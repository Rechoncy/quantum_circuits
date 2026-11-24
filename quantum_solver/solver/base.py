"""
Base data structures and helpers for step-by-step quantum evolution trajectories
and measurement statistics across QuTiP, Kraus, Trajectory, and Stim backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gates import PAULI_X, PAULI_Y, PAULI_Z
from ..core.noise import ReadoutError


def apply_operator_to_statevector(
    psi: np.ndarray,
    op_mat: np.ndarray,
    qubits: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """
    Apply a k-qubit matrix `op_mat` (2^k x 2^k) to `psi` (length 2^N) via tensor contraction
    in O(2^k * 2^N) time without constructing any 2^N x 2^N matrix.
    Uses big-endian ordering |q_0, q_1, ..., q_{N-1}>.
    """
    k = len(qubits)
    if k == 0:
        return psi
    psi_t = psi.reshape([2] * num_qubits)
    op_t = op_mat.reshape([2] * (2 * k))
    contracted = np.tensordot(op_t, psi_t, axes=(list(range(k, 2 * k)), list(qubits)))
    moved = np.moveaxis(contracted, list(range(k)), list(qubits))
    return moved.reshape(-1)


def apply_operator_to_density_matrix(
    rho: np.ndarray,
    op_mat: np.ndarray,
    qubits: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """
    Compute `op_full @ rho @ op_full.conj().T` in O(2^k * 4^N) time via tensor contraction
    instead of O(8^N) dense matrix multiplication.
    """
    k = len(qubits)
    if k == 0:
        return rho
    dim = 1 << num_qubits
    rho_t = rho.reshape([2] * (2 * num_qubits))
    op_t = op_mat.reshape([2] * (2 * k))
    ket_axes = list(qubits)
    bra_axes = [num_qubits + q for q in qubits]

    tmp = np.tensordot(op_t, rho_t, axes=(list(range(k, 2 * k)), ket_axes))
    tmp = np.moveaxis(tmp, list(range(k)), ket_axes)

    out = np.tensordot(op_t.conj(), tmp, axes=(list(range(k, 2 * k)), bra_axes))
    out = np.moveaxis(out, list(range(k)), bra_axes)
    return out.reshape((dim, dim))


def compute_bloch_vectors(rho: np.ndarray, num_qubits: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute the reduced 1-qubit Bloch vector (<X_q>, <Y_q>, <Z_q>) for every qubit q
    from an N-qubit density matrix rho using 6D block partial trace (safe for any N).
    """
    bloch: Dict[int, Tuple[float, float, float]] = {}
    for q in range(num_qubits):
        d_left = 1 << q
        d_right = 1 << (num_qubits - 1 - q)
        rho_6d = rho.reshape((d_left, 2, d_right, d_left, 2, d_right))
        rho_1q = np.einsum("aibajb->ij", rho_6d)
        rx = float(2.0 * np.real(rho_1q[0, 1]))
        ry = float(-2.0 * np.imag(rho_1q[0, 1]))
        rz = float(np.real(rho_1q[0, 0] - rho_1q[1, 1]))
        bloch[q] = (rx, ry, rz)
    return bloch


def compute_statevector_bloch(psi: np.ndarray, num_qubits: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute exact 1-qubit reduced Bloch vectors (<X_q>, <Y_q>, <Z_q>) directly from a pure
    statevector `psi` (length 2^N) in O(N * 2^N) time via contiguous inner products.
    """
    bloch: Dict[int, Tuple[float, float, float]] = {}
    for q in range(num_qubits):
        d_left = 1 << q
        d_right = 1 << (num_qubits - 1 - q)
        psi_3d = psi.reshape((d_left, 2, d_right))
        psi0 = psi_3d[:, 0, :].reshape(-1)
        psi1 = psi_3d[:, 1, :].reshape(-1)
        p0 = float(np.real(np.vdot(psi0, psi0)))
        p1 = float(np.real(np.vdot(psi1, psi1)))
        c01 = np.vdot(psi1, psi0)  # sum(psi1.conj() * psi0) = rho_01
        rx = float(2.0 * np.real(c01))
        ry = float(-2.0 * np.imag(c01))
        rz = float(p0 - p1)
        bloch[q] = (rx, ry, rz)
    return bloch


def probabilities_array_to_dict(
    probs: np.ndarray,
    num_qubits: int,
    max_entries: int = 256,
) -> Dict[str, float]:
    """
    Convert a 1D probability array of length 2^N into a bitstring dictionary.
    For N <= 8 (<= 256 states), includes all basis states; for N > 8, returns the top
    `max_entries` non-negligible basis states so large-N JSON serialization stays instant.
    """
    probs = np.clip(np.real(probs), 0.0, 1.0)
    total = float(np.sum(probs))
    if total > 0:
        probs = probs / total

    dim = probs.shape[0]
    if dim <= max_entries:
        return {
            format(idx, f"0{num_qubits}b"): float(probs[idx])
            for idx in range(dim)
        }

    # Select top `max_entries` indices in O(2^N) C time
    top_idx = np.argpartition(probs, -max_entries)[-max_entries:]
    top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
    out: Dict[str, float] = {}
    for idx in top_idx:
        p = float(probs[int(idx)])
        if p >= 1e-6 or len(out) < 8:
            out[format(int(idx), f"0{num_qubits}b")] = p
    return out


def compute_populations(
    rho: np.ndarray,
    num_qubits: int,
    max_entries: int = 256,
) -> Dict[str, float]:
    """Return computational basis probabilities {'00...0': p0, ...} from density matrix diagonal."""
    diag = np.real(np.diag(rho))
    return probabilities_array_to_dict(diag, num_qubits, max_entries=max_entries)


def state_fidelity_with_pure(rho: np.ndarray, psi_ideal: np.ndarray) -> float:
    """Compute state fidelity F(rho, |psi><psi|) = <psi| rho |psi>."""
    val = np.vdot(psi_ideal, rho @ psi_ideal)
    return float(max(0.0, min(1.0, np.real(val))))


def apply_readout_confusion_array(
    p_vec: np.ndarray,
    num_qubits: int,
    readout_errors: Dict[int, ReadoutError],
) -> np.ndarray:
    """Apply single-qubit measurement confusion matrices directly to a 2^N probability array."""
    if not readout_errors:
        return p_vec
    dim = 1 << num_qubits
    p_tensor = p_vec.astype(np.float64, copy=True).reshape([2] * num_qubits)
    for q in range(num_qubits):
        r_err = readout_errors.get(q)
        if r_err is not None and (r_err.p1_given_0 > 0 or r_err.p0_given_1 > 0):
            M = r_err.confusion_matrix()
            p_tensor = np.tensordot(M, p_tensor, axes=([1], [q]))
            p_tensor = np.moveaxis(p_tensor, 0, q)
    p_noisy = np.clip(p_tensor.reshape(dim), 0.0, 1.0)
    s = float(np.sum(p_noisy))
    if s > 0:
        p_noisy /= s
    return p_noisy


def apply_readout_confusion(
    probs_dict: Dict[str, float],
    num_qubits: int,
    readout_errors: Dict[int, ReadoutError],
) -> Dict[str, float]:
    """
    Apply single-qubit measurement assignment confusion matrices M_0 ⊗ M_1 ⊗ ... ⊗ M_{N-1}
    to an ideal computational-basis probability distribution.
    """
    if not readout_errors:
        return dict(probs_dict)

    dim = 1 << num_qubits
    p_vec = np.zeros(dim, dtype=np.float64)
    for bstr, val in probs_dict.items():
        p_vec[int(bstr, 2)] = val

    p_noisy = apply_readout_confusion_array(p_vec, num_qubits, readout_errors)
    return probabilities_array_to_dict(p_noisy, num_qubits)


def sample_bitstrings_from_prob_array(
    p_vec: np.ndarray,
    num_qubits: int,
    shots: int,
    readout_errors: Optional[Dict[int, ReadoutError]] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Sample `shots` computational-basis bitstrings directly from a 2^N probability array,
    applying per-qubit readout bit-flip confusion in O(shots * N) so even N=22..24 is fast
    and never needs a second 2^N tensor copy for readout confusion.
    """
    if shots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    p_clean = np.clip(np.real(p_vec), 0.0, 1.0)
    s = float(np.sum(p_clean))
    if s > 0:
        p_clean = p_clean / s
    dim = p_clean.shape[0]
    sampled_indices = rng.choice(dim, size=shots, replace=True, p=p_clean)

    # Unpack bits (shape: shots x num_qubits)
    bit_shifts = np.arange(num_qubits - 1, -1, -1, dtype=np.int64)
    bits = ((sampled_indices[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8)

    if readout_errors:
        for q in range(num_qubits):
            r_err = readout_errors.get(q)
            if r_err is not None and (r_err.p1_given_0 > 0 or r_err.p0_given_1 > 0):
                u = rng.random(shots)
                col = bits[:, q]
                flip_0_to_1 = (col == 0) & (u < r_err.p1_given_0)
                flip_1_to_0 = (col == 1) & (u < r_err.p0_given_1)
                col[flip_0_to_1] = 1
                col[flip_1_to_0] = 0

    return count_boolean_samples(bits, num_qubits)


def count_boolean_samples(raw_samples: np.ndarray, num_qubits: int) -> Dict[str, int]:
    """
    Count bitstring outcomes from a (shots, num_qubits) boolean/uint8 array for ANY `num_qubits`
    (1 to 10,000+ qubits) using packed bytes (`np.packbits`) with zero int64 overflow!
    """
    if raw_samples.size == 0:
        return {}
    bits_u8 = np.asarray(raw_samples, dtype=np.uint8)
    packed = np.packbits(bits_u8, axis=1)
    unique_packed, unique_cnts = np.unique(packed, axis=0, return_counts=True)
    # Sort descending by count so most frequent bitstrings come first
    order = np.argsort(unique_cnts)[::-1]
    unpacked = np.unpackbits(unique_packed[order], axis=1)[:, :num_qubits]
    counts: Dict[str, int] = {}
    for row, cnt in zip(unpacked, unique_cnts[order]):
        bstr = "".join("1" if b else "0" for b in row)
        counts[bstr] = int(cnt)
    return counts


def sample_from_probabilities(
    probs_dict: Dict[str, float],
    shots: int,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Draw multinomial measurement shot counts from a probability distribution."""
    if shots <= 0 or not probs_dict:
        return {}
    rng = np.random.default_rng(seed)
    labels = list(probs_dict.keys())
    p_vals = np.array([max(0.0, probs_dict[k]) for k in labels], dtype=np.float64)
    s = float(np.sum(p_vals))
    if s <= 0:
        return {}
    p_vals /= s
    draws = rng.multinomial(shots, p_vals)
    return {lbl: int(cnt) for lbl, cnt in zip(labels, draws) if cnt > 0}


@dataclass
class StepSnapshot:
    """
    Snapshot of the quantum system state after executing one circuit instruction.
    """

    step_index: int
    operation_label: str
    elapsed_time_ns: float
    density_matrix: Optional[np.ndarray] = None
    ideal_statevector: Optional[np.ndarray] = None
    state_fidelity: Optional[float] = None
    purity: Optional[float] = None
    bloch_vectors: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    populations: Dict[str, float] = field(default_factory=dict)
    stabilizers: Optional[List[str]] = None
    # Intra-gate continuous trajectory (populated by QuTiP pulse solver)
    pulse_times_ns: Optional[np.ndarray] = None
    pulse_bloch_trajectories: Optional[Dict[int, np.ndarray]] = None


@dataclass
class EvolutionResult:
    """
    Unified output of a quantum circuit simulation across QuTiP, Kraus, Trajectory, or Stim backends.
    Contains both step-by-step state evolution and final measurement statistics.
    """

    backend_name: str
    num_qubits: int
    steps: List[StepSnapshot]
    final_probabilities: Dict[str, float]
    measurement_counts: Dict[str, int]
    shots: int
    total_duration_ns: float
    final_fidelity: Optional[float] = None
    final_purity: Optional[float] = None
    stim_circuit_str: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def final_density_matrix(self) -> Optional[np.ndarray]:
        if self.steps and self.steps[-1].density_matrix is not None:
            return self.steps[-1].density_matrix
        return None

    def empirical_probabilities(self) -> Dict[str, float]:
        """Compute empirical bitstring probabilities from sampled measurement_counts."""
        if not self.measurement_counts or self.shots <= 0:
            return dict(self.final_probabilities)
        return {
            bstr: cnt / float(self.shots)
            for bstr, cnt in sorted(self.measurement_counts.items())
        }

    def summary_table(self) -> str:
        """Render a human-readable summary of step-by-step evolution and final measurements."""
        lines = [
            f"=== Evolution Result [{self.backend_name}] ({self.num_qubits} Qubits, Total Time: {self.total_duration_ns:.1f} ns) ===",
            f"{'Step':<5} | {'Operation':<18} | {'Time(ns)':<9} | {'Fidelity':<9} | {'Purity':<8} | {'Q0 Bloch (X,Y,Z) / Stabilizers'}",
            "-" * 96,
        ]
        for s in self.steps:
            fid_str = f"{s.state_fidelity:.5f}" if s.state_fidelity is not None else "   N/A   "
            pur_str = f"{s.purity:.5f}" if s.purity is not None else "  N/A   "
            if 0 in s.bloch_vectors:
                bx, by, bz = s.bloch_vectors[0]
                info = f"q0=({bx:+.3f}, {by:+.3f}, {bz:+.3f})"
            elif s.stabilizers:
                info = ", ".join(s.stabilizers[:8])
            else:
                info = ""
            if s.stabilizers and 0 in s.bloch_vectors:
                info += f" | Stab: [{', '.join(s.stabilizers[:6])}]"
            lines.append(
                f"{s.step_index:<5} | {s.operation_label:<18} | {s.elapsed_time_ns:<9.1f} | {fid_str:<9} | {pur_str:<8} | {info}"
            )

        lines.append("-" * 96)
        lines.append("Final Measurement Results:")
        emp = self.empirical_probabilities()
        all_keys = list(dict.fromkeys(list(self.final_probabilities.keys())[:32] + list(emp.keys())[:32]))
        for bstr in all_keys:
            p_th = self.final_probabilities.get(bstr, 0.0)
            cnt = self.measurement_counts.get(bstr, 0)
            p_emp = emp.get(bstr, 0.0)
            if p_th > 1e-4 or cnt > 0:
                lines.append(
                    f"  |{bstr}> : exact_prob = {p_th:.5f} | sampled_prob = {p_emp:.5f} ({cnt}/{self.shots} shots)"
                )
        return "\n".join(lines)
