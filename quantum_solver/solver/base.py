"""
Base data structures and helpers for step-by-step quantum evolution trajectories
and measurement statistics across QuTiP, Kraus, and Stim backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gates import PAULI_X, PAULI_Y, PAULI_Z
from ..core.noise import ReadoutError


def compute_bloch_vectors(rho: np.ndarray, num_qubits: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Compute the reduced 1-qubit Bloch vector (<X_q>, <Y_q>, <Z_q>) for every qubit q
    from a full N-qubit density matrix rho.
    """
    bloch: Dict[int, Tuple[float, float, float]] = {}
    rho_tensor = rho.reshape([2] * (2 * num_qubits))
    for q in range(num_qubits):
        # Partial trace over all qubits except q
        keep_ket = q
        keep_bra = q + num_qubits
        trace_axes = [i for i in range(num_qubits) if i != q]
        # Contract trace_axes between ket and bra dimensions via einsum
        idx_ket = list(range(num_qubits))
        idx_bra = [i if i != q else num_qubits for i in range(num_qubits)]
        rho_1q = np.einsum(rho_tensor, idx_ket + idx_bra, [q, num_qubits])
        rx = float(np.real(np.trace(rho_1q @ PAULI_X)))
        ry = float(np.real(np.trace(rho_1q @ PAULI_Y)))
        rz = float(np.real(np.trace(rho_1q @ PAULI_Z)))
        bloch[q] = (rx, ry, rz)
    return bloch


def compute_populations(rho: np.ndarray, num_qubits: int) -> Dict[str, float]:
    """Return computational basis probabilities {'00...0': p0, ...} from density matrix diagonal."""
    diag = np.real(np.diag(rho))
    diag = np.clip(diag, 0.0, 1.0)
    total = float(np.sum(diag))
    if total > 0:
        diag = diag / total
    return {
        format(idx, f"0{num_qubits}b"): float(prob)
        for idx, prob in enumerate(diag)
    }


def state_fidelity_with_pure(rho: np.ndarray, psi_ideal: np.ndarray) -> float:
    """Compute state fidelity F(rho, |psi><psi|) = <psi| rho |psi>."""
    val = np.vdot(psi_ideal, rho @ psi_ideal)
    return float(max(0.0, min(1.0, np.real(val))))


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
    for idx in range(dim):
        bstr = format(idx, f"0{num_qubits}b")
        p_vec[idx] = probs_dict.get(bstr, 0.0)

    p_tensor = p_vec.reshape([2] * num_qubits)
    for q in range(num_qubits):
        r_err = readout_errors.get(q)
        if r_err is not None and (r_err.p1_given_0 > 0 or r_err.p0_given_1 > 0):
            M = r_err.confusion_matrix()
            # Apply M along axis q
            p_tensor = np.tensordot(M, p_tensor, axes=([1], [q]))
            p_tensor = np.moveaxis(p_tensor, 0, q)

    p_noisy = np.clip(p_tensor.reshape(dim), 0.0, 1.0)
    s = float(np.sum(p_noisy))
    if s > 0:
        p_noisy /= s

    return {
        format(idx, f"0{num_qubits}b"): float(p_noisy[idx])
        for idx in range(dim)
    }


def sample_from_probabilities(
    probs_dict: Dict[str, float],
    shots: int,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Draw multinomial measurement shot counts from a probability distribution."""
    if shots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    labels = list(probs_dict.keys())
    p_vals = np.array([max(0.0, probs_dict[k]) for k in labels], dtype=np.float64)
    p_vals /= np.sum(p_vals)
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
    Unified output of a quantum circuit simulation across QuTiP, Kraus, or Stim backends.
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
                info = ", ".join(s.stabilizers)
            else:
                info = ""
            if s.stabilizers and 0 in s.bloch_vectors:
                info += f" | Stab: [{', '.join(s.stabilizers)}]"
            lines.append(
                f"{s.step_index:<5} | {s.operation_label:<18} | {s.elapsed_time_ns:<9.1f} | {fid_str:<9} | {pur_str:<8} | {info}"
            )

        lines.append("-" * 96)
        lines.append("Final Measurement Results:")
        emp = self.empirical_probabilities()
        all_keys = sorted(set(self.final_probabilities.keys()) | set(emp.keys()))
        for bstr in all_keys:
            p_th = self.final_probabilities.get(bstr, 0.0)
            cnt = self.measurement_counts.get(bstr, 0)
            p_emp = emp.get(bstr, 0.0)
            if p_th > 1e-4 or cnt > 0:
                lines.append(
                    f"  |{bstr}> : exact_prob = {p_th:.5f} | sampled_prob = {p_emp:.5f} ({cnt}/{self.shots} shots)"
                )
        return "\n".join(lines)
