"""
Base data structures and helpers for step-by-step quantum evolution trajectories
and measurement statistics across QuTiP, Kraus, Trajectory, and Stim backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gates import PAULI_X, PAULI_Y, PAULI_Z, PAULIS_1Q
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


def _kron_pauli_ops(pauli_chars: Sequence[str]) -> np.ndarray:
    """Build small 2^k x 2^k Kronecker product of 1-qubit Pauli matrices."""
    mat = PAULIS_1Q[pauli_chars[0]]
    for ch in pauli_chars[1:]:
        mat = np.kron(mat, PAULIS_1Q[ch])
    return mat


def evaluate_pauli_sum_statevector(
    psi: np.ndarray,
    pauli_terms: Sequence[Tuple[float, Tuple[Tuple[int, str], ...]]],
    num_qubits: int,
) -> float:
    """
    Compute exact expectation value <psi | H | psi> for H = sum_k c_k P_k on an N-qubit
    statevector `psi` in O(num_terms * 2^N) without allocating any 2^N x 2^N matrix.
    """
    if not pauli_terms:
        return 0.0
    total = 0.0
    probs_nd: Optional[np.ndarray] = None
    all_axes = tuple(range(num_qubits))

    for coeff, ops in pauli_terms:
        if abs(coeff) < 1e-15:
            continue
        if not ops:
            total += coeff
            continue
        qubits = tuple(q for q, _ in ops)
        p_chars = tuple(p for _, p in ops)
        if all(p == "Z" for p in p_chars):
            if probs_nd is None:
                probs_nd = np.real(psi * psi.conj()).reshape([2] * num_qubits)
            sum_axes = tuple(ax for ax in all_axes if ax not in qubits)
            marg = np.sum(probs_nd, axis=sum_axes) if sum_axes else probs_nd
            # Align marginal axes with `qubits` order (already sorted if canonical, but safe for any order)
            kept_order = [ax for ax in all_axes if ax in qubits]
            if tuple(kept_order) != qubits:
                perm = [kept_order.index(q) for q in qubits]
                marg = np.transpose(marg, perm)
            signs = np.array([1.0, -1.0], dtype=np.float64)
            parity_vec = signs
            for _ in range(1, len(qubits)):
                parity_vec = np.kron(parity_vec, signs)
            total += coeff * float(np.dot(marg.reshape(-1), parity_vec))
        else:
            op_mat = _kron_pauli_ops(p_chars)
            psi_op = apply_operator_to_statevector(psi, op_mat, qubits, num_qubits)
            total += coeff * float(np.real(np.vdot(psi, psi_op)))
    return float(total)


def evaluate_pauli_sum_density_matrix(
    rho: np.ndarray,
    pauli_terms: Sequence[Tuple[float, Tuple[Tuple[int, str], ...]]],
    num_qubits: int,
) -> float:
    """
    Compute exact expectation value Tr(rho H) for H = sum_k c_k P_k on an N-qubit
    density matrix `rho` via k-qubit reduced density matrix partial trace.
    """
    if not pauli_terms:
        return 0.0
    total = 0.0
    rho_nd = rho.reshape([2] * (2 * num_qubits))

    for coeff, ops in pauli_terms:
        if abs(coeff) < 1e-15:
            continue
        if not ops:
            total += coeff
            continue
        qubits = [q for q, _ in ops]
        p_chars = [p for _, p in ops]
        k = len(qubits)
        rest = [q for q in range(num_qubits) if q not in qubits]
        perm = qubits + rest + [num_qubits + q for q in qubits] + [num_qubits + q for q in rest]
        d_sub = 1 << k
        d_rest = 1 << (num_qubits - k)
        view = np.transpose(rho_nd, perm).reshape((d_sub, d_rest, d_sub, d_rest))
        rho_sub = np.einsum("aibi->ab", view)
        op_mat = _kron_pauli_ops(p_chars)
        total += coeff * float(np.real(np.trace(rho_sub @ op_mat)))
    return float(total)


def von_neumann_entropy_hermitian(rho_sub: np.ndarray) -> float:
    """Compute von Neumann entropy S(rho) = -Tr(rho log2 rho) in bits for a small Hermitian matrix."""
    vals = np.linalg.eigvalsh(rho_sub)
    vals = np.maximum(0.0, np.real(vals))
    s = float(np.sum(vals))
    if s > 0.0:
        vals = vals / s
    nz = vals[vals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


def compute_entanglement_from_statevector(
    psi: np.ndarray,
    num_qubits: int,
) -> Tuple[List[List[float]], float]:
    """
    Compute:
      1. N x N quantum correlation matrix (diagonal = S(rho_i) in [0,1], off-diagonal = I(q_i : q_j) in [0,2] bits)
      2. Half-chain bipartite von Neumann entanglement entropy S(rho_{0..N//2-1}) in bits
    in O(N^2 * 2^N) time via reshape & 4 x 2^(N-2) Gram products (fast up to 16Q).
    """
    n = num_qubits
    if n < 2:
        return [[0.0]], 0.0
    psi_nd = psi.reshape([2] * n)
    s1: List[float] = []
    for q in range(n):
        rest = [ax for ax in range(n) if ax != q]
        m1 = np.transpose(psi_nd, [q] + rest).reshape((2, 1 << (n - 1)))
        rho_1q = m1 @ m1.conj().T
        s1.append(von_neumann_entropy_hermitian(rho_1q))

    mi_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mi_mat[i][i] = round(s1[i], 5)
        for j in range(i + 1, n):
            rest = [ax for ax in range(n) if ax != i and ax != j]
            m2 = np.transpose(psi_nd, [i, j] + rest).reshape((4, 1 << (n - 2)))
            rho_2q = m2 @ m2.conj().T
            s2 = von_neumann_entropy_hermitian(rho_2q)
            mi = max(0.0, s1[i] + s1[j] - s2)
            val = round(mi, 5)
            mi_mat[i][j] = val
            mi_mat[j][i] = val

    k_a = max(1, n // 2)
    svals = np.linalg.svd(psi.reshape((1 << k_a, 1 << (n - k_a))), compute_uv=False)
    probs_schmidt = np.maximum(0.0, np.real(svals) ** 2)
    tot = float(np.sum(probs_schmidt))
    if tot > 0.0:
        probs_schmidt /= tot
    nz = probs_schmidt[probs_schmidt > 1e-15]
    bip_s = float(-np.sum(nz * np.log2(nz))) if nz.size > 0 else 0.0
    return mi_mat, round(bip_s, 5)


def compute_entanglement_from_density_matrix(
    rho: np.ndarray,
    num_qubits: int,
) -> Tuple[List[List[float]], float]:
    """
    Compute N x N quantum mutual information matrix I(q_i : q_j) and half-chain entropy S(rho_A)
    from a full density matrix `rho` (2 <= N <= 11).
    """
    n = num_qubits
    if n < 2:
        return [[round(von_neumann_entropy_hermitian(rho), 5)]], 0.0
    rho_nd = rho.reshape([2] * (2 * n))
    s1: List[float] = []
    for q in range(n):
        d_l = 1 << q
        d_r = 1 << (n - 1 - q)
        r6 = rho.reshape((d_l, 2, d_r, d_l, 2, d_r))
        rho_1q = np.einsum("aibajb->ij", r6)
        s1.append(von_neumann_entropy_hermitian(rho_1q))

    mi_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mi_mat[i][i] = round(s1[i], 5)
        for j in range(i + 1, n):
            rest = [ax for ax in range(n) if ax != i and ax != j]
            perm = [i, j] + rest + [n + i, n + j] + [n + ax for ax in rest]
            view = np.transpose(rho_nd, perm).reshape((4, 1 << (n - 2), 4, 1 << (n - 2)))
            rho_2q = np.einsum("aibi->ab", view)
            s2 = von_neumann_entropy_hermitian(rho_2q)
            val = round(max(0.0, s1[i] + s1[j] - s2), 5)
            mi_mat[i][j] = val
            mi_mat[j][i] = val

    k_a = max(1, min(5, n // 2))
    a_axes = list(range(k_a))
    b_axes = list(range(k_a, n))
    perm_bip = a_axes + b_axes + [n + ax for ax in a_axes] + [n + ax for ax in b_axes]
    view_bip = np.transpose(rho_nd, perm_bip).reshape((1 << k_a, 1 << (n - k_a), 1 << k_a, 1 << (n - k_a)))
    rho_a = np.einsum("aibi->ab", view_bip)
    bip_s = von_neumann_entropy_hermitian(rho_a)
    return mi_mat, round(bip_s, 5)


def gf2_rank(mat: np.ndarray) -> int:
    """Compute rank over GF(2) of a 2D uint8/bool binary matrix via Gaussian elimination."""
    a = (np.asarray(mat, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    rank = 0
    for c in range(cols):
        if rank >= rows:
            break
        piv_rel = np.nonzero(a[rank:, c])[0]
        if piv_rel.size == 0:
            continue
        piv = rank + int(piv_rel[0])
        if piv != rank:
            a[[rank, piv]] = a[[piv, rank]]
        below = np.nonzero(a[rank + 1 :, c])[0] + (rank + 1)
        if below.size > 0:
            a[below] ^= a[rank]
        rank += 1
    return rank


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
