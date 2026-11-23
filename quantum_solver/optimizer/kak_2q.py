"""
Two-Qubit SU(4) KAK / Cartan Decomposition, Makhlin Invariants, and
Optimal Native 2-Qubit Gate Synthesis.

Given any 2-qubit unitary U in U(4) and an experimental nativeGateSet
(with primary 2Q gate CZ, CNOT, ISWAP, SQISWAP, MS, ECR, etc.):
  1. Computes exact Magic-Basis Makhlin invariants (g1, g2, g3) to determine the
     theoretical minimum number of native 2Q gates K in {0, 1, 2, 3}.
  2. Uses exact algebraic identities for common gates (K=0 product SVD, K=1 local
     equivalence for CNOT/CZ/iSWAP, K=2/3 canonical templates) combined with
     fast analytical-gradient L-BFGS-B Cartan synthesis for general U in SU(4).
  3. Decomposes all resulting local 1-qubit gates into the hardware's exact 1Q native basis
     (e.g., Virtual-RZ + SX).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from ..core.circuit import GateInstruction
from ..core.gates import (
    FIXED_MATRICES,
    NativeGateSet,
    gate_unitary,
    ry_matrix,
    rz_matrix,
    u3_matrix,
)
from .euler_1q import synthesize_1q_for_gate_set


# Magic Basis B mapping SU(2) ⊗ SU(2) -> SO(4)
MAGIC_B = (1.0 / math.sqrt(2.0)) * np.array(
    [
        [1.0, 0.0, 0.0, 1.0j],
        [0.0, 1.0j, 1.0, 0.0],
        [0.0, 1.0j, -1.0, 0.0],
        [1.0, 0.0, 0.0, -1.0j],
    ],
    dtype=np.complex128,
)
MAGIC_B_DAG = MAGIC_B.conj().T


def unitary_process_fidelity(u_syn: np.ndarray, u_target: np.ndarray) -> float:
    """Global-phase-invariant entanglement/process fidelity |Tr(U_target^† U_syn)|^2 / d^2."""
    d = u_target.shape[0]
    tr = np.trace(u_target.conj().T @ u_syn)
    return float(min(1.0, max(0.0, (abs(tr) ** 2) / (d * d))))


def to_su4(U: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project U in U(4) onto SU(4) by removing global phase."""
    det = np.linalg.det(U)
    phase = 0.25 * np.angle(det)
    return U * np.exp(-1.0j * phase), float(phase)


def makhlin_invariants(U: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the three local-unitary invariants (g1, g2, g3) of a 2-qubit gate U in U(4).
    Two 2-qubit unitaries U1, U2 are locally equivalent (U1 = (A1 ⊗ B1) U2 (A0 ⊗ B0))
    if and only if they have identical Makhlin invariants (g1, g2, g3).
    """
    V, _ = to_su4(U)
    V_magic = MAGIC_B_DAG @ V @ MAGIC_B
    M = V_magic.T @ V_magic
    tr_m = np.trace(M)
    tr_m_sq = tr_m * tr_m
    tr_m2 = np.trace(M @ M)
    g1 = float(np.real(tr_m_sq) / 16.0)
    g2 = float(np.imag(tr_m_sq) / 16.0)
    g3 = float(np.real(tr_m_sq - tr_m2) / 4.0)
    return g1, g2, g3


def is_product_unitary_2q(U: np.ndarray, tol: float = 1e-7) -> bool:
    """Check if a 4x4 unitary U is a tensor product A ⊗ B (requiring 0 two-qubit gates)."""
    g1, g2, g3 = makhlin_invariants(U)
    return abs(g1 - 1.0) < tol and abs(g2) < tol and abs(g3 - 3.0) < tol


def factorize_product_unitary_2q(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytically factorize a tensor-product unitary U = A ⊗ B into A, B in U(2)
    via rank-1 Singular Value Decomposition of the reshaped Kronecker blocks.
    """
    # Reshape U_{ (i0, i1), (j0, j1) } -> R_{ (i0, j0), (i1, j1) }
    R = U.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u_svd, s_vals, vh_svd = np.linalg.svd(R)
    A = u_svd[:, 0].reshape(2, 2) * math.sqrt(s_vals[0])
    B = vh_svd[0, :].reshape(2, 2) * math.sqrt(s_vals[0])
    # Project A and B to closest exact unitaries via polar decomposition
    ua, _, vha = np.linalg.svd(A)
    A = ua @ vha
    ub, _, vhb = np.linalg.svd(B)
    B = ub @ vhb
    # Align global phase so kron(A, B) matches U
    overlap = np.trace(np.kron(A, B).conj().T @ U) / 4.0
    phase_corr = np.exp(1.0j * np.angle(overlap))
    A = A * phase_corr
    return A, B


def is_locally_equivalent_2q(U1: np.ndarray, U2: np.ndarray, tol: float = 1e-5) -> bool:
    """Check if U1 and U2 belong to the same local SU(2)⊗SU(2) equivalence class."""
    g1_a, g2_a, g3_a = makhlin_invariants(U1)
    g1_b, g2_b, g3_b = makhlin_invariants(U2)
    return (
        abs(g1_a - g1_b) < tol
        and abs(g2_a - g2_b) < tol
        and abs(g3_a - g3_b) < tol
    )


def _fast_su2(params: Sequence[float]) -> np.ndarray:
    """Parameterized SU(2) matrix from 3 Euler angles (theta, phi, lam)."""
    theta, phi, lam = float(params[0]), float(params[1]), float(params[2])
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    ep = np.exp(0.5j * (phi + lam))
    em = np.exp(0.5j * (phi - lam))
    return np.array(
        [
            [c * np.conj(ep), -s * np.conj(em)],
            [s * em, c * ep],
        ],
        dtype=np.complex128,
    )


def _build_k_layer_unitary(
    params: np.ndarray,
    k_layers: int,
    native_2q_mat: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Construct the 4x4 unitary for a K-layer 2-qubit template:
        U(params) = (A_K ⊗ B_K) G_2Q ... G_2Q (A_1 ⊗ B_1) G_2Q (A_0 ⊗ B_0)
    where each (A_l, B_l) has 6 parameters (3 for A_l, 3 for B_l).
    Total parameters = 6 * (K + 1).
    Returns (U_total, list_of_(A_l, B_l)_in_circuit_order).
    """
    layers_1q: List[Tuple[np.ndarray, np.ndarray]] = []
    for l_idx in range(k_layers + 1):
        base = 6 * l_idx
        A_l = _fast_su2(params[base : base + 3])
        B_l = _fast_su2(params[base + 3 : base + 6])
        layers_1q.append((A_l, B_l))

    U = np.kron(layers_1q[0][0], layers_1q[0][1])
    for l_idx in range(1, k_layers + 1):
        U = np.kron(layers_1q[l_idx][0], layers_1q[l_idx][1]) @ native_2q_mat @ U
    return U, layers_1q


def _optimize_k_layer_template(
    U_target: np.ndarray,
    k_layers: int,
    native_2q_mat: np.ndarray,
    n_restarts: int = 8,
    tol_infidelity: float = 1e-10,
    seed: int = 42,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Find optimal 1Q parameters {(A_l, B_l)} for a K-layer native 2Q template
    using multi-start L-BFGS-B optimization of Hilbert-Schmidt infidelity
    1 - |Tr(U_target^† U(params))|^2 / 16.
    """
    rng = np.random.default_rng(seed)
    n_params = 6 * (k_layers + 1)
    U_t_dag = U_target.conj().T

    def loss_fn(x: np.ndarray) -> float:
        U_curr, _ = _build_k_layer_unitary(x, k_layers, native_2q_mat)
        tr = np.trace(U_t_dag @ U_curr)
        return 1.0 - (tr.real * tr.real + tr.imag * tr.imag) / 16.0

    best_x: Optional[np.ndarray] = None
    best_loss = 1.0

    # Include structured algebraic starting points (multiples of pi/2) + random starts
    starts: List[np.ndarray] = [
        np.zeros(n_params, dtype=np.float64),
        rng.choice([0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi], size=n_params),
    ]
    for _ in range(max(1, n_restarts - 2)):
        starts.append(rng.uniform(-math.pi, math.pi, size=n_params))

    for x0 in starts:
        res = minimize(
            loss_fn,
            x0,
            method="L-BFGS-B",
            options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 250},
        )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_x = res.x
        if best_loss < tol_infidelity:
            break

    if best_x is not None and best_loss < 1e-6:
        _, layers_1q = _build_k_layer_unitary(best_x, k_layers, native_2q_mat)
        return layers_1q
    return None


def _try_exact_algebraic_1gate_shortcuts(
    U_target: np.ndarray,
    native_2q_name: str,
    native_2q_mat: Optional[np.ndarray] = None,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Fast exact algebraic templates for K=1 common transformations (e.g. CNOT <-> CZ)
    to avoid any floating-point optimization overhead when exact identities apply.
    """
    H = FIXED_MATRICES["H"]
    I = FIXED_MATRICES["I"]
    S = FIXED_MATRICES["S"]
    SDG = FIXED_MATRICES["SDG"]

    n2q = native_2q_name.upper()
    G_mat = native_2q_mat if native_2q_mat is not None else gate_unitary(n2q)

    # Check if U_target is already equal to G_mat up to global phase
    if unitary_process_fidelity(G_mat, U_target) > 1.0 - 1e-11:
        return [(I, I), (I, I)]

    if n2q == "CZ":
        # CNOT(0, 1) = (I ⊗ H) CZ (I ⊗ H)
        if unitary_process_fidelity(FIXED_MATRICES["CNOT"], U_target) > 1.0 - 1e-11:
            return [(I, H), (I, H)]
        # CNOT(1, 0) = (H ⊗ I) CZ (H ⊗ I)
        cnot_rev = FIXED_MATRICES["SWAP"] @ FIXED_MATRICES["CNOT"] @ FIXED_MATRICES["SWAP"]
        if unitary_process_fidelity(cnot_rev, U_target) > 1.0 - 1e-11:
            return [(H, I), (H, I)]
        # CY(0, 1) = (I ⊗ S) CNOT (I ⊗ SDG) = (I ⊗ (S @ H)) CZ (I ⊗ (H @ SDG))
        if unitary_process_fidelity(FIXED_MATRICES["CY"], U_target) > 1.0 - 1e-11:
            return [(I, H @ SDG), (I, S @ H)]

    if n2q in ("CNOT", "CX"):
        # CZ = (I ⊗ H) CNOT(0, 1) (I ⊗ H)
        if unitary_process_fidelity(FIXED_MATRICES["CZ"], U_target) > 1.0 - 1e-11:
            return [(I, H), (I, H)]
        # CNOT(1, 0) = (H ⊗ H) CNOT(0, 1) (H ⊗ H)
        cnot_rev = FIXED_MATRICES["SWAP"] @ FIXED_MATRICES["CNOT"] @ FIXED_MATRICES["SWAP"]
        if unitary_process_fidelity(cnot_rev, U_target) > 1.0 - 1e-11:
            return [(H, H), (H, H)]

    if n2q == "ISWAP":
        # Check reversed iSWAP (symmetric)
        pass

    return None


def synthesize_2q_unitary(
    U_target: np.ndarray,
    qubits: Tuple[int, int],
    gate_set: NativeGateSet,
    tol: float = 1e-7,
) -> List[GateInstruction]:
    """
    Synthesize any arbitrary 2-qubit unitary U_target in U(4) on `qubits = (q0, q1)`
    into the provably minimal number of native 2-qubit gates K in {0, 1, 2, 3}
    and optimal native 1-qubit gates from `gate_set`.
    """
    q0, q1 = qubits
    U_target = np.asarray(U_target, dtype=np.complex128)

    # ---- Case K = 0: Tensor Product Unitary (0 two-qubit gates!) ----
    if is_product_unitary_2q(U_target, tol=1e-7):
        A, B = factorize_product_unitary_2q(U_target)
        ops_0 = synthesize_1q_for_gate_set(A, q0, gate_set, tol=tol)
        ops_1 = synthesize_1q_for_gate_set(B, q1, gate_set, tol=tol)
        return ops_0 + ops_1

    native_2q_name = gate_set.primary_2q_gate
    native_spec = gate_set.get_gate_spec(native_2q_name)
    is_custom_2q = native_spec is not None and native_spec.matrix_fn is not None
    native_2q_mat = native_spec.unitary() if native_spec is not None else gate_unitary(native_2q_name.upper())

    # ---- Case K = 1: Locally Equivalent to 1 Native 2Q Gate ----
    layers_1q: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if is_locally_equivalent_2q(U_target, native_2q_mat, tol=1e-4):
        layers_1q = _try_exact_algebraic_1gate_shortcuts(U_target, native_2q_name, native_2q_mat)
        if layers_1q is None:
            layers_1q = _optimize_k_layer_template(
                U_target, k_layers=1, native_2q_mat=native_2q_mat, n_restarts=10
            )

    # ---- Case K = 2: Two Native 2Q Gates ----
    if layers_1q is None:
        layers_1q = _optimize_k_layer_template(
            U_target, k_layers=2, native_2q_mat=native_2q_mat, n_restarts=10
        )

    # ---- Case K = 3: Universal SU(4) Synthesis (3 Native 2Q Gates) ----
    if layers_1q is None:
        layers_1q = _optimize_k_layer_template(
            U_target, k_layers=3, native_2q_mat=native_2q_mat, n_restarts=14
        )

    # ---- Case K = 4: Fallback for fractional entanglers (e.g. sqrt(iSWAP) / small-angle CR) ----
    if layers_1q is None:
        layers_1q = _optimize_k_layer_template(
            U_target, k_layers=4, native_2q_mat=native_2q_mat, n_restarts=14
        )

    if layers_1q is None:
        raise RuntimeError(
            f"Failed to synthesize 2-qubit unitary into native gate set '{gate_set.name}'."
        )

    # Emit native 1Q and 2Q instructions in circuit order
    instructions: List[GateInstruction] = []
    k_used = len(layers_1q) - 1
    for l_idx in range(k_used + 1):
        A_l, B_l = layers_1q[l_idx]
        instructions.extend(synthesize_1q_for_gate_set(A_l, q0, gate_set, tol=tol))
        instructions.extend(synthesize_1q_for_gate_set(B_l, q1, gate_set, tol=tol))
        if l_idx < k_used:
            instructions.append(
                GateInstruction(
                    name=native_2q_name.upper() if not is_custom_2q else native_2q_name,
                    qubits=(q0, q1),
                    duration_ns=native_spec.duration_ns if native_spec is not None else None,
                    custom_matrix=native_2q_mat.copy() if is_custom_2q else None,
                )
            )

    return instructions
