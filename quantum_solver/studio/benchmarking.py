"""
Quantum Gate Characterization & Benchmarking Engine (`Quantum Studio EDA v2.5`).

Implements experimental-grade gate calibration & tomography protocols under realistic
hardware decoherence (T1, T2, depolarizing, ZZ crosstalk, readout error) plus coherent
control imperfections (amplitude over/under-rotation, detuning, IQ quadrature phase skew):

  1. Standard & Interleaved Randomized Benchmarking (1Q C_1 & 2Q C_2 via Stim Tableaus):
     - Fits survival probability P(m) = A * p^m + B via nonlinear least squares;
     - Extracts SPAM-free Error Per Clifford (EPC), Average Clifford Fidelity,
       Interleaved Gate Error r_gate with Magesan rigorous bounds, and T1/T2 coherence floor.
  2. AllXY 21-Sequence Pulse Calibration Diagnostic:
     - Executes all 21 canonical pulse pairs from {I, X, Y, X90, Y90};
     - Automatically diagnoses Rabi amplitude over/under-rotation sawtooth signatures,
       off-resonance detuning / IQ phase skew splits, and T1/T2 relaxation damping.
  3. Gate Set Tomography (GST — Gauge-Aligned Linear Inversion + Germ Amplification):
     - Informationally complete 6-state preparation fiducials & 4-axis measurement fiducials,
       enhanced with germ power sequences g^L for coherent angle/axis refinement;
     - Reconstructs 4x4 Pauli Transfer Matrices (PTM) in {I, X, Y, Z}, Process & Average
       Gate Fidelities, Wallman Unitarity u(E), exact 1Q Diamond Distance 0.5*||E - U||_diamond,
       Coherent vs. Stochastic error breakdown, and SPAM tomography.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm, orthogonal_procrustes, svdvals
from scipy.optimize import curve_fit, minimize
import stim

from ..core.gates import gate_unitary
from ..core.noise import NoiseChannel
from .models import CircuitOpModel, GateBenchmarkRequest, ProjectModel
from .project_engine import build_hardware_model, rabi_pulse_duration_ns


# ============================================================================
# Pauli Basis & Liouville Superoperator Utilities
# ============================================================================

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PAULIS_1Q = (_I2, _SX, _SY, _SZ)
_PAULI_LABELS_1Q = ["I", "X", "Y", "Z"]


def _unitary_to_liouville(u: np.ndarray) -> np.ndarray:
    """Convert d x d unitary U to d^2 x d^2 column-stacked (row-major vec) superoperator U ⊗ U*."""
    return np.kron(u, u.conj())


def _kraus_to_liouville(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    """Convert Kraus operators {K_j} (rho -> sum K_j rho K_j^dagger) to d^2 x d^2 superoperator."""
    d = kraus_ops[0].shape[0]
    sup = np.zeros((d * d, d * d), dtype=np.complex128)
    for k in kraus_ops:
        sup += np.kron(k, k.conj())
    return sup


def _liouville_to_ptm_1q(sup_col: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 vec-basis superoperator S (vec(E(rho)) = S vec(rho), row-major)
    into the real 4x4 Pauli Transfer Matrix R_{ij} = (1/2) Tr(P_i E(P_j)) for P in {I, X, Y, Z}.
    """
    # Basis change matrix M where columns are vec(P_j) / sqrt(2)
    m = np.column_stack([p.reshape(-1) / math.sqrt(2.0) for p in _PAULIS_1Q])
    ptm = m.conj().T @ sup_col @ m
    return np.real_if_close(ptm, tol=1e5).real.astype(np.float64)


def _ptm_to_choi_1q(ptm: np.ndarray) -> np.ndarray:
    """
    Convert 1-qubit 4x4 Pauli Transfer Matrix R into the normalized 4x4 Choi matrix J(E)
    satisfying Tr(J) = 1 (where J = (1/2) sum_{|a><b|} |a><b| ⊗ E(|a><b|)).
    Formula: J = (1/4) sum_{i,j=0}^3 R_{ij} (P_j^T ⊗ P_i).
    """
    choi = np.zeros((4, 4), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            choi += 0.25 * float(ptm[i, j]) * np.kron(_PAULIS_1Q[j].T, _PAULIS_1Q[i])
    return choi


def _project_ptm_physical_1q(ptm: np.ndarray) -> np.ndarray:
    """Ensure trace-preservation (top row = [1, 0, 0, 0]) and project Choi to nearest PSD if slightly negative."""
    r = np.array(ptm, dtype=np.float64, copy=True)
    r[0, :] = [1.0, 0.0, 0.0, 0.0]
    choi = _ptm_to_choi_1q(r)
    choi = 0.5 * (choi + choi.conj().T)
    evals, evecs = np.linalg.eigh(choi)
    if np.min(evals) < -1e-7:
        evals_clipped = np.maximum(evals, 0.0)
        s = float(np.sum(evals_clipped))
        if s > 1e-12:
            evals_clipped /= s
        choi_psd = (evecs * evals_clipped) @ evecs.conj().T
        # Convert Choi back to PTM: R_{ij} = Tr(J * (P_j^T ⊗ P_i))
        r_proj = np.zeros((4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                op = np.kron(_PAULIS_1Q[j].T, _PAULIS_1Q[i])
                r_proj[i, j] = float(np.real(np.trace(choi_psd @ op)))
        r_proj[0, :] = [1.0, 0.0, 0.0, 0.0]
        return r_proj
    return r


def _diamond_distance_1q_ptm(ptm_meas: np.ndarray, ptm_ideal: np.ndarray) -> float:
    """
    Compute the standard normalized half-diamond distance:
        d_diamond(E, U) = 0.5 * || E - U ||_diamond  in [0, 1]
    for 1-qubit channels via maximization of 0.5 * || ((E - U) ⊗ I)(|psi><psi|) ||_1
    over pure two-qubit input states |psi> in C^4 (seeded with maximally entangled Bell state,
    which is exact for all Pauli/unital channels and refined via L-BFGS-B for non-unital channels).
    """
    delta_ptm = ptm_meas - ptm_ideal
    if np.linalg.norm(delta_ptm) < 1e-12:
        return 0.0

    # Precompute 4x4 operator action of (DeltaE ⊗ I)(|psi><psi|):
    # Note J_delta = _ptm_to_choi_1q(delta_ptm) has J_delta = (1/2) sum_{a,b} |a><b| ⊗ DeltaE(|a><b|).
    # For any pure state |psi> = sum_{a,k} C_{a,k} |a>_sys |k>_anc (2x2 matrix C with ||C||_F = 1),
    # ((DeltaE ⊗ I)(|psi><psi|))_{(i,k),(j,l)} = sum_{a,b} C_{a,k} C_{b,l}^* DeltaE(|a><b|)_{i,j}.
    delta_blocks = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            eab = np.zeros((2, 2), dtype=np.complex128)
            eab[a, b] = 1.0
            # Decompose eab in Pauli basis: eab = (1/2) sum_j Tr(P_j eab) P_j
            coeffs_in = [float(np.trace(p @ eab).real) if idx == 0 else complex(np.trace(p @ eab)) for idx, p in enumerate(_PAULIS_1Q)]
            out_mat = np.zeros((2, 2), dtype=np.complex128)
            for i in range(4):
                val = sum(delta_ptm[i, j] * coeffs_in[j] for j in range(4))
                out_mat += 0.5 * val * _PAULIS_1Q[i]
            delta_blocks[a, b] = out_mat

    def eval_half_trace_norm(c_mat: np.ndarray) -> float:
        # out_{(i,k),(j,l)} = sum_{a,b} c_{a,k} conj(c_{b,l}) delta_blocks[a, b, i, j]
        out4 = np.einsum("ak,bl,abij->ikjl", c_mat, c_mat.conj(), delta_blocks).reshape(4, 4)
        return 0.5 * float(np.sum(svdvals(out4)))

    # Seed 1: Maximally entangled Bell state C = I_2 / sqrt(2) -> gives Choi trace norm!
    best_val = eval_half_trace_norm(np.eye(2, dtype=np.complex128) / math.sqrt(2.0))
    # Seed 2 & 3: Computational basis |00> and |10> (important for amplitude damping T1!)
    for diag_w in ([1.0, 0.0], [0.0, 1.0], [math.sqrt(0.3), math.sqrt(0.7)]):
        c0 = np.diag(np.array(diag_w, dtype=np.complex128))
        best_val = max(best_val, eval_half_trace_norm(c0))

    # Refine over general C in C^{2x2} (8 real vars normalized on S^7)
    def neg_obj(x: np.ndarray) -> float:
        nrm = float(np.linalg.norm(x))
        if nrm < 1e-12:
            return 0.0
        xn = x / nrm
        c_m = (xn[:4] + 1.0j * xn[4:]).reshape(2, 2)
        return -eval_half_trace_norm(c_m)

    x_bell = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) / math.sqrt(2.0)
    x_skew = np.array([0.5, 0.2, -0.1, 0.8, 0.1, -0.1, 0.05, 0.1])
    for seed_x in (x_bell, x_skew):
        res = minimize(neg_obj, seed_x, method="Powell", options={"maxiter": 60, "ftol": 1e-6})
        best_val = max(best_val, float(-res.fun))

    return float(min(1.0, max(0.0, best_val)))


# ============================================================================
# Realistic Noisy + Coherent-Error Gate Superoperator Builder
# ============================================================================

def _build_1q_noisy_pulse_superop(
    axis_angle: Optional[Tuple[float, float, float]],
    duration_ns: float,
    t1_ns: float,
    t2_ns: float,
    depol_1q: float,
    amp_error_pct: float,
    detuning_mhz: float,
    phase_skew_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (S_noisy, U_ideal) for a 1Q physical pulse:
      - If axis_angle is None: Identity / idle wait of duration_ns.
      - If axis_angle is (nx, ny, theta):
          Ideal unitary: U_ideal = exp(-i (theta/2) (nx X + ny Y))
          Physical pulse Hamiltonian with coherent calibration errors:
            theta_phys = theta * (1 + amp_error_pct / 100)
            If ny != 0 (Y-quadrature pulse), phase shifts by phase_skew_deg!
            Detuning term: +(2*pi * detuning_mhz * 1e-3 * duration_ns / 2) Z
    Followed by gate-duration T1/T2 thermal relaxation and 1Q depolarizing Kraus channels.
    """
    if axis_angle is None:
        u_ideal = _I2.copy()
        det_rad = 2.0 * math.pi * (detuning_mhz * 1e-3) * duration_ns
        u_coherent = expm(-0.5j * det_rad * _SZ) if abs(det_rad) > 1e-12 else _I2.copy()
    else:
        nx, ny, theta = axis_angle
        u_ideal = expm(-0.5j * theta * (nx * _SX + ny * _SY))

        scale = 1.0 + (amp_error_pct / 100.0)
        base_phi = math.atan2(ny, nx)
        # Apply IQ quadrature skew predominantly to the Y quadrature
        skew_rad = math.radians(phase_skew_deg) * (abs(ny) / max(1e-9, math.hypot(nx, ny)))
        phi_phys = base_phi + skew_rad
        drive_gen = (theta * scale) * (math.cos(phi_phys) * _SX + math.sin(phi_phys) * _SY)
        det_rad = 2.0 * math.pi * (detuning_mhz * 1e-3) * duration_ns
        u_coherent = expm(-0.5j * (drive_gen + det_rad * _SZ))

    s_total = _unitary_to_liouville(u_coherent)

    # Thermal relaxation T1 / T2 during pulse duration
    if duration_ns > 0 and (math.isfinite(t1_ns) or math.isfinite(t2_ns)):
        ch_tr = NoiseChannel.thermal_relaxation(qubit=0, t1_ns=t1_ns, t2_ns=t2_ns, duration_ns=duration_ns)
        s_total = _kraus_to_liouville(ch_tr.kraus_operators()) @ s_total

    # Control depolarizing error
    if depol_1q > 1e-12:
        ch_dep = NoiseChannel.depolarizing_1q(qubit=0, p=min(0.74, depol_1q))
        s_total = _kraus_to_liouville(ch_dep.kraus_operators()) @ s_total

    return s_total, u_ideal


def _extract_qubit_noise_params(project: ProjectModel, q_idx: int) -> Dict[str, float]:
    """Extract effective T1, T2, 1Q/2Q gate times, depol, ZZ, and readout error for qubit q_idx."""
    hw_model = build_hardware_model(project)
    sim = project.sim_settings
    qp = hw_model.qubit_params[min(q_idx, hw_model.num_qubits - 1)]

    t1_ns = qp.t1_ns if sim.enable_t1_t2 else float("inf")
    t2_ns = qp.t2_ns if sim.enable_t1_t2 else float("inf")
    depol_1q = qp.depol_1q if sim.enable_depol else 0.0
    p01 = qp.readout_error.p0_given_1 if sim.enable_readout else 0.0
    p10 = qp.readout_error.p1_given_0 if sim.enable_readout else 0.0

    hp = next(
        (h for h in project.hardware_profiles if h.id == project.active_hardware_id),
        project.hardware_profiles[0],
    )
    t_1q_ns = float(hp.t_1q_ns)
    t_2q_ns = float(hp.t_2q_ns)
    depol_2q = float(hp.depol_2q) if sim.enable_depol else 0.0
    zz_khz = float(hp.zz_khz) if sim.enable_zz else 0.0

    # Rabi-frequency drive model: pi and pi/2 pulses get their own physical lengths
    t_pi_ns = t_1q_ns
    t_pi2_ns = t_1q_ns
    q_spec = project.qubits[q_idx] if 0 <= q_idx < len(project.qubits) else None
    q_rabi = q_spec.rabi_mhz if q_spec is not None else None
    if hp.drive_mode == "fixed_rabi" or q_rabi is not None:
        omega = float(q_rabi if q_rabi is not None else hp.rabi_mhz)
        t_pi_ns = rabi_pulse_duration_ns(math.pi, omega, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
        t_pi2_ns = rabi_pulse_duration_ns(0.5 * math.pi, omega, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
        t_1q_ns = 0.5 * (t_pi_ns + t_pi2_ns)

    return {
        "t1_ns": t1_ns,
        "t2_ns": t2_ns,
        "t_1q_ns": t_1q_ns,
        "t_pi_ns": t_pi_ns,
        "t_pi2_ns": t_pi2_ns,
        "t_2q_ns": t_2q_ns,
        "depol_1q": depol_1q,
        "depol_2q": depol_2q,
        "zz_khz": zz_khz,
        "p1_given_0": p10,
        "p0_given_1": p01,
        "hardware_name": hp.name,
    }


def _build_primitive_1q_library(
    noise_cfg: Dict[str, float],
    amp_error_pct: float,
    detuning_mhz: float,
    phase_skew_deg: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build fundamental physical 1Q pulses used by AllXY, GST, and Clifford synthesis:
      I, X, Y, X90, Xm90, Y90, Ym90, Z, S, SDG, H
    """
    t1 = noise_cfg["t1_ns"]
    t2 = noise_cfg["t2_ns"]
    t_1q = noise_cfg["t_1q_ns"]
    t_pi = noise_cfg.get("t_pi_ns", t_1q)
    t_pi2 = noise_cfg.get("t_pi2_ns", t_1q)
    dep1 = noise_cfg["depol_1q"]

    def mk(ax: Optional[Tuple[float, float, float]], dur: float = t_1q) -> Tuple[np.ndarray, np.ndarray]:
        return _build_1q_noisy_pulse_superop(
            ax, dur, t1, t2, dep1, amp_error_pct, detuning_mhz, phase_skew_deg
        )

    lib: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "I": mk(None, t_1q),
        "X": mk((1.0, 0.0, math.pi), t_pi),
        "Y": mk((0.0, 1.0, math.pi), t_pi),
        "X90": mk((1.0, 0.0, 0.5 * math.pi), t_pi2),
        "Xm90": mk((-1.0, 0.0, 0.5 * math.pi), t_pi2),
        "Y90": mk((0.0, 1.0, 0.5 * math.pi), t_pi2),
        "Ym90": mk((0.0, -1.0, 0.5 * math.pi), t_pi2),
        # Virtual-Z frame updates (0 ns duration, exact software phase)
        "Z": (_unitary_to_liouville(_SZ), _SZ),
        "S": (_unitary_to_liouville(gate_unitary("S")), gate_unitary("S")),
        "SDG": (_unitary_to_liouville(gate_unitary("SDG")), gate_unitary("SDG")),
    }
    # H = Y90 followed by Z (or Virtual-Z S * X90 * S): physically 1 pi/2 pulse (Y90) + Virtual-Z!
    s_y90, u_y90 = lib["Y90"]
    s_z, u_z = lib["Z"]
    # Note Y90 @ Z: (-i/sqrt(2))(I*Z - i Y*Z) -> check Y90 @ Z = [[c, -s], [s, c]] @ [[1,0],[0,-1]] = [[1/sqrt2, 1/sqrt2], [1/sqrt2, -1/sqrt2]] = H!
    lib["H"] = (s_y90 @ s_z, u_y90 @ u_z)
    return lib


# ============================================================================
# Canonical 24 Single-Qubit Cliffords Decomposed into Physical Pulses
# ============================================================================

def _build_canonical_24_clifford_seqs() -> List[Tuple[str, ...]]:
    """Generate all 24 distinct elements of the 1-qubit Clifford group C_1 via minimum-pulse BFS."""
    gens = {
        "I": _I2,
        "Z": _SZ,
        "X90": expm(-0.25j * math.pi * _SX),
        "Xm90": expm(0.25j * math.pi * _SX),
        "Y90": expm(-0.25j * math.pi * _SY),
        "Ym90": expm(0.25j * math.pi * _SY),
        "X": expm(-0.5j * math.pi * _SX),
        "Y": expm(-0.5j * math.pi * _SY),
    }
    found_seqs: List[Tuple[str, ...]] = [("I",)]
    found_u: List[np.ndarray] = [_I2.copy()]
    queue: List[Tuple[Tuple[str, ...], np.ndarray]] = [((), _I2.copy())]
    idx_q = 0
    while len(found_seqs) < 24 and idx_q < len(queue):
        seq, u_cur = queue[idx_q]
        idx_q += 1
        for g_nm in ("Z", "X90", "Xm90", "Y90", "Ym90", "X", "Y"):
            u_next = gens[g_nm] @ u_cur
            if any(abs(complex(np.trace(u_old.conj().T @ u_next))) / 2.0 > 0.9999 for u_old in found_u):
                continue
            next_seq = seq + (g_nm,)
            found_seqs.append(next_seq)
            found_u.append(u_next)
            queue.append((next_seq, u_next))
            if len(found_seqs) == 24:
                break
    return found_seqs


_CLIFFORD_1Q_PULSE_SEQS: List[Tuple[str, ...]] = _build_canonical_24_clifford_seqs()


def _match_1q_clifford_index(u_target: np.ndarray, cliff_unitaries: Sequence[np.ndarray]) -> int:
    """Find the unique Clifford index k in 0..23 equal to u_target up to global phase."""
    best_idx = 0
    best_ov = -1.0
    for idx, uc in enumerate(cliff_unitaries):
        ov = abs(complex(np.trace(uc.conj().T @ u_target))) / 2.0
        if ov > best_ov:
            best_ov = ov
            best_idx = idx
        if ov > 0.99999:
            return idx
    return best_idx


# ============================================================================
# 1. Randomized Benchmarking (1Q & 2Q Standard RB + Interleaved RB)
# ============================================================================

def _run_rb_protocol(req: GateBenchmarkRequest) -> Dict[str, Any]:
    project = req.project
    n_proj = len(project.qubits)
    q0 = max(0, min(req.target_qubit, n_proj - 1))
    is_2q = req.rb_num_qubits == 2 and n_proj >= 2
    q1 = max(0, min(req.second_qubit, n_proj - 1))
    if is_2q and q1 == q0:
        q1 = (q0 + 1) % n_proj

    noise0 = _extract_qubit_noise_params(project, q0)
    rng = np.random.default_rng(req.seed)
    shots = max(0, int(req.shots))

    # Clean & sort sequence lengths
    raw_lens = sorted({max(1, min(256, int(m))) for m in req.rb_lengths})
    if len(raw_lens) < 3:
        raw_lens = [1, 2, 4, 8, 16, 32, 64, 128]
    num_seeds = max(4, min(60, int(req.rb_num_seeds)))

    interleaved_name = (req.interleaved_gate or "").strip().upper()
    if interleaved_name in ("", "NONE", "NULL"):
        interleaved_name = ""

    sample_ops: List[CircuitOpModel] = []

    if not is_2q:
        # ---- 1-Qubit Randomized Benchmarking (Exact C_1 = 24 Cliffords) ----
        prim = _build_primitive_1q_library(
            noise0, req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg
        )
        cliff_supers: List[np.ndarray] = []
        cliff_unitaries: List[np.ndarray] = []
        avg_phys_pulses = 0.0
        for seq in _CLIFFORD_1Q_PULSE_SEQS:
            s_acc = np.eye(4, dtype=np.complex128)
            u_acc = _I2.copy()
            for gname in seq:
                sg, ug = prim[gname]
                s_acc = sg @ s_acc
                u_acc = ug @ u_acc
                if gname not in ("I", "Z", "S", "SDG"):
                    avg_phys_pulses += 1.0
            cliff_supers.append(s_acc)
            cliff_unitaries.append(u_acc)
        avg_phys_pulses /= 24.0

        # Initial |0><0| vec and noisy Z-basis POVM M_0 for survival probability P(|0>)
        rho0_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        p10 = noise0["p1_given_0"]
        p01 = noise0["p0_given_1"]
        e0_mat = np.diag([1.0 - p10, p01]).astype(np.complex128)
        e0_vec_bra = e0_mat.reshape(-1).conj()

        # Optional 1Q Interleaved Clifford gate
        inter_super: Optional[np.ndarray] = None
        inter_u: Optional[np.ndarray] = None
        if interleaved_name:
            if interleaved_name in ("SX", "SQRTX", "X90"):
                inter_super, inter_u = prim["X90"]
                interleaved_name = "SX"
            elif interleaved_name == "Y90":
                inter_super, inter_u = prim["Y90"]
            elif interleaved_name in ("X", "Y", "Z", "H", "S"):
                inter_super, inter_u = prim[interleaved_name]
            else:
                # Fallback to X90 if a 2Q gate was accidentally left selected on 1Q mode
                inter_super, inter_u = prim["X90"]
                interleaved_name = "SX"

        def eval_1q_rb_curves(use_interleaved: bool) -> Tuple[List[float], List[float], List[List[float]], List[int]]:
            means: List[float] = []
            stds: List[float] = []
            raw_samples: List[List[float]] = []
            sampled_seqs: List[Tuple[int, List[int], int]] = []
            sample_m_targets = {0, min(1, len(raw_lens) - 1), min(2, len(raw_lens) - 1)}
            for m_idx, m_len in enumerate(raw_lens):
                seq_probs: List[float] = []
                for k in range(num_seeds):
                    picks = rng.integers(0, 24, size=m_len)
                    s_seq = np.eye(4, dtype=np.complex128)
                    u_seq = _I2.copy()
                    for c_i in picks:
                        s_seq = cliff_supers[int(c_i)] @ s_seq
                        u_seq = cliff_unitaries[int(c_i)] @ u_seq
                        if use_interleaved and inter_super is not None and inter_u is not None:
                            s_seq = inter_super @ s_seq
                            u_seq = inter_u @ u_seq
                    inv_idx = _match_1q_clifford_index(u_seq.conj().T, cliff_unitaries)
                    s_full = cliff_supers[inv_idx] @ s_seq
                    p_surv = float(np.clip(np.real(e0_vec_bra @ (s_full @ rho0_vec)), 0.0, 1.0))
                    if shots > 0:
                        p_surv = float(rng.binomial(shots, p_surv)) / float(shots)
                    seq_probs.append(round(p_surv, 6))
                    if m_idx in sample_m_targets and k == 0:
                        sampled_seqs.append((m_len, [int(x) for x in picks], int(inv_idx)))
                arr = np.array(seq_probs, dtype=np.float64)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0)
                raw_samples.append(seq_probs)
            return means, stds, raw_samples, sampled_seqs

        ref_means, ref_stds, ref_raw, ref_seq_samples = eval_1q_rb_curves(False)
        irb_means: Optional[List[float]] = None
        irb_stds: Optional[List[float]] = None
        irb_raw: Optional[List[List[float]]] = None
        irb_seq_samples: List[Tuple[int, List[int], int]] = []
        if inter_super is not None:
            irb_means, irb_stds, irb_raw, irb_seq_samples = eval_1q_rb_curves(True)

        # Build multi-depth Standard RB + Interleaved IRB CircuitOpModel samples for inline schematic & SVG export
        dsl_map: Dict[str, Tuple[str, List[float]]] = {
            "I": ("I", []),
            "X": ("X", []),
            "Y": ("Y", []),
            "Z": ("Z", []),
            "X90": ("SX", []),
            "Xm90": ("RX", [-round(0.5 * math.pi, 4)]),
            "Y90": ("RY", [round(0.5 * math.pi, 4)]),
            "Ym90": ("RY", [-round(0.5 * math.pi, 4)]),
        }
        inter_dsl_map: Dict[str, Tuple[str, List[float]]] = {
            "SX": ("SX", []),
            "X90": ("SX", []),
            "X": ("X", []),
            "Y": ("Y", []),
            "Z": ("Z", []),
            "H": ("H", []),
            "S": ("S", []),
            "Y90": ("RY", [round(0.5 * math.pi, 4)]),
        }

        def build_1q_rb_ops(
            tag: str, rand_c: List[int], inv_c: int, inter_gate_nm: Optional[str]
        ) -> List[CircuitOpModel]:
            ops_out: List[CircuitOpModel] = []
            sc = 0
            for ci in rand_c:
                for p_tok in _CLIFFORD_1Q_PULSE_SEQS[ci]:
                    g_nm, g_pr = dsl_map[p_tok]
                    if g_nm == "I" and len(_CLIFFORD_1Q_PULSE_SEQS[ci]) > 1:
                        continue
                    ops_out.append(
                        CircuitOpModel(
                            id=f"{tag}_{sc}",
                            step=sc,
                            kind="gate",
                            name=g_nm,
                            qubits=[q0],
                            params=list(g_pr),
                        )
                    )
                    sc += 1
                if inter_gate_nm:
                    ig_nm, ig_pr = inter_dsl_map.get(inter_gate_nm, (inter_gate_nm, []))
                    ops_out.append(
                        CircuitOpModel(
                            id=f"{tag}_ir_{sc}",
                            step=sc,
                            kind="gate",
                            name=ig_nm,
                            qubits=[q0],
                            params=list(ig_pr),
                        )
                    )
                    sc += 1
            for p_tok in _CLIFFORD_1Q_PULSE_SEQS[inv_c]:
                g_nm, g_pr = dsl_map[p_tok]
                if g_nm == "I" and len(_CLIFFORD_1Q_PULSE_SEQS[inv_c]) > 1:
                    continue
                ops_out.append(
                    CircuitOpModel(
                        id=f"{tag}_inv_{sc}",
                        step=sc,
                        kind="gate",
                        name=g_nm,
                        qubits=[q0],
                        params=list(g_pr),
                    )
                )
                sc += 1
            ops_out.append(
                CircuitOpModel(id=f"{tag}_m", step=sc, kind="measure", name="MEASURE", qubits=[q0], params=[])
            )
            return ops_out

        rb_circuit_samples: List[Dict[str, Any]] = []
        for m_val, c_list, inv_c in ref_seq_samples:
            ops_m = build_1q_rb_ops(f"rb_m{m_val}", c_list, inv_c, None)
            rb_circuit_samples.append(
                {
                    "id": f"rb_ref_m{m_val}",
                    "label": f"参考 RB (m={m_val})",
                    "subtitle": f"{m_val} 个随机 C₁ Clifford 脉冲分解 + 逆恢复门 C_inv → 测量",
                    "operations": [op.model_dump() for op in ops_m],
                }
            )
            if not sample_ops or m_val == ref_seq_samples[-1][0]:
                sample_ops = ops_m

        for m_val, c_list, inv_c in irb_seq_samples[:2]:
            ops_irb = build_1q_rb_ops(f"irb_m{m_val}", c_list, inv_c, interleaved_name)
            rb_circuit_samples.append(
                {
                    "id": f"rb_irb_m{m_val}",
                    "label": f"交织 [{interleaved_name}] IRB (m={m_val})",
                    "subtitle": f"每级随机 Clifford 后插入待测门 {interleaved_name} + 总逆恢复门 C_inv → 测量",
                    "operations": [op.model_dump() for op in ops_irb],
                }
            )

        dim = 2
        baseline_b = 0.5
        # Coherence-limited EPC floor from average physical pulse duration per Clifford
        avg_cliff_dur_ns = avg_phys_pulses * noise0["t_1q_ns"]
        gamma1 = (avg_cliff_dur_ns / noise0["t1_ns"]) if math.isfinite(noise0["t1_ns"]) and noise0["t1_ns"] > 0 else 0.0
        gamma2 = (avg_cliff_dur_ns / noise0["t2_ns"]) if math.isfinite(noise0["t2_ns"]) and noise0["t2_ns"] > 0 else 0.0
        coherence_epc_floor = float(min(0.5, max(0.0, (gamma1 / 6.0) + (gamma2 / 3.0))))
        target_label = project.qubits[q0].name

    else:
        # ---- 2-Qubit Randomized Benchmarking (Stim Tableau C_2 + 16x16 Superoperators) ----
        noise1 = _extract_qubit_noise_params(project, q1)
        prim0 = _build_primitive_1q_library(
            noise0, req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg
        )
        prim1 = _build_primitive_1q_library(
            noise1, req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg
        )

        def lift_1q_super(s1: np.ndarray, which: int) -> np.ndarray:
            # Lift 4x4 vec-superoperator on qubit 0 or 1 to 16x16 vec-superoperator on 2 qubits!
            # Recall U_2Q = U_0 ⊗ U_1 => S_2Q = (U_0 ⊗ U_1) ⊗ (U_0* ⊗ U_1*).
            # By permuting tensor axes (i0, i1, j0, j1) <-> (i0, j0, i1, j1):
            s0_mat = s1 if which == 0 else np.eye(4, dtype=np.complex128)
            s1_mat = s1 if which == 1 else np.eye(4, dtype=np.complex128)
            t8 = np.kron(s0_mat, s1_mat).reshape(2, 2, 2, 2, 2, 2, 2, 2)
            # s0 has indices (r0, c0, r0*, c0*)? Wait: s = kron(K, K*), so row=(r, r*), col=(c, c*).
            # In kron(s0, s1), row=(r0, r0*, r1, r1*), col=(c0, c0*, c1, c1*).
            # Target 2Q vec has row=(r0, r1, r0*, r1*), col=(c0, c1, c0*, c1*)!
            perm = np.transpose(t8, (0, 2, 1, 3, 4, 6, 5, 7)).reshape(16, 16)
            return perm

        local_1q_0 = {k: (lift_1q_super(v[0], 0), np.kron(v[1], _I2)) for k, v in prim0.items()}
        local_1q_1 = {k: (lift_1q_super(v[0], 1), np.kron(_I2, v[1])) for k, v in prim1.items()}

        def build_2q_entangler(g2_name: str) -> Tuple[np.ndarray, np.ndarray]:
            u_id = gate_unitary(g2_name)
            t2q = noise0["t_2q_ns"]
            # ZZ crosstalk + coherent amplitude error during 2Q entangling gate
            zz_rad = 2.0 * math.pi * (noise0["zz_khz"] * 1e-6) * t2q
            amp_rad = (req.amp_error_pct / 100.0) * (0.25 * math.pi)
            u_err = expm(-0.5j * (zz_rad + amp_rad) * np.kron(_SZ, _SZ))
            u_phys = u_err @ u_id
            s2 = _unitary_to_liouville(u_phys)
            if t2q > 0:
                if math.isfinite(noise0["t1_ns"]) or math.isfinite(noise0["t2_ns"]):
                    tr0 = NoiseChannel.thermal_relaxation(0, noise0["t1_ns"], noise0["t2_ns"], t2q)
                    s2 = lift_1q_super(_kraus_to_liouville(tr0.kraus_operators()), 0) @ s2
                if math.isfinite(noise1["t1_ns"]) or math.isfinite(noise1["t2_ns"]):
                    tr1 = NoiseChannel.thermal_relaxation(1, noise1["t1_ns"], noise1["t2_ns"], t2q)
                    s2 = lift_1q_super(_kraus_to_liouville(tr1.kraus_operators()), 1) @ s2
            if noise0["depol_2q"] > 1e-12:
                dep2 = NoiseChannel.depolarizing_2q(0, 1, min(0.90, noise0["depol_2q"]))
                s2 = _kraus_to_liouville(dep2.kraus_operators()) @ s2
            return s2, u_id

        s_cx_01, _ = build_2q_entangler("CNOT")
        # CNOT(1->0) = SWAP @ CNOT(0->1) @ SWAP
        u_swap = gate_unitary("SWAP")
        s_swap = _unitary_to_liouville(u_swap)
        s_cx_10 = s_swap @ s_cx_01 @ s_swap
        s_cz, u_cz = build_2q_entangler("CZ")
        s_iswap, u_iswap = build_2q_entangler("ISWAP")

        stim_1q_map = {
            "I": "I",
            "X": "X",
            "Y": "Y",
            "Z": "Z",
            "H": "H",
            "S": "S",
            "S_DAG": "SDG",
            "SQRT_X": "X90",
            "SQRT_X_DAG": "Xm90",
            "SQRT_Y": "Y90",
            "SQRT_Y_DAG": "Ym90",
        }

        def tableau_to_superop(tab: stim.Tableau) -> Tuple[np.ndarray, List[Tuple[str, List[int]]]]:
            circ = tab.to_circuit(method="elimination")
            s_acc = np.eye(16, dtype=np.complex128)
            gate_trace: List[Tuple[str, List[int]]] = []
            for inst in circ:
                nm = inst.name
                targs = [t.value for t in inst.targets_copy()]
                if nm in stim_1q_map:
                    p_key = stim_1q_map[nm]
                    for q_loc in targs:
                        s_g = local_1q_0[p_key][0] if q_loc == 0 else local_1q_1[p_key][0]
                        s_acc = s_g @ s_acc
                        gate_trace.append((p_key, [q0 if q_loc == 0 else q1]))
                elif nm in ("CX", "CNOT"):
                    for j in range(0, len(targs), 2):
                        c_loc, t_loc = targs[j], targs[j + 1]
                        s_acc = (s_cx_01 if (c_loc == 0 and t_loc == 1) else s_cx_10) @ s_acc
                        gate_trace.append(("CNOT", [q0 if c_loc == 0 else q1, q0 if t_loc == 0 else q1]))
                elif nm == "CZ":
                    for j in range(0, len(targs), 2):
                        s_acc = s_cz @ s_acc
                        gate_trace.append(("CZ", [q0, q1]))
                elif nm == "SWAP":
                    for j in range(0, len(targs), 2):
                        s_acc = s_cx_01 @ s_cx_10 @ s_cx_01 @ s_acc
                        gate_trace.append(("SWAP", [q0, q1]))
            return s_acc, gate_trace

        # Prebuild 24 1Q Cliffords lifted to 16x16 on qubit 0 and qubit 1 + 3 axis-cycling V gates
        c1_0_sup: List[np.ndarray] = []
        c1_0_u: List[np.ndarray] = []
        c1_1_sup: List[np.ndarray] = []
        c1_1_u: List[np.ndarray] = []
        for seq in _CLIFFORD_1Q_PULSE_SEQS:
            s0_a = np.eye(16, dtype=np.complex128)
            u0_a = np.eye(4, dtype=np.complex128)
            s1_a = np.eye(16, dtype=np.complex128)
            u1_a = np.eye(4, dtype=np.complex128)
            for p_k in seq:
                s0_a = local_1q_0[p_k][0] @ s0_a
                u0_a = local_1q_0[p_k][1] @ u0_a
                s1_a = local_1q_1[p_k][0] @ s1_a
                u1_a = local_1q_1[p_k][1] @ u1_a
            c1_0_sup.append(s0_a)
            c1_0_u.append(u0_a)
            c1_1_sup.append(s1_a)
            c1_1_u.append(u1_a)

        v_seqs: List[Tuple[str, ...]] = [("I",), ("X90", "Y90"), ("Ym90", "Xm90")]
        v0_sup = [np.eye(16, dtype=np.complex128)]
        v0_u = [np.eye(4, dtype=np.complex128)]
        v1_sup = [np.eye(16, dtype=np.complex128)]
        v1_u = [np.eye(4, dtype=np.complex128)]
        for vs in v_seqs[1:]:
            sa0 = np.eye(16, dtype=np.complex128)
            ua0 = np.eye(4, dtype=np.complex128)
            sa1 = np.eye(16, dtype=np.complex128)
            ua1 = np.eye(4, dtype=np.complex128)
            for pk in vs:
                sa0 = local_1q_0[pk][0] @ sa0
                ua0 = local_1q_0[pk][1] @ ua0
                sa1 = local_1q_1[pk][0] @ sa1
                ua1 = local_1q_1[pk][1] @ ua1
            v0_sup.append(sa0)
            v0_u.append(ua0)
            v1_sup.append(sa1)
            v1_u.append(ua1)

        u_cx_01 = gate_unitary("CNOT")
        u_cx_10 = u_swap @ u_cx_01 @ u_swap
        s_swap_phys = s_cx_01 @ s_cx_10 @ s_cx_01

        def sample_canonical_2q_clifford(c2_idx: int, rec_trace: bool) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, List[int]]]]:
            """Deterministic Koenig-Smolin bijection over all 11,520 elements of C_2 (avg 1.5 two-qubit gates)."""
            i0 = c2_idx % 24
            i1 = (c2_idx // 24) % 24
            cls_id = c2_idx // 576  # 0..19
            s_cur = c1_1_sup[i1] @ c1_0_sup[i0]
            u_cur = c1_1_u[i1] @ c1_0_u[i0]
            tr: List[Tuple[str, List[int]]] = []
            if rec_trace:
                for pk in _CLIFFORD_1Q_PULSE_SEQS[i0]:
                    tr.append((pk, [q0]))
                for pk in _CLIFFORD_1Q_PULSE_SEQS[i1]:
                    tr.append((pk, [q1]))
            if cls_id == 0:
                return s_cur, u_cur, tr
            if 1 <= cls_id <= 9:
                va = (cls_id - 1) // 3
                vb = (cls_id - 1) % 3
                s_cur = v1_sup[vb] @ v0_sup[va] @ s_cx_01 @ s_cur
                u_cur = v1_u[vb] @ v0_u[va] @ u_cx_01 @ u_cur
                if rec_trace:
                    tr.append(("CNOT", [q0, q1]))
                return s_cur, u_cur, tr
            if 10 <= cls_id <= 18:
                va = (cls_id - 10) // 3
                vb = (cls_id - 10) % 3
                s_cur = s_cx_10 @ v1_sup[vb] @ v0_sup[va] @ s_cx_01 @ s_cur
                u_cur = u_cx_10 @ v1_u[vb] @ v0_u[va] @ u_cx_01 @ u_cur
                if rec_trace:
                    tr.append(("CNOT", [q0, q1]))
                    tr.append(("CNOT", [q1, q0]))
                return s_cur, u_cur, tr
            # cls_id == 19: SWAP class
            s_cur = s_swap_phys @ s_cur
            u_cur = u_swap @ u_cur
            if rec_trace:
                tr.append(("SWAP", [q0, q1]))
            return s_cur, u_cur, tr

        # Cap 2Q RB max length at 64 so response stays snappy (< 80 ms)
        raw_lens = [m for m in raw_lens if m <= 64] or [1, 2, 4, 8, 16, 32]
        num_seeds = min(num_seeds, 20)

        inter_u_2q: Optional[np.ndarray] = None
        inter_super_2q: Optional[np.ndarray] = None
        if interleaved_name:
            if interleaved_name in ("CZ", ""):
                inter_u_2q = u_cz
                inter_super_2q = s_cz
                interleaved_name = "CZ"
            elif interleaved_name in ("CNOT", "CX"):
                inter_u_2q = u_cx_01
                inter_super_2q = s_cx_01
                interleaved_name = "CNOT"
            elif interleaved_name == "ISWAP":
                inter_u_2q = u_iswap
                inter_super_2q = s_iswap
            else:
                inter_u_2q = u_cz
                inter_super_2q = s_cz
                interleaved_name = "CZ"

        rho0_2q = np.zeros((4, 4), dtype=np.complex128)
        rho0_2q[0, 0] = 1.0
        rho0_vec = rho0_2q.reshape(-1)
        e0_0 = np.diag([1.0 - noise0["p1_given_0"], noise0["p0_given_1"]])
        e0_1 = np.diag([1.0 - noise1["p1_given_0"], noise1["p0_given_1"]])
        e00_bra = np.kron(e0_0, e0_1).reshape(-1).astype(np.complex128).conj()

        def eval_2q_rb(
            use_inter: bool,
        ) -> Tuple[List[float], List[float], List[List[float]], List[Tuple[int, List[Tuple[str, List[int]]]]]]:
            # Paired RNG seed across ref & interleaved runs for low-variance IRB comparison!
            loc_rng = np.random.default_rng(req.seed)
            means: List[float] = []
            stds: List[float] = []
            raw_samples: List[List[float]] = []
            sampled_traces: List[Tuple[int, List[Tuple[str, List[int]]]]] = []
            sample_m_set = {0, min(1, len(raw_lens) - 1)}
            for m_idx, m_len in enumerate(raw_lens):
                seq_probs: List[float] = []
                for k in range(num_seeds):
                    picks = loc_rng.integers(0, 11520, size=m_len)
                    s_seq = np.eye(16, dtype=np.complex128)
                    u_seq = np.eye(4, dtype=np.complex128)
                    local_trace: List[Tuple[str, List[int]]] = []
                    rec = (m_idx in sample_m_set) and k == 0
                    for c2_i in picks:
                        s_c, u_c, tr_c = sample_canonical_2q_clifford(int(c2_i), rec)
                        s_seq = s_c @ s_seq
                        u_seq = u_c @ u_seq
                        if rec:
                            local_trace.extend(tr_c)
                        if use_inter and inter_u_2q is not None and inter_super_2q is not None:
                            s_seq = inter_super_2q @ s_seq
                            u_seq = inter_u_2q @ u_seq
                            if rec and interleaved_name:
                                local_trace.append((interleaved_name, [q0, q1]))
                    t_inv = stim.Tableau.from_unitary_matrix(u_seq.conj().T, endian="big")
                    s_inv, tr_inv = tableau_to_superop(t_inv)
                    if rec:
                        local_trace.extend(tr_inv)
                        sampled_traces.append((m_len, local_trace))
                    s_full = s_inv @ s_seq
                    p_surv = float(np.clip(np.real(e00_bra @ (s_full @ rho0_vec)), 0.0, 1.0))
                    if shots > 0:
                        p_surv = float(loc_rng.binomial(shots, p_surv)) / float(shots)
                    seq_probs.append(round(p_surv, 6))
                arr = np.array(seq_probs, dtype=np.float64)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0)
                raw_samples.append(seq_probs)
            return means, stds, raw_samples, sampled_traces

        ref_means, ref_stds, ref_raw, ref_2q_traces = eval_2q_rb(False)
        irb_means = None
        irb_stds = None
        irb_raw = None
        irb_2q_traces: List[Tuple[int, List[Tuple[str, List[int]]]]] = []
        if inter_super_2q is not None:
            irb_means, irb_stds, irb_raw, irb_2q_traces = eval_2q_rb(True)

        name_remap = {"X90": "SX", "Xm90": "RX", "Y90": "RY", "Ym90": "RY"}
        param_remap: Dict[str, List[float]] = {
            "Xm90": [-round(0.5 * math.pi, 4)],
            "Y90": [round(0.5 * math.pi, 4)],
            "Ym90": [-round(0.5 * math.pi, 4)],
        }

        def trace_2q_to_ops(tag: str, tr_list: List[Tuple[str, List[int]]]) -> List[CircuitOpModel]:
            out_ops: List[CircuitOpModel] = []
            for idx_op, (gnm, qlist) in enumerate(tr_list[:48]):
                if gnm == "I" and len(tr_list) > 2:
                    continue
                step_i = len(out_ops)
                out_ops.append(
                    CircuitOpModel(
                        id=f"{tag}_{step_i}",
                        step=step_i,
                        kind="gate",
                        name=name_remap.get(gnm, gnm),
                        qubits=qlist,
                        params=list(param_remap.get(gnm, [])),
                    )
                )
            out_ops.append(
                CircuitOpModel(
                    id=f"{tag}_m",
                    step=len(out_ops),
                    kind="measure",
                    name="MEASURE",
                    qubits=[q0, q1],
                    params=[],
                )
            )
            return out_ops

        rb_circuit_samples = []
        for m_val, tr_seq in ref_2q_traces:
            ops_2q = trace_2q_to_ops(f"rb2_m{m_val}", tr_seq)
            rb_circuit_samples.append(
                {
                    "id": f"rb2_ref_m{m_val}",
                    "label": f"参考 2Q RB (m={m_val})",
                    "subtitle": f"{m_val} 个随机 C₂ 双比特 Clifford (Koenig-Smolin 分解) + 逆恢复门 → 联合测量",
                    "operations": [op.model_dump() for op in ops_2q],
                }
            )
            if not sample_ops:
                sample_ops = ops_2q

        for m_val, tr_seq in irb_2q_traces:
            ops_2q_irb = trace_2q_to_ops(f"irb2_m{m_val}", tr_seq)
            rb_circuit_samples.append(
                {
                    "id": f"rb2_irb_m{m_val}",
                    "label": f"交织 [{interleaved_name}] 2Q IRB (m={m_val})",
                    "subtitle": f"每级 C₂ 随机门后交织双比特门 {interleaved_name} + Stim Tableau 逆恢复 → 测量",
                    "operations": [op.model_dump() for op in ops_2q_irb],
                }
            )

        dim = 4
        baseline_b = 0.25
        # Average ~1.5 two-qubit gates per 2Q Clifford
        avg_2q_dur = 1.5 * noise0["t_2q_ns"] + 3.0 * noise0["t_1q_ns"]
        g1 = (avg_2q_dur / noise0["t1_ns"]) if math.isfinite(noise0["t1_ns"]) and noise0["t1_ns"] > 0 else 0.0
        g2 = (avg_2q_dur / noise0["t2_ns"]) if math.isfinite(noise0["t2_ns"]) and noise0["t2_ns"] > 0 else 0.0
        coherence_epc_floor = float(min(0.75, max(0.0, 2.0 * ((g1 / 6.0) + (g2 / 3.0)))))
        target_label = f"{project.qubits[q0].name}, {project.qubits[q1].name}"

    # ---- Fit exponential decay P(m) = A * p^m + B ----
    m_arr = np.array(raw_lens, dtype=np.float64)

    def fit_rb_decay(y_vals: Sequence[float], fixed_b: Optional[float] = None) -> Tuple[float, float, float]:
        y = np.array(y_vals, dtype=np.float64)
        b_anchor = fixed_b if fixed_b is not None else baseline_b
        # Check if survival curve is flat (noiseless gates, regardless of SPAM readout visibility!)
        if float(np.max(y) - np.min(y)) < 1e-5 and float(np.mean(y)) > b_anchor + 0.02:
            return float(np.mean(y) - b_anchor), 1.0, b_anchor

        a_init = max(0.05, float(y[0]) - b_anchor)
        y_shifted = np.clip((y - b_anchor) / max(1e-4, a_init), 1e-4, 1.0)
        slope = float(np.polyfit(m_arr, np.log(y_shifted), 1)[0])
        p_init = float(np.clip(math.exp(slope), 0.2, 0.99999))

        # When sequence hasn't decayed near the asymptote yet (shallow high-fidelity regime),
        # constrain B tightly around the twirled maximally-mixed asymptote to avoid A/B collinearity!
        shallow = float(np.min(y)) > b_anchor + 0.18
        b_lo = max(0.0, b_anchor - (0.015 if fixed_b is not None else (0.035 if shallow else 0.25)))
        b_hi = min(1.0, b_anchor + (0.015 if fixed_b is not None else (0.045 if shallow else 0.25)))

        def model_fn(m: np.ndarray, a: float, p: float, b: float) -> np.ndarray:
            return a * np.power(p, m) + b

        try:
            popt, _ = curve_fit(
                model_fn,
                m_arr,
                y,
                p0=[a_init, p_init, float(np.clip(b_anchor, b_lo, b_hi))],
                bounds=([0.0, 0.0, b_lo], [1.0, 1.0, b_hi]),
                maxfev=4000,
            )
            return float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            return a_init, p_init, b_anchor

    a_ref, p_ref, b_ref = fit_rb_decay(ref_means)
    epc_ref = ((dim - 1.0) / dim) * (1.0 - p_ref)
    f_cliff = 1.0 - epc_ref

    dense_m = np.linspace(0.0, float(max(raw_lens)), 80)
    ref_fit_y = (a_ref * np.power(p_ref, dense_m) + b_ref).tolist()

    irb_summary: Optional[Dict[str, Any]] = None
    if irb_means is not None and irb_stds is not None and irb_raw is not None:
        a_irb, p_irb, b_irb = fit_rb_decay(irb_means, fixed_b=b_ref)
        p_ratio = min(1.0, p_irb / max(1e-9, p_ref))
        r_gate = ((dim - 1.0) / dim) * (1.0 - p_ratio)
        # Magesan et al. (PRL 109, 080505) systematic error bound E
        e_bound = min(
            ((dim - 1.0) / dim) * (abs(p_ref - p_ratio) + (1.0 - p_ref)),
            (2.0 * (dim * dim - 1.0) / (dim * dim)) * (1.0 - p_ref)
            + 4.0 * math.sqrt(max(0.0, 1.0 - p_ref)) * math.sqrt(dim * dim - 1.0) / dim,
        )
        irb_fit_y = (a_irb * np.power(p_irb, dense_m) + b_irb).tolist()
        irb_summary = {
            "gate_name": interleaved_name,
            "mean_survival": [round(v, 6) for v in irb_means],
            "std_survival": [round(v, 6) for v in irb_stds],
            "raw_survival": irb_raw,
            "fit_y": [round(v, 6) for v in irb_fit_y],
            "p_irb": round(p_irb, 6),
            "a_irb": round(a_irb, 5),
            "b_irb": round(b_irb, 5),
            "gate_error": round(max(0.0, r_gate), 6),
            "gate_fidelity": round(min(1.0, 1.0 - max(0.0, r_gate)), 6),
            "systematic_bound": round(float(e_bound), 6),
        }

    return {
        "protocol": "rb",
        "hardware_name": noise0["hardware_name"],
        "target_label": target_label,
        "num_qubits": 2 if is_2q else 1,
        "dimension": dim,
        "lengths": raw_lens,
        "num_seeds": num_seeds,
        "shots": shots,
        "fit_x": [round(float(x), 3) for x in dense_m],
        "mean_survival": [round(v, 6) for v in ref_means],
        "std_survival": [round(v, 6) for v in ref_stds],
        "raw_survival": ref_raw,
        "fit_y": [round(v, 6) for v in ref_fit_y],
        "a_spam": round(a_ref, 5),
        "b_offset": round(b_ref, 5),
        "depolarizing_p": round(p_ref, 6),
        "epc": round(epc_ref, 6),
        "clifford_fidelity": round(f_cliff, 6),
        "coherence_epc_floor": round(coherence_epc_floor, 6),
        "interleaved": irb_summary,
        "sample_operations": [op.model_dump() for op in sample_ops],
        "circuit_samples": rb_circuit_samples,
    }


# ============================================================================
# 2. AllXY 21-Sequence Coherent & Incoherent Pulse Calibration Diagnostic
# ============================================================================

_ALLXY_SEQUENCES: List[Tuple[str, str, float, str]] = [
    # Group 1: Ground state plateau (<Z> = +1, P(|1>) = 0)
    ("I", "I", 1.0, "ground"),
    ("X", "X", 1.0, "ground"),
    ("Y", "Y", 1.0, "ground"),
    ("X", "Y", 1.0, "ground"),
    ("Y", "X", 1.0, "ground"),
    # Group 2: Equatorial superposition plateau (<Z> = 0, P(|1>) = 0.5)
    ("X90", "I", 0.0, "equator"),
    ("Y90", "I", 0.0, "equator"),
    ("X90", "Y90", 0.0, "equator"),
    ("Y90", "X90", 0.0, "equator"),
    ("X90", "Y", 0.0, "equator"),
    ("Y90", "X", 0.0, "equator"),
    ("X", "Y90", 0.0, "equator"),
    ("Y", "X90", 0.0, "equator"),
    ("X90", "X", 0.0, "equator"),
    ("X", "X90", 0.0, "equator"),
    ("Y90", "Y", 0.0, "equator"),
    ("Y", "Y90", 0.0, "equator"),
    # Group 3: Excited state inversion plateau (<Z> = -1, P(|1>) = 1)
    ("X", "I", -1.0, "excited"),
    ("Y", "I", -1.0, "excited"),
    ("X90", "X90", -1.0, "excited"),
    ("Y90", "Y90", -1.0, "excited"),
]


def _run_allxy_protocol(req: GateBenchmarkRequest) -> Dict[str, Any]:
    project = req.project
    q0 = max(0, min(req.target_qubit, len(project.qubits) - 1))
    noise0 = _extract_qubit_noise_params(project, q0)
    prim = _build_primitive_1q_library(
        noise0, req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg
    )
    rng = np.random.default_rng(req.seed)
    shots = max(0, int(req.shots))

    rho0_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    p10 = noise0["p1_given_0"]
    p01 = noise0["p0_given_1"]
    e0_bra = np.diag([1.0 - p10, p01]).reshape(-1).astype(np.complex128).conj()

    rows: List[Dict[str, Any]] = []
    sq_errs: List[float] = []

    dsl_gate_map: Dict[str, Tuple[str, List[float]]] = {
        "I": ("I", []),
        "X": ("X", []),
        "Y": ("Y", []),
        "X90": ("SX", []),
        "Y90": ("RY", [round(0.5 * math.pi, 4)]),
    }

    for idx, (g1, g2, ideal_z, grp) in enumerate(_ALLXY_SEQUENCES):
        # Gate order in AllXY(G1, G2): apply G1 first, then G2!
        s1 = prim[g1][0]
        s2 = prim[g2][0]
        rho_out = s2 @ (s1 @ rho0_vec)
        p0_exact = float(np.clip(np.real(e0_bra @ rho_out), 0.0, 1.0))
        if shots > 0:
            c0 = int(rng.binomial(shots, p0_exact))
            p0_meas = c0 / float(shots)
            sigma_p0 = math.sqrt(max(1e-9, p0_meas * (1.0 - p0_meas) / float(shots)))
        else:
            p0_meas = p0_exact
            sigma_p0 = 0.0

        z_meas = 2.0 * p0_meas - 1.0
        sigma_z = 2.0 * sigma_p0
        p1_meas = 1.0 - p0_meas
        ideal_p1 = 0.5 * (1.0 - ideal_z)
        err_z = z_meas - ideal_z
        sq_errs.append(err_z * err_z)

        # Build CircuitOpModel pair for clicking any AllXY sequence into Design canvas
        seq_ops: List[Dict[str, Any]] = []
        st = 0
        for g_tok in (g1, g2):
            nm, prm = dsl_gate_map[g_tok]
            seq_ops.append(
                CircuitOpModel(
                    id=f"axy_{idx}_{st}",
                    step=st,
                    kind="gate",
                    name=nm,
                    qubits=[q0],
                    params=prm,
                ).model_dump()
            )
            st += 1
        seq_ops.append(
            CircuitOpModel(
                id=f"axy_{idx}_m",
                step=st,
                kind="measure",
                name="MEASURE",
                qubits=[q0],
                params=[],
            ).model_dump()
        )

        rows.append(
            {
                "index": idx + 1,
                "label": f"{g1}·{g2}",
                "gate_first": g1,
                "gate_second": g2,
                "plateau": grp,
                "ideal_z": round(ideal_z, 4),
                "measured_z": round(z_meas, 5),
                "std_z": round(sigma_z, 5),
                "ideal_p1": round(ideal_p1, 4),
                "measured_p1": round(p1_meas, 5),
                "residual_z": round(err_z, 5),
                "operations": seq_ops,
            }
        )

    rms_z = math.sqrt(sum(sq_errs) / len(sq_errs))

    # Physics-informed signature extraction:
    # 1. Amplitude error signature: in AllXY, X90·I (#6, idx 5) has <Z> ≈ -pi/2 * eps_amp,
    #    whereas X90·X (#14, idx 13) and X·X90 (#15, idx 14) have <Z> ≈ +3*pi/2 * eps_amp!
    amp_sig = 0.5 * ((rows[13]["measured_z"] + rows[14]["measured_z"]) * 0.5 - rows[5]["measured_z"])
    est_amp_pct = (amp_sig / math.pi) * 100.0

    # 2. Quadrature phase skew / detuning signature:
    #    X90·Y90 (#8, idx 7) vs Y90·X90 (#9, idx 8) and X90·Y (#10, idx 9) vs Y90·X (#11, idx 10)
    phase_split = abs(rows[7]["measured_z"] - rows[8]["measured_z"]) + abs(
        rows[9]["measured_z"] - rows[10]["measured_z"]
    )
    detuning_asym = 0.5 * (rows[7]["measured_z"] + rows[8]["measured_z"])

    if rms_z < 0.018:
        verdict = "标定优良：AllXY 21 序列与理想阶梯高度吻合，未检出显著相干脉冲失配。"
        severity = "good"
    elif abs(est_amp_pct) >= 1.2 and abs(amp_sig) >= phase_split * 0.6:
        direction = "过旋转 (Over-rotation) " if est_amp_pct > 0 else "欠旋转 (Under-rotation) "
        verdict = (
            f"检出典型拉比幅度{direction}锯齿特征：赤道平台 X90·I 与 X90·X 反向偏离，"
            f"估计脉冲幅度偏差约 {est_amp_pct:+.2f}%。"
        )
        severity = "warn"
    elif phase_split >= 0.035 or abs(detuning_asym) >= 0.035:
        verdict = (
            f"检出失谐 / IQ 正交相位偏斜特征：X90·Y90 与 Y90·X90 呈非对称偏离 "
            f"(Δ⟨Z⟩ = {phase_split:.4f})，建议校准本振频率与正交相位。"
        )
        severity = "warn"
    else:
        verdict = (
            f"检出以非相干退相干 / 读出压缩为主的阶梯收缩 (RMS Δ⟨Z⟩ = {rms_z:.4f})，"
            "单脉冲旋转角与正交轴向标定基本准确。"
        )
        severity = "info"

    return {
        "protocol": "allxy",
        "hardware_name": noise0["hardware_name"],
        "target_label": project.qubits[q0].name,
        "shots": shots,
        "rms_error_z": round(rms_z, 6),
        "estimated_amp_error_pct": round(est_amp_pct, 3),
        "phase_detuning_metric": round(float(phase_split + abs(detuning_asym)), 5),
        "diagnosis": verdict,
        "severity": severity,
        "sequences": rows,
    }


# ============================================================================
# 3. Gate Set Tomography (GST — Gauge-Aligned Linear Inversion + Germ Powers)
# ============================================================================

def _run_gst_protocol(req: GateBenchmarkRequest) -> Dict[str, Any]:
    """
    Reconstruct the 1-qubit gate set G = {Gi, Gx90, Gy90, Gx, Gy} via Linear Inversion GST
    with optional Germ-Power coherent angle/axis refinement (L in {1..gst_max_length}).
    """
    project = req.project
    q0 = max(0, min(req.target_qubit, len(project.qubits) - 1))
    noise0 = _extract_qubit_noise_params(project, q0)
    prim = _build_primitive_1q_library(
        noise0, req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg
    )
    rng = np.random.default_rng(req.seed)
    shots = max(0, int(req.shots))
    l_max = max(1, min(16, int(req.gst_max_length)))

    gate_keys = [
        ("Gi", "I", "Idle 空闲门 I"),
        ("Gx90", "X90", "√X 脉冲 RX(π/2)"),
        ("Gy90", "Y90", "√Y 脉冲 RY(π/2)"),
        ("Gx", "X", "π 翻转脉冲 X"),
        ("Gy", "Y", "π 翻转脉冲 Y"),
        ("Gh", "H", "Hadamard 门 H"),
    ]

    # 6 informationally complete preparation fiducials F_j (acting on |0>)
    # States: |0>, |-i>, |+>, |1>, |+i>, |->
    prep_fiducials: List[Tuple[str, ...]] = [
        (),
        ("X90",),
        ("Y90",),
        ("X90", "X90"),
        ("X90", "X90", "X90"),
        ("Y90", "Y90", "Y90"),
    ]
    # 4 informationally complete measurement fiducials M_i (followed by Z-basis readout E_0)
    # Effective observables: +Z, +Y (after X90), -X (after Y90), -Z (after X90*X90)
    meas_fiducials: List[Tuple[str, ...]] = [
        (),
        ("X90",),
        ("Y90",),
        ("X90", "X90"),
    ]

    rho0_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    p10 = noise0["p1_given_0"]
    p01 = noise0["p0_given_1"]
    e0_noisy_bra = np.diag([1.0 - p10, p01]).reshape(-1).astype(np.complex128).conj()
    e0_ideal_bra = np.diag([1.0, 0.0]).reshape(-1).astype(np.complex128).conj()

    def seq_super(seq: Sequence[str], ideal: bool = False) -> np.ndarray:
        s_acc = np.eye(4, dtype=np.complex128)
        for g_tok in seq:
            s_step = _unitary_to_liouville(prim[g_tok][1]) if ideal else prim[g_tok][0]
            s_acc = s_step @ s_acc
        return s_acc

    def sample_prob(seq: Sequence[str]) -> float:
        s_tot = seq_super(seq, ideal=False)
        p_ex = float(np.clip(np.real(e0_noisy_bra @ (s_tot @ rho0_vec)), 0.0, 1.0))
        if shots > 0:
            return float(rng.binomial(shots, p_ex)) / float(shots)
        return p_ex

    # Build ideal Pauli-basis target fiducial matrices:
    # Ideal prep states in Pauli basis (4 x 6): col j = [1, rx, ry, rz]^T / sqrt(2)
    m_pauli = np.column_stack([p.reshape(-1) / math.sqrt(2.0) for p in _PAULIS_1Q])
    rho_ideal_pauli = np.zeros((4, len(prep_fiducials)), dtype=np.float64)
    for j, pf in enumerate(prep_fiducials):
        v = seq_super(pf, ideal=True) @ rho0_vec
        rho_ideal_pauli[:, j] = np.real(m_pauli.conj().T @ v)

    e_ideal_pauli = np.zeros((len(meas_fiducials), 4), dtype=np.float64)
    for i, mf in enumerate(meas_fiducials):
        bra_v = e0_ideal_bra @ seq_super(mf, ideal=True)
        e_ideal_pauli[i, :] = np.real(bra_v @ m_pauli)

    # Measure 4 x 6 Gram matrix g_{ij} = P(0 | M_i F_j)
    gram = np.zeros((len(meas_fiducials), len(prep_fiducials)), dtype=np.float64)
    for i, mf in enumerate(meas_fiducials):
        for j, pf in enumerate(prep_fiducials):
            gram[i, j] = sample_prob(list(pf) + list(mf))

    # Measure 4 x 6 process data tables T_k for each gate k: (T_k)_{ij} = P(0 | M_i G_k F_j)
    t_tables: Dict[str, np.ndarray] = {}
    for g_id, p_tok, _ in gate_keys:
        tk = np.zeros((len(meas_fiducials), len(prep_fiducials)), dtype=np.float64)
        for i, mf in enumerate(meas_fiducials):
            for j, pf in enumerate(prep_fiducials):
                tk[i, j] = sample_prob(list(pf) + [p_tok] + list(mf))
        t_tables[g_id] = tk

    # Exact Subspace Linear GST Inversion (Greenbaum / Blume-Kohout pyGSTi formulation):
    # Project 4x6 Gram matrix onto its exact 4D right singular subspace W (6x4, W^T W = I_4)
    # so g_sq = gram @ W = E_eff @ (Sigma_eff @ W) is an exact 4x4 invertible product,
    # giving g_tilde_k = inv(g_sq) @ (T_k @ W) = tilde_Sigma^{-1} G_k tilde_Sigma with ZERO rank leakage!
    _, _s_vals, vt = np.linalg.svd(gram, full_matrices=False)
    w_right = vt.T  # 6 x 4
    g_sq = gram @ w_right  # 4 x 4
    g_sq_inv = np.linalg.pinv(g_sq, rcond=1e-6)

    tilde_ptms: Dict[str, np.ndarray] = {}
    ideal_ptms: Dict[str, np.ndarray] = {}
    for g_id, p_tok, _ in gate_keys:
        tilde_ptms[g_id] = g_sq_inv @ (t_tables[g_id] @ w_right)
        ideal_ptms[g_id] = _liouville_to_ptm_1q(_unitary_to_liouville(prim[p_tok][1]))

    # First map from the 4D right-singular basis W into the target Pauli basis via B_0 = Sigma_ideal @ W
    # (Do NOT overwrite B_0[0, :] because W is in SVD coordinates, whereas B_0 @ tilde_G @ B_0^{-1}
    #  automatically has exact trace-preserving top row [1, 0, 0, 0] in Pauli coordinates!)
    b0 = rho_ideal_pauli @ w_right
    b0_inv = np.linalg.pinv(b0, rcond=1e-6)
    pauli_seed_ptms = {g_id: (b0 @ tilde_ptms[g_id] @ b0_inv) for g_id, _, _ in gate_keys}
    rho0_seed_pauli = b0 @ (g_sq_inv @ gram[:, 0])
    rho0_target_pauli = rho_ideal_pauli[:, 0]

    # Now refine trace-preserving gauge M in Pauli space (M[0, :] = [1, 0, 0, 0], initialized at I_4)
    # minimizing gate-set + SPAM distance to ideal target:
    m_init = np.eye(4, dtype=np.float64)[1:, :].reshape(-1)

    def gauge_loss(flat12: np.ndarray) -> float:
        m_mat = np.zeros((4, 4), dtype=np.float64)
        m_mat[0, 0] = 1.0
        m_mat[1:, :] = flat12.reshape(3, 4)
        det3 = float(np.linalg.det(m_mat[1:, 1:]))
        if abs(det3) < 1e-4:
            return 1e6
        m_inv = np.linalg.inv(m_mat)
        err_sum = 0.0
        for g_id, _, _ in gate_keys:
            diff = (m_mat @ pauli_seed_ptms[g_id] @ m_inv) - ideal_ptms[g_id]
            err_sum += float(np.sum(diff * diff))
        # Include SPAM vector alignment so M never flips -I_3 sign!
        spam_diff = (m_mat @ rho0_seed_pauli) - rho0_target_pauli
        err_sum += 0.5 * float(np.sum(spam_diff * spam_diff))
        reg = 1e-3 * float(np.sum((m_mat - np.eye(4)) ** 2))
        return err_sum + reg

    opt_g = minimize(gauge_loss, m_init, method="BFGS", options={"maxiter": 40, "gtol": 1e-7})
    m_opt = np.zeros((4, 4), dtype=np.float64)
    m_opt[0, 0] = 1.0
    m_opt[1:, :] = opt_g.x.reshape(3, 4)
    gauge_b = m_opt @ b0
    gauge_b_inv = b0_inv @ np.linalg.inv(m_opt)

    # Optional Germ-Power coherent refinement (L > 1): measure g^L Rabi oscillations on X90 & Y90
    germ_powers = [1]
    p_cur = 2
    while p_cur <= l_max:
        germ_powers.append(p_cur)
        p_cur *= 2

    germ_curves: Dict[str, List[float]] = {}
    for g_tok in ("X90", "Y90"):
        vals_l: List[float] = []
        for lp in germ_powers:
            # Prepare |0>, apply g^lp, measure Z expectation 2*P(0)-1
            p0_l = sample_prob([g_tok] * lp)
            vals_l.append(round(2.0 * p0_l - 1.0, 5))
        germ_curves[g_tok] = vals_l

    # Finalize reconstructed PTMs & compute comprehensive tomography metrics
    gate_reports: List[Dict[str, Any]] = []
    num_circuits = len(prep_fiducials) * len(meas_fiducials) * (1 + len(gate_keys)) + 2 * max(0, len(germ_powers) - 1)

    for g_id, p_tok, disp_name in gate_keys:
        r_aligned = gauge_b @ tilde_ptms[g_id] @ gauge_b_inv
        r_phys = _project_ptm_physical_1q(r_aligned)
        r_ideal = ideal_ptms[g_id]
        r_err = r_phys - r_ideal

        # Process fidelity F_proc = (1/4) Tr(R_ideal^T R_phys) in [0, 1]
        f_proc = float(np.clip(0.25 * np.trace(r_ideal.T @ r_phys), 0.0, 1.0))
        f_avg = (2.0 * f_proc + 1.0) / 3.0
        infid_avg = max(0.0, 1.0 - f_avg)

        # Gauge-invariant Unitarity u(E) = |det(tilde_G)|^(2/3) (since det_4x4(G) = 1 * det_3x3(T) in any GL(4) gauge!)
        det_unital = abs(float(np.linalg.det(tilde_ptms[g_id])))
        unitarity = float(np.clip(det_unital ** (2.0 / 3.0), 0.0, 1.0))
        # Stochastic infidelity lower bound from unitarity: r_stoch = ((d-1)/d) * (1 - sqrt(u))
        infid_incoherent = 0.5 * (1.0 - math.sqrt(unitarity))
        coherent_share = float(
            np.clip((infid_avg - infid_incoherent) / max(1e-9, infid_avg), 0.0, 1.0)
            if infid_avg > 1e-7
            else 0.0
        )

        # Exact 1Q half-diamond distance 0.5 * || E - U ||_diamond
        d_diam = _diamond_distance_1q_ptm(r_phys, r_ideal)

        gate_reports.append(
            {
                "gate_id": g_id,
                "gate_token": p_tok,
                "label": disp_name,
                "process_fidelity": round(f_proc, 6),
                "avg_gate_fidelity": round(f_avg, 6),
                "gate_infidelity": round(infid_avg, 6),
                "diamond_distance": round(d_diam, 6),
                "unitarity": round(unitarity, 6),
                "incoherent_infidelity": round(infid_incoherent, 6),
                "coherent_ratio": round(coherent_share, 4),
                "ptm_ideal": np.round(r_ideal, 4).tolist(),
                "ptm_reconstructed": np.round(r_phys, 4).tolist(),
                "ptm_error": np.round(r_err, 5).tolist(),
            }
        )

    # SPAM diagnostics: reconstructed |0> Bloch vector & readout assignment fidelity
    # Note gram[:, 0] = E_eff @ rho_0, and g_sq = E_eff @ tilde_Sigma, so tilde_Sigma^{-1} rho_0 = g_sq_inv @ gram[:, 0],
    # and in the target gauge frame rho_0_pauli = gauge_b @ (g_sq_inv @ gram[:, 0])!
    rho0_est_pauli = gauge_b @ (g_sq_inv @ gram[:, 0])
    # Recall rho_pauli = [1/sqrt2, x/sqrt2, y/sqrt2, z/sqrt2]
    scale_0 = float(rho0_est_pauli[0]) if abs(rho0_est_pauli[0]) > 1e-6 else (1.0 / math.sqrt(2.0))
    bloch_0 = [
        float(np.clip(rho0_est_pauli[1] / scale_0, -1.0, 1.0)),
        float(np.clip(rho0_est_pauli[2] / scale_0, -1.0, 1.0)),
        float(np.clip(rho0_est_pauli[3] / scale_0, -1.0, 1.0)),
    ]
    state_prep_fid = float(np.clip(0.5 * (1.0 + bloch_0[2]), 0.0, 1.0))
    # Empirical readout contrast from Gram F_0 (|0>) and F_3 (|1>) under M_0
    readout_fid = float(np.clip(0.5 * (gram[0, 0] + (1.0 - gram[0, 3])), 0.0, 1.0))

    mean_f_avg = float(np.mean([g["avg_gate_fidelity"] for g in gate_reports]))
    max_diam = float(np.max([g["diamond_distance"] for g in gate_reports]))
    mean_unitarity = float(np.mean([g["unitarity"] for g in gate_reports]))

    # Build GST Fiducial-Germ-Fiducial circuit catalog across all 6 gates + germ powers!
    gst_tok_map: Dict[str, Tuple[str, List[float]]] = {
        "I": ("I", []),
        "X90": ("SX", []),
        "Y90": ("RY", [round(0.5 * math.pi, 4)]),
        "X": ("X", []),
        "Y": ("Y", []),
        "H": ("H", []),
    }

    def _build_gst_ops(tag: str, f_prep: Sequence[str], germ_tok: str, power: int, f_meas: Sequence[str]) -> List[Dict[str, Any]]:
        ops_out: List[Dict[str, Any]] = []
        st = 0
        full_seq = list(f_prep) + ([germ_tok] * power) + list(f_meas)
        for tok in full_seq:
            gnm, gpr = gst_tok_map.get(tok, (tok, []))
            ops_out.append(
                CircuitOpModel(
                    id=f"{tag}_{st}",
                    step=st,
                    kind="gate",
                    name=gnm,
                    qubits=[q0],
                    params=gpr,
                ).model_dump()
            )
            st += 1
        ops_out.append(
            CircuitOpModel(id=f"{tag}_m", step=st, kind="measure", name="MEASURE", qubits=[q0], params=[]).model_dump()
        )
        return ops_out

    gst_circuit_samples: List[Dict[str, Any]] = []
    for g_id, p_tok, g_lbl in gate_keys:
        gst_circuit_samples.append(
            {
                "id": f"gst_{g_id}_l1",
                "gate_id": g_id,
                "label": f"{g_id} · LGST 基准三明治 (L=1)",
                "subtitle": f"F₂(Y90) → {p_tok} → M₁(X90) → 测量 ({g_lbl})",
                "operations": _build_gst_ops(f"gst_{g_id}_1", ("Y90",), p_tok, 1, ("X90",)),
            }
        )
        if l_max > 1 and p_tok != "I":
            cap_l = min(l_max, 8)
            gst_circuit_samples.append(
                {
                    "id": f"gst_{g_id}_lmax",
                    "gate_id": g_id,
                    "label": f"{g_id} · 长胚相干放大 (L={cap_l})",
                    "subtitle": f"F₁(X90) → ({p_tok})^{cap_l} → M₂(Y90) → 测量 (放大过旋转误差)",
                    "operations": _build_gst_ops(f"gst_{g_id}_L", ("X90",), p_tok, cap_l, ("Y90",)),
                }
            )
    gst_circuit_samples.append(
        {
            "id": "gst_gram_f3_m0",
            "gate_id": "SPAM",
            "label": "Gram 矩阵 SPAM 标定电路",
            "subtitle": "F₃(X90·X90) → M₀(∅) → 测量 (剥离初态与读出混淆)",
            "operations": _build_gst_ops("gst_gram", ("X90", "X90"), "I", 0, ()),
        }
    )
    sample_gst_ops = gst_circuit_samples[1]["operations"]

    return {
        "protocol": "gst",
        "hardware_name": noise0["hardware_name"],
        "target_label": project.qubits[q0].name,
        "pauli_labels": _PAULI_LABELS_1Q,
        "shots": shots,
        "max_germ_power": l_max,
        "germ_powers": germ_powers,
        "germ_curves": germ_curves,
        "num_experiments": num_circuits,
        "mean_gate_fidelity": round(mean_f_avg, 6),
        "max_diamond_distance": round(max_diam, 6),
        "mean_unitarity": round(mean_unitarity, 6),
        "spam_prep_bloch": [round(v, 4) for v in bloch_0],
        "spam_prep_fidelity": round(state_prep_fid, 5),
        "spam_readout_fidelity": round(readout_fid, 5),
        "gram_matrix": np.round(gram, 4).tolist(),
        "gates": gate_reports,
        "sample_operations": sample_gst_ops,
        "circuit_samples": gst_circuit_samples,
    }


# ============================================================================
# 4. 2-Qubit Linear Cross-Entropy Benchmarking (Linear XEB — Non-Clifford Cycles)
# ============================================================================

def _run_xeb_protocol(req: GateBenchmarkRequest) -> Dict[str, Any]:
    """
    Google Sycamore-style 2-qubit Linear Cross-Entropy Benchmarking (XEB):
      Each cycle applies independent random pi/2 rotations from {SX, SY, SW, SXdg, SYdg, SWdg}
      (where W = (X + Y)/sqrt(2) is non-Clifford!) on (q_a, q_b), followed by the target
      2-qubit entangling gate (e.g. SQISWAP / √iSWAP, ISWAP, CZ, CNOT, or any 2Q custom gate).
    """
    from .custom_gates import evaluate_custom_gate_unitary

    project = req.project
    n_proj = len(project.qubits)
    q0 = max(0, min(req.target_qubit, n_proj - 1))
    q1 = max(0, min(req.second_qubit, n_proj - 1))
    if q1 == q0 and n_proj >= 2:
        q1 = (q0 + 1) % n_proj

    noise0 = _extract_qubit_noise_params(project, q0)
    noise1 = _extract_qubit_noise_params(project, q1)
    rng = np.random.default_rng(req.seed)
    shots = max(0, int(req.shots))

    raw_lens = sorted({max(1, min(40, int(m))) for m in req.rb_lengths if int(m) <= 40})
    if len(raw_lens) < 3:
        raw_lens = [1, 2, 4, 8, 12, 16, 24]
    num_seeds = max(6, min(30, int(req.rb_num_seeds)))

    # Build 6 single-qubit pi/2 gates {+X90, +Y90, +W90, -X90, -Y90, -W90} on q0 & q1
    s2_inv = 1.0 / math.sqrt(2.0)
    axes_1q = [
        ("SX", (1.0, 0.0, 0.5 * math.pi)),
        ("SY", (0.0, 1.0, 0.5 * math.pi)),
        ("SW", (s2_inv, s2_inv, 0.5 * math.pi)),
        ("SXdg", (-1.0, 0.0, 0.5 * math.pi)),
        ("SYdg", (0.0, -1.0, 0.5 * math.pi)),
        ("SWdg", (-s2_inv, -s2_inv, 0.5 * math.pi)),
    ]

    def lift_1q(s4: np.ndarray, which: int) -> np.ndarray:
        s0_m = s4 if which == 0 else np.eye(4, dtype=np.complex128)
        s1_m = s4 if which == 1 else np.eye(4, dtype=np.complex128)
        t8 = np.kron(s0_m, s1_m).reshape(2, 2, 2, 2, 2, 2, 2, 2)
        return np.transpose(t8, (0, 2, 1, 3, 4, 6, 5, 7)).reshape(16, 16)

    g1_q0: List[Tuple[str, np.ndarray, np.ndarray]] = []
    g1_q1: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for nm, ax in axes_1q:
        s_0, u_0 = _build_1q_noisy_pulse_superop(
            ax, noise0["t_1q_ns"], noise0["t1_ns"], noise0["t2_ns"], noise0["depol_1q"],
            req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg,
        )
        s_1, u_1 = _build_1q_noisy_pulse_superop(
            ax, noise1["t_1q_ns"], noise1["t1_ns"], noise1["t2_ns"], noise1["depol_1q"],
            req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg,
        )
        g1_q0.append((nm, lift_1q(s_0, 0), np.kron(u_0, _I2)))
        g1_q1.append((nm, lift_1q(s_1, 1), np.kron(_I2, u_1)))

    # Resolve target 2Q entangling gate (supports SQISWAP, ISWAP, CZ, CNOT, or 2Q custom gates!)
    ent_key = (req.xeb_entangler or "SQISWAP").strip().upper()
    custom_match = next(
        (cg for cg in project.custom_gates if cg.name.upper() == ent_key and cg.num_qubits == 2),
        None,
    )
    if custom_match is not None:
        cg_map = {g.name.upper(): g for g in project.custom_gates}
        u_ent, _ = evaluate_custom_gate_unitary(
            custom_match, list(custom_match.default_params), cg_map
        )
        t_ent_ns = float(custom_match.duration_ns)
        dep2_eff = float(custom_match.infidelity) if project.sim_settings.enable_depol else 0.0
        ent_label = custom_match.display_label or custom_match.name
        dsl_ent_name = custom_match.name
    elif ent_key in ("SQISWAP", "SQRT_ISWAP"):
        u_ent = expm(-0.25j * math.pi * (0.5 * (np.kron(_SX, _SX) + np.kron(_SY, _SY))))
        t_ent_ns = noise0["t_2q_ns"] * 0.75
        dep2_eff = noise0["depol_2q"]
        ent_label = "√iSWAP"
        dsl_ent_name = "SQISWAP" if any(g.name.upper() == "SQISWAP" for g in project.custom_gates) else "ISWAP"
    elif ent_key in ("CZ", "CNOT", "ISWAP"):
        u_ent = gate_unitary(ent_key)
        t_ent_ns = noise0["t_2q_ns"]
        dep2_eff = noise0["depol_2q"]
        ent_label = "iSWAP" if ent_key == "ISWAP" else ent_key
        dsl_ent_name = ent_key
    else:
        u_ent = gate_unitary("CZ")
        t_ent_ns = noise0["t_2q_ns"]
        dep2_eff = noise0["depol_2q"]
        ent_label = "CZ"
        dsl_ent_name = "CZ"

    zz_rad = 2.0 * math.pi * (noise0["zz_khz"] * 1e-6) * t_ent_ns
    amp_rad = (req.amp_error_pct / 100.0) * (0.25 * math.pi)
    u_ent_phys = expm(-0.5j * (zz_rad + amp_rad) * np.kron(_SZ, _SZ)) @ u_ent
    s_ent = _unitary_to_liouville(u_ent_phys)
    if t_ent_ns > 0:
        if math.isfinite(noise0["t1_ns"]) or math.isfinite(noise0["t2_ns"]):
            tr0 = NoiseChannel.thermal_relaxation(0, noise0["t1_ns"], noise0["t2_ns"], t_ent_ns)
            s_ent = lift_1q(_kraus_to_liouville(tr0.kraus_operators()), 0) @ s_ent
        if math.isfinite(noise1["t1_ns"]) or math.isfinite(noise1["t2_ns"]):
            tr1 = NoiseChannel.thermal_relaxation(1, noise1["t1_ns"], noise1["t2_ns"], t_ent_ns)
            s_ent = lift_1q(_kraus_to_liouville(tr1.kraus_operators()), 1) @ s_ent
    if dep2_eff > 1e-12:
        dep2_ch = NoiseChannel.depolarizing_2q(0, 1, min(0.90, dep2_eff))
        s_ent = _kraus_to_liouville(dep2_ch.kraus_operators()) @ s_ent

    # Readout confusion matrix M_ro (4 x 4) on 2-qubit computational basis probabilities
    m_ro0 = np.array(
        [[1.0 - noise0["p1_given_0"], noise0["p0_given_1"]], [noise0["p1_given_0"], 1.0 - noise0["p0_given_1"]]],
        dtype=np.float64,
    )
    m_ro1 = np.array(
        [[1.0 - noise1["p1_given_0"], noise1["p0_given_1"]], [noise1["p1_given_0"], 1.0 - noise1["p0_given_1"]]],
        dtype=np.float64,
    )
    m_ro_2q = np.kron(m_ro0, m_ro1)

    psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    rho0_vec = np.outer(psi0, psi0.conj()).reshape(-1)
    diag_indices = [0, 5, 10, 15]

    xeb_means: List[float] = []
    xeb_stds: List[float] = []
    speckle_ideal: List[float] = []
    speckle_meas: List[float] = []
    sample_ops: List[CircuitOpModel] = []
    xeb_circuit_samples: List[Dict[str, Any]] = []
    xeb_sample_m_set = {0, min(1, len(raw_lens) - 1), min(2, len(raw_lens) - 1)}

    for m_idx, m_cyc in enumerate(raw_lens):
        num_sum = 0.0
        den_sum = 0.0
        per_seed_f: List[float] = []
        for k in range(num_seeds):
            psi_id = psi0.copy()
            rho_v = rho0_vec.copy()
            rec = (m_idx in xeb_sample_m_set) and k == 0
            cur_ops: List[CircuitOpModel] = []
            step_c = 0
            for _ in range(m_cyc):
                i_a = int(rng.integers(0, len(g1_q0)))
                i_b = int(rng.integers(0, len(g1_q1)))
                nm_a, s_a, u_a = g1_q0[i_a]
                nm_b, s_b, u_b = g1_q1[i_b]
                psi_id = u_ent @ (u_b @ (u_a @ psi_id))
                rho_v = s_ent @ (s_b @ (s_a @ rho_v))
                if rec:
                    for q_t, g_tok in ((q0, nm_a), (q1, nm_b)):
                        if g_tok == "SX":
                            cur_ops.append(CircuitOpModel(id=f"xeb_m{m_cyc}_{step_c}_{q_t}", step=step_c, kind="gate", name="SX", qubits=[q_t], params=[]))
                        elif g_tok == "SY":
                            cur_ops.append(CircuitOpModel(id=f"xeb_m{m_cyc}_{step_c}_{q_t}", step=step_c, kind="gate", name="RY", qubits=[q_t], params=[round(0.5 * math.pi, 4)]))
                        else:
                            phi_w = 0.25 * math.pi if "W" in g_tok else 0.0
                            th_w = -0.5 * math.pi if "dg" in g_tok else 0.5 * math.pi
                            cur_ops.append(CircuitOpModel(id=f"xeb_m{m_cyc}_{step_c}_{q_t}", step=step_c, kind="gate", name="RXY", qubits=[q_t], params=[round(th_w, 4), round(phi_w, 4)]))
                    step_c += 1
                    cur_ops.append(CircuitOpModel(id=f"xeb_m{m_cyc}_e_{step_c}", step=step_c, kind="gate", name=dsl_ent_name, qubits=[q0, q1], params=[]))
                    step_c += 1
            if rec:
                cur_ops.append(CircuitOpModel(id=f"xeb_m{m_cyc}_meas", step=step_c, kind="measure", name="MEASURE", qubits=[q0, q1], params=[]))
                xeb_circuit_samples.append(
                    {
                        "id": f"xeb_m{m_cyc}",
                        "label": f"Sycamore 周期 (m={m_cyc})",
                        "subtitle": f"{m_cyc} 周期随机单比特 {{√X, √Y, √W}} 并行层 + {ent_label} 纠缠门 → 比特串采样",
                        "operations": [op.model_dump() for op in cur_ops],
                    }
                )
                if m_idx == min(1, len(raw_lens) - 1) or not sample_ops:
                    sample_ops = cur_ops

            p_id = np.abs(psi_id) ** 2
            p_ns_raw = np.clip(np.real(rho_v[diag_indices]), 0.0, 1.0)
            p_ns = m_ro_2q @ p_ns_raw
            p_ns = np.clip(p_ns / max(1e-12, float(np.sum(p_ns))), 0.0, 1.0)
            if shots > 0:
                counts = rng.multinomial(shots, p_ns)
                p_obs = counts.astype(np.float64) / float(shots)
            else:
                p_obs = p_ns

            # Normalized Linear XEB estimator: (d * sum(p_id * p_obs) - 1) / (d * sum(p_id^2) - 1)
            raw_num = 4.0 * float(np.dot(p_id, p_obs)) - 1.0
            raw_den = max(0.05, 4.0 * float(np.dot(p_id, p_id)) - 1.0)
            num_sum += raw_num
            den_sum += raw_den
            per_seed_f.append(float(raw_num / raw_den))

            if m_idx == len(raw_lens) // 2 and len(speckle_ideal) < 80:
                for b_i in range(4):
                    speckle_ideal.append(round(float(p_id[b_i]), 5))
                    speckle_meas.append(round(float(p_obs[b_i]), 5))

        f_m = float(np.clip(num_sum / max(1e-9, den_sum), 0.0, 1.05))
        xeb_means.append(round(f_m, 6))
        xeb_stds.append(round(float(np.std(per_seed_f, ddof=1) / math.sqrt(len(per_seed_f))), 6))

    # Fit F_XEB(m) = A * f_cycle^m
    m_arr = np.array(raw_lens, dtype=np.float64)
    y_arr = np.array(xeb_means, dtype=np.float64)
    if float(np.max(y_arr) - np.min(y_arr)) < 1e-5 and float(np.mean(y_arr)) > 0.96:
        a_xeb, f_cyc = float(min(1.0, np.mean(y_arr))), 1.0
    else:
        def xeb_fn(m: np.ndarray, a: float, f: float) -> np.ndarray:
            return a * np.power(f, m)

        try:
            popt, _ = curve_fit(xeb_fn, m_arr, np.clip(y_arr, 1e-4, 1.05), p0=[0.98, 0.985], bounds=([0.2, 0.1], [1.05, 1.0]), maxfev=3000)
            a_xeb, f_cyc = float(popt[0]), float(popt[1])
        except Exception:
            a_xeb, f_cyc = float(y_arr[0]), 0.985

    # Cycle Pauli error e_cycle = (1 - 1/4) * (1 - f_cycle) = 0.75 * (1 - f_cycle)
    e_cycle = 0.75 * (1.0 - f_cyc)
    # Compute exact 2-qubit Pauli-twirled polarization of the parallel 1Q gate layer
    s0_single, u0_single = _build_1q_noisy_pulse_superop(
        axes_1q[0][1], noise0["t_1q_ns"], noise0["t1_ns"], noise0["t2_ns"], noise0["depol_1q"],
        req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg,
    )
    s1_single, u1_single = _build_1q_noisy_pulse_superop(
        axes_1q[0][1], noise1["t_1q_ns"], noise1["t1_ns"], noise1["t2_ns"], noise1["depol_1q"],
        req.amp_error_pct, req.detuning_mhz, req.phase_skew_deg,
    )
    f_proc_0 = float(np.clip(0.25 * np.trace(_liouville_to_ptm_1q(_unitary_to_liouville(u0_single)).T @ _liouville_to_ptm_1q(s0_single)), 0.0, 1.0))
    f_proc_1 = float(np.clip(0.25 * np.trace(_liouville_to_ptm_1q(_unitary_to_liouville(u1_single)).T @ _liouville_to_ptm_1q(s1_single)), 0.0, 1.0))
    p0_1q = max(0.01, (4.0 * f_proc_0 - 1.0) / 3.0)
    p1_1q = max(0.01, (4.0 * f_proc_1 - 1.0) / 3.0)
    p_layer_2q = max(0.01, (3.0 * p0_1q + 3.0 * p1_1q + 9.0 * p0_1q * p1_1q) / 15.0)
    f_ent_pol = min(1.0, f_cyc / p_layer_2q)
    entangler_fidelity = 1.0 - 0.75 * (1.0 - f_ent_pol)

    dense_m = np.linspace(0.0, float(max(raw_lens)), 80)
    fit_y = [round(float(a_xeb * (f_cyc ** xm)), 6) for xm in dense_m]

    return {
        "protocol": "xeb",
        "hardware_name": noise0["hardware_name"],
        "target_label": f"{project.qubits[q0].name}, {project.qubits[q1].name}",
        "entangler_name": ent_label,
        "lengths": raw_lens,
        "num_seeds": num_seeds,
        "shots": shots,
        "fit_x": [round(float(x), 3) for x in dense_m],
        "fit_y": fit_y,
        "mean_xeb": xeb_means,
        "std_xeb": xeb_stds,
        "a_spam": round(a_xeb, 5),
        "cycle_polarization": round(f_cyc, 6),
        "cycle_pauli_error": round(e_cycle, 6),
        "entangler_fidelity": round(entangler_fidelity, 6),
        "speckle_ideal": speckle_ideal,
        "speckle_measured": speckle_meas,
        "sample_operations": [op.model_dump() for op in sample_ops],
        "circuit_samples": xeb_circuit_samples,
    }


def execute_gate_benchmark(req: GateBenchmarkRequest) -> Dict[str, Any]:
    """Dispatch RB / Interleaved RB, AllXY, Gate Set Tomography (GST), or Linear XEB."""
    if req.protocol == "rb":
        return _run_rb_protocol(req)
    if req.protocol == "allxy":
        return _run_allxy_protocol(req)
    if req.protocol == "gst":
        return _run_gst_protocol(req)
    if req.protocol == "xeb":
        return _run_xeb_protocol(req)
    if req.protocol in ("rabi", "ramsey", "t1"):
        from .calibration import run_basic_calibration

        return run_basic_calibration(req, _extract_qubit_noise_params(req.project, req.target_qubit))
    raise ValueError(f"未知门标定协议: {req.protocol}")
