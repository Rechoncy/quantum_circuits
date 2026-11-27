"""
Standard and experimental native quantum gates, unitary matrices,
Hamiltonian generators, and hardware-native basis gate sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import expm


# ---------------------------------------------------------------------------
# Fundamental Pauli & Basis Matrices
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

PAULIS_1Q: Dict[str, np.ndarray] = {
    "I": I2,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}


# ---------------------------------------------------------------------------
# Standard Gate Unitary Constructors
# ---------------------------------------------------------------------------

def rx_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = -1.0j * math.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    e_m = np.exp(-0.5j * theta)
    e_p = np.exp(0.5j * theta)
    return np.array([[e_m, 0.0], [0.0, e_p]], dtype=np.complex128)


def phase_matrix(lam: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * lam)]], dtype=np.complex128)


def rxy_matrix(theta: float, phi: float) -> np.ndarray:
    """Experimental microwave rotation around cos(phi)*X + sin(phi)*Y axis."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    ep = np.exp(1.0j * phi)
    em = np.exp(-1.0j * phi)
    return np.array(
        [[c, -1.0j * s * em], [-1.0j * s * ep, c]],
        dtype=np.complex128,
    )


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Standard 3-parameter universal single-qubit gate U3(theta, phi, lam)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1.0j * lam) * s],
            [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def rxx_matrix(theta: float) -> np.ndarray:
    """Two-qubit XX rotation exp(-i * theta/2 * X⊗X) (e.g. trapped-ion Mølmer-Sørensen)."""
    c = math.cos(theta / 2.0)
    s = -1.0j * math.sin(theta / 2.0)
    return np.array(
        [
            [c, 0.0, 0.0, s],
            [0.0, c, s, 0.0],
            [0.0, s, c, 0.0],
            [s, 0.0, 0.0, c],
        ],
        dtype=np.complex128,
    )


def ryy_matrix(theta: float) -> np.ndarray:
    """Two-qubit YY rotation exp(-i * theta/2 * Y⊗Y)."""
    c = math.cos(theta / 2.0)
    s = 1.0j * math.sin(theta / 2.0)
    return np.array(
        [
            [c, 0.0, 0.0, s],
            [0.0, c, -s, 0.0],
            [0.0, -s, c, 0.0],
            [s, 0.0, 0.0, c],
        ],
        dtype=np.complex128,
    )


def rzz_matrix(theta: float) -> np.ndarray:
    """Two-qubit ZZ rotation exp(-i * theta/2 * Z⊗Z)."""
    em = np.exp(-0.5j * theta)
    ep = np.exp(0.5j * theta)
    return np.diag([em, ep, ep, em]).astype(np.complex128)


def cr_matrix(theta: float) -> np.ndarray:
    """Cross-Resonance gate exp(-i * theta/2 * Z⊗X)."""
    G = np.kron(PAULI_Z, PAULI_X)
    return expm(-0.5j * theta * G)


def fsim_matrix(theta: float, phi: float) -> np.ndarray:
    """Google/superconducting fermionic simulation gate fSim(theta, phi)."""
    c = math.cos(theta)
    s = -1.0j * math.sin(theta)
    ep = np.exp(-1.0j * phi)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, ep],
        ],
        dtype=np.complex128,
    )


def can_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Canonical SU(4) non-local gate exp(-i/2 * (tx X⊗X + ty Y⊗Y + tz Z⊗Z))."""
    xx = np.kron(PAULI_X, PAULI_X)
    yy = np.kron(PAULI_Y, PAULI_Y)
    zz = np.kron(PAULI_Z, PAULI_Z)
    return expm(-0.5j * (tx * xx + ty * yy + tz * zz))


# Fixed Unitary Matrices
FIXED_MATRICES: Dict[str, np.ndarray] = {
    "I": I2,
    "IDLE": I2,
    "DELAY": I2,
    "BARRIER": I2,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
    "H": np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0),
    "S": np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128),
    "SDG": np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128),
    "T": np.array([[1.0, 0.0], [0.0, np.exp(0.25j * math.pi)]], dtype=np.complex128),
    "TDG": np.array([[1.0, 0.0], [0.0, np.exp(-0.25j * math.pi)]], dtype=np.complex128),
    "SX": 0.5 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128),
    "SXDG": 0.5 * np.array([[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]], dtype=np.complex128),
    "SY": 0.5 * np.array([[1.0 + 1.0j, -1.0 - 1.0j], [1.0 + 1.0j, 1.0 + 1.0j]], dtype=np.complex128),
    "SYDG": 0.5 * np.array([[1.0 - 1.0j, 1.0 - 1.0j], [-1.0 + 1.0j, 1.0 - 1.0j]], dtype=np.complex128),
    "CNOT": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    ),
    "CX": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    ),
    "CY": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=np.complex128,
    ),
    "CZ": np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128),
    "SWAP": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    ),
    "ISWAP": np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    ),
    "SQISWAP": np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / math.sqrt(2), 1j / math.sqrt(2), 0],
            [0, 1j / math.sqrt(2), 1 / math.sqrt(2), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    ),
    "ECR": (1.0 / math.sqrt(2.0))
    * np.array(
        [
            [0, 1, 0, 1j],
            [1, 0, -1j, 0],
            [0, 1j, 0, 1],
            [-1j, 0, 1, 0],
        ],
        dtype=np.complex128,
    ),
    "MS": rxx_matrix(math.pi / 2.0),
    "CCX": np.block(
        [
            [np.eye(6, dtype=np.complex128), np.zeros((6, 2), dtype=np.complex128)],
            [np.zeros((2, 6), dtype=np.complex128), PAULI_X],
        ]
    ),
    "TOFFOLI": np.block(
        [
            [np.eye(6, dtype=np.complex128), np.zeros((6, 2), dtype=np.complex128)],
            [np.zeros((2, 6), dtype=np.complex128), PAULI_X],
        ]
    ),
    "CCZ": np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]).astype(np.complex128),
}

CLIFFORD_GATE_NAMES = {
    "I",
    "IDLE",
    "DELAY",
    "BARRIER",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "SX",
    "SXDG",
    "SY",
    "SYDG",
    "CNOT",
    "CX",
    "CY",
    "CZ",
    "SWAP",
    "ISWAP",
    "ECR",
}


def is_clifford_angle(theta: float, tol: float = 1e-4) -> bool:
    """Check if rotation angle theta is an integer multiple of pi/2 (with 4-decimal rounding tolerance)."""
    k = round(theta / (math.pi / 2.0))
    return abs(theta - k * (math.pi / 2.0)) < tol


def gate_unitary(
    name: str,
    params: Sequence[float] = (),
    custom_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return the unitary matrix for a named gate and its parameters."""
    if custom_matrix is not None:
        return np.asarray(custom_matrix, dtype=np.complex128)

    uname = name.upper()
    if uname in ("IDLE", "DELAY", "BARRIER"):
        return I2.copy()
    if uname in FIXED_MATRICES and len(params) == 0:
        return FIXED_MATRICES[uname].copy()

    if uname == "RX":
        return rx_matrix(float(params[0]))
    if uname == "RY":
        return ry_matrix(float(params[0]))
    if uname == "RZ":
        return rz_matrix(float(params[0]))
    if uname in ("P", "PHASE"):
        return phase_matrix(float(params[0]))
    if uname == "RXY":
        return rxy_matrix(float(params[0]), float(params[1]))
    if uname in ("U3", "U"):
        return u3_matrix(float(params[0]), float(params[1]), float(params[2]))
    if uname == "RXX":
        return rxx_matrix(float(params[0]))
    if uname == "RYY":
        return ryy_matrix(float(params[0]))
    if uname == "RZZ":
        return rzz_matrix(float(params[0]))
    if uname == "CR":
        return cr_matrix(float(params[0]))
    if uname == "FSIM":
        return fsim_matrix(float(params[0]), float(params[1]))
    if uname == "CAN":
        return can_matrix(float(params[0]), float(params[1]), float(params[2]))
    if uname == "MS":
        angle = float(params[0]) if len(params) > 0 else math.pi / 2.0
        return rxx_matrix(angle)

    raise ValueError(f"Unknown gate '{name}' with params {params}")


def gate_hamiltonian_generator(
    name: str,
    params: Sequence[float] = (),
    custom_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Return the effective dimensionless generator H_gen and rotation angle theta
    such that U = exp(-i * (theta / 2) * H_gen) up to a global phase.
    Used by QuTiP pulse-level solver to construct physical control pulses H(t).
    """
    uname = name.upper()
    if uname in ("I", "IDLE", "DELAY", "BARRIER"):
        return np.zeros((2, 2), dtype=np.complex128), 0.0
    if uname == "X":
        return PAULI_X, math.pi
    if uname == "Y":
        return PAULI_Y, math.pi
    if uname == "Z":
        return PAULI_Z, math.pi
    if uname == "SX":
        return PAULI_X, math.pi / 2.0
    if uname == "SXDG":
        return PAULI_X, -math.pi / 2.0
    if uname == "SY":
        return PAULI_Y, math.pi / 2.0
    if uname == "SYDG":
        return PAULI_Y, -math.pi / 2.0
    if uname == "S":
        return PAULI_Z, math.pi / 2.0
    if uname == "SDG":
        return PAULI_Z, -math.pi / 2.0
    if uname == "T":
        return PAULI_Z, math.pi / 4.0
    if uname == "TDG":
        return PAULI_Z, -math.pi / 4.0
    if uname == "H":
        return (PAULI_X + PAULI_Z) / math.sqrt(2.0), math.pi
    if uname == "RX":
        return PAULI_X, float(params[0])
    if uname == "RY":
        return PAULI_Y, float(params[0])
    if uname == "RZ":
        return PAULI_Z, float(params[0])
    if uname in ("P", "PHASE"):
        return PAULI_Z, float(params[0])
    if uname == "RXY":
        theta, phi = float(params[0]), float(params[1])
        gen = math.cos(phi) * PAULI_X + math.sin(phi) * PAULI_Y
        return gen, theta
    if uname == "RXX":
        return np.kron(PAULI_X, PAULI_X), float(params[0])
    if uname == "RYY":
        return np.kron(PAULI_Y, PAULI_Y), float(params[0])
    if uname == "RZZ":
        return np.kron(PAULI_Z, PAULI_Z), float(params[0])
    if uname == "MS":
        angle = float(params[0]) if len(params) > 0 else math.pi / 2.0
        return np.kron(PAULI_X, PAULI_X), angle
    if uname == "CR":
        return np.kron(PAULI_Z, PAULI_X), float(params[0])
    if uname == "CZ":
        # Proj |11><11| = (I - Z1 - Z2 + Z1 Z2)/4; exp(-i * pi * |11><11|) = CZ
        proj_11 = np.diag([0.0, 0.0, 0.0, 2.0]).astype(np.complex128)
        return proj_11, math.pi
    if uname == "CCZ":
        proj_111 = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]).astype(np.complex128)
        return proj_111, math.pi
    if uname in ("CNOT", "CX"):
        # |1><1| ⊗ (I - X)
        proj_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        gen = np.kron(proj_1, I2 - PAULI_X)
        return gen, math.pi
    if uname == "ISWAP":
        gen = -0.5 * (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y))
        return gen, math.pi
    if uname == "SQISWAP":
        gen = -0.5 * (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y))
        return gen, math.pi / 2.0

    # Fallback: matrix logarithm U = exp(-i * H_eff) => H_gen = 2 * i * logm(U) / pi
    from scipy.linalg import logm

    U = gate_unitary(name, params, custom_matrix)
    H_eff = 1.0j * logm(U)
    H_eff = 0.5 * (H_eff + H_eff.conj().T)
    return 2.0 * H_eff, 1.0


def sample_unit_area_envelope(
    t_ns: float,
    gate_dt: float,
    pulse_shape: str = "cosine",
    rise_ns: float = 0.0,
    fall_ns: float = 0.0,
) -> float:
    """Evaluate unit-area control envelope f(t) on [0, gate_dt] with integral_0^{gate_dt} f(t) dt = 1."""
    if gate_dt <= 0.0 or t_ns < 0.0 or t_ns > gate_dt:
        return 0.0
    shape = (pulse_shape or "cosine").lower()
    if shape in ("rectangular", "trapezoid") and (rise_ns > 0.0 or fall_ns > 0.0):
        r = max(0.0, rise_ns)
        f = max(0.0, fall_ns)
        if r + f > gate_dt:
            s = gate_dt / (r + f)
            r, f = r * s, f * s
        area = max(gate_dt - 0.5 * (r + f), 1e-12)
        amp_ft = 1.0 / area
        t_fall0 = gate_dt - f
        if shape == "trapezoid":
            if r > 0.0 and t_ns < r:
                return amp_ft * (t_ns / r)
            if f > 0.0 and t_ns > t_fall0:
                return amp_ft * ((gate_dt - t_ns) / f)
            return amp_ft
        if r > 0.0 and t_ns < r:
            return amp_ft * 0.5 * (1.0 - math.cos(math.pi * t_ns / r))
        if f > 0.0 and t_ns > t_fall0:
            return amp_ft * 0.5 * (1.0 - math.cos(math.pi * (gate_dt - t_ns) / f))
        return amp_ft

    if shape in ("rectangular", "trapezoid"):
        return 1.0 / gate_dt

    if shape == "gaussian":
        mid = 0.5 * gate_dt
        sigma = gate_dt / 4.5
        edge = math.exp(-0.5 * (mid / sigma) ** 2)
        area = math.sqrt(2.0 * math.pi) * sigma * math.erf(mid / (math.sqrt(2.0) * sigma)) - edge * gate_dt
        return max(0.0, math.exp(-0.5 * ((t_ns - mid) / sigma) ** 2) - edge) / max(area, 1e-12)

    return (1.0 - math.cos(2.0 * math.pi * t_ns / gate_dt)) / gate_dt


def simulate_3level_drag_pulse(
    theta: float,
    phi: float,
    duration_ns: float,
    anharmonicity_ghz: float,
    drag_beta: float = 0.0,
    pulse_shape: str = "cosine",
    rise_ns: float = 0.0,
    fall_ns: float = 0.0,
    detuning_mhz: float = 0.0,
    n_steps: int = 44,
) -> Tuple[np.ndarray, float, float]:
    """
    Propagate a single-qubit drive pulse in the 3-level Transmon Hilbert space {|0>, |1>, |2>}
    with anharmonicity alpha/2pi = `anharmonicity_ghz` (GHz) and DRAG quadrature derivative
    Omega_Q(t) = -drag_beta * d(Omega_I)/dt / omega_alpha.

    Returns:
      - `u_2x2_polar`: Closest 2x2 unitary in the computational subspace (capturing AC-Stark phase error)
      - `leak_0`: Leakage probability P(|2> | init |0>) = |U_{2,0}|^2
      - `leak_1`: Leakage probability P(|2> | init |1>) = |U_{2,1}|^2
    """
    if duration_ns <= 1e-9 or abs(theta) <= 1e-9 or abs(anharmonicity_ghz) <= 1e-5:
        return rxy_matrix(theta, phi), 0.0, 0.0

    w_alpha = 2.0 * math.pi * float(anharmonicity_ghz)  # rad / ns
    w_det = 2.0 * math.pi * float(detuning_mhz) * 1e-3  # rad / ns
    dt = duration_ns / float(n_steps)
    eps = max(1e-4, 0.02 * dt)

    ep = complex(math.cos(phi), math.sin(phi))
    em = ep.conjugate()
    # Lowering operator b_phi = e^{-i phi} (|0><1| + sqrt(2)|1><2|)
    s2 = math.sqrt(2.0)
    x_phi = np.array(
        [
            [0.0, em, 0.0],
            [ep, 0.0, s2 * em],
            [0.0, s2 * ep, 0.0],
        ],
        dtype=np.complex128,
    )
    y_phi = np.array(
        [
            [0.0, -1j * em, 0.0],
            [1j * ep, 0.0, -1j * s2 * em],
            [0.0, 1j * s2 * ep, 0.0],
        ],
        dtype=np.complex128,
    )
    h_static = np.diag([0.0, w_det, 2.0 * w_det + w_alpha]).astype(np.complex128)

    u3 = np.eye(3, dtype=np.complex128)
    for k in range(n_steps):
        tm = (k + 0.5) * dt
        f_mid = sample_unit_area_envelope(tm, duration_ns, pulse_shape, rise_ns, fall_ns)
        omega_i = theta * f_mid
        if abs(drag_beta) > 1e-9:
            f_p = sample_unit_area_envelope(min(duration_ns, tm + eps), duration_ns, pulse_shape, rise_ns, fall_ns)
            f_m = sample_unit_area_envelope(max(0.0, tm - eps), duration_ns, pulse_shape, rise_ns, fall_ns)
            d_omega_i = theta * (f_p - f_m) / (min(duration_ns, tm + eps) - max(0.0, tm - eps))
            omega_q = -float(drag_beta) * d_omega_i / w_alpha
        else:
            omega_q = 0.0

        h_step = h_static + 0.5 * omega_i * x_phi + 0.5 * omega_q * y_phi
        vals, vecs = np.linalg.eigh(h_step)
        phases = np.exp(-1j * vals * dt)
        u_step = (vecs * phases) @ vecs.conj().T
        u3 = u_step @ u3

    leak_0 = float(min(1.0, max(0.0, abs(u3[2, 0]) ** 2)))
    leak_1 = float(min(1.0, max(0.0, abs(u3[2, 1]) ** 2)))
    sub2 = u3[:2, :2]
    # Polar decomposition SVD: closest 2x2 unitary preserving exact AC-Stark phase shift!
    u_svd, _, vh_svd = np.linalg.svd(sub2)
    u_polar = u_svd @ vh_svd
    return u_polar, leak_0, leak_1


# ---------------------------------------------------------------------------
# Experimental Native Gate Set Definitions
# ---------------------------------------------------------------------------

@dataclass
class NativeGateSpec:
    """Specification of an experimentally available native quantum operation."""

    name: str
    num_qubits: int
    num_params: int = 0
    duration_ns: float = 20.0
    infidelity: float = 1e-3
    is_virtual: bool = False  # e.g., Virtual-Z has 0 duration and 0 physical error
    matrix_fn: Optional[Callable[..., np.ndarray]] = None

    def unitary(self, *params: float) -> np.ndarray:
        if self.matrix_fn is not None:
            return self.matrix_fn(*params)
        return gate_unitary(self.name, params)


@dataclass
class NativeGateSet:
    """
    Defines the repertoire of experimental operations available on a target device.
    Used by the Quantum Circuit Synthesizer & Optimizer to find optimal gate combinations.
    """

    name: str
    single_qubit_gates: List[NativeGateSpec] = field(default_factory=list)
    two_qubit_gates: List[NativeGateSpec] = field(default_factory=list)
    decomposition_mode_1q: str = "rz_sx_rz"  # 'rz_sx_rz', 'zyz', 'rxy_rz', 'u3', 'discrete'
    primary_2q_gate: str = "CZ"

    @classmethod
    def superconducting_cz(
        cls,
        t_1q_ns: float = 20.0,
        t_2q_ns: float = 40.0,
        err_1q: float = 5e-4,
        err_2q: float = 5e-3,
    ) -> "NativeGateSet":
        """
        Standard superconducting processor native gate set (IBM/Google/tunable coupler style):
        - Virtual-Z: arbitrary RZ(theta) with 0 duration and 0 physical error
        - Physical 1Q: SX (sqrt(X)) and X pulses
        - Physical 2Q: CZ gate
        """
        return cls(
            name="Superconducting (Virtual-RZ + SX + CZ)",
            single_qubit_gates=[
                NativeGateSpec("RZ", 1, 1, duration_ns=0.0, infidelity=0.0, is_virtual=True),
                NativeGateSpec("SX", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("X", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
            ],
            two_qubit_gates=[
                NativeGateSpec("CZ", 2, 0, duration_ns=t_2q_ns, infidelity=err_2q),
            ],
            decomposition_mode_1q="rz_sx_rz",
            primary_2q_gate="CZ",
        )

    @classmethod
    def superconducting_cnot(
        cls,
        t_1q_ns: float = 20.0,
        t_2q_ns: float = 60.0,
        err_1q: float = 5e-4,
        err_2q: float = 5e-3,
    ) -> "NativeGateSet":
        """Superconducting / standard basis with {RZ (virtual), SX, X, CNOT}."""
        return cls(
            name="Superconducting (Virtual-RZ + SX + CNOT)",
            single_qubit_gates=[
                NativeGateSpec("RZ", 1, 1, duration_ns=0.0, infidelity=0.0, is_virtual=True),
                NativeGateSpec("SX", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("X", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
            ],
            two_qubit_gates=[
                NativeGateSpec("CNOT", 2, 0, duration_ns=t_2q_ns, infidelity=err_2q),
            ],
            decomposition_mode_1q="rz_sx_rz",
            primary_2q_gate="CNOT",
        )

    @classmethod
    def superconducting_iswap(
        cls,
        t_1q_ns: float = 20.0,
        t_2q_ns: float = 32.0,
        err_1q: float = 5e-4,
        err_2q: float = 4e-3,
    ) -> "NativeGateSet":
        """Superconducting processor with capacitive XY coupling: {RZ (virtual), SX, iSWAP}."""
        return cls(
            name="Superconducting (Virtual-RZ + SX + iSWAP)",
            single_qubit_gates=[
                NativeGateSpec("RZ", 1, 1, duration_ns=0.0, infidelity=0.0, is_virtual=True),
                NativeGateSpec("SX", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("X", 1, 0, duration_ns=t_1q_ns, infidelity=err_1q),
            ],
            two_qubit_gates=[
                NativeGateSpec("ISWAP", 2, 0, duration_ns=t_2q_ns, infidelity=err_2q),
            ],
            decomposition_mode_1q="rz_sx_rz",
            primary_2q_gate="ISWAP",
        )

    @classmethod
    def trapped_ion(
        cls,
        t_1q_ns: float = 5000.0,
        t_2q_ns: float = 100000.0,
        err_1q: float = 1e-4,
        err_2q: float = 2e-3,
    ) -> "NativeGateSet":
        """
        Trapped-ion native gate set:
        - Arbitrary equatorial laser rotation RXY(theta, phi) + virtual RZ(theta)
        - Mølmer-Sørensen entangling gate MS = RXX(pi/2)
        """
        return cls(
            name="Trapped Ion (RXY + RZ + MS)",
            single_qubit_gates=[
                NativeGateSpec("RZ", 1, 1, duration_ns=0.0, infidelity=0.0, is_virtual=True),
                NativeGateSpec("RX", 1, 1, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("RY", 1, 1, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("RXY", 1, 2, duration_ns=t_1q_ns, infidelity=err_1q),
            ],
            two_qubit_gates=[
                NativeGateSpec("MS", 2, 0, duration_ns=t_2q_ns, infidelity=err_2q),
            ],
            decomposition_mode_1q="rxy_rz",
            primary_2q_gate="MS",
        )

    @classmethod
    def neutral_atom_rydberg(
        cls,
        t_1q_ns: float = 400.0,
        t_2q_ns: float = 240.0,
        err_1q: float = 8e-4,
        err_2q: float = 4.5e-3,
    ) -> "NativeGateSet":
        """
        Neutral-Atom Rydberg Array native gate set (Rb-87 / Cs-133 / Yb-171 / Sr-87 optical tweezers):
        - Raman / microwave arbitrary equatorial rotation RXY(theta, phi)
        - Virtual / AC-Stark phase shift RZ(theta)
        - Rydberg blockade Levine-Pichler CZ entangling gate
        """
        return cls(
            name="Neutral Atom Rydberg (RXY + Virtual-RZ + CZ)",
            single_qubit_gates=[
                NativeGateSpec("RZ", 1, 1, duration_ns=0.0, infidelity=0.0, is_virtual=True),
                NativeGateSpec("RXY", 1, 2, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("RX", 1, 1, duration_ns=t_1q_ns, infidelity=err_1q),
                NativeGateSpec("RY", 1, 1, duration_ns=t_1q_ns, infidelity=err_1q),
            ],
            two_qubit_gates=[
                NativeGateSpec("CZ", 2, 0, duration_ns=t_2q_ns, infidelity=err_2q),
            ],
            decomposition_mode_1q="rxy_rz",
            primary_2q_gate="CZ",
        )

    def get_gate_spec(self, name: str) -> Optional[NativeGateSpec]:
        uname = name.upper()
        for g in self.single_qubit_gates + self.two_qubit_gates:
            if g.name.upper() == uname:
                return g
        return None

    def allowed_names(self) -> List[str]:
        return [g.name.upper() for g in self.single_qubit_gates + self.two_qubit_gates]
