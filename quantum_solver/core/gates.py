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
    if uname == "I":
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
