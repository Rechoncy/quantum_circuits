"""
Exact Single-Qubit Unitary Decomposition & Hardware-Native Synthesis.
Supports:
  - ZYZ Euler decomposition: U = e^{i*phase} RZ(phi) RY(theta) RZ(lam)
  - Superconducting Virtual-Z + SX decomposition: RZ(a) SX RZ(b) SX RZ(c)
    with automatic pulse reduction to 0, 1, or 2 physical SX/X pulses!
  - Trapped-Ion / Microwave Agile RXY + Virtual-Z decomposition:
    at most 1 physical RXY(theta, phi_axis) pulse + 1 Virtual-Z gate!
  - U3(theta, phi, lam) decomposition.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from ..core.circuit import GateInstruction
from ..core.gates import NativeGateSet, gate_unitary, rz_matrix, ry_matrix


def normalize_angle(angle: float) -> float:
    """Wrap angle into (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if abs(wrapped + math.pi) < 1e-12:
        return math.pi
    return wrapped


def single_qubit_unitary_fidelity(u1: np.ndarray, u2: np.ndarray) -> float:
    """Global-phase-invariant Hilbert-Schmidt fidelity |Tr(U1^† U2)|^2 / 4 for 2x2 unitaries."""
    tr = np.trace(u1.conj().T @ u2)
    return float(min(1.0, max(0.0, (abs(tr) ** 2) / 4.0)))


def zyz_euler_angles(U: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Exact ZYZ Euler decomposition of any 2x2 unitary matrix U in U(2):
        U = exp(i * global_phase) * RZ(phi) @ RY(theta) @ RZ(lam)
    Returns (theta, phi, lam, global_phase) with theta in [0, pi].
    """
    U = np.asarray(U, dtype=np.complex128)
    det = np.linalg.det(U)
    global_phase = 0.5 * np.angle(det)
    V = U * np.exp(-1.0j * global_phase)  # V in SU(2)

    # In SU(2):
    # V = [[cos(theta/2) * exp(-i*(phi+lam)/2), -sin(theta/2) * exp(-i*(phi-lam)/2)],
    #      [sin(theta/2) * exp(+i*(phi-lam)/2),  cos(theta/2) * exp(+i*(phi+lam)/2)]]
    c = float(np.clip(abs(V[0, 0]), 0.0, 1.0))
    s = float(np.clip(abs(V[1, 0]), 0.0, 1.0))
    theta = 2.0 * math.atan2(s, c)

    if s < 1e-9:
        # theta ~ 0: V is purely diagonal RZ(phi + lam)
        phi_plus_lam = 2.0 * np.angle(V[1, 1])
        return 0.0, normalize_angle(phi_plus_lam), 0.0, float(global_phase)

    if c < 1e-9:
        # theta ~ pi: V is purely off-diagonal
        phi_minus_lam = 2.0 * np.angle(V[1, 0])
        return math.pi, normalize_angle(phi_minus_lam), 0.0, float(global_phase)

    sum_half = np.angle(V[1, 1])
    diff_half = np.angle(V[1, 0])
    phi = normalize_angle(sum_half + diff_half)
    lam = normalize_angle(sum_half - diff_half)
    return float(theta), float(phi), float(lam), float(global_phase)


def synthesize_1q_rz_sx_rz(
    U: np.ndarray,
    qubit: int,
    allow_x_gate: bool = True,
    tol: float = 1e-7,
) -> List[GateInstruction]:
    """
    Synthesize any 1-qubit unitary U into the superconducting hardware basis
    {RZ (virtual, 0 ns), SX (sqrt(X)), X} using the minimum number of physical pulses (0, 1, or 2):

      - Case 0 (theta == 0): 0 physical pulses, 1 Virtual-Z: [RZ(phi + lam)]
      - Case 1a (theta == pi and allow_x_gate): 1 physical X pulse + 1 Virtual-Z:
            RY(pi) = X @ RZ(pi)  =>  U = RZ(phi) @ X @ RZ(lam + pi) = X @ RZ(lam + pi - phi)
      - Case 1b (theta == pi/2): 1 physical SX pulse + 2 Virtual-Z:
            RY(pi/2) = RZ(-pi/2) @ RX(pi/2) @ RZ(pi/2) = RZ(-pi/2) @ SX @ RZ(pi/2)
      - Case 2 (general theta): 2 physical SX pulses + 3 Virtual-Z:
            RY(theta) = RZ(pi) @ SX @ RZ(pi - theta) @ SX
            (since SX @ RZ(alpha) @ SX = exp(i*pi/2) * RX(pi/2) RZ(alpha) RX(pi/2)
             = RZ(-pi) @ RY(-alpha), so RZ(pi) @ SX @ RZ(pi - theta) @ SX = RY(theta) @ RZ(pi)!)
            Exact identity:
            RY(theta) = SX @ RZ(pi - theta) @ SX @ RZ(pi)  Wait: let's use the exact identity:
            RY(theta) = RZ(-pi/2) @ RX(theta) @ RZ(pi/2)
            and RX(theta) = RZ(pi/2) @ RY(theta) @ RZ(-pi/2),
            while SX @ RZ(alpha) @ SX = -i * [[cos(a/2), i*sin(a/2)], [i*sin(a/2), -cos(a/2)]]
            = RZ(pi/2) @ RY(pi - alpha) @ RZ(pi/2) (up to global phase)!
            Therefore:
            RY(theta) = RZ(-pi/2) @ SX @ RZ(pi - theta) @ SX @ RZ(-pi/2)
            => U = RZ(phi - pi/2) @ SX @ RZ(pi - theta) @ SX @ RZ(lam - pi/2).
    """
    theta, phi, lam, _ = zyz_euler_angles(U)
    ops: List[GateInstruction] = []

    def _add_rz(angle: float) -> None:
        a = normalize_angle(angle)
        if abs(a) > tol:
            ops.append(GateInstruction("RZ", (qubit,), (a,), duration_ns=0.0))

    # Case 0: Pure Z rotation (0 physical pulses)
    if abs(theta) < tol:
        _add_rz(phi + lam)
        return ops

    # Case 1a: theta == pi -> 1 physical X pulse
    if abs(theta - math.pi) < tol and allow_x_gate:
        # RY(pi) = RZ(pi) @ X = X @ RZ(-pi)
        # U = RZ(phi) @ RY(pi) @ RZ(lam) = X @ RZ(lam - phi - pi)
        _add_rz(lam - phi - math.pi)
        ops.append(GateInstruction("X", (qubit,)))
        return ops

    # Case 1b: theta == pi/2 -> 1 physical SX pulse
    if abs(theta - 0.5 * math.pi) < tol:
        # RY(pi/2) = RZ(+pi/2) @ SX @ RZ(-pi/2)
        # U = RZ(phi + pi/2) @ SX @ RZ(lam - pi/2)
        # Apply right-to-left in circuit order (first gate in list acts first on |psi>):
        _add_rz(lam - 0.5 * math.pi)
        ops.append(GateInstruction("SX", (qubit,)))
        _add_rz(phi + 0.5 * math.pi)
        return ops

    # General Case 2: 2 physical SX pulses
    # Exact identity: SX @ RZ(pi - theta) @ SX = RY(theta) @ RZ(pi) (up to global phase)
    # => RY(theta) = SX @ RZ(pi - theta) @ SX @ RZ(-pi)
    # Matrix order: U = RZ(phi) @ SX @ RZ(pi - theta) @ SX @ RZ(lam - pi)
    # Circuit execution order (rightmost factor acts first):
    _add_rz(lam - math.pi)
    ops.append(GateInstruction("SX", (qubit,)))
    _add_rz(math.pi - theta)
    ops.append(GateInstruction("SX", (qubit,)))
    _add_rz(phi)
    return ops


def synthesize_1q_rxy_rz(
    U: np.ndarray,
    qubit: int,
    tol: float = 1e-7,
) -> List[GateInstruction]:
    """
    Synthesize any 1-qubit unitary U into at most ONE physical equatorial rotation
    RXY(theta, phi_axis) and ONE Virtual-Z gate RZ(alpha):
        U = RZ(phi) @ RY(theta) @ RZ(lam)
          = [RZ(phi) @ RY(theta) @ RZ(-phi)] @ RZ(phi + lam)
          = RXY(theta, phi + pi/2) @ RZ(phi + lam)
    In circuit order (right-to-left): first RZ(phi + lam), then RXY(theta, phi + pi/2).
    """
    theta, phi, lam, _ = zyz_euler_angles(U)
    ops: List[GateInstruction] = []
    rz_angle = normalize_angle(phi + lam)
    if abs(rz_angle) > tol:
        ops.append(GateInstruction("RZ", (qubit,), (rz_angle,), duration_ns=0.0))
    if abs(theta) > tol:
        axis_phi = normalize_angle(phi + 0.5 * math.pi)
        ops.append(GateInstruction("RXY", (qubit,), (theta, axis_phi)))
    return ops


def synthesize_1q_zyz(
    U: np.ndarray,
    qubit: int,
    tol: float = 1e-7,
) -> List[GateInstruction]:
    """Synthesize U into [RZ(lam), RY(theta), RZ(phi)] in circuit execution order."""
    theta, phi, lam, _ = zyz_euler_angles(U)
    ops: List[GateInstruction] = []
    if abs(normalize_angle(lam)) > tol:
        ops.append(GateInstruction("RZ", (qubit,), (normalize_angle(lam),)))
    if abs(theta) > tol:
        ops.append(GateInstruction("RY", (qubit,), (theta,)))
    if abs(normalize_angle(phi)) > tol:
        ops.append(GateInstruction("RZ", (qubit,), (normalize_angle(phi),)))
    return ops


def synthesize_1q_for_gate_set(
    U: np.ndarray,
    qubit: int,
    gate_set: NativeGateSet,
    tol: float = 1e-7,
) -> List[GateInstruction]:
    """
    Dispatch single-qubit unitary synthesis to the optimal decomposition
    for the target hardware's NativeGateSet.
    """
    mode = gate_set.decomposition_mode_1q.lower()
    allowed = set(gate_set.allowed_names())

    if mode == "rz_sx_rz" or ("SX" in allowed and "RZ" in allowed):
        return synthesize_1q_rz_sx_rz(
            U, qubit, allow_x_gate=("X" in allowed), tol=tol
        )
    if mode == "rxy_rz" or "RXY" in allowed:
        return synthesize_1q_rxy_rz(U, qubit, tol=tol)
    return synthesize_1q_zyz(U, qubit, tol=tol)
