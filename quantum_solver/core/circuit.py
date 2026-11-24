"""
Unified Quantum Circuit Intermediate Representation (QuantumCircuit IR).
Supports standard and experimental gates, inline noise channels, step-by-step
inspection, unitary matrix calculation, and ASCII circuit diagrams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .gates import CLIFFORD_GATE_NAMES, gate_unitary, is_clifford_angle
from .noise import NoiseChannel, ReadoutError


@dataclass
class GateInstruction:
    """Single quantum gate operation in a circuit."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    duration_ns: Optional[float] = None
    label: Optional[str] = None
    custom_matrix: Optional[np.ndarray] = None
    noise_after: List[NoiseChannel] = field(default_factory=list)
    layer: Optional[int] = None
    skip_hw_noise: bool = False
    custom_hamiltonian: Optional[np.ndarray] = None

    def unitary(self) -> np.ndarray:
        return gate_unitary(self.name, self.params, self.custom_matrix)

    def is_clifford(self) -> bool:
        uname = self.name.upper()
        if uname in CLIFFORD_GATE_NAMES:
            return True
        if uname in ("RX", "RY", "RZ", "P", "PHASE", "RXX", "RYY", "RZZ") and len(self.params) == 1:
            return is_clifford_angle(self.params[0])
        if uname == "RXY" and len(self.params) >= 2:
            return is_clifford_angle(self.params[0]) and is_clifford_angle(self.params[1])
        if uname == "MS":
            angle = self.params[0] if len(self.params) > 0 else math.pi / 2.0
            return is_clifford_angle(angle)
        return False

    def format_label(self) -> str:
        if self.label:
            return self.label
        if not self.params:
            return self.name.upper()
        p_str = ", ".join(f"{p:.3g}" for p in self.params)
        return f"{self.name.upper()}({p_str})"


@dataclass
class NoiseInstruction:
    """Standalone noise channel inserted at a specific point in the circuit."""

    channel: NoiseChannel
    label: Optional[str] = None


@dataclass
class MeasureInstruction:
    """Measurement instruction on one or more qubits."""

    qubits: Tuple[int, ...]
    readout_error: Optional[ReadoutError] = None


CircuitOperation = Union[GateInstruction, NoiseInstruction, MeasureInstruction]


def embed_gate_unitary(
    gate_mat: np.ndarray,
    qubits: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """
    Embed a k-qubit gate matrix into the full 2^N x 2^N Hilbert space
    using big-endian qubit ordering |q_0, q_1, ..., q_{N-1}>.
    """
    k = len(qubits)
    dim = 1 << num_qubits
    # Reshape state tensor with N indices of size 2: (q_0, ..., q_{N-1}, q_0', ..., q_{N-1}')
    gate_tensor = gate_mat.reshape([2] * (2 * k))
    full_eye = np.eye(dim, dtype=np.complex128).reshape([2] * (2 * num_qubits))

    # Apply gate_tensor along target axes of full_eye
    target_axes = list(qubits)
    rest_axes = [q for q in range(num_qubits) if q not in target_axes]
    perm = target_axes + rest_axes + [q + num_qubits for q in range(num_qubits)]
    inv_perm = np.argsort(perm)

    permuted = np.transpose(full_eye, perm).reshape(1 << k, -1)
    updated = (gate_mat @ permuted).reshape([2] * (2 * num_qubits))
    return np.transpose(updated, inv_perm).reshape(dim, dim)


class QuantumCircuit:
    """
    Unified Quantum Circuit Intermediate Representation.
    Supports fluent gate construction, inline noise injection, and multi-backend execution.
    """

    def __init__(self, num_qubits: int, name: str = "Circuit") -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = int(num_qubits)
        self.name = name
        self.operations: List[CircuitOperation] = []
        self.measured_qubits: List[int] = []

    def copy(self) -> "QuantumCircuit":
        new_circ = QuantumCircuit(self.num_qubits, name=self.name)
        for op in self.operations:
            if isinstance(op, GateInstruction):
                new_circ.operations.append(
                    GateInstruction(
                        name=op.name,
                        qubits=op.qubits,
                        params=op.params,
                        duration_ns=op.duration_ns,
                        label=op.label,
                        custom_matrix=op.custom_matrix.copy() if op.custom_matrix is not None else None,
                        noise_after=list(op.noise_after),
                        layer=op.layer,
                        skip_hw_noise=op.skip_hw_noise,
                        custom_hamiltonian=op.custom_hamiltonian.copy()
                        if op.custom_hamiltonian is not None
                        else None,
                    )
                )
            elif isinstance(op, NoiseInstruction):
                new_circ.operations.append(NoiseInstruction(channel=op.channel, label=op.label))
            elif isinstance(op, MeasureInstruction):
                new_circ.operations.append(
                    MeasureInstruction(qubits=op.qubits, readout_error=op.readout_error)
                )
        new_circ.measured_qubits = list(self.measured_qubits)
        return new_circ

    # ---- Generic Gate Addition ----

    def add_gate(
        self,
        name: str,
        qubits: Union[int, Sequence[int]],
        params: Sequence[float] = (),
        duration_ns: Optional[float] = None,
        label: Optional[str] = None,
        custom_matrix: Optional[np.ndarray] = None,
        noise_after: Optional[Sequence[NoiseChannel]] = None,
        layer: Optional[int] = None,
        skip_hw_noise: bool = False,
        custom_hamiltonian: Optional[np.ndarray] = None,
    ) -> "QuantumCircuit":
        q_tuple = (qubits,) if isinstance(qubits, int) else tuple(int(q) for q in qubits)
        for q in q_tuple:
            if q < 0 or q >= self.num_qubits:
                raise IndexError(f"Qubit index {q} out of bounds for {self.num_qubits}-qubit circuit.")
        if len(set(q_tuple)) != len(q_tuple):
            raise ValueError(f"Duplicate qubit indices in gate {name}: {q_tuple}")

        inst = GateInstruction(
            name=name.upper(),
            qubits=q_tuple,
            params=tuple(float(p) for p in params),
            duration_ns=duration_ns,
            label=label,
            custom_matrix=custom_matrix,
            noise_after=list(noise_after) if noise_after else [],
            layer=layer,
            skip_hw_noise=skip_hw_noise,
            custom_hamiltonian=custom_hamiltonian,
        )
        self.operations.append(inst)
        return self

    def add_noise(self, channel: NoiseChannel, label: Optional[str] = None) -> "QuantumCircuit":
        """Insert a standalone noise channel at the current step of the circuit."""
        self.operations.append(NoiseInstruction(channel=channel, label=label))
        return self

    # ---- Fluent Single-Qubit Gates ----

    def i(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("I", (q,), duration_ns=duration_ns)

    def x(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("X", (q,), duration_ns=duration_ns)

    def y(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("Y", (q,), duration_ns=duration_ns)

    def z(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("Z", (q,), duration_ns=duration_ns)

    def h(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("H", (q,), duration_ns=duration_ns)

    def s(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("S", (q,), duration_ns=duration_ns)

    def sdg(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("SDG", (q,), duration_ns=duration_ns)

    def t(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("T", (q,), duration_ns=duration_ns)

    def tdg(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("TDG", (q,), duration_ns=duration_ns)

    def sx(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("SX", (q,), duration_ns=duration_ns)

    def sxdg(self, q: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("SXDG", (q,), duration_ns=duration_ns)

    def rx(self, q: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RX", (q,), (theta,), duration_ns=duration_ns)

    def ry(self, q: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RY", (q,), (theta,), duration_ns=duration_ns)

    def rz(self, q: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RZ", (q,), (theta,), duration_ns=duration_ns)

    def rxy(self, q: int, theta: float, phi: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RXY", (q,), (theta, phi), duration_ns=duration_ns)

    def u3(self, q: int, theta: float, phi: float, lam: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("U3", (q,), (theta, phi, lam), duration_ns=duration_ns)

    # ---- Fluent Two-Qubit & Multi-Qubit Gates ----

    def cnot(self, control: int, target: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("CNOT", (control, target), duration_ns=duration_ns)

    def cx(self, control: int, target: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.cnot(control, target, duration_ns=duration_ns)

    def cz(self, q0: int, q1: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("CZ", (q0, q1), duration_ns=duration_ns)

    def swap(self, q0: int, q1: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("SWAP", (q0, q1), duration_ns=duration_ns)

    def iswap(self, q0: int, q1: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("ISWAP", (q0, q1), duration_ns=duration_ns)

    def sqiswap(self, q0: int, q1: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("SQISWAP", (q0, q1), duration_ns=duration_ns)

    def ecr(self, q0: int, q1: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("ECR", (q0, q1), duration_ns=duration_ns)

    def ms(self, q0: int, q1: int, theta: float = math.pi / 2.0, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("MS", (q0, q1), (theta,), duration_ns=duration_ns)

    def rxx(self, q0: int, q1: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RXX", (q0, q1), (theta,), duration_ns=duration_ns)

    def ryy(self, q0: int, q1: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RYY", (q0, q1), (theta,), duration_ns=duration_ns)

    def rzz(self, q0: int, q1: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("RZZ", (q0, q1), (theta,), duration_ns=duration_ns)

    def cr(self, q0: int, q1: int, theta: float, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("CR", (q0, q1), (theta,), duration_ns=duration_ns)

    def ccx(self, c0: int, c1: int, target: int, duration_ns: Optional[float] = None) -> "QuantumCircuit":
        return self.add_gate("CCX", (c0, c1, target), duration_ns=duration_ns)

    # ---- Fluent Noise Injection Helpers ----

    def depolarize(self, qubits: Union[int, Sequence[int]], p: float) -> "QuantumCircuit":
        q_tuple = (qubits,) if isinstance(qubits, int) else tuple(qubits)
        if len(q_tuple) == 1:
            return self.add_noise(NoiseChannel.depolarizing_1q(q_tuple[0], p))
        if len(q_tuple) == 2:
            return self.add_noise(NoiseChannel.depolarizing_2q(q_tuple[0], q_tuple[1], p))
        raise ValueError("depolarize supports 1 or 2 qubits")

    def amplitude_damping(self, qubit: int, gamma: float) -> "QuantumCircuit":
        return self.add_noise(NoiseChannel.amplitude_damping(qubit, gamma))

    def phase_damping(self, qubit: int, lam: float) -> "QuantumCircuit":
        return self.add_noise(NoiseChannel.phase_damping(qubit, lam))

    def pauli_error(self, qubit: int, px: float, py: float, pz: float) -> "QuantumCircuit":
        return self.add_noise(NoiseChannel.pauli_1q(qubit, px, py, pz))

    def thermal_relaxation(self, qubit: int, t1_ns: float, t2_ns: float, duration_ns: float) -> "QuantumCircuit":
        return self.add_noise(NoiseChannel.thermal_relaxation(qubit, t1_ns, t2_ns, duration_ns))

    # ---- Measurement ----

    def measure(
        self,
        qubits: Union[int, Sequence[int]],
        readout_error: Optional[ReadoutError] = None,
    ) -> "QuantumCircuit":
        q_tuple = (qubits,) if isinstance(qubits, int) else tuple(int(q) for q in qubits)
        self.operations.append(MeasureInstruction(qubits=q_tuple, readout_error=readout_error))
        for q in q_tuple:
            if q not in self.measured_qubits:
                self.measured_qubits.append(q)
        return self

    def measure_all(self, readout_error: Optional[ReadoutError] = None) -> "QuantumCircuit":
        return self.measure(tuple(range(self.num_qubits)), readout_error=readout_error)

    # ---- Circuit Properties & Unitary Evaluation ----

    @property
    def gate_instructions(self) -> List[GateInstruction]:
        return [op for op in self.operations if isinstance(op, GateInstruction)]

    def is_clifford(self) -> bool:
        return all(g.is_clifford() for g in self.gate_instructions)

    def gate_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for g in self.gate_instructions:
            counts[g.name] = counts.get(g.name, 0) + 1
        return counts

    def num_2q_gates(self) -> int:
        return sum(1 for g in self.gate_instructions if len(g.qubits) >= 2)

    def num_1q_gates(self) -> int:
        return sum(1 for g in self.gate_instructions if len(g.qubits) == 1)

    def depth(self) -> int:
        qubit_depths = [0] * self.num_qubits
        for g in self.gate_instructions:
            max_d = max(qubit_depths[q] for q in g.qubits)
            for q in g.qubits:
                qubit_depths[q] = max_d + 1
        return max(qubit_depths) if qubit_depths else 0

    def unitary(self) -> np.ndarray:
        """Compute the ideal total unitary matrix U of the circuit via O(G * 4^N) tensor contraction."""
        if self.num_qubits > 12:
            raise ValueError(
                f"Dense 2^{self.num_qubits}x2^{self.num_qubits} unitary matrix exceeds memory limits; "
                "use statevector probe verification or Trajectory/Stim backends."
            )
        dim = 1 << self.num_qubits
        U_tensor = np.eye(dim, dtype=np.complex128).reshape([2] * self.num_qubits + [dim])
        for g in self.gate_instructions:
            k = len(g.qubits)
            if k == 0:
                continue
            op_t = g.unitary().reshape([2] * (2 * k))
            U_tensor = np.tensordot(op_t, U_tensor, axes=(list(range(k, 2 * k)), list(g.qubits)))
            U_tensor = np.moveaxis(U_tensor, list(range(k)), list(g.qubits))
        return U_tensor.reshape((dim, dim))

    def ascii_diagram(self) -> str:
        """Render a clean text diagram of the quantum circuit."""
        lines = [f"q{q}: ─" for q in range(self.num_qubits)]
        for op in self.operations:
            if isinstance(op, GateInstruction):
                lbl = op.format_label()
                if len(op.qubits) == 1:
                    q = op.qubits[0]
                    box = f"[{lbl}]─"
                    pad = "─" * len(box)
                    for i in range(self.num_qubits):
                        lines[i] += box if i == q else pad
                elif len(op.qubits) == 2:
                    q0, q1 = op.qubits
                    b0 = f"●─" if op.name in ("CNOT", "CX", "CZ", "CR") else f"[{lbl}:0]─"
                    b1 = f"[{lbl}]─" if op.name in ("CNOT", "CX", "CZ", "CR") else f"[{lbl}:1]─"
                    width = max(len(b0), len(b1))
                    for i in range(self.num_qubits):
                        if i == q0:
                            lines[i] += b0.ljust(width, "─")
                        elif i == q1:
                            lines[i] += b1.ljust(width, "─")
                        else:
                            lines[i] += "─" * width
                else:
                    box = f"[{lbl}]─"
                    for i in range(self.num_qubits):
                        lines[i] += box if i in op.qubits else "─" * len(box)
            elif isinstance(op, NoiseInstruction):
                lbl = f"~{op.channel.kind}~"
                box = f"({lbl})─"
                for i in range(self.num_qubits):
                    lines[i] += box if i in op.channel.qubits else "─" * len(box)
            elif isinstance(op, MeasureInstruction):
                box = "[M]─"
                for i in range(self.num_qubits):
                    lines[i] += box if i in op.qubits else "─" * len(box)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(name='{self.name}', num_qubits={self.num_qubits}, "
            f"gates={len(self.gate_instructions)}, 2Q={self.num_2q_gates()}, depth={self.depth()})"
        )
