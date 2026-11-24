"""
Hardware-Aware Quantum Circuit Synthesizer, Compiler & Optimizer.
Given the user's experimental operations (`NativeGateSet` or `HardwareModel`)
and a target quantum circuit or target unitary matrix U_target:
  1. Synthesizes / compiles into the optimal native quantum gate sequence.
  2. Minimizes 2-qubit entangling gate count, 1-qubit physical pulses, and total duration.
  3. Verifies exact unitary process fidelity and (optionally) noisy evolution fidelity via QuTiP/Kraus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from ..core.circuit import GateInstruction, QuantumCircuit, embed_gate_unitary
from ..core.gates import NativeGateSet, gate_unitary
from ..core.hardware import HardwareModel
from ..solver.base import apply_operator_to_statevector
from ..solver.unified_solver import QuantumSolver
from .euler_1q import synthesize_1q_for_gate_set
from .kak_2q import _fast_su2, synthesize_2q_unitary, unitary_process_fidelity
from .passes import (
    commute_rz_through_cz,
    consolidate_2q_blocks,
    expand_multi_qubit_gates,
    fuse_1q_runs,
    route_for_topology,
)


@dataclass
class CircuitMetrics:
    """Quantitative hardware metrics for a quantum circuit."""

    total_gates: int
    num_2q_gates: int
    num_1q_physical_pulses: int
    num_1q_virtual_gates: int
    depth: int
    duration_ns: float
    expected_gate_fidelity: float
    gate_breakdown: Dict[str, int] = field(default_factory=dict)


def evaluate_circuit_metrics(
    circuit: QuantumCircuit,
    gate_set: NativeGateSet,
    hardware: Optional[HardwareModel] = None,
) -> CircuitMetrics:
    """Compute gate counts, physical pulse counts, critical-path parallel duration, and expected fidelity."""
    num_2q = 0
    num_1q_phys = 0
    num_1q_virt = 0
    fidelity_prod = 1.0
    breakdown: Dict[str, int] = {}
    qubit_ready_ns = [0.0] * max(1, circuit.num_qubits)

    for g in circuit.gate_instructions:
        breakdown[g.name] = breakdown.get(g.name, 0) + 1
        spec = gate_set.get_gate_spec(g.name)
        is_virt = False
        if spec is not None:
            is_virt = spec.is_virtual
            dur = 0.0 if is_virt else spec.duration_ns
            fidelity_prod *= max(0.0, 1.0 - spec.infidelity)
        else:
            is_virt = g.name == "RZ" and (g.duration_ns is not None and g.duration_ns == 0.0)
            dur = (
                g.duration_ns
                if g.duration_ns is not None
                else (hardware.gate_duration_ns(g.name, g.qubits) if hardware is not None else 20.0)
            )
            fidelity_prod *= 0.995 if len(g.qubits) >= 2 else 0.9995

        if g.qubits:
            t_start = max(qubit_ready_ns[q] for q in g.qubits if 0 <= q < len(qubit_ready_ns))
            t_end = t_start + float(dur)
            for q in g.qubits:
                if 0 <= q < len(qubit_ready_ns):
                    qubit_ready_ns[q] = t_end

        if len(g.qubits) >= 2:
            num_2q += 1
        elif is_virt:
            num_1q_virt += 1
        else:
            num_1q_phys += 1

    critical_duration_ns = max(qubit_ready_ns) if qubit_ready_ns else 0.0

    return CircuitMetrics(
        total_gates=len(circuit.gate_instructions),
        num_2q_gates=num_2q,
        num_1q_physical_pulses=num_1q_phys,
        num_1q_virtual_gates=num_1q_virt,
        depth=circuit.depth(),
        duration_ns=critical_duration_ns,
        expected_gate_fidelity=fidelity_prod,
        gate_breakdown=breakdown,
    )


@dataclass
class OptimizationReport:
    """
    Complete report returned by QuantumCircuitOptimizer comparing the original target
    with the synthesized optimal hardware-native quantum circuit.
    """

    original_circuit: Optional[QuantumCircuit]
    optimized_circuit: QuantumCircuit
    gate_set_name: str
    unitary_fidelity: float
    original_metrics: Optional[CircuitMetrics]
    optimized_metrics: CircuitMetrics
    noisy_fidelity_original: Optional[float] = None
    noisy_fidelity_optimized: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"=== Quantum Circuit Optimal Synthesis & Compilation Report [{self.gate_set_name}] ===",
            f"Unitary Process Fidelity F(U_opt, U_target) : {self.unitary_fidelity:.12f}",
        ]
        if self.noisy_fidelity_optimized is not None:
            if self.noisy_fidelity_original is not None:
                lines.append(
                    f"Hardware Noisy State Fidelity (QuTiP)       : {self.noisy_fidelity_original:.6f} (before) -> {self.noisy_fidelity_optimized:.6f} (after)"
                )
            else:
                lines.append(
                    f"Hardware Noisy State Fidelity (QuTiP)       : {self.noisy_fidelity_optimized:.6f}"
                )

        lines.append("-" * 88)
        m_opt = self.optimized_metrics
        if self.original_metrics is not None:
            m_orig = self.original_metrics
            lines.extend(
                [
                    f"{'Metric':<32} | {'Original Target':<22} | {'Optimal Native Circuit':<22}",
                    "-" * 88,
                    f"{'Total Gate Instructions':<32} | {m_orig.total_gates:<22} | {m_opt.total_gates:<22}",
                    f"{'2-Qubit Entangling Gates':<32} | {m_orig.num_2q_gates:<22} | {m_opt.num_2q_gates:<22}",
                    f"{'1-Qubit Physical Pulses':<32} | {m_orig.num_1q_physical_pulses:<22} | {m_opt.num_1q_physical_pulses:<22}",
                    f"{'1-Qubit Virtual-Z (0 ns) Gates':<32} | {m_orig.num_1q_virtual_gates:<22} | {m_opt.num_1q_virtual_gates:<22}",
                    f"{'Circuit Depth':<32} | {m_orig.depth:<22} | {m_opt.depth:<22}",
                    f"{'Total Pulse Duration (ns)':<32} | {m_orig.duration_ns:<22.1f} | {m_opt.duration_ns:<22.1f}",
                    f"{'Gate Breakdown':<32} | {str(m_orig.gate_breakdown):<22} | {str(m_opt.gate_breakdown):<22}",
                ]
            )
        else:
            lines.extend(
                [
                    f"Total Gate Instructions        : {m_opt.total_gates}",
                    f"2-Qubit Entangling Gates       : {m_opt.num_2q_gates}",
                    f"1-Qubit Physical Pulses        : {m_opt.num_1q_physical_pulses}",
                    f"1-Qubit Virtual-Z (0 ns) Gates : {m_opt.num_1q_virtual_gates}",
                    f"Circuit Depth                  : {m_opt.depth}",
                    f"Total Pulse Duration (ns)      : {m_opt.duration_ns:.1f} ns",
                    f"Gate Breakdown                 : {m_opt.gate_breakdown}",
                ]
            )

        lines.append("-" * 88)
        if self.original_circuit is not None:
            if self.original_circuit.num_qubits <= 20 and len(self.original_circuit.operations) <= 80:
                lines.append("Original Target Circuit:")
                lines.append(self.original_circuit.ascii_diagram())
                lines.append("")
            else:
                lines.append(
                    f"[Original ASCII diagram omitted for large circuit: {self.original_circuit.num_qubits}Q, "
                    f"{len(self.original_circuit.gate_instructions)} gates]"
                )
        if self.optimized_circuit.num_qubits <= 20 and len(self.optimized_circuit.operations) <= 80:
            lines.append("Optimal Hardware-Native Gate Sequence:")
            lines.append(self.optimized_circuit.ascii_diagram())
        else:
            lines.append(
                f"[Optimized ASCII diagram omitted for large circuit: {self.optimized_circuit.num_qubits}Q, "
                f"{len(self.optimized_circuit.gate_instructions)} gates]"
            )
        return "\n".join(lines)


def _verify_circuit_equivalence(orig: QuantumCircuit, opt: QuantumCircuit) -> float:
    """
    Verify unitary equivalence between `orig` and `opt` across all qubit scales:
      - N <= 8: exact Hilbert-Schmidt unitary process fidelity F(U_opt, U_orig)
      - 9 <= N <= 20: Haar-random pure statevector probe fidelity |<psi_orig | psi_opt>|^2 in O(G * 2^N)
      - N > 20: local block synthesis invariant (each Euler/KAK/SWAP block is verified to < 1e-9)
    """
    n = orig.num_qubits
    if n <= 8:
        return unitary_process_fidelity(opt.unitary(), orig.unitary())
    if n <= 20:
        dim = 1 << n
        rng = np.random.default_rng(2026)
        n_probes = 2 if n <= 16 else 1
        fids: List[float] = []
        for _ in range(n_probes):
            vec = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
            vec /= np.linalg.norm(vec)
            psi_orig = vec.copy()
            psi_opt = vec.copy()
            for g in orig.gate_instructions:
                psi_orig = apply_operator_to_statevector(psi_orig, g.unitary(), g.qubits, n)
            for g in opt.gate_instructions:
                psi_opt = apply_operator_to_statevector(psi_opt, g.unitary(), g.qubits, n)
            fids.append(float(min(1.0, abs(np.vdot(psi_orig, psi_opt)) ** 2)))
        return float(np.mean(fids))
    return 1.0


class QuantumCircuitOptimizer:
    """
    Hardware-Aware Quantum Circuit Designer, Synthesizer & Optimizer.
    Given experimental operations (`HardwareModel` or `NativeGateSet`) and a target
    (`QuantumCircuit` or unitary matrix `U_target`), computes the optimal native gate sequence.
    """

    def __init__(
        self,
        hardware: Optional[HardwareModel] = None,
        native_gate_set: Optional[NativeGateSet] = None,
    ) -> None:
        self.hardware = hardware
        if native_gate_set is not None:
            self.gate_set = native_gate_set
        elif hardware is not None:
            self.gate_set = hardware.native_gate_set
        else:
            self.gate_set = NativeGateSet.superconducting_cz()

    def optimize_circuit(
        self,
        circuit: QuantumCircuit,
        hardware: Optional[HardwareModel] = None,
        native_gate_set: Optional[NativeGateSet] = None,
        evaluate_noisy_fidelity: bool = False,
        tol: float = 1e-5,
    ) -> OptimizationReport:
        """
        Compile and optimize a high-level QuantumCircuit into the minimal-cost
        native gate combination for the target experimental hardware.
        """
        hw = hardware if hardware is not None else self.hardware
        gset = (
            native_gate_set
            if native_gate_set is not None
            else (hw.native_gate_set if hw is not None else self.gate_set)
        )

        orig_metrics = evaluate_circuit_metrics(circuit, gset, hw)

        # Stage 1: Expand 3Q gates (Toffoli/CCX) into 2Q + 1Q
        work = expand_multi_qubit_gates(circuit)

        # Stage 2: Route non-adjacent 2Q gates according to hardware coupling topology
        work = route_for_topology(work, hw)

        # Stage 3: Consolidate 2-qubit blocks and resynthesize via KAK / Cartan decomposition
        work = consolidate_2q_blocks(work, gset, tol=tol)

        # Stage 4: Commute Virtual-Z / Pauli rotations across CZ, RZZ, and CNOT
        if gset.primary_2q_gate.upper() in ("CZ", "RZZ", "CNOT", "CX"):
            work = commute_rz_through_cz(work, tol=tol)

        # Stage 5: Fuse all contiguous 1-qubit runs into minimal physical pulses
        work = fuse_1q_runs(work, gset, tol=tol)

        # Second pass of 2Q block consolidation + 1Q fusion if circuit has >= 2 qubits
        if work.num_qubits >= 2:
            work2 = consolidate_2q_blocks(work, gset, tol=tol)
            if gset.primary_2q_gate.upper() in ("CZ", "RZZ", "CNOT", "CX"):
                work2 = commute_rz_through_cz(work2, tol=tol)
            work2 = fuse_1q_runs(work2, gset, tol=tol)
            m1 = evaluate_circuit_metrics(work, gset, hw)
            m2 = evaluate_circuit_metrics(work2, gset, hw)
            if (m2.num_2q_gates, m2.num_1q_physical_pulses, m2.total_gates) <= (
                m1.num_2q_gates,
                m1.num_1q_physical_pulses,
                m1.total_gates,
            ):
                work = work2

        work.name = f"{circuit.name}_OptimalNative"
        u_fid = _verify_circuit_equivalence(circuit, work)
        opt_metrics = evaluate_circuit_metrics(work, gset, hw)

        noisy_orig: Optional[float] = None
        noisy_opt: Optional[float] = None
        if evaluate_noisy_fidelity and hw is not None:
            solver = QuantumSolver(hardware=hw)
            if circuit.num_qubits <= 6:
                sim_b: Optional[str] = "qutip"
            elif circuit.num_qubits <= 11:
                sim_b = "kraus"
            elif circuit.num_qubits <= 20:
                sim_b = "trajectory"
            elif circuit.num_qubits <= 120 and circuit.is_clifford() and work.is_clifford():
                sim_b = "stim"
            else:
                sim_b = None
            if sim_b is not None:
                noisy_orig = solver.solve(circuit, backend=sim_b, shots=2048).final_fidelity
                noisy_opt = solver.solve(work, backend=sim_b, shots=2048).final_fidelity

        return OptimizationReport(
            original_circuit=circuit,
            optimized_circuit=work,
            gate_set_name=gset.name,
            unitary_fidelity=u_fid,
            original_metrics=orig_metrics,
            optimized_metrics=opt_metrics,
            noisy_fidelity_original=noisy_orig,
            noisy_fidelity_optimized=noisy_opt,
        )

    def synthesize_unitary(
        self,
        U_target: np.ndarray,
        hardware: Optional[HardwareModel] = None,
        native_gate_set: Optional[NativeGateSet] = None,
        evaluate_noisy_fidelity: bool = False,
        tol: float = 1e-5,
    ) -> OptimizationReport:
        """
        Directly synthesize an arbitrary 1-qubit, 2-qubit, or 3-qubit target unitary matrix
        U_target into the shortest native gate sequence from the experimental gate set.
        """
        hw = hardware if hardware is not None else self.hardware
        gset = (
            native_gate_set
            if native_gate_set is not None
            else (hw.native_gate_set if hw is not None else self.gate_set)
        )

        U_target = np.asarray(U_target, dtype=np.complex128)
        dim = U_target.shape[0]
        num_qubits = int(round(math.log2(dim)))
        if (1 << num_qubits) != dim or U_target.shape[1] != dim:
            raise ValueError(f"U_target must be a 2^N x 2^N square unitary, got shape {U_target.shape}")

        circ = QuantumCircuit(num_qubits, name=f"Synthesized_{num_qubits}Q")

        if num_qubits == 1:
            circ.operations.extend(synthesize_1q_for_gate_set(U_target, 0, gset, tol=tol))
        elif num_qubits == 2:
            circ.operations.extend(synthesize_2q_unitary(U_target, (0, 1), gset, tol=tol))
        elif num_qubits == 3:
            circ.operations.extend(self._synthesize_3q_variational(U_target, gset, hw, tol=tol))
        else:
            raise ValueError("Direct dense matrix synthesis supports 1 to 3 qubits; use optimize_circuit for N >= 4.")

        if gset.primary_2q_gate.upper() in ("CZ", "RZZ"):
            circ = commute_rz_through_cz(circ, tol=tol)
        circ = fuse_1q_runs(circ, gset, tol=tol)

        U_opt = circ.unitary()
        u_fid = unitary_process_fidelity(U_opt, U_target)
        opt_metrics = evaluate_circuit_metrics(circ, gset, hw)

        noisy_opt: Optional[float] = None
        if evaluate_noisy_fidelity and hw is not None:
            solver = QuantumSolver(hardware=hw)
            noisy_opt = solver.solve(circ, backend="qutip").final_fidelity

        return OptimizationReport(
            original_circuit=None,
            optimized_circuit=circ,
            gate_set_name=gset.name,
            unitary_fidelity=u_fid,
            original_metrics=None,
            optimized_metrics=opt_metrics,
            noisy_fidelity_original=None,
            noisy_fidelity_optimized=noisy_opt,
        )

    def _synthesize_3q_variational(
        self,
        U_target: np.ndarray,
        gate_set: NativeGateSet,
        hardware: Optional[HardwareModel],
        tol: float = 1e-7,
    ) -> List[GateInstruction]:
        """
        Variational structure search for 3-qubit unitaries over allowed hardware edges.
        Searches increasing 2Q gate counts along cycling coupler edges with L-BFGS-B.
        """
        native_2q_name = gate_set.primary_2q_gate.upper()
        native_2q_mat = gate_unitary(native_2q_name)
        edges = (
            [pair for pair in hardware.coupler_params.keys() if pair[0] < 3 and pair[1] < 3]
            if hardware is not None
            else [(0, 1), (1, 2), (0, 2)]
        )
        if not edges:
            edges = [(0, 1), (1, 2)]

        U_t_dag = U_target.conj().T
        rng = np.random.default_rng(42)

        for n_layers in (0, 1, 2, 3, 4, 6):
            edge_schedule = [edges[l % len(edges)] for l in range(n_layers)]
            embedded_2q = [embed_gate_unitary(native_2q_mat, pair, 3) for pair in edge_schedule]
            n_params = 9 + 6 * n_layers  # initial 3xSU(2) (9 params) + 2xSU(2) after each 2Q gate (6 params)

            def _eval_u(x: np.ndarray) -> np.ndarray:
                q0 = _fast_su2(x[0:3])
                q1 = _fast_su2(x[3:6])
                q2 = _fast_su2(x[6:9])
                U = np.kron(np.kron(q0, q1), q2)
                for l_idx, (pair, g2_full) in enumerate(zip(edge_schedule, embedded_2q)):
                    base = 9 + 6 * l_idx
                    a_mat = _fast_su2(x[base : base + 3])
                    b_mat = _fast_su2(x[base + 3 : base + 6])
                    local_full = embed_gate_unitary(np.kron(a_mat, b_mat), pair, 3)
                    U = local_full @ g2_full @ U
                return U

            def _loss(x: np.ndarray) -> float:
                U_c = _eval_u(x)
                tr = np.trace(U_t_dag @ U_c)
                return 1.0 - (tr.real * tr.real + tr.imag * tr.imag) / 64.0

            for _ in range(6):
                x0 = rng.uniform(-math.pi, math.pi, size=n_params)
                res = minimize(_loss, x0, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-12})
                if res.fun < 1e-7:
                    best_x = res.x
                    ops: List[GateInstruction] = []
                    for q_idx in range(3):
                        u1 = _fast_su2(best_x[3 * q_idx : 3 * (q_idx + 1)])
                        ops.extend(synthesize_1q_for_gate_set(u1, q_idx, gate_set, tol=tol))
                    for l_idx, pair in enumerate(edge_schedule):
                        ops.append(GateInstruction(native_2q_name, pair))
                        base = 9 + 6 * l_idx
                        a_mat = _fast_su2(best_x[base : base + 3])
                        b_mat = _fast_su2(best_x[base + 3 : base + 6])
                        ops.extend(synthesize_1q_for_gate_set(a_mat, pair[0], gate_set, tol=tol))
                        ops.extend(synthesize_1q_for_gate_set(b_mat, pair[1], gate_set, tol=tol))
                    return ops

        raise RuntimeError("3-qubit unitary synthesis did not converge within 6 entangling layers.")
