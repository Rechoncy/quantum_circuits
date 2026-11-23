"""
Demo 2: Hardware-Aware Quantum Circuit Designer, Synthesizer & Optimizer
Given experimental operations (Superconducting CZ, Superconducting iSWAP, or Trapped-Ion MS)
and a target circuit or unitary matrix, computes the optimal native quantum gate sequence.
"""

from __future__ import annotations

import math
import numpy as np

from quantum_solver import (
    HardwareModel,
    NativeGateSet,
    QuantumCircuit,
    QuantumCircuitOptimizer,
    QuantumSolver,
)


def main() -> None:
    # =========================================================================
    # Scenario A: Optimizing a Logical Circuit for a Superconducting CZ Processor
    # =========================================================================
    print("=" * 88)
    print("SCENARIO A: Logical Circuit Compilation & Optimization -> Superconducting {Virtual-RZ, SX, CZ}")
    print("=" * 88)

    hw_sc_cz = HardwareModel.superconducting_processor(
        num_qubits=2,
        t1_us=30.0,
        t2_us=22.0,
        t_1q_ns=20.0,
        t_2q_ns=40.0,
        depol_1q=0.001,
        depol_2q=0.012,
        primary_2q_gate="CZ",
    )

    # User wants to implement a 2-qubit Quantum Fourier Transform (QFT) + SWAP + local rotations
    target_circ = QuantumCircuit(2, name="2Q_QFT_and_Entangler")
    target_circ.h(0)
    target_circ.cz(0, 1)
    target_circ.s(0)
    target_circ.h(1)
    target_circ.cnot(0, 1)
    target_circ.cnot(1, 0)
    target_circ.cnot(0, 1)  # Explicit 3-CNOT SWAP
    target_circ.rx(0, math.pi / 3.0)
    target_circ.ry(1, math.pi / 5.0)

    optimizer = QuantumCircuitOptimizer(hardware=hw_sc_cz)
    report_a = optimizer.optimize_circuit(target_circ, evaluate_noisy_fidelity=True)
    print(report_a.summary())
    print()

    # =========================================================================
    # Scenario B: Direct Unitary Synthesis across 3 Different Experimental Platforms
    # =========================================================================
    print("=" * 88)
    print("SCENARIO B: Direct Optimal Gate Synthesis for a Target 2-Qubit Unitary across Platforms")
    print("=" * 88)

    # Construct a 2-qubit Heisenberg interaction unitary U = exp(-i * (0.35 XX + 0.35 YY + 0.20 ZZ))
    from quantum_solver.core.gates import can_matrix

    U_heisenberg = can_matrix(0.70, 0.70, 0.40)

    for gate_set in (
        NativeGateSet.superconducting_cz(),
        NativeGateSet.superconducting_iswap(),
        NativeGateSet.trapped_ion(),
    ):
        rep = optimizer.synthesize_unitary(U_heisenberg, native_gate_set=gate_set)
        print(rep.summary())
        print()

    # =========================================================================
    # Scenario C: Execute the Synthesized Optimal Circuit on QuTiP & Stim
    # =========================================================================
    print("=" * 88)
    print("SCENARIO C: Running the Optimal Native Circuit on QuTiP & Stim Solvers")
    print("=" * 88)
    opt_circ = report_a.optimized_circuit.copy().measure_all()
    solver = QuantumSolver(hardware=hw_sc_cz)
    res_qutip = solver.solve(opt_circ, backend="qutip", shots=10000)
    print(res_qutip.summary_table())


if __name__ == "__main__":
    main()
