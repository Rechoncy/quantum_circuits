"""
Demo 1: Multi-Backend Quantum Circuit Evolution & Noisy Measurement Solver
Comparing QuTiP (Pulse/Lindblad Master Equation), Kraus (Density Matrix),
Stim (Analytic Pauli-Twirled Stabilizer Sampler), and Hybrid Stim+QuTiP.
"""

from __future__ import annotations

from pathlib import Path

from quantum_solver import (
    HardwareModel,
    NoiseChannel,
    QuantumCircuit,
    QuantumSolver,
    plot_evolution_and_measurements,
)


def main() -> None:
    # 1. Define Experimental Superconducting Hardware Model (3 Qubits)
    hw = HardwareModel.superconducting_processor(
        num_qubits=3,
        t1_us=35.0,            # T1 = 35 us
        t2_us=25.0,            # T2* = 25 us
        t_1q_ns=20.0,          # 1Q pulse duration = 20 ns
        t_2q_ns=45.0,          # 2Q gate duration = 45 ns
        depol_1q=0.002,        # 0.2% 1Q control depolarizing error
        depol_2q=0.015,        # 1.5% 2Q control depolarizing error
        zz_crosstalk_khz=60.0, # 60 kHz static ZZ crosstalk
        readout_err=0.012,     # 1.2% readout bit-flip error
        primary_2q_gate="CZ",
    )

    # 2. Construct a 3-Qubit GHZ State Circuit with an extra inline noise injection
    circ = QuantumCircuit(3, name="Noisy_3Q_GHZ")
    circ.h(0)
    circ.cnot(0, 1)
    circ.cnot(1, 2)
    # Optional custom inline phase-flip noise on qubit 2 before measurement
    circ.add_noise(NoiseChannel.phase_flip(qubit=2, p=0.01))
    circ.measure_all()

    print("Target Circuit Diagram:")
    print(circ.ascii_diagram())
    print()

    # 3. Solve across all backends (QuTiP, Kraus, Stim, and Hybrid QuTiP->Stim)
    solver = QuantumSolver(hardware=hw, intra_gate_steps=15)
    results = solver.compare_backends(circ, shots=50000, seed=2026)

    for key in ("qutip", "kraus", "stim", "stim_qutip"):
        print(results[key].summary_table())
        print()

    print("=== Generated Stim Circuit (with QuTiP-Calibrated Pauli-Twirled Noise Channels) ===")
    print(results["stim_qutip"].stim_circuit_str)

    # 4. Save side-by-side evolution & measurement comparison plot
    out_img = Path(__file__).resolve().parent / "ghz_evolution_comparison.png"
    plot_evolution_and_measurements(
        results,
        save_path=str(out_img),
        title="3-Qubit Noisy GHZ Evolution: QuTiP Master Eq. vs Kraus vs Stim Sampler",
    )
    print(f"\nSaved multi-backend evolution comparison plot to: {out_img}")


if __name__ == "__main__":
    main()
