"""
Quantum Solver & Hardware-Aware Circuit Synthesizer Platform.

Modules:
  - core: QuantumCircuit IR, standard & experimental gates, NoiseChannel, Pauli Twirling, HardwareModel
  - solver: Multi-backend evolution engines (QuTiPBackend, StimBackend, KrausBackend, QuantumSolver)
  - optimizer: Hardware-aware circuit synthesizer & optimizer (Euler 1Q, KAK 2Q, Peephole passes, QuantumCircuitOptimizer)
  - visualization: Trajectory and multi-backend measurement comparison plots
"""

import os
os.environ.setdefault("MPMATH_NOGMPY", "1")

from .core import (
    CouplerPhysicalParams,
    GateInstruction,
    HardwareModel,
    MeasureInstruction,
    NativeGateSet,
    NativeGateSpec,
    NoiseChannel,
    NoiseInstruction,
    QuantumCircuit,
    QubitPhysicalParams,
    ReadoutError,
    gate_hamiltonian_generator,
    gate_unitary,
    pauli_twirl_kraus,
)
from .optimizer import (
    CircuitMetrics,
    OptimizationReport,
    QuantumCircuitOptimizer,
    makhlin_invariants,
    synthesize_1q_for_gate_set,
    synthesize_1q_rxy_rz,
    synthesize_1q_rz_sx_rz,
    synthesize_1q_zyz,
    synthesize_2q_unitary,
    unitary_process_fidelity,
    zyz_euler_angles,
)
from .solver import (
    EvolutionResult,
    KrausBackend,
    QuantumSolver,
    QuTiPBackend,
    StepSnapshot,
    StimBackend,
)
from .visualization import plot_evolution_and_measurements

__all__ = [
    "QuantumCircuit",
    "GateInstruction",
    "NoiseInstruction",
    "MeasureInstruction",
    "NativeGateSet",
    "NativeGateSpec",
    "HardwareModel",
    "QubitPhysicalParams",
    "CouplerPhysicalParams",
    "NoiseChannel",
    "ReadoutError",
    "gate_unitary",
    "gate_hamiltonian_generator",
    "pauli_twirl_kraus",
    "QuantumSolver",
    "QuTiPBackend",
    "StimBackend",
    "KrausBackend",
    "EvolutionResult",
    "StepSnapshot",
    "QuantumCircuitOptimizer",
    "OptimizationReport",
    "CircuitMetrics",
    "zyz_euler_angles",
    "synthesize_1q_for_gate_set",
    "synthesize_1q_rz_sx_rz",
    "synthesize_1q_rxy_rz",
    "synthesize_1q_zyz",
    "makhlin_invariants",
    "synthesize_2q_unitary",
    "unitary_process_fidelity",
    "plot_evolution_and_measurements",
]
