"""Hardware-aware quantum circuit synthesis, decomposition, and optimization engines."""

from .compiler import (
    CircuitMetrics,
    OptimizationReport,
    QuantumCircuitOptimizer,
    evaluate_circuit_metrics,
)
from .euler_1q import (
    synthesize_1q_for_gate_set,
    synthesize_1q_rxy_rz,
    synthesize_1q_rz_sx_rz,
    synthesize_1q_zyz,
    zyz_euler_angles,
)
from .kak_2q import (
    factorize_product_unitary_2q,
    is_locally_equivalent_2q,
    is_product_unitary_2q,
    makhlin_invariants,
    synthesize_2q_unitary,
    unitary_process_fidelity,
)
from .passes import (
    commute_rz_through_cz,
    consolidate_2q_blocks,
    expand_multi_qubit_gates,
    fuse_1q_runs,
    route_for_topology,
)

__all__ = [
    "QuantumCircuitOptimizer",
    "OptimizationReport",
    "CircuitMetrics",
    "evaluate_circuit_metrics",
    "zyz_euler_angles",
    "synthesize_1q_rz_sx_rz",
    "synthesize_1q_rxy_rz",
    "synthesize_1q_zyz",
    "synthesize_1q_for_gate_set",
    "makhlin_invariants",
    "is_product_unitary_2q",
    "factorize_product_unitary_2q",
    "is_locally_equivalent_2q",
    "synthesize_2q_unitary",
    "unitary_process_fidelity",
    "expand_multi_qubit_gates",
    "route_for_topology",
    "consolidate_2q_blocks",
    "commute_rz_through_cz",
    "fuse_1q_runs",
]
