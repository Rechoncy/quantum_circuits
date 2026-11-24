"""
Core Project Engine for Quantum Studio v2:
  - Presets (Neutral Atom Rydberg GHZ, Superconducting QFT+SWAP, Bell Noise Compare, 1Q Rabi)
  - Project -> (QuantumCircuit, HardwareModel, initial_state) compiler
  - Custom native gate promotion into NativeGateSet for QuantumCircuitOptimizer
  - Bidirectional DSL <-> Visual Circuit synchronization & OpenQASM 3 export
  - JSON-serializable runners for Simulation, Optimization, and 1D/2D Parameter Sweeps
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import math
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize as scipy_minimize

from ..core.circuit import GateInstruction, MeasureInstruction, NoiseInstruction, QuantumCircuit
from ..core.gates import FIXED_MATRICES, NativeGateSet, NativeGateSpec
from ..core.hardware import CouplerPhysicalParams, HardwareModel, QubitPhysicalParams
from ..core.noise import NoiseChannel, ReadoutError
from ..gui.circuit_parser import eval_math_expr, extract_symbolic_parameters
from ..gui.sweep_engine import HARDWARE_SWEEP_PARAMS, apply_hardware_param
from ..optimizer.compiler import QuantumCircuitOptimizer
from ..solver.base import (
    EvolutionResult,
    compute_entanglement_from_density_matrix,
    evaluate_pauli_sum_density_matrix,
    evaluate_pauli_sum_statevector,
)
from ..solver.unified_solver import QuantumSolver
from .custom_gates import (
    _safe_complex_eval,
    evaluate_custom_gate_unitary,
    parse_pauli_hamiltonian,
    validate_unitary_matrix,
)
from .models import (
    BenchmarkHardwareRequest,
    CircuitOpModel,
    CustomGateDef,
    HardwareProfileModel,
    OptimizeRequest,
    ProjectModel,
    QubitSpec,
    SimulationSettingsModel,
    SweepRequest,
    SynthesizeUnitaryRequest,
    VariationalOptimizeRequest,
)


QUBIT_COLORS = [
    "#FF6B6B",
    "#0A84FF",
    "#34C759",
    "#FF9F0A",
    "#AF52DE",
    "#00C7BE",
    "#FF2D55",
    "#5856D6",
]


def _default_hardware_profiles() -> List[HardwareProfileModel]:
    return [
        HardwareProfileModel(
            id="hw_neutral_atom",
            name="中性原子 Rydberg",
            platform="neutral_atom",
            t1_us=1000.0,
            t2_us=800.0,
            t_1q_ns=400.0,
            t_2q_ns=240.0,
            depol_1q=0.0008,
            depol_2q=0.0045,
            zz_khz=2.0,
            readout_err=0.008,
            topology_mode="rydberg_radius",
            blockade_radius_um=6.5,
        ),
        HardwareProfileModel(
            id="hw_sc_cz",
            name="超导 CZ",
            platform="superconducting_cz",
            t1_us=35.0,
            t2_us=25.0,
            t_1q_ns=20.0,
            t_2q_ns=40.0,
            depol_1q=0.0015,
            depol_2q=0.012,
            zz_khz=50.0,
            readout_err=0.01,
            topology_mode="all",
        ),
        HardwareProfileModel(
            id="hw_sc_iswap",
            name="超导 iSWAP",
            platform="superconducting_iswap",
            t1_us=40.0,
            t2_us=28.0,
            t_1q_ns=20.0,
            t_2q_ns=32.0,
            depol_1q=0.0012,
            depol_2q=0.01,
            zz_khz=35.0,
            readout_err=0.01,
            topology_mode="all",
        ),
        HardwareProfileModel(
            id="hw_trapped_ion",
            name="离子阱 MS",
            platform="trapped_ion",
            t1_us=5000.0,
            t2_us=2000.0,
            t_1q_ns=2000.0,
            t_2q_ns=40000.0,
            depol_1q=0.0003,
            depol_2q=0.003,
            zz_khz=0.1,
            readout_err=0.003,
            topology_mode="all",
        ),
    ]


def _default_custom_gates() -> List[CustomGateDef]:
    return [
        CustomGateDef(
            id="cg_sqiswap",
            name="SQISWAP",
            display_label="√iSWAP",
            num_qubits=2,
            definition_type="matrix",
            matrix_exprs=[
                ["1", "0", "0", "0"],
                ["0", "1/sqrt(2)", "1j/sqrt(2)", "0"],
                ["0", "1j/sqrt(2)", "1/sqrt(2)", "0"],
                ["0", "0", "0", "1"],
            ],
            duration_ns=32.0,
            infidelity=0.004,
            color="#8B5CF6",
            is_native=False,
            description="超导电容耦合平方根 iSWAP 纠缠门",
        ),
        CustomGateDef(
            id="cg_cr_theta",
            name="CR_EXP",
            display_label="CR(θ)",
            num_qubits=2,
            definition_type="expression",
            expression="expm(-1j * theta / 2 * ZX)",
            param_names=["theta"],
            default_params=[round(math.pi / 2.0, 4)],
            duration_ns=120.0,
            infidelity=0.006,
            color="#7C3AED",
            is_native=False,
            description="参数化交叉共振门 exp(-i θ/2 Z⊗X)",
        ),
        CustomGateDef(
            id="cg_xy_ham",
            name="XY_PULSE",
            display_label="XY(H)",
            num_qubits=2,
            definition_type="hamiltonian",
            hamiltonian_str="0.5*XX + 0.5*YY",
            hamiltonian_scale=round(math.pi / 4.0, 4),
            duration_ns=60.0,
            infidelity=0.003,
            color="#6366F1",
            is_native=False,
            description="由 XY 交换哈密顿量生成的物理脉冲门",
        ),
        CustomGateDef(
            id="cg_bell_macro",
            name="BELL_2Q",
            display_label="Bell (复合)",
            num_qubits=2,
            definition_type="composite",
            sub_operations=[
                CircuitOpModel(id="m1", step=0, kind="gate", name="H", qubits=[0]),
                CircuitOpModel(id="m2", step=1, kind="gate", name="CNOT", qubits=[0, 1]),
            ],
            duration_ns=80.0,
            infidelity=0.005,
            color="#A855F7",
            is_native=False,
            description="Bell 态制备复合子线路 (H + CNOT)",
        ),
    ]


def build_preset_projects() -> List[ProjectModel]:
    """Return built-in starter projects showcasing Neutral Atom, Superconducting, and VQE workflows."""
    rydberg_proj = ProjectModel(
        name="Rydberg_GHZ.qproj",
        qubits=[
            QubitSpec(id=0, name="q0", color=QUBIT_COLORS[0], pos_x=0.0, pos_y=0.0),
            QubitSpec(id=1, name="q1", color=QUBIT_COLORS[1], pos_x=4.5, pos_y=0.0),
            QubitSpec(id=2, name="q2", color=QUBIT_COLORS[2], pos_x=9.0, pos_y=0.0),
            QubitSpec(id=3, name="q3", color=QUBIT_COLORS[3], pos_x=13.5, pos_y=0.0),
        ],
        custom_gates=_default_custom_gates(),
        operations=[
            CircuitOpModel(id="op1", step=0, kind="gate", name="H", qubits=[0]),
            CircuitOpModel(id="op2", step=0, kind="gate", name="RXY", qubits=[1], params=["theta", 0.0]),
            CircuitOpModel(id="op3", step=1, kind="gate", name="CZ", qubits=[0, 1]),
            CircuitOpModel(id="op4", step=2, kind="gate", name="SQISWAP", qubits=[2, 3]),
            CircuitOpModel(id="op5", step=3, kind="gate", name="CZ", qubits=[1, 2]),
            CircuitOpModel(id="op6", step=4, kind="measure", name="MEASURE", qubits=[0, 1, 2, 3]),
        ],
        variables={"theta": round(math.pi / 2.0, 4)},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_neutral_atom",
        sim_settings=SimulationSettingsModel(backend="qutip", shots=20000),
    )

    qft_opt_proj = ProjectModel(
        name="Superconducting_QFT_Opt.qproj",
        qubits=[
            QubitSpec(id=0, name="q0", color=QUBIT_COLORS[0], pos_x=0.0, pos_y=0.0),
            QubitSpec(id=1, name="q1", color=QUBIT_COLORS[1], pos_x=4.5, pos_y=0.0),
        ],
        custom_gates=_default_custom_gates(),
        operations=[
            CircuitOpModel(id="qop1", step=0, kind="gate", name="H", qubits=[0]),
            CircuitOpModel(id="qop2", step=1, kind="gate", name="CZ", qubits=[0, 1]),
            CircuitOpModel(id="qop3", step=2, kind="gate", name="S", qubits=[0]),
            CircuitOpModel(id="qop4", step=2, kind="gate", name="H", qubits=[1]),
            CircuitOpModel(id="qop5", step=3, kind="gate", name="CNOT", qubits=[0, 1]),
            CircuitOpModel(id="qop6", step=4, kind="gate", name="CNOT", qubits=[1, 0]),
            CircuitOpModel(id="qop7", step=5, kind="gate", name="CNOT", qubits=[0, 1]),
            CircuitOpModel(id="qop8", step=6, kind="gate", name="RY", qubits=[0], params=["theta"]),
            CircuitOpModel(id="qop9", step=6, kind="gate", name="RX", qubits=[1], params=["phi"]),
            CircuitOpModel(id="qop10", step=7, kind="measure", name="MEASURE", qubits=[0, 1]),
        ],
        variables={"theta": round(math.pi / 3.0, 4), "phi": round(math.pi / 5.0, 4)},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_sc_cz",
        sim_settings=SimulationSettingsModel(backend="compare", shots=20000),
    )

    bell_noise_proj = ProjectModel(
        name="Bell_Noisy_Compare.qproj",
        qubits=[
            QubitSpec(id=0, name="q0", color=QUBIT_COLORS[0], pos_x=0.0, pos_y=0.0),
            QubitSpec(id=1, name="q1", color=QUBIT_COLORS[1], pos_x=4.5, pos_y=0.0),
        ],
        custom_gates=_default_custom_gates(),
        operations=[
            CircuitOpModel(id="b1", step=0, kind="gate", name="H", qubits=[0]),
            CircuitOpModel(id="b2", step=1, kind="gate", name="CNOT", qubits=[0, 1]),
            CircuitOpModel(id="b3", step=2, kind="noise", name="DEPOLARIZE", qubits=[0, 1], params=["p_noise"]),
            CircuitOpModel(id="b4", step=3, kind="measure", name="MEASURE", qubits=[0, 1]),
        ],
        variables={"p_noise": 0.02},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_sc_cz",
        sim_settings=SimulationSettingsModel(backend="compare", shots=20000),
    )

    rabi_proj = ProjectModel(
        name="SingleQubit_Rabi.qproj",
        qubits=[
            QubitSpec(id=0, name="atom_0", color=QUBIT_COLORS[0], pos_x=0.0, pos_y=0.0),
        ],
        custom_gates=_default_custom_gates(),
        operations=[
            CircuitOpModel(id="r1", step=0, kind="gate", name="RX", qubits=[0], params=["theta"]),
            CircuitOpModel(id="r2", step=1, kind="gate", name="RZ", qubits=[0], params=["phi"]),
            CircuitOpModel(id="r3", step=2, kind="gate", name="SX", qubits=[0]),
            CircuitOpModel(id="r4", step=3, kind="measure", name="MEASURE", qubits=[0]),
        ],
        variables={"theta": round(math.pi / 2.0, 4), "phi": 0.4},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_sc_cz",
        sim_settings=SimulationSettingsModel(backend="qutip", shots=20000),
    )

    # ---- Flagship Preset 5: 16Q 2D Transverse-Field Ising VQE (4x4 Neutral Atom Grid) ----
    vqe_qubits = [
        QubitSpec(
            id=r * 4 + c,
            name=f"q{r * 4 + c}",
            color=QUBIT_COLORS[(r * 4 + c) % len(QUBIT_COLORS)],
            pos_x=round(c * 5.0, 1),
            pos_y=round(r * 5.0, 1),
        )
        for r in range(4)
        for c in range(4)
    ]
    vqe_ops: List[CircuitOpModel] = []
    for q in range(16):
        vqe_ops.append(CircuitOpModel(id=f"vqe_ry_{q}", step=0, kind="gate", name="RY", qubits=[q], params=["theta"]))
    # Horizontal even & odd CZ layers (adjacent wires, zero crossing)
    for r in range(4):
        for c in (0, 2):
            u = r * 4 + c
            vqe_ops.append(CircuitOpModel(id=f"vqe_cz_he_{u}", step=1, kind="gate", name="CZ", qubits=[u, u + 1]))
        u_odd = r * 4 + 1
        vqe_ops.append(CircuitOpModel(id=f"vqe_cz_ho_{u_odd}", step=2, kind="gate", name="CZ", qubits=[u_odd, u_odd + 1]))
    # Vertical CZ layers scheduled cleanly
    v_col = [3] * 16
    for r in range(3):
        for c in range(4):
            u, v = r * 4 + c, (r + 1) * 4 + c
            st = max(v_col[q] for q in range(u, v + 1))
            vqe_ops.append(CircuitOpModel(id=f"vqe_cz_v_{u}_{v}", step=st, kind="gate", name="CZ", qubits=[u, v]))
            for q in range(u, v + 1):
                v_col[q] = st + 1
    rx_step = max(v_col)
    for q in range(16):
        vqe_ops.append(CircuitOpModel(id=f"vqe_rx_{q}", step=rx_step, kind="gate", name="RX", qubits=[q], params=["gamma"]))
    vqe_ops.append(CircuitOpModel(id="vqe_m", step=rx_step + 1, kind="measure", name="MEASURE", qubits=list(range(16))))

    vqe_16q_proj = ProjectModel(
        name="Ising_2D_VQE_16Q.qproj",
        qubits=vqe_qubits,
        custom_gates=_default_custom_gates(),
        operations=vqe_ops,
        variables={"theta": round(math.pi / 3.0, 4), "gamma": round(math.pi / 6.0, 4)},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_neutral_atom",
        sim_settings=SimulationSettingsModel(
            backend="trajectory",
            shots=20000,
            enable_t1_t2=False,
            enable_depol=False,
            enable_zz=False,
            enable_readout=False,
        ),
    )

    # ---- Flagship Preset 6: 17Q d=3 Rotated Surface Code QEC Syndrome Cycle (Stim) ----
    sc_qubits: List[QubitSpec] = []
    for i in range(9):
        r, c = divmod(i, 3)
        sc_qubits.append(
            QubitSpec(id=i, name=f"d{i}", color=QUBIT_COLORS[1], pos_x=round(c * 6.0, 1), pos_y=round(r * 6.0, 1))
        )
    x_anc_coords = [(3.0, 3.0), (9.0, 9.0), (3.0, -3.0), (9.0, 15.0)]
    z_anc_coords = [(9.0, 3.0), (3.0, 9.0), (-3.0, 9.0), (15.0, 3.0)]
    for idx, (ax, ay) in enumerate(x_anc_coords):
        sc_qubits.append(QubitSpec(id=9 + idx, name=f"x_anc{idx}", color=QUBIT_COLORS[3], pos_x=ax, pos_y=ay))
    for idx, (zx, zy) in enumerate(z_anc_coords):
        sc_qubits.append(QubitSpec(id=13 + idx, name=f"z_anc{idx}", color=QUBIT_COLORS[2], pos_x=zx, pos_y=zy))

    sc_col = [0] * 17
    sc_ops: List[CircuitOpModel] = []
    for a in range(9, 13):
        sc_ops.append(CircuitOpModel(id=f"sc_h_in_{a}", step=0, kind="gate", name="H", qubits=[a]))
        sc_col[a] = 1
    stabilizer_pairs = [
        # X-stabilizers (control = ancilla, target = data)
        (9, 0), (9, 1), (9, 3), (9, 4),
        (10, 4), (10, 5), (10, 7), (10, 8),
        (11, 0), (11, 1),
        (12, 7), (12, 8),
        # Z-stabilizers (control = data, target = ancilla)
        (1, 13), (2, 13), (4, 13), (5, 13),
        (3, 14), (4, 14), (6, 14), (7, 14),
        (3, 15), (6, 15),
        (2, 16), (5, 16),
    ]
    for p_idx, (ctrl, targ) in enumerate(stabilizer_pairs):
        lo, hi = min(ctrl, targ), max(ctrl, targ)
        st = max(sc_col[q] for q in range(lo, hi + 1))
        sc_ops.append(CircuitOpModel(id=f"sc_cx_{p_idx}", step=st, kind="gate", name="CNOT", qubits=[ctrl, targ]))
        for q in range(lo, hi + 1):
            sc_col[q] = st + 1
    for a in range(9, 13):
        st = sc_col[a]
        sc_ops.append(CircuitOpModel(id=f"sc_h_out_{a}", step=st, kind="gate", name="H", qubits=[a]))
        sc_col[a] = st + 1
    m_st = max(sc_col)
    sc_ops.append(CircuitOpModel(id="sc_meas", step=m_st, kind="measure", name="MEASURE", qubits=list(range(17))))

    surface_17q_proj = ProjectModel(
        name="Rotated_Surface_Code_17Q.qproj",
        qubits=sc_qubits,
        custom_gates=_default_custom_gates(),
        operations=sc_ops,
        variables={"p_err": 0.005},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_neutral_atom",
        sim_settings=SimulationSettingsModel(backend="stim", shots=20000),
    )

    # ---- Flagship Preset 7: 32Q Log-Depth Binary-Tree Rydberg GHZ (5 Entangling Layers) ----
    ghz32_qubits = [
        QubitSpec(
            id=i,
            name=f"q{i}",
            color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
            pos_x=round((i % 8) * 5.0, 1),
            pos_y=round((i // 8) * 5.0, 1),
        )
        for i in range(32)
    ]
    ghz32_ops: List[CircuitOpModel] = [
        CircuitOpModel(id="g32_h0", step=0, kind="gate", name="H", qubits=[0])
    ]
    for layer_idx, stride in enumerate((16, 8, 4, 2, 1), start=1):
        for src in range(0, 32, stride * 2):
            dst = src + stride
            ghz32_ops.append(
                CircuitOpModel(
                    id=f"g32_cx_{layer_idx}_{src}_{dst}",
                    step=layer_idx,
                    kind="gate",
                    name="CNOT",
                    qubits=[src, dst],
                )
            )
    ghz32_ops.append(CircuitOpModel(id="g32_m", step=6, kind="measure", name="MEASURE", qubits=list(range(32))))

    ghz_32q_proj = ProjectModel(
        name="Rydberg_LogDepth_GHZ_32Q.qproj",
        qubits=ghz32_qubits,
        custom_gates=_default_custom_gates(),
        operations=ghz32_ops,
        variables={},
        hardware_profiles=_default_hardware_profiles(),
        active_hardware_id="hw_neutral_atom",
        sim_settings=SimulationSettingsModel(backend="stim", shots=20000),
    )

    return [
        rydberg_proj,
        qft_opt_proj,
        bell_noise_proj,
        rabi_proj,
        vqe_16q_proj,
        surface_17q_proj,
        ghz_32q_proj,
    ]


# ---------------------------------------------------------------------------
# Initial State, Topology, HardwareModel, and Circuit Compilation
# ---------------------------------------------------------------------------

def _qubit_init_ket(qspec: QubitSpec) -> np.ndarray:
    s = qspec.init_state
    if s == "|0>":
        return np.array([1.0, 0.0], dtype=np.complex128)
    if s == "|1>":
        return np.array([0.0, 1.0], dtype=np.complex128)
    if s == "|+>":
        return np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    if s == "|->":
        return np.array([1.0, -1.0], dtype=np.complex128) / math.sqrt(2.0)
    if s == "|+i>":
        return np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0)
    if s == "|-i>":
        return np.array([1.0, -1.0j], dtype=np.complex128) / math.sqrt(2.0)
    # Custom Bloch angles
    th = float(qspec.init_theta)
    ph = float(qspec.init_phi)
    return np.array(
        [math.cos(0.5 * th), np.exp(1.0j * ph) * math.sin(0.5 * th)],
        dtype=np.complex128,
    )


def build_initial_statevector(qubits: Sequence[QubitSpec]) -> Optional[np.ndarray]:
    if not qubits:
        return None
    if all(q.init_state == "|0>" for q in qubits):
        return None
    psi = np.array([1.0 + 0.0j], dtype=np.complex128)
    for q in qubits:
        psi = np.kron(psi, _qubit_init_ket(q))
    return psi


def compute_topology_edges(
    qubits: Sequence[QubitSpec],
    profile: HardwareProfileModel,
    operations: Optional[Sequence[CircuitOpModel]] = None,
) -> List[Tuple[int, int]]:
    n = len(qubits)
    if n <= 1:
        return []
    mode = profile.topology_mode
    if mode == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if mode == "custom":
        valid = [
            (min(u, v), max(u, v))
            for u, v in profile.edges
            if 0 <= u < n and 0 <= v < n and u != v
        ]
        return sorted(set(valid))
    if mode == "rydberg_radius":
        rb = max(0.1, float(profile.blockade_radius_um))
        edges: List[Tuple[int, int]] = []
        # For large arrays (> 120 qubits), bucket 2D coordinates into spatial cells of size rb for O(N) neighbor lookup
        if n > 120:
            grid: Dict[Tuple[int, int], List[int]] = {}
            for idx, q in enumerate(qubits):
                cx = int(math.floor(q.pos_x / rb))
                cy = int(math.floor(q.pos_y / rb))
                grid.setdefault((cx, cy), []).append(idx)
            for (cx, cy), cell_qs in grid.items():
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nbr_qs = grid.get((cx + dx, cy + dy))
                        if not nbr_qs:
                            continue
                        for i in cell_qs:
                            for j in nbr_qs:
                                if i < j:
                                    dist = math.hypot(
                                        qubits[i].pos_x - qubits[j].pos_x,
                                        qubits[i].pos_y - qubits[j].pos_y,
                                    )
                                    if dist <= rb + 1e-9:
                                        edges.append((i, j))
            return sorted(edges)
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(qubits[i].pos_x - qubits[j].pos_x, qubits[i].pos_y - qubits[j].pos_y)
                if dist <= rb + 1e-9:
                    edges.append((i, j))
        return edges
    # All-to-all: for n > 32, materialize linear chain + all pairs actually used by operations to avoid O(N^2) bloat
    if n > 32:
        sparse_pairs = {(i, i + 1) for i in range(n - 1)}
        if operations:
            for op in operations:
                if op.kind == "gate" and len(op.qubits) >= 2:
                    for a_idx in range(len(op.qubits)):
                        for b_idx in range(a_idx + 1, len(op.qubits)):
                            u, v = op.qubits[a_idx], op.qubits[b_idx]
                            if 0 <= u < n and 0 <= v < n and u != v:
                                sparse_pairs.add((min(u, v), max(u, v)))
        return sorted(sparse_pairs)
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def get_active_hardware_profile(project: ProjectModel) -> HardwareProfileModel:
    for hp in project.hardware_profiles:
        if hp.id == project.active_hardware_id:
            return hp
    if project.hardware_profiles:
        return project.hardware_profiles[0]
    return _default_hardware_profiles()[0]


def build_hardware_model(
    project: ProjectModel,
    preferred_native_2q: Optional[str] = None,
) -> HardwareModel:
    n = max(1, len(project.qubits))
    hp = get_active_hardware_profile(project)
    sim = project.sim_settings
    edges = compute_topology_edges(project.qubits, hp, operations=project.operations)

    t1_base = hp.t1_us if sim.enable_t1_t2 else float("inf")
    t2_base = hp.t2_us if sim.enable_t1_t2 else float("inf")
    dep1_base = hp.depol_1q if sim.enable_depol else 0.0
    dep2_base = hp.depol_2q if sim.enable_depol else 0.0
    zz_base = hp.zz_khz if sim.enable_zz else 0.0
    ro_base = hp.readout_err if sim.enable_readout else 0.0

    if hp.platform == "neutral_atom":
        hw = HardwareModel.neutral_atom_processor(
            num_qubits=n,
            t1_us=t1_base,
            t2_us=t2_base,
            t_1q_ns=hp.t_1q_ns,
            t_2q_ns=hp.t_2q_ns,
            depol_1q=dep1_base,
            depol_2q=dep2_base,
            zz_crosstalk_khz=zz_base,
            readout_err=ro_base,
            topology=edges,
        )
    else:
        p2q_map = {
            "superconducting_cz": "CZ",
            "superconducting_iswap": "ISWAP",
            "superconducting_cnot": "CNOT",
            "trapped_ion": "MS",
        }
        p2q = p2q_map.get(hp.platform, "CZ")
        hw = HardwareModel.superconducting_processor(
            num_qubits=n,
            t1_us=t1_base,
            t2_us=t2_base,
            t_1q_ns=hp.t_1q_ns,
            t_2q_ns=hp.t_2q_ns,
            depol_1q=dep1_base,
            depol_2q=dep2_base,
            zz_crosstalk_khz=zz_base,
            readout_err=ro_base,
            primary_2q_gate=p2q,
            topology=edges,
        )
        if hp.platform == "trapped_ion":
            hw.native_gate_set = NativeGateSet.trapped_ion(
                hp.t_1q_ns, hp.t_2q_ns, dep1_base, dep2_base
            )

    hw.name = hp.name
    hw.pulse_shape = hp.pulse_shape

    # Apply per-qubit overrides from Quantum Register
    for idx, qspec in enumerate(project.qubits):
        qp = hw.qubit_params[idx]
        if sim.enable_t1_t2:
            if qspec.t1_us is not None:
                qp.t1_ns = qspec.t1_us * 1000.0
            if qspec.t2_us is not None:
                qp.t2_ns = qspec.t2_us * 1000.0
        if qspec.freq_ghz is not None:
            qp.freq_ghz = qspec.freq_ghz
        if sim.enable_depol and qspec.depol_1q is not None:
            qp.depol_1q = qspec.depol_1q
        if sim.enable_readout and qspec.readout_err is not None:
            qp.readout_error = ReadoutError(qspec.readout_err, qspec.readout_err * 1.1)

    # Promote custom gates marked `is_native=True` into NativeGateSet!
    cg_lookup = {cg.name.upper(): cg for cg in project.custom_gates}
    custom_native_2q_names: List[str] = []
    for cg in project.custom_gates:
        if not cg.is_native:
            continue
        try:
            u_cg, _ = evaluate_custom_gate_unitary(cg, custom_gate_map=cg_lookup)
            u_copy = u_cg.copy()
            spec = NativeGateSpec(
                name=cg.name,
                num_qubits=cg.num_qubits,
                num_params=len(cg.param_names),
                duration_ns=cg.duration_ns,
                infidelity=cg.infidelity,
                is_virtual=False,
                matrix_fn=lambda *_args, _u=u_copy: _u,
            )
            if cg.num_qubits == 1:
                hw.native_gate_set.single_qubit_gates.append(spec)
            elif cg.num_qubits == 2:
                hw.native_gate_set.two_qubit_gates.append(spec)
                custom_native_2q_names.append(cg.name)
        except Exception:
            continue

    if preferred_native_2q:
        hw.native_gate_set.primary_2q_gate = preferred_native_2q
    elif custom_native_2q_names:
        # If user explicitly marked a custom 2Q gate as native, use it as primary 2Q entangler for optimizer
        hw.native_gate_set.primary_2q_gate = custom_native_2q_names[-1]

    return hw


def _resolve_op_params(raw_params: Sequence[Any], variables: Dict[str, float]) -> Tuple[float, ...]:
    resolved: List[float] = []
    for p in raw_params:
        if isinstance(p, (int, float)):
            resolved.append(float(p))
        else:
            resolved.append(eval_math_expr(str(p), variables))
    return tuple(resolved)


def compile_project_circuit(
    project: ProjectModel,
    variables_override: Optional[Dict[str, float]] = None,
) -> QuantumCircuit:
    """Compile a ProjectModel's operations + custom gates into an executable QuantumCircuit."""
    n = max(1, len(project.qubits))
    vars_map = dict(project.variables)
    if variables_override:
        vars_map.update(variables_override)

    custom_by_name: Dict[str, CustomGateDef] = {
        cg.name.upper(): cg for cg in project.custom_gates
    }

    circ = QuantumCircuit(num_qubits=n, name=project.name)
    # Sort operations deterministically by (step, index)
    ordered_ops = [
        op for _, op in sorted(enumerate(project.operations), key=lambda pair: (pair[1].step, pair[0]))
    ]

    for op in ordered_ops:
        if op.kind == "measure":
            valid_q = [q for q in op.qubits if 0 <= q < n]
            if valid_q:
                circ.measure(valid_q)
            continue

        if op.kind == "noise":
            p_vals = _resolve_op_params(op.params, vars_map)
            uname = op.name.upper()
            q_list = [q for q in op.qubits if 0 <= q < n]
            if not q_list:
                continue
            p0 = p_vals[0] if p_vals else 0.01
            if uname in ("DEPOLARIZE", "DEPOL", "DEPOLARIZE1", "DEPOLARIZE2"):
                if len(q_list) == 1:
                    circ.depolarize(q_list[0], p0)
                else:
                    circ.depolarize((q_list[0], q_list[1]), p0)
            elif uname in ("AMPLITUDE_DAMPING", "AD", "T1_DECAY"):
                for q in q_list:
                    circ.amplitude_damping(q, p0)
            elif uname in ("PHASE_DAMPING", "PD", "DEPHASING"):
                for q in q_list:
                    circ.phase_damping(q, p0)
            elif uname in ("BIT_FLIP", "X_ERROR"):
                for q in q_list:
                    circ.add_noise(NoiseChannel.bit_flip(q, p0))
            elif uname in ("PHASE_FLIP", "Z_ERROR"):
                for q in q_list:
                    circ.add_noise(NoiseChannel.phase_flip(q, p0))
            elif uname in ("PAULI_ERROR", "PAULI1Q") and len(p_vals) >= 3:
                for q in q_list:
                    circ.pauli_error(q, p_vals[0], p_vals[1], p_vals[2])
            continue

        # Gate operation
        uname = op.name.upper()
        p_vals = _resolve_op_params(op.params, vars_map)
        valid_q = tuple(q for q in op.qubits if 0 <= q < n)
        if not valid_q:
            continue

        noise_after: List[NoiseChannel] = []
        if op.noise_mode == "custom" and op.custom_depol is not None and op.custom_depol > 0:
            if len(valid_q) == 1:
                noise_after.append(NoiseChannel.depolarizing_1q(valid_q[0], op.custom_depol))
            elif len(valid_q) == 2:
                noise_after.append(NoiseChannel.depolarizing_2q(valid_q[0], valid_q[1], op.custom_depol))

        skip_hw = op.noise_mode == "none"

        def _emit_gate(
            g_name: str,
            g_qubits: Tuple[int, ...],
            g_params: Tuple[float, ...],
            g_dur: Optional[float],
            g_noise: List[NoiseChannel],
            g_layer: Optional[int],
            g_skip_hw: bool,
            depth: int = 0,
        ) -> None:
            if depth > 6:
                raise ValueError(f"复合门 '{g_name}' 存在循环引用或嵌套层级超过 6 层")
            g_uname = g_name.upper()
            if g_uname in custom_by_name:
                cg = custom_by_name[g_uname]
                if len(g_qubits) != cg.num_qubits:
                    return
                if cg.definition_type == "composite" and cg.sub_operations:
                    for sub in sorted(cg.sub_operations, key=lambda s: s.step):
                        mapped_q = tuple(g_qubits[lq] for lq in sub.qubits if 0 <= lq < len(g_qubits))
                        sub_p = _resolve_op_params(sub.params, vars_map)
                        if sub.kind == "gate" and mapped_q:
                            _emit_gate(
                                sub.name,
                                mapped_q,
                                sub_p,
                                sub.duration_ns,
                                [],
                                None,
                                g_skip_hw,
                                depth=depth + 1,
                            )
                    if g_noise:
                        for ch in g_noise:
                            circ.add_noise(ch)
                else:
                    u_mat, h_mat = evaluate_custom_gate_unitary(
                        cg, g_params if g_params else None, custom_gate_map=custom_by_name
                    )
                    dur = g_dur if g_dur is not None else cg.duration_ns
                    direct_h = None
                    if h_mat is not None:
                        scale = float(g_params[0]) if (cg.param_names and g_params) else float(cg.hamiltonian_scale)
                        direct_h = scale * h_mat
                    circ.add_gate(
                        name=cg.name,
                        qubits=g_qubits,
                        params=g_params,
                        duration_ns=dur,
                        label=cg.display_label or cg.name,
                        custom_matrix=u_mat,
                        noise_after=g_noise,
                        layer=g_layer,
                        skip_hw_noise=g_skip_hw,
                        custom_hamiltonian=direct_h,
                    )
            else:
                circ.add_gate(
                    name=g_uname,
                    qubits=g_qubits,
                    params=g_params,
                    duration_ns=g_dur,
                    noise_after=g_noise,
                    layer=g_layer,
                    skip_hw_noise=g_skip_hw,
                )

        _emit_gate(uname, valid_q, p_vals, op.duration_ns, noise_after, op.step, skip_hw)

    return circ


# ---------------------------------------------------------------------------
# Bidirectional DSL <-> Project Operations & OpenQASM 3 Export
# ---------------------------------------------------------------------------

def project_to_dsl(project: ProjectModel) -> str:
    """Serialize a ProjectModel's qubits and operations into clean DSL code."""
    lines = [f"qubits {len(project.qubits)}"]
    ordered = [
        op for _, op in sorted(enumerate(project.operations), key=lambda pair: (pair[1].step, pair[0]))
    ]
    for op in ordered:
        q_str = " ".join(str(q) for q in op.qubits)
        if op.kind == "measure":
            lines.append(f"MEASURE {q_str}")
        else:
            if op.params:
                p_str = ", ".join(
                    f"{p:.4g}" if isinstance(p, (int, float)) else str(p)
                    for p in op.params
                )
                lines.append(f"{op.name}({p_str}) {q_str}")
            else:
                lines.append(f"{op.name} {q_str}")
    return "\n".join(lines)


def dsl_to_project_ops(
    dsl: str,
    project: ProjectModel,
) -> Tuple[List[QubitSpec], List[CircuitOpModel], Dict[str, float]]:
    """
    Parse user-edited DSL code back into structured QubitSpec list, CircuitOpModel list
    (with ASAP column/step scheduling), and updated symbolic variables dictionary.
    Supports both built-in gates and all custom gates in `project.custom_gates`!
    Automatically delegates to `openqasm_to_project_ops` if OpenQASM 2.0/3.0 text is pasted.
    """
    stripped_head = dsl.lstrip()
    if (
        stripped_head.upper().startswith("OPENQASM")
        or re.search(r"^\s*(include\s+\"|qreg\s+\w+\s*\[|qubit\s*\[)", dsl, flags=re.MULTILINE)
    ):
        return openqasm_to_project_ops(dsl, project)

    raw_lines = [ln.split("#", 1)[0].strip() for ln in dsl.splitlines()]
    lines = [ln for ln in raw_lines if ln]

    declared_n: Optional[int] = None
    max_q = -1
    inst_re = re.compile(r"^([A-Za-z0-9_]+)(?:\(([^)]*)\))?\s+(.*)$")

    for ln in lines:
        parts = ln.split()
        if parts[0].lower() in ("qubits", "num_qubits", "n_qubits") and len(parts) >= 2:
            declared_n = int(parts[1])
            continue
        m = inst_re.match(ln)
        if m:
            q_toks = [t for t in m.group(3).replace(",", " ").split() if t.strip()]
            for t in q_toks:
                max_q = max(max_q, int(t.lstrip("qQ")))

    n_qubits = max(declared_n or len(project.qubits) or 1, max_q + 1, 1)

    # Preserve existing qubit specs where available; append new ones if n_qubits grew
    grid_cols = max(1, math.ceil(math.sqrt(n_qubits)))
    new_qubits: List[QubitSpec] = []
    for i in range(n_qubits):
        if i < len(project.qubits):
            q_copy = project.qubits[i].model_copy()
            q_copy.id = i
            new_qubits.append(q_copy)
        else:
            if n_qubits <= 8:
                px, py = round(i * 4.5, 1), 0.0
            else:
                r_idx = i // grid_cols
                c_idx = (i % grid_cols) if (r_idx % 2 == 0) else (grid_cols - 1 - (i % grid_cols))
                px, py = round(c_idx * 4.5, 1), round(r_idx * 4.5, 1)
            new_qubits.append(
                QubitSpec(
                    id=i,
                    name=f"q{i}",
                    color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
                    pos_x=px,
                    pos_y=py,
                )
            )

    noise_names = {
        "DEPOLARIZE", "DEPOL", "DEPOLARIZING", "DEPOLARIZE1", "DEPOLARIZE2",
        "AMPLITUDE_DAMPING", "AD", "T1_DECAY", "PHASE_DAMPING", "PD", "DEPHASING",
        "BIT_FLIP", "X_ERROR", "PHASE_FLIP", "Z_ERROR", "PAULI_ERROR", "PAULI1Q",
    }
    single_q_broadcast = {
        "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "RX", "RY", "RZ", "RXY", "U3",
    }

    qubit_col = [0] * n_qubits
    new_ops: List[CircuitOpModel] = []

    for idx, ln in enumerate(lines):
        parts = ln.split()
        cmd = parts[0].lower()
        if cmd in ("qubits", "num_qubits", "n_qubits"):
            continue
        if cmd in ("measure_all", "measureall", "m_all"):
            col = max(qubit_col) if qubit_col else 0
            new_ops.append(
                CircuitOpModel(
                    id=f"op_{idx}_{uuid.uuid4().hex[:6]}",
                    step=col,
                    kind="measure",
                    name="MEASURE",
                    qubits=list(range(n_qubits)),
                )
            )
            for q in range(n_qubits):
                qubit_col[q] = col + 1
            continue

        m = inst_re.match(ln)
        if not m:
            raise ValueError(f"第 {idx + 1} 行语法错误: '{ln}'")

        op_name = m.group(1).upper()
        param_str = m.group(2)
        q_tokens = [t for t in m.group(3).replace(",", " ").split() if t.strip()]
        qubits = [int(t.lstrip("qQ")) for t in q_tokens]
        if not qubits:
            raise ValueError(f"第 {idx + 1} 行缺少比特编号: '{ln}'")

        raw_params: List[Any] = []
        if param_str is not None and param_str.strip():
            for token in param_str.split(","):
                t_s = token.strip()
                if not t_s:
                    continue
                try:
                    raw_params.append(float(t_s))
                except ValueError:
                    raw_params.append(t_s)

        if op_name in ("M", "MEASURE", "MEAS"):
            col = max(qubit_col[q] for q in qubits)
            new_ops.append(
                CircuitOpModel(
                    id=f"op_{idx}_{uuid.uuid4().hex[:6]}",
                    step=col,
                    kind="measure",
                    name="MEASURE",
                    qubits=qubits,
                    params=raw_params,
                )
            )
            for q in qubits:
                qubit_col[q] = col + 1
            continue

        if op_name in noise_names:
            span = range(min(qubits), max(qubits) + 1)
            col = max(qubit_col[q] for q in span)
            new_ops.append(
                CircuitOpModel(
                    id=f"op_{idx}_{uuid.uuid4().hex[:6]}",
                    step=col,
                    kind="noise",
                    name=op_name,
                    qubits=qubits,
                    params=raw_params,
                )
            )
            for q in span:
                qubit_col[q] = col + 1
            continue

        if len(qubits) > 1 and op_name in single_q_broadcast:
            for q in qubits:
                col = qubit_col[q]
                new_ops.append(
                    CircuitOpModel(
                        id=f"op_{idx}_{q}_{uuid.uuid4().hex[:5]}",
                        step=col,
                        kind="gate",
                        name=op_name,
                        qubits=[q],
                        params=list(raw_params),
                    )
                )
                qubit_col[q] = col + 1
        else:
            span = range(min(qubits), max(qubits) + 1)
            col = max(qubit_col[q] for q in span)
            new_ops.append(
                CircuitOpModel(
                    id=f"op_{idx}_{uuid.uuid4().hex[:6]}",
                    step=col,
                    kind="gate",
                    name=op_name,
                    qubits=qubits,
                    params=raw_params,
                )
            )
            for q in span:
                qubit_col[q] = col + 1

    # Sync symbolic variables
    sym_names = extract_symbolic_parameters(dsl)
    new_vars = dict(project.variables)
    for sn in sym_names:
        if sn not in new_vars:
            new_vars[sn] = 0.015 if sn.startswith("p_") else round(math.pi / 4.0, 4)

    return new_qubits, new_ops, new_vars


def _build_register_qubits(n_qubits: int, project: ProjectModel) -> List[QubitSpec]:
    """Preserve existing qubit specs up to `n_qubits` and lay out any newly added qubits cleanly."""
    grid_cols = max(1, math.ceil(math.sqrt(n_qubits)))
    new_qubits: List[QubitSpec] = []
    for i in range(n_qubits):
        if i < len(project.qubits):
            q_copy = project.qubits[i].model_copy()
            q_copy.id = i
            new_qubits.append(q_copy)
        else:
            if n_qubits <= 8:
                px, py = round(i * 4.5, 1), 0.0
            else:
                r_idx = i // grid_cols
                c_idx = (i % grid_cols) if (r_idx % 2 == 0) else (grid_cols - 1 - (i % grid_cols))
                px, py = round(c_idx * 4.5, 1), round(r_idx * 4.5, 1)
            new_qubits.append(
                QubitSpec(
                    id=i,
                    name=f"q{i}",
                    color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
                    pos_x=px,
                    pos_y=py,
                )
            )
    return new_qubits


def project_to_openqasm3(project: ProjectModel) -> str:
    """Export the project circuit as standard, round-trippable OpenQASM 3.0 text."""
    n = len(project.qubits)
    ordered = [
        op for _, op in sorted(enumerate(project.operations), key=lambda pair: (pair[1].step, pair[0]))
    ]
    used_gate_names = {op.name.upper() for op in ordered if op.kind == "gate"}

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
    ]
    # Emit standard helper gate definitions so external QASM 3 tools (Qiskit/Braket) can run native gates too
    if "RXY" in used_gate_names:
        lines.append("gate rxy(theta, phi) a { rz(-phi) a; rx(theta) a; rz(phi) a; }")
    if "ISWAP" in used_gate_names:
        lines.append("gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }")
    if "SQISWAP" in used_gate_names:
        lines.append("gate sqiswap a, b { rx(pi/2) a; cx a, b; rx(-pi/4) a; rz(pi/4) b; cx a, b; rx(-pi/4) a; }")

    for vname, vval in project.variables.items():
        if re.match(r"^[A-Za-z_]\w*$", vname):
            lines.append(f"float[64] {vname} = {float(vval):.6g};")

    lines.append(f"qubit[{n}] q;")
    lines.append(f"bit[{n}] c;")
    lines.append("")

    vars_map = project.variables
    qasm_map = {
        "H": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "SDG": "sdg",
        "T": "t", "TDG": "tdg", "SX": "sx", "SXDG": "sxdg",
        "RX": "rx", "RY": "ry", "RZ": "rz", "P": "p", "U3": "u", "U": "u",
        "RXY": "rxy", "RXX": "rxx", "RYY": "ryy", "RZZ": "rzz", "MS": "rxx",
        "CNOT": "cx", "CX": "cx", "CY": "cy", "CZ": "cz",
        "SWAP": "swap", "ISWAP": "iswap", "SQISWAP": "sqiswap",
        "CCX": "ccx", "TOFFOLI": "ccx", "CCZ": "ccz",
    }

    def _fmt_param(p: Any) -> str:
        if isinstance(p, (int, float)):
            return f"{float(p):.6g}"
        s = str(p).strip()
        if s in vars_map and re.match(r"^[A-Za-z_]\w*$", s):
            return s
        return f"{eval_math_expr(s, vars_map):.6g}"

    for op in ordered:
        if op.kind == "measure":
            for q in op.qubits:
                lines.append(f"c[{q}] = measure q[{q}];")
        elif op.kind == "gate":
            uname = op.name.upper()
            q_args = ", ".join(f"q[{q}]" for q in op.qubits)
            g_id = qasm_map.get(uname, uname.lower())
            if uname == "MS" and not op.params:
                lines.append(f"rxx(1.570796) {q_args};")
            elif op.params:
                p_str = ", ".join(_fmt_param(p) for p in op.params)
                lines.append(f"{g_id}({p_str}) {q_args};")
            else:
                lines.append(f"{g_id} {q_args};")
    return "\n".join(lines)


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def openqasm_to_project_ops(
    qasm: str,
    project: ProjectModel,
) -> Tuple[List[QubitSpec], List[CircuitOpModel], Dict[str, float]]:
    """
    Parse OpenQASM 2.0 or OpenQASM 3.0 source code into Quantum Studio's visual
    (QubitSpec list, ASAP-scheduled CircuitOpModel list, symbolic variables dict).
    """
    # 1. Strip /* ... */ block comments and // or # line comments
    cleaned = re.sub(r"/\*.*?\*/", "", qasm, flags=re.DOTALL)
    clean_lines: List[str] = []
    for raw_ln in cleaned.splitlines():
        ln = raw_ln.split("//", 1)[0].split("#", 1)[0].strip()
        if ln:
            clean_lines.append(ln)
    src = "\n".join(clean_lines)

    # 2. Extract `gate name[(params)] qargs { body }` macro blocks and remaining top-level text
    macros: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    top_chunks: List[str] = []
    pos = 0
    gate_hdr_re = re.compile(r"\bgate\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*([^{]+)\{", re.MULTILINE)
    while pos < len(src):
        m = gate_hdr_re.search(src, pos)
        if not m:
            top_chunks.append(src[pos:])
            break
        top_chunks.append(src[pos : m.start()])
        g_name = m.group(1).strip()
        p_names = [x.strip() for x in (m.group(2) or "").split(",") if x.strip()]
        q_names = [x.strip() for x in m.group(3).split(",") if x.strip()]
        brace_depth = 1
        cur_idx = m.end()
        while cur_idx < len(src) and brace_depth > 0:
            if src[cur_idx] == "{":
                brace_depth += 1
            elif src[cur_idx] == "}":
                brace_depth -= 1
            cur_idx += 1
        body_str = src[m.end() : max(m.end(), cur_idx - 1)]
        body_stmts = [st.strip() for st in body_str.split(";") if st.strip()]
        macros[g_name.lower()] = (p_names, q_names, body_stmts)
        pos = cur_idx

    statements = [st.strip() for st in " ".join(top_chunks).split(";") if st.strip()]

    reg_map: Dict[str, List[int]] = {}
    next_q = 0
    new_vars: Dict[str, float] = dict(project.variables)
    custom_gate_names = {cg.name.upper(): cg.name for cg in project.custom_gates}

    native_qasm_map: Dict[str, str] = {
        "h": "H", "x": "X", "y": "Y", "z": "Z",
        "s": "S", "sdg": "SDG", "t": "T", "tdg": "TDG",
        "sx": "SX", "sxdg": "SXDG",
        "rx": "RX", "ry": "RY", "rz": "RZ", "p": "RZ", "u1": "RZ",
        "rxy": "RXY", "rxx": "RXX", "ryy": "RYY", "rzz": "RZZ", "ms": "MS",
        "u": "U3", "u3": "U3",
        "cx": "CNOT", "cnot": "CNOT", "CX": "CNOT",
        "cy": "CY", "cz": "CZ", "swap": "SWAP", "iswap": "ISWAP", "sqiswap": "SQISWAP",
        "ecr": "ECR", "ccx": "CCX", "toffoli": "CCX", "ccz": "CCZ",
    }
    single_q_gates = {
        "H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "RX", "RY", "RZ", "RXY", "U3",
    }
    math_reserved = {
        "pi", "e", "tau", "sin", "cos", "tan", "sqrt", "asin", "acos", "atan", "exp",
    }

    # Pass 1: Process register and variable declarations
    exec_stmts: List[str] = []
    for st in statements:
        low = st.lower()
        if low.startswith("openqasm") or low.startswith("include ") or low.startswith("barrier"):
            continue
        if low.startswith("creg ") or low.startswith("bit[") or re.match(r"^bit\s+\w+$", low):
            continue
        # OpenQASM 3: qubit[4] q; or qubit q0;
        m_q3 = re.match(r"^qubit\s*(?:\[\s*(\d+)\s*\])?\s+([A-Za-z_]\w*)$", st)
        if m_q3:
            sz = int(m_q3.group(1)) if m_q3.group(1) else 1
            rname = m_q3.group(2)
            reg_map[rname] = list(range(next_q, next_q + sz))
            next_q += sz
            continue
        # OpenQASM 2: qreg q[4];
        m_q2 = re.match(r"^qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$", st)
        if m_q2:
            rname = m_q2.group(1)
            sz = int(m_q2.group(2))
            reg_map[rname] = list(range(next_q, next_q + sz))
            next_q += sz
            continue
        # OpenQASM 3 scalar variable: [input] float[64] theta [= 1.57]; or angle phi = pi/2;
        m_var = re.match(
            r"^(?:input\s+)?(?:float|angle)(?:\s*\[\s*\d+\s*\])?\s+([A-Za-z_]\w*)(?:\s*=\s*(.+))?$",
            st,
        )
        if m_var:
            vname = m_var.group(1)
            vexpr = m_var.group(2)
            if vexpr is not None:
                new_vars[vname] = round(eval_math_expr(vexpr.strip(), new_vars), 6)
            elif vname not in new_vars:
                new_vars[vname] = round(math.pi / 4.0, 4)
            continue
        exec_stmts.append(st)

    def _resolve_qarg(tok: str, local_qmap: Optional[Dict[str, List[int]]] = None) -> List[int]:
        nonlocal next_q
        t = tok.strip()
        if local_qmap and t in local_qmap:
            return local_qmap[t]
        if t.startswith("$") and t[1:].isdigit():
            q_id = int(t[1:])
            next_q = max(next_q, q_id + 1)
            return [q_id]
        m_idx = re.match(r"^([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$", t)
        if m_idx:
            rname, idx = m_idx.group(1), int(m_idx.group(2))
            if rname not in reg_map:
                reg_map[rname] = list(range(next_q, next_q + idx + 1))
                next_q += idx + 1
            elif idx >= len(reg_map[rname]):
                extra = idx + 1 - len(reg_map[rname])
                reg_map[rname].extend(range(next_q, next_q + extra))
                next_q += extra
            return [reg_map[rname][idx]]
        if t in reg_map:
            return list(reg_map[t])
        raise ValueError(f"OpenQASM 未知量子比特参数: '{t}'")

    def _parse_param_expr(expr: str, local_pmap: Optional[Dict[str, Any]] = None) -> Union[float, str]:
        s = expr.strip()
        if local_pmap:
            for pk, pv in local_pmap.items():
                s = re.sub(rf"\b{re.escape(pk)}\b", f"({pv})", s)
        # Simplify trivial `(theta)` -> `theta`
        m_sym = re.match(r"^\(?\s*([A-Za-z_]\w*)\s*\)?$", s)
        if m_sym and m_sym.group(1).lower() not in math_reserved:
            vname = m_sym.group(1)
            if vname not in new_vars:
                new_vars[vname] = round(math.pi / 4.0, 4)
            return vname
        idents = [tok for tok in re.findall(r"[A-Za-z_]\w*", s) if tok.lower() not in math_reserved]
        if not idents:
            return round(eval_math_expr(s, {}), 6)
        for vname in idents:
            if vname not in new_vars:
                new_vars[vname] = round(math.pi / 4.0, 4)
        return s

    raw_emitted: List[Tuple[str, str, List[int], List[Any]]] = []  # (kind, name, qubits, params)

    def _process_stmt(
        st: str,
        local_pmap: Optional[Dict[str, Any]] = None,
        local_qmap: Optional[Dict[str, List[int]]] = None,
        depth: int = 0,
    ) -> None:
        if depth > 8:
            raise ValueError("OpenQASM 自定义 gate 嵌套过深或存在递归")
        s = st.strip()
        if not s or s.lower().startswith("barrier") or s.lower().startswith("reset"):
            return

        # Measurements:
        # 1) c[0] = measure q[0]; or c = measure q;
        m_m3 = re.match(r"^[A-Za-z_]\w*(?:\s*\[\s*\d+\s*\])?\s*=\s*measure\s+(.+)$", s, flags=re.IGNORECASE)
        if m_m3:
            qs = _resolve_qarg(m_m3.group(1), local_qmap)
            raw_emitted.append(("measure", "MEASURE", qs, []))
            return
        # 2) measure q[0] -> c[0]; or measure q -> c; or measure q[0];
        m_m2 = re.match(r"^measure\s+([^\-;]+?)(?:\s*->\s*.+)?$", s, flags=re.IGNORECASE)
        if m_m2:
            qs = _resolve_qarg(m_m2.group(1), local_qmap)
            raw_emitted.append(("measure", "MEASURE", qs, []))
            return

        # Gate call: gname[(params)] qarg1, qarg2...
        m_g = re.match(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s+(.+)$", s)
        if not m_g:
            raise ValueError(f"OpenQASM 语法无法解析: '{s}'")
        g_raw = m_g.group(1).strip()
        g_low = g_raw.lower()
        p_raw = m_g.group(2)
        q_raw = m_g.group(3)

        p_list = [_parse_param_expr(tok, local_pmap) for tok in _split_top_level_commas(p_raw)] if p_raw else []
        q_groups = [_resolve_qarg(qt, local_qmap) for qt in _split_top_level_commas(q_raw)]
        if not q_groups:
            raise ValueError(f"OpenQASM 门调用缺少量子比特: '{s}'")

        if g_low in ("id", "i", "nop"):
            return

        # Broadcast across register arguments if needed (e.g. `h q;` or `cx a, b;`)
        bcast_len = max(len(qg) for qg in q_groups)
        for qg in q_groups:
            if len(qg) not in (1, bcast_len):
                raise ValueError(f"OpenQASM 寄存器维度不匹配: '{s}'")

        for b_idx in range(bcast_len):
            qs = [qg[b_idx] if len(qg) > 1 else qg[0] for qg in q_groups]

            # Built-in / standard QASM gates take priority over exported compatibility stubs
            if g_low == "u2" and len(p_list) == 2:
                raw_emitted.append(("gate", "U3", qs, [round(math.pi / 2.0, 6), p_list[0], p_list[1]]))
            elif g_low in ("cp", "cu1", "crz") and len(qs) == 2 and len(p_list) == 1:
                lam = p_list[0]
                half = round(lam / 2.0, 6) if isinstance(lam, (int, float)) else f"({lam})/2"
                neg_half = round(-lam / 2.0, 6) if isinstance(lam, (int, float)) else f"-({lam})/2"
                if g_low in ("cp", "cu1"):
                    raw_emitted.append(("gate", "RZ", [qs[0]], [half]))
                raw_emitted.append(("gate", "RZ", [qs[1]], [half]))
                raw_emitted.append(("gate", "CNOT", [qs[0], qs[1]], []))
                raw_emitted.append(("gate", "RZ", [qs[1]], [neg_half]))
                raw_emitted.append(("gate", "CNOT", [qs[0], qs[1]], []))
            elif g_low == "ch" and len(qs) == 2:
                raw_emitted.append(("gate", "RY", [qs[1]], [round(math.pi / 4.0, 6)]))
                raw_emitted.append(("gate", "CNOT", [qs[0], qs[1]], []))
                raw_emitted.append(("gate", "RY", [qs[1]], [round(-math.pi / 4.0, 6)]))
            elif g_low in native_qasm_map:
                mapped = native_qasm_map[g_low]
                raw_emitted.append(("gate", mapped, qs, list(p_list)))
            elif g_raw.upper() in custom_gate_names:
                raw_emitted.append(("gate", custom_gate_names[g_raw.upper()], qs, list(p_list)))
            elif g_low in macros:
                m_pnames, m_qnames, m_body = macros[g_low]
                if len(m_qnames) != len(qs):
                    raise ValueError(f"自定义门 '{g_raw}' 需要 {len(m_qnames)} 个比特，实际传入 {len(qs)} 个")
                sub_pmap = {pn: pv for pn, pv in zip(m_pnames, p_list)}
                sub_qmap = {qn: [qv] for qn, qv in zip(m_qnames, qs)}
                for body_st in m_body:
                    _process_stmt(body_st, sub_pmap, sub_qmap, depth + 1)
            else:
                raise ValueError(f"不支持的 OpenQASM 量子门: '{g_raw}'")

    for st in exec_stmts:
        _process_stmt(st)

    n_qubits = max(next_q, len(project.qubits) if next_q == 0 else next_q, 1)
    new_qubits = _build_register_qubits(n_qubits, project)

    # Schedule onto visual columns (merging simultaneous single-qubit measurements cleanly)
    qubit_col = [0] * n_qubits
    new_ops: List[CircuitOpModel] = []
    pending_meas_step: Optional[int] = None
    pending_meas_op: Optional[CircuitOpModel] = None

    for idx, (kind, name, qs, params) in enumerate(raw_emitted):
        if kind == "measure":
            col = max(qubit_col[q] for q in qs)
            if (
                pending_meas_op is not None
                and pending_meas_step == col
                and set(qs).isdisjoint(pending_meas_op.qubits)
            ):
                pending_meas_op.qubits.extend(qs)
            else:
                # If all previously unmeasured qubits can align at the final measurement barrier, align them
                col = max(qubit_col) if len(qs) == 1 and idx >= len(raw_emitted) - n_qubits else col
                if (
                    pending_meas_op is not None
                    and pending_meas_step == col
                    and set(qs).isdisjoint(pending_meas_op.qubits)
                ):
                    pending_meas_op.qubits.extend(qs)
                else:
                    pending_meas_op = CircuitOpModel(
                        id=f"qasm_m_{idx}_{uuid.uuid4().hex[:5]}",
                        step=col,
                        kind="measure",
                        name="MEASURE",
                        qubits=list(qs),
                    )
                    pending_meas_step = col
                    new_ops.append(pending_meas_op)
            for q in qs:
                qubit_col[q] = col + 1
            continue

        pending_meas_op = None
        pending_meas_step = None
        if len(qs) == 1 and name in single_q_gates:
            q0 = qs[0]
            col = qubit_col[q0]
            new_ops.append(
                CircuitOpModel(
                    id=f"qasm_{idx}_{uuid.uuid4().hex[:5]}",
                    step=col,
                    kind="gate",
                    name=name,
                    qubits=[q0],
                    params=params,
                )
            )
            qubit_col[q0] = col + 1
        else:
            span = range(min(qs), max(qs) + 1)
            col = max(qubit_col[q] for q in span)
            new_ops.append(
                CircuitOpModel(
                    id=f"qasm_{idx}_{uuid.uuid4().hex[:5]}",
                    step=col,
                    kind="gate",
                    name=name,
                    qubits=list(qs),
                    params=params,
                )
            )
            for q in span:
                qubit_col[q] = col + 1

    return new_qubits, new_ops, new_vars


# ---------------------------------------------------------------------------
# Serialization Helpers & Simulation / Optimization / Sweep Runners
# ---------------------------------------------------------------------------

def _serialize_evolution_result(res: EvolutionResult) -> Dict[str, Any]:
    steps_json: List[Dict[str, Any]] = []
    for s in res.steps:
        rho_abs = None
        rho_re = None
        rho_im = None
        if s.density_matrix is not None and res.num_qubits <= 5:
            rho_abs = np.round(np.abs(s.density_matrix), 5).tolist()
            rho_re = np.round(np.real(s.density_matrix), 5).tolist()
            rho_im = np.round(np.imag(s.density_matrix), 5).tolist()

        pulse_traj = None
        if s.pulse_bloch_trajectories is not None:
            pulse_traj = {
                str(q): np.round(arr, 5).tolist()
                for q, arr in s.pulse_bloch_trajectories.items()
            }

        steps_json.append(
            {
                "step_index": s.step_index,
                "operation_label": s.operation_label,
                "elapsed_time_ns": round(float(s.elapsed_time_ns), 3),
                "state_fidelity": round(float(s.state_fidelity), 6) if s.state_fidelity is not None else None,
                "purity": round(float(s.purity), 6) if s.purity is not None else None,
                "bloch_vectors": {
                    str(q): [round(float(v), 5) for v in bv]
                    for q, bv in s.bloch_vectors.items()
                },
                "populations": {
                    k: round(float(v), 6) for k, v in s.populations.items()
                },
                "stabilizers": s.stabilizers,
                "pulse_times_ns": np.round(s.pulse_times_ns, 3).tolist() if s.pulse_times_ns is not None else None,
                "pulse_bloch_trajectories": pulse_traj,
                "density_matrix_abs": rho_abs,
                "density_matrix_real": rho_re,
                "density_matrix_imag": rho_im,
            }
        )

    mi_matrix: Optional[List[List[float]]] = None
    bip_entropy: Optional[float] = None
    if "mutual_information" in res.metadata:
        mi_matrix = res.metadata["mutual_information"]  # type: ignore[assignment]
        raw_bip = res.metadata.get("bipartite_entropy")
        bip_entropy = float(raw_bip) if isinstance(raw_bip, (int, float)) else None
    elif res.steps and res.steps[-1].density_matrix is not None and 2 <= res.num_qubits <= 11:
        mi_matrix, bip_entropy = compute_entanglement_from_density_matrix(
            res.steps[-1].density_matrix, res.num_qubits
        )

    return {
        "backend_name": res.backend_name,
        "num_qubits": res.num_qubits,
        "total_duration_ns": round(float(res.total_duration_ns), 3),
        "shots": res.shots,
        "final_fidelity": round(float(res.final_fidelity), 6) if res.final_fidelity is not None else None,
        "final_purity": round(float(res.final_purity), 6) if res.final_purity is not None else None,
        "bipartite_entropy": bip_entropy,
        "mutual_information": mi_matrix,
        "final_probabilities": {k: round(float(v), 6) for k, v in res.final_probabilities.items()},
        "measurement_counts": res.measurement_counts,
        "stim_circuit_str": res.stim_circuit_str,
        "steps": steps_json,
    }


def execute_project_simulation(project: ProjectModel) -> Dict[str, Any]:
    circ = compile_project_circuit(project)
    hw = build_hardware_model(project)
    init_labels = [q.init_state for q in project.qubits]
    sim = project.sim_settings

    solver = QuantumSolver(hardware=hw, intra_gate_steps=sim.intra_gate_steps)
    backend = sim.backend

    results_map: Dict[str, EvolutionResult] = {}
    warnings: List[str] = []

    if backend == "compare":
        init_psi = build_initial_statevector(project.qubits) if circ.num_qubits <= 22 else None
        if circ.num_qubits <= 7:
            results_map["qutip"] = solver.solve(
                circ, backend="qutip", hardware=hw, shots=sim.shots, initial_state=init_psi
            )
        if circ.num_qubits <= 12:
            results_map["kraus"] = solver.solve(
                circ, backend="kraus", hardware=hw, shots=sim.shots, initial_state=init_psi
            )
        if circ.num_qubits <= 22:
            results_map["trajectory"] = solver.solve(
                circ, backend="trajectory", hardware=hw, shots=sim.shots, initial_state=init_psi
            )
        if circ.num_qubits > 12:
            warnings.append(
                f"比特数 ({circ.num_qubits}Q) 超过稠密密度矩阵规模上限，已自动对比 Trajectory 态矢量与 Stim 稳定子引擎。"
            )
        try:
            results_map["stim"] = solver.solve(
                circ, backend="stim", hardware=hw, shots=sim.shots, initial_state_labels=init_labels
            )
            if circ.num_qubits <= 6:
                results_map["stim_qutip"] = solver.solve(
                    circ, backend="stim_qutip", hardware=hw, shots=sim.shots, initial_state_labels=init_labels
                )
        except Exception as exc:
            warnings.append(f"Stim 后端不支持部分自定义非 Clifford 门（已保留精确态演化结果）: {exc}")
        for pref in ("qutip", "kraus", "trajectory", "stim"):
            if pref in results_map:
                primary_key = pref
                break
        else:
            primary_key = next(iter(results_map.keys()))
    else:
        eff_backend = backend
        if eff_backend == "qutip" and circ.num_qubits > 9:
            if circ.num_qubits <= 13:
                eff_backend = "kraus"
            elif circ.is_clifford() or circ.num_qubits > 24:
                eff_backend = "stim"
            else:
                eff_backend = "trajectory"
            warnings.append(
                f"{circ.num_qubits} 比特超出 QuTiP 连续脉冲主方程内存上限，已自动切换至「{eff_backend}」高性能引擎完成仿真。"
            )
        elif eff_backend == "kraus" and circ.num_qubits > 14:
            eff_backend = "stim" if (circ.is_clifford() or circ.num_qubits > 24) else "trajectory"
            warnings.append(
                f"{circ.num_qubits} 比特超出 Kraus 稠密密度矩阵上限 (14Q)，已自动切换至「{eff_backend}」引擎完成仿真。"
            )
        elif eff_backend == "trajectory" and circ.num_qubits > 24:
            eff_backend = "stim"
            warnings.append(
                f"{circ.num_qubits} 比特超出 Trajectory 态矢量内存上限 (24Q)，已自动切换至「stim」稳定子引擎完成仿真。"
            )

        init_psi = (
            build_initial_statevector(project.qubits)
            if eff_backend in ("qutip", "kraus", "trajectory") and circ.num_qubits <= 24
            else None
        )
        res = solver.solve(
            circ,
            backend=eff_backend,
            hardware=hw,
            shots=sim.shots,
            initial_state=init_psi,
            initial_state_labels=init_labels if eff_backend in ("stim", "stim_qutip") else None,
        )
        results_map[eff_backend] = res
        primary_key = eff_backend

    all_edges = compute_topology_edges(
        project.qubits,
        get_active_hardware_profile(project),
        operations=project.operations,
    )
    return {
        "primary_backend": primary_key,
        "warnings": warnings,
        "hardware_name": hw.name,
        "topology_edges": all_edges[:300],
        "results": {k: _serialize_evolution_result(v) for k, v in results_map.items()},
    }


def _circuit_to_op_models(circuit: QuantumCircuit) -> List[CircuitOpModel]:
    """Convert a compiled QuantumCircuit back to ASAP-scheduled visual CircuitOpModel items."""
    n = circuit.num_qubits
    qubit_col = [0] * n
    ops: List[CircuitOpModel] = []
    for idx, op in enumerate(circuit.operations):
        if isinstance(op, GateInstruction):
            span = range(min(op.qubits), max(op.qubits) + 1)
            col = max(qubit_col[q] for q in span)
            ops.append(
                CircuitOpModel(
                    id=f"opt_{idx}_{uuid.uuid4().hex[:5]}",
                    step=col,
                    kind="gate",
                    name=op.name,
                    qubits=list(op.qubits),
                    params=[round(float(p), 5) for p in op.params],
                    duration_ns=op.duration_ns,
                )
            )
            for q in span:
                qubit_col[q] = col + 1
        elif isinstance(op, MeasureInstruction):
            col = max(qubit_col[q] for q in op.qubits)
            ops.append(
                CircuitOpModel(
                    id=f"opt_m_{idx}_{uuid.uuid4().hex[:5]}",
                    step=col,
                    kind="measure",
                    name="MEASURE",
                    qubits=list(op.qubits),
                )
            )
            for q in op.qubits:
                qubit_col[q] = col + 1
    return ops


def execute_project_optimization(req: OptimizeRequest) -> Dict[str, Any]:
    project = req.project
    circ = compile_project_circuit(project)
    hw = build_hardware_model(project, preferred_native_2q=req.preferred_native_2q)
    optimizer = QuantumCircuitOptimizer(hardware=hw)

    can_eval_noisy = circ.num_qubits <= 20 or (circ.num_qubits <= 120 and circ.is_clifford())
    report = optimizer.optimize_circuit(
        circ,
        hardware=hw,
        evaluate_noisy_fidelity=(req.evaluate_noisy_fidelity and can_eval_noisy),
    )

    opt_ops = _circuit_to_op_models(report.optimized_circuit)
    # Build DSL text for optimized circuit
    tmp_proj = project.model_copy(deep=True)
    tmp_proj.operations = opt_ops
    opt_dsl = project_to_dsl(tmp_proj)
    opt_qasm = project_to_openqasm3(tmp_proj)

    def _m_dict(m: Any) -> Optional[Dict[str, Any]]:
        if m is None:
            return None
        return {
            "total_gates": m.total_gates,
            "num_2q_gates": m.num_2q_gates,
            "num_1q_physical_pulses": m.num_1q_physical_pulses,
            "num_1q_virtual_gates": m.num_1q_virtual_gates,
            "depth": m.depth,
            "duration_ns": round(float(m.duration_ns), 2),
            "expected_gate_fidelity": round(float(m.expected_gate_fidelity), 6),
            "gate_breakdown": m.gate_breakdown,
        }

    return {
        "gate_set_name": report.gate_set_name,
        "primary_2q_gate": hw.native_gate_set.primary_2q_gate,
        "unitary_fidelity": round(float(report.unitary_fidelity), 10),
        "noisy_fidelity_original": round(float(report.noisy_fidelity_original), 6)
        if report.noisy_fidelity_original is not None
        else None,
        "noisy_fidelity_optimized": round(float(report.noisy_fidelity_optimized), 6)
        if report.noisy_fidelity_optimized is not None
        else None,
        "original_metrics": _m_dict(report.original_metrics),
        "optimized_metrics": _m_dict(report.optimized_metrics),
        "optimized_operations": [op.model_dump() for op in opt_ops],
        "optimized_dsl": opt_dsl,
        "optimized_qasm": opt_qasm,
        "summary_text": report.summary(),
    }


def execute_unitary_synthesis(req: SynthesizeUnitaryRequest) -> Dict[str, Any]:
    """Synthesize an arbitrary 1Q/2Q target unitary matrix or Pauli Hamiltonian into optimal native gates."""
    n_q = max(1, min(2, int(req.num_qubits)))
    dim = 1 << n_q
    if req.mode == "hamiltonian":
        if not req.hamiltonian_str or not req.hamiltonian_str.strip():
            raise ValueError("请输入泡利字符串哈密顿量，例如 '0.5*XX + 0.5*YY'")
        H = parse_pauli_hamiltonian(req.hamiltonian_str, n_q)
        U_target = expm(-1.0j * float(req.hamiltonian_scale) * H)
    else:
        if not req.matrix_exprs or len(req.matrix_exprs) != dim:
            raise ValueError(f"目标矩阵维度必须为 {dim}×{dim}")
        U_target = np.zeros((dim, dim), dtype=np.complex128)
        for r in range(dim):
            if len(req.matrix_exprs[r]) != dim:
                raise ValueError(f"第 {r + 1} 行长度必须为 {dim}")
            for c in range(dim):
                U_target[r, c] = _safe_complex_eval(req.matrix_exprs[r][c], req.project.variables)

    is_u, err, _ = validate_unitary_matrix(U_target)
    if not is_u:
        raise ValueError(f"目标矩阵不是有效酉矩阵 (‖U†U - I‖_F = {err:.4g})")

    # Build a sub-project with at least n_q qubits so hardware model matches target size
    synth_proj = req.project.model_copy(deep=True)
    if len(synth_proj.qubits) < n_q:
        for i in range(len(synth_proj.qubits), n_q):
            synth_proj.qubits.append(
                QubitSpec(id=i, name=f"q{i}", color=QUBIT_COLORS[i % len(QUBIT_COLORS)], pos_x=i * 4.5, pos_y=0.0)
            )
    else:
        synth_proj.qubits = synth_proj.qubits[:n_q]
        for idx, q in enumerate(synth_proj.qubits):
            q.id = idx

    hw = build_hardware_model(synth_proj, preferred_native_2q=req.preferred_native_2q)
    optimizer = QuantumCircuitOptimizer(hardware=hw)
    report = optimizer.synthesize_unitary(
        U_target,
        hardware=hw,
        evaluate_noisy_fidelity=req.evaluate_noisy_fidelity,
    )

    opt_ops = _circuit_to_op_models(report.optimized_circuit)
    # Add measurements on synthesized qubits so applying to Design is immediately simulatable
    max_step = max((op.step for op in opt_ops), default=-1)
    opt_ops.append(
        CircuitOpModel(
            id=f"synth_m_{uuid.uuid4().hex[:5]}",
            step=max_step + 1,
            kind="measure",
            name="MEASURE",
            qubits=list(range(n_q)),
        )
    )
    tmp_proj = synth_proj.model_copy(deep=True)
    tmp_proj.operations = opt_ops
    opt_dsl = project_to_dsl(tmp_proj)
    opt_qasm = project_to_openqasm3(tmp_proj)

    m = report.optimized_metrics
    opt_metrics_dict = {
        "total_gates": m.total_gates,
        "num_2q_gates": m.num_2q_gates,
        "num_1q_physical_pulses": m.num_1q_physical_pulses,
        "num_1q_virtual_gates": m.num_1q_virtual_gates,
        "depth": m.depth,
        "duration_ns": round(float(m.duration_ns), 2),
        "expected_gate_fidelity": round(float(m.expected_gate_fidelity), 6),
        "gate_breakdown": m.gate_breakdown,
    }

    return {
        "gate_set_name": report.gate_set_name,
        "primary_2q_gate": hw.native_gate_set.primary_2q_gate,
        "unitary_fidelity": round(float(report.unitary_fidelity), 10),
        "noisy_fidelity_original": None,
        "noisy_fidelity_optimized": round(float(report.noisy_fidelity_optimized), 6)
        if report.noisy_fidelity_optimized is not None
        else None,
        "original_metrics": None,
        "optimized_metrics": opt_metrics_dict,
        "optimized_operations": [op.model_dump() for op in opt_ops],
        "optimized_dsl": opt_dsl,
        "optimized_qasm": opt_qasm,
        "summary_text": report.summary(),
        "target_num_qubits": n_q,
    }


def parse_sparse_pauli_terms(
    expr: str,
    num_qubits: int,
) -> List[Tuple[float, Tuple[Tuple[int, str], ...]]]:
    """
    Parse a Hermitian Pauli Hamiltonian string into sparse `(coeff, ((q_idx, 'X'|'Y'|'Z'), ...))` terms.
    Supports both:
      - Indexed sparse form (for arbitrary N): `-1.0 * Z0 Z1 - 0.5 * X0 - 0.5 * X1`, `0.5*X0*X1 + 0.5*Y0*Y1 + 1.5*I`
      - Dense N-qubit Pauli strings: `-1.0*ZZ - 0.5*XI - 0.5*IX`
    """
    s = expr.strip()
    if not s:
        return []

    # Split top-level additive terms while respecting parentheses (e.g. `(1/2)*Z0*Z1`)
    chunks: List[str] = []
    depth = 0
    cur: List[str] = []
    for idx, ch in enumerate(s):
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch in "+-" and depth == 0 and idx > 0:
            prev_non_space = next((s[j] for j in range(idx - 1, -1, -1) if not s[j].isspace()), "")
            if prev_non_space not in ("*", "/", "(", "e", "E", "+", "-"):
                part = "".join(cur).strip()
                if part:
                    chunks.append(part)
                cur = [ch]
                continue
            cur.append(ch)
        else:
            cur.append(ch)
    if cur:
        part = "".join(cur).strip()
        if part:
            chunks.append(part)

    merged: Dict[Tuple[Tuple[int, str], ...], float] = {}
    indexed_re = re.compile(r"\b([IXYZixyz])_?(\d+)\b")

    for raw_term in chunks:
        t = raw_term.strip()
        if not t:
            continue
        sign = 1.0
        while t and t[0] in "+-":
            if t[0] == "-":
                sign *= -1.0
            t = t[1:].strip()
        if not t:
            continue

        # Check if term uses indexed Pauli tokens like Z0, X_1, Y12
        idx_matches = list(indexed_re.finditer(t))
        if idx_matches:
            first_pos = idx_matches[0].start()
            coeff_part = t[:first_pos].strip().rstrip("*").strip()
            coeff = sign * (eval_math_expr(coeff_part, {}) if coeff_part else 1.0)
            # Verify remainder contains only indexed Paulis, whitespace, and '*'
            rem = indexed_re.sub("", t[first_pos:])
            if rem.replace("*", "").strip():
                raise ValueError(f"无法解析泡利项 '{raw_term}' 中的多余符号 '{rem.strip()}'")
            q_paulis: Dict[int, str] = {}
            phase_sign = 1.0
            for m in idx_matches:
                p_ch = m.group(1).upper()
                q_id = int(m.group(2))
                if q_id < 0 or q_id >= num_qubits:
                    raise ValueError(f"泡利算符 '{m.group(0)}' 超出当前比特范围 (0 ~ {num_qubits - 1})")
                if p_ch == "I":
                    continue
                if q_id not in q_paulis:
                    q_paulis[q_id] = p_ch
                elif q_paulis[q_id] == p_ch:
                    del q_paulis[q_id]  # P * P = I
                else:
                    raise ValueError(f"同一项中比特 q{q_id} 重复出现非对易泡利算符 '{q_paulis[q_id]}' 与 '{p_ch}'")
            key = tuple(sorted(q_paulis.items()))
            merged[key] = merged.get(key, 0.0) + phase_sign * coeff
            continue

        # Check dense Pauli word (e.g. `0.5 * XX`, `-ZZ`, `I`, or pure constant `1.25`)
        m_dense = re.match(r"^(?:(.+?)\s*\*\s*)?([IXYZixyz]+)$", t)
        if m_dense:
            coeff_part = (m_dense.group(1) or "").strip()
            word = m_dense.group(2).upper()
            coeff = sign * (eval_math_expr(coeff_part, {}) if coeff_part else 1.0)
            if set(word) == {"I"} and (len(word) == 1 or len(word) == num_qubits):
                merged[()] = merged.get((), 0.0) + coeff
                continue
            if len(word) != num_qubits:
                raise ValueError(
                    f"稠密泡利串 '{word}' 长度为 {len(word)}，需等于寄存器比特数 {num_qubits}，或改用下标写法如 'Z0 Z1'"
                )
            key = tuple((q_idx, ch) for q_idx, ch in enumerate(word) if ch != "I")
            merged[key] = merged.get(key, 0.0) + coeff
            continue

        # Pure scalar constant offset (e.g. `+ 1.5`)
        coeff = sign * eval_math_expr(t, {})
        merged[()] = merged.get((), 0.0) + coeff

    return [(float(c), ops) for ops, c in merged.items() if abs(c) > 1e-15]


def evaluate_result_hamiltonian(
    res: EvolutionResult,
    pauli_terms: Sequence[Tuple[float, Tuple[Tuple[int, str], ...]]],
    num_qubits: int,
) -> float:
    """Evaluate <H> across QuTiP, Kraus, Trajectory (exact/MC), and Stim backends."""
    if not pauli_terms:
        return 0.0
    if "observable_expectation" in res.metadata:
        return float(res.metadata["observable_expectation"])  # type: ignore[arg-type]

    last_step = res.steps[-1]
    if last_step.density_matrix is not None:
        return evaluate_pauli_sum_density_matrix(last_step.density_matrix, pauli_terms, num_qubits)
    if last_step.ideal_statevector is not None and (res.final_purity or 0.0) > 0.999999:
        return evaluate_pauli_sum_statevector(last_step.ideal_statevector, pauli_terms, num_qubits)

    # Fallback for large-N Stim stabilizer / sampling results:
    # 1-body terms use exact reduced Bloch vectors; multi-qubit Z terms use empirical/final bitstrings.
    emp_probs = res.empirical_probabilities()
    total = 0.0
    for coeff, ops in pauli_terms:
        if not ops:
            total += coeff
        elif len(ops) == 1:
            q_id, p_ch = ops[0]
            bx, by, bz = last_step.bloch_vectors.get(q_id, (0.0, 0.0, 0.0))
            val = bx if p_ch == "X" else (by if p_ch == "Y" else bz)
            total += coeff * float(val)
        elif all(p_ch == "Z" for _, p_ch in ops) and emp_probs:
            z_exp = 0.0
            q_indices = [q_id for q_id, _ in ops]
            for bstr, prob in emp_probs.items():
                if len(bstr) > max(q_indices):
                    parity = 1.0
                    for qi in q_indices:
                        if bstr[qi] == "1":
                            parity = -parity
                    z_exp += prob * parity
            total += coeff * z_exp
    return float(total)


def execute_project_sweep(req: SweepRequest) -> Dict[str, Any]:
    project = req.project
    base_hw = build_hardware_model(project)
    init_labels = [q.init_state for q in project.qubits]
    n_q = max(1, len(project.qubits))
    eff_backend = req.backend
    if eff_backend == "qutip" and n_q > 9:
        eff_backend = "kraus" if n_q <= 13 else ("trajectory" if n_q <= 24 else "stim")
    elif eff_backend == "kraus" and n_q > 14:
        eff_backend = "trajectory" if n_q <= 24 else "stim"
    elif eff_backend == "trajectory" and n_q > 24:
        eff_backend = "stim"

    ham_clean = (req.hamiltonian_str or "").strip()
    parsed_ham = parse_sparse_pauli_terms(ham_clean, n_q) if ham_clean else []

    init_psi = (
        build_initial_statevector(project.qubits)
        if eff_backend in ("qutip", "kraus", "trajectory") and n_q <= 24
        else None
    )
    obs_q = max(0, min(n_q - 1, int(req.observed_qubit)))
    obs_q_name = project.qubits[obs_q].name if obs_q < len(project.qubits) else f"q{obs_q}"
    shots = project.sim_settings.shots
    max_workers = max(1, min(6, (os.cpu_count() or 4)))

    if req.mode == "1d":
        n_pts = max(2, min(150, int(req.x_points)))
        x_vals = np.linspace(float(req.x_start), float(req.x_stop), n_pts)

        def _eval_1d_point(
            args: Tuple[int, float],
        ) -> Tuple[Dict[str, float], float, float, List[float], Optional[float]]:
            idx, xv = args
            curr_vars = dict(project.variables)
            curr_hw = base_hw
            if req.param_x.startswith("hw:"):
                curr_hw = apply_hardware_param(base_hw, req.param_x, float(xv))
            else:
                curr_vars[req.param_x] = float(xv)
            circ = compile_project_circuit(project, variables_override=curr_vars)
            local_solver = QuantumSolver(hardware=curr_hw, intra_gate_steps=8)
            res = local_solver.solve(
                circ,
                backend=eff_backend,
                hardware=curr_hw,
                shots=shots,
                seed=42 + idx,
                initial_state=init_psi,
                initial_state_labels=init_labels if eff_backend in ("stim", "stim_qutip") else None,
                observable_terms=parsed_ham if parsed_ham else None,
            )
            fid = round(float(res.final_fidelity if res.final_fidelity is not None else 1.0), 6)
            pur = round(float(res.final_purity if res.final_purity is not None else 1.0), 6)
            bv = res.steps[-1].bloch_vectors.get(obs_q, (0.0, 0.0, 0.0))
            en = round(evaluate_result_hamiltonian(res, parsed_ham, n_q), 6) if parsed_ham else None
            return (
                res.final_probabilities,
                fid,
                pur,
                [round(float(bv[0]), 5), round(float(bv[1]), 5), round(float(bv[2]), 5)],
                en,
            )

        tasks_1d = [(idx, float(xv)) for idx, xv in enumerate(x_vals)]
        if max_workers > 1 and n_pts >= 6 and n_q <= 16:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                rows_1d = list(pool.map(_eval_1d_point, tasks_1d))
        else:
            rows_1d = [_eval_1d_point(t) for t in tasks_1d]

        if n_q <= 4:
            tracked_bstrs = [format(i, f"0{n_q}b") for i in range(1 << n_q)]
        else:
            max_by_bstr: Dict[str, float] = {"0" * n_q: 0.0, "1" * n_q: 0.0}
            for raw_pmap, _, _, _, _ in rows_1d:
                for k, v in raw_pmap.items():
                    if v > max_by_bstr.get(k, 0.0):
                        max_by_bstr[k] = float(v)
            ranked = sorted(max_by_bstr.keys(), key=lambda k: max_by_bstr[k], reverse=True)
            tracked_bstrs = ranked[:16]

        pops: Dict[str, List[float]] = {b: [] for b in tracked_bstrs}
        fids: List[float] = []
        purs: List[float] = []
        bloch_obs: List[List[float]] = []
        energies: List[float] = []
        for raw_pmap, fid, pur, bv_list, en in rows_1d:
            for b in tracked_bstrs:
                pops[b].append(round(float(raw_pmap.get(b, 0.0)), 6))
            fids.append(fid)
            purs.append(pur)
            bloch_obs.append(bv_list)
            if en is not None:
                energies.append(en)

        return {
            "mode": "1d",
            "param_x": req.param_x,
            "param_x_label": HARDWARE_SWEEP_PARAMS.get(req.param_x, req.param_x),
            "backend": eff_backend,
            "observed_qubit": obs_q,
            "observed_qubit_name": obs_q_name,
            "hamiltonian_str": ham_clean if parsed_ham else None,
            "x_values": np.round(x_vals, 5).tolist(),
            "populations": pops,
            "fidelities": fids,
            "purities": purs,
            "bloch_q0": bloch_obs,
            "energies": energies if parsed_ham else None,
        }

    # 2D Sweep (Parallelized across grid points)
    nx = max(2, min(40, int(req.x_points)))
    ny = max(2, min(40, int(req.y_points)))
    x_vals = np.linspace(float(req.x_start), float(req.x_stop), nx)
    y_vals = np.linspace(float(req.y_start), float(req.y_stop), ny)
    param_y = req.param_y or "hw:depol_2q"
    target_bstr = (
        req.target_bitstring
        if (req.target_bitstring and len(req.target_bitstring) == n_q and set(req.target_bitstring) <= {"0", "1"})
        else ("0" * n_q)
    )

    def _eval_2d_point(
        args: Tuple[int, int, float, float],
    ) -> Tuple[int, int, float, float, float, float]:
        iy, ix, xv, yv = args
        curr_vars = dict(project.variables)
        curr_hw = base_hw
        for pname, pval in ((req.param_x, xv), (param_y, yv)):
            if pname.startswith("hw:"):
                curr_hw = apply_hardware_param(curr_hw, pname, float(pval))
            else:
                curr_vars[pname] = float(pval)
        circ = compile_project_circuit(project, variables_override=curr_vars)
        local_solver = QuantumSolver(hardware=curr_hw, intra_gate_steps=8)
        res = local_solver.solve(
            circ,
            backend=eff_backend,
            hardware=curr_hw,
            shots=max(1024, shots // 4),
            seed=100 + iy * nx + ix,
            initial_state=init_psi,
            initial_state_labels=init_labels if eff_backend in ("stim", "stim_qutip") else None,
            observable_terms=parsed_ham if parsed_ham else None,
        )
        p_val = float(res.final_probabilities.get(target_bstr, 0.0))
        f_val = float(res.final_fidelity if res.final_fidelity is not None else 1.0)
        z_val = float(res.steps[-1].bloch_vectors.get(obs_q, (0.0, 0.0, 0.0))[2])
        e_val = evaluate_result_hamiltonian(res, parsed_ham, n_q) if parsed_ham else 0.0
        return iy, ix, p_val, f_val, z_val, e_val

    tasks_2d = [
        (iy, ix, float(xv), float(yv))
        for iy, yv in enumerate(y_vals)
        for ix, xv in enumerate(x_vals)
    ]
    if max_workers > 1 and len(tasks_2d) >= 8 and n_q <= 16:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows_2d = list(pool.map(_eval_2d_point, tasks_2d))
    else:
        rows_2d = [_eval_2d_point(t) for t in tasks_2d]

    pop_grid = np.zeros((ny, nx), dtype=np.float64)
    fid_grid = np.zeros((ny, nx), dtype=np.float64)
    z_grid = np.zeros((ny, nx), dtype=np.float64)
    en_grid = np.zeros((ny, nx), dtype=np.float64)
    for iy, ix, p_val, f_val, z_val, e_val in rows_2d:
        pop_grid[iy, ix] = p_val
        fid_grid[iy, ix] = f_val
        z_grid[iy, ix] = z_val
        en_grid[iy, ix] = e_val

    return {
        "mode": "2d",
        "param_x": req.param_x,
        "param_x_label": HARDWARE_SWEEP_PARAMS.get(req.param_x, req.param_x),
        "param_y": param_y,
        "param_y_label": HARDWARE_SWEEP_PARAMS.get(param_y, param_y),
        "backend": eff_backend,
        "observed_qubit": obs_q,
        "observed_qubit_name": obs_q_name,
        "target_bitstring": target_bstr,
        "hamiltonian_str": ham_clean if parsed_ham else None,
        "x_values": np.round(x_vals, 5).tolist(),
        "y_values": np.round(y_vals, 5).tolist(),
        "population_grid": np.round(pop_grid, 5).tolist(),
        "fidelity_grid": np.round(fid_grid, 5).tolist(),
        "z0_grid": np.round(z_grid, 5).tolist(),
        "energy_grid": np.round(en_grid, 5).tolist() if parsed_ham else None,
    }


def execute_cross_hardware_benchmark(req: BenchmarkHardwareRequest) -> Dict[str, Any]:
    """
    Compile and benchmark the active project circuit across all configured hardware profiles
    (Neutral Atom Rydberg, Superconducting CZ, Superconducting iSWAP, Trapped Ion MS, etc.).
    """
    project = req.project
    circ = compile_project_circuit(project)
    profiles = project.hardware_profiles or _default_hardware_profiles()

    def _bench_one(hp: HardwareProfileModel) -> Dict[str, Any]:
        sub_proj = project.model_copy(deep=True)
        sub_proj.active_hardware_id = hp.id
        hw = build_hardware_model(sub_proj)
        opt = QuantumCircuitOptimizer(hardware=hw)
        rep = opt.optimize_circuit(circ, hardware=hw, evaluate_noisy_fidelity=False)
        m = rep.optimized_metrics
        opt_ops = _circuit_to_op_models(rep.optimized_circuit)
        return {
            "hardware_id": hp.id,
            "hardware_name": hp.name,
            "platform": hp.platform,
            "gate_set_name": rep.gate_set_name,
            "primary_2q_gate": hw.native_gate_set.primary_2q_gate,
            "unitary_fidelity": round(float(rep.unitary_fidelity), 8),
            "total_gates": m.total_gates,
            "num_2q_gates": m.num_2q_gates,
            "num_1q_physical_pulses": m.num_1q_physical_pulses,
            "num_1q_virtual_gates": m.num_1q_virtual_gates,
            "depth": m.depth,
            "duration_ns": round(float(m.duration_ns), 2),
            "expected_gate_fidelity": round(float(m.expected_gate_fidelity), 6),
            "gate_breakdown": m.gate_breakdown,
            "optimized_operations": [op.model_dump() for op in opt_ops],
        }

    max_w = max(1, min(len(profiles), os.cpu_count() or 4))
    if max_w > 1 and len(profiles) > 1:
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            entries = list(pool.map(_bench_one, profiles))
    else:
        entries = [_bench_one(hp) for hp in profiles]

    best_id = ""
    if entries:
        ranked = sorted(
            entries,
            key=lambda e: (-e["expected_gate_fidelity"], e["num_2q_gates"], e["duration_ns"]),
        )
        best_id = ranked[0]["hardware_id"]

    return {
        "num_qubits": circ.num_qubits,
        "original_gate_count": len(circ.gate_instructions),
        "recommended_hardware_id": best_id,
        "entries": entries,
    }


def execute_variational_optimization(req: VariationalOptimizeRequest) -> Dict[str, Any]:
    """
    Closed-loop variational optimizer (VQE / QAOA) using `scipy.optimize.minimize`
    (COBYLA, Powell, Nelder-Mead, or L-BFGS-B) to find `theta* = argmin <psi(theta)|H|psi(theta)>`.
    """
    project = req.project
    n_q = max(1, len(project.qubits))
    ham_clean = (req.hamiltonian_str or "").strip()
    if not ham_clean:
        raise ValueError("请输入泡利哈密顿量代价函数，例如 '-1.0*Z0*Z1 - 0.5*X0 - 0.5*X1'")
    parsed_ham = parse_sparse_pauli_terms(ham_clean, n_q)
    if not parsed_ham:
        raise ValueError("哈密顿量有效项为空，请检查表达式")

    all_var_names = list(project.variables.keys())
    param_names = [p for p in req.param_names if p in project.variables] if req.param_names else all_var_names
    if not param_names:
        raise ValueError("当前项目没有可优化的符号参数（请先在右侧检查器或底部参数栏添加如 theta、gamma 等符号变量）")

    eff_backend = req.backend
    if eff_backend == "qutip" and n_q > 9:
        eff_backend = "kraus" if n_q <= 13 else ("trajectory" if n_q <= 24 else "stim")
    elif eff_backend == "kraus" and n_q > 14:
        eff_backend = "trajectory" if n_q <= 24 else "stim"
    elif eff_backend == "trajectory" and n_q > 24:
        eff_backend = "stim"

    base_hw = build_hardware_model(project)
    init_labels = [q.init_state for q in project.qubits]
    init_psi = (
        build_initial_statevector(project.qubits)
        if eff_backend in ("qutip", "kraus", "trajectory") and n_q <= 24
        else None
    )
    solver = QuantumSolver(hardware=base_hw, intra_gate_steps=8)
    shots = max(2048, min(project.sim_settings.shots, 8192))

    x0 = np.array([float(project.variables[k]) for k in param_names], dtype=np.float64)
    initial_params = {k: round(float(project.variables[k]), 6) for k in param_names}

    max_evals = max(8, min(220, int(req.max_iter)))
    energy_trace: List[float] = []
    best_so_far_trace: List[float] = []
    best_energy = float("inf")
    best_vec = x0.copy()
    best_fid = 1.0
    best_probs: Dict[str, float] = {}

    def _objective(vec: np.ndarray) -> float:
        nonlocal best_energy, best_vec, best_fid, best_probs
        curr_vars = dict(project.variables)
        for idx_p, pname in enumerate(param_names):
            curr_vars[pname] = float(vec[idx_p])
        circ = compile_project_circuit(project, variables_override=curr_vars)
        res = solver.solve(
            circ,
            backend=eff_backend,
            hardware=base_hw,
            shots=shots,
            seed=1024 + len(energy_trace),
            initial_state=init_psi,
            initial_state_labels=init_labels if eff_backend in ("stim", "stim_qutip") else None,
            observable_terms=parsed_ham,
        )
        en = float(evaluate_result_hamiltonian(res, parsed_ham, n_q))
        energy_trace.append(round(en, 6))
        if en < best_energy:
            best_energy = en
            best_vec = vec.copy()
            best_fid = float(res.final_fidelity if res.final_fidelity is not None else 1.0)
            best_probs = dict(res.final_probabilities)
        best_so_far_trace.append(round(best_energy, 6))
        return en

    # Always evaluate exact initial point first so index 0 is guaranteed to be initial_params
    _objective(x0)

    opts: Dict[str, Any] = {"maxiter": max_evals}
    if req.method == "COBYLA":
        opts["rhobeg"] = 0.5
    elif req.method == "L-BFGS-B":
        opts["eps"] = 1e-3

    scipy_minimize(_objective, x0, method=req.method, options=opts)

    optimal_params = {
        pname: round(float(best_vec[idx_p]), 6) for idx_p, pname in enumerate(param_names)
    }
    top_states = sorted(best_probs.items(), key=lambda kv: kv[1], reverse=True)[:8]

    return {
        "method": req.method,
        "backend": eff_backend,
        "hamiltonian_str": ham_clean,
        "param_names": param_names,
        "initial_params": initial_params,
        "optimal_params": optimal_params,
        "initial_energy": energy_trace[0],
        "optimal_energy": round(float(best_energy), 6),
        "energy_improvement": round(float(energy_trace[0] - best_energy), 6),
        "optimal_fidelity": round(float(best_fid), 6),
        "num_evaluations": len(energy_trace),
        "energy_trace": energy_trace,
        "best_so_far_trace": best_so_far_trace,
        "dominant_states": [
            {"bitstring": k, "probability": round(float(v), 6)} for k, v in top_states if v > 1e-4
        ],
    }
