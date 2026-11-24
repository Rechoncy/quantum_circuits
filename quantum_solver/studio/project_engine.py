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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from ..core.circuit import GateInstruction, MeasureInstruction, NoiseInstruction, QuantumCircuit
from ..core.gates import FIXED_MATRICES, NativeGateSet, NativeGateSpec
from ..core.hardware import CouplerPhysicalParams, HardwareModel, QubitPhysicalParams
from ..core.noise import NoiseChannel, ReadoutError
from ..gui.circuit_parser import eval_math_expr, extract_symbolic_parameters
from ..gui.sweep_engine import HARDWARE_SWEEP_PARAMS, apply_hardware_param
from ..optimizer.compiler import QuantumCircuitOptimizer
from ..solver.base import EvolutionResult
from ..solver.unified_solver import QuantumSolver
from .custom_gates import (
    _safe_complex_eval,
    evaluate_custom_gate_unitary,
    parse_pauli_hamiltonian,
    validate_unitary_matrix,
)
from .models import (
    CircuitOpModel,
    CustomGateDef,
    HardwareProfileModel,
    OptimizeRequest,
    ProjectModel,
    QubitSpec,
    SimulationSettingsModel,
    SweepRequest,
    SynthesizeUnitaryRequest,
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

    return [rydberg_proj, qft_opt_proj, bell_noise_proj, rabi_proj]


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
    custom_native_2q_names: List[str] = []
    for cg in project.custom_gates:
        if not cg.is_native:
            continue
        try:
            u_cg, _ = evaluate_custom_gate_unitary(cg)
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

        if uname in custom_by_name:
            cg = custom_by_name[uname]
            if len(valid_q) != cg.num_qubits:
                continue
            if cg.definition_type == "composite" and cg.sub_operations:
                # Expand composite macro directly onto mapped qubits
                for sub in sorted(cg.sub_operations, key=lambda s: s.step):
                    mapped_q = tuple(valid_q[lq] for lq in sub.qubits if 0 <= lq < len(valid_q))
                    sub_p = _resolve_op_params(sub.params, vars_map)
                    if sub.kind == "gate" and mapped_q:
                        circ.add_gate(
                            sub.name,
                            mapped_q,
                            params=sub_p,
                            skip_hw_noise=skip_hw,
                        )
            else:
                u_mat, h_mat = evaluate_custom_gate_unitary(cg, p_vals if p_vals else None)
                dur = op.duration_ns if op.duration_ns is not None else cg.duration_ns
                direct_h = None
                if h_mat is not None:
                    scale = float(p_vals[0]) if (cg.param_names and p_vals) else float(cg.hamiltonian_scale)
                    direct_h = scale * h_mat
                circ.add_gate(
                    name=cg.name,
                    qubits=valid_q,
                    params=p_vals,
                    duration_ns=dur,
                    label=cg.display_label or cg.name,
                    custom_matrix=u_mat,
                    noise_after=noise_after,
                    layer=op.step,
                    skip_hw_noise=skip_hw,
                    custom_hamiltonian=direct_h,
                )
        else:
            circ.add_gate(
                name=uname,
                qubits=valid_q,
                params=p_vals,
                duration_ns=op.duration_ns,
                noise_after=noise_after,
                layer=op.step,
                skip_hw_noise=skip_hw,
            )

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
    """
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


def project_to_openqasm3(project: ProjectModel) -> str:
    """Export the project circuit as OpenQASM 3.0 text."""
    n = len(project.qubits)
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
        "",
    ]
    vars_map = project.variables
    ordered = [
        op for _, op in sorted(enumerate(project.operations), key=lambda pair: (pair[1].step, pair[0]))
    ]
    qasm_map = {
        "H": "h", "X": "x", "Y": "y", "Z": "z", "S": "s", "SDG": "sdg",
        "T": "t", "TDG": "tdg", "SX": "sx", "RX": "rx", "RY": "ry", "RZ": "rz",
        "CNOT": "cx", "CX": "cx", "CZ": "cz", "SWAP": "swap", "CCX": "ccx",
    }
    for op in ordered:
        if op.kind == "measure":
            for q in op.qubits:
                lines.append(f"c[{q}] = measure q[{q}];")
        elif op.kind == "gate":
            uname = op.name.upper()
            q_args = ", ".join(f"q[{q}]" for q in op.qubits)
            p_vals = _resolve_op_params(op.params, vars_map)
            g_id = qasm_map.get(uname, uname.lower())
            if p_vals:
                p_str = ", ".join(f"{p:.6g}" for p in p_vals)
                lines.append(f"{g_id}({p_str}) {q_args};")
            else:
                lines.append(f"{g_id} {q_args};")
    return "\n".join(lines)


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

    return {
        "backend_name": res.backend_name,
        "num_qubits": res.num_qubits,
        "total_duration_ns": round(float(res.total_duration_ns), 3),
        "shots": res.shots,
        "final_fidelity": round(float(res.final_fidelity), 6) if res.final_fidelity is not None else None,
        "final_purity": round(float(res.final_purity), 6) if res.final_purity is not None else None,
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

    report = optimizer.optimize_circuit(
        circ,
        hardware=hw,
        evaluate_noisy_fidelity=(req.evaluate_noisy_fidelity and circ.num_qubits <= 12),
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

        def _eval_1d_point(args: Tuple[int, float]) -> Tuple[Dict[str, float], float, float, List[float]]:
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
            )
            fid = round(float(res.final_fidelity if res.final_fidelity is not None else 1.0), 6)
            pur = round(float(res.final_purity if res.final_purity is not None else 1.0), 6)
            bv = res.steps[-1].bloch_vectors.get(obs_q, (0.0, 0.0, 0.0))
            return res.final_probabilities, fid, pur, [round(float(bv[0]), 5), round(float(bv[1]), 5), round(float(bv[2]), 5)]

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
            for raw_pmap, _, _, _ in rows_1d:
                for k, v in raw_pmap.items():
                    if v > max_by_bstr.get(k, 0.0):
                        max_by_bstr[k] = float(v)
            ranked = sorted(max_by_bstr.keys(), key=lambda k: max_by_bstr[k], reverse=True)
            tracked_bstrs = ranked[:16]

        pops: Dict[str, List[float]] = {b: [] for b in tracked_bstrs}
        fids: List[float] = []
        purs: List[float] = []
        bloch_obs: List[List[float]] = []
        for raw_pmap, fid, pur, bv_list in rows_1d:
            for b in tracked_bstrs:
                pops[b].append(round(float(raw_pmap.get(b, 0.0)), 6))
            fids.append(fid)
            purs.append(pur)
            bloch_obs.append(bv_list)

        return {
            "mode": "1d",
            "param_x": req.param_x,
            "param_x_label": HARDWARE_SWEEP_PARAMS.get(req.param_x, req.param_x),
            "backend": eff_backend,
            "observed_qubit": obs_q,
            "observed_qubit_name": obs_q_name,
            "x_values": np.round(x_vals, 5).tolist(),
            "populations": pops,
            "fidelities": fids,
            "purities": purs,
            "bloch_q0": bloch_obs,
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

    def _eval_2d_point(args: Tuple[int, int, float, float]) -> Tuple[int, int, float, float, float]:
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
        )
        p_val = float(res.final_probabilities.get(target_bstr, 0.0))
        f_val = float(res.final_fidelity if res.final_fidelity is not None else 1.0)
        z_val = float(res.steps[-1].bloch_vectors.get(obs_q, (0.0, 0.0, 0.0))[2])
        return iy, ix, p_val, f_val, z_val

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
    for iy, ix, p_val, f_val, z_val in rows_2d:
        pop_grid[iy, ix] = p_val
        fid_grid[iy, ix] = f_val
        z_grid[iy, ix] = z_val

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
        "x_values": np.round(x_vals, 5).tolist(),
        "y_values": np.round(y_vals, 5).tolist(),
        "population_grid": np.round(pop_grid, 5).tolist(),
        "fidelity_grid": np.round(fid_grid, 5).tolist(),
        "z0_grid": np.round(z_grid, 5).tolist(),
    }
