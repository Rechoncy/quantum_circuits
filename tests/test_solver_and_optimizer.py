"""
Comprehensive unit tests for:
  1. Multi-backend Quantum Circuit Solver (QuTiP, Kraus, Stim, Hybrid QuTiP-to-Stim Pauli Twirling)
  2. Step-by-step state evolution tracking (density matrices, Bloch trajectories, stabilizers, fidelity)
  3. Hardware-aware Quantum Circuit Synthesizer & Optimizer (Euler 1Q, KAK 2Q, Peephole passes, Topology routing)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from quantum_solver import (
    HardwareModel,
    NativeGateSet,
    NoiseChannel,
    QuantumCircuit,
    QuantumCircuitOptimizer,
    QuantumSolver,
    gate_unitary,
    makhlin_invariants,
    synthesize_1q_rz_sx_rz,
    synthesize_2q_unitary,
    unitary_process_fidelity,
)
from quantum_solver.optimizer.euler_1q import single_qubit_unitary_fidelity


def test_1q_rz_sx_rz_exact_decomposition() -> None:
    """Verify 1-qubit superconducting Virtual-Z + SX decomposition on special and random SU(2) matrices."""
    rng = np.random.default_rng(123)

    # 1. Pure Z rotation should use 0 physical SX/X pulses
    u_rz = gate_unitary("RZ", (1.234,))
    ops_rz = synthesize_1q_rz_sx_rz(u_rz, qubit=0)
    phys_rz = [g for g in ops_rz if g.name in ("SX", "X")]
    assert len(phys_rz) == 0
    circ_rz = QuantumCircuit(1)
    circ_rz.operations.extend(ops_rz)
    assert single_qubit_unitary_fidelity(circ_rz.unitary(), u_rz) > 1.0 - 1e-12

    # 2. Hadamard gate (theta = pi/2) should use exactly 1 physical SX pulse
    u_h = gate_unitary("H")
    ops_h = synthesize_1q_rz_sx_rz(u_h, qubit=0)
    phys_h = [g for g in ops_h if g.name in ("SX", "X")]
    assert len(phys_h) == 1 and phys_h[0].name == "SX"
    circ_h = QuantumCircuit(1)
    circ_h.operations.extend(ops_h)
    assert single_qubit_unitary_fidelity(circ_h.unitary(), u_h) > 1.0 - 1e-12

    # 3. Random general U3 gates should use at most 2 physical SX pulses and have fidelity > 1 - 1e-12
    for _ in range(10):
        angles = rng.uniform(-math.pi, math.pi, size=3)
        u_rand = gate_unitary("U3", angles)
        ops = synthesize_1q_rz_sx_rz(u_rand, qubit=0)
        phys = [g for g in ops if g.name in ("SX", "X")]
        assert len(phys) <= 2
        circ = QuantumCircuit(1)
        circ.operations.extend(ops)
        assert single_qubit_unitary_fidelity(circ.unitary(), u_rand) > 1.0 - 1e-11


def test_2q_kak_synthesis_minimum_gate_counts() -> None:
    """Verify KAK 2Q synthesis finds 0, 1, 2, or 3 native 2Q gates with fidelity > 1 - 1e-9."""
    gset_cz = NativeGateSet.superconducting_cz()

    # K = 0: Tensor product H ⊗ SX requires 0 CZ gates
    U_prod = np.kron(gate_unitary("H"), gate_unitary("SX"))
    ops_0 = synthesize_2q_unitary(U_prod, (0, 1), gset_cz)
    c0 = QuantumCircuit(2)
    c0.operations.extend(ops_0)
    assert c0.num_2q_gates() == 0
    assert unitary_process_fidelity(c0.unitary(), U_prod) > 1.0 - 1e-11

    # K = 1: CNOT synthesized into CZ basis requires exactly 1 CZ gate
    U_cnot = gate_unitary("CNOT")
    ops_1 = synthesize_2q_unitary(U_cnot, (0, 1), gset_cz)
    c1 = QuantumCircuit(2)
    c1.operations.extend(ops_1)
    assert c1.num_2q_gates() == 1
    assert unitary_process_fidelity(c1.unitary(), U_cnot) > 1.0 - 1e-11

    # K = 2: iSWAP synthesized into CZ basis requires exactly 2 CZ gates
    U_iswap = gate_unitary("ISWAP")
    ops_2 = synthesize_2q_unitary(U_iswap, (0, 1), gset_cz)
    c2 = QuantumCircuit(2)
    c2.operations.extend(ops_2)
    assert c2.num_2q_gates() == 2
    assert unitary_process_fidelity(c2.unitary(), U_iswap) > 1.0 - 1e-9

    # K = 3: SWAP gate requires 3 CZ gates
    U_swap = gate_unitary("SWAP")
    ops_3 = synthesize_2q_unitary(U_swap, (0, 1), gset_cz)
    c3 = QuantumCircuit(2)
    c3.operations.extend(ops_3)
    assert c3.num_2q_gates() == 3
    assert unitary_process_fidelity(c3.unitary(), U_swap) > 1.0 - 1e-9


def test_multi_backend_solver_consistency_bell_state() -> None:
    """
    Verify that QuTiPBackend, KrausBackend, StimBackend, and Hybrid Stim+QuTiP
    produce consistent step-by-step evolution and final measurement distributions
    on a noisy 2-qubit Bell circuit.
    """
    hw = HardwareModel.superconducting_processor(
        num_qubits=2,
        t1_us=30.0,
        t2_us=20.0,
        t_1q_ns=20.0,
        t_2q_ns=40.0,
        depol_1q=0.005,
        depol_2q=0.02,
        zz_crosstalk_khz=0.0,
        readout_err=0.01,
    )

    circ = QuantumCircuit(2, name="NoisyBell")
    circ.h(0).cnot(0, 1).measure_all()

    solver = QuantumSolver(hardware=hw)
    results = solver.compare_backends(circ, shots=20000, seed=12345)

    assert set(results.keys()) == {"qutip", "kraus", "stim", "stim_qutip"}

    for bname, res in results.items():
        # Step 0 (INIT) + Step 1 (H) + Step 2 (CNOT) = 3 steps
        assert len(res.steps) == 3
        p00 = res.final_probabilities.get("00", 0.0)
        p11 = res.final_probabilities.get("11", 0.0)
        # Both |00> and |11> should be around ~0.48, and |01>, |10> around ~0.02
        assert 0.43 < p00 < 0.52, f"{bname} p00={p00}"
        assert 0.43 < p11 < 0.52, f"{bname} p11={p11}"

    # QuTiP and Kraus final density matrices should match to within < 1e-3
    rho_qutip = results["qutip"].final_density_matrix
    rho_kraus = results["kraus"].final_density_matrix
    assert rho_qutip is not None and rho_kraus is not None
    assert np.linalg.norm(rho_qutip - rho_kraus) < 2e-3

    # QuTiP intra-gate pulse trajectories should be recorded
    assert results["qutip"].steps[1].pulse_times_ns is not None
    assert len(results["qutip"].steps[1].pulse_times_ns) >= 5


def test_circuit_optimizer_reduces_redundant_gates() -> None:
    """
    Verify that QuantumCircuitOptimizer simplifies a redundant multi-gate circuit
    and improves noisy hardware fidelity.
    """
    hw = HardwareModel.superconducting_processor(
        num_qubits=2,
        t1_us=25.0,
        t2_us=20.0,
        depol_1q=0.002,
        depol_2q=0.015,
        primary_2q_gate="CZ",
    )

    # Construct a 2Q circuit with redundant gates:
    # H(1) -> CNOT(0,1) -> H(1) is equivalent to a SINGLE CZ(0,1) gate!
    # Followed by CNOT(0,1) -> CNOT(1,0) -> CNOT(0,1) (3 CNOTs = SWAP).
    # Total 4 CNOTs + 2 H gates on the same pair (0, 1) -> can be synthesized with <= 3 CZ gates!
    circ = QuantumCircuit(2, name="Redundant2Q")
    circ.h(0).s(0).h(1).cnot(0, 1).h(1).cnot(0, 1).cnot(1, 0).cnot(0, 1).t(0).tdg(0)

    optimizer = QuantumCircuitOptimizer(hardware=hw)
    report = optimizer.optimize_circuit(circ, evaluate_noisy_fidelity=True)

    assert report.unitary_fidelity > 1.0 - 1e-9
    assert report.optimized_metrics.num_2q_gates <= 3
    assert report.optimized_metrics.num_2q_gates < report.original_metrics.num_2q_gates
    assert report.noisy_fidelity_optimized > report.noisy_fidelity_original


def test_gui_circuit_parser_and_parameter_sweep() -> None:
    """Verify the GUI DSL circuit parser and 1D/2D parameter sweep engine."""
    from quantum_solver.gui import (
        ParameterSweepEngine,
        extract_symbolic_parameters,
        parse_circuit_script,
    )

    script = """
    qubits 2
    RY(theta) 0
    CNOT 0 1
    DEPOLARIZE(p_err) 0 1
    MEASURE 0 1
    """
    syms = extract_symbolic_parameters(script)
    assert syms == ["theta", "p_err"]

    circ = parse_circuit_script(script, {"theta": math.pi / 2.0, "p_err": 0.01})
    assert circ.num_qubits == 2
    assert len(circ.gate_instructions) == 2

    hw = HardwareModel.superconducting_processor(num_qubits=2)
    engine = ParameterSweepEngine()
    res_1d = engine.run_1d_sweep(
        script=script,
        base_variables={"theta": 0.0, "p_err": 0.01},
        base_hardware=hw,
        param_name="theta",
        start_val=0.0,
        stop_val=math.pi,
        num_points=7,
        backend="kraus",
        shots=1024,
    )
    assert len(res_1d.x_values) == 7
    # At theta = 0, state is |00>; at theta = pi, RY(pi)|0> = |1> -> CNOT -> |11>
    assert res_1d.populations["00"][0] > 0.94
    assert res_1d.populations["11"][-1] > 0.94


def test_neutral_atom_rydberg_platform_synthesis_and_solver() -> None:
    """
    Verify Neutral-Atom Rydberg hardware model, RXY + Virtual-RZ + CZ synthesis,
    CCZ support, and multi-backend solver consistency.
    """
    hw_na = HardwareModel.neutral_atom_processor(num_qubits=3)
    assert hw_na.native_gate_set.decomposition_mode_1q == "rxy_rz"
    assert hw_na.native_gate_set.primary_2q_gate == "CZ"

    # Arbitrary 2Q circuit compiled for Neutral Atom {RXY, Virtual-RZ, CZ}
    circ = QuantumCircuit(2, name="NA_Target")
    circ.h(0).s(0).cnot(0, 1).rx(0, 0.7).ry(1, 1.1).cnot(1, 0)

    opt = QuantumCircuitOptimizer(hardware=hw_na)
    rep = opt.optimize_circuit(circ, evaluate_noisy_fidelity=True)
    assert rep.unitary_fidelity > 1.0 - 1e-8
    allowed = set(hw_na.native_gate_set.allowed_names())
    assert all(g.name in allowed for g in rep.optimized_circuit.gate_instructions)
    assert rep.optimized_metrics.num_2q_gates <= 2

    # 3-qubit Neutral Atom GHZ circuit using RXY + Rydberg CZ across QuTiP, Kraus, Stim
    from quantum_solver.gui import PRESET_CIRCUIT_TEMPLATES, parse_circuit_script

    na_tpl_name = [k for k in PRESET_CIRCUIT_TEMPLATES if "中性原子" in k][0]
    script, default_vars = PRESET_CIRCUIT_TEMPLATES[na_tpl_name]
    na_circ = parse_circuit_script(script, default_vars)
    solver = QuantumSolver(hardware=hw_na)
    results = solver.compare_backends(na_circ, shots=10000, seed=7)
    assert set(results.keys()) == {"qutip", "kraus", "stim", "stim_qutip"}
    for res in results.values():
        p011 = res.final_probabilities.get("011", 0.0)
        p100 = res.final_probabilities.get("100", 0.0)
        assert 0.42 < p011 < 0.56
        assert 0.42 < p100 < 0.56


def test_studio_v2_custom_gates_register_and_fastapi() -> None:
    """
    Verify Quantum Studio v2 unified project engine:
      - Dynamic qubit register (add/remove/initial state)
      - 4 custom gate modes (matrix, expression, hamiltonian, composite)
      - Promoting custom 2Q gate as native for optimizer
      - Neutral Atom Rydberg blockade radius topology generation
      - Bidirectional DSL sync & OpenQASM 3 export
    """
    from quantum_solver.studio import (
        OptimizeRequest,
        SweepRequest,
        build_preset_projects,
        dsl_to_project_ops,
        execute_project_optimization,
        execute_project_simulation,
        execute_project_sweep,
        project_to_dsl,
        project_to_openqasm3,
    )
    from quantum_solver.studio.custom_gates import (
        evaluate_custom_gate_unitary,
        validate_unitary_matrix,
    )
    from quantum_solver.studio.project_engine import compute_topology_edges

    presets = build_preset_projects()
    p0 = presets[0]  # 4Q Neutral Atom Rydberg GHZ with 4 custom gates
    assert len(p0.qubits) == 4
    assert len(p0.custom_gates) == 4

    # 1. All 4 custom gate modes must produce valid unitaries
    for cg in p0.custom_gates:
        U, _ = evaluate_custom_gate_unitary(cg)
        ok, err, _ = validate_unitary_matrix(U)
        assert ok and err < 1e-10

    # 2. Rydberg blockade radius topology: atoms at x = 0, 4.5, 9.0, 13.5 um with Rb = 6.5 um -> linear edges
    hw_na = p0.hardware_profiles[0]
    edges = compute_topology_edges(p0.qubits, hw_na)
    assert edges == [(0, 1), (1, 2), (2, 3)]

    # 3. Bidirectional DSL sync adding a 5th qubit
    dsl = project_to_dsl(p0) + "\nH 4\nMEASURE 4"
    new_q, new_ops, _ = dsl_to_project_ops(dsl, p0)
    assert len(new_q) == 5
    assert any(op.name == "H" and op.qubits == [4] for op in new_ops)

    # 4. OpenQASM 3 export
    qasm = project_to_openqasm3(p0)
    assert "OPENQASM 3.0;" in qasm and "qubit[4] q;" in qasm

    # 5. Custom 2Q gate promoted as native gate for optimizer
    p_opt = presets[1].model_copy(deep=True)
    p_opt.custom_gates[0].is_native = True  # SQISWAP
    opt_res = execute_project_optimization(OptimizeRequest(project=p_opt, evaluate_noisy_fidelity=False))
    assert opt_res["primary_2q_gate"] == "SQISWAP"
    assert opt_res["unitary_fidelity"] > 1.0 - 1e-8

    # 6. End-to-end simulation & 1D sweep
    sim_res = execute_project_simulation(p0)
    assert sim_res["results"][sim_res["primary_backend"]]["final_fidelity"] > 0.95
    sw_res = execute_project_sweep(
        SweepRequest(
            project=presets[3],
            mode="1d",
            param_x="theta",
            x_start=0.0,
            x_stop=math.pi,
            x_points=5,
            backend="kraus",
        )
    )
    assert len(sw_res["x_values"]) == 5


def test_parallel_layers_ideal_gate_stim_init_sabre_and_direct_synthesis() -> None:
    """
    Verify all upgraded physics, routing, synthesis, and sweep capabilities:
      1. Parallel layer pulse evolution in QuTiP & Kraus (concurrent step timing)
      2. Single-gate noise_mode='none' (skip_hw_noise) yields F=1.0
      3. Gaussian pulse shape integration accuracy
      4. Stim non-|0> initial state preparation + custom Clifford gate exact simulation
      5. SABRE-inspired lookahead topology routing preserves exact unitary equivalence
      6. Direct target unitary & Pauli Hamiltonian synthesis API
      7. Parallel 1D/2D Sweep with selectable observed_qubit
    """
    from quantum_solver.studio import (
        CircuitOpModel,
        CustomGateDef,
        SweepRequest,
        SynthesizeUnitaryRequest,
        build_preset_projects,
        execute_project_simulation,
        execute_project_sweep,
        execute_unitary_synthesis,
    )

    presets = build_preset_projects()

    # 1. Parallel Layer Timing: Step 0 has H(q0) & RXY(q1) -> both 400 ns -> parallel layer duration = 400 ns (not 800 ns!)
    p0 = presets[0].model_copy(deep=True)
    p0.hardware_profiles[0].pulse_shape = "gaussian"
    sim_par = execute_project_simulation(p0)
    q_res = sim_par["results"]["qutip"]
    # Steps: INIT(0) + Layer0(H||RXY, 400ns) + Layer1(CZ, 240ns) + Layer2(SQISWAP, 32ns) + Layer3(CZ, 240ns) = 912 ns!
    assert len(q_res["steps"]) == 5
    assert abs(q_res["total_duration_ns"] - 912.0) < 1e-3
    assert "∥" in q_res["steps"][1]["operation_label"]
    assert q_res["final_fidelity"] > 0.96

    # 2. Single-gate noise_mode='none' yields exact F=1.0 on noisy hardware
    p_ideal_gate = presets[3].model_copy(deep=True)
    p_ideal_gate.operations = [
        CircuitOpModel(id="g1", step=0, kind="gate", name="RX", qubits=[0], params=[math.pi / 2.0], noise_mode="none"),
        CircuitOpModel(id="m1", step=1, kind="measure", name="MEASURE", qubits=[0]),
    ]
    res_ideal_g = execute_project_simulation(p_ideal_gate)["results"]["qutip"]
    assert res_ideal_g["final_fidelity"] > 1.0 - 1e-5

    # 3. Stim initial state |1> + custom Clifford gate (custom Hadamard matrix)
    p_stim = presets[3].model_copy(deep=True)
    p_stim.qubits[0].init_state = "|1>"
    p_stim.custom_gates.append(
        CustomGateDef(
            id="cg_h_cliff",
            name="MY_H",
            display_label="MyH",
            num_qubits=1,
            definition_type="matrix",
            matrix_exprs=[["1/sqrt(2)", "1/sqrt(2)"], ["1/sqrt(2)", "-1/sqrt(2)"]],
        )
    )
    p_stim.operations = [
        CircuitOpModel(id="s1", step=0, kind="gate", name="MY_H", qubits=[0], noise_mode="none"),
        CircuitOpModel(id="sm", step=1, kind="measure", name="MEASURE", qubits=[0]),
    ]
    p_stim.sim_settings.backend = "stim"
    p_stim.sim_settings.enable_readout = False
    p_stim.sim_settings.enable_t1_t2 = False
    p_stim.sim_settings.enable_depol = False
    res_stim = execute_project_simulation(p_stim)["results"]["stim"]
    # MY_H |1> = |-> => Bloch X = -1.0, P(0) ~ 0.5, P(1) ~ 0.5
    assert abs(res_stim["steps"][-1]["bloch_vectors"]["0"][0] - (-1.0)) < 1e-6

    # 4. SABRE-style lookahead topology routing on 4Q linear chain preserves exact unitary
    hw_lin = HardwareModel.superconducting_processor(
        num_qubits=4, topology=[(0, 1), (1, 2), (2, 3)], primary_2q_gate="CZ"
    )
    c_nonadj = QuantumCircuit(4, name="NonAdjLinear")
    c_nonadj.h(0).cz(0, 2).cz(0, 3).rx(1, 0.5)
    opt = QuantumCircuitOptimizer(hardware=hw_lin)
    rep = opt.optimize_circuit(c_nonadj, evaluate_noisy_fidelity=False)
    assert rep.unitary_fidelity > 1.0 - 1e-7
    for g in rep.optimized_circuit.gate_instructions:
        if len(g.qubits) == 2:
            assert hw_lin.is_connected(g.qubits[0], g.qubits[1])

    # 5. Direct target unitary & Pauli Hamiltonian synthesis
    synth_res = execute_unitary_synthesis(
        SynthesizeUnitaryRequest(
            project=presets[1],
            mode="hamiltonian",
            num_qubits=2,
            hamiltonian_str="0.5*XX + 0.5*YY",
            hamiltonian_scale=math.pi / 4.0,
            evaluate_noisy_fidelity=False,
        )
    )
    assert synth_res["unitary_fidelity"] > 1.0 - 1e-8
    assert synth_res["optimized_metrics"]["num_2q_gates"] <= 2

    # 6. Parallel 2D sweep with observed_qubit=1
    sw_2d = execute_project_sweep(
        SweepRequest(
            project=presets[1],
            mode="2d",
            param_x="theta",
            x_start=0.0,
            x_stop=math.pi,
            x_points=3,
            param_y="phi",
            y_start=0.0,
            y_stop=math.pi,
            y_points=3,
            target_bitstring="00",
            observed_qubit=1,
            backend="kraus",
        )
    )
    assert sw_2d["observed_qubit"] == 1
    assert len(sw_2d["z0_grid"]) == 3 and len(sw_2d["z0_grid"][0]) == 3


def test_natural_language_to_circuit_compiler() -> None:
    """
    Verify Natural Language -> Quantum Circuit generation & physical accuracy:
      1. 2Q Grover search for |11> achieves P(|11>) > 0.92 under ideal simulation
      2. 3Q W-state achieves ~1/3 population on |100>, |010>, |001>
      3. 4Q Neutral Atom GHZ selects hw_neutral_atom & produces ~0.5 |0000> + ~0.5 |1111>
      4. 3Q QFT & 4Q 2-layer VQE ansatz generate valid symbolic parameters
      5. Free-form Chinese multi-clause description + initial state + append mode
    """
    from quantum_solver.studio import (
        CircuitOpModel,
        NlCircuitRequest,
        QubitSpec,
        build_preset_projects,
        execute_project_simulation,
        generate_circuit_from_nl,
    )

    base = build_preset_projects()[0]

    def _sim_nl(prompt: str, mode: str = "replace", proj=base):
        res = generate_circuit_from_nl(NlCircuitRequest(prompt=prompt, project=proj, mode=mode))
        p_test = proj.model_copy(deep=True)
        p_test.qubits = [QubitSpec(**q) for q in res["qubits"]]
        p_test.operations = [CircuitOpModel(**o) for o in res["operations"]]
        p_test.variables = res["variables"]
        p_test.sim_settings.backend = "kraus"
        p_test.sim_settings.enable_t1_t2 = False
        p_test.sim_settings.enable_depol = False
        p_test.sim_settings.enable_zz = False
        p_test.sim_settings.enable_readout = False
        sim = execute_project_simulation(p_test)["results"]["kraus"]
        return res, p_test, sim

    # 1. Grover |11>
    res_g, _, sim_g = _sim_nl("生成 2 比特 Grover 量子搜索线路，目标态 |11⟩")
    assert len(res_g["qubits"]) == 2
    assert sim_g["final_probabilities"]["11"] > 0.99

    # 2. 3Q W-state
    res_w, _, sim_w = _sim_nl("制备 3 比特 W 态")
    assert len(res_w["qubits"]) == 3
    for bs in ("100", "010", "001"):
        assert 0.30 < sim_w["final_probabilities"][bs] < 0.37, f"W-state {bs}={sim_w['final_probabilities'][bs]}"

    # 3. 4Q Neutral Atom GHZ
    res_ghz, _, sim_ghz = _sim_nl("用中性原子里德堡阵列生成 4 比特 GHZ 态")
    assert res_ghz["recommended_hardware_id"] == "hw_neutral_atom"
    assert len(res_ghz["qubits"]) == 4
    assert sim_ghz["final_probabilities"]["0000"] > 0.48
    assert sim_ghz["final_probabilities"]["1111"] > 0.48

    # 4. VQE & QFT
    res_vqe = generate_circuit_from_nl(NlCircuitRequest(prompt="创建 4 比特 2 层硬件高效 VQE 变分线路", project=base))
    assert len(res_vqe["qubits"]) == 4 and len(res_vqe["variables"]) >= 8
    res_qft = generate_circuit_from_nl(NlCircuitRequest(prompt="生成 3 比特量子傅里叶变换 QFT 线路", project=base))
    assert len(res_qft["qubits"]) == 3 and "OPENQASM 3.0;" in res_qft["openqasm3"]

    # 5. Free-form Chinese multi-clause + init state + append mode
    res_free, p_free, sim_free = _sim_nl(
        "创建 3 个比特，q0 初态设为 |1⟩，然后对 q1 做 Hadamard 门，接着在 q1 和 q2 之间做 CNOT，最后测量"
    )
    assert len(res_free["qubits"]) == 3
    assert res_free["qubits"][0]["init_state"] == "|1>"
    # |1> ⊗ (|00> + |11>)/sqrt(2) => P(100) = 0.5, P(111) = 0.5
    assert sim_free["final_probabilities"]["100"] > 0.48
    assert sim_free["final_probabilities"]["111"] > 0.48

    # Append mode: add X on q0 -> flips |100>,|111> to |000>,|011>
    res_app, _, sim_app = _sim_nl("在末尾对 q0 做 X 门并测量", mode="append", proj=p_free)
    assert sim_app["final_probabilities"]["000"] > 0.48
    assert sim_app["final_probabilities"]["011"] > 0.48

    # 6. Synchronized trailing noise layer alignment (all 4 DEPOL ops share the EXACT same column step!)
    res_ghz_n = generate_circuit_from_nl(
        NlCircuitRequest(prompt="用中性原子里德堡阵列生成 4 比特 GHZ 态并加入 1% 退极化噪声后测量", project=base, mode="auto")
    )
    noise_steps = {op["step"] for op in res_ghz_n["operations"] if op["kind"] == "noise"}
    assert len(noise_steps) == 1, f"Trailing noise should align in 1 column, got steps {noise_steps}"

    # 7. Compound prompt: named algorithm + extra gate + noise in one prompt
    res_comp = generate_circuit_from_nl(
        NlCircuitRequest(
            prompt="生成 3 比特 GHZ 态，然后在 q0 上加一个 RZ(phi) 相位门，最后加 2% 退极化噪声",
            project=base,
            mode="auto",
        )
    )
    assert any(op["name"] == "RZ" and op["qubits"] == [0] for op in res_comp["operations"])
    assert "phi" in res_comp["variables"]
    assert any(op["kind"] == "noise" and op["params"] == [0.02] for op in res_comp["operations"])

    # 8. Multi-turn conversational refinement on top of res_ghz_n
    p_turn1 = base.model_copy(deep=True)
    p_turn1.name = res_ghz_n["title"]
    p_turn1.qubits = [QubitSpec(**q) for q in res_ghz_n["qubits"]]
    p_turn1.operations = [CircuitOpModel(**o) for o in res_ghz_n["operations"]]
    p_turn1.variables = res_ghz_n["variables"]

    # Turn 2: tune noise to 3%, add 1 qubit and entangle with q3 via CZ
    res_t2 = generate_circuit_from_nl(
        NlCircuitRequest(
            prompt="把退极化噪声改成 3%，再增加 1 个比特并和 q3 做 CZ 纠缠",
            project=p_turn1,
            mode="auto",
        )
    )
    assert res_t2["action_mode"] == "refine"
    assert len(res_t2["qubits"]) == 5
    t2_noise = [op for op in res_t2["operations"] if op["kind"] == "noise"]
    t2_gates = [op for op in res_t2["operations"] if op["kind"] == "gate"]
    t2_meas = [op for op in res_t2["operations"] if op["kind"] == "measure"]
    assert len(t2_noise) == 5
    assert all(op["params"] == [0.03] for op in t2_noise)
    assert len({op["step"] for op in t2_noise}) == 1, "Refined trailing noise must stay aligned in 1 vertical column"
    assert any(op["name"] == "CZ" and set(op["qubits"]) == {3, 4} for op in t2_gates)
    assert max(op["step"] for op in t2_gates) < t2_noise[0]["step"] < t2_meas[0]["step"]

    # Turn 3: replace all CZ with ISWAP & delete all noise
    p_turn2 = p_turn1.model_copy(deep=True)
    p_turn2.qubits = [QubitSpec(**q) for q in res_t2["qubits"]]
    p_turn2.operations = [CircuitOpModel(**o) for o in res_t2["operations"]]
    res_t3 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="把所有 CZ 换成 iSWAP，删掉所有噪声", project=p_turn2, mode="auto")
    )
    assert res_t3["action_mode"] == "refine"
    assert not any(op["name"] == "CZ" for op in res_t3["operations"])
    assert sum(1 for op in res_t3["operations"] if op["name"] == "ISWAP") >= 4
    assert not any(op["kind"] == "noise" for op in res_t3["operations"])






