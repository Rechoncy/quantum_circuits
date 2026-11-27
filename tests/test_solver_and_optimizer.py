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


def test_high_qubit_scaling_kraus_trajectory_stim_and_nl() -> None:
    """
    Verify large-qubit scaling across all solvers and expanded AI Assistant:
      1. 10Q tensor-contracted KrausBackend noisy simulation (< 1s)
      2. 16Q TrajectoryBackend non-Clifford VQE statevector & noisy trajectories + 8Q exact W-state
      3. 100-qubit StimBackend noisy GHZ simulation (n = 100 > 64, zero int64 overflow!)
      4. Expanded AI Assistant: 50Q Neutral Atom GHZ (2D grid + stim), 16Q VQE (trajectory),
         17Q Rotated Surface Code, 5Q Grover |10110>, and multi-turn '扩容到 30 个比特'
      5. Automatic project_engine fallback when user leaves backend='qutip' on a 15Q circuit
    """
    from quantum_solver.studio import (
        CircuitOpModel,
        NlCircuitRequest,
        QubitSpec,
        build_preset_projects,
        execute_project_simulation,
        generate_circuit_from_nl,
    )

    solver = QuantumSolver()

    # 1. 10-qubit tensor-contracted KrausBackend noisy GHZ
    hw_10 = HardwareModel.superconducting_processor(
        num_qubits=10, depol_1q=0.001, depol_2q=0.003, readout_err=0.002
    )
    c10 = QuantumCircuit(10, name="GHZ_10Q")
    c10.h(0)
    for i in range(9):
        c10.cnot(i, i + 1)
    c10.measure_all()
    res_k10 = solver.solve(c10, backend="kraus", hardware=hw_10, shots=4096, seed=11)
    assert res_k10.num_qubits == 10
    assert res_k10.final_probabilities.get("0" * 10, 0.0) > 0.42
    assert res_k10.final_probabilities.get("1" * 10, 0.0) > 0.42
    assert 0.80 < (res_k10.final_fidelity or 0.0) < 1.0

    # 2. 16-qubit TrajectoryBackend on non-Clifford hardware-efficient VQE (exact + noisy MC)
    c16 = QuantumCircuit(16, name="VQE_16Q")
    for q in range(16):
        c16.ry(q, 0.37 + 0.05 * q)
    for q in range(15):
        c16.cz(q, q + 1)
    for q in range(16):
        c16.rz(q, 0.21 * (q + 1))
    c16.measure_all()
    res_traj_exact = solver.solve(c16, backend="trajectory", hardware=None, shots=2048, seed=7)
    assert res_traj_exact.final_fidelity == 1.0
    assert sum(res_traj_exact.measurement_counts.values()) == 2048

    hw_16 = HardwareModel.neutral_atom_processor(
        num_qubits=16, depol_1q=0.001, depol_2q=0.004, readout_err=0.005
    )
    res_traj_noisy = solver.trajectory_backend.run(
        c16, hardware=hw_16, shots=2048, seed=7, num_trajectories=4
    )
    assert 0.70 < (res_traj_noisy.final_fidelity or 0.0) <= 1.0
    assert 0.50 < (res_traj_noisy.final_purity or 0.0) <= 1.0

    # 3. 100-qubit StimBackend noisy GHZ (verifies np.packbits counting beyond 64-bit int limit!)
    hw_100 = HardwareModel.neutral_atom_processor(
        num_qubits=100, t1_us=10000.0, t2_us=8000.0, depol_1q=0.0001, depol_2q=0.0003, readout_err=0.0003
    )
    c100 = QuantumCircuit(100, name="GHZ_100Q")
    c100.h(0)
    for q in range(99):
        c100.cnot(q, q + 1)
    c100.measure_all()
    res_stim_100 = solver.solve(c100, backend="stim", hardware=hw_100, shots=4000, seed=99)
    assert res_stim_100.num_qubits == 100
    assert sum(res_stim_100.measurement_counts.values()) == 4000
    assert len(next(iter(res_stim_100.measurement_counts.keys()))) == 100
    assert res_stim_100.final_probabilities.get("0" * 100, 0.0) > 0.35
    assert res_stim_100.final_probabilities.get("1" * 100, 0.0) > 0.35

    # 4. Expanded AI Assistant: 50Q Neutral Atom GHZ, 8Q W-state, 5Q Grover, 17Q Surface Code, 16Q VQE
    base = build_preset_projects()[0]

    nl_50 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="用中性原子里德堡阵列生成 50 比特 GHZ 态并测量", project=base)
    )
    assert len(nl_50["qubits"]) == 50
    assert nl_50["recommended_backend"] == "stim"
    # Verify 2D snake grid coordinates (not all y == 0!)
    assert any(q["pos_y"] > 0 for q in nl_50["qubits"])

    nl_vqe16 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="创建 16 比特 2 层硬件高效 VQE 变分线路", project=base)
    )
    assert len(nl_vqe16["qubits"]) == 16
    assert nl_vqe16["recommended_backend"] == "trajectory"

    nl_surf = generate_circuit_from_nl(
        NlCircuitRequest(prompt="生成 17 比特旋转表面码纠错线路", project=base)
    )
    assert len(nl_surf["qubits"]) == 17
    assert nl_surf["recommended_backend"] == "stim"

    # Verify exact 8-qubit W-state uniform 1/8 = 0.125 single-excitation probabilities
    nl_w8 = generate_circuit_from_nl(NlCircuitRequest(prompt="制备 8 比特精确 W 态", project=base))
    assert len(nl_w8["qubits"]) == 8
    p_w8 = base.model_copy(deep=True)
    p_w8.qubits = [QubitSpec(**q) for q in nl_w8["qubits"]]
    p_w8.operations = [CircuitOpModel(**o) for o in nl_w8["operations"]]
    p_w8.variables = nl_w8["variables"]
    p_w8.sim_settings.backend = "trajectory"
    p_w8.sim_settings.enable_t1_t2 = False
    p_w8.sim_settings.enable_depol = False
    p_w8.sim_settings.enable_zz = False
    p_w8.sim_settings.enable_readout = False
    sim_w8 = execute_project_simulation(p_w8)["results"]["trajectory"]
    for pos in range(8):
        bstr = "".join("1" if idx == pos else "0" for idx in range(8))
        assert 0.122 < sim_w8["final_probabilities"].get(bstr, 0.0) < 0.128

    # Verify 5-qubit Grover search for |10110> (5 data + 2 ancillas = 7Q total -> |1011000> dominant!)
    nl_g5 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="生成 5 比特 Grover 量子搜索线路，目标态 |10110⟩", project=base)
    )
    p_g5 = base.model_copy(deep=True)
    p_g5.qubits = [QubitSpec(**q) for q in nl_g5["qubits"]]
    p_g5.operations = [CircuitOpModel(**o) for o in nl_g5["operations"]]
    p_g5.sim_settings.backend = "trajectory"
    p_g5.sim_settings.enable_t1_t2 = False
    p_g5.sim_settings.enable_depol = False
    p_g5.sim_settings.enable_zz = False
    p_g5.sim_settings.enable_readout = False
    sim_g5 = execute_project_simulation(p_g5)["results"]["trajectory"]
    assert sim_g5["final_probabilities"].get("1011000", 0.0) > 0.95

    # Multi-turn refinement: expand 8Q W-state circuit to 30 qubits
    nl_exp30 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="扩容到 30 个比特", project=p_w8, mode="auto")
    )
    assert nl_exp30["action_mode"] == "refine"
    assert len(nl_exp30["qubits"]) == 30

    # 5. Automatic OOM guard when sim_settings.backend == 'qutip' on 16Q VQE
    p_guard = base.model_copy(deep=True)
    p_guard.qubits = [QubitSpec(**q) for q in nl_vqe16["qubits"]]
    p_guard.operations = [CircuitOpModel(**o) for o in nl_vqe16["operations"]]
    p_guard.variables = nl_vqe16["variables"]
    p_guard.sim_settings.backend = "qutip"  # intentionally too small for 16Q!
    p_guard.sim_settings.enable_t1_t2 = False
    p_guard.sim_settings.enable_depol = False
    p_guard.sim_settings.enable_zz = False
    out_guard = execute_project_simulation(p_guard)
    assert out_guard["primary_backend"] == "trajectory"
    assert len(out_guard["warnings"]) >= 1


def test_large_scale_optimizer_parallel_trajectory_and_log_depth_ghz() -> None:
    """
    Verify P0-P2 optimization pass:
      1. 60-qubit QuantumCircuitOptimizer compiles without dense 2^60 OOM and computes critical-path duration
      2. 12-qubit Haar-probe equivalence check verifies exact native compilation fidelity > 1 - 1e-9
      3. Multi-threaded TrajectoryBackend produces deterministic bit-identical noisy results on 12Q
      4. 32-qubit Log-Depth Non-Crossing Binary-Tree GHZ uses <= 6 gate columns and outperforms linear chain under T1/T2 noise
    """
    from quantum_solver.optimizer.compiler import evaluate_circuit_metrics
    from quantum_solver.studio import (
        CircuitOpModel,
        NlCircuitRequest,
        QubitSpec,
        build_preset_projects,
        execute_project_simulation,
        generate_circuit_from_nl,
    )

    # 1. Critical-path parallel duration on 60 parallel SX gates (20 ns each -> 20 ns total, NOT 1200 ns!)
    hw_60 = HardwareModel.superconducting_processor(num_qubits=60, t_1q_ns=20.0, t_2q_ns=40.0)
    c60_par = QuantumCircuit(60, name="Parallel60Q")
    for q in range(60):
        c60_par.sx(q)
    m_par = evaluate_circuit_metrics(c60_par, hw_60.native_gate_set, hw_60)
    assert abs(m_par.duration_ns - 20.0) < 1e-6

    # Compile a 60Q redundant circuit (T + TDG cancels, CNOT -> CZ + 1Q) in milliseconds
    c60 = QuantumCircuit(60, name="Redundant60Q")
    c60.h(0)
    for q in range(59):
        c60.cnot(q, q + 1).t(q).tdg(q)
    opt_60 = QuantumCircuitOptimizer(hardware=hw_60)
    rep_60 = opt_60.optimize_circuit(c60, evaluate_noisy_fidelity=False)
    assert rep_60.unitary_fidelity == 1.0
    assert rep_60.optimized_metrics.num_2q_gates == 59
    assert "omitted for large circuit" in rep_60.summary()

    # 2. 12-qubit Haar-probe equivalence verification
    hw_12 = HardwareModel.neutral_atom_processor(num_qubits=12)
    c12 = QuantumCircuit(12, name="Target12Q")
    for q in range(12):
        c12.rx(q, 0.3 + 0.1 * q).rz(q, 0.5)
    for q in range(11):
        c12.cnot(q, q + 1)
    rep_12 = QuantumCircuitOptimizer(hardware=hw_12).optimize_circuit(c12, evaluate_noisy_fidelity=False)
    assert rep_12.unitary_fidelity > 1.0 - 1e-9

    # 3. Deterministic multi-threaded TrajectoryBackend on 12Q noisy circuit
    solver = QuantumSolver(hardware=hw_12)
    r_mt1 = solver.trajectory_backend.run(c12, hardware=hw_12, shots=2048, seed=2026, num_trajectories=8)
    r_mt2 = solver.trajectory_backend.run(c12, hardware=hw_12, shots=2048, seed=2026, num_trajectories=8)
    assert r_mt1.final_fidelity == r_mt2.final_fidelity
    assert r_mt1.final_purity == r_mt2.final_purity
    assert r_mt1.measurement_counts == r_mt2.measurement_counts

    # 4. 32-qubit Log-Depth Non-Crossing Binary-Tree GHZ vs Linear Chain GHZ
    base = build_preset_projects()[0]
    nl_tree = generate_circuit_from_nl(
        NlCircuitRequest(prompt="生成 32 比特对数深度二叉树 GHZ 态并测量", project=base)
    )
    gate_ops = [op for op in nl_tree["operations"] if op["kind"] == "gate"]
    max_gate_col = max(op["step"] for op in gate_ops)
    # 1 H gate (step 0) + log2(32) = 5 non-crossing parallel CNOT layers (steps 1..5) => max_gate_col == 5!
    assert max_gate_col == 5, f"Expected 6 visual gate columns (0..5), got 0..{max_gate_col}"

    p_tree = base.model_copy(deep=True)
    p_tree.qubits = [QubitSpec(**q) for q in nl_tree["qubits"]]
    p_tree.operations = [CircuitOpModel(**o) for o in nl_tree["operations"]]
    p_tree.sim_settings.backend = "stim"
    p_tree.sim_settings.shots = 8000

    nl_chain = generate_circuit_from_nl(
        NlCircuitRequest(prompt="生成 32 比特 GHZ 态并测量", project=base)
    )
    p_chain = p_tree.model_copy(deep=True)
    p_chain.operations = [CircuitOpModel(**o) for o in nl_chain["operations"]]

    sim_tree = execute_project_simulation(p_tree)["results"]["stim"]
    sim_chain = execute_project_simulation(p_chain)["results"]["stim"]
    coh_tree = sim_tree["final_probabilities"].get("0" * 32, 0.0) + sim_tree["final_probabilities"].get("1" * 32, 0.0)
    coh_chain = sim_chain["final_probabilities"].get("0" * 32, 0.0) + sim_chain["final_probabilities"].get("1" * 32, 0.0)
    assert sim_tree["total_duration_ns"] < sim_chain["total_duration_ns"] * 0.35
    assert coh_tree > coh_chain


def test_stim_adjoint_fidelity_sparse_qutip_and_nested_composite_gates() -> None:
    """
    Verify second-pass physics & studio enhancements:
      1. StimBackend adjoint-uncomputation exact state fidelity on 25Q 2D Cluster state (F=1.0)
         and phase-flip error sensitivity on GHZ state (F ~ 0.85 under 15% Z error)
      2. Sparse CSR QuTiPBackend on 6Q noisy circuit matching KrausBackend closely
      3. Nested composite custom gate containing another custom gate (SQISWAP) validates & simulates cleanly
    """
    from quantum_solver.studio import (
        CircuitOpModel,
        CustomGateDef,
        NlCircuitRequest,
        QubitSpec,
        build_preset_projects,
        execute_project_simulation,
        generate_circuit_from_nl,
    )
    from quantum_solver.studio.custom_gates import (
        evaluate_custom_gate_unitary,
        validate_unitary_matrix,
    )

    solver = QuantumSolver()
    base = build_preset_projects()[0]

    # 1a. 25-qubit 2D Cluster state (exponentially flat computational basis -> adjoint uncompute yields exact F=1.0!)
    nl_c25 = generate_circuit_from_nl(
        NlCircuitRequest(prompt="生成 25 比特二维网格 Cluster 簇态线路", project=base)
    )
    p_c25 = base.model_copy(deep=True)
    p_c25.qubits = [QubitSpec(**q) for q in nl_c25["qubits"]]
    p_c25.operations = [CircuitOpModel(**o) for o in nl_c25["operations"]]
    p_c25.sim_settings.backend = "stim"
    p_c25.sim_settings.enable_t1_t2 = False
    p_c25.sim_settings.enable_depol = False
    p_c25.sim_settings.enable_zz = False
    p_c25.sim_settings.enable_readout = False
    sim_c25 = execute_project_simulation(p_c25)["results"]["stim"]
    assert sim_c25["final_fidelity"] == 1.0

    # 1b. 4Q GHZ with 15% pure Z phase-flip error on q0 -> exact quantum fidelity is 0.85 (classical basis is blind!)
    c_z = QuantumCircuit(4, name="GHZ_PhaseFlip")
    c_z.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)
    c_z.add_noise(NoiseChannel.phase_flip(0, 0.15))
    c_z.measure_all()
    res_stim_z = solver.solve(c_z, backend="stim", hardware=None, shots=4096, seed=77)
    res_kraus_z = solver.solve(c_z, backend="kraus", hardware=None, shots=4096, seed=77)
    assert abs((res_stim_z.final_fidelity or 0.0) - (res_kraus_z.final_fidelity or 0.0)) < 0.03
    assert 0.82 < (res_stim_z.final_fidelity or 0.0) < 0.88
    # Intermediate step fidelities interpolate smoothly from 1.0 down to final_fidelity
    assert res_stim_z.steps[0].state_fidelity == 1.0
    assert (res_stim_z.steps[2].state_fidelity or 0.0) < 1.0

    # 2. Sparse CSR QuTiPBackend on 6-qubit noisy circuit vs KrausBackend
    hw_6 = HardwareModel.neutral_atom_processor(num_qubits=6, depol_1q=0.001, depol_2q=0.004)
    c6 = QuantumCircuit(6, name="SparseQuTiP_6Q")
    for q in range(6):
        c6.add_gate("RXY", (q,), (math.pi / 2.0, -math.pi / 2.0), layer=0)
    for q in range(0, 6, 2):
        c6.add_gate("CZ", (q, q + 1), layer=1)
    c6.rz(0, 0.4)
    c6.measure_all()
    r_qt6 = solver.solve(c6, backend="qutip", hardware=hw_6, shots=2048, seed=5)
    r_kr6 = solver.solve(c6, backend="kraus", hardware=hw_6, shots=2048, seed=5)
    assert abs((r_qt6.final_fidelity or 0.0) - (r_kr6.final_fidelity or 0.0)) < 0.015

    # 3. Nested composite custom gate referencing existing custom gate 'SQISWAP'
    p_nested = build_preset_projects()[1].model_copy(deep=True)  # contains SQISWAP custom gate
    cg_map = {cg.name.upper(): cg for cg in p_nested.custom_gates}
    nested_macro = CustomGateDef(
        id="cg_nested_macro",
        name="NESTED_MACRO",
        display_label="NestM",
        num_qubits=2,
        definition_type="composite",
        sub_operations=[
            CircuitOpModel(id="m_sub1", step=0, kind="gate", name="H", qubits=[0], params=[]),
            CircuitOpModel(id="m_sub2", step=1, kind="gate", name="SQISWAP", qubits=[0, 1], params=[]),
        ],
    )
    U_nest, _ = evaluate_custom_gate_unitary(nested_macro, custom_gate_map=cg_map)
    ok_u, err_u, _ = validate_unitary_matrix(U_nest)
    assert ok_u and err_u < 1e-10

    p_nested.custom_gates.append(nested_macro)
    p_nested.operations = [
        CircuitOpModel(id="op_nm", step=0, kind="gate", name="NESTED_MACRO", qubits=[0, 1], params=[]),
        CircuitOpModel(id="op_m", step=1, kind="measure", name="MEASURE", qubits=[0, 1], params=[]),
    ]
    sim_nest = execute_project_simulation(p_nested)
    assert sim_nest["results"][sim_nest["primary_backend"]]["final_fidelity"] > 0.96


def test_openqasm_bidirectional_hamiltonian_sweep_and_flagship_presets() -> None:
    """
    Verify v2.1 polish upgrades:
      1. Bidirectional OpenQASM 3.0 export -> import round-trip (preserving RXY, SQISWAP, and symbolic variables)
      2. OpenQASM 2.0 Qiskit-style `qreg`, register broadcast `h q;`, and custom `gate` macro expansion
      3. Variational Pauli Hamiltonian `<H>` expectation in 1D & 2D sweeps across Kraus & Trajectory (16Q VQE)
      4. All 7 presets (including 16Q Ising VQE, 17Q Rotated Surface Code, 32Q Log-Depth GHZ) simulate cleanly
    """
    from quantum_solver.studio import (
        SweepRequest,
        build_preset_projects,
        dsl_to_project_ops,
        execute_project_simulation,
        execute_project_sweep,
        openqasm_to_project_ops,
        parse_sparse_pauli_terms,
        project_to_openqasm3,
    )

    presets = build_preset_projects()
    assert len(presets) == 7

    # 1. Round-trip OpenQASM 3.0 on Preset 0 (Rydberg_GHZ with RXY(theta, 0) and SQISWAP)
    p0 = presets[0].model_copy(deep=True)
    qasm3 = project_to_openqasm3(p0)
    assert "float[64] theta" in qasm3
    rt_q, rt_ops, rt_vars = openqasm_to_project_ops(qasm3, p0)
    assert len(rt_q) == 4
    assert "theta" in rt_vars
    assert any(op.name == "RXY" and op.params[0] == "theta" for op in rt_ops)
    assert any(op.name == "SQISWAP" for op in rt_ops)

    # 2. OpenQASM 2.0 with custom `gate` macro + register broadcast + auto-detection via `dsl_to_project_ops`
    qasm2_src = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate bell_pair a, b { h a; cx a, b; }
    qreg q[3];
    creg c[3];
    bell_pair q[0], q[1];
    rx(pi/2) q[2];
    measure q -> c;
    """
    q2_qubits, q2_ops, _ = dsl_to_project_ops(qasm2_src, p0)
    assert len(q2_qubits) == 3
    gate_names = [op.name for op in q2_ops if op.kind == "gate"]
    assert gate_names == ["H", "CNOT", "RX"]
    assert abs(float(q2_ops[2].params[0]) - math.pi / 2.0) < 1e-5

    # 3. Sparse & dense Pauli Hamiltonian expectation in 1D & 2D parameter sweeps
    p_rabi = presets[3].model_copy(deep=True)  # 1Q: RX(theta) -> RZ(phi) -> SX -> M
    p_rabi.operations = [
        o for o in p_rabi.operations if o.name == "RX" or o.kind == "measure"
    ]
    p_rabi.sim_settings.enable_t1_t2 = False
    p_rabi.sim_settings.enable_depol = False
    p_rabi.sim_settings.enable_zz = False
    p_rabi.sim_settings.enable_readout = False
    # For state RX(theta)|0>, <Z0> = cos(theta), so H = -1.0*Z0 + 0.5*I has <H> = 0.5 - cos(theta)
    sw_1d = execute_project_sweep(
        SweepRequest(
            project=p_rabi,
            mode="1d",
            param_x="theta",
            x_start=0.0,
            x_stop=math.pi,
            x_points=5,
            hamiltonian_str="-1.0*Z0 + 0.5*I",
            backend="kraus",
        )
    )
    assert sw_1d["energies"] is not None and len(sw_1d["energies"]) == 5
    assert abs(sw_1d["energies"][0] - (-0.5)) < 1e-4  # theta=0 -> 0.5 - 1 = -0.5
    assert abs(sw_1d["energies"][-1] - 1.5) < 1e-4    # theta=pi -> 0.5 - (-1) = 1.5

    # 4. Execute all 3 new flagship presets (16Q VQE + 2D energy landscape, 17Q Surface Code, 32Q Log-Depth GHZ)
    vqe_16q, surf_17q, ghz_32q = presets[4], presets[5], presets[6]
    assert len(vqe_16q.qubits) == 16
    assert len(surf_17q.qubits) == 17
    assert len(ghz_32q.qubits) == 32

    terms_16q = parse_sparse_pauli_terms("-1.0*Z0*Z1 - 1.0*Z1*Z2 - 0.5*X0 - 0.5*X1", 16)
    assert len(terms_16q) == 4
    sw_vqe_2d = execute_project_sweep(
        SweepRequest(
            project=vqe_16q,
            mode="2d",
            param_x="theta",
            x_start=0.0,
            x_stop=math.pi,
            x_points=3,
            param_y="gamma",
            y_start=0.0,
            y_stop=math.pi / 2.0,
            y_points=3,
            hamiltonian_str="-1.0*Z0*Z1 - 0.5*X0 - 0.5*X1",
            backend="trajectory",
        )
    )
    assert sw_vqe_2d["energy_grid"] is not None
    assert len(sw_vqe_2d["energy_grid"]) == 3 and len(sw_vqe_2d["energy_grid"][0]) == 3

    res_surf = execute_project_simulation(surf_17q)["results"]["stim"]
    assert res_surf["num_qubits"] == 17 and 0.78 < res_surf["final_fidelity"] < 0.95

    ghz_ideal = ghz_32q.model_copy(deep=True)
    ghz_ideal.sim_settings.enable_t1_t2 = False
    ghz_ideal.sim_settings.enable_depol = False
    ghz_ideal.sim_settings.enable_zz = False
    ghz_ideal.sim_settings.enable_readout = False
    res_ghz32 = execute_project_simulation(ghz_ideal)["results"]["stim"]
    assert res_ghz32["final_fidelity"] == 1.0
    p_zero32 = res_ghz32["final_probabilities"].get("0" * 32, 0.0)
    p_one32 = res_ghz32["final_probabilities"].get("1" * 32, 0.0)
    assert 0.45 < p_zero32 < 0.55 and 0.45 < p_one32 < 0.55
    # Exact GF(2) stabilizer entanglement diagnostics on 32-qubit GHZ: S_A = 1.0 bit, I(q0:q31) = 1.0 bit
    assert abs((res_ghz32["bipartite_entropy"] or 0.0) - 1.0) < 1e-5
    assert abs(res_ghz32["mutual_information"][0][31] - 1.0) < 1e-5


def test_vqe_optimizer_mutual_info_and_cross_hardware_benchmark() -> None:
    """
    Verify v2.2 scientific upgrades:
      1. Closed-loop VQE (`execute_variational_optimization`) reaches exact 2Q TFIM ground-state energy -sqrt(2)
      2. Mutual Information Matrix I(q_i:q_j) = 2.0 bits & Bipartite Entropy S_A = 1.0 bit across QuTiP, Kraus, Trajectory & Stim
      3. Cross-Hardware Benchmark (`execute_cross_hardware_benchmark`) & 16Q noisy optimizer evaluation via Trajectory
    """
    from quantum_solver.studio import (
        BenchmarkHardwareRequest,
        CircuitOpModel,
        OptimizeRequest,
        VariationalOptimizeRequest,
        build_preset_projects,
        execute_cross_hardware_benchmark,
        execute_project_optimization,
        execute_project_simulation,
        execute_variational_optimization,
    )

    presets = build_preset_projects()

    # 1. 2Q Transverse-Field Ising Model H = -1.0*Z0*Z1 - 0.5*X0 - 0.5*X1 via 2-param QAOA/HVA ansatz:
    # Exact ground-state eigenvalue is E_0 = -sqrt(J^2 + h^2) = -sqrt(1 + 1) = -sqrt(2) ≈ -1.41421356
    p_vqe2 = presets[2].model_copy(deep=True)  # 2Q template
    p_vqe2.sim_settings.enable_t1_t2 = False
    p_vqe2.sim_settings.enable_depol = False
    p_vqe2.sim_settings.enable_zz = False
    p_vqe2.sim_settings.enable_readout = False
    p_vqe2.variables = {"gamma": 0.2, "beta": 0.2}
    p_vqe2.operations = [
        CircuitOpModel(id="v_h0", step=0, kind="gate", name="H", qubits=[0], params=[]),
        CircuitOpModel(id="v_h1", step=0, kind="gate", name="H", qubits=[1], params=[]),
        CircuitOpModel(id="v_zz", step=1, kind="gate", name="RZZ", qubits=[0, 1], params=["gamma"]),
        CircuitOpModel(id="v_x0", step=2, kind="gate", name="RX", qubits=[0], params=["beta"]),
        CircuitOpModel(id="v_x1", step=2, kind="gate", name="RX", qubits=[1], params=["beta"]),
        CircuitOpModel(id="vm", step=3, kind="measure", name="MEASURE", qubits=[0, 1], params=[]),
    ]
    vqe_res = execute_variational_optimization(
        VariationalOptimizeRequest(
            project=p_vqe2,
            hamiltonian_str="-1.0*Z0*Z1 - 0.5*X0 - 0.5*X1",
            method="COBYLA",
            max_iter=60,
            backend="trajectory",
        )
    )
    assert vqe_res["optimal_energy"] < vqe_res["initial_energy"] - 0.35
    assert abs(vqe_res["optimal_energy"] - (-math.sqrt(2.0))) < 1e-3

    # 2. Exact Bell state mutual information I(0:1) = 2.0 bits and S_A = 1.0 bit across all 4 engines!
    p_bell = p_vqe2.model_copy(deep=True)
    p_bell.operations = [
        CircuitOpModel(id="b_h", step=0, kind="gate", name="H", qubits=[0], params=[]),
        CircuitOpModel(id="b_cx", step=1, kind="gate", name="CNOT", qubits=[0, 1], params=[]),
        CircuitOpModel(id="b_m", step=2, kind="measure", name="MEASURE", qubits=[0, 1], params=[]),
    ]
    p_bell.sim_settings.backend = "compare"
    bell_cmp = execute_project_simulation(p_bell)["results"]
    for bk in ("qutip", "kraus", "trajectory", "stim"):
        r_bk = bell_cmp[bk]
        assert abs((r_bk["bipartite_entropy"] or 0.0) - 1.0) < 1e-4
        assert abs(r_bk["mutual_information"][0][1] - 2.0) < 1e-4
        assert abs(r_bk["mutual_information"][0][0] - 1.0) < 1e-4

    # 3. Cross-Hardware Shootout across all 4 hardware profiles + 16Q noisy optimizer check
    bench = execute_cross_hardware_benchmark(BenchmarkHardwareRequest(project=presets[1]))
    assert len(bench["entries"]) >= 4
    assert bench["recommended_hardware_id"] in {e["hardware_id"] for e in bench["entries"]}

    p_16q_noisy = presets[4].model_copy(deep=True)  # 16Q VQE
    p_16q_noisy.sim_settings.enable_depol = True
    opt_16 = execute_project_optimization(
        OptimizeRequest(project=p_16q_noisy, evaluate_noisy_fidelity=True)
    )
    assert opt_16["unitary_fidelity"] > 0.99999
    assert opt_16["noisy_fidelity_optimized"] is not None


def test_dag_consolidation_zne_mitigation_and_parameter_shift_vqe() -> None:
    """
    Verify v2.3.0 upgrades:
      1. DAG-aware 2Q block consolidation across disjoint qubits ([CNOT(0,1), H(2), CNOT(0,1)] -> 0 2Q gates)
         + non-commuting barrier safety ([CNOT(0,1), CNOT(1,2), CNOT(0,1)] preserves exact unitary)
         + CNOT commutation cancellation ([CNOT(0,1), RZ(pi/3, 0), CNOT(0,1)] -> 0 2Q gates);
      2. Zero-Noise Extrapolation (ZNE) quantum error mitigation reducing noisy <Z> & <H> bias;
      3. Analytical Parameter-Shift + Adam VQE (including shared multi-gate chain-rule parameters).
    """
    import math
    from quantum_solver.studio import (
        CircuitOpModel,
        VariationalOptimizeRequest,
        ZneRequest,
        build_preset_projects,
        execute_variational_optimization,
        execute_zne_mitigation,
    )

    hw_sc = HardwareModel.superconducting_processor(num_qubits=3)
    opt_sc = QuantumCircuitOptimizer(hardware=hw_sc)

    # 1a. Disjoint-wire interleaved 2Q cancellation: [CNOT(0,1), H(2), CNOT(0,1)] -> 0 2Q gates
    c_disjoint = QuantumCircuit(3)
    c_disjoint.cnot(0, 1).h(2).cnot(0, 1)
    r_disjoint = opt_sc.optimize_circuit(c_disjoint, evaluate_noisy_fidelity=False)
    assert r_disjoint.optimized_metrics.num_2q_gates == 0
    assert r_disjoint.unitary_fidelity > 0.999999

    # 1b. Overlapping 2Q barrier safety: [CNOT(0,1), CNOT(1,2), CNOT(0,1)] must preserve exact unitary
    c_barrier = QuantumCircuit(3)
    c_barrier.cnot(0, 1).cnot(1, 2).cnot(0, 1)
    r_barrier = opt_sc.optimize_circuit(c_barrier, evaluate_noisy_fidelity=False)
    assert r_barrier.unitary_fidelity > 0.999999

    # 1c. CNOT control-Z commutation + KAK cancellation: [CNOT(0,1), RZ(0.7, 0), CNOT(0,1)] -> 0 2Q gates
    hw_cnot = HardwareModel.superconducting_processor(num_qubits=2, primary_2q_gate="CNOT")
    opt_cnot = QuantumCircuitOptimizer(hardware=hw_cnot)
    c_comm = QuantumCircuit(2)
    c_comm.cnot(0, 1).rz(0, 0.7).cnot(0, 1)
    r_comm = opt_cnot.optimize_circuit(c_comm, evaluate_noisy_fidelity=False)
    assert r_comm.optimized_metrics.num_2q_gates == 0
    assert r_comm.unitary_fidelity > 0.999999

    # 2. Zero-Noise Extrapolation (ZNE) reduces expectation error on a noisy circuit
    presets = build_preset_projects()
    p_zne = presets[2].model_copy(deep=True)  # 2Q noisy Bell/entangling template
    p_zne.sim_settings.enable_depol = True
    p_zne.sim_settings.enable_t1_t2 = True
    p_zne.operations = [
        CircuitOpModel(id="z_x0", step=0, kind="gate", name="X", qubits=[0], params=[]),
        CircuitOpModel(id="z_cx1", step=1, kind="gate", name="CNOT", qubits=[0, 1], params=[]),
        CircuitOpModel(id="z_cx2", step=2, kind="gate", name="CNOT", qubits=[0, 1], params=[]),
        CircuitOpModel(id="z_cx3", step=3, kind="gate", name="CNOT", qubits=[0, 1], params=[]),
        CircuitOpModel(id="z_m", step=4, kind="measure", name="MEASURE", qubits=[0, 1], params=[]),
    ]
    zne_res = execute_zne_mitigation(
        ZneRequest(
            project=p_zne,
            scale_factors=[1.0, 2.0, 3.0],
            observed_qubit=1,
            hamiltonian_str="Z0*Z1",
            backend="kraus",
        )
    )
    # Ideal state is |11>, so <Z_1>_ideal = -1.0 and <Z0 Z1>_ideal = +1.0
    assert abs(zne_res["z_ideal"] - (-1.0)) < 1e-4
    err_raw_z = abs(zne_res["z_unmitigated"] - zne_res["z_ideal"])
    err_rich_z = abs(zne_res["z_zne_richardson"] - zne_res["z_ideal"])
    assert err_rich_z < err_raw_z * 0.5, f"ZNE should cut <Z> bias by >50%: raw={err_raw_z}, zne={err_rich_z}"

    assert zne_res["energy_ideal"] is not None
    err_raw_e = abs(zne_res["energy_unmitigated"] - zne_res["energy_ideal"])
    err_rich_e = abs(zne_res["energy_zne_richardson"] - zne_res["energy_ideal"])
    assert err_rich_e < err_raw_e * 0.5

    # 3. Exact Parameter-Shift + Adam VQE on 2Q TFIM (with shared parameter 'beta' across 2 RX gates!)
    p_ps = presets[2].model_copy(deep=True)
    p_ps.sim_settings.enable_t1_t2 = False
    p_ps.sim_settings.enable_depol = False
    p_ps.sim_settings.enable_zz = False
    p_ps.sim_settings.enable_readout = False
    p_ps.variables = {"gamma": 0.25, "beta": 0.25}
    p_ps.operations = [
        CircuitOpModel(id="ps_h0", step=0, kind="gate", name="H", qubits=[0], params=[]),
        CircuitOpModel(id="ps_h1", step=0, kind="gate", name="H", qubits=[1], params=[]),
        CircuitOpModel(id="ps_zz", step=1, kind="gate", name="RZZ", qubits=[0, 1], params=["gamma"]),
        CircuitOpModel(id="ps_x0", step=2, kind="gate", name="RX", qubits=[0], params=["beta"]),
        CircuitOpModel(id="ps_x1", step=2, kind="gate", name="RX", qubits=[1], params=["beta"]),
        CircuitOpModel(id="ps_m", step=3, kind="measure", name="MEASURE", qubits=[0, 1], params=[]),
    ]
    ps_res = execute_variational_optimization(
        VariationalOptimizeRequest(
            project=p_ps,
            hamiltonian_str="-1.0*Z0*Z1 - 0.5*X0 - 0.5*X1",
            method="Parameter-Shift",
            max_iter=45,
            backend="trajectory",
        )
    )
    assert ps_res["method"] == "Parameter-Shift"
    assert abs(ps_res["optimal_energy"] - (-math.sqrt(2.0))) < 2e-3


def test_gate_benchmarking_rb_irb_allxy_and_gst() -> None:
    """
    Verify v2.5.0 Quantum Gate Characterization & Benchmarking protocols:
      1. 1Q & 2Q Standard + Interleaved Randomized Benchmarking (SPAM-insensitivity & IRB gate fidelity);
      2. AllXY 21-sequence pulse diagnostic (ideal staircase vs. +5% over-rotation sawtooth detection);
      3. Gate Set Tomography (SPAM-free PTM reconstruction, exact diamond distance sin(|dtheta|/2),
         and Wallman unitarity coherent/incoherent separation).
    """
    import math
    from quantum_solver.studio import (
        GateBenchmarkRequest,
        build_preset_projects,
        execute_gate_benchmark,
    )

    presets = build_preset_projects()
    p_base = presets[0].model_copy(deep=True)

    # 1a. SPAM-insensitivity of 1Q RB: 8% readout error reduces A_spam, NOT clifford_fidelity!
    p_spam = p_base.model_copy(deep=True)
    p_spam.sim_settings.enable_t1_t2 = False
    p_spam.sim_settings.enable_depol = False
    p_spam.sim_settings.enable_readout = True
    for hp in p_spam.hardware_profiles:
        hp.readout_err = 0.08
    rb_spam = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_spam,
            protocol="rb",
            rb_num_qubits=1,
            rb_lengths=[1, 4, 16, 64],
            rb_num_seeds=10,
            interleaved_gate="SX",
            shots=0,
        )
    )
    assert rb_spam["clifford_fidelity"] > 0.9999
    assert rb_spam["a_spam"] <= 0.421  # SPAM visibility reduced from 0.50 to 0.42 by 8% readout error
    assert rb_spam["interleaved"] is not None
    assert rb_spam["interleaved"]["gate_fidelity"] > 0.9999

    # 1b. Noisy 2Q RB + Interleaved CZ on Rydberg array
    rb_2q = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_base,
            protocol="rb",
            rb_num_qubits=2,
            rb_lengths=[1, 2, 4, 8, 16],
            rb_num_seeds=8,
            interleaved_gate="CZ",
            shots=0,
        )
    )
    assert rb_2q["num_qubits"] == 2
    assert 0.96 < rb_2q["clifford_fidelity"] < 0.999
    assert 0.985 < rb_2q["interleaved"]["gate_fidelity"] < 0.9995
    assert len(rb_2q["circuit_samples"]) >= 2

    # 2. AllXY 21-sequence coherent pulse diagnostic
    p_clean = p_base.model_copy(deep=True)
    p_clean.sim_settings.enable_t1_t2 = False
    p_clean.sim_settings.enable_depol = False
    p_clean.sim_settings.enable_readout = False
    axy_ideal = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_clean, protocol="allxy", shots=0)
    )
    assert len(axy_ideal["sequences"]) == 21
    assert axy_ideal["rms_error_z"] < 1e-6
    assert axy_ideal["severity"] == "good"

    axy_over = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_clean, protocol="allxy", amp_error_pct=5.0, shots=0)
    )
    assert axy_over["rms_error_z"] > 0.08
    assert 3.5 < axy_over["estimated_amp_error_pct"] < 6.5
    assert axy_over["severity"] == "warn"

    # 3. Gate Set Tomography (GST): SPAM-free PTM + exact coherent diamond distance sin(dtheta/2)
    # Inject +4% over-rotation WITH 5% readout SPAM error -> GST still isolates pure coherent gate error!
    gst_coh = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_spam,
            protocol="gst",
            amp_error_pct=4.0,
            gst_max_length=4,
            shots=0,
        )
    )
    gx_rep = next(g for g in gst_coh["gates"] if g["gate_id"] == "Gx")
    # For pi pulse with +4% over-rotation: dtheta = 0.04 * pi, theoretical 0.5*||Delta E||_diam = sin(0.02 * pi) = 0.06279
    theo_diam = math.sin(0.02 * math.pi)
    assert abs(gx_rep["diamond_distance"] - theo_diam) < 0.004
    assert gx_rep["unitarity"] > 0.998
    assert gx_rep["coherent_ratio"] > 0.95
    assert gst_coh["spam_prep_fidelity"] > 0.98
    assert gst_coh["spam_readout_fidelity"] < 0.95  # GST simultaneously catches the 8% readout SPAM!
    assert any(g["gate_id"] == "Gh" for g in gst_coh["gates"]), "GST should include 6th gate Gh (Hadamard)"
    assert len(gst_coh["circuit_samples"]) >= 12, "GST should expose inline circuits for all 6 gates (L=1 & germ) + SPAM"


def test_noise_error_budget_and_2q_linear_xeb() -> None:
    """
    Verify v2.6 Noise Error Budget ablation (`execute_noise_budget`) and 2Q Linear XEB
    (`execute_gate_benchmark(..., protocol='xeb')`) including non-Clifford √iSWAP.
    """
    from quantum_solver.studio import (
        GateBenchmarkRequest,
        build_preset_projects,
        execute_gate_benchmark,
        execute_noise_budget,
    )

    p_base = build_preset_projects()[0]

    # 1. Noise Error Budget isolates channels and sums shares to 100%
    nb = execute_noise_budget(p_base)
    assert 0.85 < nb["full_fidelity"] < 1.0
    assert len(nb["contributions"]) >= 5
    total_share = sum(c["share_pct"] for c in nb["contributions"])
    assert abs(total_share - 100.0) < 0.25
    assert nb["contributions"][0]["infidelity"] >= nb["contributions"][-1]["infidelity"]

    # 2. Ideal vs Noisy 2Q Linear XEB for non-Clifford √iSWAP
    p_ideal = p_base.model_copy(deep=True)
    p_ideal.sim_settings.enable_t1_t2 = False
    p_ideal.sim_settings.enable_depol = False
    p_ideal.sim_settings.enable_zz = False
    p_ideal.sim_settings.enable_readout = False
    xeb_ideal = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_ideal,
            protocol="xeb",
            rb_lengths=[1, 2, 4, 8, 12],
            rb_num_seeds=10,
            xeb_entangler="SQISWAP",
            shots=0,
        )
    )
    assert xeb_ideal["protocol"] == "xeb"
    assert xeb_ideal["cycle_polarization"] > 0.999
    assert xeb_ideal["entangler_fidelity"] > 0.999
    assert len(xeb_ideal["speckle_ideal"]) >= 20

    xeb_noisy = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_base,
            protocol="xeb",
            rb_lengths=[1, 2, 4, 8, 16],
            rb_num_seeds=12,
            xeb_entangler="SQISWAP",
            shots=0,
        )
    )
    assert 0.95 < xeb_noisy["cycle_polarization"] < 0.9995
    assert 0.96 < xeb_noisy["entangler_fidelity"] < 1.0
    assert len(xeb_noisy["sample_operations"]) > 0
    assert len(xeb_noisy["circuit_samples"]) >= 2


def test_rabi_drive_hierarchy_rise_fall_edges_and_basic_calibrations() -> None:
    """
    Verify v2.7 features:
      1. 3-tier Rabi frequency inheritance (hardware -> per-qubit -> per-gate) & fixed_rabi vs fixed_duration
      2. Flat-top & linear-trapezoid rise/fall edges (`rise_ns`, `fall_ns`) reaching >99.99% ideal fidelity in QuTiP
      3. Basic calibration experiments (`rabi`, `ramsey`, `t1`) fitting Ω_R, T2*, frequency offset, and T1
    """
    from quantum_solver.studio import (
        CircuitOpModel,
        GateBenchmarkRequest,
        SweepRequest,
        build_preset_projects,
        execute_gate_benchmark,
        execute_project_simulation,
        execute_project_sweep,
    )

    p = build_preset_projects()[0].model_copy(deep=True)
    p.qubits = p.qubits[:2]
    p.sim_settings.backend = "qutip"
    p.sim_settings.enable_t1_t2 = False
    p.sim_settings.enable_depol = False
    p.sim_settings.enable_zz = False
    p.sim_settings.enable_readout = False

    # 1. Fixed Rabi mode + Flat-top raised-cosine edges (rise_ns=15, fall_ns=15) + 3-tier override
    for hp in p.hardware_profiles:
        hp.drive_mode = "fixed_rabi"
        hp.rabi_mhz = 2.0  # t_pi = 250 + 15 = 265 ns
        hp.pulse_shape = "rectangular"
        hp.rise_ns = 15.0
        hp.fall_ns = 15.0

    p.qubits[1].rabi_mhz = 4.0  # q1 override: t_pi = 125 + 15 = 140 ns
    p.operations = [
        CircuitOpModel(id="g0", step=0, kind="gate", name="X", qubits=[0]),
        CircuitOpModel(id="g1", step=1, kind="gate", name="X", qubits=[1]),
        CircuitOpModel(id="g2", step=2, kind="gate", name="SX", qubits=[0], rabi_mhz=5.0),  # gate override: t_pi/2 = 50 + 15 = 65 ns
    ]
    sim = execute_project_simulation(p)["results"]["qutip"]
    # Total duration = 265 + 140 + 65 = 470 ns, and state should be |+1> with F > 0.9999
    assert abs(sim["total_duration_ns"] - 470.0) < 0.5
    assert sim["final_fidelity"] > 0.9999

    # Also verify trapezoid linear ramp edges integrate cleanly
    for hp in p.hardware_profiles:
        hp.pulse_shape = "trapezoid"
    sim_trap = execute_project_simulation(p)["results"]["qutip"]
    assert sim_trap["final_fidelity"] > 0.9999

    # 2. Rabi / Ramsey / T1 calibration protocols
    p_cal = build_preset_projects()[0]
    r_rabi = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_cal, protocol="rabi", amp_error_pct=4.0, shots=0)
    )
    omega_fit = next(x["value"] for x in r_rabi["params"] if x["key"] == "omega_fit")
    assert abs(omega_fit - 2.5 * 1.04) < 0.03
    assert len(r_rabi["writeback"]) >= 2
    assert len(r_rabi["circuit_samples"]) >= 3

    r_ramsey = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_cal, protocol="ramsey", ramsey_detuning_mhz=0.5, detuning_mhz=0.12, shots=0
        )
    )
    t2_fit = next(x["value"] for x in r_ramsey["params"] if x["key"] == "t2_star")
    freq_off = next(x["value"] for x in r_ramsey["params"] if x["key"] == "freq_offset")
    assert abs(t2_fit - 800.0) < 25.0
    assert abs(freq_off - 0.12) < 0.02

    r_t1 = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_cal, protocol="t1", shots=0)
    )
    t1_fit = next(x["value"] for x in r_t1["params"] if x["key"] == "t1")
    assert abs(t1_fit - 1000.0) < 25.0

    # 3. Sweeping hw:rabi_mhz works end-to-end
    sw = execute_project_sweep(
        SweepRequest(
            project=p,
            mode="1d",
            param_x="hw:rabi_mhz",
            x_start=1.0,
            x_stop=5.0,
            x_points=5,
            backend="kraus",
        )
    )
    assert len(sw["fidelities"]) == 5


def test_v28_drag_leakage_barrier_idle_readout_mitigation_echo_and_chevron() -> None:
    """
    Verify all Quantum Studio v2.8 physics & EDA upgrades:
      1. 3-Level Transmon |2> leakage & DRAG quadrature suppression (QuTiP + RB + simulate_3level_drag_pulse)
      2. Asymmetric readout confusion matrix (P(1|0) != P(0|1)) & tensor-product M^-1 readout mitigation
      3. Exact static ZZ crosstalk alignment across QuTiP, Kraus, and Trajectory solvers
      4. BARRIER & IDLE(tau) non-fusing optimization boundaries + OpenQASM 3.0 delay/barrier round-trip
      5. Trapezoid edge-integrated Rabi, 2D Rabi Chevron heatmap, and CPMG-N Spin Echo refocusing
    """
    import math
    from quantum_solver.core.gates import simulate_3level_drag_pulse
    from quantum_solver.studio import (
        CircuitOpModel,
        GateBenchmarkRequest,
        OptimizeRequest,
        build_preset_projects,
        execute_gate_benchmark,
        execute_project_optimization,
        execute_project_simulation,
        openqasm_to_project_ops,
        project_to_openqasm3,
    )

    # 1. 3-level DRAG suppresses |1> -> |2> leakage by > 4x on a fast 8 ns Transmon pi pulse
    _, _, lk_no_drag = simulate_3level_drag_pulse(
        theta=math.pi,
        phi=0.0,
        duration_ns=8.0,
        anharmonicity_ghz=-0.22,
        drag_beta=0.0,
        pulse_shape="cosine",
    )
    _, _, lk_with_drag = simulate_3level_drag_pulse(
        theta=math.pi,
        phi=0.0,
        duration_ns=8.0,
        anharmonicity_ghz=-0.22,
        drag_beta=0.5,
        pulse_shape="cosine",
    )
    assert lk_no_drag > 0.004
    assert lk_with_drag < 0.25 * lk_no_drag

    p_sc = build_preset_projects()[0].model_copy(deep=True)
    p_sc.qubits = p_sc.qubits[:2]
    p_sc.active_hardware_id = "hw_sc_cz"
    for hp in p_sc.hardware_profiles:
        if hp.id == "hw_sc_cz":
            hp.t_1q_ns = 8.0
            hp.anharmonicity_mhz = -220.0
            hp.drag_beta = 0.0
    p_sc.operations = [
        CircuitOpModel(id="x0", step=0, kind="gate", name="X", qubits=[0]),
        CircuitOpModel(id="x1", step=1, kind="gate", name="X", qubits=[0]),
        CircuitOpModel(id="m0", step=2, kind="measure", name="MEASURE", qubits=[0, 1]),
    ]
    p_sc.sim_settings.backend = "qutip"
    sim_nodrag = execute_project_simulation(p_sc)["results"]["qutip"]
    assert sim_nodrag["leakage_by_qubit"]["0"] > 1e-4

    rb_nodrag = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_sc, protocol="rb", rb_lengths=[1, 2, 4, 8, 16, 32], rb_num_seeds=10, shots=0)
    )
    for hp in p_sc.hardware_profiles:
        if hp.id == "hw_sc_cz":
            hp.drag_beta = 0.5
    rb_drag = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_sc, protocol="rb", rb_lengths=[1, 2, 4, 8, 16, 32], rb_num_seeds=10, shots=0)
    )
    assert rb_drag["epc"] < rb_nodrag["epc"]

    # 2. Asymmetric readout confusion P(1|0)=0.02, P(0|1)=0.15 + M^-1 mitigation on Bell state
    p_ro = build_preset_projects()[0].model_copy(deep=True)
    p_ro.qubits = p_ro.qubits[:2]
    for hp in p_ro.hardware_profiles:
        hp.readout_p1_given_0 = 0.02
        hp.readout_p0_given_1 = 0.15
    p_ro.sim_settings.backend = "kraus"
    p_ro.sim_settings.shots = 40000
    p_ro.sim_settings.enable_t1_t2 = False
    p_ro.sim_settings.enable_depol = False
    p_ro.sim_settings.enable_zz = False
    p_ro.sim_settings.enable_readout = True
    p_ro.operations = [
        CircuitOpModel(id="b0", step=0, kind="gate", name="H", qubits=[0]),
        CircuitOpModel(id="b1", step=1, kind="gate", name="CNOT", qubits=[0, 1]),
        CircuitOpModel(id="bm", step=2, kind="measure", name="MEASURE", qubits=[0, 1]),
    ]
    res_ro = execute_project_simulation(p_ro)["results"]["kraus"]
    assert res_ro["final_probabilities"]["11"] < 0.40  # heavily skewed by 15% P(0|1) on both qubits
    mit = res_ro["mitigated_probabilities"]
    assert mit is not None
    assert abs(mit["11"] - 0.5) < 0.025
    assert abs(mit["00"] - 0.5) < 0.025
    assert abs(sum(mit.values()) - 1.0) < 1e-4

    # 3. Static ZZ crosstalk exact alignment across QuTiP, Kraus, and Trajectory during IDLE(250ns)
    p_zz = build_preset_projects()[0].model_copy(deep=True)
    p_zz.qubits = p_zz.qubits[:2]
    for hp in p_zz.hardware_profiles:
        hp.zz_khz = 1000.0  # 1 MHz static ZZ -> pi/2 ZZ rotation in 250 ns!
        hp.topology_mode = "all"
        hp.anharmonicity_mhz = 0.0
    p_zz.sim_settings.enable_t1_t2 = False
    p_zz.sim_settings.enable_depol = False
    p_zz.sim_settings.enable_readout = False
    p_zz.sim_settings.enable_zz = True
    p_zz.operations = [
        CircuitOpModel(id="z0", step=0, kind="gate", name="H", qubits=[0], noise_mode="none"),
        CircuitOpModel(id="z1", step=0, kind="gate", name="H", qubits=[1], noise_mode="none"),
        CircuitOpModel(id="zi0", step=1, kind="gate", name="IDLE", qubits=[0], params=[250.0]),
        CircuitOpModel(id="zi1", step=1, kind="gate", name="IDLE", qubits=[1], params=[250.0]),
    ]
    fids = {}
    bxs = {}
    for bk in ("qutip", "kraus", "trajectory"):
        p_zz.sim_settings.backend = bk
        r_bk = execute_project_simulation(p_zz)["results"][bk]
        fids[bk] = r_bk["final_fidelity"]
        bxs[bk] = r_bk["steps"][-1]["bloch_vectors"]["0"][0]
    assert abs(fids["qutip"] - fids["kraus"]) < 2e-3
    assert abs(fids["kraus"] - fids["trajectory"]) < 1e-5
    assert abs(bxs["qutip"] - bxs["kraus"]) < 2e-3

    # 4. BARRIER blocks H*H cancellation + OpenQASM 3 delay/barrier round-trip
    p_bar = build_preset_projects()[0].model_copy(deep=True)
    p_bar.qubits = p_bar.qubits[:2]
    p_bar.operations = [
        CircuitOpModel(id="h_a", step=0, kind="gate", name="H", qubits=[0]),
        CircuitOpModel(id="bar", step=1, kind="gate", name="BARRIER", qubits=[0, 1]),
        CircuitOpModel(id="h_b", step=2, kind="gate", name="H", qubits=[0]),
        CircuitOpModel(id="idl", step=3, kind="gate", name="IDLE", qubits=[1], params=[150.0]),
        CircuitOpModel(id="m_all", step=4, kind="measure", name="MEASURE", qubits=[0, 1]),
    ]
    opt_bar = execute_project_optimization(OptimizeRequest(project=p_bar, evaluate_noisy_fidelity=False))
    assert opt_bar["optimized_metrics"]["num_1q_physical_pulses"] >= 2
    assert any(op["name"] == "BARRIER" for op in opt_bar["optimized_operations"])
    assert any(op["name"] == "IDLE" for op in opt_bar["optimized_operations"])

    qasm_txt = project_to_openqasm3(p_bar)
    assert "barrier" in qasm_txt
    assert "delay[150ns]" in qasm_txt
    _, rt_ops, _ = openqasm_to_project_ops(qasm_txt, p_bar)
    assert any(o.name == "BARRIER" for o in rt_ops)
    assert any(o.name == "IDLE" and abs(float(o.params[0]) - 150.0) < 1e-3 for o in rt_ops)

    # 5. Trapezoid edge-integrated Rabi + 2D Chevron + CPMG-4 Spin Echo
    p_cal = build_preset_projects()[0].model_copy(deep=True)
    for hp in p_cal.hardware_profiles:
        hp.pulse_shape = "trapezoid"
        hp.rise_ns = 15.0
        hp.fall_ns = 15.0
        hp.rabi_mhz = 3.0
    r_trap = execute_gate_benchmark(
        GateBenchmarkRequest(project=p_cal, protocol="rabi", amp_error_pct=-5.0, shots=0)
    )
    om_trap = next(x["value"] for x in r_trap["params"] if x["key"] == "omega_fit")
    assert abs(om_trap - 3.0 * 0.95) < 0.05

    r_chev = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_cal,
            protocol="chevron",
            detuning_mhz=1.1,
            chevron_detuning_max_mhz=5.0,
            chevron_freq_points=21,
            cal_points=35,
            shots=0,
        )
    )
    assert len(r_chev["z_grid"]) == 21
    assert len(r_chev["z_grid"][0]) == 35
    chev_off = next(x["value"] for x in r_chev["params"] if x["key"] == "freq_offset")
    assert abs(chev_off - 1.1) < 0.35

    # Set T2* << 2*T1 so CPMG-4 clearly extends coherence time
    for hp in p_cal.hardware_profiles:
        hp.t1_us = 200.0
        hp.t2_us = 35.0
    r_echo = execute_gate_benchmark(
        GateBenchmarkRequest(
            project=p_cal,
            protocol="echo",
            echo_num_pulses=4,
            detuning_mhz=0.75,
            shots=0,
        )
    )
    t2_echo_val = next(x["value"] for x in r_echo["params"] if x["key"] == "t2_echo")
    assert t2_echo_val > 35.0 * 1.4
    assert len(r_echo["echo_ref_ramsey"]) > 50
















