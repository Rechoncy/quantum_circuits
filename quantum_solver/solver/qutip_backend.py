"""
QuTiP Physical Master Equation & Pulse-Level Dynamics Solver Backend.
Integrates the time-dependent Lindblad Master Equation:
    d(rho)/dt = -i [H_drift + H_ctrl(t), rho] + sum_k D[L_k](rho)
using QuTiP (`qutip.mesolve`), capturing continuous intra-gate pulse trajectories,
static ZZ crosstalk, T1 energy relaxation, T2* pure dephasing, and inline Kraus noise.

Also provides `calibrate_gate_pauli_noise_with_qutip()` to extract effective Pauli
error channels from QuTiP physical pulse evolution via Pauli Twirling for Stim.
"""

from __future__ import annotations

import math
import os
os.environ.setdefault("MPMATH_NOGMPY", "1")
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import qutip as qt

from ..core.circuit import (
    GateInstruction,
    MeasureInstruction,
    NoiseInstruction,
    QuantumCircuit,
    embed_gate_unitary,
)
from ..core.gates import PAULI_X, PAULI_Y, PAULI_Z, PAULIS_1Q, gate_hamiltonian_generator
from ..core.hardware import HardwareModel
from ..core.noise import NoiseChannel, PAULI_LABELS_1Q, PAULI_LABELS_2Q, ReadoutError
from .base import (
    EvolutionResult,
    StepSnapshot,
    apply_readout_confusion,
    compute_bloch_vectors,
    compute_populations,
    sample_from_probabilities,
    state_fidelity_with_pure,
)
from .kraus_backend import apply_kraus_channel


def _build_qutip_collapse_operators(
    hardware: Optional[HardwareModel],
    num_qubits: int,
) -> List[qt.Qobj]:
    """
    Construct physical Lindblad collapse operators {L_k} in units of (ns)^{-1/2}:
      - Amplitude damping (T1): L_{1,q} = (1 / sqrt(T1_ns)) * sigma_minus_q
      - Pure dephasing (T_phi): L_{phi,q} = (1 / sqrt(2 * T_phi_ns)) * sigma_z_q
        (since D[sigma_z](rho) = sigma_z rho sigma_z - rho decays off-diagonal elements at rate 2 * gamma_phi)
    """
    if hardware is None:
        return []

    c_ops: List[qt.Qobj] = []
    dims = [[2] * num_qubits, [2] * num_qubits]
    sm_1q = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)  # |0><1|

    for q in range(num_qubits):
        qp = hardware.qubit_params[q]
        if qp.t1_ns > 0 and math.isfinite(qp.t1_ns):
            rate_1 = 1.0 / math.sqrt(qp.t1_ns)
            op_full = embed_gate_unitary(rate_1 * sm_1q, (q,), num_qubits)
            c_ops.append(qt.Qobj(op_full, dims=dims))

        t_phi = qp.t_phi_ns
        if t_phi > 0 and math.isfinite(t_phi):
            rate_phi = 1.0 / math.sqrt(2.0 * t_phi)
            op_full = embed_gate_unitary(rate_phi * PAULI_Z, (q,), num_qubits)
            c_ops.append(qt.Qobj(op_full, dims=dims))

    return c_ops


def _build_drift_hamiltonian(
    hardware: Optional[HardwareModel],
    num_qubits: int,
) -> qt.Qobj:
    """
    Construct static drift Hamiltonian H_drift (in rad/ns, i.e., GHz * 2pi):
      H_drift = sum_{(i,j)} 2*pi * (J_zz_MHz * 1e-3) * (Z_i ⊗ Z_j) / 4
    representing residual ZZ crosstalk between coupled physical qubits.
    """
    dim = 1 << num_qubits
    dims = [[2] * num_qubits, [2] * num_qubits]
    H_mat = np.zeros((dim, dim), dtype=np.complex128)

    if hardware is not None:
        zz_2q = np.kron(PAULI_Z, PAULI_Z)
        for pair, cp in hardware.coupler_params.items():
            if abs(cp.zz_crosstalk_mhz) > 1e-12 and pair[0] < num_qubits and pair[1] < num_qubits:
                # 1 MHz = 1e-3 GHz = 1e-3 cycles/ns => angular frequency = 2*pi * 1e-3 rad/ns
                omega_zz = 2.0 * math.pi * (cp.zz_crosstalk_mhz * 1e-3)
                H_mat += 0.25 * omega_zz * embed_gate_unitary(zz_2q, pair, num_qubits)

    return qt.Qobj(H_mat, dims=dims)


def _make_pulse_envelope(pulse_shape: str, gate_dt: float):
    """
    Build a smooth pulse envelope f(t) supported on [0, gate_dt] with integral_0^{gate_dt} f(t) dt = 1.
    Returns 0 for t > gate_dt so faster gates in a parallel layer naturally idle while longer gates finish.
    """
    shape = (pulse_shape or "cosine").lower()
    if shape == "rectangular":
        amp = 1.0 / gate_dt

        def rect_fn(t: float, _args: dict = {}) -> float:
            return amp if (0.0 <= t <= gate_dt) else 0.0

        return rect_fn

    if shape == "gaussian":
        # Zero-endpoint truncated Gaussian centered at gate_dt / 2 with sigma = gate_dt / 4.5
        mid = 0.5 * gate_dt
        sigma = gate_dt / 4.5
        edge = math.exp(-0.5 * (mid / sigma) ** 2)
        # Integral of exp(-(t-mid)^2/(2*sigma^2)) - edge over [0, gate_dt]
        area = math.sqrt(2.0 * math.pi) * sigma * math.erf(mid / (math.sqrt(2.0) * sigma)) - edge * gate_dt
        norm = 1.0 / max(area, 1e-12)

        def gauss_fn(t: float, _args: dict = {}) -> float:
            if t < 0.0 or t > gate_dt:
                return 0.0
            return norm * max(0.0, math.exp(-0.5 * ((t - mid) / sigma) ** 2) - edge)

        return gauss_fn

    # Default: smooth Hann / raised cosine window (1 - cos(2*pi*t/dt)) / dt
    def hann_fn(t: float, _args: dict = {}) -> float:
        if t < 0.0 or t > gate_dt:
            return 0.0
        return (1.0 - math.cos(2.0 * math.pi * t / gate_dt)) / gate_dt

    return hann_fn


class QuTiPBackend:
    """
    Physical Hamiltonian & Lindblad Master Equation Solver powered by QuTiP (`qutip.mesolve`).
    Simulates continuous-time pulse evolution during each quantum gate along with
    experimental decoherence (T1, T2*), static ZZ crosstalk, and control errors.
    """

    def __init__(self, intra_gate_steps: int = 15) -> None:
        self.intra_gate_steps = max(3, int(intra_gate_steps))

    def run(
        self,
        circuit: QuantumCircuit,
        hardware: Optional[HardwareModel] = None,
        shots: int = 4096,
        seed: Optional[int] = 42,
        initial_state: Optional[np.ndarray] = None,
    ) -> EvolutionResult:
        n = circuit.num_qubits
        dim = 1 << n
        dims = [[2] * n, [2] * n]

        if initial_state is None:
            psi_ideal = np.zeros(dim, dtype=np.complex128)
            psi_ideal[0] = 1.0
            rho = np.zeros((dim, dim), dtype=np.complex128)
            rho[0, 0] = 1.0
        else:
            arr = np.asarray(initial_state, dtype=np.complex128)
            if arr.ndim == 1:
                psi_ideal = arr / np.linalg.norm(arr)
                rho = np.outer(psi_ideal, psi_ideal.conj())
            else:
                rho = arr / np.real(np.trace(arr))
                eigvals, eigvecs = np.linalg.eigh(rho)
                psi_ideal = eigvecs[:, int(np.argmax(eigvals))]

        c_ops = _build_qutip_collapse_operators(hardware, n)
        H_drift = _build_drift_hamiltonian(hardware, n)
        pulse_shape = hardware.pulse_shape if hardware is not None else "cosine"

        steps: List[StepSnapshot] = []
        elapsed_ns = 0.0
        step_idx = 0

        steps.append(
            StepSnapshot(
                step_index=0,
                operation_label="INIT |0...0>",
                elapsed_time_ns=0.0,
                density_matrix=rho.copy(),
                ideal_statevector=psi_ideal.copy(),
                state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                purity=float(np.real(np.trace(rho @ rho))),
                bloch_vectors=compute_bloch_vectors(rho, n),
                populations=compute_populations(rho, n),
            )
        )

        readout_errors: Dict[int, ReadoutError] = {}
        if hardware is not None:
            for q in range(n):
                readout_errors[q] = hardware.qubit_params[q].readout_error

        # Group consecutive GateInstructions sharing the same non-None `layer` on disjoint qubits
        # into true parallel pulse layers; sequential circuits (layer=None) stay gate-by-gate.
        op_idx = 0
        ops_list = circuit.operations
        total_ops = len(ops_list)

        while op_idx < total_ops:
            op = ops_list[op_idx]
            if isinstance(op, GateInstruction):
                layer_gates: List[GateInstruction] = [op]
                used_qubits = set(op.qubits)
                op_idx += 1
                if op.layer is not None:
                    while op_idx < total_ops:
                        nxt = ops_list[op_idx]
                        if (
                            isinstance(nxt, GateInstruction)
                            and nxt.layer == op.layer
                            and used_qubits.isdisjoint(nxt.qubits)
                        ):
                            layer_gates.append(nxt)
                            used_qubits.update(nxt.qubits)
                            op_idx += 1
                        else:
                            break

                step_idx += 1
                gate_dts: List[float] = []
                for g in layer_gates:
                    U_g_full = embed_gate_unitary(g.unitary(), g.qubits, n)
                    psi_ideal = U_g_full @ psi_ideal
                    dt_g = g.duration_ns
                    if dt_g is None:
                        dt_g = hardware.gate_duration_ns(g.name, g.qubits) if hardware is not None else 20.0
                    gate_dts.append(float(dt_g))

                # Apply zero-duration Virtual-Z / instantaneous gates first
                finite_entries: List[Tuple[GateInstruction, float]] = []
                for g, dt_g in zip(layer_gates, gate_dts):
                    if dt_g <= 1e-9:
                        U_g_full = embed_gate_unitary(g.unitary(), g.qubits, n)
                        rho = U_g_full @ rho @ U_g_full.conj().T
                    else:
                        finite_entries.append((g, dt_g))

                dt_layer = max((dt_g for _, dt_g in finite_entries), default=0.0)

                if dt_layer <= 1e-9:
                    t_grid = np.array([elapsed_ns], dtype=np.float64)
                    bloch_now = compute_bloch_vectors(rho, n)
                    pulse_traj = {q: np.array([bloch_now[q]], dtype=np.float64) for q in range(n)}
                else:
                    all_ideal_layer = all(g.skip_hw_noise for g, _ in finite_entries)
                    layer_c_ops = [] if all_ideal_layer else c_ops
                    layer_drift = qt.Qobj(np.zeros((dim, dim), dtype=np.complex128), dims=dims) if all_ideal_layer else H_drift

                    # Build combined parallel Hamiltonian H(t) = H_drift + sum_g H_ctrl_g * env_g(t)
                    H_td_list: List = [layer_drift]
                    static_ctrl = np.zeros((dim, dim), dtype=np.complex128)
                    has_td = False

                    for g, dt_g in finite_entries:
                        if g.custom_hamiltonian is not None:
                            H_gen_full = embed_gate_unitary(g.custom_hamiltonian, g.qubits, n)
                        else:
                            H_gen_local, theta = gate_hamiltonian_generator(g.name, g.params, g.custom_matrix)
                            H_gen_full = embed_gate_unitary(0.5 * theta * H_gen_local, g.qubits, n)

                        if pulse_shape == "rectangular" and abs(dt_g - dt_layer) <= 1e-9:
                            static_ctrl += (1.0 / dt_g) * H_gen_full
                        else:
                            has_td = True
                            H_td_list.append([qt.Qobj(H_gen_full, dims=dims), _make_pulse_envelope(pulse_shape, dt_g)])

                    if np.any(np.abs(static_ctrl) > 1e-15):
                        H_td_list[0] = layer_drift + qt.Qobj(static_ctrl, dims=dims)

                    H_td = H_td_list if has_td else H_td_list[0]
                    t_local = np.linspace(0.0, dt_layer, self.intra_gate_steps)
                    rho0_qobj = qt.Qobj(rho, dims=dims)

                    res = qt.mesolve(H_td, rho0_qobj, t_local, c_ops=layer_c_ops)
                    t_grid = elapsed_ns + t_local
                    pulse_traj = {q: np.zeros((len(t_local), 3), dtype=np.float64) for q in range(n)}
                    for tidx, state_qobj in enumerate(res.states):
                        rho_t = state_qobj.full()
                        bv = compute_bloch_vectors(rho_t, n)
                        for q in range(n):
                            pulse_traj[q][tidx] = bv[q]

                    rho = res.states[-1].full()
                    rho = 0.5 * (rho + rho.conj().T)

                elapsed_ns += dt_layer

                # Apply per-gate control depolarizing error (unless skip_hw_noise) + explicit inline noise
                for g, dt_g in zip(layer_gates, gate_dts):
                    if hardware is not None and dt_g > 0 and not g.skip_hw_noise:
                        if len(g.qubits) == 1:
                            p_dep = hardware.qubit_params[g.qubits[0]].depol_1q
                            if p_dep > 0:
                                rho = apply_kraus_channel(rho, NoiseChannel.depolarizing_1q(g.qubits[0], p_dep), n)
                        elif len(g.qubits) == 2:
                            pair = (min(g.qubits[0], g.qubits[1]), max(g.qubits[0], g.qubits[1]))
                            if pair in hardware.coupler_params and hardware.coupler_params[pair].depol_2q > 0:
                                rho = apply_kraus_channel(
                                    rho,
                                    NoiseChannel.depolarizing_2q(
                                        g.qubits[0], g.qubits[1], hardware.coupler_params[pair].depol_2q
                                    ),
                                    n,
                                )
                    for ch in g.noise_after:
                        rho = apply_kraus_channel(rho, ch, n)

                layer_label = (
                    f"{layer_gates[0].format_label()} @{layer_gates[0].qubits}"
                    if len(layer_gates) == 1
                    else " ∥ ".join(f"{g.format_label()} @{g.qubits}" for g in layer_gates)
                )

                steps.append(
                    StepSnapshot(
                        step_index=step_idx,
                        operation_label=layer_label,
                        elapsed_time_ns=elapsed_ns,
                        density_matrix=rho.copy(),
                        ideal_statevector=psi_ideal.copy(),
                        state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                        purity=float(np.real(np.trace(rho @ rho))),
                        bloch_vectors=compute_bloch_vectors(rho, n),
                        populations=compute_populations(rho, n),
                        pulse_times_ns=t_grid,
                        pulse_bloch_trajectories=pulse_traj,
                    )
                )

            elif isinstance(op, NoiseInstruction):
                op_idx += 1
                step_idx += 1
                rho = apply_kraus_channel(rho, op.channel, n)
                steps.append(
                    StepSnapshot(
                        step_index=step_idx,
                        operation_label=f"NOISE[{op.channel.kind}] @{op.channel.qubits}",
                        elapsed_time_ns=elapsed_ns,
                        density_matrix=rho.copy(),
                        ideal_statevector=psi_ideal.copy(),
                        state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                        purity=float(np.real(np.trace(rho @ rho))),
                        bloch_vectors=compute_bloch_vectors(rho, n),
                        populations=compute_populations(rho, n),
                    )
                )

            elif isinstance(op, MeasureInstruction):
                op_idx += 1
                if op.readout_error is not None:
                    for q in op.qubits:
                        readout_errors[q] = op.readout_error
            else:
                op_idx += 1

        raw_probs = compute_populations(rho, n)
        final_probs = apply_readout_confusion(raw_probs, n, readout_errors)
        counts = sample_from_probabilities(final_probs, shots=shots, seed=seed)

        return EvolutionResult(
            backend_name="QuTiP-MasterEquation",
            num_qubits=n,
            steps=steps,
            final_probabilities=final_probs,
            measurement_counts=counts,
            shots=shots,
            total_duration_ns=elapsed_ns,
            final_fidelity=steps[-1].state_fidelity,
            final_purity=steps[-1].purity,
        )

    def extract_pauli_noise_for_stim(
        self,
        gate: GateInstruction,
        hardware: HardwareModel,
    ) -> Dict[str, float]:
        """
        Bridge from QuTiP physical simulation to Stim Pauli noise channel!
        Simulates the noisy physical evolution of `gate` via QuTiP's `propagator` on the
        active 1Q or 2Q subsystem, isolates the residual post-gate error superoperator
        S_err = S_noisy @ S_ideal^{-1}, and performs exact Pauli Twirling to obtain the
        Pauli error probabilities {'I': pI, 'X': pX, ...} or {'II': pII, 'IX': pIX, ...}.
        """
        k = len(gate.qubits)
        if k not in (1, 2):
            raise ValueError("extract_pauli_noise_for_stim supports 1Q or 2Q gates.")

        dt = gate.duration_ns if gate.duration_ns is not None else hardware.gate_duration_ns(gate.name, gate.qubits)
        if dt <= 1e-9:
            return {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0} if k == 1 else {lbl: (1.0 if lbl == "II" else 0.0) for lbl in PAULI_LABELS_2Q}

        # Build subsystem hardware model for the gate's active qubits
        sub_dims = [[2] * k, [2] * k]
        H_gen_local, theta = gate_hamiltonian_generator(gate.name, gate.params, gate.custom_matrix)
        H_ctrl_qobj = qt.Qobj((0.5 * theta / dt) * H_gen_local, dims=sub_dims)

        # Collapse operators on active qubits
        c_ops: List[qt.Qobj] = []
        sm_1q = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        for local_idx, phys_q in enumerate(gate.qubits):
            qp = hardware.qubit_params[phys_q]
            if qp.t1_ns > 0 and math.isfinite(qp.t1_ns):
                op = embed_gate_unitary((1.0 / math.sqrt(qp.t1_ns)) * sm_1q, (local_idx,), k)
                c_ops.append(qt.Qobj(op, dims=sub_dims))
            t_phi = qp.t_phi_ns
            if t_phi > 0 and math.isfinite(t_phi):
                op = embed_gate_unitary((1.0 / math.sqrt(2.0 * t_phi)) * PAULI_Z, (local_idx,), k)
                c_ops.append(qt.Qobj(op, dims=sub_dims))

        # Evolve Pauli basis matrices {P_a} through mesolve to construct the Pauli Transfer Matrix (PTM) R_ab
        pauli_labels = PAULI_LABELS_1Q if k == 1 else PAULI_LABELS_2Q
        pauli_mats = [
            PAULIS_1Q[lbl] if k == 1 else np.kron(PAULIS_1Q[lbl[0]], PAULIS_1Q[lbl[1]])
            for lbl in pauli_labels
        ]
        U_ideal = gate.unitary()
        U_dag = U_ideal.conj().T
        dim = 1 << k

        # For error channel E_err(rho) = E_noisy(U_dag rho U_ideal),
        # its PTM diagonal is d_a = (1/2^k) Tr(P_a E_err(P_a))
        ptm_diag = np.zeros(len(pauli_labels), dtype=np.float64)
        t_span = np.linspace(0.0, dt, max(5, self.intra_gate_steps))

        for idx, P_a in enumerate(pauli_mats):
            if idx == 0:
                ptm_diag[0] = 1.0
                continue
            # Input U_dag P_a U_ideal (shifted by I to make positive Hermitian density operator if needed,
            # though mesolve is linear and accepts any Hermitian operator!)
            H_in = qt.Qobj(U_dag @ P_a @ U_ideal, dims=sub_dims)
            res = qt.mesolve(H_ctrl_qobj, H_in, t_span, c_ops=c_ops)
            out_mat = res.states[-1].full()
            # Apply control depolarizing if present
            if k == 1:
                p_dep = hardware.qubit_params[gate.qubits[0]].depol_1q
                if p_dep > 0:
                    out_mat = (1.0 - 4.0 * p_dep / 3.0) * out_mat
            elif k == 2:
                pair = (min(gate.qubits[0], gate.qubits[1]), max(gate.qubits[0], gate.qubits[1]))
                if pair in hardware.coupler_params and hardware.coupler_params[pair].depol_2q > 0:
                    p_dep2 = hardware.coupler_params[pair].depol_2q
                    out_mat = (1.0 - 16.0 * p_dep2 / 15.0) * out_mat

            ptm_diag[idx] = float(np.real(np.trace(P_a @ out_mat)) / dim)

        # Walsh-Hadamard transform converts PTM diagonal {d_P} to Pauli error probabilities {p_Q}:
        # d_P = sum_Q (-1)^{<P, Q>} p_Q  <=>  p_Q = (1 / 4^k) sum_P (-1)^{<P, Q>} d_P
        n_paulis = len(pauli_labels)
        probs: Dict[str, float] = {}
        for q_idx, q_lbl in enumerate(pauli_labels):
            Q_mat = pauli_mats[q_idx]
            acc = 0.0
            for p_idx, P_mat in enumerate(pauli_mats):
                # Commutation sign: P Q = ± Q P => (1/2^k) Tr(P Q P Q) = ±1
                sign = float(np.real(np.trace(P_mat @ Q_mat @ P_mat @ Q_mat)) / dim)
                acc += sign * ptm_diag[p_idx]
            probs[q_lbl] = max(0.0, acc / float(n_paulis))

        total = sum(probs.values())
        if total > 0:
            for k_lbl in probs:
                probs[k_lbl] /= total
        return probs
