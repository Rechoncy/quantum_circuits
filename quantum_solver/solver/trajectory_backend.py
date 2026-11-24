"""
Pure-Statevector & Stochastic Kraus Quantum Trajectory Solver Backend (`TrajectoryBackend`).

Uses O(2^k * 2^N) tensor-axis contraction (`apply_operator_to_statevector`) without ever
allocating a 2^N x 2^N matrix or density operator, scaling universal non-Clifford circuits
(arbitrary RX/RY/RZ/RXY/RZZ/MS angles, VQE, QAOA, QFT, W-states, custom gates) up to 24 qubits.

When physical hardware noise (T1/T2 thermal relaxation, 1Q/2Q depolarizing, ZZ crosstalk) or
inline NoiseInstructions are active, unravels CPTP Kraus channels into lockstep pairs of pure
quantum trajectories (|psi_A>, |psi_B>) alongside |psi_ideal>, yielding:
  - Unbiased state fidelity F(s) = E[|<psi_ideal(s) | psi(s)>|^2]
  - Unbiased SWAP-test purity Tr(rho(s)^2) = E[|<psi_A(s) | psi_B(s)>|^2]
  - Averaged reduced 1-qubit Bloch vectors (<X_q>, <Y_q>, <Z_q>)
  - Readout-corrupted measurement bitstring counts in O(shots * N)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.circuit import (
    GateInstruction,
    MeasureInstruction,
    NoiseInstruction,
    QuantumCircuit,
)
from ..core.gates import PAULI_X, PAULI_Y, PAULI_Z, PAULIS_1Q
from ..core.hardware import HardwareModel
from ..core.noise import PAULI_LABELS_2Q, NoiseChannel, ReadoutError
from .base import (
    EvolutionResult,
    StepSnapshot,
    apply_operator_to_statevector,
    apply_readout_confusion_array,
    compute_statevector_bloch,
    probabilities_array_to_dict,
    sample_bitstrings_from_prob_array,
)


def _has_dynamical_noise(circuit: QuantumCircuit, hardware: Optional[HardwareModel]) -> bool:
    """Return True if any non-unitary noise channel or ZZ crosstalk is active during evolution."""
    for op in circuit.operations:
        if isinstance(op, NoiseInstruction):
            return True
        if isinstance(op, GateInstruction) and op.noise_after:
            return True

    if hardware is None:
        return False

    has_non_skipped_gate = any(
        isinstance(op, GateInstruction) and not op.skip_hw_noise for op in circuit.operations
    )
    if not has_non_skipped_gate:
        return False

    for qp in hardware.qubit_params.values():
        if qp.depol_1q > 0 or math.isfinite(qp.t1_ns) or math.isfinite(qp.t2_ns):
            return True
    for cp in hardware.coupler_params.values():
        if cp.depol_2q > 0 or abs(cp.zz_crosstalk_mhz) > 1e-9:
            return True
    return False


def _apply_1q_pauli_inplace(psi: np.ndarray, qubit: int, pauli_char: str, num_qubits: int) -> np.ndarray:
    """Apply single-qubit Pauli X, Y, or Z directly to a 3D reshaped view of `psi` in O(2^N)."""
    if pauli_char == "I":
        return psi
    d_left = 1 << qubit
    d_right = 1 << (num_qubits - 1 - qubit)
    psi_3d = psi.reshape((d_left, 2, d_right))
    if pauli_char == "Z":
        psi_3d[:, 1, :] *= -1.0
        return psi
    if pauli_char == "X":
        tmp = psi_3d[:, 0, :].copy()
        psi_3d[:, 0, :] = psi_3d[:, 1, :]
        psi_3d[:, 1, :] = tmp
        return psi
    if pauli_char == "Y":
        tmp = psi_3d[:, 0, :].copy()
        psi_3d[:, 0, :] = -1j * psi_3d[:, 1, :]
        psi_3d[:, 1, :] = 1j * tmp
        return psi
    return apply_operator_to_statevector(psi, PAULIS_1Q[pauli_char], (qubit,), num_qubits)


def _apply_amplitude_damping_trajectory(
    psi: np.ndarray,
    qubit: int,
    gamma: float,
    num_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stochastically unravel 1-qubit amplitude damping (T1 decay) in O(2^N)."""
    gamma = max(0.0, min(1.0, gamma))
    if gamma <= 1e-13:
        return psi
    d_left = 1 << qubit
    d_right = 1 << (num_qubits - 1 - qubit)
    psi_3d = psi.reshape((d_left, 2, d_right))
    psi1 = psi_3d[:, 1, :]
    p1 = float(np.real(np.vdot(psi1, psi1)))
    p_jump = gamma * p1
    if p_jump > 1e-15 and rng.random() < p_jump:
        # Quantum jump |1> -> |0>
        inv_norm = 1.0 / math.sqrt(max(p1, 1e-30))
        psi_3d[:, 0, :] = psi1 * inv_norm
        psi_3d[:, 1, :] = 0.0
    else:
        # No-jump non-Hermitian damping K_0 = diag(1, sqrt(1 - gamma))
        psi_3d[:, 1, :] *= math.sqrt(max(0.0, 1.0 - gamma))
        norm_sq = max(1e-30, 1.0 - p_jump)
        psi *= 1.0 / math.sqrt(norm_sq)
    return psi


def _apply_phase_damping_trajectory(
    psi: np.ndarray,
    qubit: int,
    lam: float,
    num_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stochastically unravel pure dephasing (lam = 1 - exp(-2*dt/T_phi)) via its exact unitary
    Pauli-Z equivalence: p_Z = 0.5 * (1 - sqrt(1 - lam)), requiring O(1) when no jump occurs!
    """
    lam = max(0.0, min(1.0, lam))
    if lam <= 1e-13:
        return psi
    p_z = 0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - lam)))
    if rng.random() < p_z:
        _apply_1q_pauli_inplace(psi, qubit, "Z", num_qubits)
    return psi


def sample_kraus_trajectory_step(
    psi: np.ndarray,
    channel: NoiseChannel,
    num_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one Monte-Carlo quantum trajectory realization of `channel` to pure state `psi`."""
    if channel.custom_kraus is None:
        kind = channel.kind.lower()
        if kind == "depolarizing1q":
            p = max(0.0, min(1.0, channel.params["p"]))
            u = rng.random()
            if u < p:
                idx = int((u / max(p, 1e-30)) * 3)
                pauli_c = ("X", "Y", "Z")[min(idx, 2)]
                _apply_1q_pauli_inplace(psi, channel.qubits[0], pauli_c, num_qubits)
            return psi

        if kind == "pauli1q":
            px = max(0.0, channel.params.get("px", 0.0))
            py = max(0.0, channel.params.get("py", 0.0))
            pz = max(0.0, channel.params.get("pz", 0.0))
            u = rng.random()
            if u < px:
                _apply_1q_pauli_inplace(psi, channel.qubits[0], "X", num_qubits)
            elif u < px + py:
                _apply_1q_pauli_inplace(psi, channel.qubits[0], "Y", num_qubits)
            elif u < px + py + pz:
                _apply_1q_pauli_inplace(psi, channel.qubits[0], "Z", num_qubits)
            return psi

        if kind == "depolarizing2q":
            p = max(0.0, min(1.0, channel.params["p"]))
            u = rng.random()
            if u < p:
                non_id = PAULI_LABELS_2Q[1:]  # 15 non-II Paulis
                idx = min(14, int((u / max(p, 1e-30)) * 15))
                lbl = non_id[idx]
                _apply_1q_pauli_inplace(psi, channel.qubits[0], lbl[0], num_qubits)
                _apply_1q_pauli_inplace(psi, channel.qubits[1], lbl[1], num_qubits)
            return psi

        if kind == "pauli2q":
            u = rng.random()
            cum = 0.0
            for lbl in PAULI_LABELS_2Q[1:]:
                prob = max(0.0, channel.params.get(lbl, 0.0))
                cum += prob
                if u < cum:
                    _apply_1q_pauli_inplace(psi, channel.qubits[0], lbl[0], num_qubits)
                    _apply_1q_pauli_inplace(psi, channel.qubits[1], lbl[1], num_qubits)
                    break
            return psi

        if kind == "amplitude_damping":
            return _apply_amplitude_damping_trajectory(
                psi, channel.qubits[0], channel.params["gamma"], num_qubits, rng
            )

        if kind == "phase_damping":
            return _apply_phase_damping_trajectory(
                psi, channel.qubits[0], channel.params["lam"], num_qubits, rng
            )

        if kind == "thermal_relaxation":
            q = channel.qubits[0]
            t1 = channel.params["t1_ns"]
            t2 = min(channel.params["t2_ns"], 2.0 * t1)
            dt = channel.params["duration_ns"]
            gamma_1 = 1.0 - math.exp(-dt / t1) if (t1 > 0 and math.isfinite(t1)) else 0.0
            if t2 > 0 and math.isfinite(t2):
                rate_phi = max(
                    0.0,
                    (1.0 / t2) - (0.5 / t1 if (t1 > 0 and math.isfinite(t1)) else 0.0),
                )
                lam_phi = 1.0 - math.exp(-2.0 * dt * rate_phi)
            else:
                lam_phi = 0.0
            if gamma_1 > 1e-13:
                psi = _apply_amplitude_damping_trajectory(psi, q, gamma_1, num_qubits, rng)
            if lam_phi > 1e-13:
                psi = _apply_phase_damping_trajectory(psi, q, lam_phi, num_qubits, rng)
            return psi

    # General fallback for arbitrary custom Kraus operators {K_m}
    kraus_ops = channel.kraus_operators()
    candidates: List[np.ndarray] = []
    weights: List[float] = []
    for K in kraus_ops:
        cand = apply_operator_to_statevector(psi, K, channel.qubits, num_qubits)
        w = float(max(0.0, np.real(np.vdot(cand, cand))))
        candidates.append(cand)
        weights.append(w)
    total_w = sum(weights)
    if total_w <= 1e-30:
        return psi
    probs = [w / total_w for w in weights]
    chosen = int(rng.choice(len(candidates), p=probs))
    chosen_norm = math.sqrt(max(weights[chosen], 1e-30))
    return candidates[chosen] / chosen_norm


def _apply_zz_crosstalk_layer(
    psi: np.ndarray,
    dt_ns: float,
    hardware: HardwareModel,
    active_qubits: Sequence[int],
    num_qubits: int,
) -> np.ndarray:
    """Apply residual static ZZ crosstalk phase exp(-i * pi * zz_mhz * 1e-3 * dt_ns * Z_u Z_v)."""
    if dt_ns <= 0 or not hardware.coupler_params:
        return psi
    # For large registries (> 40 couplers), restrict per-layer ZZ scan to couplers touching active qubits + linear chain
    check_all = len(hardware.coupler_params) <= 64
    active_set = set(active_qubits)
    for (u, v), cp in hardware.coupler_params.items():
        if abs(cp.zz_crosstalk_mhz) <= 1e-9:
            continue
        if not check_all and (u not in active_set and v not in active_set and abs(u - v) > 1):
            continue
        # H_zz = (2*pi * zz_mhz * 1e-3 / 2) Z_u Z_v => phase angle = pi * zz_mhz * 1e-3 * dt_ns
        phi = math.pi * cp.zz_crosstalk_mhz * 1e-3 * dt_ns
        if abs(phi) <= 1e-9:
            continue
        e_minus = complex(math.cos(phi), -math.sin(phi))
        e_plus = complex(math.cos(phi), math.sin(phi))
        u_zz = np.diag([e_minus, e_plus, e_plus, e_minus])
        psi = apply_operator_to_statevector(psi, u_zz, (u, v), num_qubits)
    return psi


class TrajectoryBackend:
    """
    Pure Statevector & Monte-Carlo Quantum Trajectory Solver Backend.
    Scales non-Clifford universal circuits up to 24 qubits with O(2^N) memory.
    """

    MAX_STATEVECTOR_QUBITS = 24

    def __init__(self, default_trajectories: Optional[int] = None) -> None:
        self.default_trajectories = default_trajectories

    @staticmethod
    def recommended_trajectory_count(num_qubits: int) -> int:
        """Choose an adaptive pair-aligned trajectory count balancing accuracy and latency."""
        if num_qubits <= 8:
            return 32
        if num_qubits <= 12:
            return 20
        if num_qubits <= 16:
            return 12
        if num_qubits <= 19:
            return 6
        if num_qubits <= 21:
            return 4
        return 2

    def _group_scheduled_ops(self, circuit: QuantumCircuit) -> List[object]:
        """Group consecutive parallel GateInstructions sharing a layer into lists."""
        grouped: List[object] = []
        ops_list = circuit.operations
        total_ops = len(ops_list)
        op_idx = 0
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
                grouped.append(layer_gates)
            else:
                grouped.append(op)
                op_idx += 1
        return grouped

    def _evolve_single_noisy_trajectory(
        self,
        psi: np.ndarray,
        step_item: object,
        hardware: Optional[HardwareModel],
        num_qubits: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Advance one noisy statevector trajectory across a grouped step item."""
        if isinstance(step_item, list):
            layer_gates: List[GateInstruction] = step_item
            used_qubits = set()
            gate_dts: List[float] = []
            for g in layer_gates:
                used_qubits.update(g.qubits)
                u_small = g.unitary()
                psi = apply_operator_to_statevector(psi, u_small, g.qubits, num_qubits)
                dt_g = g.duration_ns
                if dt_g is None:
                    dt_g = hardware.gate_duration_ns(g.name, g.qubits) if hardware is not None else 20.0
                gate_dts.append(float(dt_g))

            dt_layer = max(gate_dts) if gate_dts else 0.0
            all_ideal_layer = all(g.skip_hw_noise for g in layer_gates)

            if hardware is not None and dt_layer > 0 and not all_ideal_layer:
                psi = _apply_zz_crosstalk_layer(psi, dt_layer, hardware, list(used_qubits), num_qubits)

            for g, dt_g in zip(layer_gates, gate_dts):
                for ch in g.noise_after:
                    psi = sample_kraus_trajectory_step(psi, ch, num_qubits, rng)
                if hardware is not None and dt_g > 0 and not g.skip_hw_noise:
                    if len(g.qubits) == 1:
                        q0 = g.qubits[0]
                        p_dep = hardware.qubit_params[q0].depol_1q
                        if p_dep > 0:
                            psi = sample_kraus_trajectory_step(
                                psi, NoiseChannel.depolarizing_1q(q0, p_dep), num_qubits, rng
                            )
                    elif len(g.qubits) == 2:
                        pair = (min(g.qubits[0], g.qubits[1]), max(g.qubits[0], g.qubits[1]))
                        if pair in hardware.coupler_params:
                            p_dep2 = hardware.coupler_params[pair].depol_2q
                            if p_dep2 > 0:
                                psi = sample_kraus_trajectory_step(
                                    psi,
                                    NoiseChannel.depolarizing_2q(g.qubits[0], g.qubits[1], p_dep2),
                                    num_qubits,
                                    rng,
                                )
                    for q_act in g.qubits:
                        qp = hardware.qubit_params[q_act]
                        if math.isfinite(qp.t1_ns) or math.isfinite(qp.t2_ns):
                            ch_th = NoiseChannel.thermal_relaxation(q_act, qp.t1_ns, qp.t2_ns, dt_layer)
                            psi = sample_kraus_trajectory_step(psi, ch_th, num_qubits, rng)

            if hardware is not None and dt_layer > 0 and not all_ideal_layer and hardware.include_idle_noise:
                for q_idle in range(num_qubits):
                    if q_idle not in used_qubits:
                        qp = hardware.qubit_params[q_idle]
                        if math.isfinite(qp.t1_ns) or math.isfinite(qp.t2_ns):
                            ch_idle = NoiseChannel.thermal_relaxation(q_idle, qp.t1_ns, qp.t2_ns, dt_layer)
                            psi = sample_kraus_trajectory_step(psi, ch_idle, num_qubits, rng)
            return psi

        if isinstance(step_item, NoiseInstruction):
            return sample_kraus_trajectory_step(psi, step_item.channel, num_qubits, rng)

        return psi

    def run(
        self,
        circuit: QuantumCircuit,
        hardware: Optional[HardwareModel] = None,
        shots: int = 4096,
        seed: Optional[int] = 42,
        initial_state: Optional[np.ndarray] = None,
        num_trajectories: Optional[int] = None,
    ) -> EvolutionResult:
        n = circuit.num_qubits
        if n > self.MAX_STATEVECTOR_QUBITS:
            raise ValueError(
                f"Trajectory 态矢量后端最高支持 {self.MAX_STATEVECTOR_QUBITS} 比特（2^{n} 复振幅需要 "
                f"{(16 * (1 << n)) / (1024**3):.1f} GB 内存）。大规模稳定子/纠错线路请切换至「Stim (500+Q)」后端。"
            )
        dim = 1 << n

        if initial_state is None:
            psi_init = np.zeros(dim, dtype=np.complex128)
            psi_init[0] = 1.0
        else:
            arr = np.asarray(initial_state, dtype=np.complex128)
            if arr.ndim == 1:
                psi_init = arr.astype(np.complex128, copy=True)
                norm = float(np.linalg.norm(psi_init))
                if norm > 0:
                    psi_init /= norm
            else:
                eigvals, eigvecs = np.linalg.eigh(arr)
                psi_init = eigvecs[:, int(np.argmax(eigvals))].astype(np.complex128, copy=True)

        readout_errors: Dict[int, ReadoutError] = {}
        if hardware is not None:
            for q in range(n):
                readout_errors[q] = hardware.qubit_params[q].readout_error

        grouped_ops = self._group_scheduled_ops(circuit)
        for item in grouped_ops:
            if isinstance(item, MeasureInstruction) and item.readout_error is not None:
                for q in item.qubits:
                    readout_errors[q] = item.readout_error

        evolution_items = [it for it in grouped_ops if not isinstance(it, MeasureInstruction)]
        num_steps = 1 + len(evolution_items)  # Step 0 = INIT
        noisy_active = _has_dynamical_noise(circuit, hardware)

        # Precompute step labels and elapsed times
        step_labels: List[str] = ["INIT |0...0>"]
        step_times: List[float] = [0.0]
        elapsed_ns = 0.0
        for item in evolution_items:
            if isinstance(item, list):
                layer_gates: List[GateInstruction] = item
                dts = [
                    float(
                        g.duration_ns
                        if g.duration_ns is not None
                        else (hardware.gate_duration_ns(g.name, g.qubits) if hardware is not None else 20.0)
                    )
                    for g in layer_gates
                ]
                elapsed_ns += max(dts) if dts else 0.0
                lbl = (
                    f"{layer_gates[0].format_label()} @{layer_gates[0].qubits}"
                    if len(layer_gates) == 1
                    else " ∥ ".join(f"{g.format_label()} @{g.qubits}" for g in layer_gates)
                )
                step_labels.append(lbl)
                step_times.append(elapsed_ns)
            elif isinstance(item, NoiseInstruction):
                step_labels.append(f"NOISE[{item.channel.kind}] @{item.channel.qubits}")
                step_times.append(elapsed_ns)

        store_ideal_vec = n <= 10

        # ---- Fast Path: Exact Noiseless Statevector Evolution (1 pass) ----
        if not noisy_active:
            psi = psi_init.copy()
            steps: List[StepSnapshot] = []
            p0_arr = np.real(psi * psi.conj())
            steps.append(
                StepSnapshot(
                    step_index=0,
                    operation_label=step_labels[0],
                    elapsed_time_ns=0.0,
                    ideal_statevector=psi.copy() if store_ideal_vec else None,
                    state_fidelity=1.0,
                    purity=1.0,
                    bloch_vectors=compute_statevector_bloch(psi, n),
                    populations=probabilities_array_to_dict(p0_arr, n),
                )
            )
            for s_idx, item in enumerate(evolution_items, start=1):
                if isinstance(item, list):
                    for g in item:
                        psi = apply_operator_to_statevector(psi, g.unitary(), g.qubits, n)
                p_arr = np.real(psi * psi.conj())
                steps.append(
                    StepSnapshot(
                        step_index=s_idx,
                        operation_label=step_labels[s_idx],
                        elapsed_time_ns=step_times[s_idx],
                        ideal_statevector=psi.copy() if store_ideal_vec else None,
                        state_fidelity=1.0,
                        purity=1.0,
                        bloch_vectors=compute_statevector_bloch(psi, n),
                        populations=probabilities_array_to_dict(p_arr, n),
                    )
                )

            final_p_raw = np.real(psi * psi.conj())
            counts = sample_bitstrings_from_prob_array(
                final_p_raw, n, shots=shots, readout_errors=readout_errors, seed=seed
            )
            if readout_errors and n <= 16:
                final_p_noisy = apply_readout_confusion_array(final_p_raw, n, readout_errors)
                final_probs = probabilities_array_to_dict(final_p_noisy, n)
            elif readout_errors and shots > 0 and counts:
                final_probs = {bstr: cnt / float(shots) for bstr, cnt in list(counts.items())[:256]}
            else:
                final_probs = probabilities_array_to_dict(final_p_raw, n)

            return EvolutionResult(
                backend_name="Statevector-Exact",
                num_qubits=n,
                steps=steps,
                final_probabilities=final_probs,
                measurement_counts=counts,
                shots=shots,
                total_duration_ns=elapsed_ns,
                final_fidelity=1.0,
                final_purity=1.0,
                metadata={"trajectories": 1, "mode": "exact_statevector"},
            )

        # ---- Noisy Path: Lockstep Paired Monte-Carlo Quantum Trajectories (Multi-Threaded) ----
        m_req = num_trajectories or self.default_trajectories or self.recommended_trajectory_count(n)
        num_pairs = max(1, (int(m_req) + 1) // 2)
        total_trajs = 2 * num_pairs

        base_seed = 42 if seed is None else int(seed)
        master_rng = np.random.default_rng(base_seed)
        pair_seeds = [
            (
                idx,
                int(master_rng.integers(0, 2**31 - 1)),
                int(master_rng.integers(0, 2**31 - 1)),
            )
            for idx in range(num_pairs)
        ]

        # Precompute ideal statevector trajectory once when n <= 15 to avoid redundant work per pair
        precomputed_ideal: Optional[List[np.ndarray]] = None
        saved_ideal_vecs: Dict[int, np.ndarray] = {}
        if n <= 15:
            psi_id = psi_init.copy()
            precomputed_ideal = [psi_id.copy()]
            if store_ideal_vec:
                saved_ideal_vecs[0] = psi_id.copy()
            for s_idx, item in enumerate(evolution_items, start=1):
                if isinstance(item, list):
                    for g in item:
                        psi_id = apply_operator_to_statevector(psi_id, g.unitary(), g.qubits, n)
                precomputed_ideal.append(psi_id.copy())
                if store_ideal_vec:
                    saved_ideal_vecs[s_idx] = psi_id.copy()

        accum_step_probs = n <= 14

        def _simulate_pair(
            args: Tuple[int, int, int],
        ) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[np.ndarray],
            Dict[int, Dict[str, float]],
        ]:
            pair_idx, seed_a, seed_b = args
            rng_a = np.random.default_rng(seed_a)
            rng_b = np.random.default_rng(seed_b)

            loc_bloch = np.zeros((num_steps, n, 3), dtype=np.float64)
            loc_fid = np.zeros(num_steps, dtype=np.float64)
            loc_pur = np.zeros(num_steps, dtype=np.float64)
            loc_step_prob = (
                np.zeros((num_steps, dim), dtype=np.float64) if accum_step_probs else None
            )
            loc_rep_pops: Dict[int, Dict[str, float]] = {}

            psi_ideal_loc = None if precomputed_ideal is not None else psi_init.copy()
            psi_a = psi_init.copy()
            psi_b = psi_init.copy()

            ba0 = compute_statevector_bloch(psi_a, n)
            bb0 = compute_statevector_bloch(psi_b, n)
            for q in range(n):
                loc_bloch[0, q, 0] = ba0[q][0] + bb0[q][0]
                loc_bloch[0, q, 1] = ba0[q][1] + bb0[q][1]
                loc_bloch[0, q, 2] = ba0[q][2] + bb0[q][2]
            loc_fid[0] = 2.0
            loc_pur[0] = 1.0
            if loc_step_prob is not None:
                pa_0 = np.real(psi_a * psi_a.conj())
                pb_0 = np.real(psi_b * psi_b.conj())
                loc_step_prob[0] = pa_0 + pb_0
            elif pair_idx == 0:
                loc_rep_pops[0] = probabilities_array_to_dict(
                    np.real(psi_a * psi_a.conj()), n, max_entries=64
                )

            for s_idx, item in enumerate(evolution_items, start=1):
                if precomputed_ideal is not None:
                    ref_ideal = precomputed_ideal[s_idx]
                else:
                    assert psi_ideal_loc is not None
                    if isinstance(item, list):
                        for g in item:
                            psi_ideal_loc = apply_operator_to_statevector(
                                psi_ideal_loc, g.unitary(), g.qubits, n
                            )
                    ref_ideal = psi_ideal_loc

                psi_a = self._evolve_single_noisy_trajectory(psi_a, item, hardware, n, rng_a)
                psi_b = self._evolve_single_noisy_trajectory(psi_b, item, hardware, n, rng_b)

                fid_a = float(abs(np.vdot(ref_ideal, psi_a)) ** 2)
                fid_b = float(abs(np.vdot(ref_ideal, psi_b)) ** 2)
                loc_fid[s_idx] = fid_a + fid_b
                loc_pur[s_idx] = float(abs(np.vdot(psi_a, psi_b)) ** 2)

                ba = compute_statevector_bloch(psi_a, n)
                bb = compute_statevector_bloch(psi_b, n)
                for q in range(n):
                    loc_bloch[s_idx, q, 0] = ba[q][0] + bb[q][0]
                    loc_bloch[s_idx, q, 1] = ba[q][1] + bb[q][1]
                    loc_bloch[s_idx, q, 2] = ba[q][2] + bb[q][2]

                if loc_step_prob is not None:
                    pa_s = np.real(psi_a * psi_a.conj())
                    pb_s = np.real(psi_b * psi_b.conj())
                    loc_step_prob[s_idx] = pa_s + pb_s
                elif pair_idx == 0:
                    pa_s = np.real(psi_a * psi_a.conj())
                    pb_s = np.real(psi_b * psi_b.conj())
                    loc_rep_pops[s_idx] = probabilities_array_to_dict(
                        0.5 * (pa_s + pb_s), n, max_entries=64
                    )

            loc_final_prob = np.real(psi_a * psi_a.conj()) + np.real(psi_b * psi_b.conj())
            return loc_bloch, loc_fid, loc_pur, loc_final_prob, loc_step_prob, loc_rep_pops

        # Dispatch across CPU threads when workload benefits from parallelism and memory fits easily
        max_workers = min(num_pairs, max(1, min(4, os.cpu_count() or 4)))
        if max_workers > 1 and 8 <= n <= 20:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                pair_outputs = list(pool.map(_simulate_pair, pair_seeds))
        else:
            pair_outputs = [_simulate_pair(ps) for ps in pair_seeds]

        bloch_accum = np.zeros((num_steps, n, 3), dtype=np.float64)
        fidelity_accum = np.zeros(num_steps, dtype=np.float64)
        purity_accum = np.zeros(num_steps, dtype=np.float64)
        final_prob_accum = np.zeros(dim, dtype=np.float64)
        step_prob_accum = (
            np.zeros((num_steps, dim), dtype=np.float64) if accum_step_probs else None
        )
        rep_step_pops: Dict[int, Dict[str, float]] = {}

        for loc_bloch, loc_fid, loc_pur, loc_fprob, loc_sprob, loc_rep in pair_outputs:
            bloch_accum += loc_bloch
            fidelity_accum += loc_fid
            purity_accum += loc_pur
            final_prob_accum += loc_fprob
            if step_prob_accum is not None and loc_sprob is not None:
                step_prob_accum += loc_sprob
            if loc_rep:
                rep_step_pops.update(loc_rep)

        final_prob_accum /= float(total_trajs)

        steps_noisy: List[StepSnapshot] = []
        for s_idx in range(num_steps):
            fid_mean = float(max(0.0, min(1.0, fidelity_accum[s_idx] / float(total_trajs))))
            pur_mean = float(max(0.0, min(1.0, purity_accum[s_idx] / float(num_pairs))))
            # Physical lower bound: Tr(rho^2) >= F(rho, |psi_ideal>)^2 and >= 1/2^n
            pur_mean = max(pur_mean, fid_mean * fid_mean, 1.0 / float(dim))

            b_dict: Dict[int, Tuple[float, float, float]] = {
                q: (
                    float(bloch_accum[s_idx, q, 0] / float(total_trajs)),
                    float(bloch_accum[s_idx, q, 1] / float(total_trajs)),
                    float(bloch_accum[s_idx, q, 2] / float(total_trajs)),
                )
                for q in range(n)
            }
            if step_prob_accum is not None:
                pops = probabilities_array_to_dict(step_prob_accum[s_idx] / float(total_trajs), n)
            elif s_idx == num_steps - 1:
                pops = probabilities_array_to_dict(final_prob_accum, n)
            else:
                pops = rep_step_pops.get(s_idx, {})

            steps_noisy.append(
                StepSnapshot(
                    step_index=s_idx,
                    operation_label=step_labels[s_idx],
                    elapsed_time_ns=step_times[s_idx],
                    ideal_statevector=saved_ideal_vecs.get(s_idx),
                    state_fidelity=fid_mean,
                    purity=pur_mean,
                    bloch_vectors=b_dict,
                    populations=pops,
                )
            )

        counts = sample_bitstrings_from_prob_array(
            final_prob_accum, n, shots=shots, readout_errors=readout_errors, seed=base_seed + 7919
        )
        if readout_errors and n <= 16:
            noisy_final_arr = apply_readout_confusion_array(final_prob_accum, n, readout_errors)
            final_probs = probabilities_array_to_dict(noisy_final_arr, n)
        elif readout_errors and shots > 0 and counts:
            final_probs = {bstr: cnt / float(shots) for bstr, cnt in list(counts.items())[:256]}
        else:
            final_probs = probabilities_array_to_dict(final_prob_accum, n)

        return EvolutionResult(
            backend_name=f"Trajectory-MC ({total_trajs} trajs)",
            num_qubits=n,
            steps=steps_noisy,
            final_probabilities=final_probs,
            measurement_counts=counts,
            shots=shots,
            total_duration_ns=elapsed_ns,
            final_fidelity=steps_noisy[-1].state_fidelity,
            final_purity=steps_noisy[-1].purity,
            metadata={"trajectories": total_trajs, "mode": "monte_carlo_kraus"},
        )
