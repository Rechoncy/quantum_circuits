"""
Tensor-Contracted Density Matrix & Kraus Channel Solver Backend.
Supports arbitrary (Clifford and non-Clifford) quantum gates, inline Kraus noise channels,
and automatic HardwareModel thermal relaxation / depolarizing / readout noise.

Uses O(2^k * 4^N) tensor-axis contraction (`apply_operator_to_density_matrix`) instead of
O(8^N) dense 2^N x 2^N matrix multiplication, accelerating 10-14 qubit noisy simulations
by orders of magnitude.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..core.circuit import (
    GateInstruction,
    MeasureInstruction,
    NoiseInstruction,
    QuantumCircuit,
)
from ..core.hardware import HardwareModel
from ..core.noise import NoiseChannel, ReadoutError
from .base import (
    EvolutionResult,
    StepSnapshot,
    apply_operator_to_density_matrix,
    apply_operator_to_statevector,
    apply_readout_confusion_array,
    compute_bloch_vectors,
    compute_populations,
    mitigate_readout_probabilities,
    probabilities_array_to_dict,
    sample_from_probabilities,
    state_fidelity_with_pure,
)


def apply_kraus_channel(
    rho: np.ndarray,
    channel: NoiseChannel,
    num_qubits: int,
) -> np.ndarray:
    """Apply a 1-qubit or 2-qubit Kraus channel to an N-qubit density matrix rho via tensor contraction."""
    kraus_ops = channel.kraus_operators()
    new_rho = np.zeros_like(rho)
    for K in kraus_ops:
        new_rho += apply_operator_to_density_matrix(rho, K, channel.qubits, num_qubits)
    # Symmetrize and normalize trace to prevent numerical drift
    new_rho = 0.5 * (new_rho + new_rho.conj().T)
    tr = float(np.real(np.trace(new_rho)))
    if tr > 0:
        new_rho /= tr
    return new_rho


def _fast_purity(rho: np.ndarray) -> float:
    """Compute Tr(rho^2) = sum_{i,j} |rho_{i,j}|^2 for Hermitian rho in O(4^N) instead of O(8^N) matmul!"""
    # Since rho is Hermitian, Tr(rho @ rho) = Tr(rho @ rho^dagger) = ||rho||_F^2 !
    flat = rho.ravel()
    return float(np.real(np.vdot(flat, flat)))


class KrausBackend:
    """
    Exact gate-level density matrix and Kraus operator evolution engine (tensor-contracted).
    Scales cleanly up to ~13-14 qubits for exact mixed-state density matrix simulation.
    """

    MAX_DENSITY_MATRIX_QUBITS = 14

    def run(
        self,
        circuit: QuantumCircuit,
        hardware: Optional[HardwareModel] = None,
        shots: int = 4096,
        seed: Optional[int] = 42,
        initial_state: Optional[np.ndarray] = None,
    ) -> EvolutionResult:
        n = circuit.num_qubits
        if n > self.MAX_DENSITY_MATRIX_QUBITS:
            raise ValueError(
                f"Kraus 稠密密度矩阵后端最高支持 {self.MAX_DENSITY_MATRIX_QUBITS} 比特（2^{n}×2^{n} 矩阵需要 "
                f"{(16 * (1 << (2 * n))) / (1024**3):.1f} GB 内存）。请切换至「Trajectory · 态矢量轨迹 (≤24Q)」或「Stim · 稳定子 (500+Q)」后端。"
            )
        dim = 1 << n

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

        # Keep step-by-step full rho copies only when n <= 8 to save gigabytes of RAM at n=11..14
        store_step_rho = n <= 8

        steps: List[StepSnapshot] = []
        elapsed_ns = 0.0
        step_idx = 0

        steps.append(
            StepSnapshot(
                step_index=0,
                operation_label="INIT |0...0>",
                elapsed_time_ns=0.0,
                density_matrix=rho.copy() if store_step_rho else None,
                ideal_statevector=psi_ideal.copy() if store_step_rho else None,
                state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                purity=_fast_purity(rho),
                bloch_vectors=compute_bloch_vectors(rho, n),
                populations=compute_populations(rho, n),
            )
        )

        readout_errors: Dict[int, ReadoutError] = {}
        if hardware is not None:
            for q in range(n):
                readout_errors[q] = hardware.qubit_params[q].readout_error

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
                busy_ns: Dict[int, float] = {}
                for g in layer_gates:
                    uname = g.name.upper()
                    if uname != "BARRIER":
                        u_small = g.unitary()
                        psi_ideal = apply_operator_to_statevector(psi_ideal, u_small, g.qubits, n)
                        rho = apply_operator_to_density_matrix(rho, u_small, g.qubits, n)
                    dt_g = (
                        0.0
                        if uname == "BARRIER"
                        else (
                            float(g.params[0])
                            if uname in ("IDLE", "DELAY") and g.params
                            else (
                                g.duration_ns
                                if g.duration_ns is not None
                                else (hardware.gate_duration_ns(g.name, g.qubits) if hardware is not None else 20.0)
                            )
                        )
                    )
                    dt_val = float(max(0.0, dt_g))
                    gate_dts.append(dt_val)
                    for q_u in g.qubits:
                        busy_ns[q_u] = max(busy_ns.get(q_u, 0.0), dt_val if not g.skip_hw_noise else 0.0)

                dt_layer = max(gate_dts) if gate_dts else 0.0
                elapsed_ns += dt_layer

                all_ideal_layer = all(g.skip_hw_noise or g.name.upper() == "BARRIER" for g in layer_gates)
                # Apply static ZZ crosstalk phase across coupled pairs over dt_layer (aligned with QuTiP H_drift)
                if hardware is not None and dt_layer > 0 and not all_ideal_layer and hardware.coupler_params:
                    for (qa, qb), cp in hardware.coupler_params.items():
                        if abs(cp.zz_crosstalk_mhz) > 1e-12 and qa < n and qb < n:
                            phi_zz = 0.5 * math.pi * cp.zz_crosstalk_mhz * 1e-3 * dt_layer
                            if abs(phi_zz) > 1e-12:
                                em = complex(math.cos(phi_zz), -math.sin(phi_zz))
                                ep = complex(math.cos(phi_zz), math.sin(phi_zz))
                                u_zz = np.diag([em, ep, ep, em])
                                rho = apply_operator_to_density_matrix(rho, u_zz, (qa, qb), n)

                for g, dt_g in zip(layer_gates, gate_dts):
                    for ch in g.noise_after:
                        rho = apply_kraus_channel(rho, ch, n)
                    is_idle_op = g.name.upper() in ("IDLE", "DELAY", "BARRIER")
                    if hardware is not None and dt_g > 0 and not g.skip_hw_noise:
                        if not is_idle_op:
                            if len(g.qubits) == 1:
                                q0 = g.qubits[0]
                                p_dep = hardware.qubit_params[q0].depol_1q
                                if p_dep > 0:
                                    rho = apply_kraus_channel(rho, NoiseChannel.depolarizing_1q(q0, p_dep), n)
                            elif len(g.qubits) == 2:
                                pair = (min(g.qubits[0], g.qubits[1]), max(g.qubits[0], g.qubits[1]))
                                if pair in hardware.coupler_params:
                                    p_dep2 = hardware.coupler_params[pair].depol_2q
                                    if p_dep2 > 0:
                                        rho = apply_kraus_channel(
                                            rho, NoiseChannel.depolarizing_2q(g.qubits[0], g.qubits[1], p_dep2), n
                                        )
                        for q_act in g.qubits:
                            qp = hardware.qubit_params[q_act]
                            ch_th = NoiseChannel.thermal_relaxation(q_act, qp.t1_ns, qp.t2_ns, dt_g)
                            rho = apply_kraus_channel(rho, ch_th, n)

                if hardware is not None and dt_layer > 0 and not all_ideal_layer and hardware.include_idle_noise:
                    for q_idle in range(n):
                        dt_wait = dt_layer - busy_ns.get(q_idle, 0.0)
                        if dt_wait > 1e-9:
                            qp = hardware.qubit_params[q_idle]
                            ch_idle = NoiseChannel.thermal_relaxation(q_idle, qp.t1_ns, qp.t2_ns, dt_wait)
                            rho = apply_kraus_channel(rho, ch_idle, n)

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
                        density_matrix=rho.copy() if store_step_rho else None,
                        ideal_statevector=psi_ideal.copy() if store_step_rho else None,
                        state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                        purity=_fast_purity(rho),
                        bloch_vectors=compute_bloch_vectors(rho, n),
                        populations=compute_populations(rho, n),
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
                        density_matrix=rho.copy() if store_step_rho else None,
                        ideal_statevector=psi_ideal.copy() if store_step_rho else None,
                        state_fidelity=state_fidelity_with_pure(rho, psi_ideal),
                        purity=_fast_purity(rho),
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

        raw_diag = np.clip(np.real(np.diag(rho)), 0.0, 1.0)
        noisy_diag = apply_readout_confusion_array(raw_diag, n, readout_errors)
        final_probs = probabilities_array_to_dict(noisy_diag, n)
        counts = sample_from_probabilities(final_probs, shots=shots, seed=seed)
        emp_probs = {k: v / float(shots) for k, v in counts.items()} if shots > 0 and counts else final_probs
        mitigated_probs = mitigate_readout_probabilities(emp_probs, n, readout_errors)

        return EvolutionResult(
            backend_name="Kraus-DensityMatrix",
            num_qubits=n,
            steps=steps,
            final_probabilities=final_probs,
            measurement_counts=counts,
            shots=shots,
            total_duration_ns=elapsed_ns,
            final_fidelity=steps[-1].state_fidelity,
            final_purity=steps[-1].purity,
            metadata={"mitigated_probabilities": mitigated_probs},
        )
