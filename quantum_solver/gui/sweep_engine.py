"""
1D and 2D Parameter Sweep Engine for Quantum Circuits and Experimental Hardware Models.
Supports sweeping:
  - Symbolic circuit parameters (e.g. gate angles `theta`, `phi`, inline noise `p_noise`)
  - Hardware physical parameters (`hw:t1_us`, `hw:t2_us`, `hw:depol_1q`, `hw:depol_2q`,
    `hw:t_2q_ns`, `hw:zz_khz`, `hw:readout_err`)
across QuTiP, Kraus, Stim, or Hybrid Stim+QuTiP backends.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.hardware import HardwareModel
from ..core.noise import ReadoutError
from ..solver.unified_solver import QuantumSolver
from .circuit_parser import parse_circuit_script


HARDWARE_SWEEP_PARAMS: Dict[str, str] = {
    "hw:t1_us": "硬件 T1 弛豫时间 (μs)",
    "hw:t2_us": "硬件 T2* 退相干时间 (μs)",
    "hw:depol_1q": "单比特门退极化错误率",
    "hw:depol_2q": "两比特门退极化错误率",
    "hw:t_1q_ns": "单比特门脉冲时长 (ns)",
    "hw:rabi_mhz": "单比特拉比频率 Ω_R (MHz)",
    "hw:rise_ns": "脉冲上升沿时长 t_rise (ns)",
    "hw:fall_ns": "脉冲下降沿时长 t_fall (ns)",
    "hw:anharmonicity_mhz": "Transmon 非谐性 α/2π (MHz)",
    "hw:drag_beta": "DRAG 正交导数系数 β",
    "hw:t_2q_ns": "两比特门脉冲时长 (ns)",
    "hw:zz_khz": "静态 ZZ 串扰强度 (kHz)",
    "hw:readout_err": "测量读出翻转错误率",
}


def apply_hardware_param(hw: HardwareModel, key: str, val: float) -> HardwareModel:
    """Return a modified copy of HardwareModel with the swept hardware parameter updated."""
    new_hw = copy.deepcopy(hw)
    v = float(val)
    if key == "hw:t1_us":
        for qp in new_hw.qubit_params.values():
            qp.t1_ns = max(1.0, v * 1000.0)
    elif key == "hw:t2_us":
        for qp in new_hw.qubit_params.values():
            qp.t2_ns = max(1.0, v * 1000.0)
    elif key == "hw:depol_1q":
        for qp in new_hw.qubit_params.values():
            qp.depol_1q = max(0.0, min(1.0, v))
        for g in new_hw.native_gate_set.single_qubit_gates:
            if not g.is_virtual:
                g.infidelity = max(0.0, min(1.0, v))
    elif key == "hw:depol_2q":
        for cp in new_hw.coupler_params.values():
            cp.depol_2q = max(0.0, min(1.0, v))
        for g in new_hw.native_gate_set.two_qubit_gates:
            g.infidelity = max(0.0, min(1.0, v))
    elif key == "hw:t_1q_ns":
        for g in new_hw.native_gate_set.single_qubit_gates:
            if not g.is_virtual:
                g.duration_ns = max(0.1, v)
    elif key == "hw:anharmonicity_mhz":
        for qp in new_hw.qubit_params.values():
            qp.anharmonicity_ghz = v * 1e-3
    elif key == "hw:drag_beta":
        new_hw.drag_beta = v
        for qp in new_hw.qubit_params.values():
            qp.drag_beta = v
    elif key == "hw:t_2q_ns":
        for cp in new_hw.coupler_params.values():
            cp.gate_duration_ns = max(0.1, v)
        for g in new_hw.native_gate_set.two_qubit_gates:
            g.duration_ns = max(0.1, v)
    elif key == "hw:zz_khz":
        for cp in new_hw.coupler_params.values():
            cp.zz_crosstalk_mhz = v / 1000.0
    elif key == "hw:readout_err":
        p = max(0.0, min(0.5, v))
        for qp in new_hw.qubit_params.values():
            qp.readout_error = ReadoutError(p, p * 1.1)
    return new_hw


@dataclass
class Sweep1DResult:
    """Result container for a 1D parameter sweep."""

    param_name: str
    param_label: str
    x_values: np.ndarray
    backend_name: str
    num_qubits: int
    populations: Dict[str, np.ndarray]        # bitstring -> shape (N_pts,)
    fidelities: np.ndarray                    # shape (N_pts,)
    purities: np.ndarray                      # shape (N_pts,)
    bloch_expectations: Dict[int, np.ndarray] # qubit -> shape (N_pts, 3) for (<X>, <Y>, <Z>)


@dataclass
class Sweep2DResult:
    """Result container for a 2D parameter sweep."""

    param_x_name: str
    param_y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    backend_name: str
    target_bitstring: str
    population_grid: np.ndarray  # shape (len(y_values), len(x_values))
    fidelity_grid: np.ndarray    # shape (len(y_values), len(x_values))
    z0_grid: np.ndarray          # shape (len(y_values), len(x_values))


class ParameterSweepEngine:
    """Executes 1D and 2D parameter sweeps over circuit or hardware parameters."""

    def run_1d_sweep(
        self,
        script: str,
        base_variables: Dict[str, float],
        base_hardware: HardwareModel,
        param_name: str,
        start_val: float,
        stop_val: float,
        num_points: int = 25,
        backend: str = "qutip",
        shots: int = 4096,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Sweep1DResult:
        n_pts = max(2, int(num_points))
        x_vals = np.linspace(float(start_val), float(stop_val), n_pts)

        # Parse once with base variables to determine num_qubits
        probe_vars = dict(base_variables)
        if not param_name.startswith("hw:"):
            probe_vars[param_name] = float(x_vals[0])
        probe_circ = parse_circuit_script(script, probe_vars)
        n_q = probe_circ.num_qubits
        all_bstrs = [format(i, f"0{n_q}b") for i in range(1 << min(n_q, 8))]

        pops: Dict[str, np.ndarray] = {b: np.zeros(n_pts, dtype=np.float64) for b in all_bstrs}
        fidelities = np.zeros(n_pts, dtype=np.float64)
        purities = np.zeros(n_pts, dtype=np.float64)
        bloch_exp: Dict[int, np.ndarray] = {q: np.zeros((n_pts, 3), dtype=np.float64) for q in range(n_q)}

        solver = QuantumSolver(hardware=base_hardware, intra_gate_steps=10)
        b_label = backend

        for idx, x_v in enumerate(x_vals):
            curr_vars = dict(base_variables)
            curr_hw = base_hardware
            if param_name.startswith("hw:"):
                curr_hw = apply_hardware_param(base_hardware, param_name, float(x_v))
            else:
                curr_vars[param_name] = float(x_v)

            circ = parse_circuit_script(script, curr_vars)
            res = solver.solve(circ, backend=backend, hardware=curr_hw, shots=shots, seed=42 + idx)
            b_label = res.backend_name

            for b in all_bstrs:
                pops[b][idx] = res.final_probabilities.get(b, 0.0)
            fidelities[idx] = res.final_fidelity if res.final_fidelity is not None else 1.0
            purities[idx] = res.final_purity if res.final_purity is not None else 1.0

            last_step = res.steps[-1]
            for q in range(n_q):
                if q in last_step.bloch_vectors:
                    bloch_exp[q][idx] = last_step.bloch_vectors[q]

            if progress_cb is not None:
                progress_cb(idx + 1, n_pts)

        p_label = HARDWARE_SWEEP_PARAMS.get(param_name, f"线路参数 {param_name}")
        return Sweep1DResult(
            param_name=param_name,
            param_label=p_label,
            x_values=x_vals,
            backend_name=b_label,
            num_qubits=n_q,
            populations=pops,
            fidelities=fidelities,
            purities=purities,
            bloch_expectations=bloch_exp,
        )

    def run_2d_sweep(
        self,
        script: str,
        base_variables: Dict[str, float],
        base_hardware: HardwareModel,
        param_x: str,
        x_range: Tuple[float, float, int],
        param_y: str,
        y_range: Tuple[float, float, int],
        target_bitstring: str = "00",
        backend: str = "kraus",
        shots: int = 2048,
    ) -> Sweep2DResult:
        nx = max(2, int(x_range[2]))
        ny = max(2, int(y_range[2]))
        x_vals = np.linspace(float(x_range[0]), float(x_range[1]), nx)
        y_vals = np.linspace(float(y_range[0]), float(y_range[1]), ny)

        pop_grid = np.zeros((ny, nx), dtype=np.float64)
        fid_grid = np.zeros((ny, nx), dtype=np.float64)
        z0_grid = np.zeros((ny, nx), dtype=np.float64)

        solver = QuantumSolver(hardware=base_hardware, intra_gate_steps=8)
        b_label = backend

        for iy, y_v in enumerate(y_vals):
            for ix, x_v in enumerate(x_vals):
                curr_vars = dict(base_variables)
                curr_hw = base_hardware
                for pname, pval in ((param_x, x_v), (param_y, y_v)):
                    if pname.startswith("hw:"):
                        curr_hw = apply_hardware_param(curr_hw, pname, float(pval))
                    else:
                        curr_vars[pname] = float(pval)

                circ = parse_circuit_script(script, curr_vars)
                t_bstr = target_bitstring if len(target_bitstring) == circ.num_qubits else "0" * circ.num_qubits
                res = solver.solve(circ, backend=backend, hardware=curr_hw, shots=shots, seed=100 + iy * nx + ix)
                b_label = res.backend_name

                pop_grid[iy, ix] = res.final_probabilities.get(t_bstr, 0.0)
                fid_grid[iy, ix] = res.final_fidelity if res.final_fidelity is not None else 1.0
                z0_grid[iy, ix] = res.steps[-1].bloch_vectors.get(0, (0.0, 0.0, 0.0))[2]

        return Sweep2DResult(
            param_x_name=param_x,
            param_y_name=param_y,
            x_values=x_vals,
            y_values=y_vals,
            backend_name=b_label,
            target_bitstring=target_bitstring,
            population_grid=pop_grid,
            fidelity_grid=fid_grid,
            z0_grid=z0_grid,
        )
