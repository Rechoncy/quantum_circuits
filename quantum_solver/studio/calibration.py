"""
Basic single-qubit calibration experiments for the Characterization workspace:

  * Rabi   - drive with a flat-top pulse of variable length, fit P1(t) to extract the true
             Rabi frequency Omega_R, pi / pi-2 pulse lengths and drive-amplitude miscalibration.
  * Ramsey - pi/2 . free precession(tau) . pi/2 with an artificial detuning, fit decaying fringes
             to extract T2* and the qubit-frequency offset.
  * T1     - pi pulse then wait(tau), fit exponential energy relaxation.

All experiments integrate the exact single-qubit Lindblad master equation (4x4 Liouvillian),
include readout confusion and optional binomial shot noise, and return suggested write-back
values for the Device workspace.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from .models import CircuitOpModel, GateBenchmarkRequest
from .project_engine import get_active_hardware_profile, rabi_pulse_duration_ns

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_SM = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1| lowering (|0> = ground)


def _liouvillian(omega_mhz: float, delta_mhz: float, phase: float, t1_ns: float, t2_ns: float) -> np.ndarray:
    """Row-major vectorized Liouvillian (rad / ns) for a driven, decohering qubit in the rotating frame."""
    wx = 2.0 * math.pi * omega_mhz * 1e-3
    wz = 2.0 * math.pi * delta_mhz * 1e-3
    h = 0.5 * wx * (math.cos(phase) * _SX + math.sin(phase) * _SY) + 0.5 * wz * _SZ
    lv = -1j * (np.kron(h, _I2) - np.kron(_I2, h.T))

    def diss(c: np.ndarray) -> np.ndarray:
        cdc = c.conj().T @ c
        return np.kron(c, c.conj()) - 0.5 * np.kron(cdc, _I2) - 0.5 * np.kron(_I2, cdc.T)

    if math.isfinite(t1_ns) and t1_ns > 0:
        lv = lv + diss(math.sqrt(1.0 / t1_ns) * _SM)
        g1 = 1.0 / t1_ns
    else:
        g1 = 0.0
    if math.isfinite(t2_ns) and t2_ns > 0:
        g_phi = max(0.0, 1.0 / t2_ns - 0.5 * g1)
        if g_phi > 0:
            lv = lv + diss(math.sqrt(0.5 * g_phi) * _SZ)
    return lv


def _unitary_super(u: np.ndarray) -> np.ndarray:
    return np.kron(u, u.conj())


def _rot(theta: float, phase: float) -> np.ndarray:
    n = math.cos(phase) * _SX + math.sin(phase) * _SY
    return math.cos(0.5 * theta) * _I2 - 1j * math.sin(0.5 * theta) * n


def _p1(rho_vec: np.ndarray) -> float:
    return float(np.real(rho_vec[3]))


def _measure(p1: np.ndarray, p10: float, p01: float, shots: int, rng: np.random.Generator) -> np.ndarray:
    p_meas = p1 * (1.0 - p01) + (1.0 - p1) * p10
    p_meas = np.clip(p_meas, 0.0, 1.0)
    if shots > 0:
        return rng.binomial(shots, p_meas) / float(shots)
    return p_meas


def _fmt(val: float, nd: int = 4) -> float:
    return float(round(val, nd)) if math.isfinite(val) else float("nan")


def run_basic_calibration(req: GateBenchmarkRequest, noise: Dict[str, Any]) -> Dict[str, Any]:
    project = req.project
    hp = get_active_hardware_profile(project)
    q = max(0, min(len(project.qubits) - 1, int(req.target_qubit)))
    q_spec = project.qubits[q]
    q_name = q_spec.name

    t1 = float(noise["t1_ns"])
    t2 = float(noise["t2_ns"])
    p10 = float(noise["p1_given_0"])
    p01 = float(noise["p0_given_1"])
    shots = max(0, int(req.shots))
    n_pts = max(11, min(301, int(req.cal_points)))
    rng = np.random.default_rng(req.seed)

    omega_set = float(q_spec.rabi_mhz if q_spec.rabi_mhz is not None else hp.rabi_mhz)
    omega_true = omega_set * (1.0 + req.amp_error_pct / 100.0)
    det_true = float(req.detuning_mhz)

    rho0 = np.array([1, 0, 0, 0], dtype=np.complex128)
    params: List[Dict[str, Any]] = []
    writeback: List[Dict[str, Any]] = []

    if req.protocol == "rabi":
        period_ns = 1000.0 / max(1e-6, omega_true)
        t_max = float(req.cal_max_ns) if req.cal_max_ns else 3.0 * period_ns
        xs = np.linspace(0.0, t_max, n_pts)
        lv = _liouvillian(omega_true, det_true, 0.0, t1, t2)
        dt = xs[1] - xs[0]
        step = expm(lv * dt)
        p1_true = np.zeros(n_pts)
        r = rho0.copy()
        for i in range(n_pts):
            p1_true[i] = _p1(r)
            r = step @ r
        ys = _measure(p1_true, p10, p01, shots, rng)

        def model(t, a, b, f, tau):
            return a - b * np.exp(-t / tau) * np.cos(2.0 * math.pi * f * t)

        f0 = math.sqrt(omega_true ** 2 + det_true ** 2) * 1e-3
        try:
            popt, _ = curve_fit(
                model, xs, ys, p0=[0.5, 0.5, f0, max(t_max * 4, 1.0)],
                bounds=([0.0, 0.0, 1e-7, 1.0], [1.0, 1.0, 1.0, 1e9]), maxfev=8000,
            )
        except Exception:
            popt = np.array([0.5, 0.5, f0, 1e9])
        a, b, f_fit, tau = [float(v) for v in popt]
        omega_fit = f_fit * 1e3
        t_pi_rect = 0.5 / f_fit if f_fit > 0 else float("nan")
        t_pi_env = rabi_pulse_duration_ns(math.pi, omega_fit, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
        t_pi2_env = rabi_pulse_duration_ns(0.5 * math.pi, omega_fit, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
        fit_x = np.linspace(0.0, t_max, 300)
        fit_y = model(fit_x, *popt)
        params = [
            {"key": "omega_fit", "label": "拟合拉比频率 Ω_R", "value": _fmt(omega_fit, 5), "unit": "MHz"},
            {"key": "omega_set", "label": "器件设定 Ω_R", "value": _fmt(omega_set, 5), "unit": "MHz"},
            {"key": "t_pi_env", "label": f"π 脉冲时长 ({hp.pulse_shape})", "value": _fmt(t_pi_env, 2), "unit": "ns"},
            {"key": "t_pi2_env", "label": f"π/2 脉冲时长 ({hp.pulse_shape})", "value": _fmt(t_pi2_env, 2), "unit": "ns"},
            {"key": "t_pi", "label": "方波基准 π 时长", "value": _fmt(t_pi_rect, 2), "unit": "ns"},
            {"key": "tau", "label": "拉比振荡衰减 τ_R", "value": _fmt(tau / 1000.0, 3), "unit": "μs"},
            {"key": "contrast", "label": "振荡对比度 2B", "value": _fmt(2 * b, 4), "unit": ""},
        ]
        writeback = [
            {"scope": "qubit", "qubit": q, "field": "rabi_mhz", "value": _fmt(omega_fit, 5),
             "label": f"写回 {q_name} Ω_R = {omega_fit:.4f} MHz"},
            {"scope": "hardware", "field": "rabi_mhz", "value": _fmt(omega_fit, 5),
             "label": f"写回硬件默认 Ω_R = {omega_fit:.4f} MHz"},
            {"scope": "hardware", "field": "t_1q_ns", "value": _fmt(t_pi_env, 2),
             "label": f"写回硬件 t_1q = {t_pi_env:.1f} ns"},
        ]
        summary = (
            f"{q_name} 实测拉比频率 Ω_R = {omega_fit:.4f} MHz（设定 {omega_set:.4f} MHz，"
            f"偏差 {100 * (omega_fit / omega_set - 1):+.2f}%），当前包络 π 脉冲 ≈ {t_pi_env:.1f} ns。"
        )
        x_label, y_label = "驱动脉冲时长 t (ns)", "激发态概率 P(|1⟩)"

    elif req.protocol == "ramsey":
        art = float(req.ramsey_detuning_mhz)
        det_tot = art + det_true
        if req.cal_max_ns:
            t_max = float(req.cal_max_ns)
        else:
            fringes = 8.0 / max(1e-4, abs(det_tot)) * 1000.0 if abs(det_tot) > 1e-6 else 2e4
            t_max = min(fringes, 3.0 * t2) if math.isfinite(t2) else fringes
        xs = np.linspace(0.0, t_max, n_pts)
        u90 = _unitary_super(_rot(0.5 * math.pi, 0.0))
        lv_free = _liouvillian(0.0, det_tot, 0.0, t1, t2)
        dt = xs[1] - xs[0]
        step = expm(lv_free * dt)
        r = u90 @ rho0
        p1_true = np.zeros(n_pts)
        for i in range(n_pts):
            p1_true[i] = _p1(u90 @ r)
            r = step @ r
        ys = _measure(p1_true, p10, p01, shots, rng)

        def model(t, a, b, f, t2s, ph):
            return a + b * np.exp(-t / t2s) * np.cos(2.0 * math.pi * f * t + ph)

        f0 = abs(det_tot) * 1e-3
        t2_guess = t2 if math.isfinite(t2) else t_max * 10
        best = None
        for ph0 in (0.0, math.pi):
            try:
                popt, _ = curve_fit(
                    model, xs, ys, p0=[0.5, 0.5, max(f0, 1e-7), t2_guess, ph0],
                    bounds=([0.0, 0.0, 0.0, 1.0, -2 * math.pi], [1.0, 1.0, 1.0, 1e10, 2 * math.pi]),
                    maxfev=10000,
                )
                res = float(np.sum((model(xs, *popt) - ys) ** 2))
                if best is None or res < best[1]:
                    best = (popt, res)
            except Exception:
                continue
        popt = best[0] if best else np.array([0.5, 0.5, f0, t2_guess, 0.0])
        a, b, f_fit, t2s, ph = [float(v) for v in popt]
        f_mhz = f_fit * 1e3
        offset = f_mhz - abs(art)
        fit_x = np.linspace(0.0, t_max, 400)
        fit_y = model(fit_x, *popt)
        params = [
            {"key": "t2_star", "label": "拟合退相干时间 T2*", "value": _fmt(t2s / 1000.0, 3), "unit": "μs"},
            {"key": "fringe", "label": "Ramsey 条纹频率", "value": _fmt(f_mhz, 5), "unit": "MHz"},
            {"key": "art_det", "label": "人工失谐 δ_art", "value": _fmt(art, 4), "unit": "MHz"},
            {"key": "freq_offset", "label": "估计比特频率偏移 Δ", "value": _fmt(offset, 5), "unit": "MHz"},
            {"key": "t2_set", "label": "器件设定 T2*", "value": _fmt(t2 / 1000.0, 3) if math.isfinite(t2) else float("nan"), "unit": "μs"},
        ]
        writeback = [
            {"scope": "qubit", "qubit": q, "field": "t2_us", "value": _fmt(t2s / 1000.0, 3),
             "label": f"{q_name} T2* = {t2s / 1000.0:.2f} μs"},
        ]
        if q_spec.freq_ghz is not None and abs(offset) > 1e-4:
            writeback.append(
                {"scope": "qubit", "qubit": q, "field": "freq_ghz", "value": _fmt(q_spec.freq_ghz + offset * 1e-3, 7),
                 "label": f"{q_name} f_q 修正 {offset:+.4f} MHz"}
            )
        summary = f"{q_name} 拟合 T2* = {t2s / 1000.0:.2f} μs，条纹频率 {f_mhz:.4f} MHz，比特频率偏移 Δ ≈ {offset:+.4f} MHz。"
        x_label, y_label = "自由演化间隔 τ (ns)", "激发态概率 P(|1⟩)"

    else:  # T1
        t_max = float(req.cal_max_ns) if req.cal_max_ns else (4.0 * t1 if math.isfinite(t1) else 1e5)
        xs = np.linspace(0.0, t_max, n_pts)
        upi = _unitary_super(_rot(math.pi, 0.0))
        lv_free = _liouvillian(0.0, det_true, 0.0, t1, t2)
        dt = xs[1] - xs[0]
        step = expm(lv_free * dt)
        r = upi @ rho0
        p1_true = np.zeros(n_pts)
        for i in range(n_pts):
            p1_true[i] = _p1(r)
            r = step @ r
        ys = _measure(p1_true, p10, p01, shots, rng)

        def model(t, a, t1f, c):
            return a * np.exp(-t / t1f) + c

        t1_guess = t1 if math.isfinite(t1) else t_max * 10
        try:
            popt, _ = curve_fit(model, xs, ys, p0=[0.95, t1_guess, 0.02],
                                bounds=([0.0, 1.0, -0.2], [1.2, 1e10, 0.5]), maxfev=8000)
        except Exception:
            popt = np.array([0.95, t1_guess, 0.02])
        a, t1f, c = [float(v) for v in popt]
        fit_x = np.linspace(0.0, t_max, 300)
        fit_y = model(fit_x, *popt)
        params = [
            {"key": "t1", "label": "拟合能量弛豫时间 T1", "value": _fmt(t1f / 1000.0, 3), "unit": "μs"},
            {"key": "amp", "label": "初始激发幅度 A", "value": _fmt(a, 4), "unit": ""},
            {"key": "offset", "label": "读出基线 C", "value": _fmt(c, 4), "unit": ""},
            {"key": "t1_set", "label": "器件设定 T1", "value": _fmt(t1 / 1000.0, 3) if math.isfinite(t1) else float("nan"), "unit": "μs"},
        ]
        writeback = [
            {"scope": "qubit", "qubit": q, "field": "t1_us", "value": _fmt(t1f / 1000.0, 3),
             "label": f"{q_name} T1 = {t1f / 1000.0:.2f} μs"},
        ]
        summary = f"{q_name} 拟合 T1 = {t1f / 1000.0:.2f} μs（读出基线 {c:.3f}，激发幅度 {a:.3f}）。"
        x_label, y_label = "π 脉冲后等待时间 τ (ns)", "激发态概率 P(|1⟩)"

    circuit_samples: List[Dict[str, Any]] = []
    if req.protocol == "rabi":
        circuit_samples = [
            {
                "id": "rabi_var",
                "label": "Rabi 变宽共振驱动序列",
                "subtitle": f"{q_name}: RXY(Ω_R·t, 0) → M",
                "operations": [
                    CircuitOpModel(id="cal_r0", step=0, kind="gate", name="RXY", qubits=[q], params=["theta", 0.0], rabi_mhz=round(omega_set, 4)).model_dump(),
                    CircuitOpModel(id="cal_rm", step=1, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            },
            {
                "id": "rabi_pi2",
                "label": "校准后 π/2 脉冲 (SX)",
                "subtitle": f"Ω_R = {params[0]['value']} MHz",
                "operations": [
                    CircuitOpModel(id="cal_p2", step=0, kind="gate", name="SX", qubits=[q], params=[], rabi_mhz=params[0]["value"]).model_dump(),
                    CircuitOpModel(id="cal_p2m", step=1, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            },
            {
                "id": "rabi_pi",
                "label": "校准后 π 翻转脉冲 (X)",
                "subtitle": f"Ω_R = {params[0]['value']} MHz",
                "operations": [
                    CircuitOpModel(id="cal_pi", step=0, kind="gate", name="X", qubits=[q], params=[], rabi_mhz=params[0]["value"]).model_dump(),
                    CircuitOpModel(id="cal_pim", step=1, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            },
        ]
    elif req.protocol == "ramsey":
        circuit_samples = [
            {
                "id": "ramsey_seq",
                "label": "Ramsey π/2 − τ − π/2 干涉序列",
                "subtitle": f"{q_name}: SX → 自由进动 τ (δ={req.ramsey_detuning_mhz:g}MHz) → SX → M",
                "operations": [
                    CircuitOpModel(id="ram_0", step=0, kind="gate", name="SX", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="ram_idle", step=1, kind="gate", name="RZ", qubits=[q], params=[round(0.5 * math.pi, 4)]).model_dump(),
                    CircuitOpModel(id="ram_deph", step=2, kind="noise", name="PHASE_DAMPING", qubits=[q], params=[0.05]).model_dump(),
                    CircuitOpModel(id="ram_1", step=3, kind="gate", name="SX", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="ram_m", step=4, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            }
        ]
    else:
        circuit_samples = [
            {
                "id": "t1_seq",
                "label": "T1 激发态弛豫测量序列",
                "subtitle": f"{q_name}: X(π) → 自由弛豫等待 τ → M",
                "operations": [
                    CircuitOpModel(id="t1_x", step=0, kind="gate", name="X", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="t1_ad", step=1, kind="noise", name="AMPLITUDE_DAMPING", qubits=[q], params=[0.1]).model_dump(),
                    CircuitOpModel(id="t1_m", step=2, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            }
        ]

    return {
        "protocol": req.protocol,
        "hardware_name": str(noise.get("hardware_name", hp.name)),
        "target_label": q_name,
        "qubit": q,
        "shots": shots,
        "x": [float(round(v, 4)) for v in xs],
        "measured": [float(round(v, 6)) for v in ys],
        "ideal": [float(round(v, 6)) for v in p1_true],
        "fit_x": [float(round(v, 4)) for v in fit_x],
        "fit_y": [float(round(v, 6)) for v in fit_y],
        "x_label": x_label,
        "y_label": y_label,
        "params": params,
        "writeback": writeback,
        "circuit_samples": circuit_samples,
        "summary": summary,
    }
