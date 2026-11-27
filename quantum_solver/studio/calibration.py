"""
Basic single-qubit calibration experiments for the Characterization workspace (`Quantum Studio v2.8`):

  * Rabi    - variable-length drive integrated over the physical pulse envelope (cosine, gaussian,
              flat-top rectangular with rise/fall edges, trapezoid), fitting true peak Omega_R.
  * Chevron - 2D frequency-detuning x pulse-length Rabi Chevron scan extracting both resonance
              offset Delta_0 and Rabi frequency Omega_R.
  * Ramsey  - pi/2 . free precession(tau) . pi/2 with artificial detuning, extracting T2* and Delta_0.
  * Echo    - Hahn Spin Echo (N=1) and Meiboom-Gill CPMG-N dynamical decoupling sequence
              X_{pi/2} - [tau/(2N) - Y_pi - tau/(2N)]^N - X_{pi/2}, refocusing static detuning and
              low-frequency quasistatic dephasing to extract T2_echo.
  * T1      - pi pulse then wait(tau), fitting exponential energy relaxation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from .models import CircuitOpModel, GateBenchmarkRequest
from .project_engine import (
    envelope_area_factor,
    get_active_hardware_profile,
    rabi_pulse_duration_ns,
)

_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_SM = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0><1| lowering (|0> = ground)

_GX_SUPER = -1j * (np.kron(0.5 * _SX, _I2) - np.kron(_I2, 0.5 * _SX.T))
_GZ_SUPER = -1j * (np.kron(0.5 * _SZ, _I2) - np.kron(_I2, 0.5 * _SZ.T))


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


def _sample_peak_envelope_vec(
    t_ns: np.ndarray,
    dur_ns: np.ndarray,
    pulse_shape: str,
    rise_ns: float,
    fall_ns: float,
) -> np.ndarray:
    """Vectorized peak-normalized envelope e_peak(t_ns; dur_ns) in [0, 1]."""
    shape = (pulse_shape or "cosine").lower()
    out = np.zeros_like(t_ns, dtype=np.float64)
    valid = dur_ns > 1e-12
    if not np.any(valid):
        return out
    tv = t_ns[valid]
    dv = dur_ns[valid]
    if shape in ("rectangular", "trapezoid"):
        r0 = max(0.0, float(rise_ns))
        f0 = max(0.0, float(fallNs := fall_ns))
        if r0 == 0.0 and f0 == 0.0:
            out[valid] = 1.0
            return out
        tot_rf = r0 + fallNs
        scale = np.where(tot_rf > dv, dv / max(1e-12, tot_rf), 1.0)
        rv = r0 * scale
        fv = fallNs * scale
        val = np.ones_like(tv)
        m_rise = (rv > 1e-12) & (tv < rv)
        if np.any(m_rise):
            ur = tv[m_rise] / rv[m_rise]
            val[m_rise] = ur if shape == "trapezoid" else 0.5 * (1.0 - np.cos(np.pi * ur))
        m_fall = (fv > 1e-12) & (tv > dv - fv) & (~m_rise)
        if np.any(m_fall):
            uf = np.clip((dv[m_fall] - tv[m_fall]) / fv[m_fall], 0.0, 1.0)
            val[m_fall] = uf if shape == "trapezoid" else 0.5 * (1.0 - np.cos(np.pi * uf))
        out[valid] = np.clip(val, 0.0, 1.0)
        return out

    u = np.clip(tv / dv, 0.0, 1.0)
    if shape == "gaussian":
        sigma = 1.0 / 4.5
        edge = math.exp(-0.5 * (0.5 / sigma) ** 2)
        out[valid] = np.maximum(0.0, (np.exp(-0.5 * ((u - 0.5) / sigma) ** 2) - edge) / (1.0 - edge))
        return out
    out[valid] = 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
    return out


def _effective_area_time_vec(
    xs: np.ndarray,
    pulse_shape: str,
    rise_ns: float,
    fall_ns: float,
) -> np.ndarray:
    return np.array(
        [float(t) * envelope_area_factor(pulse_shape, float(t), rise_ns, fall_ns) for t in xs],
        dtype=np.float64,
    )


def _propagate_shaped_pulses_batch(
    durations_ns: np.ndarray,
    detunings_mhz: np.ndarray,
    omega_peak_mhz: float,
    t1_ns: float,
    t2_ns: float,
    pulse_shape: str,
    rise_ns: float,
    fall_ns: float,
    n_rk_steps: int = 28,
) -> np.ndarray:
    """
    Propagate initial state |0><0| under shaped envelopes of total duration `durations_ns[k]`
    and detuning `detunings_mhz[k]` for all batch items simultaneously via 4th-order Runge-Kutta.
    Returns excited population P(|1>) array of shape `(len(durations_ns),)`.
    """
    n_batch = len(durations_ns)
    r = np.zeros((n_batch, 4), dtype=np.complex128)
    r[:, 0] = 1.0

    lv_diss = _liouvillian(0.0, 0.0, 0.0, t1_ns, t2_ns)
    wz_vec = 2.0 * math.pi * detunings_mhz * 1e-3  # (n_batch,)
    wx_peak = 2.0 * math.pi * omega_peak_mhz * 1e-3
    dt_vec = durations_ns / float(n_rk_steps)  # (n_batch,)

    def rhs(state: np.ndarray, u_frac: float) -> np.ndarray:
        t_cur = u_frac * durations_ns
        env = _sample_peak_envelope_vec(t_cur, durations_ns, pulse_shape, rise_ns, fall_ns)
        wx = wx_peak * env
        # state @ lv_diss.T + wx[:, None] * (state @ _GX_SUPER.T) + wz_vec[:, None] * (state @ _GZ_SUPER.T)
        return (
            state @ lv_diss.T
            + wx[:, None] * (state @ _GX_SUPER.T)
            + wz_vec[:, None] * (state @ _GZ_SUPER.T)
        )

    du = 1.0 / float(n_rk_steps)
    for step_i in range(n_rk_steps):
        u0 = step_i * du
        k1 = rhs(r, u0)
        k2 = rhs(r + 0.5 * dt_vec[:, None] * k1, u0 + 0.5 * du)
        k3 = rhs(r + 0.5 * dt_vec[:, None] * k2, u0 + 0.5 * du)
        k4 = rhs(r + dt_vec[:, None] * k3, u0 + du)
        r = r + (dt_vec[:, None] / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return np.clip(np.real(r[:, 3]), 0.0, 1.0)


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
    extra_fields: Dict[str, Any] = {}

    if req.protocol == "rabi":
        kappa_nom = envelope_area_factor(hp.pulse_shape, max(40.0, hp.t_1q_ns), hp.rise_ns, hp.fall_ns)
        period_eff_ns = 1000.0 / max(1e-6, omega_true * max(0.25, kappa_nom)) + 0.5 * (hp.rise_ns + hp.fall_ns)
        t_max = float(req.cal_max_ns) if req.cal_max_ns else 3.0 * period_eff_ns
        xs = np.linspace(0.0, t_max, n_pts)
        p1_true = _propagate_shaped_pulses_batch(
            xs,
            np.full(n_pts, det_true, dtype=np.float64),
            omega_true,
            t1,
            t2,
            hp.pulse_shape,
            hp.rise_ns,
            hp.fall_ns,
        )
        ys = _measure(p1_true, p10, p01, shots, rng)

        def model(t: np.ndarray, a: float, b: float, f_peak: float, tau: float) -> np.ndarray:
            t_eff = _effective_area_time_vec(t, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
            return a - b * np.exp(-t / tau) * np.cos(2.0 * math.pi * f_peak * t_eff)

        f0 = math.sqrt(omega_true ** 2 + det_true ** 2) * 1e-3
        try:
            popt, _ = curve_fit(
                model,
                xs,
                ys,
                p0=[0.5, 0.5, f0, max(t_max * 4, 1.0)],
                bounds=([0.0, 0.0, 1e-7, 1.0], [1.0, 1.0, 1.0, 1e9]),
                maxfev=8000,
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
        edge_delay = 0.5 * (hp.rise_ns + hp.fall_ns)
        params = [
            {"key": "omega_fit", "label": "拟合峰值拉比频率 Ω_R", "value": _fmt(omega_fit, 5), "unit": "MHz"},
            {"key": "omega_set", "label": "器件设定 Ω_R", "value": _fmt(omega_set, 5), "unit": "MHz"},
            {"key": "t_pi_env", "label": f"π 脉冲时长 ({hp.pulse_shape})", "value": _fmt(t_pi_env, 2), "unit": "ns"},
            {"key": "t_pi2_env", "label": f"π/2 脉冲时长 ({hp.pulse_shape})", "value": _fmt(t_pi2_env, 2), "unit": "ns"},
            {"key": "t_pi", "label": "方波等效 π 时长", "value": _fmt(t_pi_rect, 2), "unit": "ns"},
            {"key": "edge_offset", "label": "边沿面积补偿", "value": _fmt(edge_delay, 2), "unit": "ns"},
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
            f"{q_name} 真实包络 ({hp.pulse_shape}, 上升/下降沿 {hp.rise_ns:g}/{hp.fall_ns:g} ns) 拟合峰值 Ω_R = {omega_fit:.4f} MHz"
            f"（设定 {omega_set:.4f} MHz，偏差 {100 * (omega_fit / max(1e-6, omega_set) - 1):+.2f}%），π 脉冲 ≈ {t_pi_env:.1f} ns。"
        )
        x_label, y_label = "驱动脉冲总时长 t (ns)", "激发态概率 P(|1⟩)"

    elif req.protocol == "chevron":
        n_time = max(15, min(101, int(req.cal_points)))
        n_freq = max(9, min(61, int(req.chevron_freq_points)))
        det_span = max(0.2, float(req.chevron_detuning_max_mhz))
        kappa_nom = envelope_area_factor(hp.pulse_shape, max(40.0, hp.t_1q_ns), hp.rise_ns, hp.fall_ns)
        period_eff_ns = 1000.0 / max(1e-6, omega_true * max(0.25, kappa_nom)) + 0.5 * (hp.rise_ns + hp.fall_ns)
        t_max = float(req.cal_max_ns) if req.cal_max_ns else 2.6 * period_eff_ns

        xs = np.linspace(0.0, t_max, n_time)
        y_dets = np.linspace(-det_span, det_span, n_freq)
        tt_grid, dd_grid = np.meshgrid(xs, y_dets + det_true)
        p1_flat = _propagate_shaped_pulses_batch(
            tt_grid.ravel(),
            dd_grid.ravel(),
            omega_true,
            t1,
            t2,
            hp.pulse_shape,
            hp.rise_ns,
            hp.fall_ns,
            n_rk_steps=24,
        )
        p1_grid = p1_flat.reshape(n_freq, n_time)
        meas_grid = _measure(p1_grid, p10, p01, shots, rng)

        # Resonance occurs where max population peak / total swing is largest (slowest generalized Rabi frequency)
        row_peaks = np.max(meas_grid, axis=1) - np.min(meas_grid, axis=1)
        best_j = int(np.argmax(row_peaks))
        # Sub-grid quadratic interpolation around best_j
        if 0 < best_j < n_freq - 1:
            y_m, y_0, y_p = float(row_peaks[best_j - 1]), float(row_peaks[best_j]), float(row_peaks[best_j + 1])
            denom = y_m - 2.0 * y_0 + y_p
            shift = 0.5 * (y_m - y_p) / denom if abs(denom) > 1e-9 else 0.0
            shift = max(-0.75, min(0.75, shift))
            step_d = float(y_dets[1] - y_dets[0])
            res_drive_det = float(y_dets[best_j]) + shift * step_d
        else:
            res_drive_det = float(y_dets[best_j])
        est_qubit_offset = -res_drive_det  # since delta_eff = y_det + det_true = 0 => det_true = -y_det

        p1_true = p1_grid[best_j]
        ys = meas_grid[best_j]

        def res_model(t: np.ndarray, a: float, b: float, f_peak: float, tau: float) -> np.ndarray:
            t_eff = _effective_area_time_vec(t, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
            return a - b * np.exp(-t / tau) * np.cos(2.0 * math.pi * f_peak * t_eff)

        f0 = omega_true * 1e-3
        try:
            popt, _ = curve_fit(
                res_model, xs, ys, p0=[0.5, 0.5, f0, max(t_max * 4, 1.0)],
                bounds=([0.0, 0.0, 1e-7, 1.0], [1.0, 1.0, 1.0, 1e9]), maxfev=6000,
            )
        except Exception:
            popt = np.array([0.5, 0.5, f0, 1e9])
        omega_fit = float(popt[2]) * 1e3
        t_pi_env = rabi_pulse_duration_ns(math.pi, omega_fit, hp.pulse_shape, hp.rise_ns, hp.fall_ns)
        fit_x = np.linspace(0.0, t_max, 240)
        fit_y = res_model(fit_x, *popt)

        params = [
            {"key": "omega_fit", "label": "共振峰值 Ω_R", "value": _fmt(omega_fit, 5), "unit": "MHz"},
            {"key": "freq_offset", "label": "Chevron 谐振频偏 Δ", "value": _fmt(est_qubit_offset, 4), "unit": "MHz"},
            {"key": "t_pi_env", "label": f"共振 π 脉冲 ({hp.pulse_shape})", "value": _fmt(t_pi_env, 2), "unit": "ns"},
            {"key": "grid_size", "label": "二维扫描网格", "value": float(n_freq * n_time), "unit": "点"},
        ]
        writeback = [
            {"scope": "qubit", "qubit": q, "field": "rabi_mhz", "value": _fmt(omega_fit, 5),
             "label": f"写回 {q_name} Ω_R = {omega_fit:.4f} MHz"},
        ]
        if q_spec.freq_ghz is not None and abs(est_qubit_offset) > 1e-4:
            writeback.append(
                {"scope": "qubit", "qubit": q, "field": "freq_ghz", "value": _fmt(q_spec.freq_ghz + est_qubit_offset * 1e-3, 7),
                 "label": f"写回 {q_name} 共振频率修正 {est_qubit_offset:+.3f} MHz"}
            )
        extra_fields = {
            "z_grid": [[float(round(v, 5)) for v in row] for row in meas_grid],
            "y_detunings": [float(round(v, 4)) for v in y_dets],
            "y_grid_label": "驱动频率失谐 Δ_drive / 2π (MHz)",
        }
        summary = (
            f"{q_name} 二维 Rabi Chevron ({n_freq}×{n_time}) 定位共振频偏 Δ ≈ {est_qubit_offset:+.3f} MHz，"
            f"共振拉比频率 Ω_R = {omega_fit:.4f} MHz（π 脉冲 ≈ {t_pi_env:.1f} ns）。"
        )
        x_label, y_label = "驱动脉冲时长 t (ns)", "共振切片 P(|1⟩)"

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

    elif req.protocol == "echo":
        n_echo = max(1, min(32, int(req.echo_num_pulses)))
        # Decompose pure dephasing into quasistatic/low-frequency (refocused by Hahn/CPMG-N) + white Markovian floor
        g1 = (1.0 / t1) if (math.isfinite(t1) and t1 > 0) else 0.0
        g_phi_total = max(0.0, (1.0 / t2 - 0.5 * g1)) if (math.isfinite(t2) and t2 > 0) else 0.0
        # CPMG-N filter suppresses low-frequency dephasing rate ~ g_slow / (1 + 2.5 * N^0.85)
        g_phi_fast = 0.25 * g_phi_total
        g_phi_slow_res = (0.75 * g_phi_total) / (1.0 + 2.5 * (float(n_echo) ** 0.85))
        g2_echo = 0.5 * g1 + g_phi_fast + g_phi_slow_res
        t2_echo_true = (1.0 / g2_echo) if g2_echo > 1e-15 else float("inf")

        if req.cal_max_ns:
            t_max = float(req.cal_max_ns)
        else:
            ref_scale = t2_echo_true if math.isfinite(t2_echo_true) else (3.0 * t2 if math.isfinite(t2) else 5e4)
            t_max = min(2.5 * ref_scale, 4.0 * t1 if math.isfinite(t1) else 2.5 * ref_scale)
        xs = np.linspace(0.0, t_max, n_pts)

        # Meiboom-Gill CPMG-N: X_{pi/2} - [tau/(2N) - Y_pi - tau/(2N)]^N - X_{pi/2}
        # Include realistic coherent pulse errors (amp_error_pct, phase_skew_deg) on X90 & Y180!
        scale_amp = 1.0 + req.amp_error_pct / 100.0
        skew_rad = math.radians(req.phase_skew_deg)
        u_x90 = _unitary_super(_rot(0.5 * math.pi * scale_amp, 0.0))
        u_y180 = _unitary_super(_rot(math.pi * scale_amp, 0.5 * math.pi + skew_rad))
        lv_echo_free = _liouvillian(0.0, det_true, 0.0, t1, t2_echo_true)

        p1_true = np.zeros(n_pts)
        for i, tau_tot in enumerate(xs):
            dt_half = float(tau_tot) / (2.0 * n_echo)
            s_half = expm(lv_echo_free * dt_half) if dt_half > 0 else np.eye(4, dtype=np.complex128)
            s_unit = s_half @ u_y180 @ s_half
            s_seq = np.linalg.matrix_power(s_unit, n_echo)
            r_out = u_x90 @ (s_seq @ (u_x90 @ rho0))
            p1_true[i] = _p1(r_out)
        ys = _measure(p1_true, p10, p01, shots, rng)

        def echo_model(t: np.ndarray, a: float, b: float, t2e: float) -> np.ndarray:
            return a + b * np.exp(-t / t2e)

        t2e_guess = t2_echo_true if math.isfinite(t2_echo_true) else t_max * 5.0
        try:
            popt, _ = curve_fit(
                echo_model,
                xs,
                ys,
                p0=[0.5, 0.5, t2e_guess],
                bounds=([0.0, -0.2, 1.0], [1.0, 1.0, 1e10]),
                maxfev=8000,
            )
        except Exception:
            popt = np.array([0.5, 0.5, t2e_guess])
        a_e, b_e, t2e_fit = [float(v) for v in popt]
        fit_x = np.linspace(0.0, t_max, 300)
        fit_y = echo_model(fit_x, *popt)

        # Un-refocused Ramsey reference curve on fit_x for direct visual contrast
        t2_ref = t2 if (math.isfinite(t2) and t2 > 0) else 1e12
        ref_ramsey = 0.5 + 0.5 * np.exp(-fit_x / t2_ref) * np.cos(2.0 * math.pi * (det_true * 1e-3) * fit_x)
        gain_ratio = (t2e_fit / t2) if (math.isfinite(t2) and t2 > 0) else 1.0
        seq_title = "Hahn Echo (N=1)" if n_echo == 1 else f"CPMG-{n_echo} 动态解耦"

        params = [
            {"key": "t2_echo", "label": f"拟合相干时间 T2_echo ({seq_title})", "value": _fmt(t2e_fit / 1000.0, 3), "unit": "μs"},
            {"key": "t2_star_ref", "label": "未解耦基准 T2*", "value": _fmt(t2 / 1000.0, 3) if math.isfinite(t2) else float("nan"), "unit": "μs"},
            {"key": "echo_gain", "label": "相干延长倍率 T2_E / T2*", "value": _fmt(gain_ratio, 3), "unit": "×"},
            {"key": "t1_limit", "label": "弛豫理论上限 2·T1", "value": _fmt(2.0 * t1 / 1000.0, 2) if math.isfinite(t1) else float("nan"), "unit": "μs"},
            {"key": "num_pulses", "label": "重聚焦 π 脉冲数 N", "value": float(n_echo), "unit": ""},
        ]
        writeback = [
            {"scope": "qubit", "qubit": q, "field": "t2_us", "value": _fmt(t2e_fit / 1000.0, 3),
             "label": f"写回 {q_name} T2_echo = {t2e_fit / 1000.0:.2f} μs"},
        ]
        extra_fields = {
            "echo_ref_ramsey": [float(round(v, 6)) for v in ref_ramsey],
            "echo_num_pulses": n_echo,
        }
        summary = (
            f"{q_name} {seq_title} 拟合 T2_echo = {t2e_fit / 1000.0:.2f} μs"
            f"（较自由进动 T2* 提升 {gain_ratio:.2f}×，静态频偏 Δ = {det_true:+.3f} MHz 已完全重聚焦）。"
        )
        x_label, y_label = "总自由演化时间 τ (ns)", "重聚焦存活概率 P(|1⟩)"

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
    if req.protocol in ("rabi", "chevron"):
        circuit_samples = [
            {
                "id": "rabi_var",
                "label": "Rabi 变宽驱动序列" if req.protocol == "rabi" else "Chevron (Δ, t) 二维扫频变宽序列",
                "subtitle": f"{q_name}: RXY(Ω_R·t, 0) [{hp.pulse_shape}] → M",
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
                "label": "Ramsey π/2 − IDLE(τ) − π/2 干涉序列",
                "subtitle": f"{q_name}: SX → IDLE(τ) + RZ(δ·τ) → SX → M",
                "operations": [
                    CircuitOpModel(id="ram_0", step=0, kind="gate", name="SX", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="ram_idle", step=1, kind="gate", name="IDLE", qubits=[q], params=[200.0]).model_dump(),
                    CircuitOpModel(id="ram_rz", step=2, kind="gate", name="RZ", qubits=[q], params=[round(0.5 * math.pi, 4)]).model_dump(),
                    CircuitOpModel(id="ram_1", step=3, kind="gate", name="SX", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="ram_m", step=4, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump(),
                ],
            }
        ]
    elif req.protocol == "echo":
        n_disp = min(4, max(1, int(req.echo_num_pulses)))
        echo_ops: List[Dict[str, Any]] = [
            CircuitOpModel(id="ec_in", step=0, kind="gate", name="SX", qubits=[q], params=[]).model_dump(),
        ]
        st = 1
        for k_p in range(n_disp):
            echo_ops.append(CircuitOpModel(id=f"ec_w1_{k_p}", step=st, kind="gate", name="IDLE", qubits=[q], params=[100.0]).model_dump())
            st += 1
            echo_ops.append(CircuitOpModel(id=f"ec_pi_{k_p}", step=st, kind="gate", name="Y", qubits=[q], params=[]).model_dump())
            st += 1
            echo_ops.append(CircuitOpModel(id=f"ec_w2_{k_p}", step=st, kind="gate", name="IDLE", qubits=[q], params=[100.0]).model_dump())
            st += 1
            echo_ops.append(CircuitOpModel(id=f"ec_bar_{k_p}", step=st, kind="gate", name="BARRIER", qubits=[q], params=[]).model_dump())
            st += 1
        echo_ops.append(CircuitOpModel(id="ec_out", step=st, kind="gate", name="SX", qubits=[q], params=[]).model_dump())
        echo_ops.append(CircuitOpModel(id="ec_m", step=st + 1, kind="measure", name="MEASURE", qubits=[q], params=[]).model_dump())
        circuit_samples = [
            {
                "id": "echo_seq",
                "label": f"{'Hahn Echo' if req.echo_num_pulses <= 1 else f'CPMG-{req.echo_num_pulses}'} 动态解耦序列",
                "subtitle": f"{q_name}: SX − [IDLE(τ/2N) − Y_π − IDLE(τ/2N)]×{req.echo_num_pulses} − SX → M",
                "operations": echo_ops,
            }
        ]
    else:
        circuit_samples = [
            {
                "id": "t1_seq",
                "label": "T1 激发态弛豫测量序列",
                "subtitle": f"{q_name}: X(π) → IDLE(τ) → M",
                "operations": [
                    CircuitOpModel(id="t1_x", step=0, kind="gate", name="X", qubits=[q], params=[]).model_dump(),
                    CircuitOpModel(id="t1_idle", step=1, kind="gate", name="IDLE", qubits=[q], params=[500.0]).model_dump(),
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
        **extra_fields,
    }
