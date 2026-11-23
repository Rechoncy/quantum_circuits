import React, { useState, useMemo } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  GitCompareArrows,
  Maximize2,
  Minimize2,
  Play,
  Pause,
  RefreshCw,
  SkipBack,
  SkipForward,
  X,
} from 'lucide-react';
import type { ProjectModel, RunHistoryItem, SimulationResponse } from '../types';
import { PlotlyChart } from './PlotlyChart';
import { BlochSphere3D } from './BlochSphere3D';

interface SimulateResultsProps {
  project: ProjectModel;
  simResponse: SimulationResponse | null;
  timeCursorStep: number;
  maxCursorStep: number;
  onTimeCursorChange?: (step: number) => void;
  onStepJump?: (targetStep: number) => void;
  isPlayingTimeline?: boolean;
  onTogglePlayTimeline?: () => void;
  playbackSpeed?: 0.5 | 1 | 2;
  onPlaybackSpeedChange?: (speed: 0.5 | 1 | 2) => void;
  observedQubit: number;
  onObservedQubitChange?: (q: number) => void;
  darkMode: boolean;
  circuitCollapsed?: boolean;
  onToggleCircuitCollapsed?: () => void;
  compareRun?: RunHistoryItem | null;
  onClearCompareRun?: () => void;
  isStale?: boolean;
  onRefreshSim?: () => void;
}

const CHART_PALETTE = [
  '#0A84FF',
  '#10B981',
  '#F59E0B',
  '#8B5CF6',
  '#EC4899',
  '#06B6D4',
  '#EF4444',
  '#6366F1',
];

function formatOpLabelClean(raw: string): string {
  return raw
    .replace(/@\((\d+),\s*\)/g, '@ q$1')
    .replace(/@\((\d+(?:,\s*\d+)+)\)/g, (_, list: string) =>
      `@ ${list
        .split(',')
        .map((s) => `q${s.trim()}`)
        .join(',')}`
    );
}

function shortGateToken(raw: string): string {
  if (raw.includes('∥')) {
    return raw
      .split('∥')
      .map((part) => part.trim().split(/[\s(]/)[0])
      .slice(0, 2)
      .join('∥');
  }
  const token = raw.trim().split(/[\s(]/)[0];
  return token.length > 7 ? token.slice(0, 6) : token;
}

export const SimulateResults: React.FC<SimulateResultsProps> = ({
  project,
  simResponse,
  timeCursorStep,
  maxCursorStep,
  onTimeCursorChange,
  onStepJump,
  isPlayingTimeline = false,
  onTogglePlayTimeline,
  playbackSpeed = 1,
  onPlaybackSpeedChange,
  observedQubit,
  onObservedQubitChange,
  darkMode,
  circuitCollapsed = false,
  onToggleCircuitCollapsed,
  compareRun = null,
  onClearCompareRun,
  isStale = false,
  onRefreshSim,
}) => {
  const [tab, setTab] = useState<'evolution' | 'populations' | 'measurement' | 'density' | 'steps'>('evolution');
  const [rhoPart, setRhoPart] = useState<'abs' | 'real' | 'imag'>('abs');
  const [measUnit, setMeasUnit] = useState<'prob' | 'counts'>('prob');
  const [filterSmallBars, setFilterSmallBars] = useState(true);
  const [focusedEvoPanel, setFocusedEvoPanel] = useState<'none' | 'curve' | 'sphere'>('none');

  const textColor = darkMode ? '#E5E5EA' : '#1D1D1F';
  const mutedColor = darkMode ? '#98989D' : '#6E6E73';
  const gridColor = darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.07)';
  const plotBg = 'rgba(0,0,0,0)';

  const primaryRes = simResponse ? simResponse.results[simResponse.primary_backend] : null;
  const clampedCursor = primaryRes
    ? Math.max(0, Math.min(timeCursorStep, primaryRes.steps.length - 1))
    : 0;
  const stepFloor = Math.floor(clampedCursor);
  const stepCeil = primaryRes ? Math.min(primaryRes.steps.length - 1, stepFloor + 1) : 0;
  const stepFrac = clampedCursor - stepFloor;
  const stepIdx = stepFrac > 0.04 ? stepCeil : stepFloor;
  const snap0 = primaryRes?.steps[stepFloor] ?? null;
  const snap1 = primaryRes?.steps[stepCeil] ?? snap0;
  const activeSnap = primaryRes?.steps[stepIdx] ?? snap0;

  // Build continuous or stepwise Bloch curves + 3D trajectory (memoized only on [primaryRes, observedQubit])
  const blochData = useMemo(() => {
    if (!primaryRes) return null;
    const qKey = String(observedQubit);
    const hasPulse = primaryRes.steps.some(
      (s) => s.pulse_times_ns && s.pulse_times_ns.length > 1 && s.pulse_bloch_trajectories?.[qKey]
    );
    const xs: number[] = [];
    const bx: number[] = [];
    const by: number[] = [];
    const bz: number[] = [];
    const trajectory: [number, number, number][] = [];
    const stepPointIndices: number[] = [];
    const stepMarkers: { x: number; label: string }[] = [];

    if (hasPulse) {
      // Seed initial state from step 0 if step 0 has no pulse points
      const s0 = primaryRes.steps[0];
      if (s0 && (!s0.pulse_times_ns || s0.pulse_times_ns.length === 0)) {
        const bv0 = s0.bloch_vectors[qKey] || [0, 0, 1];
        xs.push(0);
        bx.push(bv0[0]);
        by.push(bv0[1]);
        bz.push(bv0[2]);
        trajectory.push([bv0[0], bv0[1], bv0[2]]);
        stepPointIndices.push(0);
      }
      for (let si = 0; si < primaryRes.steps.length; si++) {
        const s = primaryRes.steps[si];
        const pts = s.pulse_bloch_trajectories?.[qKey];
        if (s.pulse_times_ns && pts && pts.length > 0) {
          for (let i = 0; i < s.pulse_times_ns.length; i++) {
            // Avoid exact duplicate boundary timestamp
            if (xs.length > 0 && Math.abs(s.pulse_times_ns[i] - xs[xs.length - 1]) < 1e-6) {
              continue;
            }
            xs.push(s.pulse_times_ns[i]);
            bx.push(pts[i][0]);
            by.push(pts[i][1]);
            bz.push(pts[i][2]);
            trajectory.push([pts[i][0], pts[i][1], pts[i][2]]);
          }
          stepPointIndices[si] = Math.max(0, trajectory.length - 1);
          if (si > 0) {
            stepMarkers.push({ x: s.elapsed_time_ns, label: shortGateToken(s.operation_label) });
          }
        } else {
          // Zero-duration step (e.g. Virtual-RZ or INIT)
          const bv = s.bloch_vectors[qKey] || [0, 0, 1];
          if (si > 0) {
            xs.push(s.elapsed_time_ns);
            bx.push(bv[0]);
            by.push(bv[1]);
            bz.push(bv[2]);
            trajectory.push([bv[0], bv[1], bv[2]]);
          }
          stepPointIndices[si] = Math.max(0, trajectory.length - 1);
        }
      }
    } else {
      for (let si = 0; si < primaryRes.steps.length; si++) {
        const s = primaryRes.steps[si];
        const bv = s.bloch_vectors[qKey] || [0, 0, 1];
        xs.push(s.step_index);
        bx.push(bv[0]);
        by.push(bv[1]);
        bz.push(bv[2]);
        trajectory.push([bv[0], bv[1], bv[2]]);
        stepPointIndices[si] = trajectory.length - 1;
        if (si > 0) {
          stepMarkers.push({ x: s.step_index, label: shortGateToken(s.operation_label) });
        }
      }
    }

    return {
      hasPulse,
      xs,
      bx,
      by,
      bz,
      trajectory,
      stepPointIndices,
      stepMarkers,
    };
  }, [primaryRes, observedQubit]);

  if (!simResponse || !primaryRes || !activeSnap) {
    return (
      <div className="flex-1 bg-[var(--bg-app)] border-t border-[var(--border-hairline)] flex items-center justify-center text-[13px] text-[var(--text-secondary)]">
        点击右上角「▶ 运行」即可查看多后端逐步演化、态布居、测量分布与密度矩阵。
      </div>
    );
  }

  const allBasis = Object.keys(primaryRes.final_probabilities).sort();
  const topBasis = allBasis.slice(0, 16);
  const obsName = project.qubits[observedQubit]?.name || `q${observedQubit}`;

  // Continuous sub-pulse trajectory index & spherical-radial Bloch vector interpolation
  const { cursorFloatIdx, curBlochVec, cursorXVal } = (() => {
    if (!blochData || blochData.trajectory.length === 0) {
      const fb = (activeSnap.bloch_vectors[String(observedQubit)] as [number, number, number]) || [0, 0, 1];
      return {
        cursorFloatIdx: 0,
        curBlochVec: fb,
        cursorXVal: activeSnap.elapsed_time_ns,
      };
    }
    const p0 = blochData.stepPointIndices[stepFloor] ?? 0;
    const p1 = blochData.stepPointIndices[stepCeil] ?? p0;
    const floatIdx = Math.max(0, Math.min(blochData.trajectory.length - 1, p0 + stepFrac * (p1 - p0)));
    const t0 = Math.floor(floatIdx);
    const t1 = Math.min(blochData.trajectory.length - 1, t0 + 1);
    const tFrac = floatIdx - t0;

    const v0 = blochData.trajectory[t0];
    const v1 = blochData.trajectory[t1];
    let lx = v0[0] + tFrac * (v1[0] - v0[0]);
    let ly = v0[1] + tFrac * (v1[1] - v0[1]);
    let lz = v0[2] + tFrac * (v1[2] - v0[2]);
    const r0 = Math.hypot(v0[0], v0[1], v0[2]);
    const r1 = Math.hypot(v1[0], v1[1], v1[2]);
    const targetR = r0 + tFrac * (r1 - r0);
    const lLen = Math.hypot(lx, ly, lz);
    if (lLen > 1e-4 && r0 > 0.05 && r1 > 0.05) {
      const scale = targetR / lLen;
      lx *= scale;
      ly *= scale;
      lz *= scale;
    }

    const x0 = blochData.xs[t0] ?? 0;
    const x1 = blochData.xs[t1] ?? x0;
    const cx = x0 + tFrac * (x1 - x0);
    return {
      cursorFloatIdx: floatIdx,
      curBlochVec: [lx, ly, lz] as [number, number, number],
      cursorXVal: cx,
    };
  })();

  // Continuous interpolated time, fidelity, purity & basis populations
  const interpElapsedNs =
    snap0 && snap1
      ? snap0.elapsed_time_ns + stepFrac * (snap1.elapsed_time_ns - snap0.elapsed_time_ns)
      : activeSnap.elapsed_time_ns;
  const interpFidelity =
    snap0 && snap1 && snap0.state_fidelity !== null && snap1.state_fidelity !== null
      ? snap0.state_fidelity + stepFrac * (snap1.state_fidelity - snap0.state_fidelity)
      : activeSnap.state_fidelity;
  const interpPurity =
    snap0 && snap1 && snap0.purity !== null && snap1.purity !== null
      ? snap0.purity + stepFrac * (snap1.purity - snap0.purity)
      : activeSnap.purity;

  // Top-8 computational basis states at current interpolated cursor for fluid progress bars
  const activeStepPopChips = (() => {
    if (!snap0 || !snap1) return [];
    const keys = Object.keys(snap1.populations);
    const entries: [string, number][] = keys.map((k) => {
      const p0 = snap0.populations[k] ?? 0;
      const p1 = snap1.populations[k] ?? 0;
      return [k, p0 + stepFrac * (p1 - p0)];
    });
    return entries
      .filter(([, p]) => p > 0.002)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  })();
  const fallbackPopChips =
    activeStepPopChips.length > 0
      ? activeStepPopChips
      : topBasis.slice(0, 6).map((b) => [b, primaryRes.final_probabilities[b] ?? 0] as [string, number]);

  // Measurement filtering (hide near-zero bars when 4+ qubits if toggle enabled)
  const measBasis = (() => {
    if (!filterSmallBars || allBasis.length <= 8) return allBasis.slice(0, 24);
    const significant = allBasis.filter((b) =>
      Object.values(simResponse.results).some((r) => (r.final_probabilities[b] ?? 0) >= 0.01)
    );
    return (significant.length > 0 ? significant : allBasis).slice(0, 24);
  })();

  const dominantEntry = Object.entries(primaryRes.final_probabilities).sort((a, b) => b[1] - a[1])[0];

  // Dynamic density matrix colorscale range so large diagonal entries (>= 0.5) are never clipped
  const rhoMatrix = activeSnap.density_matrix_abs
    ? rhoPart === 'abs'
      ? activeSnap.density_matrix_abs
      : rhoPart === 'real'
      ? activeSnap.density_matrix_real
      : activeSnap.density_matrix_imag
    : null;
  let rhoMaxAbs = 1;
  if (rhoMatrix) {
    let m = 0.01;
    for (const row of rhoMatrix) {
      for (const v of row) {
        const av = Math.abs(v);
        if (av > m) m = av;
      }
    }
    rhoMaxAbs = Math.min(1, Math.max(0.25, m));
  }

  const xMaxPlot = blochData && blochData.xs.length > 0 ? Math.max(...blochData.xs) : 1;
  const cleanActiveOp = formatOpLabelClean(activeSnap.operation_label);

  // Deduplicate / space out gate annotations so adjacent gates (e.g. CZ + √iSWAP) never collide
  const spacedStepAnnotations = useMemo(() => {
    if (!blochData) return [];
    const minGap = Math.max(xMaxPlot * 0.11, 0.8);
    const placed: { x: number; text: string }[] = [];
    for (const m of blochData.stepMarkers) {
      const prev = placed[placed.length - 1];
      if (prev && Math.abs(m.x - prev.x) < minGap) {
        if (!prev.text.includes(m.label)) {
          prev.text = `${prev.text}·${m.label}`;
        }
        prev.x = (prev.x + m.x) / 2;
      } else {
        placed.push({ x: m.x, text: m.label });
      }
    }
    return placed.slice(0, 7);
  }, [blochData, xMaxPlot]);

  // Memoize Plotly traces & layout for Evolution tab so 60fps cursor updates never trigger Plotly.react
  const evoPlotData = useMemo(() => {
    if (!blochData) return [];
    return [
      {
        x: blochData.xs,
        y: blochData.bx,
        type: 'scatter',
        mode: blochData.hasPulse ? 'lines' : 'lines+markers',
        name: '⟨X⟩',
        line: { color: '#0A84FF', width: 2.4, shape: blochData.hasPulse ? 'spline' : 'linear', smoothing: 0.4 },
        marker: { size: 5 },
        hovertemplate: '⟨X⟩ = %{y:.3f}<extra></extra>',
      },
      {
        x: blochData.xs,
        y: blochData.by,
        type: 'scatter',
        mode: blochData.hasPulse ? 'lines' : 'lines+markers',
        name: '⟨Y⟩',
        line: { color: '#F59E0B', width: 2.4, shape: blochData.hasPulse ? 'spline' : 'linear', smoothing: 0.4 },
        marker: { size: 5 },
        hovertemplate: '⟨Y⟩ = %{y:.3f}<extra></extra>',
      },
      {
        x: blochData.xs,
        y: blochData.bz,
        type: 'scatter',
        mode: blochData.hasPulse ? 'lines' : 'lines+markers',
        name: '⟨Z⟩',
        line: { color: '#10B981', width: 2.4, shape: blochData.hasPulse ? 'spline' : 'linear', smoothing: 0.4 },
        marker: { size: 5 },
        hovertemplate: '⟨Z⟩ = %{y:.3f}<extra></extra>',
      },
    ];
  }, [blochData]);

  const evoPlotLayout = useMemo(() => {
    if (!blochData) return {};
    return {
      paper_bgcolor: plotBg,
      plot_bgcolor: plotBg,
      showlegend: false,
      hovermode: 'x unified',
      margin: { l: 46, r: 28, t: 22, b: 38 },
      xaxis: {
        title: {
          text: blochData.hasPulse ? '物理演化时间 t (ns)' : '演化步骤 Step',
          font: { size: 11, color: mutedColor },
          standoff: 8,
        },
        range: blochData.hasPulse ? [-xMaxPlot * 0.015, xMaxPlot * 1.03] : undefined,
        gridcolor: gridColor,
        zeroline: false,
        tickfont: { color: mutedColor, size: 10.5 },
      },
      yaxis: {
        title: {
          text: `${obsName} 期望值`,
          font: { size: 11, color: mutedColor },
          standoff: 6,
        },
        range: [-1.1, 1.12],
        dtick: 0.5,
        gridcolor: gridColor,
        zerolinecolor: darkMode ? 'rgba(255,255,255,0.16)' : 'rgba(15,23,42,0.16)',
        zerolinewidth: 1,
        tickfont: { color: mutedColor, size: 10.5 },
      },
      shapes: blochData.stepMarkers.slice(0, 10).map((m) => ({
        type: 'line',
        x0: m.x,
        x1: m.x,
        y0: -1.1,
        y1: 1.12,
        line: {
          color: darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.09)',
          width: 1,
          dash: 'dot',
        },
      })),
      annotations: spacedStepAnnotations.map((m) => ({
        x: m.x,
        y: 1.09,
        text: m.text,
        showarrow: false,
        font: { size: 9.5, color: mutedColor },
        xanchor: 'center',
      })),
    };
  }, [blochData, plotBg, mutedColor, xMaxPlot, gridColor, obsName, darkMode, spacedStepAnnotations]);

  return (
    <div className="flex-1 min-h-0 bg-[var(--bg-app)] flex flex-col overflow-hidden">
      {/* Sub-Tabs Bar (Always clean, never overflows!) */}
      <div className="h-10 px-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-4 lg:gap-5 h-full shrink-0">
          {(
            [
              { id: 'evolution', label: '演化' },
              { id: 'populations', label: '布居' },
              { id: 'measurement', label: '测量' },
              { id: 'density', label: '密度矩阵 ρ' },
              { id: 'steps', label: '指标 & 明细' },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`h-full text-[12.5px] border-b-2 transition-colors cursor-pointer whitespace-nowrap ${
                tab === t.id
                  ? 'border-[#0A84FF] text-[var(--text-primary)] font-semibold'
                  : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 text-[11.5px] font-mono-num text-[var(--text-secondary)] min-w-0">
          {!circuitCollapsed && (
            <span className="hidden xl:inline truncate max-w-[180px]" title={`Step ${activeSnap.step_index}: ${cleanActiveOp}`}>
              Step {activeSnap.step_index}: {cleanActiveOp}
            </span>
          )}
          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11px] shrink-0">
            {interpElapsedNs.toFixed(1)} ns
          </span>
          {interpFidelity !== null && (
            <span className="px-1.5 py-0.5 rounded bg-[#0A84FF]/10 text-[#0A84FF] font-semibold text-[11px] shrink-0">
              F = {interpFidelity.toFixed(4)}
            </span>
          )}
          {onToggleCircuitCollapsed && (
            <button
              onClick={onToggleCircuitCollapsed}
              title={circuitCollapsed ? '展开上方线路预览' : '收起上方线路，放大绘图区域'}
              className="px-2 py-1 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11px] font-sans font-medium flex items-center gap-1 cursor-pointer transition-colors shrink-0 whitespace-nowrap"
            >
              {circuitCollapsed ? (
                <>
                  <ChevronDown className="w-3 h-3" />
                  <span>展开线路</span>
                </>
              ) : (
                <>
                  <ChevronUp className="w-3 h-3" />
                  <span>放大绘图</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Dedicated Slim Timeline Strip when Top Circuit Preview is Collapsed */}
      {circuitCollapsed && onTimeCursorChange && (
        <div className="h-9 px-4 bg-[var(--bg-panel)]/95 border-b border-[var(--border-hairline)] flex items-center gap-2.5 shrink-0">
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => (onStepJump || onTimeCursorChange)(Math.max(0, Math.ceil(timeCursorStep) - 1))}
              title="上一步"
              className="w-6 h-6 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer"
            >
              <SkipBack className="w-3 h-3" />
            </button>
            {onTogglePlayTimeline && (
              <button
                type="button"
                onClick={onTogglePlayTimeline}
                className="w-6 h-6 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] hover:bg-[var(--accent)]/20 flex items-center justify-center cursor-pointer"
                title={isPlayingTimeline ? '暂停 (Space)' : '播放连续脉冲演化 (Space)'}
              >
                {isPlayingTimeline ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
              </button>
            )}
            <button
              type="button"
              onClick={() => (onStepJump || onTimeCursorChange)(Math.min(maxCursorStep, Math.floor(timeCursorStep) + 1))}
              title="下一步"
              className="w-6 h-6 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer"
            >
              <SkipForward className="w-3 h-3" />
            </button>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxCursorStep)}
            step={0.01}
            value={timeCursorStep}
            onChange={(e) => onTimeCursorChange(Number(e.target.value))}
            className="mac-slider flex-1"
          />
          {onPlaybackSpeedChange && (
            <div className="bg-black/[0.045] dark:bg-white/[0.07] p-0.5 rounded-md inline-flex text-[10px] font-mono-num shrink-0">
              {([0.5, 1, 2] as const).map((sp) => (
                <button
                  key={sp}
                  type="button"
                  onClick={() => onPlaybackSpeedChange(sp)}
                  className={`px-1.5 py-0.25 rounded cursor-pointer ${
                    playbackSpeed === sp
                      ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold shadow-2xs'
                      : 'text-[var(--text-secondary)]'
                  }`}
                >
                  {sp}×
                </button>
              ))}
            </div>
          )}
          <span className="text-[11px] font-mono-num text-[var(--text-secondary)] shrink-0">
            Step {activeSnap.step_index} / {maxCursorStep} · {cleanActiveOp}
          </span>
        </div>
      )}

      {simResponse.warnings.length > 0 && (
        <div className="px-4 py-1.5 bg-amber-50/95 dark:bg-amber-950/45 border-b border-amber-200 dark:border-amber-800 text-[11.5px] text-amber-800 dark:text-amber-200 flex items-center gap-2 shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">{simResponse.warnings.join(' · ')}</span>
        </div>
      )}

      {/* Stale Result / Run Diff Comparison Banner */}
      {(isStale || compareRun) && (
        <div className="px-4 py-1 bg-[var(--bg-panel)]/95 border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2 text-[11.5px] shrink-0">
          <div className="flex items-center gap-2">
            {isStale && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-amber-500/12 text-amber-700 dark:text-amber-300 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                线路或参数已变更，当前为上次结果
                {onRefreshSim && (
                  <button
                    onClick={onRefreshSim}
                    className="ml-1 underline hover:opacity-80 inline-flex items-center gap-0.5 cursor-pointer font-semibold"
                  >
                    <RefreshCw className="w-3 h-3" />
                    立即更新
                  </button>
                )}
              </span>
            )}
          </div>
          {compareRun && (
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-md bg-purple-500/12 text-purple-700 dark:text-purple-300 font-mono-num">
              <GitCompareArrows className="w-3.5 h-3.5 shrink-0" />
              <span>
                已叠加对比基准: <b>{compareRun.label}</b> ({compareRun.backend}
                {compareRun.fidelity !== null && primaryRes.final_fidelity !== null
                  ? ` · ΔF = ${primaryRes.final_fidelity >= compareRun.fidelity ? '+' : ''}${(
                      primaryRes.final_fidelity - compareRun.fidelity
                    ).toFixed(4)}`
                  : ''}
                )
              </span>
              {onClearCompareRun && (
                <button
                  onClick={onClearCompareRun}
                  title="退出对比"
                  className="p-0.5 rounded hover:bg-purple-500/20 cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Compact shadcn-analytics 4-KPI Telemetry Strip */}
      <div className="px-3.5 pt-2.5 grid grid-cols-2 lg:grid-cols-4 gap-2.5 shrink-0">
        {(() => {
          const curFid = interpFidelity ?? primaryRes.final_fidelity;
          const curPur = interpPurity ?? primaryRes.final_purity;
          const curDom = fallbackPopChips[0] || dominantEntry;
          const fidDelta =
            compareRun && compareRun.fidelity !== null && curFid !== null
              ? curFid - compareRun.fidelity
              : null;
          const fidTone =
            curFid === null
              ? 'text-[var(--text-secondary)] bg-[var(--bg-subtle)]'
              : curFid >= 0.95
              ? 'text-emerald-700 dark:text-emerald-300 bg-emerald-500/12'
              : curFid >= 0.8
              ? 'text-amber-700 dark:text-amber-300 bg-amber-500/12'
              : 'text-rose-700 dark:text-rose-300 bg-rose-500/12';
          const purLabel =
            curPur === null ? '—' : curPur >= 0.98 ? '纯态' : curPur >= 0.85 ? '弱退相干' : '混态';

          return (
            <>
              <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
                <div className="min-w-0">
                  <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                    态保真度 F
                  </div>
                  <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                    {curFid !== null ? curFid.toFixed(4) : '—'}
                  </div>
                </div>
                <span className={`px-1.5 py-0.5 rounded-md text-[10px] font-mono-num font-semibold shrink-0 ${fidTone}`}>
                  {fidDelta !== null
                    ? `${fidDelta >= 0 ? '+' : ''}${fidDelta.toFixed(3)}`
                    : curFid !== null
                    ? curFid >= 0.95
                      ? '高保真'
                      : curFid >= 0.8
                      ? '中度噪'
                      : '强噪声'
                    : primaryRes.backend_name}
                </span>
              </div>

              <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
                <div className="min-w-0">
                  <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                    量子纯度 Tr(ρ²)
                  </div>
                  <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                    {curPur !== null ? curPur.toFixed(4) : '—'}
                  </div>
                </div>
                <span className="px-1.5 py-0.5 rounded-md bg-[var(--bg-subtle)] text-[var(--text-secondary)] text-[10px] font-medium shrink-0">
                  {purLabel}
                </span>
              </div>

              <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
                <div className="min-w-0">
                  <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                    演化时序
                  </div>
                  <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                    {interpElapsedNs.toFixed(1)}{' '}
                    <span className="text-[11px] font-normal text-[var(--text-secondary)]">ns</span>
                  </div>
                </div>
                <span className="px-1.5 py-0.5 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] text-[10px] font-mono-num font-semibold shrink-0">
                  S{activeSnap.step_index}/{maxCursorStep}
                </span>
              </div>

              <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
                <div className="min-w-0">
                  <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                    主导基态
                  </div>
                  <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5 truncate">
                    {curDom ? `|${curDom[0]}⟩` : '—'}
                  </div>
                </div>
                {curDom && (
                  <span className="px-1.5 py-0.5 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] text-[10.5px] font-mono-num font-semibold shrink-0">
                    {(curDom[1] * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </>
          );
        })()}
      </div>

      {/* Main Content Stage */}
      <div className="flex-1 min-h-0 px-3.5 py-3 overflow-y-auto">
        {/* =============================================================== */}
        {/* TAB 1: EVOLUTION (Balanced Curve + High-DPI Interactive Bloch 3D) */}
        {/* =============================================================== */}
        {tab === 'evolution' && blochData && (
          <div className="h-full flex flex-col gap-3 min-h-[340px]">
            <div className="flex-1 grid grid-cols-1 xl:grid-cols-12 gap-3 min-h-[265px]">
              {/* Left Card: Continuous / Stepwise Bloch Curves */}
              {focusedEvoPanel !== 'sphere' && (
                <div
                  className={`${
                    focusedEvoPanel === 'curve' ? 'xl:col-span-12' : 'xl:col-span-7'
                  } flex flex-col studio-card overflow-hidden min-h-[245px]`}
                >
                  {/* Crisp Card Header with Inline Qubit Pills & Clean External Legend */}
                  <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0">
                    <div className="flex items-center gap-2 min-w-0 overflow-hidden">
                      <span className="text-[12.5px] font-semibold text-[var(--text-primary)] whitespace-nowrap">
                        Bloch 分量演化
                      </span>
                      {onObservedQubitChange && project.qubits.length > 1 && (
                        <div className="flex items-center gap-0.5 bg-black/[0.045] dark:bg-white/[0.07] p-0.5 rounded-lg shrink-0 overflow-x-auto">
                          {project.qubits.slice(0, 5).map((q) => (
                            <button
                              key={q.id}
                              onClick={() => onObservedQubitChange(q.id)}
                              className={`px-1.5 py-0.5 rounded-md text-[10.5px] font-mono-num flex items-center gap-1 transition-all cursor-pointer shrink-0 ${
                                observedQubit === q.id
                                  ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                              }`}
                            >
                              <span
                                className="w-1.5 h-1.5 rounded-full"
                                style={{ backgroundColor: q.color }}
                              />
                              {q.name}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <div className="hidden 2xl:flex items-center gap-2.5 text-[11px] font-mono-num text-[var(--text-secondary)]">
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-[2.5px] rounded-full bg-[#0A84FF]" />⟨X⟩
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-[2.5px] rounded-full bg-[#F59E0B]" />⟨Y⟩
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="w-2.5 h-[2.5px] rounded-full bg-[#10B981]" />⟨Z⟩
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setFocusedEvoPanel(focusedEvoPanel === 'curve' ? 'none' : 'curve')
                        }
                        title={focusedEvoPanel === 'curve' ? '还原分屏视图' : '放大 Bloch 演化曲线'}
                        className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                      >
                        {focusedEvoPanel === 'curve' ? (
                          <Minimize2 className="w-3.5 h-3.5" />
                        ) : (
                          <Maximize2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Plot Body */}
                  <div className="flex-1 min-h-0 px-1.5 pb-1 pt-0.5">
                    <PlotlyChart
                      data={evoPlotData}
                      layout={evoPlotLayout}
                      cursorX={cursorXVal}
                      progressiveReveal
                      cursorTips={[
                        { y: curBlochVec[0], color: '#0A84FF' },
                        { y: curBlochVec[1], color: '#F59E0B' },
                        { y: curBlochVec[2], color: '#10B981' },
                      ]}
                    />
                  </div>
                </div>
              )}

              {/* Right Card: High-DPI Interactive 3D Bloch Sphere */}
              {focusedEvoPanel !== 'curve' && (
                <div className={focusedEvoPanel === 'sphere' ? 'xl:col-span-12 min-h-[260px]' : 'xl:col-span-5 min-h-[260px]'}>
                  <BlochSphere3D
                    qubitName={obsName}
                    trajectory={blochData.trajectory}
                    cursorIdx={cursorFloatIdx}
                    currentVector={curBlochVec}
                    stepIndices={blochData.stepPointIndices}
                    darkMode={darkMode}
                    expanded={focusedEvoPanel === 'sphere'}
                    onToggleExpand={() =>
                      setFocusedEvoPanel(focusedEvoPanel === 'sphere' ? 'none' : 'sphere')
                    }
                  />
                </div>
              )}
            </div>

            {/* Bottom Labeled Basis State Probability Strip (Full basis labels, never truncated!) */}
            <div className="shrink-0 studio-card px-3.5 py-2 flex items-center gap-3">
              <div className="shrink-0 leading-tight">
                <div className="text-[11.5px] font-semibold text-[var(--text-primary)]">
                  布居分布
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                  Step {activeSnap.step_index} · Top {fallbackPopChips.length}
                </div>
              </div>
              <div className="flex-1 flex items-center gap-2 overflow-x-auto min-w-0 py-0.5">
                {fallbackPopChips.map(([bstr, p]) => {
                  const pct = Math.max(0, Math.min(100, p * 100));
                  return (
                    <div
                      key={bstr}
                      title={`计算基态 |${bstr}⟩ 概率: ${pct.toFixed(2)}%`}
                      className="relative h-8 min-w-[116px] flex-1 max-w-[178px] shrink-0 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-subtle)] overflow-hidden px-2.5 flex items-center justify-between gap-2 font-mono-num"
                    >
                      <div
                        className="absolute inset-y-0 left-0 bg-[#0A84FF]/18 dark:bg-[#0A84FF]/28"
                        style={{ width: `${pct}%` }}
                      />
                      <span className="relative z-10 text-[11px] font-semibold text-[var(--text-primary)] whitespace-nowrap">
                        |{bstr}⟩
                      </span>
                      <span className="relative z-10 text-[10.5px] font-semibold text-[#0A84FF] whitespace-nowrap shrink-0">
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* TAB 2: STATE POPULATIONS                                         */}
        {/* =============================================================== */}
        {tab === 'populations' && (
          <div className="h-full grid grid-cols-12 gap-3 min-h-[320px]">
            <div className="col-span-6 flex flex-col studio-card overflow-hidden">
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                  计算基态布居演化曲线
                </span>
                <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">
                  Step 0 → {primaryRes.steps.length - 1}
                </span>
              </div>
              <div className="flex-1 min-h-0 p-1.5">
                <PlotlyChart
                  data={topBasis
                    .filter(
                      (b) =>
                        primaryRes.steps.some((s) => (s.populations[b] ?? 0) > 0.015) ||
                        topBasis.length <= 4
                    )
                    .slice(0, 8)
                    .map((b, idx) => ({
                      x: primaryRes.steps.map((s) => s.step_index),
                      y: primaryRes.steps.map((s) => s.populations[b] ?? 0),
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: `|${b}⟩`,
                      line: { width: 2.2, color: CHART_PALETTE[idx % CHART_PALETTE.length] },
                      marker: { size: 5 },
                      hovertemplate: `|${b}⟩: %{y:.3f}<extra></extra>`,
                    }))}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    hovermode: 'x unified',
                    margin: { l: 46, r: 16, t: 14, b: 40 },
                    xaxis: {
                      title: { text: '演化步骤 Step', font: { size: 11, color: mutedColor } },
                      dtick: 1,
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    yaxis: {
                      title: { text: '概率 P(|x⟩)', font: { size: 11, color: mutedColor } },
                      range: [-0.04, 1.06],
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    legend: {
                      orientation: 'h',
                      y: 1.1,
                      x: 0,
                      font: { size: 10.5, color: textColor },
                    },
                  }}
                  cursorX={clampedCursor}
                  progressiveReveal
                />
              </div>
            </div>

            <div className="col-span-6 flex flex-col studio-card overflow-hidden">
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                  布居数时空演化热力图
                </span>
                <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">
                  Step 0 → {Math.ceil(clampedCursor)} / {primaryRes.steps.length - 1}
                </span>
              </div>
              <div className="flex-1 min-h-0 p-1.5">
                <PlotlyChart
                  data={[
                    {
                      type: 'heatmap',
                      x: primaryRes.steps.map((s) => `S${s.step_index}`),
                      y: topBasis.map((b) => `|${b}⟩`),
                      z: topBasis.map((b) =>
                        primaryRes.steps.map((s) =>
                          s.step_index <= Math.ceil(clampedCursor) ? s.populations[b] ?? 0 : null
                        )
                      ),
                      colorscale: [
                        [0, darkMode ? '#1F2430' : '#F4F7FB'],
                        [0.35, '#93C5FD'],
                        [0.7, '#2563EB'],
                        [1, '#1E3A8A'],
                      ],
                      zmin: 0,
                      zmax: 1,
                      colorbar: { thickness: 11, len: 0.88, tickfont: { size: 10, color: mutedColor } },
                      hovertemplate: '%{y} @ %{x}<br>P = %{z:.4f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    margin: { l: 58, r: 14, t: 10, b: 38 },
                    xaxis: { tickfont: { color: mutedColor, size: 10.5 } },
                    yaxis: { tickfont: { color: mutedColor, size: 10.5 }, autorange: 'reversed' },
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* TAB 3: FINAL MEASUREMENT HISTOGRAM & MULTI-BACKEND COMPARISON    */}
        {/* =============================================================== */}
        {tab === 'measurement' && (
          <div className="h-full flex flex-col studio-card overflow-hidden min-h-[320px]">
            <div className="h-10 px-4 border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)] whitespace-nowrap">
                  最终含噪测量分布
                </span>
                <div className="hidden md:flex items-center gap-2 text-[11px] font-mono-num">
                  <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-secondary)]">
                    Shots: {project.sim_settings.shots.toLocaleString()}
                  </span>
                  {dominantEntry && (
                    <span className="px-2 py-0.5 rounded-md bg-[#0A84FF]/10 text-[#0A84FF] font-semibold">
                      主导态 |{dominantEntry[0]}⟩ ({(dominantEntry[1] * 100).toFixed(1)}%)
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2.5 shrink-0">
                {allBasis.length > 8 && (
                  <label className="flex items-center gap-1.5 text-[11.5px] text-[var(--text-secondary)] cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={filterSmallBars}
                      onChange={(e) => setFilterSmallBars(e.target.checked)}
                      className="accent-[#0A84FF] rounded"
                    />
                    仅看 P ≥ 1%
                  </label>
                )}
                <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex">
                  <button
                    onClick={() => setMeasUnit('prob')}
                    className={`px-2.5 py-0.5 rounded-md text-[11.5px] cursor-pointer ${
                      measUnit === 'prob'
                        ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs text-[var(--text-primary)]'
                        : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    概率 P
                  </button>
                  <button
                    onClick={() => setMeasUnit('counts')}
                    className={`px-2.5 py-0.5 rounded-md text-[11.5px] cursor-pointer ${
                      measUnit === 'counts'
                        ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs text-[var(--text-primary)]'
                        : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    计数 Shots
                  </button>
                </div>
              </div>
            </div>

            <div className="flex-1 min-h-0 p-2">
              <PlotlyChart
                data={[
                  ...Object.entries(simResponse.results).map(([_k, r], idx) => {
                    const palette = ['#0A84FF', '#10B981', '#F59E0B', '#8B5CF6'];
                    const vals = measBasis.map((b) =>
                      measUnit === 'prob'
                        ? r.final_probabilities[b] ?? 0
                        : r.measurement_counts[b] ?? 0
                    );
                    return {
                      type: 'bar',
                      name: r.backend_name,
                      x: measBasis.map((b) => `|${b}⟩`),
                      y: vals,
                      text:
                        measBasis.length <= 16
                          ? vals.map((v) =>
                              measUnit === 'prob'
                                ? v >= 0.005
                                  ? `${(v * 100).toFixed(1)}%`
                                  : ''
                                : v > 0
                                ? String(v)
                                : ''
                            )
                          : undefined,
                      textposition: 'outside',
                      textfont: { size: 10, color: mutedColor },
                      marker: {
                        color: palette[idx % palette.length],
                        opacity: 0.9,
                        line: { width: 0 },
                      },
                    };
                  }),
                  ...(compareRun
                    ? (() => {
                        const cmpRes = compareRun.response.results[compareRun.response.primary_backend];
                        if (!cmpRes) return [];
                        const vals = measBasis.map((b) =>
                          measUnit === 'prob'
                            ? cmpRes.final_probabilities[b] ?? 0
                            : cmpRes.measurement_counts[b] ?? 0
                        );
                        return [
                          {
                            type: 'bar',
                            name: `基准 ${compareRun.label} (${cmpRes.backend_name})`,
                            x: measBasis.map((b) => `|${b}⟩`),
                            y: vals,
                            marker: {
                              color: '#A855F7',
                              opacity: 0.55,
                              line: { color: '#7E22CE', width: 1 },
                            },
                          },
                        ];
                      })()
                    : []),
                ]}
                layout={{
                  barmode: 'group',
                  bargap: 0.28,
                  paper_bgcolor: plotBg,
                  plot_bgcolor: plotBg,
                  margin: { l: 54, r: 16, t: 26, b: 44 },
                  xaxis: { tickfont: { color: mutedColor, size: 11 } },
                  yaxis: {
                    title: {
                      text: measUnit === 'prob' ? '测量概率 P' : '采样计数 Shots',
                      font: { size: 11.5, color: mutedColor },
                    },
                    rangemode: 'tozero',
                    gridcolor: gridColor,
                    tickfont: { color: mutedColor, size: 11 },
                  },
                  legend: { orientation: 'h', y: 1.12, font: { size: 11, color: textColor } },
                }}
              />
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* TAB 4: DENSITY MATRIX RHO & PER-QUBIT TOMOGRAPHY                 */}
        {/* =============================================================== */}
        {tab === 'density' && (
          <div className="h-full grid grid-cols-12 gap-3 min-h-[320px]">
            <div className="col-span-7 flex flex-col studio-card overflow-hidden">
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                  Step {activeSnap.step_index} 密度矩阵 ρ
                </span>
                <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex">
                  {(
                    [
                      ['abs', '|ρ| 模'],
                      ['real', 'Re(ρ)'],
                      ['imag', 'Im(ρ)'],
                    ] as const
                  ).map(([k, lbl]) => (
                    <button
                      key={k}
                      onClick={() => setRhoPart(k)}
                      className={`px-2 py-0.5 rounded-md text-[11px] cursor-pointer ${
                        rhoPart === k
                          ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs text-[var(--text-primary)]'
                          : 'text-[var(--text-secondary)]'
                      }`}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
              </div>
              {rhoMatrix ? (
                <div className="flex-1 min-h-0 p-2">
                  <PlotlyChart
                    data={[
                      {
                        type: 'heatmap',
                        z: rhoMatrix,
                        x: allBasis.map((b) => `|${b}⟩`),
                        y: allBasis.map((b) => `|${b}⟩`),
                        colorscale:
                          rhoPart === 'abs'
                            ? [
                                [0, darkMode ? '#1F2430' : '#F4F7FB'],
                                [0.35, '#93C5FD'],
                                [0.7, '#2563EB'],
                                [1, '#1E3A8A'],
                              ]
                            : 'RdBu',
                        zmin: rhoPart === 'abs' ? 0 : -rhoMaxAbs,
                        zmax: rhoMaxAbs,
                        colorbar: {
                          thickness: 11,
                          len: 0.88,
                          tickfont: { size: 10, color: mutedColor },
                        },
                        hovertemplate: '⟨%{y}|ρ|%{x}⟩ = %{z:.4f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      margin: { l: 54, r: 12, t: 10, b: 48 },
                      yaxis: {
                        autorange: 'reversed',
                        scaleanchor: 'x',
                        scaleratio: 1,
                        tickfont: { size: 10, color: mutedColor },
                      },
                      xaxis: { tickfont: { size: 10, color: mutedColor } },
                    }}
                  />
                </div>
              ) : (
                <div className="flex-1 flex items-center justify-center text-[12px] text-[var(--text-secondary)]">
                  当前为高比特或稳定子模式，密度矩阵已省略
                </div>
              )}
            </div>

            <div className="col-span-5 flex flex-col studio-card overflow-hidden">
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                  各比特约化 Bloch 态层析
                </span>
                <span className="text-[10.5px] text-[var(--text-secondary)]">点击切换观测比特</span>
              </div>
              <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
                {Object.entries(activeSnap.bloch_vectors).map(([qIdxStr, bv]) => {
                  const qIdx = Number(qIdxStr);
                  const qSpec = project.qubits[qIdx];
                  const isObs = qIdx === observedQubit;
                  const rLen = Math.hypot(bv[0], bv[1], bv[2]);
                  return (
                    <button
                      key={qIdxStr}
                      onClick={() => onObservedQubitChange && onObservedQubitChange(qIdx)}
                      className={`w-full text-left p-2.5 rounded-xl border transition-all cursor-pointer ${
                        isObs
                          ? 'bg-[#0A84FF]/8 border-[#0A84FF]/60 shadow-2xs'
                          : 'bg-[var(--bg-subtle)]/70 border-[var(--border-hairline)] hover:border-[#0A84FF]/40'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: qSpec?.color || '#0A84FF' }}
                          />
                          <span className="text-[12px] font-semibold font-mono-num">
                            {qSpec?.name || `q${qIdx}`}
                          </span>
                        </div>
                        <span className="text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                          |r| = {rLen.toFixed(3)}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-1.5 text-[10.5px] font-mono-num">
                        {[
                          { k: 'X', v: bv[0], c: '#0A84FF' },
                          { k: 'Y', v: bv[1], c: '#F59E0B' },
                          { k: 'Z', v: bv[2], c: '#10B981' },
                        ].map((comp) => (
                          <div key={comp.k} className="space-y-0.5">
                            <div className="flex justify-between text-[10px]">
                              <span className="text-[var(--text-secondary)]">{comp.k}</span>
                              <span>
                                {comp.v >= 0 ? '+' : ''}
                                {comp.v.toFixed(2)}
                              </span>
                            </div>
                            <div className="h-1.5 rounded-full bg-black/[0.06] dark:bg-white/[0.09] relative overflow-hidden">
                              <div
                                className="absolute top-0 bottom-0 rounded-full"
                                style={{
                                  backgroundColor: comp.c,
                                  left: comp.v >= 0 ? '50%' : `${50 + comp.v * 50}%`,
                                  width: `${Math.abs(comp.v) * 50}%`,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </button>
                  );
                })}

                {activeSnap.stabilizers && activeSnap.stabilizers.length > 0 && (
                  <div className="pt-2 border-t border-[var(--border-hairline)] space-y-1.5">
                    <div className="text-[11.5px] font-semibold text-[var(--text-secondary)]">
                      Stim 稳定子生成元
                    </div>
                    <div className="flex flex-wrap gap-1.5 font-mono-num text-[11.5px]">
                      {activeSnap.stabilizers.map((st, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)]"
                        >
                          {st}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {primaryRes.stim_circuit_str && (
                  <div className="pt-2 border-t border-[var(--border-hairline)] space-y-1">
                    <div className="text-[11.5px] font-semibold text-[var(--text-secondary)]">
                      生成的 Stim 含噪指令
                    </div>
                    <pre className="p-2 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[10.5px] font-mono-num overflow-x-auto max-h-36">
                      {primaryRes.stim_circuit_str}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* =============================================================== */}
        {/* TAB 5: FIDELITY/PURITY CURVE + CLICKABLE STEP TABLE              */}
        {/* =============================================================== */}
        {tab === 'steps' && (
          <div className="h-full flex flex-col gap-3 min-h-[320px]">
            {/* Top Card: Fidelity & Purity Decay vs Step */}
            <div className="h-[44%] min-h-[160px] flex flex-col studio-card overflow-hidden shrink-0">
              <div className="h-8 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                  逐步含噪保真度 F 与纯度 Tr(ρ²) 衰减
                </span>
                <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">
                  {primaryRes.backend_name}
                </span>
              </div>
              <div className="flex-1 min-h-0 px-1.5 pb-1">
                <PlotlyChart
                  data={[
                    ...Object.entries(simResponse.results).flatMap(([_bk, r], idx) => {
                      const fColor = CHART_PALETTE[(idx * 2) % CHART_PALETTE.length];
                      const pColor = CHART_PALETTE[(idx * 2 + 2) % CHART_PALETTE.length];
                      const traces: any[] = [];
                      if (r.steps.some((s) => s.state_fidelity !== null)) {
                        traces.push({
                          x: r.steps.map((s) => s.step_index),
                          y: r.steps.map((s) => s.state_fidelity),
                          type: 'scatter',
                          mode: 'lines+markers',
                          name: `${r.backend_name} 保真度 F`,
                          line: { color: fColor, width: 2.2 },
                          marker: { size: 5 },
                        });
                      }
                      if (r.steps.some((s) => s.purity !== null)) {
                        traces.push({
                          x: r.steps.map((s) => s.step_index),
                          y: r.steps.map((s) => s.purity),
                          type: 'scatter',
                          mode: 'lines+markers',
                          name: `${r.backend_name} 纯度 Tr(ρ²)`,
                          line: { color: pColor, width: 1.8, dash: 'dot' },
                          marker: { size: 4 },
                        });
                      }
                      return traces;
                    }),
                    ...(compareRun
                      ? (() => {
                          const cmpRes = compareRun.response.results[compareRun.response.primary_backend];
                          if (!cmpRes || !cmpRes.steps.some((s) => s.state_fidelity !== null)) return [];
                          return [
                            {
                              x: cmpRes.steps.map((s) => s.step_index),
                              y: cmpRes.steps.map((s) => s.state_fidelity),
                              type: 'scatter',
                              mode: 'lines+markers',
                              name: `基准 ${compareRun.label} 保真度 F`,
                              line: { color: '#A855F7', width: 2, dash: 'dash' },
                              marker: { size: 5, symbol: 'diamond' },
                            },
                          ];
                        })()
                      : []),
                  ]}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    hovermode: 'x unified',
                    margin: { l: 48, r: 16, t: 10, b: 30 },
                    xaxis: {
                      dtick: 1,
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10 },
                    },
                    yaxis: {
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10 },
                    },
                    legend: { orientation: 'h', y: 1.15, font: { size: 10, color: textColor } },
                  }}
                  cursorX={clampedCursor}
                  progressiveReveal
                />
              </div>
            </div>

            {/* Bottom Card: Clickable Step Table */}
            <div className="flex-1 min-h-[150px] studio-card overflow-auto">
              <table className="w-full text-[12px] font-mono-num border-collapse">
                <thead className="sticky top-0 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] text-[11px] text-[var(--text-secondary)] text-left z-10">
                  <tr>
                    <th className="py-2 px-3">Step</th>
                    <th className="py-2 px-3">操作</th>
                    <th className="py-2 px-3">累计时间 (ns)</th>
                    <th className="py-2 px-3">态保真度 F</th>
                    <th className="py-2 px-3">纯度 Tr(ρ²)</th>
                    <th className="py-2 px-3">{obsName} Bloch (X, Y, Z)</th>
                    <th className="py-2 px-3">稳定子 / 主要布居</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-hairline)]">
                  {primaryRes.steps.map((s) => {
                    const isCur = s.step_index === stepIdx;
                    const isFuture = s.step_index > Math.ceil(clampedCursor + 0.02);
                    const bv = s.bloch_vectors[String(observedQubit)] || [0, 0, 0];
                    const topP = Object.entries(s.populations)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .filter(([, p]) => p > 0.01)
                      .map(([b, p]) => `|${b}⟩:${(p * 100).toFixed(1)}%`)
                      .join('  ');
                    return (
                      <tr
                        key={s.step_index}
                        onClick={() => (onStepJump || onTimeCursorChange)?.(s.step_index)}
                        className={`cursor-pointer transition-all ${
                          isCur
                            ? 'bg-[#0A84FF]/12 text-[#0A84FF] font-semibold'
                            : isFuture
                            ? 'opacity-40 hover:opacity-80 hover:bg-black/[0.025] dark:hover:bg-white/[0.04]'
                            : 'hover:bg-black/[0.025] dark:hover:bg-white/[0.04]'
                        }`}
                      >
                        <td className="py-1.5 px-3">{s.step_index}</td>
                        <td className="py-1.5 px-3">{formatOpLabelClean(s.operation_label)}</td>
                        <td className="py-1.5 px-3">{s.elapsed_time_ns.toFixed(1)}</td>
                        <td className="py-1.5 px-3">
                          {s.state_fidelity !== null ? s.state_fidelity.toFixed(5) : '—'}
                        </td>
                        <td className="py-1.5 px-3">
                          {s.purity !== null ? s.purity.toFixed(5) : '—'}
                        </td>
                        <td className="py-1.5 px-3">
                          ({bv[0] >= 0 ? '+' : ''}
                          {bv[0].toFixed(3)}, {bv[1] >= 0 ? '+' : ''}
                          {bv[1].toFixed(3)}, {bv[2] >= 0 ? '+' : ''}
                          {bv[2].toFixed(3)})
                        </td>
                        <td className="py-1.5 px-3 text-[11px] text-[var(--text-secondary)]">
                          {s.stabilizers ? s.stabilizers.join(', ') : topP}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
