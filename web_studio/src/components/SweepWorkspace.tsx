import React, { useState } from 'react';
import { ArrowRightLeft, CheckCircle2, Download, Maximize2, Minimize2, Play, Sparkles } from 'lucide-react';
import type { SweepResponse, VariationalOptimizeResponse } from '../types';
import { PlotlyChart } from './PlotlyChart';

interface SweepWorkspaceProps {
  activeSubMode?: '1d' | '2d' | 'vqe';
  sweepResponse: SweepResponse | null;
  vqeResponse?: VariationalOptimizeResponse | null;
  darkMode: boolean;
  onRunSweep: () => void;
  onApplyOptimalParams?: (params: Record<string, number>, runSim: boolean) => void;
  isBusy: boolean;
}

const SWEEP_PALETTE = [
  '#0A84FF',
  '#10B981',
  '#F59E0B',
  '#8B5CF6',
  '#EC4899',
  '#06B6D4',
  '#EF4444',
  '#6366F1',
];

function formatSweepKet(bstr: string): string {
  if (!bstr) return '|—⟩';
  if (bstr.length <= 10) return `|${bstr}⟩`;
  if (/^0+$/.test(bstr)) return `|0⟩⊗${bstr.length}`;
  if (/^1+$/.test(bstr)) return `|1⟩⊗${bstr.length}`;
  return `|${bstr.slice(0, 4)}…${bstr.slice(-4)}⟩`;
}

export const SweepWorkspace: React.FC<SweepWorkspaceProps> = ({
  activeSubMode = '1d',
  sweepResponse,
  vqeResponse = null,
  darkMode,
  onRunSweep,
  onApplyOptimalParams,
  isBusy,
}) => {
  const [focused1D, setFocused1D] = useState<'none' | 'pop' | 'bloch' | 'fid' | 'table'>('none');
  const [topLeft1DMode, setTopLeft1DMode] = useState<'auto' | 'energy' | 'pop'>('auto');
  const [view2D, setView2D] = useState<'all' | 'energy' | 'pop' | 'fid' | 'z0'>('all');

  const textColor = darkMode ? '#E5E5EA' : '#1D1D1F';
  const mutedColor = darkMode ? '#98989D' : '#6E6E73';
  const gridColor = darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.07)';
  const plotBg = 'rgba(0,0,0,0)';

  const handleExportCsv = () => {
    if (!sweepResponse || sweepResponse.mode !== '1d') return;
    const basis = Object.keys(sweepResponse.populations);
    const hasEn = Array.isArray(sweepResponse.energies) && sweepResponse.energies.length === sweepResponse.x_values.length;
    const header = [
      sweepResponse.param_x,
      ...(hasEn ? ['energy_H'] : []),
      'fidelity',
      'purity',
      'X0',
      'Y0',
      'Z0',
      ...basis.map((b) => `P_${b}`),
    ].join(',');
    const rows = sweepResponse.x_values.map((xv, i) =>
      [
        xv,
        ...(hasEn ? [sweepResponse.energies![i]] : []),
        sweepResponse.fidelities[i],
        sweepResponse.purities[i],
        sweepResponse.bloch_q0[i][0],
        sweepResponse.bloch_q0[i][1],
        sweepResponse.bloch_q0[i][2],
        ...basis.map((b) => sweepResponse.populations[b][i]),
      ].join(',')
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sweep_${sweepResponse.param_x}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // =========================================================================
  // VQE / QAOA CLOSED-LOOP VARIATIONAL OPTIMIZER VIEW
  // =========================================================================
  if (activeSubMode === 'vqe') {
    if (!vqeResponse) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center dot-grid-canvas gap-3 p-8">
          <div className="studio-card max-w-md p-6 text-center space-y-3">
            <div className="text-[15px] font-semibold text-[var(--text-primary)] flex items-center justify-center gap-1.5">
              <Sparkles className="w-4 h-4 text-[#0A84FF]" />
              VQE / QAOA 闭环变分基态求解器
            </div>
            <p className="text-[12.5px] text-[var(--text-secondary)] leading-relaxed">
              利用 SciPy 经典优化器（COBYLA / Powell / Nelder-Mead / L-BFGS-B）联合优化线路中所有符号变分参数，自动搜索目标泡利哈密顿量 ⟨ψ(θ)|H|ψ(θ)⟩ 的最低基态能量与最优参数向量。
            </p>
            <button
              onClick={onRunSweep}
              disabled={isBusy}
              className="mx-auto px-4 py-2 rounded-xl studio-btn-primary text-[13px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-60"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              立即启动变分基态寻优
            </button>
          </div>
        </div>
      );
    }

    const evalSteps = vqeResponse.energy_trace.map((_, i) => i + 1);
    const handleExportVqeCsv = () => {
      const lines = [
        'eval_step,energy,best_energy_so_far',
        ...vqeResponse.energy_trace.map(
          (e, i) => `${i + 1},${e},${vqeResponse.best_so_far_trace[i] ?? e}`
        ),
      ];
      const blob = new Blob(['\uFEFF' + lines.join('\n')], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vqe_convergence_${vqeResponse.method}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    };

    return (
      <div className="flex-1 min-h-0 flex flex-col bg-[var(--bg-app)] overflow-hidden">
        {/* Header Bar */}
        <div className="h-11 px-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 text-[12px] min-w-0">
            <span className="font-semibold text-[var(--text-primary)] whitespace-nowrap">
              VQE / QAOA 闭环变分寻优
            </span>
            <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[11px] text-[var(--text-secondary)] truncate">
              {vqeResponse.method} · {vqeResponse.backend} · {vqeResponse.param_names.length} 参数
            </span>
            <span className="hidden md:inline-flex px-2 py-0.5 rounded-md bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 font-mono-num text-[11px] font-semibold">
              E* = {vqeResponse.optimal_energy.toFixed(5)}
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleExportVqeCsv}
              className="px-2.5 py-1 rounded-lg border border-[var(--border-input)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11.5px] font-medium flex items-center gap-1 cursor-pointer transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              导出收敛 CSV
            </button>
            {onApplyOptimalParams && (
              <button
                onClick={() => onApplyOptimalParams(vqeResponse.optimal_params, true)}
                className="px-3 py-1 rounded-lg studio-btn-primary text-[11.5px] font-semibold flex items-center gap-1.5 cursor-pointer"
              >
                <ArrowRightLeft className="w-3.5 h-3.5" />
                写回最优参数并演化
              </button>
            )}
          </div>
        </div>

        {/* 4-KPI Strip */}
        <div className="px-3.5 pt-3 grid grid-cols-2 lg:grid-cols-4 gap-2.5 shrink-0">
          <div className="studio-kpi px-3 py-2">
            <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              初始期望能量 E₀
            </div>
            <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] mt-0.5">
              {vqeResponse.initial_energy.toFixed(5)}
            </div>
          </div>
          <div className="studio-kpi px-3 py-2">
            <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              收敛基态能量 E*
            </div>
            <div className="text-[15px] font-semibold font-mono-num text-emerald-600 dark:text-emerald-400 mt-0.5 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
              {vqeResponse.optimal_energy.toFixed(5)}
            </div>
          </div>
          <div className="studio-kpi px-3 py-2">
            <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              变分能量下降 ΔE
            </div>
            <div className="text-[15px] font-semibold font-mono-num text-[#0A84FF] mt-0.5">
              -{Math.max(0, vqeResponse.energy_improvement).toFixed(5)}
            </div>
          </div>
          <div className="studio-kpi px-3 py-2">
            <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              线路评估次数 / 保真度
            </div>
            <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] mt-0.5">
              {vqeResponse.num_evaluations} 次 · F={vqeResponse.optimal_fidelity.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Convergence Curve + Optimal Parameters Table */}
        <div className="flex-1 min-h-0 p-3.5 grid grid-cols-1 lg:grid-cols-12 gap-3 overflow-hidden">
          <div className="lg:col-span-7 flex flex-col studio-card overflow-hidden min-h-[250px]">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                变分能量收敛曲线 ⟨ψ(θₖ)|H|ψ(θₖ)⟩
              </span>
              <span className="text-[11px] font-mono-num text-[var(--text-secondary)] truncate max-w-[220px]">
                H = {vqeResponse.hamiltonian_str}
              </span>
            </div>
            <div className="flex-1 min-h-0 p-2">
              <PlotlyChart
                data={[
                  {
                    x: evalSteps,
                    y: vqeResponse.energy_trace,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '当步评估 ⟨H⟩',
                    line: { width: 1.5, color: 'rgba(10,132,255,0.55)' },
                    marker: { size: 4.5, color: '#0A84FF' },
                  },
                  {
                    x: evalSteps,
                    y: vqeResponse.best_so_far_trace,
                    type: 'scatter',
                    mode: 'lines',
                    name: '最优包络 min ⟨H⟩',
                    line: { width: 2.6, color: '#10B981' },
                  },
                ]}
                layout={{
                  paper_bgcolor: plotBg,
                  plot_bgcolor: plotBg,
                  hovermode: 'x unified',
                  margin: { l: 54, r: 16, t: 16, b: 42 },
                  xaxis: {
                    title: { text: '目标函数评估步数 k', font: { size: 11, color: mutedColor } },
                    gridcolor: gridColor,
                    tickfont: { color: mutedColor, size: 10.5 },
                  },
                  yaxis: {
                    title: { text: '能量期望值 ⟨H⟩', font: { size: 11, color: mutedColor } },
                    gridcolor: gridColor,
                    tickfont: { color: mutedColor, size: 10.5 },
                  },
                  legend: { orientation: 'h', y: 1.12, font: { size: 10.5, color: textColor } },
                }}
              />
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col studio-card overflow-hidden min-h-[250px]">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                最优变分参数 θ* 与基态主导分布
              </span>
              {onApplyOptimalParams && (
                <button
                  onClick={() => onApplyOptimalParams(vqeResponse.optimal_params, false)}
                  className="text-[11px] text-[#0A84FF] hover:underline font-medium cursor-pointer"
                >
                  仅写回变量表
                </button>
              )}
            </div>
            <div className="flex-1 overflow-auto p-3 space-y-3">
              <table className="w-full text-[11.5px] font-mono-num border-collapse">
                <thead>
                  <tr className="border-b border-[var(--border-hairline)] text-left text-[var(--text-secondary)]">
                    <th className="py-1.5 pr-2">符号参数</th>
                    <th className="py-1.5 pr-2">初始值 θ₀</th>
                    <th className="py-1.5 pr-2">最优值 θ*</th>
                    <th className="py-1.5">位移 Δθ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-hairline)]">
                  {vqeResponse.param_names.map((pname) => {
                    const v0 = vqeResponse.initial_params[pname] ?? 0;
                    const v1 = vqeResponse.optimal_params[pname] ?? 0;
                    const diff = v1 - v0;
                    return (
                      <tr key={pname}>
                        <td className="py-1.5 pr-2 font-semibold text-[#0A84FF]">{pname}</td>
                        <td className="py-1.5 pr-2 text-[var(--text-secondary)]">{v0.toFixed(4)}</td>
                        <td className="py-1.5 pr-2 font-semibold text-emerald-600 dark:text-emerald-400">
                          {v1.toFixed(4)}
                        </td>
                        <td className="py-1.5 text-[var(--text-secondary)]">
                          {diff >= 0 ? '+' : ''}
                          {diff.toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {vqeResponse.dominant_states.length > 0 && (
                <div className="pt-2 border-t border-[var(--border-hairline)] space-y-1.5">
                  <div className="text-[11px] font-semibold text-[var(--text-secondary)]">
                    最优态 |ψ(θ*)⟩ 主导计算基态分布
                  </div>
                  {vqeResponse.dominant_states.slice(0, 6).map((st) => (
                    <div key={st.bitstring} className="space-y-0.5 font-mono-num text-[11px]">
                      <div className="flex justify-between">
                        <span>{formatSweepKet(st.bitstring)}</span>
                        <span>{(st.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-black/[0.06] dark:bg-white/[0.09] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{ width: `${Math.min(100, st.probability * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!sweepResponse) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center dot-grid-canvas gap-3 p-8">
        <div className="studio-card max-w-md p-6 text-center space-y-3">
          <div className="text-[15px] font-semibold text-[var(--text-primary)]">
            参数扫描实验室 (1D / 2D)
          </div>
          <p className="text-[12.5px] text-[var(--text-secondary)] leading-relaxed">
            在右侧检查器选择任意线路符号参数（如 theta、phi、p_noise）或硬件物理参数（T1、T2*、门时长、退极化率、ZZ 串扰），点击运行即可生成交互式扫描曲线或二维相图。
          </p>
          <button
            onClick={onRunSweep}
            disabled={isBusy}
            className="mx-auto px-4 py-2 rounded-xl studio-btn-primary text-[13px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-60"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            立即执行参数扫描
          </button>
        </div>
      </div>
    );
  }

  const obsName = sweepResponse.observed_qubit_name || `q${sweepResponse.observed_qubit ?? 0}`;

  // =========================================================================
  // 2D SWEEP PHASE DIAGRAM VIEW (3-card or 2x2 with Energy Landscape <H>)
  // =========================================================================
  if (sweepResponse.mode === '2d') {
    const hasEnergy2D =
      Array.isArray(sweepResponse.energy_grid) && sweepResponse.energy_grid.length > 0;

    let minE2D: { val: number; x: number; y: number } | null = null;
    if (hasEnergy2D && sweepResponse.energy_grid) {
      let best = Infinity;
      let bx = sweepResponse.x_values[0] ?? 0;
      let by = sweepResponse.y_values[0] ?? 0;
      sweepResponse.energy_grid.forEach((row, iy) => {
        row.forEach((v, ix) => {
          if (v < best) {
            best = v;
            bx = sweepResponse.x_values[ix] ?? bx;
            by = sweepResponse.y_values[iy] ?? by;
          }
        });
      });
      if (Number.isFinite(best)) minE2D = { val: best, x: bx, y: by };
    }

    const cards: {
      id: 'energy' | 'pop' | 'fid' | 'z0';
      title: string;
      z: number[][];
      colorscale: any;
      zmin?: number;
      zmax?: number;
      unit: string;
      minMarker?: { x: number; y: number; val: number } | null;
    }[] = [
      ...(hasEnergy2D && sweepResponse.energy_grid
        ? [
            {
              id: 'energy' as const,
              title: '变分能量景观 ⟨H⟩(x, y)',
              z: sweepResponse.energy_grid,
              colorscale: 'Plasma',
              unit: '⟨H⟩',
              minMarker: minE2D,
            },
          ]
        : []),
      {
        id: 'pop',
        title: `基态布居 P(${formatSweepKet(sweepResponse.target_bitstring)})`,
        z: sweepResponse.population_grid,
        colorscale: 'Viridis',
        zmin: 0,
        zmax: 1,
        unit: 'P',
      },
      {
        id: 'fid',
        title: '含噪态保真度 F(x, y)',
        z: sweepResponse.fidelity_grid,
        colorscale: 'Magma',
        unit: 'F',
      },
      {
        id: 'z0',
        title: `⟨Z(${obsName})⟩(x, y) 期望值相图`,
        z: sweepResponse.z0_grid,
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1,
        unit: `⟨Z(${obsName})⟩`,
      },
    ];

    const effectiveView2D = view2D === 'energy' && !hasEnergy2D ? 'all' : view2D;
    const visibleCards =
      effectiveView2D === 'all' ? cards : cards.filter((c) => c.id === effectiveView2D);

    const navTabs: [typeof view2D, string][] = [
      ['all', hasEnergy2D ? '四图总览' : '三图并排'],
      ...(hasEnergy2D ? ([['energy', '能量景观 ⟨H⟩']] as [typeof view2D, string][]) : []),
      ['pop', '布居相图'],
      ['fid', '保真度'],
      ['z0', `⟨Z(${obsName})⟩`],
    ];

    return (
      <div className="flex-1 min-h-0 flex flex-col bg-[var(--bg-app)] overflow-hidden">
        {/* Top Summary Header */}
        <div className="h-11 px-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 text-[12px] min-w-0">
            <span className="font-semibold text-[var(--text-primary)] whitespace-nowrap">
              2D 参数相图
            </span>
            <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[11px] text-[var(--text-secondary)] truncate">
              {sweepResponse.param_x} × {sweepResponse.param_y} · {sweepResponse.x_values.length}×
              {sweepResponse.y_values.length} · {sweepResponse.backend}
            </span>
            {minE2D && (
              <span className="hidden xl:inline-flex px-2 py-0.5 rounded-md bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 font-mono-num text-[11px] font-semibold whitespace-nowrap">
                E_min = {minE2D.val.toFixed(4)} @ ({minE2D.x.toFixed(2)}, {minE2D.y.toFixed(2)})
              </span>
            )}
          </div>

          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex shrink-0">
            {navTabs.map(([k, lbl]) => (
              <button
                key={k}
                onClick={() => setView2D(k)}
                className={`px-2.5 py-0.5 rounded-md text-[11px] cursor-pointer transition-colors ${
                  effectiveView2D === k
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>

        {/* Balanced Phase Diagram Grid */}
        <div
          className={`flex-1 min-h-0 p-3.5 grid gap-3 overflow-auto ${
            effectiveView2D === 'all'
              ? hasEnergy2D
                ? 'grid-cols-1 lg:grid-cols-2'
                : 'grid-cols-1 lg:grid-cols-3'
              : 'grid-cols-1'
          }`}
        >
          {visibleCards.map((c) => (
            <div
              key={c.id}
              className="flex flex-col studio-card overflow-hidden min-h-[260px]"
            >
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                  {c.title}
                </span>
                <button
                  onClick={() => setView2D(effectiveView2D === c.id ? 'all' : c.id)}
                  title={effectiveView2D === c.id ? '返回并排视图' : '放大查看此相图'}
                  className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                >
                  {effectiveView2D === c.id ? (
                    <Minimize2 className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize2 className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
              <div className="flex-1 min-h-0 p-2">
                <PlotlyChart
                  data={[
                    {
                      type: 'heatmap',
                      x: sweepResponse.x_values,
                      y: sweepResponse.y_values,
                      z: c.z,
                      colorscale: c.colorscale,
                      zmin: c.zmin,
                      zmax: c.zmax,
                      colorbar: { thickness: 11, len: 0.9, tickfont: { size: 10, color: mutedColor } },
                      hovertemplate: `${sweepResponse.param_x}=%{x:.3f}<br>${sweepResponse.param_y}=%{y:.3f}<br>${c.unit}=%{z:.4f}<extra></extra>`,
                    },
                    ...(c.minMarker
                      ? [
                          {
                            type: 'scatter' as const,
                            mode: 'markers' as const,
                            x: [c.minMarker.x],
                            y: [c.minMarker.y],
                            name: 'E_min',
                            marker: {
                              symbol: 'star',
                              size: 13,
                              color: '#10B981',
                              line: { color: '#FFFFFF', width: 1.5 },
                            },
                            hovertemplate: `最优点 E_min=${c.minMarker.val.toFixed(4)}<br>(%{x:.3f}, %{y:.3f})<extra></extra>`,
                          },
                        ]
                      : []),
                  ]}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    showlegend: false,
                    margin: { l: 54, r: 12, t: 10, b: 44 },
                    xaxis: {
                      title: { text: sweepResponse.param_x_label, font: { size: 11, color: mutedColor } },
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    yaxis: {
                      title: { text: sweepResponse.param_y_label, font: { size: 11, color: mutedColor } },
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // =========================================================================
  // 1D SWEEP VIEW (Header KPI Bar + Expandable 2x2 Elevated macOS Cards)
  // =========================================================================
  const basisKeys = Object.keys(sweepResponse.populations);
  const activeBasis = basisKeys
    .filter((b) => Math.max(...sweepResponse.populations[b]) > 0.015 || basisKeys.length <= 4)
    .slice(0, 8);
  const maxFid = sweepResponse.fidelities.length > 0 ? Math.max(...sweepResponse.fidelities) : null;
  const minFid = sweepResponse.fidelities.length > 0 ? Math.min(...sweepResponse.fidelities) : null;

  const hasEnergy1D =
    Array.isArray(sweepResponse.energies) &&
    sweepResponse.energies.length === sweepResponse.x_values.length;
  const showEnergy1D =
    hasEnergy1D && (topLeft1DMode === 'auto' || topLeft1DMode === 'energy');

  let minE1D: { val: number; x: number; idx: number } | null = null;
  if (hasEnergy1D && sweepResponse.energies) {
    let best = Infinity;
    let bIdx = 0;
    sweepResponse.energies.forEach((v, idx) => {
      if (v < best) {
        best = v;
        bIdx = idx;
      }
    });
    if (Number.isFinite(best)) {
      minE1D = { val: best, x: sweepResponse.x_values[bIdx] ?? 0, idx: bIdx };
    }
  }

  const isVisible = (id: 'pop' | 'bloch' | 'fid' | 'table') =>
    focused1D === 'none' || focused1D === id;

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-[var(--bg-app)] overflow-hidden">
      {/* Top Summary Header Bar */}
      <div className="h-11 px-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2.5 text-[12px] min-w-0">
          <span className="font-semibold text-[var(--text-primary)] whitespace-nowrap">
            1D 参数扫描
          </span>
          <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[11px] text-[var(--text-secondary)] truncate">
            {sweepResponse.param_x_label} · {sweepResponse.x_values.length} 点 · {sweepResponse.backend}
          </span>
          {minE1D && (
            <span className="hidden sm:inline-flex px-2 py-0.5 rounded-md bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 font-mono-num text-[11px] font-semibold whitespace-nowrap">
              E_min = {minE1D.val.toFixed(4)} @ {sweepResponse.param_x}={minE1D.x.toFixed(3)}
            </span>
          )}
          {maxFid !== null && minFid !== null && (
            <span className="hidden lg:inline-flex px-2 py-0.5 rounded-md bg-[#0A84FF]/10 text-[#0A84FF] font-mono-num text-[11px] font-semibold">
              F ∈ [{minFid.toFixed(4)}, {maxFid.toFixed(4)}]
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {focused1D !== 'none' && (
            <button
              onClick={() => setFocused1D('none')}
              className="px-2.5 py-1 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11.5px] font-medium cursor-pointer"
            >
              返回四宫格
            </button>
          )}
          <button
            onClick={handleExportCsv}
            className="px-2.5 py-1 rounded-lg border border-[var(--border-input)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11.5px] font-medium flex items-center gap-1 cursor-pointer transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            导出 CSV
          </button>
        </div>
      </div>

      {/* Main 2x2 or Focused Card Stage */}
      <div
        className={`flex-1 min-h-0 p-3.5 grid gap-3 overflow-hidden ${
          focused1D === 'none' ? 'grid-cols-2 grid-rows-2' : 'grid-cols-1 grid-rows-1'
        }`}
      >
        {/* Top-Left: Variational Energy <H> OR State Populations */}
        {isVisible('pop') && (
          <div className="flex flex-col studio-card overflow-hidden min-h-0">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                {showEnergy1D
                  ? `变分能量 ⟨H⟩ 随 [${sweepResponse.param_x}] 扫描`
                  : `态布居随 [${sweepResponse.param_x}] 扫描`}
              </span>
              <div className="flex items-center gap-1.5 shrink-0">
                {hasEnergy1D && (
                  <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-md inline-flex text-[10.5px]">
                    <button
                      onClick={() => setTopLeft1DMode('energy')}
                      className={`px-2 py-0.5 rounded cursor-pointer ${
                        showEnergy1D
                          ? 'bg-[var(--bg-panel)] text-emerald-600 dark:text-emerald-400 font-semibold shadow-2xs'
                          : 'text-[var(--text-secondary)]'
                      }`}
                    >
                      ⟨H⟩ 能量
                    </button>
                    <button
                      onClick={() => setTopLeft1DMode('pop')}
                      className={`px-2 py-0.5 rounded cursor-pointer ${
                        !showEnergy1D
                          ? 'bg-[var(--bg-panel)] text-[#0A84FF] font-semibold shadow-2xs'
                          : 'text-[var(--text-secondary)]'
                      }`}
                    >
                      态布居
                    </button>
                  </div>
                )}
                <button
                  onClick={() => setFocused1D(focused1D === 'pop' ? 'none' : 'pop')}
                  className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                  title={focused1D === 'pop' ? '还原四宫格' : '放大此图'}
                >
                  {focused1D === 'pop' ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
            <div className="flex-1 min-h-0 p-1.5">
              {showEnergy1D && sweepResponse.energies ? (
                <PlotlyChart
                  data={[
                    {
                      x: sweepResponse.x_values,
                      y: sweepResponse.energies,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: '⟨H⟩',
                      line: { width: 2.4, color: '#10B981' },
                      marker: { size: 4.5, color: '#10B981' },
                      hovertemplate: '⟨H⟩ = %{y:.4f}<extra></extra>',
                    },
                    ...(minE1D
                      ? [
                          {
                            x: [minE1D.x],
                            y: [minE1D.val],
                            type: 'scatter' as const,
                            mode: 'markers' as const,
                            name: 'E_min',
                            marker: {
                              symbol: 'star',
                              size: 12,
                              color: '#F59E0B',
                              line: { color: '#FFFFFF', width: 1.2 },
                            },
                            hovertemplate: `基态最优点: E_min = ${minE1D.val.toFixed(4)}<extra></extra>`,
                          },
                        ]
                      : []),
                  ]}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    hovermode: 'x unified',
                    margin: { l: 50, r: 14, t: 12, b: 38 },
                    xaxis: {
                      title: { text: sweepResponse.param_x_label, font: { size: 10.5, color: mutedColor } },
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    yaxis: {
                      title: { text: '⟨H⟩', font: { size: 10.5, color: mutedColor } },
                      gridcolor: gridColor,
                      zerolinecolor: darkMode ? 'rgba(255,255,255,0.16)' : 'rgba(15,23,42,0.16)',
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    legend: { orientation: 'h', y: 1.13, font: { size: 10, color: textColor } },
                  }}
                />
              ) : (
                <PlotlyChart
                  data={activeBasis.map((b, idx) => ({
                    x: sweepResponse.x_values,
                    y: sweepResponse.populations[b],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: formatSweepKet(b),
                    line: { width: 2.2, color: SWEEP_PALETTE[idx % SWEEP_PALETTE.length] },
                    marker: { size: 4.5 },
                    hovertemplate: `${formatSweepKet(b)}: %{y:.3f}<extra></extra>`,
                  }))}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
                    hovermode: 'x unified',
                    margin: { l: 46, r: 14, t: 12, b: 38 },
                    xaxis: {
                      title: { text: sweepResponse.param_x_label, font: { size: 10.5, color: mutedColor } },
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    yaxis: {
                      title: { text: 'P(|x⟩)', font: { size: 10.5, color: mutedColor } },
                      range: [-0.04, 1.06],
                      gridcolor: gridColor,
                      tickfont: { color: mutedColor, size: 10.5 },
                    },
                    legend: { orientation: 'h', y: 1.13, font: { size: 10, color: textColor } },
                  }}
                />
              )}
            </div>
          </div>
        )}

        {/* Top-Right: Q0 Bloch Expectations */}
        {isVisible('bloch') && (
          <div className="flex flex-col studio-card overflow-hidden min-h-0">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)] whitespace-nowrap">
                  {obsName} Bloch · [{sweepResponse.param_x}]
                </span>
                <div className="flex items-center gap-2 shrink-0">
                  <div className="hidden lg:flex items-center gap-2 text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-[2px] bg-[#0A84FF]" />⟨X⟩
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-[2px] bg-[#F59E0B]" />⟨Y⟩
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-[2px] bg-[#10B981]" />⟨Z⟩
                    </span>
                  </div>
                <button
                  onClick={() => setFocused1D(focused1D === 'bloch' ? 'none' : 'bloch')}
                  className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                  title={focused1D === 'bloch' ? '还原四宫格' : '放大此图'}
                >
                  {focused1D === 'bloch' ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
            <div className="flex-1 min-h-0 p-1.5">
              <PlotlyChart
                data={[
                  {
                    x: sweepResponse.x_values,
                    y: sweepResponse.bloch_q0.map((v) => v[0]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '⟨X₀⟩',
                    line: { color: '#0A84FF', width: 2.2 },
                    marker: { size: 4.5 },
                  },
                  {
                    x: sweepResponse.x_values,
                    y: sweepResponse.bloch_q0.map((v) => v[1]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '⟨Y₀⟩',
                    line: { color: '#F59E0B', width: 2.2 },
                    marker: { size: 4.5 },
                  },
                  {
                    x: sweepResponse.x_values,
                    y: sweepResponse.bloch_q0.map((v) => v[2]),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '⟨Z₀⟩',
                    line: { color: '#10B981', width: 2.2 },
                    marker: { size: 4.5 },
                  },
                ]}
                layout={{
                  paper_bgcolor: plotBg,
                  plot_bgcolor: plotBg,
                  showlegend: false,
                  hovermode: 'x unified',
                  margin: { l: 46, r: 14, t: 12, b: 38 },
                  xaxis: {
                    title: { text: sweepResponse.param_x_label, font: { size: 10.5, color: mutedColor } },
                    gridcolor: gridColor,
                    tickfont: { color: mutedColor, size: 10.5 },
                  },
                  yaxis: {
                    range: [-1.1, 1.1],
                    dtick: 0.5,
                    gridcolor: gridColor,
                    zerolinecolor: darkMode ? 'rgba(255,255,255,0.16)' : 'rgba(15,23,42,0.16)',
                    tickfont: { color: mutedColor, size: 10.5 },
                  },
                }}
              />
            </div>
          </div>
        )}

        {/* Bottom-Left: Fidelity & Purity */}
        {isVisible('fid') && (
          <div className="flex flex-col studio-card overflow-hidden min-h-0">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                含噪保真度 F 与纯度 Tr(ρ²)
              </span>
              <button
                onClick={() => setFocused1D(focused1D === 'fid' ? 'none' : 'fid')}
                className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                title={focused1D === 'fid' ? '还原四宫格' : '放大此图'}
              >
                {focused1D === 'fid' ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              </button>
            </div>
            <div className="flex-1 min-h-0 p-1.5">
              <PlotlyChart
                data={[
                  {
                    x: sweepResponse.x_values,
                    y: sweepResponse.fidelities,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '保真度 F',
                    line: { color: '#0A84FF', width: 2.3 },
                    marker: { size: 4.5 },
                  },
                  {
                    x: sweepResponse.x_values,
                    y: sweepResponse.purities,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '纯度 Tr(ρ²)',
                    line: { color: '#8B5CF6', width: 1.9, dash: 'dot' },
                    marker: { size: 4 },
                  },
                ]}
                layout={{
                  paper_bgcolor: plotBg,
                  plot_bgcolor: plotBg,
                  hovermode: 'x unified',
                  margin: { l: 48, r: 14, t: 12, b: 38 },
                  xaxis: {
                    title: { text: sweepResponse.param_x_label, font: { size: 10.5, color: mutedColor } },
                    gridcolor: gridColor,
                    tickfont: { color: mutedColor, size: 10.5 },
                  },
                  yaxis: { gridcolor: gridColor, tickfont: { color: mutedColor, size: 10.5 } },
                  legend: { orientation: 'h', y: 1.13, font: { size: 10, color: textColor } },
                }}
              />
            </div>
          </div>
        )}

        {/* Bottom-Right: Sweep Data Table */}
        {isVisible('table') && (
          <div className="flex flex-col studio-card overflow-hidden min-h-0">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                扫描采样数据明细
              </span>
              <button
                onClick={() => setFocused1D(focused1D === 'table' ? 'none' : 'table')}
                className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                title={focused1D === 'table' ? '还原四宫格' : '放大数据表'}
              >
                {focused1D === 'table' ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              </button>
            </div>
            <div className="flex-1 overflow-auto px-3 py-1.5">
              <table className="w-full text-[11.5px] font-mono-num border-collapse">
                <thead className="sticky top-0 bg-[var(--bg-panel)]">
                  <tr className="border-b border-[var(--border-hairline)] text-left text-[var(--text-secondary)]">
                    <th className="py-1.5 pr-2">#</th>
                    <th className="py-1.5 pr-2">{sweepResponse.param_x}</th>
                    {hasEnergy1D && <th className="py-1.5 pr-2 text-emerald-600 dark:text-emerald-400">⟨H⟩</th>}
                    <th className="py-1.5 pr-2">F</th>
                    <th className="py-1.5 pr-2">Tr(ρ²)</th>
                    <th className="py-1.5 pr-2">⟨Z({obsName})⟩</th>
                    <th className="py-1.5">主导基态</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-hairline)]">
                  {sweepResponse.x_values.map((xv, i) => {
                    const top2 = [...activeBasis]
                      .sort((a, b) => (sweepResponse.populations[b][i] ?? 0) - (sweepResponse.populations[a][i] ?? 0))
                      .slice(0, 2);
                    return (
                      <tr key={i} className="hover:bg-black/[0.025] dark:hover:bg-white/[0.03]">
                        <td className="py-1.5 pr-1.5 text-[var(--text-secondary)]">{i + 1}</td>
                        <td className="py-1.5 pr-1.5 font-semibold">{xv.toFixed(3)}</td>
                        {hasEnergy1D && sweepResponse.energies && (
                          <td className="py-1.5 pr-1.5 text-emerald-600 dark:text-emerald-400 font-semibold">
                            {sweepResponse.energies[i].toFixed(4)}
                          </td>
                        )}
                        <td className="py-1.5 pr-1.5 text-[#0A84FF] font-semibold">
                          {sweepResponse.fidelities[i].toFixed(4)}
                        </td>
                        <td className="py-1.5 pr-1.5">{sweepResponse.purities[i].toFixed(3)}</td>
                        <td className="py-1.5 pr-1.5">{sweepResponse.bloch_q0[i][2].toFixed(2)}</td>
                        <td className="py-1.5 text-[10.5px] text-[var(--text-secondary)] whitespace-nowrap">
                          {top2
                            .map((b) => `${formatSweepKet(b)} ${(sweepResponse.populations[b][i] * 100).toFixed(0)}%`)
                            .join(' · ')}
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
