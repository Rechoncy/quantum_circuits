import React, { useState } from 'react';
import { Download, Maximize2, Minimize2, Play } from 'lucide-react';
import type { SweepResponse } from '../types';
import { PlotlyChart } from './PlotlyChart';

interface SweepWorkspaceProps {
  sweepResponse: SweepResponse | null;
  darkMode: boolean;
  onRunSweep: () => void;
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
  sweepResponse,
  darkMode,
  onRunSweep,
  isBusy,
}) => {
  const [focused1D, setFocused1D] = useState<'none' | 'pop' | 'bloch' | 'fid' | 'table'>('none');
  const [view2D, setView2D] = useState<'all' | 'pop' | 'fid' | 'z0'>('all');

  const textColor = darkMode ? '#E5E5EA' : '#1D1D1F';
  const mutedColor = darkMode ? '#98989D' : '#6E6E73';
  const gridColor = darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(15,23,42,0.07)';
  const plotBg = 'rgba(0,0,0,0)';

  const handleExportCsv = () => {
    if (!sweepResponse || sweepResponse.mode !== '1d') return;
    const basis = Object.keys(sweepResponse.populations);
    const header = [
      sweepResponse.param_x,
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
  // 2D SWEEP PHASE DIAGRAM VIEW (Clean 3-card side-by-side or single focus)
  // =========================================================================
  if (sweepResponse.mode === '2d') {
    const cards: {
      id: 'pop' | 'fid' | 'z0';
      title: string;
      z: number[][];
      colorscale: any;
      zmin?: number;
      zmax?: number;
      unit: string;
    }[] = [
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

    const visibleCards = view2D === 'all' ? cards : cards.filter((c) => c.id === view2D);

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
          </div>

          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex shrink-0">
            {(
              [
                ['all', '三图并排'],
                ['pop', '布居相图'],
                ['fid', '保真度'],
                ['z0', `⟨Z(${obsName})⟩ 相图`],
              ] as const
            ).map(([k, lbl]) => (
              <button
                key={k}
                onClick={() => setView2D(k)}
                className={`px-2.5 py-0.5 rounded-md text-[11px] cursor-pointer transition-colors ${
                  view2D === k
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
            view2D === 'all' ? 'grid-cols-1 lg:grid-cols-3' : 'grid-cols-1'
          }`}
        >
          {visibleCards.map((c) => (
            <div
              key={c.id}
              className="flex flex-col studio-card overflow-hidden min-h-[290px]"
            >
              <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                  {c.title}
                </span>
                <button
                  onClick={() => setView2D(view2D === c.id ? 'all' : c.id)}
                  title={view2D === c.id ? '返回并排视图' : '放大查看此相图'}
                  className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                >
                  {view2D === c.id ? (
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
                  ]}
                  layout={{
                    paper_bgcolor: plotBg,
                    plot_bgcolor: plotBg,
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
          {maxFid !== null && minFid !== null && (
            <span className="hidden md:inline-flex px-2 py-0.5 rounded-md bg-[#0A84FF]/10 text-[#0A84FF] font-mono-num text-[11px] font-semibold">
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
        {/* Top-Left: State Populations */}
        {isVisible('pop') && (
          <div className="flex flex-col studio-card overflow-hidden min-h-0">
            <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
              <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                态布居随 [{sweepResponse.param_x}] 扫描
              </span>
              <button
                onClick={() => setFocused1D(focused1D === 'pop' ? 'none' : 'pop')}
                className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                title={focused1D === 'pop' ? '还原四宫格' : '放大此图'}
              >
                {focused1D === 'pop' ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              </button>
            </div>
            <div className="flex-1 min-h-0 p-1.5">
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
