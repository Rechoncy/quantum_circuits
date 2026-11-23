import React, { useState } from 'react';
import { ArrowRightLeft, CheckCircle2, Copy, Check, Zap, Sparkles } from 'lucide-react';
import type { OptimizeResponse, ProjectModel } from '../types';
import { CircuitCanvas } from './CircuitCanvas';

export interface DirectSynthPayload {
  mode: 'matrix' | 'hamiltonian';
  num_qubits: number;
  matrix_exprs?: string[][];
  hamiltonian_str?: string;
  hamiltonian_scale?: number;
}

interface OptimizeWorkspaceProps {
  project: ProjectModel;
  optResponse: OptimizeResponse | null;
  onRunOptimize: () => void;
  onRunUnitarySynthesis: (payload: DirectSynthPayload) => void;
  onApplyOptimizedToDesign: (targetNumQubits?: number) => void;
  isBusy: boolean;
}

const MATRIX_PRESETS: {
  id: string;
  label: string;
  mode: 'matrix' | 'hamiltonian';
  num_qubits: number;
  matrix?: string[][];
  ham?: string;
  scale?: number;
}[] = [
  {
    id: 'qft2',
    label: 'QFT₂',
    mode: 'matrix',
    num_qubits: 2,
    matrix: [
      ['0.5', '0.5', '0.5', '0.5'],
      ['0.5', '0.5j', '-0.5', '-0.5j'],
      ['0.5', '-0.5', '0.5', '-0.5'],
      ['0.5', '-0.5j', '-0.5', '0.5j'],
    ],
  },
  {
    id: 'swap',
    label: 'SWAP',
    mode: 'matrix',
    num_qubits: 2,
    matrix: [
      ['1', '0', '0', '0'],
      ['0', '0', '1', '0'],
      ['0', '1', '0', '0'],
      ['0', '0', '0', '1'],
    ],
  },
  {
    id: 'sqiswap',
    label: '√iSWAP',
    mode: 'matrix',
    num_qubits: 2,
    matrix: [
      ['1', '0', '0', '0'],
      ['0', '1/sqrt(2)', '1j/sqrt(2)', '0'],
      ['0', '1j/sqrt(2)', '1/sqrt(2)', '0'],
      ['0', '0', '0', '1'],
    ],
  },
  {
    id: 'fsim',
    label: 'fSim(π/4,π/6)',
    mode: 'matrix',
    num_qubits: 2,
    matrix: [
      ['1', '0', '0', '0'],
      ['0', 'cos(pi/4)', '-1j*sin(pi/4)', '0'],
      ['0', '-1j*sin(pi/4)', 'cos(pi/4)', '0'],
      ['0', '0', '0', 'exp(-1j*pi/6)'],
    ],
  },
  {
    id: 'xy_ham',
    label: 'XY 哈密顿量',
    mode: 'hamiltonian',
    num_qubits: 2,
    ham: '0.5*XX + 0.5*YY + 0.25*ZZ',
    scale: 0.7854,
  },
  {
    id: 'h1q',
    label: 'H (1Q)',
    mode: 'matrix',
    num_qubits: 1,
    matrix: [
      ['1/sqrt(2)', '1/sqrt(2)'],
      ['1/sqrt(2)', '-1/sqrt(2)'],
    ],
  },
];

export const OptimizeWorkspace: React.FC<OptimizeWorkspaceProps> = ({
  project,
  optResponse,
  onRunOptimize,
  onRunUnitarySynthesis,
  onApplyOptimizedToDesign,
  isBusy,
}) => {
  const [optMode, setOptMode] = useState<'circuit' | 'unitary'>('circuit');
  const [synthInputMode, setSynthInputMode] = useState<'matrix' | 'hamiltonian'>('matrix');
  const [synthQubits, setSynthQubits] = useState<number>(2);
  const [matrixExprs, setMatrixExprs] = useState<string[][]>(MATRIX_PRESETS[0].matrix!);
  const [hamStr, setHamStr] = useState('0.5*XX + 0.5*YY + 0.25*ZZ');
  const [hamScale, setHamScale] = useState(0.7854);

  const [codeTab, setCodeTab] = useState<'dsl' | 'qasm'>('dsl');
  const [copied, setCopied] = useState(false);

  const handleCopy = (txt: string) => {
    navigator.clipboard.writeText(txt).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const applyPreset = (p: (typeof MATRIX_PRESETS)[number]) => {
    setSynthInputMode(p.mode);
    setSynthQubits(p.num_qubits);
    if (p.matrix) setMatrixExprs(p.matrix.map((row) => [...row]));
    if (p.ham) setHamStr(p.ham);
    if (p.scale !== undefined) setHamScale(p.scale);
  };

  const handleTriggerSynth = () => {
    onRunUnitarySynthesis({
      mode: synthInputMode,
      num_qubits: synthQubits,
      matrix_exprs: synthInputMode === 'matrix' ? matrixExprs : undefined,
      hamiltonian_str: synthInputMode === 'hamiltonian' ? hamStr : undefined,
      hamiltonian_scale: synthInputMode === 'hamiltonian' ? hamScale : undefined,
    });
  };

  const synthPreviewProject: ProjectModel =
    optResponse?.target_num_qubits && optResponse.target_num_qubits < project.qubits.length
      ? { ...project, qubits: project.qubits.slice(0, optResponse.target_num_qubits) }
      : project;

  const m0 = optResponse?.original_metrics ?? null;
  const m1 = optResponse?.optimized_metrics ?? null;
  const fOrig = optResponse?.noisy_fidelity_original ?? null;
  const fOpt = optResponse?.noisy_fidelity_optimized ?? null;

  return (
    <div className="flex-1 min-h-0 flex flex-col overflow-hidden bg-[var(--bg-app)]">
      {/* Top Unified Mode Switch & Summary Strip */}
      <div className="min-h-12 py-1.5 px-4 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2 shrink-0">
        <div className="flex flex-wrap items-center gap-2.5 text-[12px]">
          {/* Switch between Circuit Optimization & Direct Unitary Synthesis */}
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex shrink-0">
            <button
              onClick={() => setOptMode('circuit')}
              className={`px-2.5 py-1 rounded-md text-[11.5px] cursor-pointer transition-colors ${
                optMode === 'circuit'
                  ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              优化当前线路
            </button>
            <button
              onClick={() => setOptMode('unitary')}
              className={`px-2.5 py-1 rounded-md text-[11.5px] cursor-pointer transition-colors flex items-center gap-1 ${
                optMode === 'unitary'
                  ? 'bg-[var(--bg-panel)] text-[#0A84FF] font-semibold shadow-2xs'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              <Sparkles className="w-3 h-3" />
              目标酉矩阵直出综合
            </button>
          </div>

          {optResponse && (
            <>
              <span className="font-semibold text-[var(--text-primary)] hidden xl:inline">
                {optResponse.gate_set_name}
              </span>
              <span className="px-2 py-0.5 rounded-md bg-[#0A84FF]/10 text-[#0A84FF] font-mono-num text-[11.5px] font-semibold">
                酉保真度 F = {optResponse.unitary_fidelity.toFixed(6)}
              </span>
              {fOpt !== null && (
                <span className="px-2 py-0.5 rounded-md bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 font-mono-num text-[11.5px] font-semibold flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {fOrig !== null
                    ? `QuTiP 含噪: ${fOrig.toFixed(4)} → ${fOpt.toFixed(4)} (${fOpt >= fOrig ? '+' : ''}${(fOpt - fOrig).toFixed(4)})`
                    : `QuTiP 含噪: ${fOpt.toFixed(4)}`}
                </span>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {optResponse && (
            <button
              onClick={() => onApplyOptimizedToDesign(optResponse.target_num_qubits)}
              className="px-3.5 py-1.5 rounded-lg studio-btn-primary text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer"
            >
              <ArrowRightLeft className="w-3.5 h-3.5" />
              应用到设计画布
            </button>
          )}
        </div>
      </div>

      {/* Direct Target Unitary / Hamiltonian Input Card (when optMode === 'unitary') */}
      {optMode === 'unitary' && (
        <div className="px-4 py-3 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] shrink-0 space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11.5px] font-semibold text-[var(--text-secondary)] mr-1">
                快速模板:
              </span>
              {MATRIX_PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => applyPreset(p)}
                  className="px-2 py-0.5 rounded-md border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11px] font-mono-num cursor-pointer transition-colors"
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex text-[11px]">
                <button
                  onClick={() => setSynthInputMode('matrix')}
                  className={`px-2 py-0.5 rounded-md cursor-pointer ${
                    synthInputMode === 'matrix'
                      ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs'
                      : 'text-[var(--text-secondary)]'
                  }`}
                >
                  酉矩阵 U
                </button>
                <button
                  onClick={() => setSynthInputMode('hamiltonian')}
                  className={`px-2 py-0.5 rounded-md cursor-pointer ${
                    synthInputMode === 'hamiltonian'
                      ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs'
                      : 'text-[var(--text-secondary)]'
                  }`}
                >
                  泡利哈密顿量 exp(-i t H)
                </button>
              </div>

              <button
                onClick={handleTriggerSynth}
                disabled={isBusy}
                className="px-3 py-1 rounded-lg studio-btn-primary text-[11.5px] font-semibold flex items-center gap-1 cursor-pointer disabled:opacity-60"
              >
                <Zap className="w-3.5 h-3.5" />
                综合最优原生门
              </button>
            </div>
          </div>

          {synthInputMode === 'matrix' ? (
            <div className="flex items-start gap-4">
              <div className="flex flex-col gap-1.5 shrink-0">
                <span className="text-[11px] text-[var(--text-secondary)]">目标维度</span>
                <div className="inline-flex bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg text-[11px]">
                  {[1, 2].map((nq) => (
                    <button
                      key={nq}
                      onClick={() => {
                        setSynthQubits(nq);
                        const dim = 1 << nq;
                        setMatrixExprs(
                          Array.from({ length: dim }, (_, r) =>
                            Array.from({ length: dim }, (_, c) => (r === c ? '1' : '0'))
                          )
                        );
                      }}
                      className={`px-2 py-0.5 rounded-md cursor-pointer ${
                        synthQubits === nq
                          ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs'
                          : 'text-[var(--text-secondary)]'
                      }`}
                    >
                      {nq}Q ({1 << nq}×{1 << nq})
                    </button>
                  ))}
                </div>
              </div>

              <div
                className="grid gap-1.5 flex-1 max-w-2xl"
                style={{ gridTemplateColumns: `repeat(${1 << synthQubits}, minmax(0, 1fr))` }}
              >
                {matrixExprs.map((row, r) =>
                  row.map((val, c) => (
                    <input
                      key={`${r}-${c}`}
                      value={val}
                      onChange={(e) => {
                        const next = matrixExprs.map((rr) => [...rr]);
                        next[r][c] = e.target.value;
                        setMatrixExprs(next);
                      }}
                      className="px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[11.5px] text-center outline-none focus:border-[#0A84FF]"
                    />
                  ))
                )}
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex-1 min-w-[240px]">
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  泡利哈密顿量 H（支持 X/Y/Z/I 组合，如 0.5*XX + 0.5*YY + 0.25*ZZ）
                </label>
                <input
                  value={hamStr}
                  onChange={(e) => setHamStr(e.target.value)}
                  className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
                />
              </div>
              <div className="w-28">
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  演化相位 t (rad)
                </label>
                <input
                  type="number"
                  step={0.1}
                  value={hamScale}
                  onChange={(e) => setHamScale(Number(e.target.value))}
                  className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
                />
              </div>
              <div className="w-24">
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">比特数</label>
                <select
                  value={synthQubits}
                  onChange={(e) => setSynthQubits(Number(e.target.value))}
                  className="w-full px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px]"
                >
                  <option value={1}>1Q</option>
                  <option value={2}>2Q</option>
                </select>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Body: Empty state or Visual Comparison + Metrics/Code Dock */}
      {!optResponse || !m1 ? (
        <div className="flex-1 flex flex-col items-center justify-center dot-grid-canvas gap-3 p-8">
          <div className="studio-card max-w-md p-6 text-center space-y-3">
            <div className="text-[15px] font-semibold text-[var(--text-primary)]">
              硬件感知最优量子门组合设计器
            </div>
            <p className="text-[12.5px] text-[var(--text-secondary)] leading-relaxed">
              根据当前实验硬件（及已勾选「作为原生门」的自定义门），利用 Euler 分解、KAK/Cartan 最优综合与 SABRE 拓扑路由，自动计算最短、最高保真度的原生门序列。
            </p>
            <button
              onClick={optMode === 'unitary' ? handleTriggerSynth : onRunOptimize}
              disabled={isBusy}
              className="mx-auto px-4 py-2 rounded-xl studio-btn-primary text-[13px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-60"
            >
              <Zap className="w-4 h-4" />
              {optMode === 'unitary' ? '综合目标酉矩阵最优原生门' : '立即计算最优量子门组合'}
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* shadcn-analytics 4-KPI Optimization Summary Strip */}
          <div className="px-3.5 py-2.5 grid grid-cols-2 lg:grid-cols-4 gap-2.5 border-b border-[var(--border-hairline)] bg-[var(--bg-app)] shrink-0">
            <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
              <div className="min-w-0">
                <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                  2Q 纠缠门数
                </div>
                <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                  {m1.num_2q_gates}
                </div>
              </div>
              <span className="px-1.5 py-0.5 rounded-md bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 text-[10px] font-mono-num font-semibold shrink-0">
                {m0 && m1.num_2q_gates < m0.num_2q_gates
                  ? `-${m0.num_2q_gates - m1.num_2q_gates}`
                  : optResponse.primary_2q_gate}
              </span>
            </div>

            <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
              <div className="min-w-0">
                <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                  物理 1Q 脉冲
                </div>
                <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                  {m1.num_1q_physical_pulses}
                </div>
              </div>
              <span className="px-1.5 py-0.5 rounded-md bg-[var(--bg-subtle)] text-[var(--text-secondary)] text-[10px] font-mono-num shrink-0">
                +{m1.num_1q_virtual_gates} VZ
              </span>
            </div>

            <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
              <div className="min-w-0">
                <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                  总脉冲时长
                </div>
                <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                  {m1.duration_ns}{' '}
                  <span className="text-[11px] font-normal text-[var(--text-secondary)]">ns</span>
                </div>
              </div>
              <span className="px-1.5 py-0.5 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] text-[10px] font-mono-num font-semibold shrink-0">
                {m0 && m0.duration_ns > m1.duration_ns
                  ? `-${(m0.duration_ns - m1.duration_ns).toFixed(0)}ns`
                  : `深度 ${m1.depth}`}
              </span>
            </div>

            <div className="studio-kpi px-3 py-2 flex items-center justify-between gap-2 min-w-0">
              <div className="min-w-0">
                <div className="text-[10px] font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                  {fOpt !== null ? '含噪保真度' : '酉保真度'}
                </div>
                <div className="text-[15px] font-semibold font-mono-num text-[var(--text-primary)] leading-tight mt-0.5">
                  {(fOpt ?? optResponse.unitary_fidelity).toFixed(4)}
                </div>
              </div>
              <span className="px-1.5 py-0.5 rounded-md bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 text-[10px] font-mono-num font-semibold shrink-0">
                {fOpt !== null && fOrig !== null
                  ? `${fOpt >= fOrig ? '+' : ''}${(fOpt - fOrig).toFixed(3)}`
                  : `U=${optResponse.unitary_fidelity.toFixed(4)}`}
              </span>
            </div>
          </div>

          {/* Middle Visual Comparison: Original vs Optimized (or Synthesized only for Direct Unitary mode) */}
          <div className="flex-1 min-h-0 flex flex-col divide-y divide-[var(--border-hairline)] overflow-hidden">
            {m0 && optMode === 'circuit' && (
              <div className="flex-1 min-h-0 flex flex-col">
                <CircuitCanvas
                  project={project}
                  readOnly
                  compact
                  selection={{ type: 'none' }}
                  activeToolGate={null}
                  onSelect={() => {}}
                  onPickGateTool={() => {}}
                  onPlaceGate={() => {}}
                  onDeleteOp={() => {}}
                  onAddQubit={() => {}}
                  titleBadge={`① 原始输入线路 · ${m0.num_2q_gates} 个 2Q 门 · ${m0.num_1q_physical_pulses} 个物理 1Q 脉冲 · ${m0.duration_ns} ns`}
                />
              </div>
            )}
            <div className="flex-1 min-h-0 flex flex-col">
              <CircuitCanvas
                project={synthPreviewProject}
                operationsOverride={optResponse.optimized_operations}
                readOnly
                compact
                selection={{ type: 'none' }}
                activeToolGate={null}
                onSelect={() => {}}
                onPickGateTool={() => {}}
                onPlaceGate={() => {}}
                onDeleteOp={() => {}}
                onAddQubit={() => {}}
                titleBadge={`${m0 && optMode === 'circuit' ? '② ' : ''}最优硬件原生组合 (2Q 基底: ${optResponse.primary_2q_gate}) · ${m1.num_2q_gates} 个 2Q 门 · ${m1.num_1q_physical_pulses} 个物理 1Q 脉冲 · ${m1.num_1q_virtual_gates} 个 Virtual-Z · ${m1.duration_ns} ns`}
              />
            </div>
          </div>

          {/* Bottom Dock: Metrics Table + Native Code Export */}
          <div className="h-[220px] shrink-0 bg-[var(--bg-panel)] border-t border-[var(--border-hairline)] grid grid-cols-12 divide-x divide-[var(--border-hairline)]">
            <div className="col-span-7 p-3.5 overflow-y-auto">
              <div className="text-[12px] font-semibold mb-2 text-[var(--text-primary)]">
                硬件原生综合关键指标对比
              </div>
              <table className="w-full text-[12px] border-collapse">
                <thead>
                  <tr className="border-b border-[var(--border-hairline)] text-left text-[11px] text-[var(--text-secondary)]">
                    <th className="py-1.5 font-medium">评估维度</th>
                    <th className="py-1.5 font-medium">原始线路</th>
                    <th className="py-1.5 font-medium">最优原生门</th>
                    <th className="py-1.5 font-medium">优化变化</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-hairline)] font-mono-num">
                  <tr>
                    <td className="py-1.5 font-sans">两比特纠缠门 (2Q)</td>
                    <td className="py-1.5">{m0?.num_2q_gates ?? '—'}</td>
                    <td className="py-1.5 font-semibold text-[#0A84FF]">{m1.num_2q_gates}</td>
                    <td className="py-1.5 text-emerald-600 font-sans">
                      {m0
                        ? m1.num_2q_gates < m0.num_2q_gates
                          ? `减少 ${m0.num_2q_gates - m1.num_2q_gates} 个`
                          : '已达 KAK 下界'
                        : 'KAK 最优综合'}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1.5 font-sans">单比特物理脉冲</td>
                    <td className="py-1.5">{m0?.num_1q_physical_pulses ?? '—'}</td>
                    <td className="py-1.5 font-semibold">{m1.num_1q_physical_pulses}</td>
                    <td className="py-1.5 text-[var(--text-secondary)] font-sans">
                      含 {m1.num_1q_virtual_gates} 个零耗时 Virtual-Z
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1.5 font-sans">总脉冲时长</td>
                    <td className="py-1.5">{m0 ? `${m0.duration_ns} ns` : '—'}</td>
                    <td className="py-1.5 font-semibold">{m1.duration_ns} ns</td>
                    <td className="py-1.5 text-emerald-600 font-sans">
                      {m0 && m0.duration_ns > m1.duration_ns
                        ? `缩短 ${(m0.duration_ns - m1.duration_ns).toFixed(1)} ns`
                        : '—'}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1.5 font-sans">线路深度 / 总门数</td>
                    <td className="py-1.5">{m0 ? `${m0.depth} / ${m0.total_gates}` : '—'}</td>
                    <td className="py-1.5">
                      {m1.depth} / {m1.total_gates}
                    </td>
                    <td className="py-1.5 text-[var(--text-secondary)] text-[11px] truncate max-w-[160px]">
                      {Object.entries(m1.gate_breakdown)
                        .map(([k, v]) => `${k}:${v}`)
                        .join(' ')}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="col-span-5 p-3.5 flex flex-col min-h-0">
              <div className="flex items-center justify-between mb-1.5 shrink-0">
                <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-md inline-flex">
                  <button
                    onClick={() => setCodeTab('dsl')}
                    className={`px-2 py-0.5 rounded text-[11px] cursor-pointer ${
                      codeTab === 'dsl'
                        ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs'
                        : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    最优门 DSL
                  </button>
                  <button
                    onClick={() => setCodeTab('qasm')}
                    className={`px-2 py-0.5 rounded text-[11px] cursor-pointer ${
                      codeTab === 'qasm'
                        ? 'bg-[var(--bg-panel)] font-semibold shadow-2xs'
                        : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    OpenQASM 3
                  </button>
                </div>
                <button
                  onClick={() =>
                    handleCopy(codeTab === 'dsl' ? optResponse.optimized_dsl : optResponse.optimized_qasm)
                  }
                  className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 text-[var(--text-secondary)] cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
              <pre className="flex-1 overflow-auto p-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11.5px] font-mono-num leading-relaxed">
                {codeTab === 'dsl' ? optResponse.optimized_dsl : optResponse.optimized_qasm}
              </pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
