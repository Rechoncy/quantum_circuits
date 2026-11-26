import React from 'react';
import {
  Trash2,
  Copy,
  ArrowLeft,
  ArrowRight,
  ArrowUpDown,
  RotateCcw,
  Pencil,
  Zap,
  Play,
  Square,
} from 'lucide-react';
import type {
  BenchmarkProtocol,
  CircuitOpModel,
  CustomGateDef,
  HardwareProfileModel,
  InitStateType,
  ProjectModel,
  QubitSpec,
  SelectionTarget,
  SimulationResponse,
  SolverBackendType,
  WorkMode,
} from '../types';
import { BUILTIN_GATES, QUBIT_DOT_PALETTE } from '../constants';
import { TopologyEditor } from './TopologyEditor';
import {
  impliedRabiMhz,
  isVirtual1QGate,
  rabiPulseDurationNs,
  singleQubitRotationAngle,
} from '../utils/drivePhysics';

export interface SweepConfigState {
  mode: '1d' | '2d' | 'vqe';
  param_x: string;
  x_start: number;
  x_stop: number;
  x_points: number;
  param_y: string;
  y_start: number;
  y_stop: number;
  y_points: number;
  target_bitstring: string;
  observed_qubit: number;
  hamiltonian_str: string;
  vqe_method: 'Parameter-Shift' | 'COBYLA' | 'Powell' | 'Nelder-Mead' | 'L-BFGS-B';
  vqe_max_iter: number;
  backend: 'qutip' | 'kraus' | 'trajectory' | 'stim' | 'stim_qutip';
}

export interface BenchmarkConfigState {
  protocol: BenchmarkProtocol;
  target_qubit: number;
  second_qubit: number;
  rb_num_qubits: 1 | 2;
  rb_lengths_str: string;
  rb_num_seeds: number;
  interleaved_gate: string;
  xeb_entangler: string;
  gst_max_length: number;
  cal_points: number;
  cal_max_ns: number | null;
  ramsey_detuning_mhz: number;
  shots: number;
  amp_error_pct: number;
  detuning_mhz: number;
  phase_skew_deg: number;
}

export function buildPresetHamiltonian(
  kind: 'tfim' | 'heisenberg' | 'maxcut',
  project: ProjectModel
): string {
  const n = Math.max(1, project.qubits.length);
  if (n === 1) {
    if (kind === 'tfim') return '-1.0*Z0 - 0.5*X0';
    if (kind === 'heisenberg') return '0.5*X0 + 0.5*Z0';
    return '0.5*I - 0.5*Z0';
  }
  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];
  for (const op of project.operations) {
    if (op.kind === 'gate' && op.qubits.length >= 2) {
      const u = Math.min(op.qubits[0], op.qubits[1]);
      const v = Math.max(op.qubits[0], op.qubits[1]);
      const key = `${u}_${v}`;
      if (u !== v && u >= 0 && v < n && !edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push([u, v]);
      }
    }
  }
  if (edges.length === 0) {
    for (let i = 0; i < Math.min(n - 1, 32); i++) edges.push([i, i + 1]);
  }
  const cappedEdges = edges.slice(0, 32);
  if (kind === 'tfim') {
    const zz = cappedEdges.map(([u, v]) => `-1.0*Z${u}*Z${v}`).join(' ');
    const hx = Array.from({ length: Math.min(n, 24) }, (_, q) => `- 0.5*X${q}`).join(' ');
    return `${zz} ${hx}`.trim();
  }
  if (kind === 'heisenberg') {
    return cappedEdges
      .slice(0, 16)
      .map(([u, v], idx) =>
        `${idx > 0 ? '+ ' : ''}0.25*X${u}*X${v} + 0.25*Y${u}*Y${v} + 0.25*Z${u}*Z${v}`
      )
      .join(' ');
  }
  const constOffset = (0.5 * cappedEdges.length).toFixed(1);
  const zzCut = cappedEdges.map(([u, v]) => `- 0.5*Z${u}*Z${v}`).join(' ');
  return `${constOffset}*I ${zzCut}`.trim();
}

function defaultSweepRange(paramKey: string): [number, number] {
  if (paramKey === 'hw:t1_us' || paramKey === 'hw:t2_us') return [5, 200];
  if (paramKey === 'hw:rabi_mhz') return [0.5, 25];
  if (paramKey === 'hw:rise_ns' || paramKey === 'hw:fall_ns') return [0, 80];
  if (paramKey.includes('depol') || paramKey.includes('readout') || paramKey.startsWith('p_')) {
    return [0, 0.08];
  }
  if (paramKey === 'hw:zz_khz') return [0, 500];
  if (paramKey === 'hw:t_1q_ns' || paramKey === 'hw:t_2q_ns') return [10, 400];
  return [0, 6.283];
}

interface RightInspectorProps {
  mode: WorkMode;
  project: ProjectModel;
  selection: SelectionTarget;
  simResponse: SimulationResponse | null;
  observedQubit: number;
  onObservedQubitChange: (q: number) => void;
  onUpdateOp: (opId: string, patch: Partial<CircuitOpModel>) => void;
  onDeleteOp: (opId: string) => void;
  onDuplicateOp: (opId: string) => void;
  onUpdateQubit: (qubitId: number, patch: Partial<QubitSpec>) => void;
  onApplyLatticeLayout?: (positions: { x: number; y: number }[]) => void;
  onUpdateHardware: (hwId: string, patch: Partial<HardwareProfileModel>) => void;
  onToggleCustomEdge: (u: number, v: number) => void;
  onUpdateSimSettings: (patch: Partial<ProjectModel['sim_settings']>) => void;
  onToggleCustomGateNative: (gateId: string, isNative: boolean) => void;
  onEditCustomGate: (gate: CustomGateDef) => void;
  onEnsureVariable: (name: string, val: number) => void;
  // Optimize controls
  preferredNative2Q: string;
  onPreferredNative2QChange: (name: string) => void;
  evalNoisyOpt: boolean;
  onEvalNoisyOptChange: (v: boolean) => void;
  onRunOptimize: () => void;
  // Sweep controls
  sweepConfig: SweepConfigState;
  onSweepConfigChange: (patch: Partial<SweepConfigState>) => void;
  hardwareSweepParams: Record<string, string>;
  onRunSweep: () => void;
  onCancelSweep?: () => void;
  onRunZne?: () => void;
  onRunNoiseBudget?: () => void;
  // Gate Benchmarking controls
  benchmarkConfig?: BenchmarkConfigState;
  onBenchmarkConfigChange?: (patch: Partial<BenchmarkConfigState>) => void;
  onRunGateBenchmark?: () => void;
  onOpenDeviceWorkspace?: () => void;
  isBusy: boolean;
}

const IOSSwitch: React.FC<{ checked: boolean; onChange: (v: boolean) => void }> = ({
  checked,
  onChange,
}) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={() => onChange(!checked)}
    className={`w-9 h-[18px] rounded-[2px] border p-0.5 transition-colors cursor-pointer shrink-0 flex items-center ${
      checked
        ? 'bg-[var(--accent)] border-[var(--accent)]'
        : 'bg-[var(--bg-subtle)] border-[var(--border-input)]'
    }`}
  >
    <div
      className={`w-3.5 h-3 rounded-[1px] bg-white transition-transform ${
        checked ? 'translate-x-4' : 'translate-x-0'
      }`}
    />
  </button>
);

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-[10.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-secondary)] mb-1.5 flex items-center gap-1.5">
    <span className="w-1 h-2.5 bg-[var(--accent)] rounded-[1px]" />
    <span>{children}</span>
  </div>
);

export const RightInspector: React.FC<RightInspectorProps> = ({
  mode,
  project,
  selection,
  simResponse,
  observedQubit,
  onObservedQubitChange,
  onUpdateOp,
  onDeleteOp,
  onDuplicateOp,
  onUpdateQubit,
  onApplyLatticeLayout,
  onUpdateHardware,
  onToggleCustomEdge,
  onUpdateSimSettings,
  onToggleCustomGateNative,
  onEditCustomGate,
  onEnsureVariable,
  preferredNative2Q,
  onPreferredNative2QChange,
  evalNoisyOpt,
  onEvalNoisyOptChange,
  onRunOptimize,
  sweepConfig,
  onSweepConfigChange,
  hardwareSweepParams,
  onRunSweep,
  onCancelSweep,
  onRunZne,
  onRunNoiseBudget,
  benchmarkConfig,
  onBenchmarkConfigChange,
  onRunGateBenchmark,
  onOpenDeviceWorkspace,
  isBusy,
}) => {
  const activeHw =
    project.hardware_profiles.find((h) => h.id === project.active_hardware_id) ||
    project.hardware_profiles[0];

  // =========================================================================
  // 1. SIMULATE MODE INSPECTOR (Matches Mockup 2)
  // =========================================================================
  if (mode === 'simulate') {
    const primaryRes = simResponse ? simResponse.results[simResponse.primary_backend] : null;
    const sim = project.sim_settings;

    return (
      <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-5">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">模拟设置</div>

        {/* Solver Segmented Control */}
        <div>
          <SectionTitle>求解器</SectionTitle>
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg grid grid-cols-4 gap-0.5">
            {(['qutip', 'kraus', 'trajectory', 'stim'] as SolverBackendType[]).map((b) => (
              <button
                key={b}
                onClick={() => onUpdateSimSettings({ backend: b })}
                title={
                  b === 'qutip'
                    ? 'QuTiP 连续脉冲主方程 (≤9Q)'
                    : b === 'kraus'
                    ? 'Kraus 张量密度矩阵 (≤14Q)'
                    : b === 'trajectory'
                    ? 'Trajectory 纯态矢量蒙特卡洛轨迹 (≤24Q)'
                    : 'Stim 稳定子与泡利帧采样 (500+Q)'
                }
                className={`py-1.5 rounded-[3px] text-[11.5px] font-medium transition-all cursor-pointer ${
                  sim.backend === b
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {b === 'qutip' ? 'QuTiP' : b === 'kraus' ? 'Kraus' : b === 'trajectory' ? 'Traj' : 'Stim'}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-1.5 mt-1.5">
            {(
              [
                ['stim_qutip', 'Stim+QuTiP 桥接'],
                ['compare', '多后端同屏对比'],
              ] as const
            ).map(([b, lbl]) => (
              <button
                key={b}
                onClick={() => onUpdateSimSettings({ backend: b })}
                className={`py-1 rounded-md text-[11px] border transition-colors cursor-pointer ${
                  sim.backend === b
                    ? 'bg-[#0A84FF]/12 border-[#0A84FF] text-[#0A84FF] font-semibold'
                    : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)] text-[var(--text-secondary)]'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>

          <div className="mt-3 flex items-center justify-between">
            <span className="text-[12.5px] text-[var(--text-primary)]">Shots</span>
            <input
              type="number"
              min={100}
              max={1000000}
              step={1000}
              value={sim.shots}
              onChange={(e) => onUpdateSimSettings({ shots: Math.max(100, Number(e.target.value)) })}
              className="w-28 px-2.5 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-right font-mono-num text-[12.5px] outline-none focus:border-[#0A84FF]"
            />
          </div>
        </div>

        {/* Noise Model iOS Switches (settings-page inset group) */}
        <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2">
          <SectionTitle>噪声模型</SectionTitle>
          <div className="studio-inset-group p-3 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">T1 / T2 退相干</span>
              <IOSSwitch
                checked={sim.enable_t1_t2}
                onChange={(v) => onUpdateSimSettings({ enable_t1_t2: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">退极化控制误差</span>
              <IOSSwitch
                checked={sim.enable_depol}
                onChange={(v) => onUpdateSimSettings({ enable_depol: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">静态 ZZ 串扰</span>
              <IOSSwitch
                checked={sim.enable_zz}
                onChange={(v) => onUpdateSimSettings({ enable_zz: v })}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">读出误差</span>
              <IOSSwitch
                checked={sim.enable_readout}
                onChange={(v) => onUpdateSimSettings({ enable_readout: v })}
              />
            </div>
          </div>
        </div>

        {/* Observation & Summary Card */}
        <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
          <SectionTitle>观测与汇总</SectionTitle>
          <div className="flex items-center justify-between gap-2">
            <span className="text-[12.5px] shrink-0">观测比特</span>
            <select
              value={observedQubit}
              onChange={(e) => onObservedQubitChange(Number(e.target.value))}
              className="flex-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12.5px] font-mono-num outline-none focus:border-[var(--accent)]"
            >
              {project.qubits.map((q, idx) => (
                <option key={q.id} value={idx}>
                  {q.name}
                </option>
              ))}
            </select>
          </div>

          <div className="studio-inset-group p-3 space-y-2">
            <div className="flex items-center justify-between text-[12.5px]">
              <span className="text-[var(--text-secondary)]">最终保真度</span>
              <span className="font-mono-num font-semibold text-[var(--accent)]">
                {primaryRes?.final_fidelity !== null && primaryRes?.final_fidelity !== undefined
                  ? primaryRes.final_fidelity.toFixed(4)
                  : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between text-[12.5px]">
              <span className="text-[var(--text-secondary)]">纯度 Tr(ρ²)</span>
              <span className="font-mono-num font-semibold text-[var(--text-primary)]">
                {primaryRes?.final_purity !== null && primaryRes?.final_purity !== undefined
                  ? primaryRes.final_purity.toFixed(4)
                  : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between text-[12.5px]">
              <span className="text-[var(--text-secondary)]">总脉冲时长</span>
              <span className="font-mono-num font-semibold text-[var(--text-primary)]">
                {primaryRes ? `${primaryRes.total_duration_ns.toFixed(1)} ns` : '—'}
              </span>
            </div>
          </div>

          {(onRunZne || onRunNoiseBudget) && (
            <div className="pt-2 space-y-2">
              {onRunZne && (
                <button
                  type="button"
                  onClick={onRunZne}
                  disabled={isBusy}
                  className="w-full py-1.5 px-3 rounded-[3px] border border-[#0A84FF]/40 bg-[#0A84FF]/10 hover:bg-[#0A84FF]/18 text-[#0A84FF] text-[11.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-55 transition-colors"
                >
                  <Zap className="w-3.5 h-3.5" />
                  执行 ZNE 零噪声外推 (λ=1,2,3)
                </button>
              )}
              {onRunNoiseBudget && (
                <button
                  type="button"
                  onClick={onRunNoiseBudget}
                  disabled={isBusy}
                  className="w-full py-1.5 px-3 rounded-[3px] border border-amber-500/40 bg-amber-500/10 hover:bg-amber-500/18 text-amber-500 text-[11.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-55 transition-colors"
                >
                  <Zap className="w-3.5 h-3.5" />
                  噪声误差预算消融拆解 (Budget)
                </button>
              )}
              <p className="text-[10.5px] text-[var(--text-secondary)] leading-snug">
                支持 Richardson 零噪声外推缓解，或逐通道消融定位 T1 / T2* / 门退极化 / ZZ / 读出瓶颈。
              </p>
            </div>
          )}
        </div>
      </aside>
    );
  }

  // =========================================================================
  // 2. OPTIMIZE MODE INSPECTOR
  // =========================================================================
  if (mode === 'optimize') {
    const custom2Q = project.custom_gates.filter((g) => g.num_qubits === 2);
    const native2QChoices = [
      { id: '', label: `平台默认 (${activeHw.name})` },
      { id: 'CZ', label: 'CZ (超导 / 中性原子里德堡)' },
      { id: 'CNOT', label: 'CNOT (交叉共振)' },
      { id: 'ISWAP', label: 'iSWAP (电容耦合)' },
      { id: 'MS', label: 'MS (离子阱 XX)' },
      ...custom2Q.map((cg) => ({ id: cg.name, label: `${cg.display_label || cg.name} (自定义原生门)` })),
    ];

    return (
      <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-5">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">最优门综合配置</div>

        <div className="space-y-2.5">
          <SectionTitle>目标实验原生门集</SectionTitle>
          <div>
            <label className="block text-[11.5px] text-[var(--text-secondary)] mb-1">
              优先两比特纠缠基底
            </label>
            <select
              value={preferredNative2Q}
              onChange={(e) => onPreferredNative2QChange(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none focus:border-[#0A84FF]"
            >
              {native2QChoices.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-between pt-1">
            <span className="text-[12.5px]">用 QuTiP 验证含噪保真度</span>
            <IOSSwitch checked={evalNoisyOpt} onChange={onEvalNoisyOptChange} />
          </div>
        </div>

        <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2">
          <SectionTitle>自定义原生门开关</SectionTitle>
          <p className="text-[11px] text-[var(--text-secondary)]">
            勾选后，该自定义门会加入实验可用操作集，优化器可直接以它为基底综合最短门组合。
          </p>
          {project.custom_gates.length === 0 ? (
            <div className="text-[12px] text-[var(--text-secondary)]">暂无自定义门</div>
          ) : (
            <div className="space-y-1.5">
              {project.custom_gates.map((cg) => (
                <div
                  key={cg.id}
                  className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)]"
                >
                  <div className="min-w-0">
                    <div className="text-[12px] font-mono-num font-semibold truncate">
                      {cg.display_label || cg.name}
                    </div>
                    <div className="text-[10.5px] text-[var(--text-secondary)]">
                      {cg.num_qubits}Q · {cg.duration_ns} ns
                    </div>
                  </div>
                  <IOSSwitch
                    checked={cg.is_native}
                    onChange={(v) => onToggleCustomGateNative(cg.id, v)}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          onClick={onRunOptimize}
          disabled={isBusy}
          className="w-full py-2 rounded-xl studio-btn-primary text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-60"
        >
          <Zap className="w-3.5 h-3.5" />
          计算最优量子门组合
        </button>
      </aside>
    );
  }

  // =========================================================================
  // 3. SWEEP MODE INSPECTOR
  // =========================================================================
  if (mode === 'sweep') {
    const symKeys = Object.keys(project.variables);
    const paramOptions = [
      ...symKeys.map((k) => ({ key: k, label: `符号参数: ${k}` })),
      ...Object.entries(hardwareSweepParams).map(([k, lbl]) => ({ key: k, label: lbl })),
    ];

    return (
      <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-5">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">
          参数扫描与变分寻优
        </div>

        <div className="space-y-3">
          <SectionTitle>任务模式与后端</SectionTitle>
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg grid grid-cols-3 gap-0.5">
            {(
              [
                ['1d', '1D 曲线'],
                ['2d', '2D 相图'],
                ['vqe', 'VQE 寻优'],
              ] as const
            ).map(([m, lbl]) => (
              <button
                key={m}
                onClick={() => {
                  if (m === 'vqe' && !sweepConfig.hamiltonian_str.trim()) {
                    onSweepConfigChange({
                      mode: m,
                      hamiltonian_str: buildPresetHamiltonian('tfim', project),
                    });
                  } else {
                    onSweepConfigChange({ mode: m });
                  }
                }}
                className={`py-1.5 rounded-[3px] text-[11.5px] font-medium cursor-pointer ${
                  sweepConfig.mode === m
                    ? 'bg-[var(--bg-panel)] text-[#0A84FF] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)]'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>

          <div>
            <label className="block text-[11px] text-[var(--text-secondary)] mb-1">求解引擎</label>
            <select
              value={sweepConfig.backend}
              onChange={(e) =>
                onSweepConfigChange({ backend: e.target.value as SweepConfigState['backend'] })
              }
              className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
            >
              <option value="qutip">QuTiP · 物理主方程 (≤9Q)</option>
              <option value="kraus">Kraus · 张量密度矩阵 (≤14Q)</option>
              <option value="trajectory">Trajectory · 态矢量轨迹 (≤24Q)</option>
              <option value="stim">Stim · 稳定子高速采样 (500+Q)</option>
              <option value="stim_qutip">Stim + QuTiP 标定</option>
            </select>
          </div>

          {sweepConfig.mode !== 'vqe' && (
            <div>
              <label className="block text-[11px] text-[var(--text-secondary)] mb-1">
                观测比特 (Bloch / ⟨Z⟩)
              </label>
              <select
                value={Math.min(sweepConfig.observed_qubit, Math.max(0, project.qubits.length - 1))}
                onChange={(e) => onSweepConfigChange({ observed_qubit: Number(e.target.value) })}
                className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
              >
                {project.qubits.map((q, idx) => (
                  <option key={q.id} value={idx}>
                    {q.name} (index {idx})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {sweepConfig.mode === 'vqe' ? (
          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-3">
            <SectionTitle>闭环变分优化器 (SciPy)</SectionTitle>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  优化算法
                </label>
                <select
                  value={sweepConfig.vqe_method}
                  onChange={(e) =>
                    onSweepConfigChange({
                      vqe_method: e.target.value as SweepConfigState['vqe_method'],
                    })
                  }
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                >
                  <option value="Parameter-Shift">Parameter-Shift (解析梯度)</option>
                  <option value="COBYLA">COBYLA (信赖域)</option>
                  <option value="Powell">Powell (方向搜索)</option>
                  <option value="Nelder-Mead">Nelder-Mead</option>
                  <option value="L-BFGS-B">L-BFGS-B (准牛顿)</option>
                </select>
              </div>
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  最大迭代步数
                </label>
                <input
                  type="number"
                  min={10}
                  max={200}
                  step={10}
                  value={sweepConfig.vqe_max_iter}
                  onChange={(e) => onSweepConfigChange({ vqe_max_iter: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
            </div>
            <div className="p-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11px] text-[var(--text-secondary)] leading-relaxed">
              待寻优符号参数 ({symKeys.length}):{' '}
              <span className="font-mono-num text-[#0A84FF] font-semibold">
                {symKeys.length > 0 ? symKeys.join(', ') : '无（请先添加符号变量）'}
              </span>
            </div>
          </div>
        ) : (
          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2.5">
            <SectionTitle>主扫描参数 X</SectionTitle>
            <select
              value={sweepConfig.param_x}
              onChange={(e) => {
                const k = e.target.value;
                const [s0, s1] = defaultSweepRange(k);
                onSweepConfigChange({ param_x: k, x_start: s0, x_stop: s1 });
              }}
              className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
            >
              {paramOptions.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">起始</label>
                <input
                  type="number"
                  step={0.1}
                  value={sweepConfig.x_start}
                  onChange={(e) => onSweepConfigChange({ x_start: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">终止</label>
                <input
                  type="number"
                  step={0.1}
                  value={sweepConfig.x_stop}
                  onChange={(e) => onSweepConfigChange({ x_stop: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">点数</label>
                <input
                  type="number"
                  min={3}
                  max={120}
                  value={sweepConfig.x_points}
                  onChange={(e) => onSweepConfigChange({ x_points: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
            </div>
          </div>
        )}

        {sweepConfig.mode === '2d' && (
          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2.5">
            <SectionTitle>第二扫描参数 Y (2D)</SectionTitle>
            <select
              value={sweepConfig.param_y}
              onChange={(e) => {
                const k = e.target.value;
                const [s0, s1] = defaultSweepRange(k);
                onSweepConfigChange({ param_y: k, y_start: s0, y_stop: s1 });
              }}
              className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
            >
              {paramOptions.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </select>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">起始</label>
                <input
                  type="number"
                  step={0.01}
                  value={sweepConfig.y_start}
                  onChange={(e) => onSweepConfigChange({ y_start: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">终止</label>
                <input
                  type="number"
                  step={0.01}
                  value={sweepConfig.y_stop}
                  onChange={(e) => onSweepConfigChange({ y_stop: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">点数</label>
                <input
                  type="number"
                  min={3}
                  max={30}
                  value={sweepConfig.y_points}
                  onChange={(e) => onSweepConfigChange({ y_points: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                目标计算基态 |x⟩ ({project.qubits.length} 位二进制)
              </label>
              <input
                value={sweepConfig.target_bitstring}
                placeholder={'0'.repeat(Math.max(1, project.qubits.length))}
                onChange={(e) =>
                  onSweepConfigChange({
                    target_bitstring: e.target.value.replace(/[^01]/g, '').slice(0, project.qubits.length),
                  })
                }
                className="w-full px-2.5 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
            </div>
          </div>
        )}

        {/* Variational Pauli Hamiltonian <H> Cost Observer (VQE / QAOA) */}
        <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2">
          <div className="flex items-center justify-between">
            <SectionTitle>变分能量哈密顿量 ⟨H⟩ (可选)</SectionTitle>
            {sweepConfig.hamiltonian_str.trim() && (
              <button
                type="button"
                onClick={() => onSweepConfigChange({ hamiltonian_str: '' })}
                className="text-[10.5px] text-[var(--text-secondary)] hover:text-red-500 mb-2 cursor-pointer"
              >
                清空
              </button>
            )}
          </div>
          <textarea
            rows={2}
            value={sweepConfig.hamiltonian_str}
            onChange={(e) => onSweepConfigChange({ hamiltonian_str: e.target.value })}
            placeholder="如 -1.0*Z0*Z1 - 0.5*X0 - 0.5*X1（支持任意比特下标或稠密泡利串）"
            spellCheck={false}
            className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] font-mono-num leading-snug outline-none focus:border-[#0A84FF] resize-y"
          />
          <div className="flex flex-wrap items-center gap-1">
            {(
              [
                ['tfim', '伊辛 TFIM'],
                ['heisenberg', '海森堡 XXX'],
                ['maxcut', 'MaxCut 代价'],
              ] as const
            ).map(([k, label]) => (
              <button
                key={k}
                type="button"
                onClick={() =>
                  onSweepConfigChange({ hamiltonian_str: buildPresetHamiltonian(k, project) })
                }
                title="根据当前线路耦合拓扑一键生成泡利哈密顿量"
                className="px-2 py-0.5 rounded-md bg-[#0A84FF]/10 hover:bg-[#0A84FF]/18 text-[#0A84FF] text-[10.5px] font-medium cursor-pointer transition-colors"
              >
                + {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={onRunSweep}
            disabled={isBusy}
            className="flex-1 py-2 rounded-xl studio-btn-primary text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-60"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {sweepConfig.mode === 'vqe' ? '启动变分基态寻优' : '执行参数扫描'}
          </button>
          {isBusy && onCancelSweep && (
            <button
              onClick={onCancelSweep}
              title="中止当前扫描任务"
              className="px-2.5 py-2 rounded-lg bg-red-500/12 hover:bg-red-500/20 text-red-600 dark:text-red-400 text-[12px] font-semibold flex items-center gap-1 cursor-pointer"
            >
              <Square className="w-3 h-3 fill-current" />
              取消
            </button>
          )}
        </div>
      </aside>
    );
  }

  // =========================================================================
  // 3.5 GATE BENCHMARKING & TOMOGRAPHY INSPECTOR (RB / AllXY / GST)
  // =========================================================================
  if (mode === 'benchmark' && benchmarkConfig && onBenchmarkConfigChange && onRunGateBenchmark) {
    const bc = benchmarkConfig;
    const gateChoices =
      bc.rb_num_qubits === 2
        ? [
            ['', '无 (仅标准 2Q RB)'],
            ['CZ', 'CZ (受控相位门)'],
            ['CNOT', 'CNOT (受控非门)'],
            ['ISWAP', 'iSWAP (交换门)'],
          ]
        : [
            ['', '无 (仅标准 1Q RB)'],
            ['SX', 'SX (√X π/2 脉冲)'],
            ['X', 'X (π 翻转脉冲)'],
            ['Y', 'Y (π 翻转脉冲)'],
            ['H', 'H (Hadamard 门)'],
            ['S', 'S (Virtual-Z π/2)'],
          ];

    return (
      <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-4">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">
          量子门标定与层析配置
        </div>

        {/* Protocol Switcher: Basic Calibrations + Gate Benchmarks */}
        <div className="space-y-2">
          <SectionTitle>基础物理标定实验</SectionTitle>
          <div className="bg-[var(--bg-subtle)] border border-[var(--border-hairline)] p-0.5 rounded-[3px] grid grid-cols-3 gap-0.5">
            {(
              [
                ['rabi', 'Rabi 振荡'],
                ['ramsey', 'Ramsey'],
                ['t1', 'T1 弛豫'],
              ] as const
            ).map(([pid, lbl]) => (
              <button
                key={pid}
                onClick={() => onBenchmarkConfigChange({ protocol: pid })}
                className={`py-1.5 rounded-[2px] text-[11px] font-mono-num font-medium cursor-pointer ${
                  bc.protocol === pid
                    ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold border border-[var(--border-hairline)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>

          <SectionTitle>门保真度与层析基准</SectionTitle>
          <div className="bg-[var(--bg-subtle)] border border-[var(--border-hairline)] p-0.5 rounded-[3px] grid grid-cols-4 gap-0.5">
            {(
              [
                ['rb', 'RB'],
                ['allxy', 'AllXY'],
                ['gst', 'GST'],
                ['xeb', 'XEB'],
              ] as const
            ).map(([pid, lbl]) => (
              <button
                key={pid}
                onClick={() => onBenchmarkConfigChange({ protocol: pid })}
                className={`py-1.5 rounded-[2px] text-[11px] font-mono-num font-medium cursor-pointer ${
                  bc.protocol === pid
                    ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold border border-[var(--border-hairline)]'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>
        </div>

        {/* Target Qubit(s) & Sampling */}
        <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
          <SectionTitle>标定目标与采样</SectionTitle>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                目标比特 q_a
              </label>
              <select
                value={bc.target_qubit}
                onChange={(e) => onBenchmarkConfigChange({ target_qubit: Number(e.target.value) })}
                className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
              >
                {project.qubits.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                测量采样 Shots
              </label>
              <select
                value={bc.shots}
                onChange={(e) => onBenchmarkConfigChange({ shots: Number(e.target.value) })}
                className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
              >
                <option value={0}>∞ 精确密度矩阵</option>
                <option value={1024}>1,024</option>
                <option value={4096}>4,096</option>
                <option value={16384}>16,384</option>
                <option value={65536}>65,536</option>
              </select>
            </div>
          </div>

          {(bc.protocol === 'rabi' || bc.protocol === 'ramsey' || bc.protocol === 't1') && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    扫描采样点数
                  </label>
                  <input
                    type="number"
                    min={15}
                    max={251}
                    step={10}
                    value={bc.cal_points}
                    onChange={(e) =>
                      onBenchmarkConfigChange({ cal_points: Math.max(15, Number(e.target.value)) })
                    }
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                  />
                </div>
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    最大扫描时长 (ns)
                  </label>
                  <input
                    type="number"
                    min={10}
                    placeholder="自动推算"
                    value={bc.cal_max_ns ?? ''}
                    onChange={(e) =>
                      onBenchmarkConfigChange({
                        cal_max_ns: e.target.value === '' ? null : Math.max(5, Number(e.target.value)),
                      })
                    }
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                  />
                </div>
              </div>
              {bc.protocol === 'ramsey' && (
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    Ramsey 人工注入条纹频率 δ_art (MHz)
                  </label>
                  <input
                    type="number"
                    step={0.1}
                    value={bc.ramsey_detuning_mhz}
                    onChange={(e) =>
                      onBenchmarkConfigChange({ ramsey_detuning_mhz: Number(e.target.value) })
                    }
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                  />
                </div>
              )}
            </>
          )}

          {bc.protocol === 'rb' && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    Clifford 群维度
                  </label>
                  <select
                    value={bc.rb_num_qubits}
                    disabled={project.qubits.length < 2}
                    onChange={(e) => {
                      const nq = Number(e.target.value) as 1 | 2;
                      onBenchmarkConfigChange({
                        rb_num_qubits: nq,
                        interleaved_gate: nq === 2 ? 'CZ' : 'SX',
                        rb_lengths_str: nq === 2 ? '1, 2, 4, 8, 16, 32' : '1, 2, 4, 8, 16, 32, 64, 128',
                      });
                    }}
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none"
                  >
                    <option value={1}>1Q 单比特 (C₁)</option>
                    <option value={2}>2Q 双比特 (C₂)</option>
                  </select>
                </div>

                {bc.rb_num_qubits === 2 ? (
                  <div>
                    <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                      耦合配对比特 q_b
                    </label>
                    <select
                      value={bc.second_qubit}
                      onChange={(e) =>
                        onBenchmarkConfigChange({ second_qubit: Number(e.target.value) })
                      }
                      className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                    >
                      {project.qubits
                        .filter((q) => q.id !== bc.target_qubit)
                        .map((q) => (
                          <option key={q.id} value={q.id}>
                            {q.name}
                          </option>
                        ))}
                    </select>
                  </div>
                ) : (
                  <div>
                    <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                      随机种子数 K
                    </label>
                    <input
                      type="number"
                      min={4}
                      max={50}
                      value={bc.rb_num_seeds}
                      onChange={(e) =>
                        onBenchmarkConfigChange({ rb_num_seeds: Number(e.target.value) })
                      }
                      className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                    />
                  </div>
                )}
              </div>

              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  交织标定门 (Interleaved RB)
                </label>
                <select
                  value={bc.interleaved_gate}
                  onChange={(e) => onBenchmarkConfigChange({ interleaved_gate: e.target.value })}
                  className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                >
                  {gateChoices.map(([val, lbl]) => (
                    <option key={val} value={val}>
                      {lbl}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  序列深度列表 m
                </label>
                <input
                  value={bc.rb_lengths_str}
                  onChange={(e) => onBenchmarkConfigChange({ rb_lengths_str: e.target.value })}
                  className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none"
                />
              </div>
            </>
          )}

          {bc.protocol === 'gst' && (
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                最大胚序列放大倍数 L_max (Germ Power)
              </label>
              <select
                value={bc.gst_max_length}
                onChange={(e) =>
                  onBenchmarkConfigChange({ gst_max_length: Number(e.target.value) })
                }
                className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
              >
                <option value={1}>L = 1 (标准线性反演 LGST)</option>
                <option value={4}>L = 4 (长序列相干放大)</option>
                <option value={8}>L = 8 (高精度过旋转探测)</option>
              </select>
            </div>
          )}

          {bc.protocol === 'xeb' && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    耦合配对比特 q_b
                  </label>
                  <select
                    value={bc.second_qubit}
                    disabled={project.qubits.length < 2}
                    onChange={(e) =>
                      onBenchmarkConfigChange({ second_qubit: Number(e.target.value) })
                    }
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                  >
                    {project.qubits
                      .filter((q) => q.id !== bc.target_qubit)
                      .map((q) => (
                        <option key={q.id} value={q.id}>
                          {q.name}
                        </option>
                      ))}
                  </select>
                </div>
                <div>
                  <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                    随机电路数 K
                  </label>
                  <input
                    type="number"
                    min={4}
                    max={60}
                    value={bc.rb_num_seeds}
                    onChange={(e) =>
                      onBenchmarkConfigChange({ rb_num_seeds: Number(e.target.value) })
                    }
                    className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  周期双比特纠缠门 (Cycle Entangler)
                </label>
                <select
                  value={bc.xeb_entangler}
                  onChange={(e) => onBenchmarkConfigChange({ xeb_entangler: e.target.value })}
                  className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none"
                >
                  <option value="SQISWAP">√iSWAP (Sycamore fSim 纠缠门)</option>
                  <option value="ISWAP">iSWAP (电容交换门)</option>
                  <option value="CZ">CZ (受控相位门)</option>
                  <option value="CNOT">CNOT (受控非门)</option>
                  {project.custom_gates
                    .filter((cg) => cg.num_qubits === 2 && cg.name.toUpperCase() !== 'SQISWAP')
                    .map((cg) => (
                      <option key={cg.id} value={cg.name}>
                        {cg.display_label || cg.name} (自定义 2Q 门)
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                  XEB 周期深度列表 m
                </label>
                <input
                  value={bc.rb_lengths_str}
                  onChange={(e) => onBenchmarkConfigChange({ rb_lengths_str: e.target.value })}
                  className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none"
                />
              </div>
            </>
          )}
        </div>

        {/* Coherent Calibration Error Injection Simulator */}
        <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
          <div className="flex items-center justify-between">
            <SectionTitle>相干校准失配注入 (脉冲故障仿真)</SectionTitle>
          </div>
          <div className="flex flex-wrap gap-1">
            {(
              [
                ['理想校准', 0, 0, 0],
                ['+4% 过旋转', 4.0, 0, 0],
                ['-3.5% 欠旋转', -3.5, 0, 0],
                ['3 MHz 失谐', 0, 3.0, 0],
                ['+5° 相位偏斜', 0, 0, 5.0],
              ] as const
            ).map(([lbl, amp, det, skew]) => (
              <button
                key={lbl}
                type="button"
                onClick={() =>
                  onBenchmarkConfigChange({
                    amp_error_pct: amp,
                    detuning_mhz: det,
                    phase_skew_deg: skew,
                  })
                }
                className="px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] border border-[var(--border-hairline)] text-[10px] font-mono-num text-[var(--text-secondary)] cursor-pointer transition-colors"
              >
                {lbl}
              </button>
            ))}
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-0.5">
              <span className="text-[var(--text-secondary)]">拉比幅度偏差 ε_amp</span>
              <span className="font-mono-num font-semibold text-[var(--accent)]">
                {bc.amp_error_pct >= 0 ? '+' : ''}
                {bc.amp_error_pct.toFixed(1)}%
              </span>
            </div>
            <input
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={bc.amp_error_pct}
              onChange={(e) =>
                onBenchmarkConfigChange({ amp_error_pct: Number(e.target.value) })
              }
              className="mac-slider w-full"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-0.5">
              <span className="text-[var(--text-secondary)]">本振频率失谐 Δ/2π</span>
              <span className="font-mono-num font-semibold text-[var(--accent)]">
                {bc.detuning_mhz.toFixed(1)} MHz
              </span>
            </div>
            <input
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={bc.detuning_mhz}
              onChange={(e) =>
                onBenchmarkConfigChange({ detuning_mhz: Number(e.target.value) })
              }
              className="mac-slider w-full"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-0.5">
              <span className="text-[var(--text-secondary)]">IQ 正交相位偏斜 δφ</span>
              <span className="font-mono-num font-semibold text-[var(--accent)]">
                {bc.phase_skew_deg >= 0 ? '+' : ''}
                {bc.phase_skew_deg.toFixed(1)}°
              </span>
            </div>
            <input
              type="range"
              min={-10}
              max={10}
              step={0.5}
              value={bc.phase_skew_deg}
              onChange={(e) =>
                onBenchmarkConfigChange({ phase_skew_deg: Number(e.target.value) })
              }
              className="mac-slider w-full"
            />
          </div>
        </div>

        <button
          onClick={onRunGateBenchmark}
          disabled={isBusy}
          className="w-full py-2 rounded-[3px] studio-btn-primary text-[12px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-60"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          执行 {bc.protocol.toUpperCase()} 门标定实验
        </button>
      </aside>
    );
  }

  // =========================================================================
  // 4. DESIGN MODE CONTEXT-SENSITIVE INSPECTOR (Matches Mockup 1)
  // =========================================================================

  // 4A: Selected Gate Operation
  if (selection.type === 'op') {
    const op = project.operations.find((o) => o.id === selection.opId);
    if (op) {
      const builtin = BUILTIN_GATES.find((g) => g.name.toUpperCase() === op.name.toUpperCase());
      const customG = project.custom_gates.find((g) => g.name.toUpperCase() === op.name.toUpperCase());
      const paramLabels = customG?.param_names.length
        ? customG.param_names
        : builtin?.paramNames || op.params.map((_, i) => `p${i}`);
      const qNames = op.qubits.map((q) => project.qubits[q]?.name || `q${q}`).join(', ');
      const defaultDur =
        customG?.duration_ns ?? (op.qubits.length >= 2 ? activeHw.t_2q_ns : activeHw.t_1q_ns);

      return (
        <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="text-[14px] font-semibold text-[var(--text-primary)] font-mono-num truncate">
              选中: {customG?.display_label || op.name} · {qNames}
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => onDuplicateOp(op.id)}
                title="复制门 (Ctrl+D)"
                className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 text-[var(--text-secondary)]"
              >
                <Copy className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => onDeleteOp(op.id)}
                title="删除门 (Delete)"
                className="p-1 rounded hover:bg-red-500/15 text-[var(--text-secondary)] hover:text-red-500"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Target / Control Qubit Assignment & 2Q Flip */}
          {op.kind !== 'measure' && project.qubits.length >= 1 && (
            <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2">
              <div className="flex items-center justify-between">
                <SectionTitle>作用比特</SectionTitle>
                {op.qubits.length === 2 && (
                  <button
                    onClick={() => onUpdateOp(op.id, { qubits: [op.qubits[1], op.qubits[0]] })}
                    title="翻转控制比特与目标比特"
                    className="text-[11px] text-[#0A84FF] hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    <ArrowUpDown className="w-3 h-3" />
                    翻转控制/目标
                  </button>
                )}
              </div>
              <div className="space-y-1.5">
                {op.qubits.map((qId, slotIdx) => {
                  const slotLabel =
                    op.qubits.length === 1
                      ? '目标比特'
                      : op.qubits.length === 2
                      ? slotIdx === 0
                        ? '控制 / 比特 A'
                        : '目标 / 比特 B'
                      : `比特 ${slotIdx + 1}`;
                  return (
                    <div key={slotIdx} className="flex items-center justify-between gap-2">
                      <span className="text-[12px] text-[var(--text-secondary)]">{slotLabel}</span>
                      <select
                        value={qId}
                        onChange={(e) => {
                          const chosen = Number(e.target.value);
                          const nextQ = [...op.qubits];
                          const dupIdx = nextQ.findIndex((x, i) => i !== slotIdx && x === chosen);
                          if (dupIdx >= 0) {
                            nextQ[dupIdx] = qId; // swap cleanly to avoid duplicate qubits
                          }
                          nextQ[slotIdx] = chosen;
                          onUpdateOp(op.id, { qubits: nextQ });
                        }}
                        className="w-28 px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
                      >
                        {project.qubits.map((q) => (
                          <option key={q.id} value={q.id}>
                            {q.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Parameters Section */}
          {op.params.length > 0 && (
            <div className="pt-3 border-t border-[var(--border-hairline)] space-y-3.5">
              <SectionTitle>参数</SectionTitle>
              {op.params.map((pVal, idx) => {
                const isSymbolic = typeof pVal === 'string';
                const numVal = isSymbolic
                  ? project.variables[pVal] ?? 1.5708
                  : Number(pVal);
                const lbl = paramLabels[idx] || `p${idx}`;

                return (
                  <div key={idx} className="space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[13px] font-mono-num font-medium">{lbl}</span>
                      {isSymbolic ? (
                        <input
                          value={pVal}
                          onChange={(e) => {
                            const next = [...op.params];
                            next[idx] = e.target.value;
                            onUpdateOp(op.id, { params: next });
                          }}
                          className="w-24 px-2 py-1 rounded-lg bg-[#0A84FF]/10 border border-[#0A84FF] text-[#0A84FF] text-right font-mono-num text-[12.5px] font-semibold outline-none"
                        />
                      ) : (
                        <input
                          type="number"
                          step={0.05}
                          value={Number(numVal.toFixed(4))}
                          onChange={(e) => {
                            const next = [...op.params];
                            next[idx] = Number(e.target.value);
                            onUpdateOp(op.id, { params: next });
                          }}
                          className="w-24 px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-right font-mono-num text-[12.5px] outline-none focus:border-[#0A84FF]"
                        />
                      )}
                    </div>

                    {!isSymbolic && (
                      <input
                        type="range"
                        min={-3.1416}
                        max={3.1416}
                        step={0.01}
                        value={Math.max(-3.1416, Math.min(3.1416, numVal))}
                        onChange={(e) => {
                          const next = [...op.params];
                          next[idx] = Number(e.target.value);
                          onUpdateOp(op.id, { params: next });
                        }}
                        className="mac-slider w-full"
                      />
                    )}

                    <div className="flex items-center justify-between pt-0.5">
                      <span className="text-[12px] text-[var(--text-secondary)]">设为符号参数</span>
                      <IOSSwitch
                        checked={isSymbolic}
                        onChange={(toSym) => {
                          const next = [...op.params];
                          if (toSym) {
                            const varName = idx === 0 ? 'theta' : idx === 1 ? 'phi' : `p_${idx}`;
                            next[idx] = varName;
                            onEnsureVariable(varName, numVal);
                          } else {
                            next[idx] = Number(numVal.toFixed(4));
                          }
                          onUpdateOp(op.id, { params: next });
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Timing, Rabi Frequency & Noise Section */}
          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-3">
            <SectionTitle>驱动拉比、时序与噪声</SectionTitle>
            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">时间列 Step</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => onUpdateOp(op.id, { step: Math.max(0, op.step - 1) })}
                  className="p-1 rounded-md border border-[var(--border-input)] hover:border-[#0A84FF]"
                >
                  <ArrowLeft className="w-3 h-3" />
                </button>
                <span className="w-8 text-center font-mono-num text-[12.5px] font-semibold">
                  {op.step + 1}
                </span>
                <button
                  onClick={() => onUpdateOp(op.id, { step: op.step + 1 })}
                  className="p-1 rounded-md border border-[var(--border-input)] hover:border-[#0A84FF]"
                >
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>
            </div>

            {op.kind === 'gate' && op.qubits.length === 1 && !isVirtual1QGate(op.name) && (() => {
              const qObj = project.qubits[op.qubits[0]];
              const inheritedRabi = qObj?.rabi_mhz ?? activeHw.rabi_mhz ?? 2.5;
              const theta = singleQubitRotationAngle(op.name, op.params, project.variables);
              const effRabi = op.rabi_mhz ?? inheritedRabi;
              const usesRabiDur =
                op.duration_ns == null &&
                (op.rabi_mhz != null ||
                  qObj?.rabi_mhz != null ||
                  activeHw.drive_mode === 'fixed_rabi');
              const derivedDur = usesRabiDur
                ? rabiPulseDurationNs(
                    theta,
                    effRabi,
                    activeHw.pulse_shape,
                    activeHw.rise_ns ?? 0,
                    activeHw.fall_ns ?? 0
                  )
                : op.duration_ns ?? defaultDur;
              const effPeakRabi =
                op.duration_ns != null
                  ? impliedRabiMhz(
                      theta,
                      op.duration_ns,
                      activeHw.pulse_shape,
                      activeHw.rise_ns ?? 0,
                      activeHw.fall_ns ?? 0
                    )
                  : effRabi;

              return (
                <div className="p-2.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[12px] font-medium">拉比频率 Ω_R/2π</span>
                    <div className="flex items-center gap-1">
                      <input
                        type="number"
                        min={0.01}
                        step={0.25}
                        placeholder={`继承 ${inheritedRabi}`}
                        value={op.rabi_mhz ?? ''}
                        onChange={(e) =>
                          onUpdateOp(op.id, {
                            rabi_mhz:
                              e.target.value === '' ? null : Math.max(0.01, Number(e.target.value)),
                            ...(e.target.value !== '' ? { duration_ns: null } : {}),
                          })
                        }
                        className={`w-22 px-2 py-0.5 rounded-[2px] border text-right font-mono-num text-[12px] outline-none focus:border-[var(--accent)] ${
                          op.rabi_mhz != null
                            ? 'bg-amber-500/12 border-amber-500/60 text-amber-400 font-semibold'
                            : 'bg-[var(--bg-panel)] border-[var(--border-input)]'
                        }`}
                      />
                      <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">MHz</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                    <span>实际脉冲宽度: {derivedDur.toFixed(1)} ns</span>
                    <span>峰值 Ω_R: {effPeakRabi.toFixed(2)} MHz</span>
                  </div>
                </div>
              );
            })()}

            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">强制门时长覆盖</span>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  placeholder={String(defaultDur)}
                  value={op.duration_ns ?? ''}
                  onChange={(e) =>
                    onUpdateOp(op.id, {
                      duration_ns: e.target.value === '' ? null : Number(e.target.value),
                    })
                  }
                  className="w-20 px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
                />
                <span className="text-[11.5px] text-[var(--text-secondary)]">ns</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">噪声</span>
              <select
                value={op.noise_mode || 'inherit'}
                onChange={(e) =>
                  onUpdateOp(op.id, {
                    noise_mode: e.target.value as CircuitOpModel['noise_mode'],
                  })
                }
                className="px-2.5 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none focus:border-[#0A84FF]"
              >
                <option value="inherit">继承硬件</option>
                <option value="none">无噪声 (理想门)</option>
                <option value="custom">自定义退极化</option>
              </select>
            </div>

            {op.noise_mode === 'custom' && (
              <div className="flex items-center justify-between">
                <span className="text-[12px] text-[var(--text-secondary)]">附加退极化率</span>
                <input
                  type="number"
                  step={0.002}
                  min={0}
                  max={0.5}
                  value={op.custom_depol ?? 0.01}
                  onChange={(e) => onUpdateOp(op.id, { custom_depol: Number(e.target.value) })}
                  className="w-24 px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none"
                />
              </div>
            )}
          </div>
        </aside>
      );
    }
  }

  // 4B: Selected Qubit (Quantum Register Inspector)
  if (selection.type === 'qubit') {
    const q = project.qubits.find((item) => item.id === selection.qubitId);
    if (q) {
      return (
        <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: q.color }} />
              <span className="text-[14px] font-semibold font-mono-num">比特属性 · {q.name}</span>
            </div>
          </div>

          <div className="space-y-3 pt-2 border-t border-[var(--border-hairline)]">
            <SectionTitle>基本设置</SectionTitle>
            <div className="flex items-center justify-between gap-2">
              <span className="text-[12.5px]">名称</span>
              <input
                value={q.name}
                onChange={(e) => onUpdateQubit(q.id, { name: e.target.value })}
                className="w-32 px-2 py-1 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12.5px] outline-none focus:border-[#0A84FF]"
              />
            </div>

            <div>
              <span className="block text-[12px] text-[var(--text-secondary)] mb-1.5">标识色</span>
              <div className="flex items-center gap-1.5">
                {QUBIT_DOT_PALETTE.map((col) => (
                  <button
                    key={col}
                    onClick={() => onUpdateQubit(q.id, { color: col })}
                    style={{ backgroundColor: col }}
                    className={`w-5 h-5 rounded-full cursor-pointer transition-transform ${
                      q.color === col ? 'ring-2 ring-offset-2 ring-[#0A84FF] scale-110' : ''
                    }`}
                  />
                ))}
              </div>
            </div>

            <div>
              <span className="block text-[12px] text-[var(--text-secondary)] mb-1.5">初始量子态</span>
              <div className="grid grid-cols-3 gap-1.5">
                {(['|0>', '|1>', '|+>', '|->', '|+i>', '|-i>'] as InitStateType[]).map((st) => (
                  <button
                    key={st}
                    onClick={() => onUpdateQubit(q.id, { init_state: st })}
                    className={`py-1 rounded-md border font-mono-num text-[12px] cursor-pointer ${
                      q.init_state === st
                        ? 'bg-[#0A84FF]/12 border-[#0A84FF] text-[#0A84FF] font-semibold'
                        : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)] text-[var(--text-primary)]'
                    }`}
                  >
                    {st.replace('>', '⟩')}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2.5">
            <div className="flex items-center justify-between">
              <SectionTitle>单比特物理参数覆盖</SectionTitle>
              <button
                onClick={() =>
                  onUpdateQubit(q.id, {
                    rabi_mhz: null,
                    t1_us: null,
                    t2_us: null,
                    depol_1q: null,
                    readout_err: null,
                  })
                }
                title="恢复全部继承硬件配置"
                className="text-[11px] text-[#0A84FF] flex items-center gap-0.5 hover:underline cursor-pointer"
              >
                <RotateCcw className="w-3 h-3" />
                重置继承
              </button>
            </div>
            {(
              [
                ['拉比频率 Ω_R (MHz)', 'rabi_mhz', activeHw.rabi_mhz ?? 2.5],
                ['T1 (μs)', 't1_us', activeHw.t1_us],
                ['T2* (μs)', 't2_us', activeHw.t2_us],
                ['1Q 退极化', 'depol_1q', activeHw.depol_1q],
                ['读出误差', 'readout_err', activeHw.readout_err],
              ] as const
            ).map(([label, field, inherited]) => {
              const val = q[field] ?? null;
              const isOverridden = val !== null;
              return (
                <div key={field} className="flex items-center justify-between gap-2">
                  <span
                    className={`text-[12px] ${
                      isOverridden ? 'text-amber-600 font-medium' : 'text-[var(--text-secondary)]'
                    }`}
                  >
                    {label}
                  </span>
                  <input
                    type="number"
                    placeholder={`继承 ${inherited}`}
                    value={val ?? ''}
                    onChange={(e) =>
                      onUpdateQubit(q.id, {
                        [field]: e.target.value === '' ? null : Number(e.target.value),
                      })
                    }
                    className={`w-28 px-2 py-1 rounded-lg border text-right font-mono-num text-[12px] outline-none focus:border-[#0A84FF] ${
                      isOverridden
                        ? 'bg-amber-50/70 dark:bg-amber-950/30 border-amber-400'
                        : 'bg-[var(--bg-subtle)] border-[var(--border-input)]'
                    }`}
                  />
                </div>
              );
            })}
          </div>

          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-2">
            <SectionTitle>空间坐标 (中性原子光镊 μm)</SectionTitle>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10.5px] text-[var(--text-secondary)]">X (μm)</label>
                <input
                  type="number"
                  step={0.5}
                  value={q.pos_x}
                  onChange={(e) => onUpdateQubit(q.id, { pos_x: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px]"
                />
              </div>
              <div>
                <label className="text-[10.5px] text-[var(--text-secondary)]">Y (μm)</label>
                <input
                  type="number"
                  step={0.5}
                  value={q.pos_y}
                  onChange={(e) => onUpdateQubit(q.id, { pos_y: Number(e.target.value) })}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px]"
                />
              </div>
            </div>
          </div>
        </aside>
      );
    }
  }

  // 4C: Selected Custom Gate
  if (selection.type === 'custom_gate') {
    const cg = project.custom_gates.find((g) => g.id === selection.gateId);
    if (cg) {
      return (
        <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-[14px] font-semibold font-mono-num">
              自定义门 · {cg.display_label || cg.name}
            </div>
            <button
              onClick={() => onEditCustomGate(cg)}
              className="px-2 py-1 rounded-md bg-[#0A84FF]/10 text-[#0A84FF] text-[11.5px] font-medium flex items-center gap-1 cursor-pointer"
            >
              <Pencil className="w-3 h-3" />
              编辑
            </button>
          </div>

          <div className="p-3 rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-1.5 text-[12px]">
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">定义方式</span>
              <span className="font-mono-num font-medium">{cg.definition_type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">作用比特</span>
              <span className="font-mono-num font-medium">{cg.num_qubits} Qubits</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">门时长</span>
              <span className="font-mono-num font-medium">{cg.duration_ns} ns</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">不保真度</span>
              <span className="font-mono-num font-medium">{cg.infidelity}</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-[var(--border-hairline)]">
            <div>
              <div className="text-[12.5px] font-medium">作为实验原生门</div>
              <div className="text-[11px] text-[var(--text-secondary)]">
                开启后优化器将优先使用该门综合线路
              </div>
            </div>
            <IOSSwitch
              checked={cg.is_native}
              onChange={(v) => onToggleCustomGateNative(cg.id, v)}
            />
          </div>

          {cg.description && (
            <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">
              {cg.description}
            </p>
          )}
        </aside>
      );
    }
  }

  // 4D: Default / Selected Hardware Profile + Interactive Topology & Neutral Atom Map
  const hw =
    (selection.type === 'hardware' &&
      project.hardware_profiles.find((h) => h.id === selection.hardwareId)) ||
    activeHw;

  return (
    <aside className="w-[288px] shrink-0 bg-[var(--bg-sidebar)] border-l border-[var(--border-hairline)] overflow-y-auto p-3.5 space-y-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-[var(--text-primary)]">{hw.name}</div>
          <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
            Ω_R = {hw.rabi_mhz ?? 2.5} MHz ·{' '}
            {(hw.drive_mode || 'fixed_duration') === 'fixed_rabi' ? '固定拉比频率' : '固定门时长'}
          </div>
        </div>
        {onOpenDeviceWorkspace && (
          <button
            type="button"
            onClick={onOpenDeviceWorkspace}
            className="px-2 py-1 rounded-[3px] bg-[var(--accent-soft)] hover:bg-[var(--accent)]/25 text-[var(--accent)] text-[11px] font-semibold shrink-0 cursor-pointer"
          >
            器件工作区 →
          </button>
        )}
      </div>

      {/* Topology & Neutral Atom 2D Coordinates Map */}
      <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
        <div className="flex items-center justify-between">
          <SectionTitle>拓扑与原子排布</SectionTitle>
          <select
            value={hw.topology_mode}
            onChange={(e) =>
              onUpdateHardware(hw.id, {
                topology_mode: e.target.value as HardwareProfileModel['topology_mode'],
              })
            }
            className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11px] outline-none"
          >
            <option value="rydberg_radius">里德堡阻塞半径</option>
            <option value="all">全连接</option>
            <option value="linear">一维链</option>
            <option value="custom">手动连边</option>
          </select>
        </div>

        <TopologyEditor
          qubits={project.qubits}
          hardware={hw}
          onUpdateQubitPos={(qid, x, y) => onUpdateQubit(qid, { pos_x: x, pos_y: y })}
          onApplyLatticeLayout={onApplyLatticeLayout}
          onToggleCustomEdge={onToggleCustomEdge}
        />

        {hw.topology_mode === 'rydberg_radius' && (
          <div className="space-y-1">
            <div className="flex justify-between text-[11.5px]">
              <span className="text-[var(--text-secondary)]">里德堡阻塞半径 R_b</span>
              <span className="font-mono-num font-semibold text-[#0A84FF]">
                {hw.blockade_radius_um.toFixed(1)} μm
              </span>
            </div>
            <input
              type="range"
              min={2}
              max={15}
              step={0.5}
              value={hw.blockade_radius_um}
              onChange={(e) =>
                onUpdateHardware(hw.id, { blockade_radius_um: Number(e.target.value) })
              }
              className="mac-slider w-full"
            />
          </div>
        )}
      </div>

      {/* Physical Parameters */}
      <div className="pt-3 border-t border-[var(--border-hairline)] space-y-2.5">
        <div className="flex items-center justify-between">
          <SectionTitle>物理噪声与门时序</SectionTitle>
          <select
            value={hw.pulse_shape || 'cosine'}
            onChange={(e) =>
              onUpdateHardware(hw.id, {
                pulse_shape: e.target.value as HardwareProfileModel['pulse_shape'],
              })
            }
            title="QuTiP 连续脉冲包络形状"
            className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11px] outline-none"
          >
            <option value="cosine">余弦脉冲 (Hann)</option>
            <option value="gaussian">高斯脉冲 (Gaussian)</option>
            <option value="rectangular">平顶余弦边沿 (Flat-Top)</option>
            <option value="trapezoid">平顶线性梯形 (Trapezoid)</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
              驱动模式
            </label>
            <select
              value={hw.drive_mode || 'fixed_duration'}
              onChange={(e) =>
                onUpdateHardware(hw.id, {
                  drive_mode: e.target.value as HardwareProfileModel['drive_mode'],
                })
              }
              className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] outline-none"
            >
              <option value="fixed_rabi">固定 Ω_R</option>
              <option value="fixed_duration">固定 t_1q</option>
            </select>
          </div>
          <div>
            <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
              拉比 Ω_R (MHz)
            </label>
            <input
              type="number"
              step={0.25}
              min={0.01}
              value={hw.rabi_mhz ?? 2.5}
              onChange={(e) =>
                onUpdateHardware(hw.id, { rabi_mhz: Math.max(0.01, Number(e.target.value)) })
              }
              className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
            />
          </div>
          {(
            [
              ['上升沿 (ns)', 'rise_ns', 1],
              ['下降沿 (ns)', 'fall_ns', 1],
              ['T1 (μs)', 't1_us', 10],
              ['T2* (μs)', 't2_us', 10],
              ['1Q 时长 (ns)', 't_1q_ns', 10],
              ['2Q 时长 (ns)', 't_2q_ns', 10],
              ['1Q 退极化', 'depol_1q', 0.0005],
              ['2Q 退极化', 'depol_2q', 0.001],
              ['ZZ 串扰 (kHz)', 'zz_khz', 1],
              ['读出误差', 'readout_err', 0.002],
            ] as const
          ).map(([lbl, key, step]) => (
            <div key={key}>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                {lbl}
              </label>
              <input
                type="number"
                step={step}
                value={hw[key]}
                onChange={(e) => onUpdateHardware(hw.id, { [key]: Number(e.target.value) })}
                className="w-full px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[#0A84FF]"
              />
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
};
