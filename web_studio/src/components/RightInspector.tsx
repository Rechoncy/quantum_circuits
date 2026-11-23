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

export interface SweepConfigState {
  mode: '1d' | '2d';
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
  backend: 'qutip' | 'kraus' | 'stim' | 'stim_qutip';
}

function defaultSweepRange(paramKey: string): [number, number] {
  if (paramKey === 'hw:t1_us' || paramKey === 'hw:t2_us') return [5, 200];
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
    className={`w-9 h-5 rounded-full p-0.5 transition-colors cursor-pointer shrink-0 ${
      checked ? 'bg-[#0A84FF]' : 'bg-[#D1D1D6] dark:bg-[#48484A]'
    }`}
  >
    <div
      className={`w-4 h-4 rounded-full bg-white shadow-xs transition-transform ${
        checked ? 'translate-x-4' : 'translate-x-0'
      }`}
    />
  </button>
);

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] mb-2">
    {children}
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
      <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-5">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">模拟设置</div>

        {/* Solver Segmented Control */}
        <div>
          <SectionTitle>求解器</SectionTitle>
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg grid grid-cols-3 gap-0.5">
            {(['qutip', 'kraus', 'stim'] as SolverBackendType[]).map((b) => (
              <button
                key={b}
                onClick={() => onUpdateSimSettings({ backend: b })}
                className={`py-1.5 rounded-[7px] text-[12px] font-medium transition-all cursor-pointer ${
                  sim.backend === b
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {b === 'qutip' ? 'QuTiP' : b === 'kraus' ? 'Kraus' : 'Stim'}
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
      <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-5">
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
      <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-5">
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">参数扫描设置</div>

        <div className="space-y-3">
          <SectionTitle>扫描维度与后端</SectionTitle>
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg grid grid-cols-2 gap-0.5">
            {(['1d', '2d'] as const).map((m) => (
              <button
                key={m}
                onClick={() => onSweepConfigChange({ mode: m })}
                className={`py-1.5 rounded-[7px] text-[12px] font-medium cursor-pointer ${
                  sweepConfig.mode === m
                    ? 'bg-[var(--bg-panel)] text-[#0A84FF] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)]'
                }`}
              >
                {m === '1d' ? '一维曲线 (1D)' : '二维相图 (2D)'}
              </button>
            ))}
          </div>

          <div>
            <label className="block text-[11px] text-[var(--text-secondary)] mb-1">扫描引擎</label>
            <select
              value={sweepConfig.backend}
              onChange={(e) =>
                onSweepConfigChange({ backend: e.target.value as SweepConfigState['backend'] })
              }
              className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
            >
              <option value="qutip">QuTiP · 物理主方程</option>
              <option value="kraus">Kraus · 快速密度矩阵</option>
              <option value="stim">Stim · 稳定子高速采样</option>
              <option value="stim_qutip">Stim + QuTiP 标定</option>
            </select>
          </div>

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
        </div>

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

        <div className="flex items-center gap-2">
          <button
            onClick={onRunSweep}
            disabled={isBusy}
            className="flex-1 py-2 rounded-xl studio-btn-primary text-[12.5px] font-semibold flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-60"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            执行参数扫描
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
        <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-5">
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

          {/* Timing & Noise Section */}
          <div className="pt-4 border-t border-[var(--border-hairline)] space-y-3">
            <SectionTitle>时序与噪声</SectionTitle>
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

            <div className="flex items-center justify-between">
              <span className="text-[12.5px]">时长</span>
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
        <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-5">
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
                ['T1 (μs)', 't1_us', activeHw.t1_us],
                ['T2* (μs)', 't2_us', activeHw.t2_us],
                ['1Q 退极化', 'depol_1q', activeHw.depol_1q],
                ['读出误差', 'readout_err', activeHw.readout_err],
              ] as const
            ).map(([label, field, inherited]) => {
              const val = q[field];
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
        <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-4">
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
    <aside className="w-[296px] shrink-0 bg-[var(--bg-panel)] border-l border-[var(--border-hairline)] overflow-y-auto p-4 space-y-4">
      <div>
        <div className="text-[14px] font-semibold text-[var(--text-primary)]">{hw.name}</div>
        <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
          {hw.platform === 'neutral_atom'
            ? '光镊原子阵列 · 拉曼 RXY + 里德堡阻塞 CZ'
            : hw.platform === 'trapped_ion'
            ? '线性保罗阱 · 赤道旋转 RXY + Mølmer-Sørensen'
            : '超导 Transmon · Virtual-Z + 微波脉冲'}
        </div>
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
            <option value="rectangular">矩形脉冲 (Rect)</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {(
            [
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
