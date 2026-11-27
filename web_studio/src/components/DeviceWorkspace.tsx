import React, { useMemo } from 'react';
import {
  Activity,
  ArrowLeftRight,
  Check,
  Cpu,
  Gauge,
  Layers,
  Pencil,
  Plus,
  RotateCcw,
  Sliders,
  Trash2,
} from 'lucide-react';
import type {
  BenchmarkProtocol,
  CustomGateDef,
  DeviceSubTab,
  HardwareProfileModel,
  InitStateType,
  ProjectModel,
  QubitSpec,
} from '../types';
import { PlotlyChart } from './PlotlyChart';
import { TopologyEditor } from './TopologyEditor';
import {
  envelopeAreaFactor,
  estimateTransmonLeakage,
  impliedRabiMhz,
  rabiPulseDurationNs,
  sampleDragQuadratureMhz,
  samplePeakNormalizedEnvelope,
} from '../utils/drivePhysics';

interface DeviceWorkspaceProps {
  project: ProjectModel;
  activeTab: DeviceSubTab;
  onTabChange: (tab: DeviceSubTab) => void;
  darkMode: boolean;
  onUpdateHardware: (hwId: string, patch: Partial<HardwareProfileModel>) => void;
  onSelectHardware: (hwId: string) => void;
  onUpdateQubit: (qubitId: number, patch: Partial<QubitSpec>) => void;
  onAddQubit: () => void;
  onDeleteQubit: (qubitId: number) => void;
  onApplyLatticeLayout?: (positions: { x: number; y: number }[]) => void;
  onToggleCustomEdge: (u: number, v: number) => void;
  onToggleCustomGateNative: (gateId: string, isNative: boolean) => void;
  onOpenNewGateModal: () => void;
  onEditCustomGate: (gate: CustomGateDef) => void;
  onDeleteCustomGate: (gateId: string) => void;
  onJumpToCalibrate: (qubitId: number, protocol: BenchmarkProtocol) => void;
}

export const DeviceWorkspace: React.FC<DeviceWorkspaceProps> = ({
  project,
  activeTab,
  onTabChange,
  darkMode,
  onUpdateHardware,
  onSelectHardware,
  onUpdateQubit,
  onAddQubit,
  onDeleteQubit,
  onApplyLatticeLayout,
  onToggleCustomEdge,
  onToggleCustomGateNative,
  onOpenNewGateModal,
  onEditCustomGate,
  onDeleteCustomGate,
  onJumpToCalibrate,
}) => {
  const hw =
    project.hardware_profiles.find((h) => h.id === project.active_hardware_id) ||
    project.hardware_profiles[0];

  const driveMode = hw.drive_mode || 'fixed_duration';
  const pulseShape = hw.pulse_shape || 'cosine';
  const riseNs = hw.rise_ns ?? 0;
  const fallNs = hw.fall_ns ?? 0;
  const rabiMhz = hw.rabi_mhz ?? 2.5;
  const anharmMhz = hw.anharmonicity_mhz ?? 0;
  const dragBeta = hw.drag_beta ?? 0;

  const mutedColor = darkMode ? '#8B96A5' : '#475569';
  const gridColor = darkMode ? 'rgba(255,255,255,0.07)' : 'rgba(15,23,42,0.08)';
  const plotBg = 'rgba(0,0,0,0)';

  // Derived physical pulse metrics for pi (X/Y) and pi/2 (SX/H), including 3-level DRAG quadrature Q(t)
  const driveMetrics = useMemo(() => {
    const tPi =
      driveMode === 'fixed_rabi'
        ? rabiPulseDurationNs(Math.PI, rabiMhz, pulseShape, riseNs, fallNs)
        : hw.t_1q_ns;
    const tPi2 =
      driveMode === 'fixed_rabi'
        ? rabiPulseDurationNs(0.5 * Math.PI, rabiMhz, pulseShape, riseNs, fallNs)
        : hw.t_1q_ns;
    const omegaPi =
      driveMode === 'fixed_rabi'
        ? rabiMhz
        : impliedRabiMhz(Math.PI, hw.t_1q_ns, pulseShape, riseNs, fallNs);
    const omegaPi2 =
      driveMode === 'fixed_rabi'
        ? rabiMhz
        : impliedRabiMhz(0.5 * Math.PI, hw.t_1q_ns, pulseShape, riseNs, fallNs);
    const kappa = envelopeAreaFactor(pulseShape, Math.max(1, tPi), riseNs, fallNs);

    const tMax = Math.max(10, tPi, tPi2) * 1.08;
    const nPts = 220;
    const xs = Array.from({ length: nPts }, (_, i) => (i / (nPts - 1)) * tMax);
    const yPi = xs.map((t) =>
      t <= tPi ? omegaPi * samplePeakNormalizedEnvelope(t, tPi, pulseShape, riseNs, fallNs) : 0
    );
    const yPi2 = xs.map((t) =>
      t <= tPi2 ? omegaPi2 * samplePeakNormalizedEnvelope(t, tPi2, pulseShape, riseNs, fallNs) : 0
    );
    const yDragQ = xs.map((t) =>
      t <= tPi
        ? sampleDragQuadratureMhz(t, tPi, omegaPi, anharmMhz, dragBeta, pulseShape, riseNs, fallNs)
        : 0
    );
    const hasDragTrace = Math.abs(anharmMhz) >= 1 && Math.abs(dragBeta) > 1e-4;
    const minY = hasDragTrace ? Math.min(0, ...yDragQ) * 1.25 : 0;
    const maxY = Math.max(omegaPi, omegaPi2, ...yDragQ, 0.2) * 1.18;
    const estLeakNoDrag = estimateTransmonLeakage(
      Math.PI,
      tPi,
      anharmMhz,
      0,
      pulseShape,
      riseNs,
      fallNs
    );
    const estLeakWithDrag = estimateTransmonLeakage(
      Math.PI,
      tPi,
      anharmMhz,
      dragBeta,
      pulseShape,
      riseNs,
      fallNs
    );
    const effRise = Math.min(riseNs, tPi * 0.5);
    const effFall = Math.min(fallNs, tPi * 0.5);
    return {
      tPi,
      tPi2,
      omegaPi,
      omegaPi2,
      kappa,
      tMax,
      xs,
      yPi,
      yPi2,
      yDragQ,
      hasDragTrace,
      minY,
      maxY,
      estLeakNoDrag,
      estLeakWithDrag,
      effRise,
      effFall,
    };
  }, [driveMode, rabiMhz, hw.t_1q_ns, pulseShape, riseNs, fallNs, anharmMhz, dragBeta]);

  const qubitOverrideCount = project.qubits.filter((q) => q.rabi_mhz != null).length;
  const opOverrideCount = project.operations.filter(
    (o) => o.kind === 'gate' && o.rabi_mhz != null
  ).length;

  const rabiSliderMax =
    hw.platform === 'trapped_ion'
      ? 5
      : hw.platform === 'neutral_atom'
      ? 20
      : 120;
  const edgeSliderMax =
    hw.platform === 'trapped_ion'
      ? 500
      : hw.platform === 'neutral_atom'
      ? 150
      : 15;

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-[var(--bg-app)] overflow-hidden">
      {/* Level-2 Sub-Tab Bar for Device & Hardware Domain */}
      <div className="h-9 px-3.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0 select-none">
        <div className="flex items-center gap-4 h-full overflow-x-auto min-w-0">
          {(
            [
              { id: 'drive', label: '单比特驱动与上升/下降沿', icon: Sliders },
              { id: 'qubits', label: `量子寄存器与逐比特参数 (${project.qubits.length}Q)`, icon: Activity },
              { id: 'topology', label: '耦合拓扑与光镊阵列', icon: Layers },
              { id: 'platforms', label: '硬件平台与原生门集', icon: Cpu },
            ] as const
          ).map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => onTabChange(t.id)}
                className={`h-full text-[12px] border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  active
                    ? 'border-[var(--accent)] text-[var(--text-primary)] font-semibold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${active ? 'text-[var(--accent)]' : 'opacity-70'}`} />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>

        {/* Active Platform Quick Switcher */}
        <div className="flex items-center gap-2 shrink-0 text-[11.5px]">
          <span className="text-[var(--text-secondary)] hidden lg:inline">当前激活器件:</span>
          <select
            value={hw.id}
            onChange={(e) => onSelectHardware(e.target.value)}
            className="h-6 px-2 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[11.5px] text-[var(--text-primary)] outline-none focus:border-[var(--accent)] cursor-pointer"
          >
            {project.hardware_profiles.map((hp) => (
              <option key={hp.id} value={hp.id}>
                {hp.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* =================================================================== */}
      {/* TAB 1: SINGLE-QUBIT RABI DRIVE, PULSE ENVELOPE & RISE/FALL EDGES    */}
      {/* =================================================================== */}
      {activeTab === 'drive' && (
        <div className="flex-1 min-h-0 overflow-y-auto p-4">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
            {/* Left 6 Cols: Drive Mode, Rabi Frequency, Rise/Fall Edges & Decoherence */}
            <div className="xl:col-span-6 space-y-4">
              {/* Card A: Drive Mode & Rabi Frequency */}
              <div className="studio-card p-4 space-y-3.5">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-[13px] font-bold text-[var(--text-primary)]">
                      单比特拉比驱动模型 (Ω_R ↔ 脉冲时长关系)
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                      满足积分面积关系 θ = ∫₀ᵀ Ω(t) dt = 2π · Ω_peak · κ · T（支持硬件 → 比特 → 单门三级继承）
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onJumpToCalibrate(0, 'rabi')}
                    className="px-2.5 py-1 rounded-[3px] bg-[var(--accent-soft)] hover:bg-[var(--accent)]/25 text-[var(--accent)] text-[11px] font-semibold flex items-center gap-1 shrink-0 cursor-pointer"
                  >
                    <Gauge className="w-3.5 h-3.5" />
                    去测 Rabi 振荡
                  </button>
                </div>

                {/* Segmented Toggle: Fixed Rabi vs Fixed Duration */}
                <div className="bg-[var(--bg-subtle)] border border-[var(--border-hairline)] p-0.5 rounded-[3px] grid grid-cols-2 gap-1">
                  <button
                    type="button"
                    onClick={() => onUpdateHardware(hw.id, { drive_mode: 'fixed_rabi' })}
                    className={`py-2 px-2.5 rounded-[2px] text-left transition-colors cursor-pointer ${
                      driveMode === 'fixed_rabi'
                        ? 'bg-[var(--bg-panel)] border border-[var(--accent)] text-[var(--text-primary)]'
                        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
                    <div className="text-[12px] font-semibold flex items-center justify-between">
                      <span>固定拉比频率 Ω_R</span>
                      {driveMode === 'fixed_rabi' && (
                        <span className="text-[10px] font-mono-num text-[var(--accent)]">ACTIVE</span>
                      )}
                    </div>
                    <div className="text-[10.5px] text-[var(--text-secondary)] mt-0.5">
                      峰值拉比频率恒定，脉冲时长随旋转角 θ 自动缩放
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => onUpdateHardware(hw.id, { drive_mode: 'fixed_duration' })}
                    className={`py-2 px-2.5 rounded-[2px] text-left transition-colors cursor-pointer ${
                      driveMode === 'fixed_duration'
                        ? 'bg-[var(--bg-panel)] border border-[var(--accent)] text-[var(--text-primary)]'
                        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
                    <div className="text-[12px] font-semibold flex items-center justify-between">
                      <span>固定门时长 t_1q</span>
                      {driveMode === 'fixed_duration' && (
                        <span className="text-[10px] font-mono-num text-[var(--accent)]">ACTIVE</span>
                      )}
                    </div>
                    <div className="text-[10.5px] text-[var(--text-secondary)] mt-0.5">
                      所有单比特门等长 (AWG 同步帧)，驱动幅度随 θ 缩放
                    </div>
                  </button>
                </div>

                {/* Rabi Frequency & 1Q/2Q Duration Controls */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                  <div className="p-3 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11.5px] font-semibold text-[var(--text-primary)]">
                        硬件默认拉比频率 Ω_R / 2π
                      </span>
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          min={0.01}
                          max={500}
                          step={hw.platform === 'trapped_ion' ? 0.05 : 0.25}
                          value={Number(rabiMhz.toFixed(4))}
                          onChange={(e) =>
                            onUpdateHardware(hw.id, {
                              rabi_mhz: Math.max(0.01, Number(e.target.value)),
                            })
                          }
                          className="w-20 px-2 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">MHz</span>
                      </div>
                    </div>
                    <input
                      type="range"
                      min={0.05}
                      max={rabiSliderMax}
                      step={hw.platform === 'trapped_ion' ? 0.05 : 0.25}
                      value={Math.min(rabiSliderMax, rabiMhz)}
                      onChange={(e) =>
                        onUpdateHardware(hw.id, { rabi_mhz: Number(e.target.value) })
                      }
                      className="mac-slider w-full"
                    />
                    <div className="flex items-center justify-between text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                      <span>2π 拉比周期 T_2π = {(1000 / Math.max(1e-4, rabiMhz)).toFixed(1)} ns</span>
                      <button
                        type="button"
                        onClick={() => {
                          const syncedT = rabiPulseDurationNs(
                            Math.PI,
                            rabiMhz,
                            pulseShape,
                            riseNs,
                            fallNs
                          );
                          onUpdateHardware(hw.id, { t_1q_ns: Number(syncedT.toFixed(2)) });
                        }}
                        title="用当前 Ω_R 推导出的 π 脉冲时长覆盖右侧 t_1q"
                        className="text-[var(--accent)] hover:underline inline-flex items-center gap-0.5 cursor-pointer"
                      >
                        <ArrowLeftRight className="w-3 h-3" />
                        同步至 t_1q
                      </button>
                    </div>
                  </div>

                  <div className="p-3 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[11.5px] font-semibold text-[var(--text-primary)]">
                        基准单比特门时长 t_1q
                      </span>
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          min={1}
                          step={5}
                          value={hw.t_1q_ns}
                          onChange={(e) =>
                            onUpdateHardware(hw.id, {
                              t_1q_ns: Math.max(0.5, Number(e.target.value)),
                            })
                          }
                          className="w-20 px-2 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">ns</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between pt-0.5">
                      <span className="text-[11.5px] text-[var(--text-secondary)]">
                        两比特纠缠门时长 t_2q
                      </span>
                      <div className="flex items-center gap-1">
                        <input
                          type="number"
                          min={1}
                          step={10}
                          value={hw.t_2q_ns}
                          onChange={(e) =>
                            onUpdateHardware(hw.id, {
                              t_2q_ns: Math.max(1, Number(e.target.value)),
                            })
                          }
                          className="w-20 px-2 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">ns</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-[10.5px] font-mono-num text-[var(--text-secondary)] pt-0.5">
                      <span>
                        等效 π 峰值:{' '}
                        {impliedRabiMhz(Math.PI, hw.t_1q_ns, pulseShape, riseNs, fallNs).toFixed(2)}{' '}
                        MHz
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          const imp = impliedRabiMhz(
                            Math.PI,
                            hw.t_1q_ns,
                            pulseShape,
                            riseNs,
                            fallNs
                          );
                          onUpdateHardware(hw.id, { rabi_mhz: Number(imp.toFixed(4)) });
                        }}
                        title="根据当前 t_1q 反推 π 脉冲所需的峰值拉比频率并写入 Ω_R"
                        className="text-[var(--accent)] hover:underline inline-flex items-center gap-0.5 cursor-pointer"
                      >
                        <ArrowLeftRight className="w-3 h-3" />
                        反推至 Ω_R
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card B: Pulse Envelope & Device Rise/Fall Edges */}
              <div className="studio-card p-4 space-y-3.5">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-[13px] font-bold text-[var(--text-primary)]">
                      AWG 脉冲包络与器件上升 / 下降沿 (Rise & Fall Edges)
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                      平顶包络 (Flat-Top / 梯形) 直接由上升沿 t_rise 与下降沿 t_fall 控制边沿平滑过渡，抑制频谱泄漏
                    </div>
                  </div>
                </div>

                {/* 4 Envelope Shape Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {(
                    [
                      ['cosine', '余弦 Hann', '全平滑零边沿'],
                      ['gaussian', '截断高斯', '低频谱旁瓣'],
                      ['rectangular', '平顶余弦边沿', '可控上升/下降沿'],
                      ['trapezoid', '平顶线性梯形', '线性斜坡边沿'],
                    ] as const
                  ).map(([shapeId, title, desc]) => {
                    const active = pulseShape === shapeId;
                    return (
                      <button
                        key={shapeId}
                        type="button"
                        onClick={() => onUpdateHardware(hw.id, { pulse_shape: shapeId })}
                        className={`p-2 rounded-[3px] border text-left transition-colors cursor-pointer ${
                          active
                            ? 'bg-[var(--accent-soft)] border-[var(--accent)] text-[var(--text-primary)]'
                            : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                        }`}
                      >
                        <div className="text-[11.5px] font-semibold">{title}</div>
                        <div className="text-[10px] opacity-80 mt-0.5">{desc}</div>
                      </button>
                    );
                  })}
                </div>

                {/* Rise & Fall Edge Controls */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-2.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-1.5">
                    <div className="flex items-center justify-between text-[11.5px]">
                      <span className="font-medium text-[var(--text-primary)]">
                        上升沿时长 t_rise
                      </span>
                      <div className="flex items-center gap-1 font-mono-num">
                        <input
                          type="number"
                          min={0}
                          step={1}
                          value={riseNs}
                          onChange={(e) => {
                            const v = Math.max(0, Number(e.target.value));
                            onUpdateHardware(hw.id, {
                              rise_ns: v,
                              ...(pulseShape !== 'rectangular' && pulseShape !== 'trapezoid' && v > 0
                                ? { pulse_shape: 'rectangular' }
                                : {}),
                            });
                          }}
                          className="w-16 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[10.5px] text-[var(--text-secondary)]">ns</span>
                      </div>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={edgeSliderMax}
                      step={hw.platform.startsWith('superconducting') ? 0.5 : 2}
                      value={Math.min(edgeSliderMax, riseNs)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        onUpdateHardware(hw.id, {
                          rise_ns: v,
                          ...(pulseShape !== 'rectangular' && pulseShape !== 'trapezoid' && v > 0
                            ? { pulse_shape: 'rectangular' }
                            : {}),
                        });
                      }}
                      className="mac-slider w-full"
                    />
                  </div>

                  <div className="p-2.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-1.5">
                    <div className="flex items-center justify-between text-[11.5px]">
                      <span className="font-medium text-[var(--text-primary)]">
                        下降沿时长 t_fall
                      </span>
                      <div className="flex items-center gap-1 font-mono-num">
                        <input
                          type="number"
                          min={0}
                          step={1}
                          value={fallNs}
                          onChange={(e) => {
                            const v = Math.max(0, Number(e.target.value));
                            onUpdateHardware(hw.id, {
                              fall_ns: v,
                              ...(pulseShape !== 'rectangular' && pulseShape !== 'trapezoid' && v > 0
                                ? { pulse_shape: 'rectangular' }
                                : {}),
                            });
                          }}
                          className="w-16 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[10.5px] text-[var(--text-secondary)]">ns</span>
                      </div>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={edgeSliderMax}
                      step={hw.platform.startsWith('superconducting') ? 0.5 : 2}
                      value={Math.min(edgeSliderMax, fallNs)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        onUpdateHardware(hw.id, {
                          fall_ns: v,
                          ...(pulseShape !== 'rectangular' && pulseShape !== 'trapezoid' && v > 0
                            ? { pulse_shape: 'rectangular' }
                            : {}),
                        });
                      }}
                      className="mac-slider w-full"
                    />
                  </div>
                </div>

                {pulseShape !== 'rectangular' && pulseShape !== 'trapezoid' && (
                  <div className="text-[11px] text-[var(--text-secondary)] flex items-center justify-between bg-[var(--bg-subtle)] px-2.5 py-1.5 rounded-[3px] border border-[var(--border-hairline)]">
                    <span>
                      当前为全窗口调制包络 ({pulseShape})；拖动上方滑条会自动切换为「平顶余弦边沿」以启用独立上升/下降沿。
                    </span>
                  </div>
                )}
              </div>

              {/* Card C: 3-Level Transmon Anharmonicity & DRAG Quadrature Control */}
              <div className="studio-card p-4 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-[13px] font-bold text-[var(--text-primary)]">
                      三能级非谐性 α/2π 与 DRAG 正交导数抑漏脉冲
                    </div>
                    <div className="text-[11px] text-[var(--text-secondary)] mt-0.5">
                      注入正交分量 Ω_Q(t) = −β · dΩ_I/dt / α 抵消 |1⟩→|2⟩ 谱泄漏与 AC-Stark 相移（0 MHz 表示理想二能级）
                    </div>
                  </div>
                  {Math.abs(anharmMhz) >= 1 && (
                    <span className="px-2 py-0.5 rounded-[2px] bg-amber-500/12 border border-amber-500/40 text-amber-400 font-mono-num text-[10.5px] shrink-0">
                      估计 |2⟩ 泄漏: {(driveMetrics.estLeakWithDrag * 100).toFixed(3)}%
                      {Math.abs(dragBeta) > 0.01 &&
                        ` (无DRAG: ${(driveMetrics.estLeakNoDrag * 100).toFixed(2)}%)`}
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="p-2.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-1.5">
                    <div className="flex items-center justify-between text-[11.5px]">
                      <span className="font-medium text-[var(--text-primary)]">非谐性 α/2π</span>
                      <div className="flex items-center gap-1 font-mono-num">
                        <input
                          type="number"
                          step={10}
                          value={anharmMhz}
                          onChange={(e) =>
                            onUpdateHardware(hw.id, { anharmonicity_mhz: Number(e.target.value) })
                          }
                          className="w-20 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right text-[12px] outline-none focus:border-[var(--accent)]"
                        />
                        <span className="text-[10.5px] text-[var(--text-secondary)]">MHz</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1 text-[10px] font-mono-num">
                      {[
                        ['理想 2能级', 0],
                        ['Transmon -240', -240],
                        ['弱非谐 -180', -180],
                      ].map(([lbl, val]) => (
                        <button
                          key={String(lbl)}
                          type="button"
                          onClick={() =>
                            onUpdateHardware(hw.id, {
                              anharmonicity_mhz: Number(val),
                              ...(Number(val) !== 0 && !hw.drag_beta ? { drag_beta: 0.5 } : {}),
                            })
                          }
                          className="px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-hairline)] hover:border-[var(--accent)] text-[var(--text-secondary)] cursor-pointer"
                        >
                          {lbl}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="p-2.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] space-y-1.5">
                    <div className="flex items-center justify-between text-[11.5px]">
                      <span className="font-medium text-[var(--text-primary)]">DRAG 系数 β</span>
                      <input
                        type="number"
                        step={0.05}
                        min={-2}
                        max={2}
                        value={dragBeta}
                        onChange={(e) =>
                          onUpdateHardware(hw.id, { drag_beta: Number(e.target.value) })
                        }
                        className="w-20 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                      />
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1.2}
                      step={0.05}
                      value={Math.max(0, Math.min(1.2, dragBeta))}
                      onChange={(e) =>
                        onUpdateHardware(hw.id, { drag_beta: Number(e.target.value) })
                      }
                      className="mac-slider w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Card D: Global Decoherence, ZZ & Asymmetric Readout Confusion Matrix */}
              <div className="studio-card p-4 space-y-3">
                <div className="text-[13px] font-bold text-[var(--text-primary)]">
                  器件退相干、ZZ 串扰与非对称读出混淆矩阵
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  {(
                    [
                      ['T1 弛豫 (μs)', 't1_us', 10],
                      ['T2* 退相干 (μs)', 't2_us', 10],
                      ['1Q 门退极化', 'depol_1q', 0.0005],
                      ['2Q 门退极化', 'depol_2q', 0.001],
                      ['静态 ZZ (kHz)', 'zz_khz', 1],
                      ['对称读出基准', 'readout_err', 0.002],
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
                        className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                      />
                    </div>
                  ))}
                  <div>
                    <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                      P(1|0) 基态误判
                    </label>
                    <input
                      type="number"
                      step={0.002}
                      min={0}
                      max={0.49}
                      placeholder={`默认 ${hw.readout_err}`}
                      value={hw.readout_p1_given_0 ?? ''}
                      onChange={(e) =>
                        onUpdateHardware(hw.id, {
                          readout_p1_given_0:
                            e.target.value === '' ? null : Math.max(0, Number(e.target.value)),
                        })
                      }
                      className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                    />
                  </div>
                  <div>
                    <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                      P(0|1) 激发弛豫误判
                    </label>
                    <input
                      type="number"
                      step={0.002}
                      min={0}
                      max={0.49}
                      placeholder={`默认 ${hw.readout_err}`}
                      value={hw.readout_p0_given_1 ?? ''}
                      onChange={(e) =>
                        onUpdateHardware(hw.id, {
                          readout_p0_given_1:
                            e.target.value === '' ? null : Math.max(0, Number(e.target.value)),
                        })
                      }
                      className="w-full px-2 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] font-mono-num text-[12px] outline-none focus:border-[var(--accent)]"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Right 6 Cols: Live Oscilloscope Envelope Preview + Inheritance Hierarchy */}
            <div className="xl:col-span-6 space-y-4">
              {/* 4 Derived KPI Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                <div className="studio-kpi p-2.5">
                  <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)]">
                    π 脉冲时长 (X / Y)
                  </div>
                  <div className="text-[17px] font-bold font-mono-num text-[var(--accent)] mt-0.5">
                    {driveMetrics.tPi.toFixed(1)} ns
                  </div>
                  <div className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                    峰值 {driveMetrics.omegaPi.toFixed(2)} MHz
                  </div>
                </div>

                <div className="studio-kpi p-2.5">
                  <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)]">
                    π/2 脉冲时长 (SX / H)
                  </div>
                  <div className="text-[17px] font-bold font-mono-num text-emerald-500 mt-0.5">
                    {driveMetrics.tPi2.toFixed(1)} ns
                  </div>
                  <div className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                    峰值 {driveMetrics.omegaPi2.toFixed(2)} MHz
                  </div>
                </div>

                <div className="studio-kpi p-2.5">
                  <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)]">
                    包络面积因子 κ
                  </div>
                  <div className="text-[17px] font-bold font-mono-num text-[var(--text-primary)] mt-0.5">
                    {driveMetrics.kappa.toFixed(3)}
                  </div>
                  <div className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                    ∫f(t)dt / (f_max·T)
                  </div>
                </div>

                <div className="studio-kpi p-2.5">
                  <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)]">
                    三级拉比覆盖状态
                  </div>
                  <div className="text-[15px] font-bold font-mono-num text-amber-500 mt-0.5">
                    {qubitOverrideCount} 比特 · {opOverrideCount} 门
                  </div>
                  <button
                    type="button"
                    onClick={() => onTabChange('qubits')}
                    className="text-[10px] font-mono-num text-[var(--accent)] hover:underline cursor-pointer"
                  >
                    管理逐比特覆盖 →
                  </button>
                </div>
              </div>

              {/* Live Physical Pulse Waveform Preview */}
              <div className="studio-card flex flex-col overflow-hidden h-[340px]">
                <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0">
                  <span className="text-[12.5px] font-semibold text-[var(--text-primary)]">
                    单比特物理脉冲波形实时预览 Ω(t) / 2π
                  </span>
                  <div className="flex items-center gap-3 text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-[2px] bg-[#2F81F7]" />
                      π 脉冲 (X/Y)
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-[2px] bg-[#10B981]" />
                      π/2 脉冲 (SX/H)
                    </span>
                    {(pulseShape === 'rectangular' || pulseShape === 'trapezoid') &&
                      (riseNs > 0 || fallNs > 0) && (
                        <span className="text-amber-500">
                          阴影: 上升/下降沿 ({riseNs}ns / {fallNs}ns)
                        </span>
                      )}
                  </div>
                </div>
                <div className="flex-1 min-h-0 p-1.5">
                  <PlotlyChart
                    data={[
                      {
                        x: driveMetrics.xs,
                        y: driveMetrics.yPi,
                        type: 'scatter',
                        mode: 'lines',
                        fill: 'tozeroy',
                        fillcolor: 'rgba(47, 129, 247, 0.14)',
                        name: 'π 脉冲 Ω(t)/2π',
                        line: { color: '#2F81F7', width: 2.4 },
                        hovertemplate: 't=%{x:.1f} ns · Ω/2π=%{y:.3f} MHz<extra>π 脉冲</extra>',
                      },
                      {
                        x: driveMetrics.xs,
                        y: driveMetrics.yPi2,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'π/2 脉冲 Ω_I(t)/2π',
                        line: { color: '#10B981', width: 2.1, dash: 'dash' },
                        hovertemplate: 't=%{x:.1f} ns · Ω_I/2π=%{y:.3f} MHz<extra>π/2 脉冲</extra>',
                      },
                      ...(driveMetrics.hasDragTrace
                        ? [
                            {
                              x: driveMetrics.xs,
                              y: driveMetrics.yDragQ,
                              type: 'scatter',
                              mode: 'lines',
                              name: 'DRAG 正交导数 Ω_Q(t)/2π',
                              line: { color: '#F59E0B', width: 2.0, dash: 'dot' },
                              hovertemplate:
                                't=%{x:.1f} ns · Ω_Q/2π=%{y:.3f} MHz<extra>DRAG 正交</extra>',
                            },
                          ]
                        : []),
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      showlegend: false,
                      hovermode: 'x unified',
                      margin: { l: 54, r: 20, t: 16, b: 40 },
                      xaxis: {
                        title: { text: '脉冲时间 t (ns)', font: { size: 11, color: mutedColor } },
                        range: [0, driveMetrics.tMax],
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                      yaxis: {
                        title: { text: '双正交脉冲包络 Ω_I, Ω_Q / 2π (MHz)', font: { size: 11, color: mutedColor } },
                        range: [driveMetrics.minY, driveMetrics.maxY],
                        gridcolor: gridColor,
                        zeroline: driveMetrics.hasDragTrace,
                        zerolinecolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                      shapes:
                        (pulseShape === 'rectangular' || pulseShape === 'trapezoid') &&
                        (driveMetrics.effRise > 0 || driveMetrics.effFall > 0)
                          ? [
                              ...(driveMetrics.effRise > 0
                                ? [
                                    {
                                      type: 'rect',
                                      x0: 0,
                                      x1: driveMetrics.effRise,
                                      y0: 0,
                                      y1: driveMetrics.omegaPi * 1.12,
                                      fillcolor: 'rgba(245, 158, 11, 0.12)',
                                      line: { width: 0 },
                                    },
                                  ]
                                : []),
                              ...(driveMetrics.effFall > 0
                                ? [
                                    {
                                      type: 'rect',
                                      x0: Math.max(0, driveMetrics.tPi - driveMetrics.effFall),
                                      x1: driveMetrics.tPi,
                                      y0: 0,
                                      y1: driveMetrics.omegaPi * 1.12,
                                      fillcolor: 'rgba(245, 158, 11, 0.12)',
                                      line: { width: 0 },
                                    },
                                  ]
                                : []),
                            ]
                          : [],
                    }}
                  />
                </div>
              </div>

              {/* Explanation Banner of 3-Tier Inheritance */}
              <div className="studio-card p-3.5 text-[11.5px] text-[var(--text-secondary)] space-y-1.5 leading-relaxed">
                <div className="font-semibold text-[var(--text-primary)]">
                  三级拉比频率优先级规则（Single Source of Truth）：
                </div>
                <div>
                  1. <strong className="text-[var(--accent)]">单个量子门覆盖</strong>：在「量子线路 → 原理图设计」选中任意单比特门，可在右侧属性栏为该门单独指定拉比频率 <code className="px-1 bg-[var(--bg-subtle)] rounded">op.rabi_mhz</code>；
                </div>
                <div>
                  2. <strong className="text-emerald-500">逐比特寄存器覆盖</strong>：在「量子寄存器与逐比特参数」表或标定实验一键写回设置 <code className="px-1 bg-[var(--bg-subtle)] rounded">qubit.rabi_mhz</code>，使该比特所有单比特门按实测拉比频率演化；
                </div>
                <div>
                  3. <strong className="text-[var(--text-primary)]">器件全局默认</strong>：未设置覆盖的比特与门自动继承本页的器件默认配置。
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* TAB 2: FULL QUBIT REGISTER PARAMETER SPREADSHEET                    */}
      {/* =================================================================== */}
      {activeTab === 'qubits' && (
        <div className="flex-1 min-h-0 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 shrink-0">
            <div>
              <div className="text-[13.5px] font-bold text-[var(--text-primary)]">
                量子寄存器逐比特物理参数表（留空表示继承 {hw.name} 默认值）
              </div>
              <div className="text-[11px] text-[var(--text-secondary)]">
                支持逐比特覆盖拉比频率 Ω_R、跃迁频率 f_q、相干时间 T1 / T2*、门退极化与读出混淆误差，或一键跳转门表征实验标定。
              </div>
            </div>
            <button
              type="button"
              onClick={onAddQubit}
              className="h-7 px-3 rounded-[3px] studio-btn-primary text-[11.5px] font-semibold flex items-center gap-1 cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              添加量子比特
            </button>
          </div>

          <div className="flex-1 min-h-0 studio-card overflow-auto">
            <table className="w-full text-[11.5px] font-mono-num border-collapse">
              <thead className="sticky top-0 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] text-[10.5px] text-[var(--text-secondary)] text-left z-10">
                <tr>
                  <th className="py-2 px-2.5">比特</th>
                  <th className="py-2 px-2.5">初态</th>
                  <th className="py-2 px-2.5">拉比 Ω_R (MHz)</th>
                  <th className="py-2 px-2.5">推导 t_π</th>
                  <th className="py-2 px-2.5">频率 f_q (GHz)</th>
                  <th className="py-2 px-2.5">T1 (μs)</th>
                  <th className="py-2 px-2.5">T2* (μs)</th>
                  <th className="py-2 px-2.5">1Q 退极化</th>
                  <th className="py-2 px-2.5">读出误差</th>
                  <th className="py-2 px-2.5">操作 / 标定</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-hairline)]">
                {project.qubits.slice(0, 120).map((q) => {
                  const effOmega = q.rabi_mhz ?? rabiMhz;
                  const effTPi =
                    q.rabi_mhz != null || driveMode === 'fixed_rabi'
                      ? rabiPulseDurationNs(Math.PI, effOmega, pulseShape, riseNs, fallNs)
                      : hw.t_1q_ns;
                  const hasAnyOverride =
                    q.rabi_mhz != null ||
                    q.t1_us != null ||
                    q.t2_us != null ||
                    q.freq_ghz != null ||
                    q.depol_1q != null ||
                    q.readout_err != null;

                  return (
                    <tr key={q.id} className="hover:bg-black/[0.02] dark:hover:bg-white/[0.03]">
                      <td className="py-1.5 px-2.5">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="w-2.5 h-2.5 rounded-full shrink-0"
                            style={{ backgroundColor: q.color }}
                          />
                          <input
                            value={q.name}
                            onChange={(e) => onUpdateQubit(q.id, { name: e.target.value })}
                            className="w-16 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-semibold outline-none focus:border-[var(--accent)]"
                          />
                        </div>
                      </td>

                      <td className="py-1.5 px-2.5">
                        <select
                          value={q.init_state}
                          onChange={(e) =>
                            onUpdateQubit(q.id, { init_state: e.target.value as InitStateType })
                          }
                          className="px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] outline-none"
                        >
                          {(['|0>', '|1>', '|+>', '|->', '|+i>', '|-i>'] as InitStateType[]).map(
                            (st) => (
                              <option key={st} value={st}>
                                {st.replace('>', '⟩')}
                              </option>
                            )
                          )}
                        </select>
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={0.1}
                          min={0.01}
                          placeholder={`继承 ${rabiMhz}`}
                          value={q.rabi_mhz ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              rabi_mhz:
                                e.target.value === '' ? null : Math.max(0.01, Number(e.target.value)),
                            })
                          }
                          className={`w-24 px-2 py-0.5 rounded-[2px] border text-right outline-none focus:border-[var(--accent)] ${
                            q.rabi_mhz != null
                              ? 'bg-amber-500/10 border-amber-500/60 text-amber-400 font-semibold'
                              : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)]'
                          }`}
                        />
                      </td>

                      <td className="py-1.5 px-2.5 text-[var(--accent)]">
                        {effTPi.toFixed(1)} ns
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={0.01}
                          placeholder={hw.platform === 'neutral_atom' ? '6.834' : (5 + 0.12 * q.id).toFixed(2)}
                          value={q.freq_ghz ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              freq_ghz: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          className="w-20 px-2 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-right outline-none focus:border-[var(--accent)]"
                        />
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={5}
                          placeholder={`继承 ${hw.t1_us}`}
                          value={q.t1_us ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              t1_us: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          className={`w-22 px-2 py-0.5 rounded-[2px] border text-right outline-none focus:border-[var(--accent)] ${
                            q.t1_us != null
                              ? 'bg-amber-500/10 border-amber-500/60'
                              : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)]'
                          }`}
                        />
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={5}
                          placeholder={`继承 ${hw.t2_us}`}
                          value={q.t2_us ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              t2_us: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          className={`w-22 px-2 py-0.5 rounded-[2px] border text-right outline-none focus:border-[var(--accent)] ${
                            q.t2_us != null
                              ? 'bg-amber-500/10 border-amber-500/60'
                              : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)]'
                          }`}
                        />
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={0.0005}
                          placeholder={`继承 ${hw.depol_1q}`}
                          value={q.depol_1q ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              depol_1q: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          className="w-22 px-2 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-right outline-none focus:border-[var(--accent)]"
                        />
                      </td>

                      <td className="py-1.5 px-2.5">
                        <input
                          type="number"
                          step={0.002}
                          placeholder={`继承 ${hw.readout_err}`}
                          value={q.readout_err ?? ''}
                          onChange={(e) =>
                            onUpdateQubit(q.id, {
                              readout_err: e.target.value === '' ? null : Number(e.target.value),
                            })
                          }
                          className="w-22 px-2 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-right outline-none focus:border-[var(--accent)]"
                        />
                      </td>

                      <td className="py-1.5 px-2.5">
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => onJumpToCalibrate(q.id, 'rabi')}
                            title={`对 ${q.name} 运行 Rabi 拉比频率标定`}
                            className="px-1.5 py-0.5 rounded-[2px] bg-[var(--accent-soft)] text-[var(--accent)] text-[10px] font-sans font-semibold cursor-pointer"
                          >
                            Rabi
                          </button>
                          <button
                            type="button"
                            onClick={() => onJumpToCalibrate(q.id, 'echo')}
                            title={`对 ${q.name} 运行 Hahn Echo / CPMG 动态解耦相干标定`}
                            className="px-1.5 py-0.5 rounded-[2px] bg-emerald-500/12 text-emerald-500 text-[10px] font-sans font-semibold cursor-pointer"
                          >
                            Echo
                          </button>
                          {hasAnyOverride && (
                            <button
                              type="button"
                              onClick={() =>
                                onUpdateQubit(q.id, {
                                  rabi_mhz: null,
                                  t1_us: null,
                                  t2_us: null,
                                  freq_ghz: null,
                                  depol_1q: null,
                                  readout_err: null,
                                })
                              }
                              title="清除该比特所有覆盖，恢复继承器件默认"
                              className="p-1 rounded-[2px] text-[var(--text-secondary)] hover:text-amber-500 cursor-pointer"
                            >
                              <RotateCcw className="w-3 h-3" />
                            </button>
                          )}
                          {project.qubits.length > 1 && (
                            <button
                              type="button"
                              onClick={() => onDeleteQubit(q.id)}
                              title="删除此量子比特"
                              className="p-1 rounded-[2px] text-[var(--text-secondary)] hover:text-red-500 cursor-pointer"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* TAB 3: COUPLING TOPOLOGY & NEUTRAL ATOM TWEEZER ARRAY               */}
      {/* =================================================================== */}
      {activeTab === 'topology' && (
        <div className="flex-1 min-h-0 overflow-y-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-7 studio-card p-4 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-[13px] font-bold text-[var(--text-primary)]">
                    二维比特排布与耦合连接拓扑编辑器
                  </div>
                  <div className="text-[11px] text-[var(--text-secondary)]">
                    支持拖拽原子光镊坐标、一键生成晶格阵列或手动点击比特对切换耦合边
                  </div>
                </div>
                <select
                  value={hw.topology_mode}
                  onChange={(e) =>
                    onUpdateHardware(hw.id, {
                      topology_mode: e.target.value as HardwareProfileModel['topology_mode'],
                    })
                  }
                  className="px-2.5 py-1 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
                >
                  <option value="rydberg_radius">里德堡阻塞半径 (d ≤ R_b)</option>
                  <option value="all">全连接耦合 (All-to-All)</option>
                  <option value="linear">一维最近邻链 (1D Chain)</option>
                  <option value="custom">自定义手动连边 (Custom)</option>
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
                <div className="pt-2 space-y-1">
                  <div className="flex justify-between text-[12px]">
                    <span className="text-[var(--text-secondary)]">里德堡偶极阻塞半径 R_b</span>
                    <span className="font-mono-num font-semibold text-[var(--accent)]">
                      {hw.blockade_radius_um.toFixed(1)} μm
                    </span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={18}
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

            <div className="lg:col-span-5 studio-card p-4 flex flex-col max-h-[460px]">
              <div className="text-[13px] font-bold text-[var(--text-primary)] mb-2">
                二维光镊空间坐标表 (μm)
              </div>
              <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
                {project.qubits.slice(0, 64).map((q) => (
                  <div
                    key={q.id}
                    className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11.5px] font-mono-num"
                  >
                    <span className="flex items-center gap-1.5 font-semibold">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: q.color }}
                      />
                      {q.name}
                    </span>
                    <div className="flex items-center gap-2">
                      <label className="text-[10.5px] text-[var(--text-secondary)]">X</label>
                      <input
                        type="number"
                        step={0.5}
                        value={q.pos_x}
                        onChange={(e) => onUpdateQubit(q.id, { pos_x: Number(e.target.value) })}
                        className="w-16 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right"
                      />
                      <label className="text-[10.5px] text-[var(--text-secondary)]">Y</label>
                      <input
                        type="number"
                        step={0.5}
                        value={q.pos_y}
                        onChange={(e) => onUpdateQubit(q.id, { pos_y: Number(e.target.value) })}
                        className="w-16 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-input)] text-right"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* TAB 4: HARDWARE PLATFORMS & CUSTOM NATIVE GATE SET                  */}
      {/* =================================================================== */}
      {activeTab === 'platforms' && (
        <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            {project.hardware_profiles.map((hp) => {
              const active = hp.id === project.active_hardware_id;
              return (
                <div
                  key={hp.id}
                  onClick={() => onSelectHardware(hp.id)}
                  className={`studio-card p-3.5 cursor-pointer transition-all space-y-2 ${
                    active ? 'border-[var(--accent)] ring-1 ring-[var(--accent)]' : 'hover:border-[var(--border-input)]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[13px] font-bold text-[var(--text-primary)]">{hp.name}</span>
                    {active && (
                      <span className="px-1.5 py-0.5 rounded-[2px] bg-[var(--accent)] text-white text-[10px] font-mono-num font-bold flex items-center gap-0.5">
                        <Check className="w-3 h-3" />
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-1.5 text-[11px] font-mono-num text-[var(--text-secondary)] pt-1">
                    <div>Ω_R: <strong className="text-[var(--text-primary)]">{hp.rabi_mhz ?? 2.5} MHz</strong></div>
                    <div>t_1q: <strong className="text-[var(--text-primary)]">{hp.t_1q_ns} ns</strong></div>
                    <div>T1: <strong className="text-[var(--text-primary)]">{hp.t1_us} μs</strong></div>
                    <div>T2*: <strong className="text-[var(--text-primary)]">{hp.t2_us} μs</strong></div>
                    <div>e_1q: <strong className="text-[var(--text-primary)]">{(hp.depol_1q * 100).toFixed(2)}%</strong></div>
                    <div>e_2q: <strong className="text-[var(--text-primary)]">{(hp.depol_2q * 100).toFixed(2)}%</strong></div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="studio-card p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[13px] font-bold text-[var(--text-primary)]">
                  自定义实验原生门集 (Custom Native Gate Library)
                </div>
                <div className="text-[11px] text-[var(--text-secondary)]">
                  开启「作为实验原生门」后，编译综合器与 XEB 门标定可直接调用该自定义门作为底层硬件原语。
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenNewGateModal}
                className="h-7 px-3 rounded-[3px] studio-btn-primary text-[11.5px] font-semibold flex items-center gap-1 cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                新建自定义门
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2.5">
              {project.custom_gates.map((cg) => (
                <div
                  key={cg.id}
                  className="p-3 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-between gap-2"
                >
                  <div className="min-w-0">
                    <div className="text-[12.5px] font-mono-num font-bold text-[var(--text-primary)] truncate">
                      {cg.display_label || cg.name}{' '}
                      <span className="text-[10.5px] font-normal text-[var(--text-secondary)]">
                        ({cg.num_qubits}Q · {cg.duration_ns} ns)
                      </span>
                    </div>
                    <div className="text-[10.5px] text-[var(--text-secondary)] truncate">
                      {cg.description || `定义模式: ${cg.definition_type}`}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <label className="flex items-center gap-1 text-[10.5px] text-[var(--text-secondary)] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={cg.is_native}
                        onChange={(e) => onToggleCustomGateNative(cg.id, e.target.checked)}
                        className="accent-[var(--accent)]"
                      />
                      原生
                    </label>
                    <button
                      type="button"
                      onClick={() => onEditCustomGate(cg)}
                      className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 text-[var(--text-secondary)] cursor-pointer"
                    >
                      <Pencil className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteCustomGate(cg.id)}
                      className="p-1 rounded hover:bg-red-500/15 text-[var(--text-secondary)] hover:text-red-500 cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
