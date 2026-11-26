import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowRightLeft,
  Check,
  CheckCircle2,
  CircuitBoard,
  Code2,
  Download,
  FileText,
  Gauge,
  Info,
  Play,
  ShieldCheck,
} from 'lucide-react';
import type {
  AllXyBenchmarkResponse,
  BasicCalibrationResponse,
  BenchmarkProtocol,
  CalibrationWritebackItem,
  CharacterizationCircuitSample,
  CircuitOpModel,
  GateBenchmarkResponse,
  GstBenchmarkResponse,
  ProjectModel,
  RbBenchmarkResponse,
  XebBenchmarkResponse,
} from '../types';
import { CircuitCanvas } from './CircuitCanvas';
import { PlotlyChart } from './PlotlyChart';
import { buildCircuitSvg, buildQuantikzLatex, downloadTextFile } from '../utils/exportCircuitSvg';

interface CharacterizationCircuitPanelProps {
  project: ProjectModel;
  protocolCode: string;
  title: string;
  samples: CharacterizationCircuitSample[];
  selectedSampleId?: string | null;
  onSelectSample?: (sample: CharacterizationCircuitSample) => void;
  onLoadCircuitToDesign: (ops: CircuitOpModel[], label: string) => void;
}

const CharacterizationCircuitPanel: React.FC<CharacterizationCircuitPanelProps> = ({
  project,
  protocolCode,
  title,
  samples,
  selectedSampleId,
  onSelectSample,
  onLoadCircuitToDesign,
}) => {
  const [localId, setLocalId] = useState<string>('');

  const activeSample = useMemo(() => {
    if (samples.length === 0) return null;
    if (selectedSampleId) {
      const ext = samples.find((s) => s.id === selectedSampleId);
      if (ext) return ext;
    }
    if (localId) {
      const loc = samples.find((s) => s.id === localId);
      if (loc) return loc;
    }
    return samples[0];
  }, [samples, selectedSampleId, localId]);

  const { focusedProject, remappedOps, usedQubitNames, gateCount, depthCols } = useMemo(() => {
    if (!activeSample) {
      return {
        focusedProject: project,
        remappedOps: [] as CircuitOpModel[],
        usedQubitNames: 'q0',
        gateCount: 0,
        depthCols: 1,
      };
    }
    const rawSet = new Set<number>();
    activeSample.operations.forEach((op) => {
      op.qubits.forEach((q) => {
        if (q >= 0 && q < project.qubits.length) rawSet.add(q);
      });
    });
    if (rawSet.size === 0 && project.qubits.length > 0) {
      rawSet.add(0);
    }
    const sortedQ = Array.from(rawSet).sort((a, b) => a - b);
    const qIndexMap = new Map<number, number>();
    const subQubits = sortedQ.map((origIdx, compactIdx) => {
      qIndexMap.set(origIdx, compactIdx);
      const spec = project.qubits[origIdx];
      return {
        ...(spec || {
          id: compactIdx,
          name: `q${origIdx}`,
          color: '#2F81F7',
          init_state: '|0>' as const,
          init_theta: 0,
          init_phi: 0,
          t1_us: null,
          t2_us: null,
          freq_ghz: null,
          depol_1q: null,
          readout_err: null,
          pos_x: 0,
          pos_y: 0,
        }),
        id: compactIdx,
      };
    });

    const mapped = activeSample.operations.map((op) => ({
      ...op,
      qubits: op.qubits.map((q) => qIndexMap.get(q) ?? 0),
    }));
    const gCnt = mapped.filter((o) => o.kind === 'gate').length;
    const maxS = mapped.reduce((m, o) => Math.max(m, o.step), 0);

    return {
      focusedProject: {
        ...project,
        qubits: subQubits,
        operations: mapped,
      },
      remappedOps: mapped,
      usedQubitNames: subQubits.map((q) => q.name).join(', '),
      gateCount: gCnt,
      depthCols: maxS + 1,
    };
  }, [activeSample, project]);

  if (!activeSample || samples.length === 0) return null;

  const safeSlug = `${protocolCode}_${activeSample.id}`.replace(/[^a-zA-Z0-9_-]+/g, '_');

  const handleExportSampleSvg = () => {
    const svg = buildCircuitSvg(focusedProject, remappedOps, `${title} · ${activeSample.label}`);
    downloadTextFile(`${safeSlug}.svg`, svg, 'image/svg+xml;charset=utf-8');
  };

  const handleExportSampleTex = () => {
    const tex = buildQuantikzLatex(focusedProject);
    downloadTextFile(`${safeSlug}_quantikz.tex`, tex, 'text/plain;charset=utf-8');
  };

  const canvasHeightCls = focusedProject.qubits.length >= 2 ? 'h-[126px]' : 'h-[92px]';

  return (
    <div className="studio-card flex flex-col overflow-hidden">
      {/* Header + Sequence Switcher Pills + Export Actions */}
      <div className="px-3 py-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-panel)] flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="inline-flex items-center gap-1 text-[11.5px] font-semibold text-[var(--text-primary)] shrink-0">
            <CircuitBoard className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span>{title}</span>
          </span>

          <div className="flex items-center gap-1 overflow-x-auto py-0.5 min-w-0">
            {samples.map((s) => {
              const isAct = s.id === activeSample.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => {
                    setLocalId(s.id);
                    onSelectSample?.(s);
                  }}
                  className={`px-2 py-0.5 rounded-[3px] text-[10.5px] font-mono-num whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                    isAct
                      ? 'bg-[var(--accent)] text-white font-semibold'
                      : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border-hairline)]'
                  }`}
                >
                  {s.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={handleExportSampleSvg}
            title="导出当前标定门序列矢量原理图 (.svg)"
            className="h-6 px-2 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[10.5px] font-mono-num text-[var(--text-secondary)] flex items-center gap-1 cursor-pointer transition-colors"
          >
            <Download className="w-3 h-3" />
            <span>线路 SVG</span>
          </button>
          <button
            type="button"
            onClick={handleExportSampleTex}
            title="导出当前标定序列的论文级 LaTeX Quantikz 源码 (.tex)"
            className="h-6 px-2 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[10.5px] font-mono-num text-[var(--text-secondary)] flex items-center gap-1 cursor-pointer transition-colors"
          >
            <Code2 className="w-3 h-3" />
            <span>Quantikz .tex</span>
          </button>
          <button
            type="button"
            onClick={() =>
              onLoadCircuitToDesign(activeSample.operations, `${safeSlug}.qproj`)
            }
            title="将此标定门序列发送到「量子线路」工作区进行脉冲级仿真或编辑"
            className="h-6 px-2 rounded-[3px] bg-[var(--accent-soft)] hover:bg-[var(--accent)] hover:text-white text-[var(--accent)] border border-[var(--accent)]/35 text-[10.5px] font-semibold flex items-center gap-1 cursor-pointer transition-colors"
          >
            <ArrowRightLeft className="w-3 h-3" />
            <span>发送到量子线路</span>
          </button>
        </div>
      </div>

      {/* Subtitle + Gate Count Telemetry */}
      <div className="px-3 py-1 bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2 text-[10.5px] font-mono-num text-[var(--text-secondary)]">
        <span className="truncate">{activeSample.subtitle || activeSample.label}</span>
        <span className="shrink-0">
          目标比特: <strong className="text-[var(--text-primary)]">{usedQubitNames}</strong> · {gateCount} 门 · {depthCols} 时间步
        </span>
      </div>

      {/* Compact Read-Only Circuit Schematic Canvas */}
      <div className={`${canvasHeightCls} flex flex-col overflow-hidden`}>
        <CircuitCanvas
          project={focusedProject}
          operationsOverride={remappedOps}
          readOnly
          compact
          selection={{ type: 'none' }}
          activeToolGate={null}
          onSelect={() => {}}
          onPickGateTool={() => {}}
          onPlaceGate={() => {}}
          onDeleteOp={() => {}}
          onAddQubit={() => {}}
        />
      </div>
    </div>
  );
};

interface BenchmarkWorkspaceProps {
  project: ProjectModel;
  activeProtocol: BenchmarkProtocol;
  onSwitchProtocol: (p: BenchmarkProtocol) => void;
  response: GateBenchmarkResponse | null;
  darkMode: boolean;
  onRunBenchmark: () => void;
  onLoadCircuitToDesign: (ops: CircuitOpModel[], label: string) => void;
  onApplyWriteback?: (item: CalibrationWritebackItem) => void;
  isBusy: boolean;
}

export const BenchmarkWorkspace: React.FC<BenchmarkWorkspaceProps> = ({
  project,
  activeProtocol,
  onSwitchProtocol,
  response,
  darkMode,
  onRunBenchmark,
  onLoadCircuitToDesign,
  onApplyWriteback,
  isBusy,
}) => {
  const [allxyUnit, setAllxyUnit] = useState<'z' | 'p1'>('z');
  const [selectedAllxyIdx, setSelectedAllxyIdx] = useState<number>(7); // default #8 X90·Y90
  const [selectedGstGateIdx, setSelectedGstGateIdx] = useState(1); // default Gx90
  const [selectedGstSampleId, setSelectedGstSampleId] = useState<string | null>('gst_Gx90_l1');
  const [appliedWbKey, setAppliedWbKey] = useState<string | null>(null);

  const textColor = darkMode ? '#E6EDF3' : '#0F172A';
  const mutedColor = darkMode ? '#8B96A5' : '#475569';
  const gridColor = darkMode ? 'rgba(255,255,255,0.07)' : 'rgba(15,23,42,0.08)';
  const plotBg = 'rgba(0,0,0,0)';

  const matchingResp = response && response.protocol === activeProtocol ? response : null;

  const handleExportCsv = () => {
    if (!matchingResp) return;
    if (matchingResp.protocol === 'rb') {
      const r = matchingResp;
      const hasIrb = Boolean(r.interleaved);
      const header = [
        'clifford_length_m',
        'ref_survival_mean',
        'ref_survival_stderr',
        ...(hasIrb ? [`irb_${r.interleaved!.gate_name}_mean`, `irb_${r.interleaved!.gate_name}_stderr`] : []),
      ].join(',');
      const rows = r.lengths.map((m, i) =>
        [
          m,
          r.mean_survival[i],
          r.std_survival[i],
          ...(hasIrb ? [r.interleaved!.mean_survival[i], r.interleaved!.std_survival[i]] : []),
        ].join(',')
      );
      downloadTextFile(`rb_${r.target_label.replace(/\s+/g, '')}.csv`, '\uFEFF' + [header, ...rows].join('\n'), 'text/csv;charset=utf-8');
    } else if (matchingResp.protocol === 'allxy') {
      const r = matchingResp;
      const header = 'index,sequence,plateau,ideal_Z,measured_Z,stderr_Z,ideal_P1,measured_P1,residual_Z';
      const rows = r.sequences.map((s) =>
        [
          s.index,
          s.label,
          s.plateau,
          s.ideal_z,
          s.measured_z,
          s.std_z,
          s.ideal_p1,
          s.measured_p1,
          s.residual_z,
        ].join(',')
      );
      downloadTextFile(`allxy_${r.target_label}.csv`, '\uFEFF' + [header, ...rows].join('\n'), 'text/csv;charset=utf-8');
    } else if (matchingResp.protocol === 'gst') {
      const r = matchingResp;
      const header = 'gate_id,label,avg_gate_fidelity,process_fidelity,gate_infidelity,diamond_distance,unitarity,coherent_ratio';
      const rows = r.gates.map((g) =>
        [
          g.gate_id,
          `"${g.label}"`,
          g.avg_gate_fidelity,
          g.process_fidelity,
          g.gate_infidelity,
          g.diamond_distance,
          g.unitarity,
          g.coherent_ratio,
        ].join(',')
      );
      downloadTextFile(`gst_${r.target_label}.csv`, '\uFEFF' + [header, ...rows].join('\n'), 'text/csv;charset=utf-8');
    } else if (matchingResp.protocol === 'xeb') {
      const r = matchingResp;
      const header = 'cycle_depth_m,mean_xeb,stderr_xeb';
      const rows = r.lengths.map((m, i) => [m, r.mean_xeb[i], r.std_xeb[i]].join(','));
      downloadTextFile(`xeb_${r.entangler_name}.csv`, '\uFEFF' + [header, ...rows].join('\n'), 'text/csv;charset=utf-8');
    } else {
      const r = matchingResp;
      const header = 'x_ns,measured_p1,ideal_p1';
      const rows = r.x.map((xv, i) => [xv, r.measured[i], r.ideal[i]].join(','));
      downloadTextFile(
        `${r.protocol}_${r.target_label}.csv`,
        '\uFEFF' + [header, ...rows].join('\n'),
        'text/csv;charset=utf-8'
      );
    }
  };

  const handleExportReport = () => {
    if (!matchingResp) return;
    let md = `# Quantum Studio EDA 量子比特与门表征报告 (${matchingResp.protocol.toUpperCase()})\n\n`;
    md += `- **硬件平台**: ${matchingResp.hardware_name}\n`;
    md += `- **标定对象**: ${matchingResp.target_label}\n\n`;
    if (matchingResp.protocol === 'rb') {
      md += `## Randomized Benchmarking 结果\n`;
      md += `- 平均单 Clifford 保真度 $F_{\\text{Clifford}}$: **${(matchingResp.clifford_fidelity * 100).toFixed(4)}%**\n`;
      md += `- 单 Clifford 错误率 EPC: **${matchingResp.epc.toExponential(3)}**\n`;
      md += `- 退极化衰减因子 $p$: **${matchingResp.depolarizing_p.toFixed(6)}**\n`;
      md += `- $T_1/T_2$ 相干极限 EPC 下界: **${matchingResp.coherence_epc_floor.toExponential(3)}**\n`;
      if (matchingResp.interleaved) {
        md += `- 交织门 (${matchingResp.interleaved.gate_name}) 保真度: **${(matchingResp.interleaved.gate_fidelity * 100).toFixed(4)}%** (门错误率 ${matchingResp.interleaved.gate_error.toExponential(3)})\n`;
      }
    } else if (matchingResp.protocol === 'allxy') {
      md += `## AllXY 21 序列脉冲标定诊断\n`;
      md += `- RMS 阶梯残差 $\\Delta\\langle Z \\rangle$: **${matchingResp.rms_error_z.toFixed(5)}**\n`;
      md += `- 估计拉比幅度偏差: **${matchingResp.estimated_amp_error_pct.toFixed(2)}%**\n`;
      md += `- 自动诊断结论: ${matchingResp.diagnosis}\n`;
    } else if (matchingResp.protocol === 'gst') {
      md += `## Gate Set Tomography (GST) 门集层析汇总\n`;
      md += `- 门集平均保真度: **${(matchingResp.mean_gate_fidelity * 100).toFixed(4)}%**\n`;
      md += `- 最大半钻石距离 $\\frac{1}{2}\\|\\Delta\\mathcal{E}\\|_\\diamond$: **${matchingResp.max_diamond_distance.toFixed(5)}**\n`;
      md += `- 平均幺正度 Unitarity: **${matchingResp.mean_unitarity.toFixed(5)}**\n`;
      md += `- SPAM 初态制备保真度: **${(matchingResp.spam_prep_fidelity * 100).toFixed(2)}%** | 读出保真度: **${(matchingResp.spam_readout_fidelity * 100).toFixed(2)}%**\n\n`;
      md += `| 门标识 | 名称 | 平均保真度 | 过程保真度 | 半钻石距离 | 幺正度 u | 相干误差占比 |\n`;
      md += `|---|---|---|---|---|---|---|\n`;
      matchingResp.gates.forEach((g) => {
        md += `| ${g.gate_id} | ${g.label} | ${(g.avg_gate_fidelity * 100).toFixed(3)}% | ${(g.process_fidelity * 100).toFixed(3)}% | ${g.diamond_distance.toFixed(5)} | ${g.unitarity.toFixed(5)} | ${(g.coherent_ratio * 100).toFixed(1)}% |\n`;
      });
    } else if (matchingResp.protocol === 'xeb') {
      md += `## 双比特线性交叉熵基准 (2Q Linear XEB)\n`;
      md += `- 待测纠缠门: **${matchingResp.entangler_name}**\n`;
      md += `- 单周期极化度 $f_{\\text{cycle}}$: **${(matchingResp.cycle_polarization * 100).toFixed(4)}%**\n`;
      md += `- 单周期泡利错误率 $e_{\\text{cycle}}$: **${(matchingResp.cycle_pauli_error * 100).toFixed(4)}%**\n`;
      md += `- 剥离单比特层后纠缠门保真度: **${(matchingResp.entangler_fidelity * 100).toFixed(4)}%**\n`;
    } else {
      md += `## 基础物理标定拟合结论\n`;
      md += `- ${matchingResp.summary}\n\n`;
      matchingResp.params.forEach((p) => {
        md += `- **${p.label}**: ${p.value} ${p.unit}\n`;
      });
    }
    downloadTextFile(`calibration_report_${matchingResp.protocol}.md`, md, 'text/markdown;charset=utf-8');
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-[var(--bg-app)] overflow-hidden">
      {/* Level-2 Sub-Tab Bar: Basic Calibrations (Rabi / Ramsey / T1) | Gate Benchmarks (RB / AllXY / GST / XEB) */}
      <div className="h-9 px-3 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0 select-none">
        <div className="flex items-center gap-3 h-full overflow-x-auto min-w-0">
          <span className="text-[10px] font-mono-num uppercase text-[var(--text-tertiary)] font-bold shrink-0">
            基础标定
          </span>
          {(
            [
              { id: 'rabi', label: 'Rabi 拉比振荡' },
              { id: 'ramsey', label: 'Ramsey 相干条纹' },
              { id: 't1', label: 'T1 能量弛豫' },
            ] as const
          ).map((t) => {
            const active = activeProtocol === t.id;
            return (
              <button
                key={t.id}
                onClick={() => onSwitchProtocol(t.id)}
                className={`h-full text-[12px] border-b-2 transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  active
                    ? 'border-[var(--accent)] text-[var(--text-primary)] font-semibold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {t.label}
              </button>
            );
          })}

          <div className="h-4 w-[1px] bg-[var(--border-hairline)] shrink-0 mx-0.5" />

          <span className="text-[10px] font-mono-num uppercase text-[var(--text-tertiary)] font-bold shrink-0">
            门基准层析
          </span>
          {(
            [
              { id: 'rb', label: 'RB 随机基准' },
              { id: 'allxy', label: 'AllXY 脉冲诊断' },
              { id: 'gst', label: 'GST 门集层析' },
              { id: 'xeb', label: 'XEB 交叉熵' },
            ] as const
          ).map((t) => {
            const active = activeProtocol === t.id;
            return (
              <button
                key={t.id}
                onClick={() => onSwitchProtocol(t.id)}
                className={`h-full text-[12px] border-b-2 transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  active
                    ? 'border-[var(--accent)] text-[var(--text-primary)] font-semibold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {matchingResp && (
            <>
              <button
                onClick={handleExportCsv}
                title="导出当前标定实验数据表 (.csv)"
                className="h-6 px-2 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[11px] font-mono-num text-[var(--text-secondary)] flex items-center gap-1 cursor-pointer"
              >
                <Download className="w-3 h-3" />
                CSV
              </button>
              <button
                onClick={handleExportReport}
                title="导出 Markdown 标定报告 (.md)"
                className="h-6 px-2 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[11px] font-mono-num text-[var(--text-secondary)] flex items-center gap-1 cursor-pointer"
              >
                <FileText className="w-3 h-3" />
                报告
              </button>
            </>
          )}
          <button
            onClick={onRunBenchmark}
            disabled={isBusy}
            className="h-6 px-2.5 rounded-[3px] studio-btn-primary text-[11px] font-semibold flex items-center gap-1 cursor-pointer disabled:opacity-60"
          >
            <Play className="w-3 h-3 fill-current" />
            运行实验
          </button>
        </div>
      </div>

      {/* Empty State */}
      {!matchingResp && (
        <div className="flex-1 dot-grid-canvas flex items-center justify-center p-6">
          <div className="studio-card max-w-lg p-6 text-center space-y-3">
            <div className="w-10 h-10 mx-auto rounded-[4px] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center">
              <Gauge className="w-5 h-5" />
            </div>
            <div className="text-[14px] font-bold text-[var(--text-primary)]">
              {activeProtocol === 'rabi'
                ? 'Rabi 时间域拉比振荡标定 (拟合实测 Ω_R 与 π / π/2 脉冲宽度)'
                : activeProtocol === 'ramsey'
                ? 'Ramsey 自由演化干涉条纹标定 (拟合相干时间 T2* 与频率失谐 Δ)'
                : activeProtocol === 't1'
                ? 'T1 能量弛豫衰减标定 (π 激发后测量自发辐射寿命 T1)'
                : activeProtocol === 'rb'
                ? 'Clifford 随机基准与交织门保真度标定 (RB & Interleaved RB)'
                : activeProtocol === 'allxy'
                ? 'AllXY 21 序列相干过旋转 / 失谐 / 正交偏斜脉冲诊断'
                : activeProtocol === 'gst'
                ? '门集层析成像 Gate Set Tomography (6 门 PTM 重构与钻石范数误差)'
                : '双比特线性交叉熵基准 Linear XEB (支持 √iSWAP 等非 Clifford 原生纠缠门)'}
            </div>
            <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">
              {activeProtocol === 'rabi'
                ? '扫描共振驱动脉冲时长 t，通过阻尼余弦最小二乘拟合精确提取真实拉比频率 Ω_R 及当前包络下的 π / π/2 脉冲时长，支持一键写回器件或逐比特参数表。'
                : activeProtocol === 'ramsey'
                ? '执行 π/2 − τ − π/2 序列并引入人工失谐条纹，同时拟合纯退相干包络 T2* 与本振频率偏移 Δ。'
                : activeProtocol === 't1'
                ? '执行 π 翻转脉冲制备 |1⟩ 态并扫描自由等待时间 τ，拟合指数弛豫曲线提取能量弛豫寿命 T1。'
                : activeProtocol === 'rb'
                ? '在右侧检查器选择 1Q (C₁=24) 或 2Q (C₂=11,520) Clifford 序列深度及待测交织门（如 SX、CZ、CNOT），通过指数衰减拟合剥离 SPAM 读出误差，精确提取单门平均失真度与相干极限。'
                : activeProtocol === 'allxy'
                ? '依次执行 21 组 {I, X, Y, X90, Y90} 双脉冲组合，对照 (+1, 0, -1) 理想三阶平台自动鉴别拉比幅度过/欠旋转锯齿、本振失谐分裂与 T1/T2 退相干压缩。'
                : activeProtocol === 'gst'
                ? '利用完备态制备与测量基准组（IC Fiducials）实施免 SPAM 干扰的规范对准线性反演层析，重构含 Gh (Hadamard) 在内的 6 门 4×4 泡利转移矩阵 (PTM)，定量分离相干标定误差与随机退相干误差。'
                : '采用 Google Sycamore 随机单比特 {√X, √Y, √W} + 双比特纠缠门交替周期，无需逆恢复门即可通过 Porter-Thomas 散斑比特串分布标定任意非 Clifford 双比特原生门（如 √iSWAP、fSim 或自定义纠缠门）的周期极化度与门保真度。'}
            </p>
            <button
              onClick={onRunBenchmark}
              disabled={isBusy}
              className="mx-auto h-8 px-4 rounded-[3px] studio-btn-primary text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-60"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              立即启动 {activeProtocol.toUpperCase()} 表征实验
            </button>
          </div>
        </div>
      )}

      {/* =================================================================== */}
      {/* VIEW 0: BASIC CALIBRATIONS (RABI / RAMSEY / T1 + WRITE-BACK)        */}
      {/* =================================================================== */}
      {matchingResp &&
        (matchingResp.protocol === 'rabi' ||
          matchingResp.protocol === 'ramsey' ||
          matchingResp.protocol === 't1') &&
        (() => {
          const cal: BasicCalibrationResponse = matchingResp;
          const topKpis = cal.params.slice(0, 4);
          const tPiParam = cal.params.find((p) => p.key === 't_pi');
          const rabiShapes =
            cal.protocol === 'rabi' && tPiParam && Number.isFinite(tPiParam.value)
              ? [
                  {
                    type: 'line',
                    x0: 0.5 * tPiParam.value,
                    x1: 0.5 * tPiParam.value,
                    y0: -0.02,
                    y1: 1.04,
                    line: { color: '#10B981', width: 1.3, dash: 'dot' },
                  },
                  {
                    type: 'line',
                    x0: tPiParam.value,
                    x1: tPiParam.value,
                    y0: -0.02,
                    y1: 1.04,
                    line: { color: '#F59E0B', width: 1.4, dash: 'dash' },
                  },
                ]
              : [];
          const calSamples: CharacterizationCircuitSample[] = cal.circuit_samples ?? [];

          return (
            <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3">
              {/* Summary + One-Click Write-Back Banner */}
              <div className="studio-card px-3.5 py-2.5 border-l-2 border-l-[var(--accent)] flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-[12px] text-[var(--text-primary)] font-medium">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                  <span>{cal.summary}</span>
                </div>
                {onApplyWriteback && cal.writeback.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                    {cal.writeback.map((wb, idx) => {
                      const wbKey = `${wb.scope}_${wb.qubit ?? 'hw'}_${wb.field}_${wb.value}`;
                      const done = appliedWbKey === wbKey;
                      return (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => {
                            onApplyWriteback(wb);
                            setAppliedWbKey(wbKey);
                          }}
                          className={`px-2.5 py-1 rounded-[3px] text-[11px] font-mono-num font-semibold flex items-center gap-1 transition-colors cursor-pointer ${
                            done
                              ? 'bg-emerald-600 text-white'
                              : 'bg-[var(--accent-soft)] hover:bg-[var(--accent)] hover:text-white text-[var(--accent)] border border-[var(--accent)]/40'
                          }`}
                        >
                          {done ? <Check className="w-3 h-3" /> : <ArrowRightLeft className="w-3 h-3" />}
                          <span>{done ? `已写回: ${wb.label}` : wb.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Top KPI Strip */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
                {topKpis.map((p) => (
                  <div key={p.key} className="studio-kpi p-2.5 flex flex-col justify-between">
                    <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                      {p.label}
                    </div>
                    <div className="text-[18px] font-bold font-mono-num text-[var(--accent)] mt-0.5 leading-tight">
                      {Number.isFinite(p.value) ? p.value : '—'}{' '}
                      <span className="text-[11px] font-normal text-[var(--text-secondary)]">
                        {p.unit}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Inline Calibration Pulse Sequence Schematic */}
              {calSamples.length > 0 && (
                <CharacterizationCircuitPanel
                  project={project}
                  protocolCode={cal.protocol.toUpperCase()}
                  title={
                    cal.protocol === 'rabi'
                      ? 'Rabi 驱动标定与 π/2 · π 门等效线路'
                      : cal.protocol === 'ramsey'
                      ? 'Ramsey π/2 − τ − π/2 自由演化线路'
                      : 'T1 π 激发与弛豫等待测量线路'
                  }
                  samples={calSamples}
                  onLoadCircuitToDesign={onLoadCircuitToDesign}
                />
              )}

              {/* Curve Plot + Fitted Parameter Table */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
                <div className="xl:col-span-8 studio-card flex flex-col overflow-hidden h-[315px]">
                  <div className="px-3.5 py-2 border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2">
                    <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                      {cal.protocol === 'rabi'
                        ? `Rabi 拉比振荡拟合曲线 (${cal.target_label})`
                        : cal.protocol === 'ramsey'
                        ? `Ramsey 自由进动干涉条纹 (${cal.target_label})`
                        : `T1 能量弛豫指数衰减曲线 (${cal.target_label})`}
                    </span>
                    <div className="flex items-center gap-3 text-[10px] font-mono-num text-[var(--text-secondary)] mr-12">
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-[#2F81F7]" />
                        实测采样点
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2.5 h-[2px] bg-[#10B981]" />
                        最小二乘拟合
                      </span>
                      {cal.protocol === 'rabi' && (
                        <span className="text-amber-500">虚线: t_π/2 与 t_π</span>
                      )}
                    </div>
                  </div>
                  <div className="flex-1 min-h-0 px-1.5 pb-1">
                    <PlotlyChart
                      data={[
                        {
                          x: cal.x,
                          y: cal.ideal,
                          type: 'scatter',
                          mode: 'lines',
                          name: '无读出混淆真值',
                          line: { color: 'rgba(148, 163, 184, 0.5)', width: 1.4, dash: 'dot' },
                        },
                        {
                          x: cal.fit_x,
                          y: cal.fit_y,
                          type: 'scatter',
                          mode: 'lines',
                          name: '拟合模型曲线',
                          line: { color: '#10B981', width: 2.4 },
                        },
                        {
                          x: cal.x,
                          y: cal.measured,
                          type: 'scatter',
                          mode: 'markers',
                          name: '实测 P(|1⟩)',
                          marker: { color: '#2F81F7', size: 6.5 },
                          hovertemplate: 't=%{x:.1f} ns · P(|1⟩)=%{y:.4f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        paper_bgcolor: plotBg,
                        plot_bgcolor: plotBg,
                        showlegend: false,
                        hovermode: 'closest',
                        margin: { l: 52, r: 18, t: 14, b: 40 },
                        xaxis: {
                          title: { text: cal.x_label, font: { size: 11, color: mutedColor } },
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 10.5 },
                        },
                        yaxis: {
                          title: { text: cal.y_label, font: { size: 11, color: mutedColor } },
                          range: [-0.04, 1.06],
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 10.5 },
                        },
                        shapes: rabiShapes,
                      }}
                    />
                  </div>
                </div>

                <div className="xl:col-span-4 studio-card flex flex-col overflow-hidden max-h-[315px]">
                  <div className="h-9 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                    <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                      标定参数拟合明细表
                    </span>
                    <span className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                      {cal.shots > 0 ? `${cal.shots} Shots` : '精确主方程'}
                    </span>
                  </div>
                  <div className="flex-1 overflow-auto">
                    <table className="w-full text-[11.5px] font-mono-num border-collapse">
                      <tbody className="divide-y divide-[var(--border-hairline)]">
                        {cal.params.map((p) => (
                          <tr key={p.key} className="hover:bg-black/[0.02] dark:hover:bg-white/[0.03]">
                            <td className="py-2 px-3 font-sans text-[var(--text-secondary)]">
                              {p.label}
                            </td>
                            <td className="py-2 px-3 text-right font-semibold text-[var(--text-primary)]">
                              {Number.isFinite(p.value) ? p.value : '—'}{' '}
                              <span className="text-[10.5px] font-normal text-[var(--text-secondary)]">
                                {p.unit}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

      {/* =================================================================== */}
      {/* VIEW 1: RANDOMIZED BENCHMARKING (1Q / 2Q RB + INTERLEAVED RB)       */}
      {/* =================================================================== */}
      {matchingResp && matchingResp.protocol === 'rb' && (() => {
        const rb: RbBenchmarkResponse = matchingResp;
        const irb = rb.interleaved;
        const rawX: number[] = [];
        const rawY: number[] = [];
        rb.lengths.forEach((m, idx) => {
          (rb.raw_survival[idx] || []).forEach((val) => {
            rawX.push(m);
            rawY.push(val);
          });
        });
        const rbSamples: CharacterizationCircuitSample[] =
          rb.circuit_samples && rb.circuit_samples.length > 0
            ? rb.circuit_samples
            : [
                {
                  id: 'rb_fallback',
                  label: `参考 RB (${rb.num_qubits}Q)`,
                  subtitle: '随机 Clifford 门序列 + 逆恢复门 C_inv → 测量',
                  operations: rb.sample_operations,
                },
              ];

        return (
          <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3">
            {/* 4 KPI Strip */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  Clifford 平均保真度 ({rb.num_qubits}Q)
                </div>
                <div className="text-[19px] font-bold font-mono-num text-emerald-500 mt-0.5 leading-tight">
                  {(rb.clifford_fidelity * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  衰减因子 p = {rb.depolarizing_p.toFixed(5)}
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  单 Clifford 错误率 (EPC)
                </div>
                <div className="text-[19px] font-bold font-mono-num text-[var(--text-primary)] mt-0.5 leading-tight">
                  {(rb.epc * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--accent)] truncate mt-0.5">
                  T1/T2 下界 {(rb.coherence_epc_floor * 100).toFixed(3)}%
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  {irb ? `交织门 [${irb.gate_name}] 保真度` : 'SPAM 可见度 A'}
                </div>
                {irb ? (
                  <>
                    <div className="text-[19px] font-bold font-mono-num text-[var(--accent)] mt-0.5 leading-tight">
                      {(irb.gate_fidelity * 100).toFixed(3)}%
                    </div>
                    <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                      r_gate = {(irb.gate_error * 100).toFixed(3)}%
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-[19px] font-bold font-mono-num text-[var(--accent)] mt-0.5 leading-tight">
                      {rb.a_spam.toFixed(4)}
                    </div>
                    <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                      基线 B = {rb.b_offset.toFixed(4)}
                    </div>
                  </>
                )}
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  标定目标 · {rb.target_label}
                </div>
                <div className="text-[16px] font-bold font-mono-num text-[var(--text-primary)] mt-0.5 leading-tight truncate">
                  {rb.num_seeds} Seeds × {rb.lengths.length} 深度
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  {rb.shots > 0 ? `${rb.shots} Shots/序列` : '精确密度矩阵极限'}
                </div>
              </div>
            </div>

            {/* Inline RB / Interleaved IRB Quantum Circuit Schematic */}
            <CharacterizationCircuitPanel
              project={project}
              protocolCode={`RB_${rb.num_qubits}Q`}
              title="RB / 交织 IRB 随机 Clifford 标定线路原理图"
              samples={rbSamples}
              onLoadCircuitToDesign={onLoadCircuitToDesign}
            />

            {/* Decay Plot + Per-Depth Table */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
              <div className="xl:col-span-7 studio-card flex flex-col overflow-hidden h-[305px]">
                <div className="px-3.5 py-2 border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                    随机基准存活率衰减曲线 P(|0⟩) = A · p^m + B
                  </span>
                  <div className="flex flex-wrap items-center gap-2.5 text-[10px] font-mono-num text-[var(--text-secondary)] mr-12">
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-[2px] bg-[#2F81F7]" />
                      参考 RB
                    </span>
                    {irb && (
                      <span className="flex items-center gap-1">
                        <span className="w-2.5 h-[2px] bg-[#F59E0B]" />
                        交织 {irb.gate_name}
                      </span>
                    )}
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#2F81F7]/40" />
                      单序列散点
                    </span>
                  </div>
                </div>
                <div className="flex-1 min-h-0 px-1.5 pb-1">
                  <PlotlyChart
                    data={[
                      {
                        x: rawX,
                        y: rawY,
                        type: 'scatter',
                        mode: 'markers',
                        name: '随机序列样本',
                        marker: { color: 'rgba(47, 129, 247, 0.24)', size: 5 },
                        hoverinfo: 'skip',
                      },
                      {
                        x: rb.fit_x,
                        y: rb.fit_y,
                        type: 'scatter',
                        mode: 'lines',
                        name: `参考 RB (p=${rb.depolarizing_p.toFixed(4)})`,
                        line: { color: '#2F81F7', width: 2.4 },
                      },
                      {
                        x: rb.lengths,
                        y: rb.mean_survival,
                        error_y: {
                          type: 'data',
                          array: rb.std_survival,
                          visible: true,
                          color: '#2F81F7',
                          thickness: 1.5,
                          width: 3,
                        },
                        type: 'scatter',
                        mode: 'markers',
                        name: '参考 RB 均值 ±1σ',
                        marker: { color: '#2F81F7', size: 8, symbol: 'circle' },
                      },
                      ...(irb
                        ? [
                            {
                              x: rb.fit_x,
                              y: irb.fit_y,
                              type: 'scatter',
                              mode: 'lines',
                              name: `交织 ${irb.gate_name} (p=${irb.p_irb.toFixed(4)})`,
                              line: { color: '#F59E0B', width: 2.2, dash: 'dash' },
                            },
                            {
                              x: rb.lengths,
                              y: irb.mean_survival,
                              error_y: {
                                type: 'data',
                                array: irb.std_survival,
                                visible: true,
                                color: '#F59E0B',
                                thickness: 1.5,
                                width: 3,
                              },
                              type: 'scatter',
                              mode: 'markers',
                              name: `交织 ${irb.gate_name} 均值 ±1σ`,
                              marker: { color: '#F59E0B', size: 8, symbol: 'diamond' },
                            },
                          ]
                        : []),
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      showlegend: false,
                      hovermode: 'closest',
                      margin: { l: 52, r: 18, t: 14, b: 40 },
                      xaxis: {
                        title: {
                          text: `Clifford 序列长度 m (共 ${rb.num_seeds} 组随机种子均值)`,
                          font: { size: 11, color: mutedColor },
                        },
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                      yaxis: {
                        title: {
                          text: rb.num_qubits === 2 ? '存活率 P(|00⟩)' : '存活率 P(|0⟩)',
                          font: { size: 11, color: mutedColor },
                        },
                        range: [Math.max(0, rb.b_offset - 0.12), 1.04],
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                    }}
                  />
                </div>
              </div>

              <div className="xl:col-span-5 studio-card flex flex-col overflow-hidden max-h-[305px]">
                <div className="h-9 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                    各深度存活概率明细
                  </span>
                  <span className="text-[10px] font-mono-num text-[var(--text-secondary)]">
                    A={rb.a_spam.toFixed(3)} B={rb.b_offset.toFixed(3)}
                  </span>
                </div>
                <div className="flex-1 overflow-auto">
                  <table className="w-full text-[11px] font-mono-num border-collapse">
                    <thead className="sticky top-0 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] text-[10px] text-[var(--text-secondary)] text-left">
                      <tr>
                        <th className="py-1.5 px-2">m</th>
                        <th className="py-1.5 px-2">参考均值</th>
                        <th className="py-1.5 px-2">±1σ</th>
                        {irb && <th className="py-1.5 px-2">交织 {irb.gate_name}</th>}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-hairline)]">
                      {rb.lengths.map((m, idx) => (
                        <tr key={m} className="hover:bg-black/[0.025] dark:hover:bg-white/[0.04]">
                          <td className="py-1.5 px-2 text-[var(--accent)] font-semibold">{m}</td>
                          <td className="py-1.5 px-2">{rb.mean_survival[idx].toFixed(4)}</td>
                          <td className="py-1.5 px-2 text-[var(--text-secondary)]">
                            ±{rb.std_survival[idx].toFixed(4)}
                          </td>
                          {irb && (
                            <td className="py-1.5 px-2 text-amber-500 font-medium">
                              {irb.mean_survival[idx].toFixed(4)}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {irb && (
                  <div className="px-3 py-2 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                    Magesan 系统误差界: r_gate = {(irb.gate_error * 100).toFixed(3)}% ±{' '}
                    {(irb.systematic_bound * 100).toFixed(2)}%
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })()}

      {/* =================================================================== */}
      {/* VIEW 2: ALLXY 21-SEQUENCE COHERENT / INCOHERENT PULSE DIAGNOSTIC    */}
      {/* =================================================================== */}
      {matchingResp && matchingResp.protocol === 'allxy' && (() => {
        const axy: AllXyBenchmarkResponse = matchingResp;
        const labels = axy.sequences.map((s) => s.label);
        const xNums = axy.sequences.map((s) => s.index);
        const idealY = axy.sequences.map((s) => (allxyUnit === 'z' ? s.ideal_z : s.ideal_p1));
        const measY = axy.sequences.map((s) => (allxyUnit === 'z' ? s.measured_z : s.measured_p1));
        const errBar = axy.sequences.map((s) => (allxyUnit === 'z' ? s.std_z : s.std_z * 0.5));
        const residuals = axy.sequences.map((s) =>
          allxyUnit === 'z' ? s.residual_z : -0.5 * s.residual_z
        );
        const safeAxyIdx = Math.min(Math.max(0, selectedAllxyIdx), axy.sequences.length - 1);
        const allxySamples: CharacterizationCircuitSample[] = axy.sequences.map((s) => ({
          id: `allxy_${s.index}`,
          label: `#${s.index} ${s.label}`,
          subtitle: `平台 ${s.plateau} · 理想 ⟨Z⟩=${s.ideal_z.toFixed(1)} · 实测 ⟨Z⟩=${s.measured_z.toFixed(4)} (残差 Δ=${s.residual_z >= 0 ? '+' : ''}${s.residual_z.toFixed(4)})`,
          operations: s.operations,
        }));

        const bannerCls =
          axy.severity === 'good'
            ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
            : axy.severity === 'warn'
            ? 'border-amber-500/45 bg-amber-500/12 text-amber-300'
            : 'border-[var(--accent)]/40 bg-[var(--accent-soft)] text-[var(--text-primary)]';

        return (
          <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3">
            {/* Automated Diagnostic Banner */}
            <div className={`px-3.5 py-2.5 rounded-[4px] border flex flex-wrap items-center justify-between gap-2 ${bannerCls}`}>
              <div className="flex items-center gap-2 text-[12px]">
                {axy.severity === 'good' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                ) : axy.severity === 'warn' ? (
                  <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />
                ) : (
                  <Info className="w-4 h-4 text-[var(--accent)] shrink-0" />
                )}
                <span className="font-medium">{axy.diagnosis}</span>
              </div>
              <div className="flex items-center gap-3 shrink-0 text-[11px] font-mono-num">
                <span>RMS Δ⟨Z⟩ = {axy.rms_error_z.toFixed(4)}</span>
                <span>
                  幅度估计 {axy.estimated_amp_error_pct >= 0 ? '+' : ''}
                  {axy.estimated_amp_error_pct.toFixed(2)}%
                </span>
              </div>
            </div>

            {/* Inline AllXY Dual-Pulse Sequence Schematic */}
            <CharacterizationCircuitPanel
              project={project}
              protocolCode="AllXY"
              title="AllXY 双脉冲诊断线路原理图"
              samples={allxySamples}
              selectedSampleId={`allxy_${axy.sequences[safeAxyIdx]?.index ?? 1}`}
              onSelectSample={(smp) => {
                const idx = axy.sequences.findIndex((s) => `allxy_${s.index}` === smp.id);
                if (idx >= 0) setSelectedAllxyIdx(idx);
              }}
              onLoadCircuitToDesign={onLoadCircuitToDesign}
            />

            {/* Staircase + Residuals + Table */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
              <div className="xl:col-span-7 flex flex-col gap-3">
                <div className="studio-card h-[245px] flex flex-col overflow-hidden">
                  <div className="px-3.5 py-1.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2">
                    <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                      AllXY 21 序列三阶标定阶梯图 ({axy.target_label})
                    </span>
                    <div className="p-0.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex text-[10px] font-mono-num mr-12">
                      <button
                        onClick={() => setAllxyUnit('z')}
                        className={`px-2 py-0.5 rounded-[2px] cursor-pointer ${
                          allxyUnit === 'z'
                            ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        ⟨Z⟩ (+1,0,-1)
                      </button>
                      <button
                        onClick={() => setAllxyUnit('p1')}
                        className={`px-2 py-0.5 rounded-[2px] cursor-pointer ${
                          allxyUnit === 'p1'
                            ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        P(|1⟩) (0,0.5,1)
                      </button>
                    </div>
                  </div>
                  <div className="flex-1 min-h-0 px-1.5 pb-1">
                    <PlotlyChart
                      data={[
                        {
                          x: xNums,
                          y: idealY,
                          type: 'scatter',
                          mode: 'lines',
                          name: '理想无差阶梯',
                          line: { color: '#10B981', width: 2, shape: 'hv', dash: 'dot' },
                        },
                        {
                          x: xNums,
                          y: measY,
                          text: labels,
                          error_y: {
                            type: 'data',
                            array: errBar,
                            visible: axy.shots > 0,
                            color: '#2F81F7',
                            thickness: 1.4,
                            width: 3,
                          },
                          type: 'scatter',
                          mode: 'lines+markers',
                          name: '实测脉冲响应',
                          line: { color: 'rgba(47,129,247,0.4)', width: 1.4 },
                          marker: {
                            size: 8,
                            color: axy.sequences.map((s) =>
                              s.plateau === 'ground'
                                ? '#2F81F7'
                                : s.plateau === 'equator'
                                ? '#F59E0B'
                                : '#EF4444'
                            ),
                          },
                          hovertemplate: '#%{x} (%{text}): %{y:.4f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        paper_bgcolor: plotBg,
                        plot_bgcolor: plotBg,
                        showlegend: false,
                        margin: { l: 48, r: 16, t: 12, b: 56 },
                        xaxis: {
                          tickmode: 'array',
                          tickvals: xNums,
                          ticktext: labels,
                          tickangle: -45,
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 8.5 },
                        },
                        yaxis: {
                          title: {
                            text: allxyUnit === 'z' ? '期望值 ⟨Z⟩' : '激发态概率 P(|1⟩)',
                            font: { size: 11, color: mutedColor },
                          },
                          range: allxyUnit === 'z' ? [-1.15, 1.15] : [-0.08, 1.08],
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 10 },
                        },
                      }}
                    />
                  </div>
                </div>

                {/* Residual Sawtooth Bar Chart */}
                <div className="studio-card h-[135px] flex flex-col overflow-hidden shrink-0">
                  <div className="h-7 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between text-[11px]">
                    <span className="font-semibold text-[var(--text-primary)]">
                      序列残差偏离 Δ = 实测 − 理想（锯齿极性指示幅度/相位偏差符号）
                    </span>
                  </div>
                  <div className="flex-1 min-h-0 px-1.5 pb-1">
                    <PlotlyChart
                      data={[
                        {
                          x: xNums,
                          y: residuals,
                          text: labels,
                          type: 'bar',
                          marker: {
                            color: residuals.map((v) => (v >= 0 ? '#2F81F7' : '#EF4444')),
                          },
                          hovertemplate: '#%{x} (%{text}): Δ = %{y:+.4f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        paper_bgcolor: plotBg,
                        plot_bgcolor: plotBg,
                        showlegend: false,
                        margin: { l: 48, r: 16, t: 8, b: 22 },
                        xaxis: {
                          dtick: 2,
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 9 },
                        },
                        yaxis: {
                          gridcolor: gridColor,
                          tickfont: { color: mutedColor, size: 9.5 },
                        },
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Right 5 Cols: Clickable 21-Sequence Table (Selects Inline Schematic!) */}
              <div className="xl:col-span-5 studio-card flex flex-col overflow-hidden max-h-[392px]">
                <div className="h-8 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                    21 组脉冲序列明细 (点击行查看对应线路)
                  </span>
                </div>
                <div className="flex-1 overflow-auto">
                  <table className="w-full text-[11px] font-mono-num border-collapse">
                    <thead className="sticky top-0 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] text-[10px] text-[var(--text-secondary)] text-left">
                      <tr>
                        <th className="py-1.5 px-2">#</th>
                        <th className="py-1.5 px-2">脉冲对</th>
                        <th className="py-1.5 px-2">理想⟨Z⟩</th>
                        <th className="py-1.5 px-2">实测⟨Z⟩</th>
                        <th className="py-1.5 px-2">残差 Δ</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-hairline)]">
                      {axy.sequences.map((s, idx) => (
                        <tr
                          key={s.index}
                          onClick={() => setSelectedAllxyIdx(idx)}
                          title={`点击在上方内联画布查看 AllXY #${s.index} (${s.label}) 脉冲序列`}
                          className={`cursor-pointer transition-colors ${
                            idx === safeAxyIdx
                              ? 'bg-[var(--accent-soft)] font-semibold'
                              : 'hover:bg-black/[0.025] dark:hover:bg-white/[0.04]'
                          }`}
                        >
                          <td className="py-1 px-2 text-[var(--text-secondary)]">{s.index}</td>
                          <td className="py-1 px-2 font-semibold text-[var(--accent)]">{s.label}</td>
                          <td className="py-1 px-2">{s.ideal_z.toFixed(1)}</td>
                          <td className="py-1 px-2">{s.measured_z.toFixed(4)}</td>
                          <td
                            className={`py-1 px-2 font-semibold ${
                              Math.abs(s.residual_z) > 0.05
                                ? 'text-amber-500'
                                : 'text-emerald-500'
                            }`}
                          >
                            {s.residual_z >= 0 ? '+' : ''}
                            {s.residual_z.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* =================================================================== */}
      {/* VIEW 3: GATE SET TOMOGRAPHY (GST — PTM & DIAMOND NORM)              */}
      {/* =================================================================== */}
      {matchingResp && matchingResp.protocol === 'gst' && (() => {
        const gst: GstBenchmarkResponse = matchingResp;
        const safeIdx = Math.min(selectedGstGateIdx, gst.gates.length - 1);
        const activeGate = gst.gates[safeIdx] || gst.gates[0];
        const pLabels = gst.pauli_labels;
        const gstSamples: CharacterizationCircuitSample[] =
          gst.circuit_samples && gst.circuit_samples.length > 0
            ? gst.circuit_samples
            : [
                {
                  id: 'gst_fallback',
                  label: 'GST 完备层析电路',
                  subtitle: '态制备 Fiducial F_j → 待测门 Germ G^L → 测量基准 M_i',
                  operations: gst.sample_operations,
                },
              ];

        const maxAbsErr = Math.max(
          0.02,
          ...activeGate.ptm_error.flat().map((v) => Math.abs(v))
        );

        const makeHeatmapAnnotations = (mat: number[][], isError: boolean) => {
          const anns: any[] = [];
          for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
              const rawV = mat[r][c];
              const cleanV = Math.abs(rawV) < 5e-5 ? 0 : rawV;
              const txt = isError
                ? cleanV === 0
                  ? '0'
                  : (cleanV > 0 ? '+' : '') + cleanV.toFixed(2)
                : cleanV.toFixed(2);
              anns.push({
                x: pLabels[c],
                y: pLabels[r],
                text: txt,
                showarrow: false,
                font: {
                  size: 10.5,
                  family: 'JetBrains Mono, Consolas, monospace',
                  color: Math.abs(cleanV) > (isError ? maxAbsErr * 0.6 : 0.55) ? '#FFFFFF' : textColor,
                },
              });
            }
          }
          return anns;
        };

        return (
          <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3">
            {/* Top 4 KPI Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  门集平均保真度 F_avg
                </div>
                <div className="text-[19px] font-bold font-mono-num text-emerald-500 mt-0.5 leading-tight">
                  {(gst.mean_gate_fidelity * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  共 {gst.num_experiments} 组完备层析电路
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  最大半钻石距离 ½‖Δℰ‖_⋄
                </div>
                <div className="text-[19px] font-bold font-mono-num text-amber-500 mt-0.5 leading-tight">
                  {(gst.max_diamond_distance * 100).toFixed(2)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  容错量子纠错最坏误差上界
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  规范不变量幺正度 u(ℰ)
                </div>
                <div className="text-[19px] font-bold font-mono-num text-[var(--accent)] mt-0.5 leading-tight">
                  {gst.mean_unitarity.toFixed(5)}
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  分离相干旋转与非相干退相干
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  SPAM 态制备 / 读出保真度
                </div>
                <div className="text-[17px] font-bold font-mono-num text-[var(--text-primary)] mt-0.5 leading-tight">
                  {(gst.spam_prep_fidelity * 100).toFixed(1)}% / {(gst.spam_readout_fidelity * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  初态 |0⟩ 与 POVM 独立剥离
                </div>
              </div>
            </div>

            {/* Gate Selector Strip */}
            <div className="studio-card px-3 py-1.5 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[11px] font-mono-num font-bold text-[var(--text-secondary)] mr-1">
                  层析门:
                </span>
                {gst.gates.map((g, idx) => (
                  <button
                    key={g.gate_id}
                    onClick={() => {
                      setSelectedGstGateIdx(idx);
                      setSelectedGstSampleId(`gst_${g.gate_id}_l1`);
                    }}
                    className={`px-2 py-0.5 rounded-[3px] text-[11px] font-mono-num transition-colors cursor-pointer ${
                      idx === safeIdx
                        ? 'bg-[var(--accent)] text-white font-semibold'
                        : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border-hairline)]'
                    }`}
                  >
                    {g.gate_id} ({g.gate_token})
                  </button>
                ))}
              </div>
              <div className="text-[11px] font-mono-num text-[var(--text-secondary)] flex items-center gap-3">
                <span>
                  F_avg: <strong className="text-emerald-500">{(activeGate.avg_gate_fidelity * 100).toFixed(3)}%</strong>
                </span>
                <span>
                  ½‖Δℰ‖_⋄: <strong className="text-amber-500">{activeGate.diamond_distance.toFixed(4)}</strong>
                </span>
                <span>
                  相干占比: <strong className="text-[var(--accent)]">{(activeGate.coherent_ratio * 100).toFixed(1)}%</strong>
                </span>
              </div>
            </div>

            {/* Inline GST Fiducial + Germ Tomography Circuit Schematic (Synced with Selected Gate!) */}
            <CharacterizationCircuitPanel
              project={project}
              protocolCode="GST"
              title="GST 态制备 F_j · 胚门放大 G^L · 测量基准 M_i 线路原理图"
              samples={gstSamples}
              selectedSampleId={selectedGstSampleId}
              onSelectSample={(smp) => {
                setSelectedGstSampleId(smp.id);
                if (smp.gate_id) {
                  const gIdx = gst.gates.findIndex((g) => g.gate_id === smp.gate_id);
                  if (gIdx >= 0) setSelectedGstGateIdx(gIdx);
                }
              }}
              onLoadCircuitToDesign={onLoadCircuitToDesign}
            />

            {/* Spacious 2-Column 4x4 PTM Heatmaps */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="studio-card flex flex-col overflow-hidden h-[255px]">
                <div className="h-8 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between">
                  <span className="text-[11.5px] font-semibold text-[var(--text-primary)]">
                    实测泡利转移矩阵 R_meas ({activeGate.gate_id})
                  </span>
                  <span className="text-[10px] font-mono-num text-[var(--text-secondary)] mr-12">
                    基组 &#123;I, X, Y, Z&#125; ∈ [-1, +1]
                  </span>
                </div>
                <div className="flex-1 min-h-0 p-1">
                  <PlotlyChart
                    data={[
                      {
                        z: activeGate.ptm_reconstructed,
                        x: pLabels,
                        y: pLabels,
                        type: 'heatmap',
                        zmin: -1,
                        zmax: 1,
                        colorscale: [
                          [0, '#EF4444'],
                          [0.5, darkMode ? '#1A222D' : '#F8FAFC'],
                          [1, '#2F81F7'],
                        ],
                        showscale: false,
                        hovertemplate: '输出 %{y} ← 输入 %{x}: %{z:.4f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      margin: { l: 36, r: 16, t: 10, b: 30 },
                      xaxis: { tickfont: { color: textColor, size: 11 } },
                      yaxis: { autorange: 'reversed', tickfont: { color: textColor, size: 11 } },
                      annotations: makeHeatmapAnnotations(activeGate.ptm_reconstructed, false),
                    }}
                  />
                </div>
              </div>

              <div className="studio-card flex flex-col overflow-hidden h-[255px]">
                <div className="h-8 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between">
                  <span className="text-[11.5px] font-semibold text-[var(--text-primary)]">
                    过程误差矩阵 ΔR = R_meas − R_ideal
                  </span>
                  <span className="text-[10px] font-mono-num text-amber-500 mr-12">
                    量程 ±{maxAbsErr.toFixed(3)}
                  </span>
                </div>
                <div className="flex-1 min-h-0 p-1">
                  <PlotlyChart
                    data={[
                      {
                        z: activeGate.ptm_error,
                        x: pLabels,
                        y: pLabels,
                        type: 'heatmap',
                        zmin: -maxAbsErr,
                        zmax: maxAbsErr,
                        colorscale: [
                          [0, '#EF4444'],
                          [0.5, darkMode ? '#1A222D' : '#F8FAFC'],
                          [1, '#10B981'],
                        ],
                        showscale: false,
                        hovertemplate: 'ΔR(%{y}, %{x}) = %{z:+.5f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      margin: { l: 36, r: 16, t: 10, b: 30 },
                      xaxis: { tickfont: { color: textColor, size: 11 } },
                      yaxis: { autorange: 'reversed', tickfont: { color: textColor, size: 11 } },
                      annotations: makeHeatmapAnnotations(activeGate.ptm_error, true),
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Bottom Row: Bar Comparison + Full Gate Set Summary Table */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
              <div className="xl:col-span-5 studio-card flex flex-col overflow-hidden h-[225px]">
                <div className="h-8 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2">
                  <span className="text-[11.5px] font-semibold text-[var(--text-primary)] truncate">
                    失真度 (1−F) vs 半钻石距离
                  </span>
                  <div className="flex items-center gap-2 text-[10px] font-mono-num shrink-0 mr-12">
                    <span className="text-[#2F81F7]">■ 1−F_avg</span>
                    <span className="text-[#F59E0B]">■ ½‖Δℰ‖_⋄</span>
                  </div>
                </div>
                <div className="flex-1 min-h-0 p-1">
                  <PlotlyChart
                    data={[
                      {
                        x: gst.gates.map((g) => g.gate_id),
                        y: gst.gates.map((g) => g.gate_infidelity),
                        type: 'bar',
                        name: '平均门失真度 1 - F_avg',
                        marker: { color: '#2F81F7' },
                      },
                      {
                        x: gst.gates.map((g) => g.gate_id),
                        y: gst.gates.map((g) => g.diamond_distance),
                        type: 'bar',
                        name: '半钻石距离 ½‖Δℰ‖_⋄',
                        marker: { color: '#F59E0B' },
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      barmode: 'group',
                      showlegend: false,
                      margin: { l: 48, r: 14, t: 10, b: 32 },
                      xaxis: { tickangle: 0, tickfont: { color: textColor, size: 10.5 } },
                      yaxis: {
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10 },
                      },
                    }}
                  />
                </div>
              </div>

              <div className="xl:col-span-7 studio-card overflow-hidden flex flex-col">
                <div className="h-8 px-3 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0">
                  <span className="text-[11.5px] font-semibold text-[var(--text-primary)] truncate">
                    GST 门集层析指标表 (点击行切换 PTM 与层析线路)
                  </span>
                  <span className="text-[10px] font-mono-num text-[var(--text-secondary)] flex items-center gap-1 shrink-0">
                    <ShieldCheck className="w-3 h-3 text-emerald-500" />
                    CPTP 规范对准
                  </span>
                </div>
                <div className="overflow-x-auto flex-1">
                  <table className="w-full text-[11px] font-mono-num border-collapse">
                    <thead className="bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)] text-[10px] text-[var(--text-secondary)] text-left">
                      <tr>
                        <th className="py-1.5 px-2.5">门</th>
                        <th className="py-1.5 px-2.5">F_avg</th>
                        <th className="py-1.5 px-2.5">F_proc</th>
                        <th className="py-1.5 px-2.5">½‖Δℰ‖_⋄</th>
                        <th className="py-1.5 px-2.5">幺正度 u</th>
                        <th className="py-1.5 px-2.5">相干占比</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-hairline)]">
                      {gst.gates.map((g, idx) => (
                        <tr
                          key={g.gate_id}
                          onClick={() => {
                            setSelectedGstGateIdx(idx);
                            setSelectedGstSampleId(`gst_${g.gate_id}_l1`);
                          }}
                          className={`cursor-pointer transition-colors ${
                            idx === safeIdx
                              ? 'bg-[var(--accent-soft)] font-semibold'
                              : 'hover:bg-black/[0.025] dark:hover:bg-white/[0.04]'
                          }`}
                        >
                          <td className="py-1.5 px-2.5 text-[var(--accent)] font-bold">{g.gate_id}</td>
                          <td className="py-1.5 px-2.5 text-emerald-500">
                            {(g.avg_gate_fidelity * 100).toFixed(3)}%
                          </td>
                          <td className="py-1.5 px-2.5">{(g.process_fidelity * 100).toFixed(3)}%</td>
                          <td className="py-1.5 px-2.5 text-amber-500">
                            {g.diamond_distance.toFixed(5)}
                          </td>
                          <td className="py-1.5 px-2.5">{g.unitarity.toFixed(5)}</td>
                          <td className="py-1.5 px-2.5">{(g.coherent_ratio * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* =================================================================== */}
      {/* VIEW 4: 2Q LINEAR CROSS-ENTROPY BENCHMARKING (LINEAR XEB)           */}
      {/* =================================================================== */}
      {matchingResp && matchingResp.protocol === 'xeb' && (() => {
        const xeb: XebBenchmarkResponse = matchingResp;
        const maxSpk = Math.max(0.55, ...xeb.speckle_ideal, ...xeb.speckle_measured) * 1.06;
        const xebSamples: CharacterizationCircuitSample[] =
          xeb.circuit_samples && xeb.circuit_samples.length > 0
            ? xeb.circuit_samples
            : [
                {
                  id: 'xeb_fallback',
                  label: `Sycamore 周期 (${xeb.entangler_name})`,
                  subtitle: `随机单比特 {√X, √Y, √W} 层 + ${xeb.entangler_name} 纠缠门`,
                  operations: xeb.sample_operations,
                },
              ];

        return (
          <div className="flex-1 min-h-0 overflow-y-auto p-3.5 space-y-3">
            {/* 4 KPI Strip */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  纠缠门 [{xeb.entangler_name}] 保真度
                </div>
                <div className="text-[19px] font-bold font-mono-num text-emerald-500 mt-0.5 leading-tight">
                  {(xeb.entangler_fidelity * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  已剥离并行单比特层极化损耗
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  单周期极化度 f_cycle
                </div>
                <div className="text-[19px] font-bold font-mono-num text-[var(--accent)] mt-0.5 leading-tight">
                  {(xeb.cycle_polarization * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  SPAM 可见度 A = {xeb.a_spam.toFixed(4)}
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  单周期泡利错误率 e_cycle
                </div>
                <div className="text-[19px] font-bold font-mono-num text-amber-500 mt-0.5 leading-tight">
                  {(xeb.cycle_pauli_error * 100).toFixed(3)}%
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  e_c = ¾(1 − f_cycle)
                </div>
              </div>

              <div className="studio-kpi p-2.5 flex flex-col justify-between">
                <div className="text-[10px] font-mono-num uppercase text-[var(--text-secondary)] truncate">
                  标定对 · {xeb.target_label}
                </div>
                <div className="text-[16px] font-bold font-mono-num text-[var(--text-primary)] mt-0.5 leading-tight truncate">
                  {xeb.num_seeds} 随机电路 × {xeb.lengths.length} 深度
                </div>
                <div className="text-[10px] font-mono-num text-[var(--text-secondary)] truncate mt-0.5">
                  {xeb.shots > 0 ? `${xeb.shots} Shots/电路` : '精确密度矩阵极限'}
                </div>
              </div>
            </div>

            {/* Inline 2Q XEB Sycamore Cycle Circuit Schematic */}
            <CharacterizationCircuitPanel
              project={project}
              protocolCode={`XEB_${xeb.entangler_name}`}
              title={`2Q 线性交叉熵 (XEB) 随机单比特 + ${xeb.entangler_name} 交变周期线路`}
              samples={xebSamples}
              onLoadCircuitToDesign={onLoadCircuitToDesign}
            />

            {/* XEB Decay Curve + Porter-Thomas Speckle Scatter */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
              <div className="xl:col-span-7 studio-card flex flex-col overflow-hidden h-[305px]">
                <div className="px-3.5 py-2 border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                    线性交叉熵保真度衰减曲线 F_XEB(m) = A · f_cycle^m
                  </span>
                  <span className="text-[10px] font-mono-num text-[var(--text-secondary)] mr-12">
                    随机 &#123;√X, √Y, √W&#125; + {xeb.entangler_name}
                  </span>
                </div>
                <div className="flex-1 min-h-0 px-1.5 pb-1">
                  <PlotlyChart
                    data={[
                      {
                        x: xeb.fit_x,
                        y: xeb.fit_y,
                        type: 'scatter',
                        mode: 'lines',
                        name: `指数拟合 (f=${xeb.cycle_polarization.toFixed(4)})`,
                        line: { color: '#10B981', width: 2.4 },
                      },
                      {
                        x: xeb.lengths,
                        y: xeb.mean_xeb,
                        error_y: {
                          type: 'data',
                          array: xeb.std_xeb,
                          visible: true,
                          color: '#2F81F7',
                          thickness: 1.5,
                          width: 3,
                        },
                        type: 'scatter',
                        mode: 'markers',
                        name: '实测 F_XEB ±1σ',
                        marker: { color: '#2F81F7', size: 8.5, symbol: 'circle' },
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      showlegend: false,
                      hovermode: 'closest',
                      margin: { l: 52, r: 18, t: 14, b: 40 },
                      xaxis: {
                        title: {
                          text: '交变纠缠周期深度 m (Cycles)',
                          font: { size: 11, color: mutedColor },
                        },
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                      yaxis: {
                        title: {
                          text: '归一化线性交叉熵 F_XEB',
                          font: { size: 11, color: mutedColor },
                        },
                        range: [-0.04, 1.06],
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10.5 },
                      },
                    }}
                  />
                </div>
              </div>

              <div className="xl:col-span-5 studio-card flex flex-col overflow-hidden h-[305px]">
                <div className="px-3.5 py-2 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)] truncate">
                    Porter-Thomas 散斑概率相关性
                  </span>
                  <span className="text-[10px] font-mono-num text-[var(--text-secondary)] shrink-0 mr-12">
                    p_meas vs p_ideal
                  </span>
                </div>
                <div className="flex-1 min-h-0 px-1.5 pb-1">
                  <PlotlyChart
                    data={[
                      {
                        x: [0, maxSpk],
                        y: [0, maxSpk],
                        type: 'scatter',
                        mode: 'lines',
                        name: '无噪理想对角线 y = x',
                        line: { color: '#10B981', width: 1.8, dash: 'dot' },
                      },
                      {
                        x: [0, maxSpk],
                        y: [0.25, 0.25],
                        type: 'scatter',
                        mode: 'lines',
                        name: '完全退极化均匀极限 1/4',
                        line: { color: '#EF4444', width: 1.2, dash: 'dash' },
                      },
                      {
                        x: xeb.speckle_ideal,
                        y: xeb.speckle_measured,
                        type: 'scatter',
                        mode: 'markers',
                        name: '比特串散斑点',
                        marker: { color: '#2F81F7', size: 6.5, opacity: 0.78 },
                        hovertemplate: '理想 p=%{x:.3f}<br>实测 p=%{y:.3f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      paper_bgcolor: plotBg,
                      plot_bgcolor: plotBg,
                      showlegend: false,
                      hovermode: 'closest',
                      margin: { l: 48, r: 16, t: 14, b: 40 },
                      xaxis: {
                        title: { text: '理想无噪概率 p_ideal(x)', font: { size: 10.5, color: mutedColor } },
                        range: [0, maxSpk],
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10 },
                      },
                      yaxis: {
                        title: { text: '含噪实测概率 p_meas(x)', font: { size: 10.5, color: mutedColor } },
                        range: [0, maxSpk],
                        gridcolor: gridColor,
                        tickfont: { color: mutedColor, size: 10 },
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
