import React, { useMemo, useRef, useState } from 'react';
import {
  Plus,
  Play,
  Pause,
  Layers,
  Trash2,
  Gauge,
  AlertTriangle,
  Download,
  Sparkles,
  SkipBack,
  SkipForward,
} from 'lucide-react';
import type {
  CircuitOpModel,
  CustomGateDef,
  ProjectModel,
  SelectionTarget,
} from '../types';
import { BUILTIN_GATES } from '../constants';
import { buildCircuitSvg, downloadTextFile } from '../utils/exportCircuitSvg';

interface CircuitCanvasProps {
  project: ProjectModel;
  operationsOverride?: CircuitOpModel[];
  readOnly?: boolean;
  compact?: boolean;
  selection: SelectionTarget;
  activeToolGate: string | null;
  onSelect: (sel: SelectionTarget) => void;
  onPickGateTool: (gateName: string | null) => void;
  onPlaceGate: (gateName: string, step: number, qubits: number[]) => void;
  onMoveOp?: (opId: string, newStep: number, newAnchorQubit: number) => void;
  onDeleteOp: (opId: string) => void;
  onAddQubit: () => void;
  onPackageMacro?: (selectedOpIds: string[]) => void;
  onOpenNlModal?: () => void;
  // Simulate mode timeline cursor
  timeCursorStep?: number;
  maxCursorStep?: number;
  onTimeCursorChange?: (step: number) => void;
  onStepJump?: (targetStep: number) => void;
  isPlayingTimeline?: boolean;
  onTogglePlayTimeline?: () => void;
  playbackSpeed?: 0.5 | 1 | 2;
  onPlaybackSpeedChange?: (speed: 0.5 | 1 | 2) => void;
  titleBadge?: string;
}

const ROW_H = 54;
const ROW_H_COMPACT = 35;
const COL_W = 86;
const LEFT_PAD = 92;

const PALETTE_1Q = ['H', 'X', 'SX', 'RXY', 'RX', 'RY', 'RZ'];
const PALETTE_2Q = ['CZ', 'CNOT', 'ISWAP'];
const PALETTE_UTIL = ['DEPOLARIZE', 'MEASURE'];

export const CircuitCanvas: React.FC<CircuitCanvasProps> = ({
  project,
  operationsOverride,
  readOnly = false,
  compact = false,
  selection,
  activeToolGate,
  onSelect,
  onPickGateTool,
  onPlaceGate,
  onMoveOp,
  onDeleteOp,
  onAddQubit,
  onPackageMacro,
  onOpenNlModal,
  timeCursorStep,
  maxCursorStep,
  onTimeCursorChange,
  onStepJump,
  isPlayingTimeline,
  onTogglePlayTimeline,
  playbackSpeed = 1,
  onPlaybackSpeedChange,
  titleBadge,
}) => {
  const ops = operationsOverride ?? project.operations;
  const rowH = compact ? ROW_H_COMPACT : ROW_H;
  const [pendingFirstQubit, setPendingFirstQubit] = useState<{
    gateName: string;
    step: number;
    qubits: number[];
    needed: number;
  } | null>(null);
  const [multiSelectedIds, setMultiSelectedIds] = useState<string[]>([]);
  const [extraCols, setExtraCols] = useState(2);
  const [dragHoverCell, setDragHoverCell] = useState<{ q: number; c: number } | null>(null);
  const [hoverCell, setHoverCell] = useState<{ q: number; c: number } | null>(null);
  const [vpScroll, setVpScroll] = useState({ top: 0, left: 0, height: 720, width: 1280 });
  const gridRef = useRef<HTMLDivElement>(null);
  const scrollWrapRef = useRef<HTMLDivElement>(null);

  const customGateMap = useMemo(() => {
    const m = new Map<string, CustomGateDef>();
    project.custom_gates.forEach((g) => m.set(g.name.toUpperCase(), g));
    return m;
  }, [project.custom_gates]);

  const builtinMap = useMemo(() => {
    const m = new Map(BUILTIN_GATES.map((g) => [g.name.toUpperCase(), g]));
    return m;
  }, []);

  // O(1) / sparse hardware coupling lookup to flag topology/blockade violations live on canvas
  const activeHwTopology = useMemo(() => {
    const hw =
      project.hardware_profiles.find((h) => h.id === project.active_hardware_id) ||
      project.hardware_profiles[0];
    const customSet = new Set<string>();
    if (hw?.topology_mode === 'custom') {
      for (const [u, v] of hw.edges) {
        if (u !== v) customSet.add(`${Math.min(u, v)}-${Math.max(u, v)}`);
      }
    }
    return {
      mode: hw?.topology_mode || 'all',
      rb: Math.max(0.1, hw?.blockade_radius_um ?? 6.5),
      customSet,
    };
  }, [project.hardware_profiles, project.active_hardware_id]);

  const isPairTopologyViolated = (validQ: number[]) => {
    if (validQ.length !== 2) return false;
    const [qa, qb] = validQ;
    const mode = activeHwTopology.mode;
    if (mode === 'all') return false;
    if (mode === 'linear') return Math.abs(qa - qb) !== 1;
    if (mode === 'rydberg_radius') {
      const a = project.qubits[qa];
      const b = project.qubits[qb];
      if (!a || !b) return false;
      const d = Math.hypot(a.pos_x - b.pos_x, a.pos_y - b.pos_y);
      return d > activeHwTopology.rb + 1e-9;
    }
    const key = `${Math.min(qa, qb)}-${Math.max(qa, qb)}`;
    return !activeHwTopology.customSet.has(key);
  };

  const handleExportSvg = () => {
    const svg = buildCircuitSvg(project, operationsOverride, titleBadge || project.name);
    const fname = `${(project.name || 'circuit').replace(/\.[^.]+$/, '')}${operationsOverride ? '_optimized' : ''}.svg`;
    downloadTextFile(fname, svg);
  };

  const maxStepInOps = useMemo(() => {
    let m = 0;
    for (const op of ops) m = Math.max(m, op.step);
    return m;
  }, [ops]);

  const numCols = Math.max(compact ? maxStepInOps + 1 : 6, maxStepInOps + (readOnly ? 1 : extraCols));

  const getGateArity = (gateName: string): number => {
    const u = gateName.toUpperCase();
    const cg = customGateMap.get(u);
    if (cg) return cg.num_qubits;
    const bg = builtinMap.get(u);
    if (bg) return bg.numQubits;
    return 1;
  };

  const handleCellClick = (qubitIdx: number, step: number) => {
    if (readOnly) return;

    if (pendingFirstQubit) {
      if (pendingFirstQubit.qubits.includes(qubitIdx)) {
        setPendingFirstQubit(null);
        return;
      }
      const nextQ = [...pendingFirstQubit.qubits, qubitIdx];
      if (nextQ.length >= pendingFirstQubit.needed) {
        onPlaceGate(pendingFirstQubit.gateName, pendingFirstQubit.step, nextQ);
        setPendingFirstQubit(null);
      } else {
        setPendingFirstQubit({ ...pendingFirstQubit, qubits: nextQ });
      }
      return;
    }

    if (!activeToolGate) {
      onSelect({ type: 'none' });
      setMultiSelectedIds([]);
      return;
    }

    const arity = getGateArity(activeToolGate);
    if (arity === 1) {
      onPlaceGate(activeToolGate, step, [qubitIdx]);
    } else {
      setPendingFirstQubit({
        gateName: activeToolGate,
        step,
        qubits: [qubitIdx],
        needed: arity,
      });
    }
  };

  const handleDropOnCell = (e: React.DragEvent, qubitIdx: number, step: number) => {
    e.preventDefault();
    setDragHoverCell(null);
    if (readOnly) return;

    // Case 1: Moving an existing placed gate on the canvas
    const moveOpId = e.dataTransfer.getData('application/quantum-move-op');
    if (moveOpId && onMoveOp) {
      onMoveOp(moveOpId, step, qubitIdx);
      return;
    }

    // Case 2: Placing a new gate dragged from palette or left library
    const gateName = e.dataTransfer.getData('application/quantum-gate') || activeToolGate;
    if (!gateName) return;
    const arity = getGateArity(gateName);
    if (arity === 1) {
      onPlaceGate(gateName, step, [qubitIdx]);
    } else if (arity === 2) {
      const target = qubitIdx + 1 < project.qubits.length ? qubitIdx + 1 : Math.max(0, qubitIdx - 1);
      if (target !== qubitIdx) {
        onPlaceGate(gateName, step, [qubitIdx, target]);
      }
    } else if (arity === 3 && project.qubits.length >= 3) {
      const base = Math.min(qubitIdx, project.qubits.length - 3);
      onPlaceGate(gateName, step, [base, base + 1, base + 2]);
    }
  };

  const formatOpParams = (op: CircuitOpModel) => {
    if (!op.params || op.params.length === 0) return '';
    return op.params
      .map((p) => {
        if (typeof p === 'string') return p;
        const ratio = p / Math.PI;
        if (Math.abs(ratio - 1) < 1e-3) return 'π';
        if (Math.abs(ratio + 1) < 1e-3) return '-π';
        if (Math.abs(ratio - 0.5) < 1e-3) return 'π/2';
        if (Math.abs(ratio + 0.5) < 1e-3) return '-π/2';
        if (Math.abs(ratio - 0.25) < 1e-3) return 'π/4';
        if (Math.abs(ratio + 0.25) < 1e-3) return '-π/4';
        return Number(p).toFixed(2);
      })
      .join(',');
  };

  const getGateStyle = (op: CircuitOpModel) => {
    const u = op.name.toUpperCase();
    const cg = customGateMap.get(u);
    if (cg) {
      return {
        bg: 'bg-[#f5f3ff] dark:bg-[#251d3a]',
        border: 'border-violet-400 dark:border-violet-400/70',
        text: 'text-violet-900 dark:text-violet-200',
        label: cg.display_label || cg.name,
        isCustom: true,
      };
    }
    if (op.kind === 'measure') {
      return {
        bg: 'bg-[#e2e8f0] dark:bg-[#242e3d]',
        border: 'border-slate-400 dark:border-slate-500/80',
        text: 'text-slate-800 dark:text-slate-200',
        label: 'M',
        isCustom: false,
      };
    }
    if (op.kind === 'noise') {
      return {
        bg: 'bg-[#fffbeb] dark:bg-[#332613]',
        border: 'border-amber-400 border-dashed dark:border-amber-400/75',
        text: 'text-amber-900 dark:text-amber-200',
        label: op.name.slice(0, 5),
        isCustom: false,
      };
    }
    const bgInfo = builtinMap.get(u);
    if (bgInfo?.tint === 'teal') {
      return {
        bg: 'bg-[#ecfdf5] dark:bg-[#122b26]',
        border: u === 'RZ' ? 'border-emerald-500 border-dashed dark:border-emerald-400/80' : 'border-emerald-400 dark:border-emerald-400/70',
        text: 'text-emerald-900 dark:text-emerald-200',
        label: bgInfo.label,
        isCustom: false,
      };
    }
    if (bgInfo?.tint === 'indigo') {
      return {
        bg: 'bg-[#eef2ff] dark:bg-[#1c2340]',
        border: 'border-indigo-400 dark:border-indigo-400/75',
        text: 'text-indigo-900 dark:text-indigo-200',
        label: bgInfo.label,
        isCustom: false,
      };
    }
    return {
      bg: 'bg-[#eff6ff] dark:bg-[#15263d]',
      border: 'border-sky-400 dark:border-sky-400/75',
      text: 'text-sky-950 dark:text-sky-200',
      label: bgInfo?.label || op.name,
      isCustom: false,
    };
  };

  // Match backend parallel-layer grouping (disjoint gates on same `step` form one snapshot)
  const orderedLayerSteps = useMemo(() => {
    const exec = [...ops]
      .filter((o) => o.kind !== 'measure')
      .sort((a, b) => a.step - b.step);
    const layerCols: number[] = [];
    let i = 0;
    while (i < exec.length) {
      const cur = exec[i];
      if (cur.kind === 'gate') {
        const used = new Set(cur.qubits);
        let j = i + 1;
        while (
          j < exec.length &&
          exec[j].kind === 'gate' &&
          exec[j].step === cur.step &&
          exec[j].qubits.every((q) => !used.has(q))
        ) {
          exec[j].qubits.forEach((q) => used.add(q));
          j++;
        }
        layerCols.push(cur.step);
        i = j;
      } else {
        layerCols.push(cur.step);
        i++;
      }
    }
    return layerCols;
  }, [ops]);

  const { cursorX, activeExecCol } = useMemo(() => {
    if (timeCursorStep === undefined || maxCursorStep === undefined) {
      return { cursorX: null, activeExecCol: null };
    }
    const clamped = Math.max(0, Math.min(maxCursorStep, timeCursorStep));
    const k0 = Math.floor(clamped);
    const k1 = Math.min(maxCursorStep, k0 + 1);
    const frac = clamped - k0;

    const colOfStep = (sIdx: number): number => {
      if (sIdx <= 0) return -1;
      if (sIdx >= maxCursorStep && ops.some((o) => o.kind === 'measure')) {
        return Math.max(...ops.filter((o) => o.kind === 'measure').map((o) => o.step));
      }
      const layerIdx = Math.min(Math.max(0, sIdx - 1), orderedLayerSteps.length - 1);
      return orderedLayerSteps[layerIdx] ?? numCols - 1;
    };

    let cx: number;
    if (clamped <= 0) {
      cx = LEFT_PAD - 6;
    } else if (frac > 1e-3 && k1 > k0) {
      // Sweep symmetrically across the active executing gate column (from 15% to 85%, passing center at frac=0.5)
      const activeCol = colOfStep(k1);
      const colStart = LEFT_PAD + activeCol * COL_W + COL_W * 0.15;
      const colEnd = LEFT_PAD + activeCol * COL_W + COL_W * 0.85;
      cx = colStart + frac * (colEnd - colStart);
    } else {
      const doneCol = colOfStep(k0);
      cx =
        k0 >= maxCursorStep && ops.some((o) => o.kind === 'measure')
          ? LEFT_PAD + doneCol * COL_W + COL_W * 0.55
          : LEFT_PAD + doneCol * COL_W + COL_W * 0.85;
    }

    let actCol: number | null = null;
    if (clamped > 0.02) {
      const activeStepInt = frac > 0.04 ? k1 : k0;
      if (activeStepInt >= maxCursorStep && ops.some((o) => o.kind === 'measure')) {
        actCol = Math.max(...ops.filter((o) => o.kind === 'measure').map((o) => o.step));
      } else {
        const lIdx = Math.min(Math.max(0, activeStepInt - 1), orderedLayerSteps.length - 1);
        actCol = orderedLayerSteps[lIdx] ?? null;
      }
    }
    return { cursorX: cx, activeExecCol: actCol };
  }, [timeCursorStep, maxCursorStep, orderedLayerSteps, ops, numCols]);

  const canvasW = LEFT_PAD + numCols * COL_W + 52;
  const canvasH = Math.max(1, project.qubits.length) * rowH + (compact ? 22 : 38);

  const renderPaletteButton = (gName: string) => {
    const info = builtinMap.get(gName);
    const active = activeToolGate === gName;
    const tintCls =
      gName === 'MEASURE'
        ? 'bg-[#e2e8f0] dark:bg-[#242e3d] border-slate-400 dark:border-slate-500/80 text-slate-800 dark:text-slate-200'
        : gName === 'DEPOLARIZE'
        ? 'bg-[#fffbeb] dark:bg-[#332613] border-amber-400 border-dashed text-amber-900 dark:text-amber-200'
        : info?.tint === 'teal'
        ? 'bg-[#ecfdf5] dark:bg-[#122b26] border-emerald-400 dark:border-emerald-400/70 text-emerald-900 dark:text-emerald-200'
        : info?.tint === 'indigo'
        ? 'bg-[#eef2ff] dark:bg-[#1c2340] border-indigo-400 dark:border-indigo-400/70 text-indigo-900 dark:text-indigo-200'
        : 'bg-[#eff6ff] dark:bg-[#15263d] border-sky-400 dark:border-sky-400/70 text-sky-950 dark:text-sky-200';
    return (
      <button
        key={gName}
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData('application/quantum-gate', gName);
          onPickGateTool(gName);
        }}
        onClick={() => onPickGateTool(active ? null : gName)}
        title={`${info?.description || gName}（拖放到原理图格点，或点击选中后放置）`}
        className={`h-[26px] min-w-[30px] px-2 rounded-[3px] border text-[11px] font-bold font-mono-num transition-colors cursor-grab active:cursor-grabbing ${tintCls} ${
          active
            ? 'ring-2 ring-[var(--accent)] border-[var(--accent)]'
            : 'hover:border-[var(--accent)]'
        }`}
      >
        {info?.label || gName}
      </button>
    );
  };

  return (
    <div className="relative flex-1 flex flex-col dot-grid-canvas overflow-hidden">
      {/* Docked Industrial EDA Schematic Component Toolbar */}
      {!readOnly && (
        <div className="min-h-[38px] px-3 py-1 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 z-10 shrink-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              1Q
            </span>
            {PALETTE_1Q.map(renderPaletteButton)}
            <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-1" />
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              2Q
            </span>
            {PALETTE_2Q.map(renderPaletteButton)}
            {project.custom_gates.length > 0 && (
              <>
                <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-1" />
                {project.custom_gates.slice(0, 2).map((cg) => {
                  const active = activeToolGate === cg.name;
                  return (
                    <button
                      key={cg.id}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('application/quantum-gate', cg.name);
                        onPickGateTool(cg.name);
                      }}
                      onClick={() => onPickGateTool(active ? null : cg.name)}
                      title={`自定义门: ${cg.display_label || cg.name}`}
                      className={`h-[26px] px-2 rounded-[3px] border bg-[#f5f3ff] dark:bg-[#251d3a] border-violet-400 dark:border-violet-400/70 text-violet-900 dark:text-violet-200 text-[11px] font-bold font-mono-num transition-colors cursor-grab ${
                        active ? 'ring-2 ring-[var(--accent)]' : 'hover:border-[var(--accent)]'
                      }`}
                    >
                      {cg.display_label || cg.name}
                    </button>
                  );
                })}
              </>
            )}
            <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-1" />
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              UTIL
            </span>
            {PALETTE_UTIL.map(renderPaletteButton)}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {multiSelectedIds.length >= 2 && onPackageMacro && (
              <button
                onClick={() => {
                  onPackageMacro(multiSelectedIds);
                  setMultiSelectedIds([]);
                }}
                className="h-[26px] px-2.5 rounded-[3px] bg-[var(--accent)] text-white text-[11px] font-semibold flex items-center gap-1 cursor-pointer"
              >
                <Layers className="w-3 h-3" />
                封装宏门 ({multiSelectedIds.length})
              </button>
            )}
            {onOpenNlModal && (
              <button
                onClick={onOpenNlModal}
                title="用自然语言对话综合量子线路 (Ctrl+K)"
                className="h-[26px] px-2 rounded-[3px] border border-[var(--accent)]/40 bg-[var(--accent-soft)] hover:bg-[var(--accent)]/25 text-[var(--accent)] text-[11px] font-semibold flex items-center gap-1 cursor-pointer transition-colors"
              >
                <Sparkles className="w-3 h-3" />
                <span>AI</span>
              </button>
            )}
            <button
              onClick={handleExportSvg}
              title="导出当前量子原理图矢量文件 (SVG)"
              className="h-[26px] px-2 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[var(--text-secondary)] text-[11px] font-mono-num font-medium flex items-center gap-1 cursor-pointer transition-colors"
            >
              <Download className="w-3 h-3" />
              <span>SVG</span>
            </button>
          </div>
        </div>
      )}

      {/* Multi-qubit placement status banner */}
      {pendingFirstQubit && (
        <div className="px-3 py-1 bg-[var(--accent)] text-white text-[11.5px] font-mono-num font-medium flex items-center justify-between gap-2 z-20 shrink-0">
          <span>
            [放置多比特门 {pendingFirstQubit.gateName}] 已锚定:{' '}
            {pendingFirstQubit.qubits.map((q) => project.qubits[q]?.name || `q${q}`).join(', ')} → 请点击目标比特连线
          </span>
          <button
            onClick={() => setPendingFirstQubit(null)}
            className="px-1.5 py-0.25 rounded-[2px] bg-black/20 hover:bg-black/35 text-[11px] cursor-pointer"
          >
            ESC 取消
          </button>
        </div>
      )}

      {/* Simulate Mode Timeline Scrubber Bar */}
      {readOnly && timeCursorStep !== undefined && maxCursorStep !== undefined && onTimeCursorChange && (
        <div className="py-1 px-3 border-b border-[var(--border-hairline)] bg-[var(--bg-panel)] flex items-center justify-center z-10 shrink-0">
          <div className="w-full max-w-3xl flex items-center gap-2.5">
            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => (onStepJump || onTimeCursorChange)(Math.max(0, Math.ceil(timeCursorStep) - 1))}
                title="跳至上一步"
                className="w-6 h-6 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer transition-colors"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              {onTogglePlayTimeline && (
                <button
                  type="button"
                  onClick={onTogglePlayTimeline}
                  title={isPlayingTimeline ? '暂停连续演化 (Space)' : '播放 60fps 连续脉冲演化 (Space)'}
                  className="w-7 h-6 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] hover:bg-[var(--accent)]/20 flex items-center justify-center cursor-pointer transition-colors"
                >
                  {isPlayingTimeline ? (
                    <Pause className="w-3.5 h-3.5" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current" />
                  )}
                </button>
              )}
              <button
                type="button"
                onClick={() => (onStepJump || onTimeCursorChange)(Math.min(maxCursorStep, Math.floor(timeCursorStep) + 1))}
                title="跳至下一步"
                className="w-6 h-6 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer transition-colors"
              >
                <SkipForward className="w-3.5 h-3.5" />
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
              <div
                className="bg-black/[0.045] dark:bg-white/[0.07] p-0.5 rounded-lg inline-flex text-[10px] font-mono-num shrink-0"
                title="演化播放倍速"
              >
                {([0.5, 1, 2] as const).map((sp) => (
                  <button
                    key={sp}
                    type="button"
                    onClick={() => onPlaybackSpeedChange(sp)}
                    className={`px-1.5 py-0.5 rounded-md cursor-pointer transition-colors ${
                      playbackSpeed === sp
                        ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold shadow-2xs'
                        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
                    {sp}×
                  </button>
                ))}
              </div>
            )}

            <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11px] font-mono-num text-[var(--text-primary)] font-medium shrink-0 min-w-[78px] text-center">
              Step {Math.round(timeCursorStep)} / {maxCursorStep}
            </span>
          </div>
        </div>
      )}

      {titleBadge && (
        <div className="px-3 py-1 border-b border-[var(--border-hairline)] bg-[var(--bg-panel)] text-[11px] font-mono-num font-semibold text-[var(--text-primary)] flex items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="w-1.5 h-3 rounded-[1px] bg-[var(--accent)] shrink-0" />
            <span className="truncate">{titleBadge}</span>
          </div>
          <button
            onClick={handleExportSvg}
            title="导出此线路矢量图 (SVG)"
            className="px-2 py-0.5 rounded-md border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[10.5px] text-[var(--text-secondary)] font-medium flex items-center gap-1 shrink-0 cursor-pointer"
          >
            <Download className="w-3 h-3" />
            SVG
          </button>
        </div>
      )}

      {/* Scrollable Circuit Grid Area (with 2D Viewport Windowing for 50Q~500Q!) */}
      <div
        ref={scrollWrapRef}
        onScroll={(e) => {
          if (project.qubits.length <= 48 && numCols <= 40) return;
          const el = e.currentTarget;
          if (
            Math.abs(el.scrollTop - vpScroll.top) > rowH * 2 ||
            Math.abs(el.scrollLeft - vpScroll.left) > COL_W * 2 ||
            Math.abs(el.clientHeight - vpScroll.height) > 40
          ) {
            setVpScroll({
              top: el.scrollTop,
              left: el.scrollLeft,
              height: el.clientHeight || 720,
              width: el.clientWidth || 1280,
            });
          }
        }}
        className={`flex-1 overflow-auto px-4 ${compact ? 'py-1.5' : 'py-2'}`}
      >
        <div className="relative mx-auto" style={{ width: canvasW, minHeight: canvasH }}>
          {/* Step Ruler Header */}
          {!compact && (
            <div className="flex items-center h-7 text-[11px] font-mono-num text-[var(--text-secondary)]">
              <div
                style={{ width: LEFT_PAD }}
                className="sticky left-0 z-30 pl-2 h-full flex items-center font-sans font-medium bg-[var(--bg-subtle)] border-r border-[var(--border-hairline)]/60"
              >
                Step
              </div>
              {Array.from({ length: numCols }).map((_, c) => (
                <div key={c} style={{ width: COL_W }} className="text-center shrink-0">
                  {c + 1}
                </div>
              ))}
              {!readOnly && (
                <button
                  onClick={() => setExtraCols((v) => v + 1)}
                  title="增加时间列"
                  className="w-6 h-6 rounded-md border border-dashed border-[var(--border-input)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[#0A84FF] hover:border-[#0A84FF] cursor-pointer shrink-0"
                >
                  <Plus className="w-3 h-3" />
                </button>
              )}
            </div>
          )}

          {/* Qubit Wires & Zero-Empty-Cell Coordinate Hit-Testing Stage */}
          <div
            ref={gridRef}
            className={`relative ${
              !readOnly && (activeToolGate || pendingFirstQubit) ? 'cursor-crosshair' : ''
            }`}
            style={{ height: project.qubits.length * rowH }}
            onMouseMove={(e) => {
              if (readOnly || (!activeToolGate && !pendingFirstQubit) || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - LEFT_PAD) / COL_W);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                if (hoverCell?.q !== q || hoverCell?.c !== c) setHoverCell({ q, c });
              } else if (hoverCell !== null) {
                setHoverCell(null);
              }
            }}
            onMouseLeave={() => {
              if (hoverCell !== null) setHoverCell(null);
            }}
            onClick={(e) => {
              if (readOnly || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - LEFT_PAD) / COL_W);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                handleCellClick(q, c);
              }
            }}
            onDragOver={(e) => {
              if (readOnly || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - LEFT_PAD) / COL_W);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                e.preventDefault();
                if (dragHoverCell?.q !== q || dragHoverCell?.c !== c) setDragHoverCell({ q, c });
              }
            }}
            onDragLeave={(e) => {
              if (e.currentTarget === e.target && dragHoverCell !== null) {
                setDragHoverCell(null);
              }
            }}
            onDrop={(e) => {
              if (readOnly || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - LEFT_PAD) / COL_W);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                handleDropOnCell(e, q, c);
              } else {
                setDragHoverCell(null);
              }
            }}
          >
            {/* Blue Vertical Time Cursor in Simulate Mode (60fps GPU translate3d) */}
            {cursorX !== null && (
              <div
                className="absolute top-0 bottom-0 left-0 w-[2px] -ml-[1px] bg-[#0A84FF] shadow-[0_0_10px_rgba(10,132,255,0.6)] pointer-events-none z-30 will-change-transform"
                style={{ transform: `translate3d(${cursorX}px, 0, 0)` }}
              >
                <div className="w-2.5 h-2.5 rounded-full bg-[#0A84FF] ring-2 ring-white dark:ring-[#222226] -ml-[4px] -mt-1 shadow-sm" />
              </div>
            )}

            {/* Single Lightweight Hover / Drag-Drop Preview Highlight */}
            {!readOnly && (dragHoverCell || ((activeToolGate || pendingFirstQubit) && hoverCell)) && (() => {
              const target = dragHoverCell || hoverCell!;
              const isDrag = Boolean(dragHoverCell);
              return (
                <div
                  className={`absolute rounded-lg pointer-events-none transition-transform duration-75 ${
                    isDrag
                      ? 'bg-[#0A84FF]/18 ring-1 ring-inset ring-[#0A84FF]/55'
                      : 'bg-[#0A84FF]/8 ring-1 ring-inset ring-[#0A84FF]/25'
                  }`}
                  style={{
                    left: LEFT_PAD + target.c * COL_W + 4,
                    top: target.q * rowH + 3,
                    width: COL_W - 8,
                    height: rowH - 6,
                  }}
                />
              );
            })()}

            {/* Pending Multi-Qubit First-Click Anchor Dots */}
            {pendingFirstQubit &&
              pendingFirstQubit.qubits.map((pq) => (
                <span
                  key={`pend_${pq}`}
                  className="absolute w-3.5 h-3.5 rounded-full bg-[#0A84FF] ring-4 ring-[#0A84FF]/25 pointer-events-none z-20"
                  style={{
                    left: LEFT_PAD + pendingFirstQubit.step * COL_W + COL_W / 2 - 7,
                    top: pq * rowH + rowH / 2 - 7,
                  }}
                />
              ))}

            {(() => {
              const nQ = project.qubits.length;
              const useVirtual = nQ > 48 || numCols > 40;
              const visRowStart = useVirtual ? Math.max(0, Math.floor(vpScroll.top / rowH) - 8) : 0;
              const visRowEnd = useVirtual
                ? Math.min(nQ, Math.ceil((vpScroll.top + vpScroll.height) / rowH) + 8)
                : nQ;
              const visColStart = useVirtual
                ? Math.max(0, Math.floor(Math.max(0, vpScroll.left - LEFT_PAD) / COL_W) - 6)
                : 0;
              const visColEnd = useVirtual
                ? Math.min(numCols, Math.ceil((Math.max(0, vpScroll.left - LEFT_PAD) + vpScroll.width) / COL_W) + 6)
                : numCols;
              const visibleQubits = project.qubits.slice(visRowStart, visRowEnd);
              const visibleOps = useVirtual
                ? ops.filter((op) => {
                    if (op.step < visColStart || op.step > visColEnd) return false;
                    const vq = op.qubits.filter((q) => q >= 0 && q < nQ);
                    if (vq.length === 0) return false;
                    return Math.max(...vq) >= visRowStart && Math.min(...vq) <= visRowEnd;
                  })
                : ops;

              return (
                <>
                  {visRowStart > 0 && <div style={{ height: visRowStart * rowH }} />}
                  {visibleQubits.map((q) => {
                    const isQSelected = selection.type === 'qubit' && selection.qubitId === q.id;
                    return (
                      <div key={q.id} className="relative flex items-center" style={{ height: rowH }}>
                        {/* Sticky Qubit Label (Opaque Mask so scrolled gates slide cleanly underneath) */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!readOnly) onSelect({ type: 'qubit', qubitId: q.id });
                          }}
                          style={{ width: LEFT_PAD, height: rowH }}
                          className={`sticky left-0 z-30 pr-3 text-right flex items-center justify-end gap-1.5 bg-[var(--bg-subtle)] border-r border-[var(--border-hairline)]/60 transition-colors cursor-pointer shrink-0 ${
                            isQSelected
                              ? 'text-[#0A84FF] font-semibold'
                              : 'text-[var(--text-primary)] hover:text-[#0A84FF]'
                          }`}
                        >
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ backgroundColor: q.color }}
                          />
                          <span className="text-[12.5px] font-mono-num font-semibold truncate">
                            {q.name}
                          </span>
                          <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">
                            {q.init_state.replace('>', '⟩')}
                          </span>
                        </button>

                        {/* Continuous Horizontal Wire */}
                        <div
                          className="absolute h-[1.5px] bg-[#8E8E93]/45 pointer-events-none"
                          style={{
                            left: LEFT_PAD,
                            width: numCols * COL_W,
                            top: rowH / 2,
                          }}
                        />
                      </div>
                    );
                  })}

                  {/* Rendered Operations Overlay (Draggable to reposition in Design Mode!) */}
                  {visibleOps.map((op) => {
                    const validQ = op.qubits.filter((q) => q >= 0 && q < nQ);
                    if (validQ.length === 0) return null;
                    const isSelected =
                      (selection.type === 'op' && selection.opId === op.id) ||
                      multiSelectedIds.includes(op.id);
                    const isExecActive = readOnly && activeExecCol !== null && op.step === activeExecCol;
                    const isFutureStep =
                      readOnly &&
                      timeCursorStep !== undefined &&
                      (activeExecCol === null || op.step > activeExecCol);
                    const simStateCls = isExecActive
                      ? 'studio-gate-active'
                      : isFutureStep
                      ? 'opacity-60'
                      : '';
                    const style = getGateStyle(op);
                    const xCenter = LEFT_PAD + op.step * COL_W + COL_W / 2;
                    const uname = op.name.toUpperCase();
                    const paramTxt = formatOpParams(op);

                    const handleOpClick = (e: React.MouseEvent) => {
                      e.stopPropagation();
                      if (readOnly) return;
                      if (e.shiftKey) {
                        setMultiSelectedIds((prev) =>
                          prev.includes(op.id) ? prev.filter((id) => id !== op.id) : [...prev, op.id]
                        );
                      } else {
                        setMultiSelectedIds([]);
                        onSelect({ type: 'op', opId: op.id });
                      }
                    };

                    const dragHandlers = readOnly
                      ? {}
                      : {
                          draggable: true,
                          onDragStart: (e: React.DragEvent) => {
                            e.stopPropagation();
                            e.dataTransfer.setData('application/quantum-move-op', op.id);
                          },
                        };

                    // Measurement across multiple qubits
                    if (op.kind === 'measure') {
                      const mH = compact ? 26 : 34;
                      const mW = compact ? 32 : 38;
                      return (
                        <React.Fragment key={op.id}>
                          {validQ.map((qIdx) => (
                            <div
                              key={`${op.id}_m_${qIdx}`}
                              {...dragHandlers}
                              onClick={handleOpClick}
                              style={{
                                left: xCenter - mW / 2,
                                top: qIdx * rowH + rowH / 2 - mH / 2,
                                width: mW,
                                height: mH,
                              }}
                              className={`studio-gate-node absolute rounded-lg border flex items-center justify-center cursor-pointer z-10 ${style.bg} ${style.border} ${style.text} ${simStateCls} ${
                                isSelected ? 'ring-2 ring-[#0A84FF] ring-offset-1' : 'hover:border-[#0A84FF]/60'
                              }`}
                              title="测量 Measurement（可拖拽移动时间列）"
                            >
                              <Gauge className={compact ? 'w-3.5 h-3.5 stroke-[1.7]' : 'w-4 h-4 stroke-[1.7]'} />
                            </div>
                          ))}
                        </React.Fragment>
                      );
                    }

                    // Standard SWAP / CZ / CNOT / CX / CCZ / CCX / TOFFOLI vertical connector rendering
                    if (
                      validQ.length >= 2 &&
                      (uname === 'SWAP' ||
                        uname === 'CZ' ||
                        uname === 'CNOT' ||
                        uname === 'CX' ||
                        uname === 'CCZ' ||
                        uname === 'CCX' ||
                        uname === 'TOFFOLI')
                    ) {
                      const minQ = Math.min(...validQ);
                      const maxQ = Math.max(...validQ);
                      const topY = minQ * rowH + rowH / 2;
                      const botY = maxQ * rowH + rowH / 2;
                      const isTargetXor =
                        uname === 'CNOT' || uname === 'CX' || uname === 'CCX' || uname === 'TOFFOLI';
                      const targetQ = validQ[validQ.length - 1];
                      const controls =
                        uname === 'SWAP'
                          ? []
                          : uname === 'CZ' || uname === 'CCZ'
                          ? validQ
                          : validQ.slice(0, -1);
                      const topoWarn = isPairTopologyViolated(validQ);

                      return (
                        <div
                          key={op.id}
                          {...dragHandlers}
                          onClick={handleOpClick}
                          style={{
                            left: xCenter - 18,
                            top: topY - 14,
                            width: 36,
                            height: botY - topY + 28,
                          }}
                          className={`studio-gate-node group absolute cursor-grab active:cursor-grabbing z-10 rounded-lg ${simStateCls} ${
                            isSelected
                              ? 'bg-[#0A84FF]/10 ring-2 ring-[#0A84FF]'
                              : topoWarn
                              ? 'bg-amber-500/12 ring-1 ring-amber-500/80 ring-dashed'
                              : 'hover:bg-indigo-500/8'
                          }`}
                          title={
                            topoWarn
                              ? `${op.name} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) — ⚠ 超出当前硬件拓扑/阻塞半径，优化编译时将自动插入 SWAP 路由`
                              : `${op.name} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) · 可拖拽移动`
                          }
                        >
                          <div
                            className={`absolute w-[2px] ${
                              topoWarn ? 'bg-amber-500' : 'bg-[#1E3A8A] dark:bg-indigo-400'
                            }`}
                            style={{ left: 17, top: 14, height: botY - topY }}
                          />
                          {uname === 'SWAP' &&
                            validQ.map((sq) => (
                              <svg
                                key={sq}
                                width="16"
                                height="16"
                                viewBox="0 0 16 16"
                                className={`absolute ${
                                  topoWarn ? 'text-amber-500' : 'text-[#1E3A8A] dark:text-indigo-400'
                                }`}
                                style={{ left: 10, top: (sq - minQ) * rowH + 6 }}
                              >
                                <path
                                  d="M3 3L13 13M13 3L3 13"
                                  stroke="currentColor"
                                  strokeWidth="2.2"
                                  strokeLinecap="round"
                                />
                              </svg>
                            ))}
                          {controls.map((cq) => (
                            <div
                              key={cq}
                              className={`absolute w-3 h-3 rounded-full ${
                                topoWarn ? 'bg-amber-500' : 'bg-[#1E3A8A] dark:bg-indigo-400'
                              }`}
                              style={{ left: 12, top: (cq - minQ) * rowH + 8 }}
                            />
                          ))}
                          {isTargetXor && (
                            <div
                              className={`absolute w-7 h-7 rounded-full border-2 bg-[var(--bg-panel)] flex items-center justify-center font-bold text-sm leading-none ${
                                topoWarn
                                  ? 'border-amber-500 text-amber-600 dark:text-amber-400'
                                  : 'border-[#1E3A8A] dark:border-indigo-400 text-[#1E3A8A] dark:text-indigo-300'
                              }`}
                              style={{ left: 4, top: (targetQ - minQ) * rowH }}
                            >
                              +
                            </div>
                          )}
                          {(uname === 'CZ' || uname === 'CCZ') && (
                            <span
                              className={`absolute left-6 text-[9.5px] font-mono-num font-bold border px-1 rounded ${
                                topoWarn
                                  ? 'text-amber-700 dark:text-amber-200 bg-amber-50 dark:bg-amber-950/90 border-amber-400'
                                  : 'text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/80 border-indigo-200 dark:border-indigo-700'
                              }`}
                              style={{ top: (botY - topY) / 2 + 5 }}
                            >
                              {uname}
                            </span>
                          )}
                          {topoWarn && !compact && (
                            <span
                              className="absolute -top-2.5 -right-2 w-4 h-4 rounded-full bg-amber-500 text-white flex items-center justify-center shadow-xs"
                              title="超出当前硬件拓扑/里德堡阻塞半径，需路由"
                            >
                              <AlertTriangle className="w-2.5 h-2.5" />
                            </span>
                          )}
                        </div>
                      );
                    }

                    // Multi-qubit custom or entangling gate box
                    if (validQ.length >= 2) {
                      const minQ = Math.min(...validQ);
                      const maxQ = Math.max(...validQ);
                      const padHalf = compact ? 14 : 20;
                      const boxTop = minQ * rowH + rowH / 2 - padHalf;
                      const boxH = (maxQ - minQ) * rowH + padHalf * 2;
                      const topoWarn = isPairTopologyViolated(validQ);
                      return (
                        <div
                          key={op.id}
                          {...dragHandlers}
                          onClick={handleOpClick}
                          style={{
                            left: xCenter - (compact ? 30 : 33),
                            top: boxTop,
                            width: compact ? 60 : 66,
                            height: boxH,
                          }}
                          className={`studio-gate-node absolute rounded-xl border flex flex-col items-center justify-center px-1 cursor-grab active:cursor-grabbing z-10 shadow-2xs ${style.bg} ${
                            topoWarn ? 'border-amber-500 border-dashed' : style.border
                          } ${style.text} ${simStateCls} ${
                            isSelected ? 'ring-2 ring-[#0A84FF] ring-offset-1' : 'hover:border-[#0A84FF]/70'
                          }`}
                          title={
                            topoWarn
                              ? `${style.label} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) — ⚠ 超出当前硬件拓扑/阻塞半径，优化编译时将自动路由`
                              : `${style.label} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) · 可拖拽移动`
                          }
                        >
                          {topoWarn && !compact && (
                            <span className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-amber-500 text-white flex items-center justify-center shadow-xs">
                              <AlertTriangle className="w-2.5 h-2.5" />
                            </span>
                          )}
                          {/* Active qubit side port dots so skipped wires are clearly distinguishable */}
                          {validQ.map((aq) => (
                            <span
                              key={aq}
                              className="absolute -left-[4px] w-2 h-2 rounded-full bg-purple-500 border border-white"
                              style={{ top: (aq - minQ) * rowH + padHalf - 4 }}
                            />
                          ))}
                          <span className={`${compact ? 'text-[11px]' : 'text-[12px]'} font-bold font-mono-num leading-tight text-center`}>
                            {style.label}
                          </span>
                          {paramTxt && !compact && (
                            <span className="text-[9.5px] font-mono-num opacity-80 truncate max-w-full mt-0.5">
                              {paramTxt}
                            </span>
                          )}
                          {style.isCustom && !compact && (
                            <span className="mt-1 px-1.5 py-0 rounded bg-white/80 dark:bg-black/25 text-[9px] font-medium">
                              自定义
                            </span>
                          )}
                        </div>
                      );
                    }

                    // Single-qubit gate or noise box
                    const qIdx = validQ[0];
                    const boxW = paramTxt && !compact ? Math.min(78, Math.max(46, Math.round(paramTxt.length * 6.4 + 14))) : compact ? 42 : 44;
                    const boxH = compact ? 26 : 38;
                    return (
                      <div
                        key={op.id}
                        {...dragHandlers}
                        onClick={handleOpClick}
                        style={{
                          left: xCenter - boxW / 2,
                          top: qIdx * rowH + rowH / 2 - boxH / 2,
                          width: boxW,
                          height: boxH,
                        }}
                        className={`studio-gate-node group absolute rounded-lg border flex flex-col items-center justify-center px-1 cursor-grab active:cursor-grabbing z-10 shadow-2xs ${style.bg} ${style.border} ${style.text} ${simStateCls} ${
                          isSelected ? 'ring-2 ring-[#0A84FF] ring-offset-1' : 'hover:border-[#0A84FF]/70'
                        }`}
                        title={`${op.name}${paramTxt ? `(${paramTxt})` : ''} @ ${project.qubits[qIdx]?.name} · 可拖拽移动`}
                      >
                        <span className="text-[12px] font-bold font-mono-num leading-none">
                          {style.label}
                        </span>
                        {paramTxt && !compact && (
                          <span className="text-[9.5px] font-mono-num opacity-85 leading-none mt-1 truncate max-w-full">
                            {paramTxt}
                          </span>
                        )}
                        {!readOnly && isSelected && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteOp(op.id);
                            }}
                            title="删除此门"
                            className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-red-500 text-white flex items-center justify-center shadow-xs hover:bg-red-600 cursor-pointer"
                          >
                            <Trash2 className="w-2.5 h-2.5" />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </>
              );
            })()}
          </div>

          {/* Dashed "+ 添加比特" Row Directly Below Last Wire (Design Mode) */}
          {!readOnly && (
            <button
              onClick={onAddQubit}
              style={{ marginLeft: LEFT_PAD, width: Math.max(360, numCols * COL_W) }}
              className="mt-3 h-9 rounded-lg border border-dashed border-[var(--border-input)] hover:border-[#0A84FF] hover:bg-[#0A84FF]/5 text-[var(--text-secondary)] hover:text-[#0A84FF] text-[12px] font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>添加比特</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
