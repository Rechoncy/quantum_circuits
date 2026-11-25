import React, { useMemo, useRef, useState, useCallback } from 'react';
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
  MousePointer,
  ZoomIn,
  ZoomOut,
  RotateCcw,
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
  onDeleteMultipleOps?: (opIds: string[]) => void;
  onAddQubit: () => void;
  onPackageMacro?: (selectedOpIds: string[]) => void;
  onOpenNlModal?: () => void;
  // Simulate mode timeline cursor
  timeCursorStep?: number;
  maxCursorStep?: number;
  cursorReadoutLabel?: string | null;
  onTimeCursorChange?: (step: number) => void;
  onStepJump?: (targetStep: number) => void;
  isPlayingTimeline?: boolean;
  onTogglePlayTimeline?: () => void;
  playbackSpeed?: 0.5 | 1 | 2;
  onPlaybackSpeedChange?: (speed: 0.5 | 1 | 2) => void;
  titleBadge?: string;
}

const BASE_ROW_H = 54;
const BASE_ROW_H_COMPACT = 35;
const BASE_COL_W = 86;
const BASE_LEFT_PAD = 92;

const PALETTE_1Q = ['H', 'X', 'SX', 'RXY', 'RX', 'RY', 'RZ'];
const PALETTE_2Q = ['CZ', 'CNOT', 'ISWAP', 'SWAP'];
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
  onDeleteMultipleOps,
  onAddQubit,
  onPackageMacro,
  onOpenNlModal,
  timeCursorStep,
  maxCursorStep,
  cursorReadoutLabel,
  onTimeCursorChange,
  onStepJump,
  isPlayingTimeline,
  onTogglePlayTimeline,
  playbackSpeed = 1,
  onPlaybackSpeedChange,
  titleBadge,
}) => {
  const ops = operationsOverride ?? project.operations;
  const [zoom, setZoom] = useState(1);
  const effZoom = compact ? 1 : zoom;
  const rowH = Math.round((compact ? BASE_ROW_H_COMPACT : BASE_ROW_H) * effZoom);
  const colW = Math.round(BASE_COL_W * effZoom);
  const leftPad = Math.round(BASE_LEFT_PAD * Math.max(0.85, Math.min(1.15, effZoom)));

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
  const [marquee, setMarquee] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  const [isScrubbingCanvas, setIsScrubbingCanvas] = useState(false);
  const suppressNextClickRef = useRef(false);

  const [vpScroll, setVpScroll] = useState({ top: 0, left: 0, height: 720, width: 1280 });
  const gridRef = useRef<HTMLDivElement>(null);
  const scrollWrapRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);

  const customGateMap = useMemo(() => {
    const m = new Map<string, CustomGateDef>();
    project.custom_gates.forEach((g) => m.set(g.name.toUpperCase(), g));
    return m;
  }, [project.custom_gates]);

  const builtinMap = useMemo(() => {
    const m = new Map(BUILTIN_GATES.map((g) => [g.name.toUpperCase(), g]));
    return m;
  }, []);

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

    const moveOpId = e.dataTransfer.getData('application/quantum-move-op');
    if (moveOpId && onMoveOp) {
      onMoveOp(moveOpId, step, qubitIdx);
      return;
    }

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
      const q1 = (qubitIdx + 1) % project.qubits.length;
      const q2 = (qubitIdx + 2) % project.qubits.length;
      onPlaceGate(gateName, step, [qubitIdx, q1, q2]);
    }
  };

  const formatOpParams = (op: CircuitOpModel): string => {
    if (!op.params || op.params.length === 0) return '';
    return op.params
      .map((p) => {
        if (typeof p === 'string') return p;
        const ratio = p / Math.PI;
        if (Math.abs(p) < 1e-4) return '0';
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

  const getGateStyleByName = (gateName: string, kind: CircuitOpModel['kind'] = 'gate') => {
    const u = gateName.toUpperCase();
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
    if (kind === 'measure' || u === 'MEASURE') {
      return {
        bg: 'bg-[#e2e8f0] dark:bg-[#242e3d]',
        border: 'border-slate-400 dark:border-slate-500/80',
        text: 'text-slate-800 dark:text-slate-200',
        label: 'M',
        isCustom: false,
      };
    }
    if (kind === 'noise' || u === 'DEPOLARIZE' || u === 'PHASE_FLIP') {
      return {
        bg: 'bg-[#fffbeb] dark:bg-[#332613]',
        border: 'border-amber-400 border-dashed dark:border-amber-400/75',
        text: 'text-amber-900 dark:text-amber-200',
        label: u.slice(0, 5),
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
      label: bgInfo?.label || gateName,
      isCustom: false,
    };
  };

  const getGateStyle = (op: CircuitOpModel) => getGateStyleByName(op.name, op.kind);

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

  // Step <-> Canvas X Pixel bidirectional mapping for 60fps Oscilloscope Probe Scrubbing!
  const stepToX = useCallback(
    (stepVal: number): number => {
      if (maxCursorStep === undefined || maxCursorStep <= 0) return leftPad - 6;
      const clamped = Math.max(0, Math.min(maxCursorStep, stepVal));
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

      if (clamped <= 0) return leftPad - 6;
      if (frac > 1e-3 && k1 > k0) {
        const activeCol = colOfStep(k1);
        const colStart = leftPad + activeCol * colW + colW * 0.15;
        const colEnd = leftPad + activeCol * colW + colW * 0.85;
        return colStart + frac * (colEnd - colStart);
      }
      const doneCol = colOfStep(k0);
      return k0 >= maxCursorStep && ops.some((o) => o.kind === 'measure')
        ? leftPad + doneCol * colW + colW * 0.55
        : leftPad + doneCol * colW + colW * 0.85;
    },
    [maxCursorStep, ops, orderedLayerSteps, numCols, leftPad, colW]
  );

  // Monotonic piecewise-linear inverse mapping X pixel -> continuous step [0..maxCursorStep]
  const xPxToStep = useCallback(
    (xPx: number, snapNearInteger: boolean): number => {
      if (!maxCursorStep || maxCursorStep <= 0) return 0;
      if (xPx <= leftPad - 2) return 0;
      // Sample monotonic knot table of stepToX(0..maxCursorStep)
      const knots: { s: number; x: number }[] = [{ s: 0, x: leftPad - 6 }];
      for (let s = 1; s <= maxCursorStep; s++) {
        const prevX = knots[knots.length - 1].x;
        const sx = Math.max(prevX + 4, stepToX(s));
        knots.push({ s, x: sx });
      }
      if (xPx >= knots[knots.length - 1].x) return maxCursorStep;
      for (let i = 0; i < knots.length - 1; i++) {
        const a = knots[i];
        const b = knots[i + 1];
        if (xPx >= a.x && xPx <= b.x) {
          const t = (xPx - a.x) / Math.max(1, b.x - a.x);
          const raw = a.s + t * (b.s - a.s);
          if (snapNearInteger && Math.abs(raw - Math.round(raw)) < 0.08) {
            return Math.round(raw);
          }
          return Number(raw.toFixed(3));
        }
      }
      return maxCursorStep;
    },
    [maxCursorStep, leftPad, stepToX]
  );

  const { cursorX, activeExecCol } = useMemo(() => {
    if (timeCursorStep === undefined || maxCursorStep === undefined) {
      return { cursorX: null, activeExecCol: null };
    }
    const clamped = Math.max(0, Math.min(maxCursorStep, timeCursorStep));
    const k0 = Math.floor(clamped);
    const k1 = Math.min(maxCursorStep, k0 + 1);
    const frac = clamped - k0;
    const cx = stepToX(clamped);

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
  }, [timeCursorStep, maxCursorStep, orderedLayerSteps, ops, stepToX]);

  const canvasW = leftPad + numCols * colW + 52;
  const canvasH = Math.max(1, project.qubits.length) * rowH + (compact ? 22 : 38);

  const isSelectMode = !readOnly && !activeToolGate && !pendingFirstQubit;

  // Interactive Simulate Canvas Scrubbing handler
  const startSimulateCanvasScrub = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!readOnly || !onTimeCursorChange || !gridRef.current || e.button !== 0) return;
    e.preventDefault();
    setIsScrubbingCanvas(true);
    const updateFromEvent = (clientX: number, shiftKey: boolean) => {
      if (!gridRef.current || !onTimeCursorChange) return;
      const rect = gridRef.current.getBoundingClientRect();
      const xPx = clientX - rect.left;
      onTimeCursorChange(xPxToStep(xPx, !shiftKey));
    };
    updateFromEvent(e.clientX, e.shiftKey);

    const onMove = (ev: PointerEvent) => updateFromEvent(ev.clientX, ev.shiftKey);
    const onUp = () => {
      setIsScrubbingCanvas(false);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Interactive Custom Stepped Timeline Scrubber Track handler
  const startTrackScrub = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!onTimeCursorChange || maxCursorStep === undefined || !trackRef.current || e.button !== 0) return;
    e.preventDefault();
    const updateTrack = (clientX: number, shiftKey: boolean) => {
      if (!trackRef.current || !onTimeCursorChange || maxCursorStep === undefined) return;
      const rect = trackRef.current.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
      const raw = ratio * maxCursorStep;
      const snapped = !shiftKey && Math.abs(raw - Math.round(raw)) < 0.06 ? Math.round(raw) : Number(raw.toFixed(3));
      onTimeCursorChange(snapped);
    };
    updateTrack(e.clientX, e.shiftKey);
    const onMove = (ev: PointerEvent) => updateTrack(ev.clientX, ev.shiftKey);
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Design Mode Marquee Box Selection handler
  const startMarqueeDrag = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isSelectMode || e.button !== 0 || !gridRef.current) return;
    // Only start marquee if clicking directly on the empty wire/grid stage (not on a gate or sticky label)
    if ((e.target as HTMLElement).closest('.studio-gate-node, button')) return;
    const rect = gridRef.current.getBoundingClientRect();
    const x0 = e.clientX - rect.left;
    const y0 = e.clientY - rect.top;
    if (x0 < leftPad) return;

    const onMove = (ev: MouseEvent) => {
      if (!gridRef.current) return;
      const r = gridRef.current.getBoundingClientRect();
      const x1 = ev.clientX - r.left;
      const y1 = ev.clientY - r.top;
      if (Math.hypot(x1 - x0, y1 - y0) > 5) {
        suppressNextClickRef.current = true;
        setMarquee({ x0, y0, x1, y1 });
        const minX = Math.min(x0, x1);
        const maxX = Math.max(x0, x1);
        const minY = Math.min(y0, y1);
        const maxY = Math.max(y0, y1);
        const hitIds = ops
          .filter((op) => {
            const vq = op.qubits.filter((q) => q >= 0 && q < project.qubits.length);
            if (vq.length === 0) return false;
            const gx = leftPad + op.step * colW + colW / 2;
            const gyTop = Math.min(...vq) * rowH + rowH / 2;
            const gyBot = Math.max(...vq) * rowH + rowH / 2;
            return gx >= minX && gx <= maxX && gyBot >= minY && gyTop <= maxY;
          })
          .map((o) => o.id);
        setMultiSelectedIds(hitIds);
      }
    };
    const onUp = () => {
      setMarquee(null);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

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
          setPendingFirstQubit(null);
          onPickGateTool(gName);
        }}
        onClick={() => {
          setPendingFirstQubit(null);
          onPickGateTool(active ? null : gName);
        }}
        title={`${info?.description || gName}（点击吸附至光标放置，右键或按 V 键退回指针选择）`}
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

  const trackPct =
    timeCursorStep !== undefined && maxCursorStep && maxCursorStep > 0
      ? Math.max(0, Math.min(100, (timeCursorStep / maxCursorStep) * 100))
      : 0;

  return (
    <div
      className="relative flex-1 flex flex-col dot-grid-canvas overflow-hidden"
      onContextMenu={(e) => {
        if (readOnly) return;
        if (pendingFirstQubit || activeToolGate) {
          e.preventDefault();
          setPendingFirstQubit(null);
          onPickGateTool(null);
        }
      }}
    >
      {/* Docked Industrial EDA Schematic Component Toolbar */}
      {!readOnly && (
        <div className="min-h-[38px] px-3 py-1 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 z-10 shrink-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* Explicit Select / Pointer Mode Tool (V / Esc) */}
            <button
              type="button"
              onClick={() => {
                setPendingFirstQubit(null);
                onPickGateTool(null);
              }}
              title="指针选择与矩形框选工具 (快捷键 V / Esc，或画布右键切换)"
              className={`h-[26px] px-2 rounded-[3px] border text-[11px] font-semibold flex items-center gap-1 cursor-pointer transition-colors ${
                isSelectMode
                  ? 'bg-[var(--accent)] border-[var(--accent)] text-white'
                  : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent)]'
              }`}
            >
              <MousePointer className="w-3 h-3" />
              <span>选择</span>
            </button>

            <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-0.5" />
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              1Q
            </span>
            {PALETTE_1Q.map(renderPaletteButton)}
            <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-0.5" />
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              2Q
            </span>
            {PALETTE_2Q.map(renderPaletteButton)}
            {project.custom_gates.length > 0 && (
              <>
                <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-0.5" />
                {project.custom_gates.slice(0, 2).map((cg) => {
                  const active = activeToolGate === cg.name;
                  return (
                    <button
                      key={cg.id}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('application/quantum-gate', cg.name);
                        setPendingFirstQubit(null);
                        onPickGateTool(cg.name);
                      }}
                      onClick={() => {
                        setPendingFirstQubit(null);
                        onPickGateTool(active ? null : cg.name);
                      }}
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
            <div className="w-[1px] h-4 bg-[var(--border-hairline)] mx-0.5" />
            <span className="text-[9.5px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-tertiary)] mr-0.5">
              UTIL
            </span>
            {PALETTE_UTIL.map(renderPaletteButton)}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {multiSelectedIds.length >= 2 && (
              <>
                {onDeleteMultipleOps && (
                  <button
                    onClick={() => {
                      onDeleteMultipleOps(multiSelectedIds);
                      setMultiSelectedIds([]);
                    }}
                    className="h-[26px] px-2 rounded-[3px] bg-red-600/15 hover:bg-red-600/25 border border-red-500/40 text-red-500 text-[11px] font-semibold flex items-center gap-1 cursor-pointer"
                  >
                    <Trash2 className="w-3 h-3" />
                    删除 ({multiSelectedIds.length})
                  </button>
                )}
                {onPackageMacro && (
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
              </>
            )}

            {/* Schematic Zoom Controls (60% ~ 150%) */}
            <div className="hidden md:flex items-center h-[26px] rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10.5px] font-mono-num">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(0.6, Number((z - 0.1).toFixed(2))))}
                title="缩小原理图 (Ctrl + 滚轮向下)"
                className="px-1.5 h-full text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <ZoomOut className="w-3 h-3" />
              </button>
              <button
                type="button"
                onClick={() => setZoom(1)}
                title="点击重置为 100% 原始比例"
                className="px-1.5 h-full border-x border-[var(--border-hairline)] text-[var(--text-primary)] hover:text-[var(--accent)] cursor-pointer min-w-[42px]"
              >
                {Math.round(zoom * 100)}%
              </button>
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(1.5, Number((z + 0.1).toFixed(2))))}
                title="放大原理图 (Ctrl + 滚轮向上)"
                className="px-1.5 h-full text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <ZoomIn className="w-3 h-3" />
              </button>
              {zoom !== 1 && (
                <button
                  type="button"
                  onClick={() => setZoom(1)}
                  title="复位缩放"
                  className="px-1 h-full border-l border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--accent)] cursor-pointer"
                >
                  <RotateCcw className="w-2.5 h-2.5" />
                </button>
              )}
            </div>

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
            [正在连线 {pendingFirstQubit.gateName}] 已锚定控制端:{' '}
            {pendingFirstQubit.qubits.map((q) => project.qubits[q]?.name || `q${q}`).join(', ')} → 移动光标预览橡皮筋连线并点击目标比特（右键或 ESC 取消）
          </span>
          <button
            onClick={() => setPendingFirstQubit(null)}
            className="px-1.5 py-0.25 rounded-[2px] bg-black/20 hover:bg-black/35 text-[11px] cursor-pointer"
          >
            ESC 取消
          </button>
        </div>
      )}

      {/* Simulate Mode Oscilloscope Precision Stepped Scrubber Bar */}
      {readOnly && timeCursorStep !== undefined && maxCursorStep !== undefined && onTimeCursorChange && (
        <div className="py-1 px-3 border-b border-[var(--border-hairline)] bg-[var(--bg-panel)] flex items-center justify-center z-10 shrink-0 select-none">
          <div className="w-full max-w-4xl flex items-center gap-2.5">
            <div className="flex items-center gap-0.5 shrink-0">
              <button
                type="button"
                onClick={() => (onStepJump || onTimeCursorChange)(Math.max(0, Math.ceil(timeCursorStep - 0.02) - 1))}
                title="上一步门边界 (← 方向键)"
                className="w-6 h-6 rounded-[3px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer transition-colors"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              {onTogglePlayTimeline && (
                <button
                  type="button"
                  onClick={onTogglePlayTimeline}
                  title={isPlayingTimeline ? '暂停连续演化 (Space)' : '播放 60fps 连续脉冲演化 (Space)'}
                  className="w-7 h-6 rounded-[3px] bg-[var(--accent-soft)] text-[var(--accent)] hover:bg-[var(--accent)]/20 flex items-center justify-center cursor-pointer transition-colors"
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
                onClick={() => (onStepJump || onTimeCursorChange)(Math.min(maxCursorStep, Math.floor(timeCursorStep + 0.02) + 1))}
                title="下一步门边界 (→ 方向键)"
                className="w-6 h-6 rounded-[3px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center justify-center cursor-pointer transition-colors"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Custom Stepped Oscilloscope Scrubber Track with Gate Tick Marks */}
            <div
              ref={trackRef}
              onPointerDown={startTrackScrub}
              title="点击或拖拽游标探针（按住 Shift 可无吸附微调脉冲内时刻，也可直接拖动下方原理图或波形图游标）"
              className="relative flex-1 h-6 flex items-center cursor-ew-resize group"
            >
              {/* Rail Base */}
              <div className="w-full h-[5px] rounded-[1px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] relative overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 bg-[var(--accent)]"
                  style={{ width: `${trackPct}%` }}
                />
              </div>

              {/* Discrete Step Tick Marks (up to 40 steps) */}
              {maxCursorStep > 0 &&
                maxCursorStep <= 40 &&
                Array.from({ length: maxCursorStep + 1 }).map((_, sIdx) => {
                  const leftPct = (sIdx / maxCursorStep) * 100;
                  const passed = sIdx <= timeCursorStep + 0.02;
                  return (
                    <span
                      key={sIdx}
                      style={{ left: `${leftPct}%` }}
                      className={`absolute top-1/2 -translate-y-1/2 -ml-[1px] w-[2px] h-2.5 pointer-events-none ${
                        passed ? 'bg-[#60a5fa]' : 'bg-[var(--border-input)]'
                      }`}
                    />
                  );
                })}

              {/* Rectangular Precision Probe Thumb */}
              <div
                style={{ left: `${trackPct}%` }}
                className="absolute top-1/2 -translate-y-1/2 -ml-[5px] w-[10px] h-4 rounded-[2px] bg-[var(--accent)] border border-white shadow-xs pointer-events-none"
              />
            </div>

            {onPlaybackSpeedChange && (
              <div
                className="bg-[var(--bg-subtle)] border border-[var(--border-hairline)] p-0.5 rounded-[3px] inline-flex text-[10px] font-mono-num shrink-0"
                title="演化播放倍速"
              >
                {([0.5, 1, 2] as const).map((sp) => (
                  <button
                    key={sp}
                    type="button"
                    onClick={() => onPlaybackSpeedChange(sp)}
                    className={`px-1.5 py-0.5 rounded-[2px] cursor-pointer transition-colors ${
                      playbackSpeed === sp
                        ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold'
                        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
                    {sp}×
                  </button>
                ))}
              </div>
            )}

            <span className="px-2 py-0.5 rounded-[3px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[11px] font-mono-num text-[var(--text-primary)] font-medium shrink-0 min-w-[92px] text-center">
              Step {Number.isInteger(Math.round(timeCursorStep * 10) / 10) ? Math.round(timeCursorStep) : timeCursorStep.toFixed(1)} / {maxCursorStep}
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
            className="px-2 py-0.5 rounded-[3px] border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[10.5px] text-[var(--text-secondary)] font-medium flex items-center gap-1 shrink-0 cursor-pointer"
          >
            <Download className="w-3 h-3" />
            SVG
          </button>
        </div>
      )}

      {/* Scrollable Circuit Grid Area (with 2D Viewport Windowing + Ctrl+Wheel Zoom) */}
      <div
        ref={scrollWrapRef}
        onWheel={(e) => {
          if (compact || !(e.ctrlKey || e.metaKey)) return;
          e.preventDefault();
          const delta = e.deltaY < 0 ? 0.1 : -0.1;
          setZoom((z) => Math.max(0.6, Math.min(1.5, Number((z + delta).toFixed(2)))));
        }}
        onScroll={(e) => {
          if (project.qubits.length <= 48 && numCols <= 40) return;
          const el = e.currentTarget;
          if (
            Math.abs(el.scrollTop - vpScroll.top) > rowH * 2 ||
            Math.abs(el.scrollLeft - vpScroll.left) > colW * 2 ||
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
            <div className="flex items-center h-7 text-[11px] font-mono-num text-[var(--text-secondary)] select-none">
              <div
                style={{ width: leftPad }}
                className="sticky left-0 z-30 pl-2 h-full flex items-center font-sans font-medium bg-[var(--bg-subtle)] border-r border-[var(--border-hairline)]/60"
              >
                Step
              </div>
              {Array.from({ length: numCols }).map((_, c) => {
                const isHoverCol = !readOnly && (activeToolGate || pendingFirstQubit) && hoverCell?.c === c;
                return (
                  <div
                    key={c}
                    style={{ width: colW }}
                    className={`text-center shrink-0 transition-colors ${
                      isHoverCol ? 'text-[var(--accent)] font-bold' : ''
                    }`}
                  >
                    {c + 1}
                  </div>
                );
              })}
              {!readOnly && (
                <button
                  onClick={() => setExtraCols((v) => v + 1)}
                  title="增加时间列"
                  className="w-6 h-6 rounded-[3px] border border-dashed border-[var(--border-input)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--accent)] hover:border-[var(--accent)] cursor-pointer shrink-0"
                >
                  <Plus className="w-3 h-3" />
                </button>
              )}
            </div>
          )}

          {/* Qubit Wires & Interactive Coordinate Stage */}
          <div
            ref={gridRef}
            onPointerDown={readOnly && onTimeCursorChange ? startSimulateCanvasScrub : undefined}
            onMouseDown={!readOnly ? startMarqueeDrag : undefined}
            className={`relative ${
              readOnly && onTimeCursorChange
                ? isScrubbingCanvas
                  ? 'cursor-ew-resize select-none'
                  : 'cursor-pointer'
                : !readOnly && (activeToolGate || pendingFirstQubit)
                ? 'cursor-crosshair'
                : 'cursor-default'
            }`}
            style={{ height: project.qubits.length * rowH }}
            onMouseMove={(e) => {
              if (readOnly || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - leftPad) / colW);
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
              if (suppressNextClickRef.current) {
                suppressNextClickRef.current = false;
                return;
              }
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - leftPad) / colW);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                handleCellClick(q, c);
              }
            }}
            onDragOver={(e) => {
              if (readOnly || !gridRef.current) return;
              const rect = gridRef.current.getBoundingClientRect();
              const c = Math.floor((e.clientX - rect.left - leftPad) / colW);
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
              const c = Math.floor((e.clientX - rect.left - leftPad) / colW);
              const q = Math.floor((e.clientY - rect.top) / rowH);
              if (c >= 0 && c < numCols && q >= 0 && q < project.qubits.length) {
                handleDropOnCell(e, q, c);
              } else {
                setDragHoverCell(null);
              }
            }}
          >
            {/* Interactive Oscilloscope Probe Playhead in Simulate Mode */}
            {cursorX !== null && (
              <div
                className="absolute -top-2 bottom-0 left-0 w-[14px] -ml-[7px] z-30 will-change-transform cursor-ew-resize group"
                style={{ transform: `translate3d(${cursorX}px, 0, 0)` }}
                title="按住左右拖动演化探针游标（支持按住 Shift 精细观察脉冲内演化）"
              >
                {/* Core 2px Glowing Probe Line */}
                <div className="mx-auto w-[2px] h-full bg-[#2F81F7] shadow-[0_0_10px_rgba(47,129,247,0.75)]" />
                {/* Top Oscilloscope Probe Triangle ▼ */}
                <svg
                  width="12"
                  height="9"
                  viewBox="0 0 12 9"
                  className="absolute left-[1px] top-0 text-[#2F81F7] drop-shadow-xs"
                >
                  <polygon points="0,0 12,0 6,9" fill="currentColor" />
                </svg>
                {cursorReadoutLabel && (
                  <div className="absolute -top-1 left-4 px-1.5 py-0 rounded-[2px] bg-[#0f141b]/95 border border-[#2F81F7]/75 text-[#60a5fa] text-[9.5px] font-mono-num font-semibold whitespace-nowrap pointer-events-none shadow-xs">
                    {cursorReadoutLabel}
                  </div>
                )}
              </div>
            )}

            {/* Subtle CAD Crosshair Alignment Guides + 1:1 Ghost Gate Preview */}
            {!readOnly && (dragHoverCell || ((activeToolGate || pendingFirstQubit) && hoverCell)) && (() => {
              const target = dragHoverCell || hoverCell!;
              const previewStep = pendingFirstQubit ? pendingFirstQubit.step : target.c;
              const xCenter = leftPad + previewStep * colW + colW / 2;
              const yCenter = target.q * rowH + rowH / 2;

              // Case A: Multi-Qubit Rubber-Band Connector Preview when anchor qubit(s) already chosen!
              if (pendingFirstQubit && !dragHoverCell) {
                const previewQs = pendingFirstQubit.qubits.includes(target.q)
                  ? pendingFirstQubit.qubits
                  : [...pendingFirstQubit.qubits, target.q];
                const minQ = Math.min(...previewQs);
                const maxQ = Math.max(...previewQs);
                const topY = minQ * rowH + rowH / 2;
                const botY = maxQ * rowH + rowH / 2;
                const uname = pendingFirstQubit.gateName.toUpperCase();
                const isTargetXor =
                  uname === 'CNOT' || uname === 'CX' || uname === 'CCX' || uname === 'TOFFOLI';
                const topoWarn = isPairTopologyViolated(previewQs);

                return (
                  <div className="pointer-events-none z-20">
                    {/* Vertical Rubber-Band Stem */}
                    {maxQ > minQ && (
                      <div
                        className={`absolute w-[2px] border-l-2 border-dashed ${
                          topoWarn ? 'border-amber-500' : 'border-[#2F81F7]'
                        }`}
                        style={{ left: xCenter - 1, top: topY, height: botY - topY }}
                      />
                    )}
                    {/* Target Hover Preview End-Cap */}
                    {!pendingFirstQubit.qubits.includes(target.q) && (
                      <div
                        style={{ left: xCenter - 13, top: yCenter - 13, width: 26, height: 26 }}
                        className={`absolute rounded-full border-2 border-dashed flex items-center justify-center text-xs font-bold ${
                          topoWarn
                            ? 'border-amber-500 bg-amber-500/15 text-amber-400'
                            : 'border-[#2F81F7] bg-[#2F81F7]/15 text-[#60a5fa]'
                        }`}
                      >
                        {isTargetXor ? '+' : uname === 'SWAP' ? '×' : '●'}
                      </div>
                    )}
                    {topoWarn && (
                      <div
                        style={{ left: xCenter + 16, top: yCenter - 10 }}
                        className="absolute px-1.5 py-0.5 rounded-[2px] bg-amber-500 text-black text-[9.5px] font-mono-num font-bold whitespace-nowrap"
                      >
                        ⚠ 超拓扑半径(自动路由)
                      </div>
                    )}
                  </div>
                );
              }

              // Case B: Drag-over highlight OR 1:1 Ghost Gate Stamp Preview + Crosshair Guides
              const gName = activeToolGate || 'GATE';
              const gStyle = getGateStyleByName(gName);
              const arity = activeToolGate ? getGateArity(activeToolGate) : 1;
              const ghostW = Math.round(42 * effZoom);
              const ghostH = Math.round(34 * effZoom);

              return (
                <div className="pointer-events-none z-20">
                  {/* Subtle Column & Row CAD Alignment Hairlines */}
                  <div
                    className="absolute top-0 bottom-0 w-[1px] border-l border-dashed border-[#2F81F7]/30"
                    style={{ left: xCenter }}
                  />
                  <div
                    className="absolute h-[1px] border-t border-dashed border-[#2F81F7]/30"
                    style={{ left: leftPad, width: numCols * colW, top: yCenter }}
                  />

                  {dragHoverCell ? (
                    <div
                      className="absolute rounded-[3px] bg-[#2F81F7]/18 ring-1 ring-inset ring-[#2F81F7]"
                      style={{
                        left: leftPad + target.c * colW + 4,
                        top: target.q * rowH + 4,
                        width: colW - 8,
                        height: rowH - 8,
                      }}
                    />
                  ) : arity >= 2 ? (
                    <div
                      style={{ left: xCenter - 12, top: yCenter - 12, width: 24, height: 24 }}
                      className="absolute rounded-full border-2 border-dashed border-[#2F81F7] bg-[#2F81F7]/20 flex items-center justify-center text-[10px] font-mono-num font-bold text-[#60a5fa]"
                    >
                      ●
                    </div>
                  ) : (
                    <div
                      style={{
                        left: xCenter - ghostW / 2,
                        top: yCenter - ghostH / 2,
                        width: ghostW,
                        height: ghostH,
                      }}
                      className={`absolute rounded-[3px] border border-dashed opacity-65 flex items-center justify-center font-mono-num font-bold text-[11px] ${gStyle.bg} ${gStyle.border} ${gStyle.text}`}
                    >
                      {gStyle.label}
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Marquee Box Selection Overlay */}
            {marquee && (
              <div
                className="absolute border border-[#2F81F7] bg-[#2F81F7]/15 pointer-events-none z-30"
                style={{
                  left: Math.min(marquee.x0, marquee.x1),
                  top: Math.min(marquee.y0, marquee.y1),
                  width: Math.abs(marquee.x1 - marquee.x0),
                  height: Math.abs(marquee.y1 - marquee.y0),
                }}
              />
            )}

            {/* Pending Multi-Qubit First-Click Anchor Dots */}
            {pendingFirstQubit &&
              pendingFirstQubit.qubits.map((pq) => (
                <span
                  key={`pend_${pq}`}
                  className="absolute w-3.5 h-3.5 rounded-full bg-[#2F81F7] ring-4 ring-[#2F81F7]/25 pointer-events-none z-20"
                  style={{
                    left: leftPad + pendingFirstQubit.step * colW + colW / 2 - 7,
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
                ? Math.max(0, Math.floor(Math.max(0, vpScroll.left - leftPad) / colW) - 6)
                : 0;
              const visColEnd = useVirtual
                ? Math.min(numCols, Math.ceil((Math.max(0, vpScroll.left - leftPad) + vpScroll.width) / colW) + 6)
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
                        {/* Sticky Qubit Label */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!readOnly) onSelect({ type: 'qubit', qubitId: q.id });
                          }}
                          style={{ width: leftPad, height: rowH }}
                          className={`sticky left-0 z-30 pr-3 text-right flex items-center justify-end gap-1.5 bg-[var(--bg-subtle)] border-r border-[var(--border-hairline)]/60 transition-colors cursor-pointer shrink-0 ${
                            isQSelected
                              ? 'text-[var(--accent)] font-semibold'
                              : 'text-[var(--text-primary)] hover:text-[var(--accent)]'
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
                            left: leftPad,
                            width: numCols * colW,
                            top: rowH / 2,
                          }}
                        />
                      </div>
                    );
                  })}

                  {/* Rendered Operations Overlay */}
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
                      ? 'opacity-55'
                      : '';
                    const style = getGateStyle(op);
                    const xCenter = leftPad + op.step * colW + colW / 2;
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

                    const handleOpContextMenu = (e: React.MouseEvent) => {
                      if (readOnly) return;
                      e.preventDefault();
                      e.stopPropagation();
                      onDeleteOp(op.id);
                    };

                    const dragHandlers = readOnly
                      ? {}
                      : {
                          draggable: true,
                          onDragStart: (e: React.DragEvent) => {
                            e.stopPropagation();
                            e.dataTransfer.setData('application/quantum-move-op', op.id);
                          },
                          onContextMenu: handleOpContextMenu,
                        };

                    // Measurement across multiple qubits
                    if (op.kind === 'measure') {
                      const mH = Math.round((compact ? 26 : 34) * effZoom);
                      const mW = Math.round((compact ? 32 : 38) * effZoom);
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
                              className={`studio-gate-node absolute rounded-[3px] border flex items-center justify-center cursor-pointer z-10 ${style.bg} ${style.border} ${style.text} ${simStateCls} ${
                                isSelected ? 'ring-2 ring-[var(--accent)]' : 'hover:border-[var(--accent)]'
                              }`}
                              title="测量 Measurement（可拖拽移动时间列，右键快速删除）"
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
                          className={`studio-gate-node group absolute cursor-grab active:cursor-grabbing z-10 rounded-[3px] ${simStateCls} ${
                            isSelected
                              ? 'bg-[#2F81F7]/15 ring-2 ring-[#2F81F7]'
                              : topoWarn
                              ? 'bg-amber-500/12 ring-1 ring-amber-500/80 ring-dashed'
                              : 'hover:bg-indigo-500/10'
                          }`}
                          title={
                            topoWarn
                              ? `${op.name} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) — ⚠ 超出当前硬件拓扑/阻塞半径，编译时自动插入 SWAP`
                              : `${op.name} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) · 拖拽移动 / 右键删除`
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
                              className={`absolute left-6 text-[9.5px] font-mono-num font-bold border px-1 rounded-[2px] ${
                                topoWarn
                                  ? 'text-amber-700 dark:text-amber-200 bg-amber-50 dark:bg-amber-950 border-amber-400'
                                  : 'text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-[#1c2340] border-indigo-300 dark:border-indigo-700'
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
                      const padHalf = Math.round((compact ? 14 : 20) * effZoom);
                      const boxTop = minQ * rowH + rowH / 2 - padHalf;
                      const boxH = (maxQ - minQ) * rowH + padHalf * 2;
                      const boxW = Math.round((compact ? 60 : 66) * effZoom);
                      const topoWarn = isPairTopologyViolated(validQ);
                      return (
                        <div
                          key={op.id}
                          {...dragHandlers}
                          onClick={handleOpClick}
                          style={{
                            left: xCenter - boxW / 2,
                            top: boxTop,
                            width: boxW,
                            height: boxH,
                          }}
                          className={`studio-gate-node absolute rounded-[4px] border flex flex-col items-center justify-center px-1 cursor-grab active:cursor-grabbing z-10 ${style.bg} ${
                            topoWarn ? 'border-amber-500 border-dashed' : style.border
                          } ${style.text} ${simStateCls} ${
                            isSelected ? 'ring-2 ring-[var(--accent)]' : 'hover:border-[var(--accent)]'
                          }`}
                          title={
                            topoWarn
                              ? `${style.label} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) — ⚠ 超出硬件拓扑`
                              : `${style.label} (${validQ.map((q) => project.qubits[q]?.name).join(', ')}) · 拖拽移动 / 右键删除`
                          }
                        >
                          {topoWarn && !compact && (
                            <span className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-amber-500 text-white flex items-center justify-center shadow-xs">
                              <AlertTriangle className="w-2.5 h-2.5" />
                            </span>
                          )}
                          {validQ.map((aq) => (
                            <span
                              key={aq}
                              className="absolute -left-[4px] w-2 h-2 rounded-full bg-violet-500 border border-white"
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
                            <span className="mt-1 px-1.5 py-0 rounded-[2px] bg-white/80 dark:bg-black/30 text-[9px] font-medium">
                              自定义
                            </span>
                          )}
                        </div>
                      );
                    }

                    // Single-qubit gate or noise box
                    const qIdx = validQ[0];
                    const rawW = paramTxt && !compact ? Math.min(78, Math.max(46, Math.round(paramTxt.length * 6.4 + 14))) : compact ? 42 : 44;
                    const boxW = Math.round(rawW * effZoom);
                    const boxH = Math.round((compact ? 26 : 38) * effZoom);
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
                        className={`studio-gate-node group absolute rounded-[3px] border flex flex-col items-center justify-center px-1 cursor-grab active:cursor-grabbing z-10 ${style.bg} ${style.border} ${style.text} ${simStateCls} ${
                          isSelected ? 'ring-2 ring-[var(--accent)]' : 'hover:border-[var(--accent)]'
                        }`}
                        title={`${op.name}${paramTxt ? `(${paramTxt})` : ''} @ ${project.qubits[qIdx]?.name} · 拖拽移动 / 右键删除`}
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
                            className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-red-600 text-white flex items-center justify-center shadow-xs hover:bg-red-500 cursor-pointer"
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
              style={{ marginLeft: leftPad, width: Math.max(360, numCols * colW) }}
              className="mt-3 h-8 rounded-[3px] border border-dashed border-[var(--border-input)] hover:border-[var(--accent)] hover:bg-[var(--accent-soft)] text-[var(--text-secondary)] hover:text-[var(--accent)] text-[12px] font-medium flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
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
