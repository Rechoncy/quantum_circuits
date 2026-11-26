import { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  CircuitBoard,
  Cpu,
  LineChart,
  Loader2,
} from 'lucide-react';
import type {
  CalibrationWritebackItem,
  CircuitOpModel,
  CircuitSubMode,
  CustomGateDef,
  DeviceSubTab,
  GateBenchmarkResponse,
  HardwareBenchmarkResponse,
  NlCircuitResponse,
  NoiseBudgetResponse,
  OptimizeResponse,
  PrimaryDomain,
  ProjectModel,
  QubitSpec,
  RunHistoryItem,
  SelectionTarget,
  SimulationResponse,
  SweepResponse,
  VariationalOptimizeResponse,
  WorkMode,
  ZneResponse,
} from './types';
import { BUILTIN_GATES, QUBIT_DOT_PALETTE } from './constants';
import {
  fetchPresets,
  runCrossHardwareBenchmark,
  runGateBenchmark,
  runNoiseBudget,
  runOptimization,
  runSimulation,
  runSweep,
  runUnitarySynthesis,
  runVariationalOptimization,
  runZneMitigation,
  syncDslToProject,
  syncProjectToDsl,
  syncQasmToProject,
} from './api';
import { TopToolbar } from './components/TopToolbar';
import { ActivityBar } from './components/ActivityBar';
import { LeftNavigator } from './components/LeftNavigator';
import { CircuitCanvas } from './components/CircuitCanvas';
import {
  RightInspector,
  buildPresetHamiltonian,
  type BenchmarkConfigState,
  type SweepConfigState,
} from './components/RightInspector';
import { BottomDock } from './components/BottomDock';
import { SimulateResults } from './components/SimulateResults';
import { OptimizeWorkspace, type DirectSynthPayload } from './components/OptimizeWorkspace';
import { SweepWorkspace } from './components/SweepWorkspace';
import { BenchmarkWorkspace } from './components/BenchmarkWorkspace';
import { DeviceWorkspace } from './components/DeviceWorkspace';
import { CustomGateModal } from './components/CustomGateModal';
import { NlCircuitModal } from './components/NlCircuitModal';
import { buildCircuitSvg, downloadTextFile } from './utils/exportCircuitSvg';

const DRAFT_STORAGE_KEY = 'quantum_studio_v2_draft_project';

export default function App() {
  const [presets, setPresets] = useState<ProjectModel[]>([]);
  const [hardwareSweepParams, setHardwareSweepParams] = useState<Record<string, string>>({});
  const [project, setProject] = useState<ProjectModel | null>(null);

  // Undo/Redo history stacks
  const [past, setPast] = useState<ProjectModel[]>([]);
  const [future, setFuture] = useState<ProjectModel[]>([]);

  // Three Parallel Top-Level Domains: 量子线路 | 比特门表征 | 器件与硬件
  const [domain, setDomain] = useState<PrimaryDomain>('circuits');
  const [circuitSubMode, setCircuitSubMode] = useState<CircuitSubMode>('design');
  const [deviceSubTab, setDeviceSubTab] = useState<DeviceSubTab>('drive');
  const mode: WorkMode =
    domain === 'circuits'
      ? circuitSubMode
      : domain === 'characterization'
      ? 'benchmark'
      : 'device';
  const setMode = useCallback((m: WorkMode) => {
    if (m === 'benchmark') {
      setDomain('characterization');
    } else if (m === 'device') {
      setDomain('device');
    } else {
      setDomain('circuits');
      setCircuitSubMode(m);
    }
  }, []);

  const [darkMode, setDarkMode] = useState(true);
  const [showLeft, setShowLeft] = useState(true);
  const [showRight, setShowRight] = useState(true);
  const [selection, setSelection] = useState<SelectionTarget>({ type: 'none' });
  const [activeToolGate, setActiveToolGate] = useState<string | null>(null);

  // Synchronized code & logs
  const [dslText, setDslText] = useState('');
  const [qasmText, setQasmText] = useState('');
  const [dslError, setDslError] = useState<string | null>(null);
  const [qasmError, setQasmError] = useState<string | null>(null);
  const [logs, setLogs] = useState<{ time: string; level: 'info' | 'warn' | 'error'; text: string }[]>([]);

  // Execution state & results
  const [isBusy, setIsBusy] = useState(false);
  const [statusMsg, setStatusMsg] = useState('正在初始化 Quantum Studio...');
  const [simResponse, setSimResponse] = useState<SimulationResponse | null>(null);
  const [zneResponse, setZneResponse] = useState<ZneResponse | null>(null);
  const [noiseBudgetResponse, setNoiseBudgetResponse] = useState<NoiseBudgetResponse | null>(null);
  const [simDirty, setSimDirty] = useState(false);
  const [runHistory, setRunHistory] = useState<RunHistoryItem[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [compareRunId, setCompareRunId] = useState<string | null>(null);
  const [timeCursorStep, setTimeCursorStep] = useState(0);
  const timeCursorRef = useRef(0);
  timeCursorRef.current = timeCursorStep;
  const [isPlayingTimeline, setIsPlayingTimeline] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<0.5 | 1 | 2>(1);
  const stepTweenRef = useRef<number | null>(null);
  const [observedQubit, setObservedQubit] = useState(1);
  const [simCircuitCollapsed, setSimCircuitCollapsed] = useState(false);
  const [simCircuitHeight, setSimCircuitHeight] = useState(206);

  // Optimize state
  const [optResponse, setOptResponse] = useState<OptimizeResponse | null>(null);
  const [benchResponse, setBenchResponse] = useState<HardwareBenchmarkResponse | null>(null);
  const [preferredNative2Q, setPreferredNative2Q] = useState('');
  const [evalNoisyOpt, setEvalNoisyOpt] = useState(true);

  // Sweep, VQE & Characterization state
  const [sweepResponse, setSweepResponse] = useState<SweepResponse | null>(null);
  const [vqeResponse, setVqeResponse] = useState<VariationalOptimizeResponse | null>(null);
  const [gateBenchResponse, setGateBenchResponse] = useState<GateBenchmarkResponse | null>(null);
  const [benchmarkConfig, setBenchmarkConfig] = useState<BenchmarkConfigState>({
    protocol: 'rabi',
    target_qubit: 0,
    second_qubit: 1,
    rb_num_qubits: 1,
    rb_lengths_str: '1, 2, 4, 8, 16, 32, 64, 128',
    rb_num_seeds: 20,
    interleaved_gate: 'SX',
    xeb_entangler: 'SQISWAP',
    gst_max_length: 4,
    cal_points: 61,
    cal_max_ns: null,
    ramsey_detuning_mhz: 0.5,
    shots: 4096,
    amp_error_pct: 0,
    detuning_mhz: 0,
    phase_skew_deg: 0,
  });
  const [sweepConfig, setSweepConfig] = useState<SweepConfigState>({
    mode: '1d',
    param_x: 'theta',
    x_start: 0,
    x_stop: 6.283,
    x_points: 25,
    param_y: 'hw:depol_2q',
    y_start: 0,
    y_stop: 0.05,
    y_points: 10,
    target_bitstring: '0000',
    observed_qubit: 1,
    hamiltonian_str: '',
    vqe_method: 'COBYLA',
    vqe_max_iter: 50,
    backend: 'kraus',
  });
  const taskAbortRef = useRef<AbortController | null>(null);

  // Custom Gate & Natural Language Modal state
  const [gateModalOpen, setGateModalOpen] = useState(false);
  const [nlModalOpen, setNlModalOpen] = useState(false);
  const [editingCustomGate, setEditingCustomGate] = useState<CustomGateDef | null>(null);
  const [macroSeedOps, setMacroSeedOps] = useState<CircuitOpModel[] | null>(null);
  const [macroSeedQubits, setMacroSeedQubits] = useState<number>(2);
  const [exportToast, setExportToast] = useState<string | null>(null);

  const runCounterRef = useRef(1);

  const addLog = useCallback((text: string, level: 'info' | 'warn' | 'error' = 'info') => {
    const now = new Date();
    const t = now.toTimeString().slice(0, 8);
    setLogs((prev) => [{ time: t, level, text }, ...prev].slice(0, 60));
  }, []);

  useEffect(() => {
    let timer: number | undefined;
    const onExport = (ev: Event) => {
      const detail = (ev as CustomEvent<{ filename?: string; path?: string }>).detail;
      const target = detail?.path || detail?.filename || '文件';
      const msg = `文件已导出: ${target}`;
      setExportToast(msg);
      setStatusMsg(msg);
      addLog(msg, 'info');
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => setExportToast(null), 3800);
    };
    window.addEventListener('qstudio:export-toast', onExport);
    return () => {
      window.removeEventListener('qstudio:export-toast', onExport);
      if (timer) window.clearTimeout(timer);
    };
  }, [addLog]);

  // Toggle dark class on root HTML element
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // Project mutation helper with Undo/Redo snapshot + stale-result flag
  const mutateProject = useCallback(
    (updater: (prev: ProjectModel) => ProjectModel, recordHistory = true) => {
      setProject((prev) => {
        if (!prev) return prev;
        const next = updater(prev);
        if (recordHistory) {
          setPast((p) => [...p.slice(-30), prev]);
          setFuture([]);
        }
        return next;
      });
      setSimDirty(true);
    },
    []
  );

  // Auto-save current project draft to localStorage (debounced 500ms)
  useEffect(() => {
    if (!project) return;
    const t = window.setTimeout(() => {
      try {
        window.localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(project));
      } catch {
        // ignore storage quota errors
      }
    }, 500);
    return () => window.clearTimeout(t);
  }, [project]);

  // Unified Stop / Abort handler for Simulate, Optimize, Synthesis, and Sweep
  const handleStopTask = useCallback(() => {
    if (taskAbortRef.current) {
      taskAbortRef.current.abort();
      taskAbortRef.current = null;
      setIsBusy(false);
      setStatusMsg('已中止当前计算任务');
      addLog('已手动中止当前计算任务', 'warn');
    }
  }, [addLog]);

  // Run Simulation action
  const handleRunSimulation = useCallback(
    async (projOverride?: ProjectModel, switchToSim = false) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg(`正在使用 ${target.sim_settings.backend.toUpperCase()} 求解 ${target.qubits.length} 比特演化...`);
      try {
        const res = await runSimulation(target, controller.signal);
        setSimResponse(res);
        setSimDirty(false);
        const pRes = res.results[res.primary_backend];
        const lastStep = Math.max(0, (pRes?.steps.length ?? 1) - 1);
        setTimeCursorStep(lastStep);
        setObservedQubit((q) => Math.min(q, Math.max(0, target.qubits.length - 1)));

        const runNum = runCounterRef.current++;
        const item: RunHistoryItem = {
          id: `run_${Date.now()}`,
          label: `run #${runNum}`,
          timestamp: new Date().toTimeString().slice(0, 8),
          backend: pRes.backend_name,
          num_qubits: target.qubits.length,
          fidelity: pRes.final_fidelity,
          duration_ns: pRes.total_duration_ns,
          response: res,
        };
        setRunHistory((prev) => [item, ...prev].slice(0, 15));
        setActiveRunId(item.id);
        const fStr = pRes.final_fidelity !== null ? ` · F = ${pRes.final_fidelity.toFixed(4)}` : '';
        const msg = `求解完成 · ${pRes.backend_name} · ${target.qubits.length} qubits · ${pRes.total_duration_ns.toFixed(0)} ns${fStr}`;
        setStatusMsg(msg);
        addLog(msg, 'info');
        if (switchToSim) setMode('simulate');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`求解失败: ${errTxt}`);
        addLog(`求解失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, addLog]
  );

  // Run Optimization action
  const handleRunOptimize = useCallback(
    async (projOverride?: ProjectModel) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg('正在执行硬件感知最优量子门综合 (Euler + KAK Cartan)...');
      try {
        const res = await runOptimization(
          target,
          evalNoisyOpt,
          preferredNative2Q || null,
          controller.signal
        );
        setOptResponse(res);
        const msg = `最优综合完成 · ${res.gate_set_name} · 2Q门: ${res.optimized_metrics.num_2q_gates} · 酉保真度 F = ${res.unitary_fidelity.toFixed(6)}`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`优化失败: ${errTxt}`);
        addLog(`优化失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, evalNoisyOpt, preferredNative2Q, addLog]
  );

  // Run Direct Target Unitary / Hamiltonian Synthesis
  const handleRunUnitarySynthesis = useCallback(
    async (payload: DirectSynthPayload) => {
      if (!project) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg('正在根据当前硬件原生门基综合目标酉矩阵 / 哈密顿量...');
      try {
        const res = await runUnitarySynthesis(
          {
            project,
            mode: payload.mode,
            num_qubits: payload.num_qubits,
            matrix_exprs: payload.matrix_exprs,
            hamiltonian_str: payload.hamiltonian_str,
            hamiltonian_scale: payload.hamiltonian_scale,
            evaluate_noisy_fidelity: evalNoisyOpt,
            preferred_native_2q: preferredNative2Q || null,
          },
          controller.signal
        );
        setOptResponse(res);
        const msg = `目标酉矩阵综合完成 · ${res.gate_set_name} · 2Q门: ${res.optimized_metrics.num_2q_gates} · F = ${res.unitary_fidelity.toFixed(6)}`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`酉矩阵综合失败: ${errTxt}`);
        addLog(`酉矩阵综合失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, evalNoisyOpt, preferredNative2Q, addLog]
  );

  // Run Cross-Hardware Benchmark
  const handleRunHwBenchmark = useCallback(
    async (projOverride?: ProjectModel) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg('正在并发执行跨硬件平台原生门编译对比基准...');
      try {
        const res = await runCrossHardwareBenchmark(target, controller.signal);
        setBenchResponse(res);
        const msg = `跨硬件平台基准对比完成 · 共评测 ${res.entries.length} 种硬件体系`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`硬件对比失败: ${errTxt}`);
        addLog(`硬件对比失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, addLog]
  );

  // Quick Alt+1 / Alt+2 / Alt+3 top-level domain switching
  useEffect(() => {
    const onDomainShortcut = (e: KeyboardEvent) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === '1') {
        e.preventDefault();
        setDomain('circuits');
      } else if (e.key === '2') {
        e.preventDefault();
        setDomain('characterization');
      } else if (e.key === '3') {
        e.preventDefault();
        setDomain('device');
      }
    };
    window.addEventListener('keydown', onDomainShortcut);
    return () => window.removeEventListener('keydown', onDomainShortcut);
  }, []);

  // Run Quantum Qubit & Gate Characterization (Rabi / Ramsey / T1 / RB / AllXY / GST / XEB)
  const handleRunGateBenchmark = useCallback(
    async (projOverride?: ProjectModel, cfgOverride?: Partial<BenchmarkConfigState>) => {
      const target = projOverride || project;
      if (!target) return;
      const cfg = { ...benchmarkConfig, ...(cfgOverride || {}) };
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      const parsedLengths = cfg.rb_lengths_str
        .split(/[\s,;]+/)
        .map((s) => Number(s.trim()))
        .filter((n) => Number.isFinite(n) && n >= 1)
        .map((n) => Math.round(n));

      setIsBusy(true);
      setStatusMsg(`正在执行 ${cfg.protocol.toUpperCase()} 量子比特与门表征实验...`);
      try {
        const res = await runGateBenchmark(
          {
            project: target,
            protocol: cfg.protocol,
            target_qubit: Math.min(cfg.target_qubit, Math.max(0, target.qubits.length - 1)),
            second_qubit: Math.min(cfg.second_qubit, Math.max(0, target.qubits.length - 1)),
            rb_num_qubits: target.qubits.length >= 2 ? cfg.rb_num_qubits : 1,
            rb_lengths: parsedLengths.length >= 3 ? parsedLengths : [1, 2, 4, 8, 16, 32, 64, 128],
            rb_num_seeds: cfg.rb_num_seeds,
            interleaved_gate: cfg.interleaved_gate || null,
            xeb_entangler: cfg.xeb_entangler || 'SQISWAP',
            gst_max_length: cfg.gst_max_length,
            cal_points: cfg.cal_points,
            cal_max_ns: cfg.cal_max_ns,
            ramsey_detuning_mhz: cfg.ramsey_detuning_mhz,
            shots: cfg.shots,
            amp_error_pct: cfg.amp_error_pct,
            detuning_mhz: cfg.detuning_mhz,
            phase_skew_deg: cfg.phase_skew_deg,
          },
          controller.signal
        );
        setGateBenchResponse(res);
        const summary =
          res.protocol === 'rb'
            ? `RB 标定完成 · F_Clifford=${(res.clifford_fidelity * 100).toFixed(3)}% · EPC=${(res.epc * 100).toFixed(3)}%`
            : res.protocol === 'allxy'
            ? `AllXY 脉冲诊断完成 · RMS Δ⟨Z⟩=${res.rms_error_z.toFixed(4)}`
            : res.protocol === 'gst'
            ? `GST 门集层析完成 · 平均保真度=${(res.mean_gate_fidelity * 100).toFixed(3)}% · 最大 ½‖Δℰ‖_⋄=${res.max_diamond_distance.toFixed(4)}`
            : res.protocol === 'xeb'
            ? `2Q Linear XEB 完成 · ${res.entangler_name} 保真度=${(res.entangler_fidelity * 100).toFixed(3)}% · f_cycle=${(res.cycle_polarization * 100).toFixed(3)}%`
            : `${res.protocol.toUpperCase()} 标定完成 · ${res.summary}`;
        setStatusMsg(summary);
        addLog(summary, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`门表征失败: ${errTxt}`);
        addLog(`门表征失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, benchmarkConfig, addLog]
  );

  // Run Zero-Noise Extrapolation (ZNE) quantum error mitigation
  const handleRunZne = useCallback(
    async (projOverride?: ProjectModel) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg('正在执行 ZNE 零噪声外推量子误差缓解 (λ = 1, 2, 3 → 0)...');
      try {
        const res = await runZneMitigation(
          {
            project: target,
            scale_factors: [1.0, 2.0, 3.0],
            observed_qubit: Math.min(observedQubit, Math.max(0, target.qubits.length - 1)),
            hamiltonian_str: sweepConfig.hamiltonian_str.trim() || undefined,
          },
          controller.signal
        );
        setZneResponse(res);
        setMode('simulate');
        const msg = `ZNE 零噪声外推完成 · 原始 ⟨Z⟩=${res.z_unmitigated.toFixed(4)} → Richardson ⟨Z⟩=${res.z_zne_richardson.toFixed(4)}`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`ZNE 外推失败: ${errTxt}`);
        addLog(`ZNE 外推失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, observedQubit, sweepConfig.hamiltonian_str, addLog]
  );

  // Run Noise Error Budget Ablation
  const handleRunNoiseBudget = useCallback(
    async (projOverride?: ProjectModel) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      setIsBusy(true);
      setStatusMsg('正在执行含噪线路误差预算消融分析 (T1 / T2* / 1Q / 2Q / ZZ / Readout)...');
      try {
        const res = await runNoiseBudget(
          {
            project: target,
            backend:
              target.sim_settings.backend === 'compare' ? undefined : target.sim_settings.backend,
          },
          controller.signal
        );
        setNoiseBudgetResponse(res);
        setMode('simulate');
        const msg = `误差预算拆解完成 · F=${res.full_fidelity.toFixed(4)} · ${res.bottleneck_summary}`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`误差预算拆解失败: ${errTxt}`);
        addLog(`误差预算拆解失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, addLog]
  );

  // Run Parameter Sweep OR Closed-Loop VQE Optimization (with AbortController cancelation support)
  const handleCancelSweep = handleStopTask;

  const handleRunSweep = useCallback(
    async (projOverride?: ProjectModel) => {
      const target = projOverride || project;
      if (!target) return;
      if (taskAbortRef.current) taskAbortRef.current.abort();
      const controller = new AbortController();
      taskAbortRef.current = controller;

      if (sweepConfig.mode === 'vqe') {
        const hamStr =
          sweepConfig.hamiltonian_str.trim() || buildPresetHamiltonian('tfim', target);
        if (!sweepConfig.hamiltonian_str.trim()) {
          setSweepConfig((prev) => ({ ...prev, hamiltonian_str: hamStr }));
        }
        setIsBusy(true);
        setStatusMsg(`正在执行 ${sweepConfig.vqe_method} 闭环变分寻优...`);
        try {
          const vRes = await runVariationalOptimization(
            {
              project: target,
              hamiltonian_str: hamStr,
              method: sweepConfig.vqe_method,
              max_iter: sweepConfig.vqe_max_iter,
              backend: sweepConfig.backend,
            },
            controller.signal
          );
          setVqeResponse(vRes);
          const msg = `变分寻优完成 · ${vRes.method} (${vRes.num_evaluations} 步) · E* = ${vRes.optimal_energy.toFixed(5)}`;
          setStatusMsg(msg);
          addLog(msg, 'info');
        } catch (err) {
          if ((err as Error)?.name === 'AbortError') return;
          const errTxt = err instanceof Error ? err.message : String(err);
          setStatusMsg(`变分寻优失败: ${errTxt}`);
          addLog(`变分寻优失败: ${errTxt}`, 'error');
        } finally {
          if (taskAbortRef.current === controller) {
            taskAbortRef.current = null;
            setIsBusy(false);
          }
        }
        return;
      }

      setIsBusy(true);
      setStatusMsg(`正在并发执行 ${sweepConfig.mode.toUpperCase()} 参数扫描 (${sweepConfig.param_x})...`);
      try {
        const res = await runSweep(
          {
            project: target,
            mode: sweepConfig.mode,
            param_x: sweepConfig.param_x,
            x_start: sweepConfig.x_start,
            x_stop: sweepConfig.x_stop,
            x_points: sweepConfig.x_points,
            param_y: sweepConfig.param_y,
            y_start: sweepConfig.y_start,
            y_stop: sweepConfig.y_stop,
            y_points: sweepConfig.y_points,
            target_bitstring: sweepConfig.target_bitstring,
            observed_qubit: sweepConfig.observed_qubit,
            hamiltonian_str: sweepConfig.hamiltonian_str.trim() || undefined,
            backend: sweepConfig.backend,
          },
          controller.signal
        );
        setSweepResponse(res);
        const msg = `参数扫描完成 · ${res.param_x_label} (${res.backend})`;
        setStatusMsg(msg);
        addLog(msg, 'info');
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') return;
        const errTxt = err instanceof Error ? err.message : String(err);
        setStatusMsg(`参数扫描失败: ${errTxt}`);
        addLog(`参数扫描失败: ${errTxt}`, 'error');
      } finally {
        if (taskAbortRef.current === controller) {
          taskAbortRef.current = null;
          setIsBusy(false);
        }
      }
    },
    [project, sweepConfig, addLog]
  );

  // Initial load presets (auto-restoring localStorage draft if present) & run first simulation
  useEffect(() => {
    (async () => {
      try {
        const data = await fetchPresets();
        setPresets(data.presets);
        setHardwareSweepParams(data.hardware_sweep_params);
        if (data.presets.length > 0) {
          let initial = data.presets[0];
          let restoredDraft = false;
          try {
            const rawDraft = window.localStorage.getItem(DRAFT_STORAGE_KEY);
            if (rawDraft) {
              const parsed = JSON.parse(rawDraft) as ProjectModel;
              if (parsed && Array.isArray(parsed.qubits) && parsed.qubits.length > 0 && Array.isArray(parsed.operations)) {
                initial = parsed;
                restoredDraft = true;
              }
            }
          } catch {
            // ignore corrupt draft
          }
          setProject(initial);
          if (initial.operations.length > 1) {
            setSelection({ type: 'op', opId: initial.operations[1].id });
          } else if (initial.operations.length === 1) {
            setSelection({ type: 'op', opId: initial.operations[0].id });
          }
          const firstVar = Object.keys(initial.variables)[0] || 'hw:depol_2q';
          setSweepConfig((prev) => ({
            ...prev,
            param_x: firstVar,
            target_bitstring: '0'.repeat(Math.max(1, initial.qubits.length)),
            observed_qubit: Math.min(1, Math.max(0, initial.qubits.length - 1)),
          }));
          if (restoredDraft) {
            addLog(`已自动恢复上次草稿 (${initial.name})`, 'info');
          }
          await handleRunSimulation(initial, false);
        }
      } catch (err) {
        setStatusMsg(`无法连接后端服务: ${String(err)}`);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Bidirectional sync: Project -> DSL & OpenQASM 3 ONLY when code-relevant structure changes (debounced 150ms)
  const projectRef = useRef<ProjectModel | null>(null);
  projectRef.current = project;

  const dslStructuralSig = useMemo(() => {
    if (!project) return '';
    return JSON.stringify({
      n: project.qubits.length,
      ops: project.operations.map((o) => [o.step, o.kind, o.name, o.qubits, o.params]),
      vars: project.variables,
    });
  }, [project?.qubits.length, project?.operations, project?.variables]);

  useEffect(() => {
    if (!dslStructuralSig || !projectRef.current) return;
    let cancelled = false;
    const delay = dslText === '' ? 0 : 150;
    const timer = window.setTimeout(() => {
      const snap = projectRef.current;
      if (!snap) return;
      syncProjectToDsl(snap)
        .then((res) => {
          if (!cancelled) {
            setDslText(res.dsl);
            setQasmText(res.openqasm3);
            setDslError(null);
            setQasmError(null);
          }
        })
        .catch(() => {});
    }, delay);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dslStructuralSig]);

  // Smoothly animate timeCursorStep to a discrete target step (e.g. clicking step buttons or table rows)
  const handleSmoothStepJump = useCallback((targetStep: number) => {
    setIsPlayingTimeline(false);
    if (stepTweenRef.current) {
      cancelAnimationFrame(stepTweenRef.current);
      stepTweenRef.current = null;
    }
    const startVal = timeCursorRef.current;
    const diff = targetStep - startVal;
    if (Math.abs(diff) < 1e-3) {
      setTimeCursorStep(targetStep);
      return;
    }
    const t0 = performance.now();
    const dur = 240;
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      const ease = 1 - Math.pow(1 - p, 3);
      setTimeCursorStep(startVal + diff * ease);
      if (p < 1) {
        stepTweenRef.current = requestAnimationFrame(tick);
      } else {
        stepTweenRef.current = null;
        setTimeCursorStep(targetStep);
      }
    };
    stepTweenRef.current = requestAnimationFrame(tick);
  }, []);

  const handleDirectTimeCursorChange = useCallback((val: number) => {
    setIsPlayingTimeline(false);
    if (stepTweenRef.current) {
      cancelAnimationFrame(stepTweenRef.current);
      stepTweenRef.current = null;
    }
    setTimeCursorStep(val);
  }, []);

  // 60fps continuous requestAnimationFrame timeline playback in Simulate mode
  useEffect(() => {
    if (!isPlayingTimeline || !simResponse) return;
    if (stepTweenRef.current) {
      cancelAnimationFrame(stepTweenRef.current);
      stepTweenRef.current = null;
    }
    const pRes = simResponse.results[simResponse.primary_backend];
    const maxS = Math.max(0, (pRes?.steps.length ?? 1) - 1);
    if (maxS <= 0) {
      setIsPlayingTimeline(false);
      return;
    }
    let rafId: number | null = null;
    let lastT = performance.now();
    const stepsPerSec = 1.45 * playbackSpeed;

    const frame = (now: number) => {
      const dt = Math.min(0.06, Math.max(0, (now - lastT) / 1000));
      lastT = now;
      const next = timeCursorRef.current + dt * stepsPerSec;
      if (next >= maxS) {
        setTimeCursorStep(maxS);
        setIsPlayingTimeline(false);
        return;
      }
      setTimeCursorStep(next);
      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);
    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [isPlayingTimeline, simResponse, playbackSpeed]);

  // Global keyboard shortcuts (Ctrl+K, Ctrl+Z, Ctrl+Y, Ctrl+D, Ctrl+Enter, Space, Esc, Delete)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setNlModalOpen((v) => !v);
        return;
      }
      const targetEl = e.target as HTMLElement | null;
      const tag = targetEl?.tagName;
      const inputType = (targetEl as HTMLInputElement | null)?.type;
      const isTextEntry =
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        (tag === 'INPUT' && inputType !== 'range' && inputType !== 'checkbox');

      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (nlModalOpen) return;
        e.preventDefault();
        if (isBusy) return;
        if (mode === 'optimize') handleRunOptimize();
        else if (mode === 'sweep') handleRunSweep();
        else if (mode === 'benchmark') handleRunGateBenchmark();
        else handleRunSimulation(undefined, mode === 'design');
        return;
      }
      if (e.key === ' ' && !isTextEntry && !nlModalOpen && !gateModalOpen && mode === 'simulate' && simResponse) {
        e.preventDefault();
        const pRes = simResponse.results[simResponse.primary_backend];
        const maxS = Math.max(0, (pRes?.steps.length ?? 1) - 1);
        setTimeCursorStep((cur) => (cur >= maxS - 0.02 ? 0 : cur));
        setIsPlayingTimeline((v) => !v);
        return;
      }
      if (isTextEntry || tag === 'INPUT') return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (past.length > 0 && project) {
          const prev = past[past.length - 1];
          setPast((p) => p.slice(0, -1));
          setFuture((f) => [project, ...f]);
          setProject(prev);
          setSimDirty(true);
        }
      } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
        e.preventDefault();
        if (future.length > 0 && project) {
          const next = future[0];
          setFuture((f) => f.slice(1));
          setPast((p) => [...p, project]);
          setProject(next);
          setSimDirty(true);
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd' && selection.type === 'op' && mode === 'design') {
        e.preventDefault();
        const opId = selection.opId;
        mutateProject((p) => {
          const src = p.operations.find((o) => o.id === opId);
          if (!src) return p;
          const copy: CircuitOpModel = {
            ...src,
            id: `op_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
            step: src.step + 1,
          };
          return { ...p, operations: [...p.operations, copy] };
        });
      } else if (e.key === ' ' && mode === 'simulate' && simResponse) {
        e.preventDefault();
        const pRes = simResponse.results[simResponse.primary_backend];
        const maxS = Math.max(0, (pRes?.steps.length ?? 1) - 1);
        setTimeCursorStep((cur) => (cur >= maxS ? 0 : cur));
        setIsPlayingTimeline((v) => !v);
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === 'v' && mode === 'design') {
        e.preventDefault();
        setActiveToolGate(null);
      } else if (
        mode === 'simulate' &&
        simResponse &&
        (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'Home' || e.key === 'End')
      ) {
        e.preventDefault();
        const pRes = simResponse.results[simResponse.primary_backend];
        const maxS = Math.max(0, (pRes?.steps.length ?? 1) - 1);
        if (e.key === 'Home') {
          handleSmoothStepJump(0);
        } else if (e.key === 'End') {
          handleSmoothStepJump(maxS);
        } else if (e.shiftKey) {
          const delta = e.key === 'ArrowRight' ? 0.05 : -0.05;
          handleDirectTimeCursorChange(Math.max(0, Math.min(maxS, timeCursorRef.current + delta)));
        } else {
          const nextStep =
            e.key === 'ArrowRight'
              ? Math.min(maxS, Math.floor(timeCursorRef.current + 1e-4) + 1)
              : Math.max(0, Math.ceil(timeCursorRef.current - 1e-4) - 1);
          handleSmoothStepJump(nextStep);
        }
      } else if (e.key === 'Escape') {
        setActiveToolGate(null);
        setSelection({ type: 'none' });
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selection.type === 'op' && mode === 'design') {
        e.preventDefault();
        const opId = selection.opId;
        mutateProject((p) => ({
          ...p,
          operations: p.operations.filter((o) => o.id !== opId),
        }));
        setSelection({ type: 'none' });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [past, future, project, selection, mode, isBusy, nlModalOpen, simResponse, mutateProject, handleRunOptimize, handleRunSweep, handleRunSimulation]);

  // ---- Handlers for Quantum Register ----
  const handleResizeRegister = (rawTarget: number) => {
    const targetN = Math.max(1, Math.min(500, Math.round(rawTarget)));
    mutateProject((p) => {
      const oldN = p.qubits.length;
      if (targetN === oldN) return p;

      if (targetN < oldN) {
        const keptQ = p.qubits.slice(0, targetN).map((q, idx) => ({ ...q, id: idx }));
        const keptOps: CircuitOpModel[] = [];
        for (const op of p.operations) {
          if (op.kind === 'measure') {
            const mq = op.qubits.filter((q) => q < targetN);
            if (mq.length > 0) keptOps.push({ ...op, qubits: mq });
          } else if (op.qubits.every((q) => q < targetN)) {
            keptOps.push(op);
          }
        }
        return { ...p, qubits: keptQ, operations: keptOps };
      }

      // Growing register: layout on 2D boustrophedon (snake) grid if targetN > 8
      const cols = Math.max(1, Math.ceil(Math.sqrt(targetN)));
      const allOn1D = p.qubits.every((q) => Math.abs(q.pos_y) < 1e-6);
      const nextQ: QubitSpec[] = [];
      for (let i = 0; i < targetN; i++) {
        const gridPos =
          targetN <= 8
            ? { x: Number((i * 4.5).toFixed(1)), y: 0 }
            : (() => {
                const r = Math.floor(i / cols);
                const c = r % 2 === 0 ? i % cols : cols - 1 - (i % cols);
                return { x: Number((c * 4.5).toFixed(1)), y: Number((r * 4.5).toFixed(1)) };
              })();
        if (i < oldN) {
          nextQ.push({
            ...p.qubits[i],
            id: i,
            pos_x: targetN > 8 && allOn1D ? gridPos.x : p.qubits[i].pos_x,
            pos_y: targetN > 8 && allOn1D ? gridPos.y : p.qubits[i].pos_y,
          });
        } else {
          nextQ.push({
            id: i,
            name: `q${i}`,
            color: QUBIT_DOT_PALETTE[i % QUBIT_DOT_PALETTE.length],
            init_state: '|0>',
            init_theta: 0,
            init_phi: 0,
            t1_us: null,
            t2_us: null,
            freq_ghz: null,
            depol_1q: null,
            readout_err: null,
            pos_x: gridPos.x,
            pos_y: gridPos.y,
          });
        }
      }

      const nextOps = p.operations.map((op) =>
        op.kind === 'measure' && op.qubits.length >= oldN
          ? { ...op, qubits: Array.from({ length: targetN }, (_, idx) => idx) }
          : op
      );

      let nextBackend = p.sim_settings.backend;
      if (targetN > 24 && (nextBackend === 'qutip' || nextBackend === 'kraus' || nextBackend === 'trajectory')) {
        nextBackend = 'stim';
      } else if (targetN > 14 && (nextBackend === 'qutip' || nextBackend === 'kraus')) {
        nextBackend = 'trajectory';
      } else if (targetN > 9 && nextBackend === 'qutip') {
        nextBackend = 'kraus';
      }

      return {
        ...p,
        qubits: nextQ,
        operations: nextOps,
        sim_settings: { ...p.sim_settings, backend: nextBackend },
      };
    });
    addLog(`已调整量子寄存器规模为 ${targetN} Qubits`, 'info');
  };

  const handleAddQubit = () => {
    if (!project) return;
    handleResizeRegister(project.qubits.length + 1);
  };

  const handleDeleteQubit = (qubitId: number) => {
    mutateProject((p) => {
      if (p.qubits.length <= 1) return p;
      const idx = p.qubits.findIndex((q) => q.id === qubitId);
      if (idx < 0) return p;
      const remaining = p.qubits
        .filter((_, i) => i !== idx)
        .map((q, newIdx) => ({ ...q, id: newIdx }));
      // Remap or remove operations referencing deleted qubit
      const nextOps: CircuitOpModel[] = [];
      for (const op of p.operations) {
        if (op.kind === 'measure') {
          const kept = op.qubits.filter((q) => q !== idx).map((q) => (q > idx ? q - 1 : q));
          if (kept.length > 0) nextOps.push({ ...op, qubits: kept });
        } else if (!op.qubits.includes(idx)) {
          nextOps.push({ ...op, qubits: op.qubits.map((q) => (q > idx ? q - 1 : q)) });
        }
      }
      return { ...p, qubits: remaining, operations: nextOps };
    });
    if (selection.type === 'qubit' && selection.qubitId === qubitId) {
      setSelection({ type: 'none' });
    }
  };

  const handleReorderQubit = (fromIdx: number, toIdx: number) => {
    mutateProject((p) => {
      const nextQ = [...p.qubits];
      const [moved] = nextQ.splice(fromIdx, 1);
      nextQ.splice(toIdx, 0, moved);
      // Build oldIndex -> newIndex permutation map
      const oldToNew = new Map<number, number>();
      nextQ.forEach((q, newIdx) => oldToNew.set(q.id, newIdx));
      const reindexedQ = nextQ.map((q, newIdx) => ({ ...q, id: newIdx }));
      const remappedOps = p.operations.map((op) => ({
        ...op,
        qubits: op.qubits.map((oldQ) => oldToNew.get(oldQ) ?? oldQ),
      }));
      return { ...p, qubits: reindexedQ, operations: remappedOps };
    });
  };

  // ---- Collision-free placement helper ----
  const insertOpWithoutOverlap = (
    existingOps: CircuitOpModel[],
    incoming: CircuitOpModel
  ): CircuitOpModel[] => {
    const incomingSet = new Set(incoming.qubits);
    const hasCollision = existingOps.some(
      (o) =>
        o.id !== incoming.id &&
        o.step === incoming.step &&
        o.qubits.some((q) => incomingSet.has(q))
    );
    if (!hasCollision) {
      return [...existingOps.filter((o) => o.id !== incoming.id), incoming];
    }
    // Shift existing ops at >= incoming.step rightward by +1 so nothing ever stacks on top
    const shifted = existingOps
      .filter((o) => o.id !== incoming.id)
      .map((o) => (o.step >= incoming.step ? { ...o, step: o.step + 1 } : o));
    return [...shifted, incoming];
  };

  // ---- Handlers for Circuit Operations ----
  const handlePlaceGate = (gateName: string, step: number, qubits: number[]) => {
    const u = gateName.toUpperCase();
    const builtin = BUILTIN_GATES.find((g) => g.name.toUpperCase() === u);
    const customG = project?.custom_gates.find((g) => g.name.toUpperCase() === u);

    const kind: CircuitOpModel['kind'] =
      u === 'MEASURE'
        ? 'measure'
        : builtin?.category === 'noise'
        ? 'noise'
        : 'gate';

    const params = customG
      ? [...customG.default_params]
      : builtin
      ? [...builtin.defaultParams]
      : [];

    const newOp: CircuitOpModel = {
      id: `op_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      step,
      kind,
      name: customG ? customG.name : u,
      qubits,
      params,
      duration_ns: customG ? customG.duration_ns : null,
      noise_mode: 'inherit',
    };

    mutateProject((p) => ({
      ...p,
      operations: insertOpWithoutOverlap(p.operations, newOp),
    }));
    setSelection({ type: 'op', opId: newOp.id });
  };

  const handleMoveOp = (opId: string, newStep: number, newAnchorQubit: number) => {
    mutateProject((p) => {
      const src = p.operations.find((o) => o.id === opId);
      if (!src) return p;
      const nQ = p.qubits.length;
      let nextQubits = src.qubits;
      if (src.kind !== 'measure' && src.qubits.length > 0) {
        const minQ = Math.min(...src.qubits);
        const maxQ = Math.max(...src.qubits);
        const span = maxQ - minQ;
        const clampedAnchor = Math.max(0, Math.min(nQ - 1 - span, newAnchorQubit));
        const delta = clampedAnchor - minQ;
        nextQubits = src.qubits.map((q) => q + delta);
      }
      const moved: CircuitOpModel = {
        ...src,
        step: Math.max(0, newStep),
        qubits: nextQubits,
      };
      return {
        ...p,
        operations: insertOpWithoutOverlap(p.operations, moved),
      };
    });
    setSelection({ type: 'op', opId });
  };

  const handleNewBlankProject = () => {
    if (!project) return;
    const blank: ProjectModel = {
      ...project,
      name: 'Untitled_Circuit.qproj',
      qubits: [
        {
          id: 0,
          name: 'q0',
          color: QUBIT_DOT_PALETTE[0],
          init_state: '|0>',
          init_theta: 0,
          init_phi: 0,
          t1_us: null,
          t2_us: null,
          freq_ghz: null,
          depol_1q: null,
          readout_err: null,
          pos_x: 0,
          pos_y: 0,
        },
        {
          id: 1,
          name: 'q1',
          color: QUBIT_DOT_PALETTE[1],
          init_state: '|0>',
          init_theta: 0,
          init_phi: 0,
          t1_us: null,
          t2_us: null,
          freq_ghz: null,
          depol_1q: null,
          readout_err: null,
          pos_x: 4.5,
          pos_y: 0,
        },
      ],
      operations: [],
      variables: {},
    };
    setProject(blank);
    setPast([]);
    setFuture([]);
    setSimResponse(null);
    setOptResponse(null);
    setSweepResponse(null);
    setSelection({ type: 'none' });
    setMode('design');
    setStatusMsg('已新建空白 2 比特项目，可从顶部门条拖放门或添加比特');
    addLog('已新建空白项目 (2 Qubits)', 'info');
  };

  const handleDuplicateOp = (opId: string) => {
    mutateProject((p) => {
      const src = p.operations.find((o) => o.id === opId);
      if (!src) return p;
      const copy: CircuitOpModel = {
        ...src,
        id: `op_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        step: src.step + 1,
      };
      return { ...p, operations: [...p.operations, copy] };
    });
  };

  const handlePackageMacro = (selectedOpIds: string[]) => {
    if (!project) return;
    const picked = project.operations
      .filter((o) => selectedOpIds.includes(o.id) && o.kind === 'gate')
      .sort((a, b) => a.step - b.step);
    if (picked.length === 0) return;
    const usedQ = Array.from(new Set(picked.flatMap((o) => o.qubits))).sort((a, b) => a - b);
    const qMap = new Map(usedQ.map((q, idx) => [q, idx]));
    const minStep = Math.min(...picked.map((o) => o.step));
    const normalized: CircuitOpModel[] = picked.map((o, idx) => ({
      ...o,
      id: `sub_${idx}`,
      step: o.step - minStep,
      qubits: o.qubits.map((q) => qMap.get(q) ?? 0),
    }));
    setEditingCustomGate(null);
    setMacroSeedOps(normalized);
    setMacroSeedQubits(Math.max(1, Math.min(2, usedQ.length)));
    setGateModalOpen(true);
  };

  // ---- Handlers for DSL & OpenQASM Sync ----
  const applySyncedCircuit = (
    synced: {
      qubits: QubitSpec[];
      operations: CircuitOpModel[];
      variables: Record<string, number>;
    },
    sourceLabel: string
  ) => {
    setDslError(null);
    setQasmError(null);
    mutateProject((p) => {
      const targetN = synced.qubits.length;
      let nextBackend = p.sim_settings.backend;
      if (targetN > 24 && (nextBackend === 'qutip' || nextBackend === 'kraus' || nextBackend === 'trajectory')) {
        nextBackend = 'stim';
      } else if (targetN > 14 && (nextBackend === 'qutip' || nextBackend === 'kraus')) {
        nextBackend = 'trajectory';
      } else if (targetN > 9 && nextBackend === 'qutip') {
        nextBackend = 'kraus';
      }
      return {
        ...p,
        qubits: synced.qubits,
        operations: synced.operations,
        variables: synced.variables,
        sim_settings: { ...p.sim_settings, backend: nextBackend },
      };
    });
    addLog(`${sourceLabel}已同步到可视化线路画布 (${synced.qubits.length}Q · ${synced.operations.length} 指令)`, 'info');
  };

  const handleApplyDsl = async (dsl: string) => {
    if (!project) return;
    try {
      const synced = await syncDslToProject(dsl, project);
      applySyncedCircuit(synced, 'DSL 代码');
    } catch (err) {
      setDslError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleApplyQasm = async (qasm: string) => {
    if (!project) return;
    try {
      const synced = await syncQasmToProject(qasm, project);
      applySyncedCircuit(synced, 'OpenQASM 代码');
    } catch (err) {
      setQasmError(err instanceof Error ? err.message : String(err));
    }
  };

  // ---- Handlers for Custom Gate Library ----
  const handleSaveCustomGate = (gate: CustomGateDef) => {
    mutateProject((p) => {
      const exists = p.custom_gates.some((g) => g.id === gate.id);
      const next = exists
        ? p.custom_gates.map((g) => (g.id === gate.id ? gate : g))
        : [...p.custom_gates, gate];
      return { ...p, custom_gates: next };
    });
    setActiveToolGate(gate.name);
    addLog(`自定义门 ${gate.display_label || gate.name} 已保存至门库`, 'info');
  };

  // ---- Project Import / Export ----
  const handleExportJson = () => {
    if (!project) return;
    const fname = project.name.endsWith('.json') ? project.name : `${project.name}.json`;
    downloadTextFile(fname, JSON.stringify(project, null, 2), 'application/json;charset=utf-8');
  };

  const handleApplyNlResult = (res: NlCircuitResponse, runImmediately: boolean) => {
    if (!project) return;
    const nQ = res.qubits.length;
    const curB = project.sim_settings.backend;
    const shouldSwitchBackend =
      (nQ > 6 && curB === 'qutip') ||
      (nQ > 12 && curB === 'kraus') ||
      (nQ > 24 && curB === 'trajectory') ||
      (nQ <= 6 && (curB === 'stim' || curB === 'trajectory'));
    const chosenBackend = shouldSwitchBackend ? res.recommended_backend : curB;

    const nextProj: ProjectModel = {
      ...project,
      name: res.title || project.name,
      qubits: res.qubits,
      operations: res.operations,
      variables: res.variables,
      active_hardware_id: res.recommended_hardware_id || project.active_hardware_id,
      sim_settings: {
        ...project.sim_settings,
        backend: chosenBackend,
      },
    };
    setPast((p) => [...p.slice(-30), project]);
    setFuture([]);
    setProject(nextProj);
    setOptResponse(null);
    setSweepResponse(null);
    setSimDirty(true);
    setSelection(nextProj.operations[0] ? { type: 'op', opId: nextProj.operations[0].id } : { type: 'none' });
    const firstVar = Object.keys(nextProj.variables)[0] || 'hw:depol_2q';
    setSweepConfig((prev) => ({
      ...prev,
      param_x: firstVar,
      target_bitstring: '0'.repeat(Math.max(1, nextProj.qubits.length)),
      observed_qubit: Math.min(prev.observed_qubit, Math.max(0, nextProj.qubits.length - 1)),
      backend:
        chosenBackend === 'compare'
          ? nQ > 24
            ? 'stim'
            : nQ > 12
            ? 'trajectory'
            : 'kraus'
          : chosenBackend,
    }));
    setStatusMsg(`AI 已生成线路: ${res.title} (${res.qubits.length}Q · ${res.operations.filter((o) => o.kind === 'gate').length} Gates)`);
    addLog(`自然语言生成线路: ${res.explanation}`, 'info');
    if (runImmediately) {
      handleRunSimulation(nextProj, true);
    } else {
      setMode('design');
    }
  };

  const handleExportSvg = () => {
    if (!project) return;
    const svg = buildCircuitSvg(project, undefined, project.name);
    const fname = `${project.name.replace(/\.[^.]+$/, '')}.svg`;
    downloadTextFile(fname, svg);
    addLog(`已导出量子线路矢量图: ${fname}`, 'info');
  };

  const handleImportJson = (file: File) => {
    const reader = new FileReader();
    reader.onload = async () => {
      const rawText = String(reader.result || '');
      const isQasm =
        file.name.toLowerCase().endsWith('.qasm') ||
        /^\s*(OPENQASM|include\s+"|qreg\s+\w+\s*\[|qubit\s*\[)/im.test(rawText);
      if (isQasm) {
        if (!project) return;
        try {
          const synced = await syncQasmToProject(rawText, project);
          const targetN = synced.qubits.length;
          const nextBackend =
            targetN > 24 ? 'stim' : targetN > 14 ? 'trajectory' : targetN > 8 ? 'kraus' : 'qutip';
          const importedProj: ProjectModel = {
            ...project,
            name: file.name.replace(/\.qasm$/i, '.qproj'),
            qubits: synced.qubits,
            operations: synced.operations,
            variables: synced.variables,
            sim_settings: { ...project.sim_settings, backend: nextBackend },
          };
          setProject(importedProj);
          setPast([]);
          setFuture([]);
          setOptResponse(null);
          setBenchResponse(null);
          setSweepResponse(null);
          setVqeResponse(null);
          setZneResponse(null);
          setSelection({ type: 'none' });
          setQasmError(null);
          addLog(`已导入 OpenQASM 线路: ${file.name} (${targetN}Q)`, 'info');
          handleRunSimulation(importedProj, false);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          setStatusMsg(`导入 OpenQASM 失败: ${msg}`);
          addLog(`导入 OpenQASM 失败: ${msg}`, 'error');
        }
        return;
      }
      try {
        const parsed = JSON.parse(rawText) as ProjectModel;
        if (parsed && Array.isArray(parsed.qubits) && Array.isArray(parsed.operations)) {
          setProject(parsed);
          setSelection({ type: 'none' });
          addLog(`已导入项目: ${parsed.name}`, 'info');
          handleRunSimulation(parsed, false);
        }
      } catch (err) {
        addLog(`导入项目失败: ${String(err)}`, 'error');
      }
    };
    reader.readAsText(file);
  };

  if (!project) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center gap-3 text-[var(--text-secondary)]">
        <Loader2 className="w-6 h-6 animate-spin text-[#0A84FF]" />
        <div className="text-[13px]">{statusMsg}</div>
      </div>
    );
  }

  const primaryRes = simResponse ? simResponse.results[simResponse.primary_backend] : null;
  const maxCursorStep = Math.max(0, (primaryRes?.steps.length ?? 1) - 1);

  const activeHwSummary =
    project.hardware_profiles.find((h) => h.id === project.active_hardware_id) ||
    project.hardware_profiles[0];

  const handleSwitchDomain = (nextDomain: PrimaryDomain) => {
    setDomain(nextDomain);
    if (nextDomain === 'characterization' && !gateBenchResponse) {
      handleRunGateBenchmark();
    } else if (nextDomain === 'circuits') {
      if (circuitSubMode === 'optimize' && !optResponse) handleRunOptimize();
      if (circuitSubMode === 'sweep' && !sweepResponse) handleRunSweep();
    }
  };

  const handleSwitchCircuitSubMode = (sub: CircuitSubMode) => {
    setDomain('circuits');
    setCircuitSubMode(sub);
    if (sub === 'simulate' && simResponse) {
      const pRes = simResponse.results[simResponse.primary_backend];
      if (pRes) {
        const fStr = pRes.final_fidelity !== null ? ` · F = ${pRes.final_fidelity.toFixed(4)}` : '';
        setStatusMsg(`求解完成 · ${pRes.backend_name} · ${project.qubits.length} qubits · ${pRes.total_duration_ns.toFixed(0)} ns${fStr}`);
      }
    }
    if (sub === 'optimize' && !optResponse) handleRunOptimize();
    if (sub === 'sweep' && !sweepResponse) handleRunSweep();
  };

  const handleApplyCalibrationWriteback = (wb: CalibrationWritebackItem) => {
    if (wb.scope === 'qubit' && wb.qubit != null) {
      mutateProject((p) => ({
        ...p,
        qubits: p.qubits.map((q) =>
          q.id === wb.qubit ? { ...q, [wb.field]: wb.value } : q
        ),
      }));
    } else {
      mutateProject((p) => ({
        ...p,
        hardware_profiles: p.hardware_profiles.map((h) =>
          h.id === p.active_hardware_id ? { ...h, [wb.field]: wb.value } : h
        ),
      }));
    }
    const msg = `已将标定结果写回器件参数: ${wb.label}`;
    setStatusMsg(msg);
    addLog(msg, 'info');
  };

  return (
    <div className="h-full w-full flex flex-col bg-[var(--bg-app)] text-[var(--text-primary)] overflow-hidden">
      {/* Top Level-1 Parallel Domain Toolbar */}
      <TopToolbar
        project={project}
        presets={presets}
        domain={domain}
        circuitSubMode={circuitSubMode}
        onDomainChange={handleSwitchDomain}
        onSelectPreset={(p) => {
          const clone = JSON.parse(JSON.stringify(p)) as ProjectModel;
          setProject(clone);
          setPast([]);
          setFuture([]);
          setOptResponse(null);
          setBenchResponse(null);
          setSweepResponse(null);
          setVqeResponse(null);
          setZneResponse(null);
          setNoiseBudgetResponse(null);
          setGateBenchResponse(null);
          const varKeys = Object.keys(clone.variables);
          const firstVar = varKeys[0] || 'hw:depol_2q';
          const secondVar = varKeys[1] || 'hw:depol_2q';
          const isVqe = /vqe|qaoa|ising/i.test(clone.name);
          const autoBackend =
            clone.sim_settings.backend === 'compare'
              ? clone.qubits.length > 24
                ? 'stim'
                : clone.qubits.length > 14
                ? 'trajectory'
                : 'kraus'
              : clone.sim_settings.backend;
          setSweepConfig((prev) => ({
            ...prev,
            param_x: firstVar,
            param_y: secondVar,
            target_bitstring: '0'.repeat(Math.max(1, clone.qubits.length)),
            observed_qubit: Math.min(1, Math.max(0, clone.qubits.length - 1)),
            hamiltonian_str: isVqe ? buildPresetHamiltonian('tfim', clone) : '',
            backend: autoBackend,
          }));
          setSelection(clone.operations[0] ? { type: 'op', opId: clone.operations[0].id } : { type: 'none' });
          handleRunSimulation(clone, false);
        }}
        onRenameProject={(name) => mutateProject((p) => ({ ...p, name }), false)}
        onBackendChange={(backend) =>
          mutateProject((p) => ({ ...p, sim_settings: { ...p.sim_settings, backend } }), false)
        }
        onShotsChange={(shots) =>
          mutateProject((p) => ({ ...p, sim_settings: { ...p.sim_settings, shots } }), false)
        }
        onRun={() => {
          if (domain === 'characterization') handleRunGateBenchmark();
          else if (domain === 'device') handleRunSimulation(undefined, true);
          else if (circuitSubMode === 'optimize') handleRunOptimize();
          else if (circuitSubMode === 'sweep') handleRunSweep();
          else handleRunSimulation(undefined, circuitSubMode === 'design');
        }}
        onStop={handleStopTask}
        isBusy={isBusy}
        isDirty={simDirty}
        canUndo={past.length > 0}
        canRedo={future.length > 0}
        onUndo={() => {
          if (past.length === 0) return;
          const prev = past[past.length - 1];
          setPast((p) => p.slice(0, -1));
          setFuture((f) => [project, ...f]);
          setProject(prev);
          setSimDirty(true);
        }}
        onRedo={() => {
          if (future.length === 0) return;
          const next = future[0];
          setFuture((f) => f.slice(1));
          setPast((p) => [...p, project]);
          setProject(next);
          setSimDirty(true);
        }}
        darkMode={darkMode}
        onToggleDark={() => setDarkMode((d) => !d)}
        showLeft={showLeft}
        showRight={showRight}
        onToggleLeft={() => setShowLeft((v) => !v)}
        onToggleRight={() => setShowRight((v) => !v)}
        onNewBlankProject={handleNewBlankProject}
        onExportJson={handleExportJson}
        onExportSvg={handleExportSvg}
        onImportJson={handleImportJson}
        onOpenNlModal={() => setNlModalOpen(true)}
      />

      {/* Main IDE Workbench: 3-Domain ActivityBar + Context Sidebars + Center Stage */}
      <div className="flex-1 min-h-0 flex overflow-hidden">
        <ActivityBar
          domain={domain}
          onDomainChange={handleSwitchDomain}
          showLeft={showLeft}
          showRight={showRight}
          onToggleLeft={() => setShowLeft((v) => !v)}
          onToggleRight={() => setShowRight((v) => !v)}
          darkMode={darkMode}
          onToggleDark={() => setDarkMode((d) => !d)}
          onOpenNlModal={() => setNlModalOpen(true)}
        />

        {/* Left Circuit Explorer Sidebar (Dedicated to Quantum Circuits Domain) */}
        {domain === 'circuits' && showLeft && (
          <LeftNavigator
            project={project}
            selection={selection}
            activeToolGate={activeToolGate}
            runHistory={runHistory}
            activeRunId={activeRunId}
            compareRunId={compareRunId}
            onSelect={setSelection}
            onAddQubit={handleAddQubit}
            onResizeRegister={handleResizeRegister}
            onDeleteQubit={handleDeleteQubit}
            onRenameQubit={(qid, name) =>
              mutateProject((p) => ({
                ...p,
                qubits: p.qubits.map((q) => (q.id === qid ? { ...q, name } : q)),
              }))
            }
            onReorderQubit={handleReorderQubit}
            onPickGateTool={(gName) => {
              setActiveToolGate(gName);
              if (mode !== 'design') setMode('design');
            }}
            onOpenNewGateModal={() => {
              setEditingCustomGate(null);
              setMacroSeedOps(null);
              setGateModalOpen(true);
            }}
            onEditCustomGate={(cg) => {
              setEditingCustomGate(cg);
              setMacroSeedOps(null);
              setGateModalOpen(true);
            }}
            onDeleteCustomGate={(gid) =>
              mutateProject((p) => ({
                ...p,
                custom_gates: p.custom_gates.filter((g) => g.id !== gid),
              }))
            }
            onSelectHardware={(hwId) =>
              mutateProject((p) => ({ ...p, active_hardware_id: hwId }), false)
            }
            onSelectRunHistory={(run) => {
              setSimResponse(run.response);
              setActiveRunId(run.id);
              setSimDirty(false);
              if (compareRunId === run.id) setCompareRunId(null);
              const pRes = run.response.results[run.response.primary_backend];
              setTimeCursorStep(Math.max(0, (pRes?.steps.length ?? 1) - 1));
              setMode('simulate');
            }}
            onToggleCompareRun={(runId) => {
              setCompareRunId((prev) => (prev === runId ? null : runId));
              setMode('simulate');
            }}
          />
        )}

        {/* Center Main Stage */}
        <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
          {/* Level-2 Sub-Tab Bar for Domain 1: 量子线路 (Design | Simulate | Optimize | Sweep) */}
          {domain === 'circuits' && (
            <div className="h-9 px-3.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0 select-none">
              <div className="flex items-center gap-4 h-full overflow-x-auto min-w-0">
                {(
                  [
                    { id: 'design', label: '原理图设计', code: 'SCH', icon: CircuitBoard },
                    { id: 'simulate', label: '演化仿真', code: 'SIM', icon: Activity },
                    { id: 'optimize', label: '编译综合', code: 'SYN', icon: Cpu },
                    { id: 'sweep', label: '参数扫描 & VQE', code: 'SWP', icon: LineChart },
                  ] as const
                ).map((t) => {
                  const Icon = t.icon;
                  const active = circuitSubMode === t.id;
                  return (
                    <button
                      key={t.id}
                      onClick={() => handleSwitchCircuitSubMode(t.id)}
                      className={`h-full text-[12px] border-b-2 flex items-center gap-1.5 transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                        active
                          ? 'border-[var(--accent)] text-[var(--text-primary)] font-semibold'
                          : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      <Icon className={`w-3.5 h-3.5 ${active ? 'text-[var(--accent)]' : 'opacity-70'}`} />
                      <span>{t.label}</span>
                      <span className="text-[9.5px] font-mono-num text-[var(--text-tertiary)]">
                        {t.code}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Quick Hardware Drive Telemetry Link */}
              <button
                type="button"
                onClick={() => {
                  setDomain('device');
                  setDeviceSubTab('drive');
                }}
                title="点击进入「器件与硬件」调整单比特拉比频率 Ω_R 与脉冲上升/下降沿"
                className="hidden lg:inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[3px] bg-[var(--bg-subtle)] hover:border-[var(--accent)] border border-[var(--border-hairline)] text-[11px] font-mono-num text-[var(--text-secondary)] hover:text-[var(--accent)] transition-colors cursor-pointer shrink-0"
              >
                <Cpu className="w-3 h-3 text-[var(--accent)]" />
                <span>{activeHwSummary?.name}</span>
                <span>·</span>
                <span>Ω_R={activeHwSummary?.rabi_mhz ?? 2.5} MHz</span>
                <span>
                  ({(activeHwSummary?.drive_mode || 'fixed_duration') === 'fixed_rabi' ? '固定Ω_R' : `t_1q=${activeHwSummary?.t_1q_ns}ns`})
                </span>
              </button>
            </div>
          )}

          {mode === 'design' && (
            <>
              <CircuitCanvas
                project={project}
                selection={selection}
                activeToolGate={activeToolGate}
                onSelect={setSelection}
                onPickGateTool={setActiveToolGate}
                onPlaceGate={handlePlaceGate}
                onMoveOp={handleMoveOp}
                onDeleteOp={(opId) => {
                  mutateProject((p) => ({
                    ...p,
                    operations: p.operations.filter((o) => o.id !== opId),
                  }));
                  if (selection.type === 'op' && selection.opId === opId) {
                    setSelection({ type: 'none' });
                  }
                }}
                onDeleteMultipleOps={(opIds) => {
                  const drop = new Set(opIds);
                  mutateProject((p) => ({
                    ...p,
                    operations: p.operations.filter((o) => !drop.has(o.id)),
                  }));
                  setSelection({ type: 'none' });
                }}
                onAddQubit={handleAddQubit}
                onPackageMacro={handlePackageMacro}
                onOpenNlModal={() => setNlModalOpen(true)}
              />
              <BottomDock
                project={project}
                dslText={dslText}
                qasmText={qasmText}
                logs={logs}
                dslError={dslError}
                qasmError={qasmError}
                onApplyDsl={handleApplyDsl}
                onApplyQasm={handleApplyQasm}
                onUpdateVariable={(name, val) =>
                  mutateProject(
                    (p) => ({ ...p, variables: { ...p.variables, [name]: val } }),
                    false
                  )
                }
                onAddVariable={(name, val) =>
                  mutateProject((p) => ({ ...p, variables: { ...p.variables, [name]: val } }))
                }
              />
            </>
          )}

          {mode === 'simulate' && (
            <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
              {!simCircuitCollapsed && (
                <>
                  <div
                    style={{ height: simCircuitHeight }}
                    className="shrink-0 min-h-[120px] max-h-[55%] flex flex-col overflow-hidden"
                  >
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
                      timeCursorStep={timeCursorStep}
                      maxCursorStep={maxCursorStep}
                      cursorReadoutLabel={
                        primaryRes
                          ? (() => {
                              const sf = Math.max(0, Math.min(primaryRes.steps.length - 1, Math.floor(timeCursorStep)));
                              const sc = Math.min(primaryRes.steps.length - 1, sf + 1);
                              const frac = Math.max(0, timeCursorStep - sf);
                              const tNs =
                                primaryRes.steps[sf].elapsed_time_ns +
                                frac * (primaryRes.steps[sc].elapsed_time_ns - primaryRes.steps[sf].elapsed_time_ns);
                              return `S${Math.round(timeCursorStep)} · ${tNs.toFixed(0)}ns`;
                            })()
                          : null
                      }
                      onTimeCursorChange={handleDirectTimeCursorChange}
                      onStepJump={handleSmoothStepJump}
                      isPlayingTimeline={isPlayingTimeline}
                      onTogglePlayTimeline={() => {
                        if (timeCursorStep >= maxCursorStep - 0.02) setTimeCursorStep(0);
                        setIsPlayingTimeline((v) => !v);
                      }}
                      playbackSpeed={playbackSpeed}
                      onPlaybackSpeedChange={setPlaybackSpeed}
                    />
                  </div>
                  <div
                    title="上下拖拽调整线路与绘图区域比例（双击重置）"
                    onDoubleClick={() => setSimCircuitHeight(206)}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      const startY = e.clientY;
                      const startH = simCircuitHeight;
                      const onMove = (ev: MouseEvent) => {
                        const next = Math.max(120, Math.min(420, startH + (ev.clientY - startY)));
                        setSimCircuitHeight(next);
                      };
                      const onUp = () => {
                        window.removeEventListener('mousemove', onMove);
                        window.removeEventListener('mouseup', onUp);
                      };
                      window.addEventListener('mousemove', onMove);
                      window.addEventListener('mouseup', onUp);
                    }}
                    className="h-2 shrink-0 bg-[var(--bg-app)] border-y border-[var(--border-hairline)] hover:bg-[#0A84FF]/15 cursor-row-resize flex items-center justify-center group transition-colors select-none"
                  >
                    <span className="w-10 h-[3px] rounded-full bg-[var(--border-input)] group-hover:bg-[#0A84FF] transition-colors" />
                  </div>
                </>
              )}
              <SimulateResults
                project={project}
                simResponse={simResponse}
                zneResponse={zneResponse}
                onRunZne={() => handleRunZne()}
                noiseBudgetResponse={noiseBudgetResponse}
                onRunNoiseBudget={() => handleRunNoiseBudget()}
                isBusy={isBusy}
                timeCursorStep={timeCursorStep}
                maxCursorStep={maxCursorStep}
                onTimeCursorChange={handleDirectTimeCursorChange}
                onStepJump={handleSmoothStepJump}
                isPlayingTimeline={isPlayingTimeline}
                onTogglePlayTimeline={() => {
                  if (timeCursorStep >= maxCursorStep - 0.02) setTimeCursorStep(0);
                  setIsPlayingTimeline((v) => !v);
                }}
                playbackSpeed={playbackSpeed}
                onPlaybackSpeedChange={setPlaybackSpeed}
                observedQubit={observedQubit}
                onObservedQubitChange={setObservedQubit}
                darkMode={darkMode}
                circuitCollapsed={simCircuitCollapsed}
                onToggleCircuitCollapsed={() => setSimCircuitCollapsed((v) => !v)}
                compareRun={runHistory.find((r) => r.id === compareRunId) ?? null}
                onClearCompareRun={() => setCompareRunId(null)}
                isStale={simDirty}
                onRefreshSim={() => handleRunSimulation(undefined, false)}
              />
            </div>
          )}

          {mode === 'optimize' && (
            <OptimizeWorkspace
              project={project}
              optResponse={optResponse}
              benchResponse={benchResponse}
              darkMode={darkMode}
              onRunOptimize={() => handleRunOptimize()}
              onRunBenchmark={() => handleRunHwBenchmark()}
              onRunUnitarySynthesis={handleRunUnitarySynthesis}
              onApplyOptimizedToDesign={(targetNumQubits) => {
                if (!optResponse) return;
                mutateProject((p) => {
                  let nextQ = p.qubits;
                  if (targetNumQubits && targetNumQubits > p.qubits.length) {
                    nextQ = [...p.qubits];
                    for (let i = p.qubits.length; i < targetNumQubits; i++) {
                      nextQ.push({
                        id: i,
                        name: `q${i}`,
                        color: QUBIT_DOT_PALETTE[i % QUBIT_DOT_PALETTE.length],
                        init_state: '|0>',
                        init_theta: 0,
                        init_phi: 0,
                        t1_us: null,
                        t2_us: null,
                        freq_ghz: null,
                        depol_1q: null,
                        readout_err: null,
                        pos_x: Number((i * 4.5).toFixed(1)),
                        pos_y: 0,
                      });
                    }
                  }
                  return {
                    ...p,
                    qubits: nextQ,
                    operations: optResponse.optimized_operations,
                  };
                });
                setMode('design');
                addLog('已将最优硬件原生门序列应用到设计画布', 'info');
              }}
              onSelectHardwareAndApply={(hardwareId, ops) => {
                mutateProject((p) => ({
                  ...p,
                  active_hardware_id: hardwareId,
                  operations: ops,
                }));
                setMode('design');
                addLog(`已切换硬件配置 (${hardwareId}) 并应用最优原生门序列`, 'info');
              }}
              isBusy={isBusy}
            />
          )}

          {mode === 'sweep' && (
            <SweepWorkspace
              activeSubMode={sweepConfig.mode}
              sweepResponse={sweepResponse}
              vqeResponse={vqeResponse}
              darkMode={darkMode}
              onRunSweep={() => handleRunSweep()}
              onApplyOptimalParams={(optParams, runSim) => {
                const updated: ProjectModel = {
                  ...project,
                  variables: { ...project.variables, ...optParams },
                };
                setPast((p) => [...p.slice(-30), project]);
                setFuture([]);
                setProject(updated);
                setSimDirty(true);
                addLog(
                  `已写回变分最优参数: ${Object.entries(optParams)
                    .map(([k, v]) => `${k}=${v.toFixed(4)}`)
                    .join(', ')}`,
                  'info'
                );
                if (runSim) {
                  handleRunSimulation(updated, true);
                }
              }}
              isBusy={isBusy}
            />
          )}

          {domain === 'characterization' && (
            <BenchmarkWorkspace
              project={project}
              activeProtocol={benchmarkConfig.protocol}
              onSwitchProtocol={(proto) => {
                setBenchmarkConfig((prev) => ({ ...prev, protocol: proto }));
                handleRunGateBenchmark(undefined, { protocol: proto });
              }}
              response={gateBenchResponse}
              darkMode={darkMode}
              onRunBenchmark={() => handleRunGateBenchmark()}
              onApplyWriteback={handleApplyCalibrationWriteback}
              onLoadCircuitToDesign={(ops, label) => {
                mutateProject((p) => ({
                  ...p,
                  name: label,
                  operations: ops,
                }));
                setMode('design');
                addLog(`已将标定序列 (${label}) 导入设计画布`, 'info');
              }}
              isBusy={isBusy}
            />
          )}

          {domain === 'device' && (
            <DeviceWorkspace
              project={project}
              activeTab={deviceSubTab}
              onTabChange={setDeviceSubTab}
              darkMode={darkMode}
              onUpdateHardware={(hwId, patch) =>
                mutateProject((p) => ({
                  ...p,
                  hardware_profiles: p.hardware_profiles.map((h) =>
                    h.id === hwId ? { ...h, ...patch } : h
                  ),
                }))
              }
              onSelectHardware={(hwId) =>
                mutateProject((p) => ({ ...p, active_hardware_id: hwId }), false)
              }
              onUpdateQubit={(qubitId, patch) =>
                mutateProject((p) => ({
                  ...p,
                  qubits: p.qubits.map((q) => (q.id === qubitId ? { ...q, ...patch } : q)),
                }))
              }
              onAddQubit={handleAddQubit}
              onDeleteQubit={handleDeleteQubit}
              onApplyLatticeLayout={(positions) => {
                mutateProject((p) => ({
                  ...p,
                  qubits: p.qubits.map((q, idx) =>
                    positions[idx] ? { ...q, pos_x: positions[idx].x, pos_y: positions[idx].y } : q
                  ),
                }));
                addLog(`已应用二维中性原子晶格排布 (${positions.length} 原子)`, 'info');
              }}
              onToggleCustomEdge={(u, v) => {
                const pair: [number, number] = [Math.min(u, v), Math.max(u, v)];
                mutateProject((p) => ({
                  ...p,
                  hardware_profiles: p.hardware_profiles.map((h) => {
                    if (h.id !== p.active_hardware_id) return h;
                    const exists = h.edges.some(([a, b]) => a === pair[0] && b === pair[1]);
                    const nextEdges = exists
                      ? h.edges.filter(([a, b]) => !(a === pair[0] && b === pair[1]))
                      : [...h.edges, pair];
                    return { ...h, edges: nextEdges };
                  }),
                }));
              }}
              onToggleCustomGateNative={(gateId, isNative) =>
                mutateProject((p) => ({
                  ...p,
                  custom_gates: p.custom_gates.map((g) =>
                    g.id === gateId ? { ...g, is_native: isNative } : g
                  ),
                }))
              }
              onOpenNewGateModal={() => {
                setEditingCustomGate(null);
                setMacroSeedOps(null);
                setGateModalOpen(true);
              }}
              onEditCustomGate={(cg) => {
                setEditingCustomGate(cg);
                setMacroSeedOps(null);
                setGateModalOpen(true);
              }}
              onDeleteCustomGate={(gid) =>
                mutateProject((p) => ({
                  ...p,
                  custom_gates: p.custom_gates.filter((g) => g.id !== gid),
                }))
              }
              onJumpToCalibrate={(qid, proto) => {
                setDomain('characterization');
                setBenchmarkConfig((prev) => ({
                  ...prev,
                  target_qubit: qid,
                  protocol: proto,
                }));
                handleRunGateBenchmark(undefined, { target_qubit: qid, protocol: proto });
              }}
            />
          )}
        </main>

        {/* Right Context-Sensitive Inspector (Shown in Circuits & Characterization; Device workspace uses full stage) */}
        {domain !== 'device' && showRight && (
          <RightInspector
            mode={mode}
            project={project}
            selection={selection}
            simResponse={simResponse}
            observedQubit={observedQubit}
            onObservedQubitChange={setObservedQubit}
            onUpdateOp={(opId, patch) =>
              mutateProject((p) => ({
                ...p,
                operations: p.operations.map((o) => (o.id === opId ? { ...o, ...patch } : o)),
              }))
            }
            onDeleteOp={(opId) => {
              mutateProject((p) => ({
                ...p,
                operations: p.operations.filter((o) => o.id !== opId),
              }));
              setSelection({ type: 'none' });
            }}
            onDuplicateOp={handleDuplicateOp}
            onUpdateQubit={(qubitId, patch) =>
              mutateProject((p) => ({
                ...p,
                qubits: p.qubits.map((q) => (q.id === qubitId ? { ...q, ...patch } : q)),
              }))
            }
            onApplyLatticeLayout={(positions) => {
              mutateProject((p) => ({
                ...p,
                qubits: p.qubits.map((q, idx) =>
                  positions[idx] ? { ...q, pos_x: positions[idx].x, pos_y: positions[idx].y } : q
                ),
              }));
              addLog(`已应用二维中性原子晶格排布 (${positions.length} 原子)`, 'info');
            }}
            onUpdateHardware={(hwId, patch) =>
              mutateProject((p) => ({
                ...p,
                hardware_profiles: p.hardware_profiles.map((h) =>
                  h.id === hwId ? { ...h, ...patch } : h
                ),
              }))
            }
            onToggleCustomEdge={(u, v) => {
              const pair: [number, number] = [Math.min(u, v), Math.max(u, v)];
              mutateProject((p) => ({
                ...p,
                hardware_profiles: p.hardware_profiles.map((h) => {
                  if (h.id !== p.active_hardware_id) return h;
                  const exists = h.edges.some(([a, b]) => a === pair[0] && b === pair[1]);
                  const nextEdges = exists
                    ? h.edges.filter(([a, b]) => !(a === pair[0] && b === pair[1]))
                    : [...h.edges, pair];
                  return { ...h, edges: nextEdges };
                }),
              }));
            }}
            onUpdateSimSettings={(patch) => {
              const nextProj: ProjectModel = {
                ...project,
                sim_settings: { ...project.sim_settings, ...patch },
              };
              setProject(nextProj);
              if (mode === 'simulate' && nextProj.qubits.length <= 6 && !isBusy) {
                handleRunSimulation(nextProj, false);
              }
            }}
            onToggleCustomGateNative={(gateId, isNative) =>
              mutateProject((p) => ({
                ...p,
                custom_gates: p.custom_gates.map((g) =>
                  g.id === gateId ? { ...g, is_native: isNative } : g
                ),
              }))
            }
            onEditCustomGate={(cg) => {
              setEditingCustomGate(cg);
              setMacroSeedOps(null);
              setGateModalOpen(true);
            }}
            onEnsureVariable={(name, val) =>
              mutateProject(
                (p) =>
                  name in p.variables
                    ? p
                    : { ...p, variables: { ...p.variables, [name]: Number(val.toFixed(4)) } },
                false
              )
            }
            preferredNative2Q={preferredNative2Q}
            onPreferredNative2QChange={setPreferredNative2Q}
            evalNoisyOpt={evalNoisyOpt}
            onEvalNoisyOptChange={setEvalNoisyOpt}
            onRunOptimize={() => handleRunOptimize()}
            sweepConfig={sweepConfig}
            onSweepConfigChange={(patch) =>
              setSweepConfig((prev) => ({ ...prev, ...patch }))
            }
            hardwareSweepParams={hardwareSweepParams}
            onRunSweep={() => handleRunSweep()}
            onRunZne={() => handleRunZne()}
            onRunNoiseBudget={() => handleRunNoiseBudget()}
            onCancelSweep={handleCancelSweep}
            benchmarkConfig={benchmarkConfig}
            onBenchmarkConfigChange={(patch) =>
              setBenchmarkConfig((prev) => ({ ...prev, ...patch }))
            }
            onRunGateBenchmark={() => handleRunGateBenchmark()}
            onOpenDeviceWorkspace={() => {
              setDomain('device');
              setDeviceSubTab('drive');
            }}
            isBusy={isBusy}
          />
        )}
      </div>

      {/* Bottom Industrial EDA Segmented Status Bar */}
      <footer className="h-[24px] shrink-0 bg-[var(--bg-activity)] border-t border-[var(--border-hairline)] px-2.5 flex items-center justify-between text-[11px] font-mono-num text-[var(--text-secondary)] select-none">
        <div className="flex items-center gap-2 truncate min-w-0">
          {isBusy ? (
            <span className="px-1.5 py-0 rounded-[2px] bg-[var(--accent)] text-white text-[9.5px] font-bold flex items-center gap-1 shrink-0">
              <Loader2 className="w-2.5 h-2.5 animate-spin" />
              BUSY
            </span>
          ) : statusMsg.includes('失败') ? (
            <span className="px-1.5 py-0 rounded-[2px] bg-red-600 text-white text-[9.5px] font-bold flex items-center gap-1 shrink-0">
              <AlertCircle className="w-2.5 h-2.5" />
              ERR
            </span>
          ) : (
            <span className="px-1.5 py-0 rounded-[2px] bg-emerald-600/90 text-white text-[9.5px] font-bold flex items-center gap-1 shrink-0">
              <CheckCircle2 className="w-2.5 h-2.5" />
              READY
            </span>
          )}
          <span className="truncate text-[var(--text-primary)]/90">{statusMsg}</span>
        </div>
        <div className="flex items-center h-full shrink-0 divide-x divide-[var(--border-hairline)]">
          <span className="px-2.5 hidden md:inline">
            HW:{' '}
            {project.hardware_profiles.find((h) => h.id === project.active_hardware_id)?.name ||
              project.active_hardware_id}
          </span>
          <span className="px-2.5 uppercase">ENG: {project.sim_settings.backend}</span>
          <span className="px-2.5">{project.qubits.length}Q</span>
          <span className="px-2.5">{project.operations.filter((o) => o.kind === 'gate').length} GATES</span>
          <span className="pl-2.5">{project.custom_gates.length} MACRO</span>
        </div>
      </footer>

      {/* Export / Save-As Confirmation Toast */}
      {exportToast && (
        <div className="fixed bottom-8 right-4 z-50 max-w-md px-3.5 py-2 rounded-[4px] bg-[var(--bg-panel)] border border-emerald-500/60 shadow-xl flex items-center gap-2 text-[11.5px] font-mono-num text-[var(--text-primary)] pointer-events-none">
          <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
          <span className="truncate">{exportToast}</span>
        </div>
      )}

      {/* Custom Gate Builder Sheet Modal */}
      <CustomGateModal
        open={gateModalOpen}
        editingGate={editingCustomGate}
        existingCustomGates={project.custom_gates}
        macroSeedOps={macroSeedOps}
        macroSeedQubits={macroSeedQubits}
        onClose={() => setGateModalOpen(false)}
        onSave={handleSaveCustomGate}
      />

      {/* Natural Language -> Quantum Circuit AI Modal */}
      <NlCircuitModal
        open={nlModalOpen}
        project={project}
        onClose={() => setNlModalOpen(false)}
        onApplyResult={handleApplyNlResult}
      />
    </div>
  );
}
