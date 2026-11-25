import React, { useState, useRef, useEffect } from 'react';
import {
  Activity,
  Atom,
  Bot,
  ChevronDown,
  CircuitBoard,
  Cpu,
  Download,
  FolderOpen,
  Gauge,
  Keyboard,
  LineChart,
  Loader2,
  Moon,
  PanelLeft,
  PanelRight,
  Play,
  Redo2,
  Square,
  Sun,
  Undo2,
  Upload,
} from 'lucide-react';
import type { ProjectModel, SolverBackendType, WorkMode } from '../types';
import { BACKEND_LABELS } from '../constants';

interface TopToolbarProps {
  project: ProjectModel;
  presets: ProjectModel[];
  mode: WorkMode;
  onModeChange: (m: WorkMode) => void;
  onSelectPreset: (p: ProjectModel) => void;
  onRenameProject: (name: string) => void;
  onBackendChange: (b: SolverBackendType) => void;
  onShotsChange: (shots: number) => void;
  onRun: () => void;
  onStop?: () => void;
  isBusy: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  darkMode: boolean;
  onToggleDark: () => void;
  showLeft: boolean;
  showRight: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  onNewBlankProject: () => void;
  onExportJson: () => void;
  onExportSvg: () => void;
  onImportJson: (file: File) => void;
  onOpenNlModal: () => void;
  isDirty?: boolean;
}

const MODES: {
  id: WorkMode;
  zh: string;
  short: string;
  code: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  { id: 'design', zh: '原理图设计', short: '设计', code: 'SCH', icon: CircuitBoard },
  { id: 'simulate', zh: '演化仿真', short: '仿真', code: 'SIM', icon: Activity },
  { id: 'optimize', zh: '编译综合', short: '综合', code: 'SYN', icon: Cpu },
  { id: 'sweep', zh: '扫描 / VQE', short: '扫描', code: 'SWP', icon: LineChart },
  { id: 'benchmark', zh: '门标定基准', short: '标定', code: 'CAL', icon: Gauge },
];

export const TopToolbar: React.FC<TopToolbarProps> = ({
  project,
  presets,
  mode,
  onModeChange,
  onSelectPreset,
  onRenameProject,
  onBackendChange,
  onShotsChange,
  onRun,
  onStop,
  isBusy,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  darkMode,
  onToggleDark,
  showLeft,
  showRight,
  onToggleLeft,
  onToggleRight,
  onNewBlankProject,
  onExportJson,
  onExportSvg,
  onImportJson,
  onOpenNlModal,
  isDirty = false,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(project.name);
  const menuRef = useRef<HTMLDivElement>(null);
  const shortcutsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setNameDraft(project.name);
  }, [project.name]);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
      if (shortcutsRef.current && !shortcutsRef.current.contains(e.target as Node)) {
        setShortcutsOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);

  const formatShots = (s: number) => (s >= 1000 ? `${Math.round(s / 1000)}k` : String(s));

  return (
    <header className="h-[40px] shrink-0 bg-[var(--bg-header)] border-b border-[var(--border-hairline)] px-2 flex items-center justify-between gap-2 select-none z-30">
      {/* Left: Brand Emblem + Project File Selector + Undo/Redo + IDE Workbench Tabs */}
      <div className="flex items-center h-full min-w-0 overflow-hidden">
        {/* Industrial Brand Badge */}
        <div className="flex items-center gap-2 px-2 h-full border-r border-[var(--border-hairline)] shrink-0">
          <div className="w-6 h-6 rounded-[3px] bg-[var(--accent)] text-white flex items-center justify-center shrink-0">
            <Atom className="w-3.5 h-3.5 stroke-[2]" />
          </div>
          <div className="hidden sm:flex items-baseline gap-1.5">
            <span className="text-[12.5px] font-bold tracking-tight text-[var(--text-primary)]">
              QUANTUM STUDIO
            </span>
            <span className="text-[10px] font-mono-num text-[var(--accent)] font-semibold">
              EDA
            </span>
          </div>
        </div>

        {/* Project Selector Dropdown */}
        <div className="relative flex items-center h-full px-2 border-r border-[var(--border-hairline)]" ref={menuRef}>
          {editingName ? (
            <input
              value={nameDraft}
              autoFocus
              onChange={(e) => setNameDraft(e.target.value)}
              onBlur={() => {
                setEditingName(false);
                if (nameDraft.trim()) onRenameProject(nameDraft.trim());
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  setEditingName(false);
                  if (nameDraft.trim()) onRenameProject(nameDraft.trim());
                }
              }}
              className="h-6 text-[11.5px] px-1.5 rounded-[2px] border border-[var(--accent)] bg-[var(--bg-panel)] outline-none w-40 font-mono-num"
            />
          ) : (
            <button
              onClick={() => setMenuOpen((o) => !o)}
              onDoubleClick={(e) => {
                e.stopPropagation();
                setMenuOpen(false);
                setEditingName(true);
              }}
              className="h-7 px-2 rounded-[3px] bg-[var(--bg-panel)] border border-[var(--border-hairline)] hover:border-[var(--border-input)] text-[11.5px] font-mono-num text-[var(--text-primary)] flex items-center gap-1.5 max-w-[200px] cursor-pointer transition-colors"
              title="点击切换预设工程或导入/导出文件，双击重命名当前工程"
            >
              <FolderOpen className="w-3.5 h-3.5 text-[var(--accent)] shrink-0" />
              <span className="truncate">{project.name}</span>
              <ChevronDown className="w-3 h-3 text-[var(--text-secondary)] shrink-0" />
            </button>
          )}

          {/* Preset / File Menu */}
          {menuOpen && (
            <div className="absolute left-2 top-[38px] w-80 studio-card shadow-xl py-1 z-50">
              <div className="px-3 py-1 text-[10px] font-mono-num font-bold uppercase tracking-wider text-[var(--text-secondary)] bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)]">
                工程模板库 PRESETS ({presets.length})
              </div>
              {presets.map((p) => (
                <button
                  key={p.name}
                  onClick={() => {
                    onSelectPreset(p);
                    setMenuOpen(false);
                  }}
                  className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] flex items-center justify-between gap-2 cursor-pointer"
                >
                  <span className="truncate font-mono-num">{p.name}</span>
                  <span className="shrink-0 flex items-center gap-1 text-[10px] font-mono-num text-[var(--text-secondary)]">
                    <span className="px-1.5 py-0.25 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                      {p.qubits.length}Q
                    </span>
                    <span className="uppercase opacity-80">{p.sim_settings.backend}</span>
                  </span>
                </button>
              ))}
              <div className="my-1 border-t border-[var(--border-hairline)]" />
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onNewBlankProject();
                }}
                className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] flex items-center gap-2 font-medium cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5" />
                新建空白工程 (2 Qubits)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  setEditingName(true);
                }}
                className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-black/5 dark:hover:bg-white/5 flex items-center gap-2 cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                重命名当前工程
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onExportJson();
                }}
                className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-black/5 dark:hover:bg-white/5 flex items-center gap-2 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                导出工程文件 (.qproj.json)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onExportSvg();
                }}
                className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-black/5 dark:hover:bg-white/5 flex items-center gap-2 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-[var(--accent)]" />
                导出原理图矢量图 (.svg)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  fileInputRef.current?.click();
                }}
                className="w-full text-left px-3 py-1.5 text-[12px] hover:bg-black/5 dark:hover:bg-white/5 flex items-center gap-2 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                导入工程或 OpenQASM (.json, .qasm)
              </button>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.qproj,.qasm,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onImportJson(f);
              e.target.value = '';
            }}
          />
        </div>

        {/* Undo / Redo */}
        <div className="hidden xl:flex items-center gap-0.5 px-1.5 h-full border-r border-[var(--border-hairline)]">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            title="撤销 (Ctrl+Z)"
            className="w-7 h-7 rounded-[3px] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/8 disabled:opacity-30 cursor-pointer"
          >
            <Undo2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            title="重做 (Ctrl+Y)"
            className="w-7 h-7 rounded-[3px] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/8 disabled:opacity-30 cursor-pointer"
          >
            <Redo2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Flush IDE Workbench Mode Tabs (auto-collapses on narrow screens where left ActivityBar handles switching) */}
        <div className="hidden xl:flex items-center h-full shrink-0">
          {MODES.map((m) => {
            const Icon = m.icon;
            const active = mode === m.id;
            return (
              <button
                key={m.id}
                onClick={() => onModeChange(m.id)}
                className={`relative h-full px-2.5 2xl:px-3 border-r border-[var(--border-hairline)] flex items-center gap-1.5 text-[12px] transition-colors cursor-pointer whitespace-nowrap ${
                  active
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.03] dark:hover:bg-white/[0.03]'
                }`}
              >
                {active && (
                  <span className="absolute top-0 left-0 right-0 h-[2px] bg-[var(--accent)]" />
                )}
                <Icon className={`w-3.5 h-3.5 ${active ? 'text-[var(--accent)]' : 'opacity-75'}`} />
                <span className="hidden 2xl:inline">{m.zh}</span>
                <span className="2xl:hidden">{m.short}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right: AI Synthesis + Solver Engine + Shots + Industrial Run/Stop + Quick Toggles */}
      <div className="flex items-center gap-1.5 shrink-0">
        <button
          onClick={onOpenNlModal}
          title="用自然语言对话综合与精调量子线路 (Ctrl+K)"
          className="h-7 px-2.5 rounded-[3px] border border-[var(--accent)]/40 bg-[var(--accent-soft)] hover:bg-[var(--accent)]/25 text-[var(--accent)] text-[11.5px] font-semibold flex items-center gap-1.5 cursor-pointer transition-colors shrink-0"
        >
          <Bot className="w-3.5 h-3.5 shrink-0" />
          <span className="hidden lg:inline">AI 综合</span>
          <kbd className="hidden 2xl:inline text-[9.5px] font-mono-num px-1 py-0 rounded-[2px] bg-[var(--bg-panel)] border border-[var(--border-hairline)]">
            Ctrl+K
          </kbd>
        </button>

        <div className="relative flex items-center">
          <select
            value={project.sim_settings.backend}
            onChange={(e) => onBackendChange(e.target.value as SolverBackendType)}
            className="h-7 appearance-none bg-[var(--bg-panel)] border border-[var(--border-hairline)] hover:border-[var(--border-input)] rounded-[3px] pl-2 pr-6 text-[11.5px] font-mono-num font-medium text-[var(--text-primary)] outline-none focus:border-[var(--accent)] cursor-pointer transition-colors"
          >
            {Object.entries(BACKEND_LABELS).map(([k, lbl]) => (
              <option key={k} value={k}>
                {lbl}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3 h-3 text-[var(--text-secondary)] pointer-events-none absolute right-1.5" />
        </div>

        <div
          className="hidden md:flex h-7 items-center gap-1 bg-[var(--bg-panel)] border border-[var(--border-hairline)] rounded-[3px] px-2"
          title="测量采样次数 Shots"
        >
          <span className="text-[9.5px] font-mono-num text-[var(--text-tertiary)] uppercase font-bold">
            SHOTS
          </span>
          <select
            value={project.sim_settings.shots}
            onChange={(e) => onShotsChange(Number(e.target.value))}
            className="bg-[var(--bg-panel)] text-[11.5px] font-semibold font-mono-num text-[var(--text-primary)] outline-none cursor-pointer"
          >
            {[1000, 4096, 10000, 20000, 50000, 100000].map((v) => (
              <option key={v} value={v}>
                {formatShots(v)}
              </option>
            ))}
          </select>
        </div>

        {isBusy && onStop ? (
          <button
            onClick={onStop}
            title="中止当前正在执行的计算任务"
            className="h-7 px-3 rounded-[3px] bg-red-600 hover:bg-red-500 text-white text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer transition-colors"
          >
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <Square className="w-2.5 h-2.5 fill-current" />
            <span>停止</span>
          </button>
        ) : (
          <button
            onClick={onRun}
            disabled={isBusy}
            title={isDirty ? '线路或参数已变更，点击或按 Ctrl+Enter 重新求解' : '运行当前工作区任务 (Ctrl+Enter)'}
            className="relative h-7 px-3.5 rounded-[3px] bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-400/25 text-[12px] font-semibold flex items-center gap-1.5 disabled:opacity-60 cursor-pointer transition-colors"
          >
            {isBusy ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3 h-3 fill-current" />
            )}
            <span>
              {mode === 'optimize'
                ? '编译综合'
                : mode === 'sweep'
                ? '执行扫描'
                : mode === 'benchmark'
                ? '运行标定'
                : '运行仿真'}
            </span>
            {isDirty && !isBusy && (
              <span
                className="w-2 h-2 rounded-full bg-amber-300"
                title="线路或参数已变更，待重新运行"
              />
            )}
          </button>
        )}

        <div className="h-4 w-[1px] bg-[var(--border-hairline)] mx-0.5" />

        <div className="relative" ref={shortcutsRef}>
          <button
            onClick={() => setShortcutsOpen((o) => !o)}
            title="快捷键参考手册"
            className={`w-7 h-7 rounded-[3px] flex items-center justify-center transition-colors cursor-pointer ${
              shortcutsOpen
                ? 'text-[var(--accent)] bg-[var(--accent-soft)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.05] dark:hover:bg-white/[0.08]'
            }`}
          >
            <Keyboard className="w-3.5 h-3.5" />
          </button>

          {shortcutsOpen && (
            <div className="absolute right-0 top-9 w-72 studio-card shadow-xl p-2.5 z-50 space-y-2">
              <div className="flex items-center justify-between border-b border-[var(--border-hairline)] pb-1.5">
                <span className="text-[11.5px] font-bold text-[var(--text-primary)]">
                  EDA 工作台快捷键
                </span>
                <span className="text-[10px] font-mono-num text-[var(--accent)]">v2.5</span>
              </div>
              <div className="space-y-1.5 text-[11px]">
                {[
                  ['Ctrl + K', '唤起 / 关闭 AI 线路综合助手'],
                  ['Ctrl + Enter', '执行运行 / 编译底部代码'],
                  ['Space', '播放 / 暂停演化时间轴 (SIM)'],
                  ['Ctrl + Z / Y', '撤销 / 重做编辑'],
                  ['Ctrl + D', '复制当前选中量子门'],
                  ['Delete', '删除当前选中量子门'],
                  ['Shift + 框选', '批量框选并封装为宏门'],
                  ['Esc', '释放放置工具 / 清除选中'],
                ].map(([keyStr, desc]) => (
                  <div key={keyStr} className="flex items-center justify-between gap-2">
                    <span className="text-[var(--text-secondary)] truncate">{desc}</span>
                    <kbd className="shrink-0 px-1.5 py-0.5 rounded-[2px] bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[10px] font-semibold text-[var(--text-primary)]">
                      {keyStr}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <button
          onClick={onToggleLeft}
          title={showLeft ? '隐藏左侧资源管理器' : '显示左侧资源管理器'}
          className={`w-7 h-7 rounded-[3px] flex items-center justify-center transition-colors cursor-pointer ${
            showLeft
              ? 'text-[var(--text-primary)] bg-black/[0.05] dark:bg-white/[0.08]'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
          }`}
        >
          <PanelLeft className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={onToggleRight}
          title={showRight ? '隐藏右侧属性检查器' : '显示右侧属性检查器'}
          className={`w-7 h-7 rounded-[3px] flex items-center justify-center transition-colors cursor-pointer ${
            showRight
              ? 'text-[var(--text-primary)] bg-black/[0.05] dark:bg-white/[0.08]'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
          }`}
        >
          <PanelRight className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={onToggleDark}
          title={darkMode ? '切换浅色原理图模式' : '切换深色工业模式'}
          className="w-7 h-7 rounded-[3px] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.05] dark:hover:bg-white/[0.08] cursor-pointer transition-colors"
        >
          {darkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>
      </div>
    </header>
  );
};
