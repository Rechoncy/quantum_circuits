import React, { useState, useRef, useEffect } from 'react';
import {
  Atom,
  Play,
  Square,
  ChevronDown,
  Undo2,
  Redo2,
  Sun,
  Moon,
  FolderOpen,
  Download,
  Upload,
  Loader2,
  PanelLeft,
  PanelRight,
  Sparkles,
  Keyboard,
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

const MODES: { id: WorkMode; zh: string; en: string }[] = [
  { id: 'design', zh: '设计', en: 'Design' },
  { id: 'simulate', zh: '模拟', en: 'Simulate' },
  { id: 'optimize', zh: '优化', en: 'Optimize' },
  { id: 'sweep', zh: '扫描', en: 'Sweep' },
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
    <header className="h-[52px] shrink-0 studio-vibrancy border-b border-[var(--border-hairline)] px-3.5 flex items-center justify-between gap-2 select-none z-30">
      {/* Left: Traffic Lights + Brand + Project Menu */}
      <div className="flex items-center gap-2.5 min-w-0">
        <div className="flex items-center gap-1.5 mr-0.5">
          <span className="w-3 h-3 rounded-full bg-[#FF5F56] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#FFBD2E] ring-1 ring-black/10" />
          <span className="w-3 h-3 rounded-full bg-[#27C93F] ring-1 ring-black/10" />
        </div>

        <button
          onClick={onToggleLeft}
          title={showLeft ? '隐藏导航栏' : '显示导航栏'}
          className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
            showLeft
              ? 'text-[var(--text-primary)] bg-black/[0.04] dark:bg-white/[0.07] hover:bg-black/[0.07] dark:hover:bg-white/[0.11]'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.08]'
          }`}
        >
          <PanelLeft className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2.5 relative" ref={menuRef}>
          <div className="w-7 h-7 rounded-lg bg-[var(--accent-soft)] ring-1 ring-[var(--accent)]/20 text-[var(--accent)] flex items-center justify-center shrink-0">
            <Atom className="w-4 h-4 stroke-[1.9]" />
          </div>
          <div className="leading-tight min-w-0">
            <div className="text-[13px] font-semibold tracking-tight text-[var(--text-primary)]">
              Quantum Studio
            </div>
            <div className="flex items-center gap-1">
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
                  className="text-[11px] px-1.5 py-0 rounded border border-[var(--accent)] bg-[var(--bg-panel)] outline-none w-36 font-mono-num"
                />
              ) : (
                <button
                  onClick={() => setMenuOpen((o) => !o)}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    setMenuOpen(false);
                    setEditingName(true);
                  }}
                  className="text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] flex items-center gap-0.5 truncate max-w-[170px] cursor-pointer"
                  title="点击切换预设项目或导入导出，双击重命名"
                >
                  <span className="truncate">{project.name}</span>
                  <ChevronDown className="w-3 h-3 shrink-0" />
                </button>
              )}
            </div>
          </div>

          {/* Preset / File Dropdown */}
          {menuOpen && (
            <div className="absolute left-0 top-11 w-80 studio-card shadow-xl py-1.5 z-50">
              <div className="px-3 py-1 text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                预设项目模板 ({presets.length})
              </div>
              {presets.map((p) => (
                <button
                  key={p.name}
                  onClick={() => {
                    onSelectPreset(p);
                    setMenuOpen(false);
                  }}
                  className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-[#0A84FF]/10 hover:text-[#0A84FF] flex items-center justify-between gap-2 cursor-pointer"
                >
                  <span className="truncate">{p.name}</span>
                  <span className="shrink-0 flex items-center gap-1 text-[10.5px] font-mono-num text-[var(--text-secondary)]">
                    <span className="px-1.5 py-0.25 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                      {p.qubits.length}Q
                    </span>
                    <span className="uppercase opacity-75">{p.sim_settings.backend}</span>
                  </span>
                </button>
              ))}
              <div className="my-1 border-t border-[var(--border-hairline)]" />
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onNewBlankProject();
                }}
                className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-[#0A84FF]/10 hover:text-[#0A84FF] flex items-center gap-2 font-medium cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5" />
                新建空白项目 (2 Qubits)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  setEditingName(true);
                }}
                className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-2 cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                重命名当前项目
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onExportJson();
                }}
                className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-2 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                导出项目 (.qproj.json)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onExportSvg();
                }}
                className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-2 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-[#0A84FF]" />
                导出线路矢量图 (.svg)
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  fileInputRef.current?.click();
                }}
                className="w-full text-left px-3 py-1.5 text-[12.5px] hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-2 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                导入项目或 QASM (.json, .qasm)
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
        <div className="hidden xl:flex items-center gap-0.5 ml-2 pl-2 border-l border-[var(--border-hairline)]">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            title="撤销 (Ctrl+Z)"
            className="p-1.5 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30"
          >
            <Undo2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            title="重做 (Ctrl+Y)"
            className="p-1.5 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30"
          >
            <Redo2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Center: macOS 4-Mode Segmented Control */}
      <div className="bg-black/[0.045] dark:bg-white/[0.06] border border-[var(--border-hairline)] p-0.5 rounded-xl flex items-center gap-0.5 shrink-0">
        {MODES.map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onModeChange(m.id)}
              className={`px-3.5 py-1 rounded-[9px] transition-all flex flex-col items-center leading-none min-w-[62px] cursor-pointer ${
                active
                  ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] shadow-xs ring-1 ring-black/[0.04] dark:ring-white/[0.06]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              <span className={`text-[12px] ${active ? 'font-semibold' : 'font-medium'}`}>
                {m.zh}
              </span>
              <span className={`text-[9.5px] mt-0.5 ${active ? 'text-[var(--accent)] font-medium' : 'opacity-65'}`}>
                {m.en}
              </span>
            </button>
          );
        })}
      </div>

      {/* Right: AI Generate + Backend Picker + Shots + Primary Run Button */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onOpenNlModal}
          title="用自然语言对话生成与精调量子线路 (Ctrl+K)"
          className="h-8 px-2.5 rounded-lg border border-[var(--accent)]/28 bg-[var(--accent-soft)] hover:bg-[var(--accent)]/16 text-[var(--accent)] text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer transition-colors shrink-0"
        >
          <Sparkles className="w-3.5 h-3.5 shrink-0" />
          <span className="hidden lg:inline">AI 助手</span>
          <span className="hidden 2xl:inline text-[9.5px] font-mono-num opacity-75 px-1 py-0.25 rounded bg-[var(--bg-panel)]/70">
            ⌘K
          </span>
        </button>

        <div className="relative flex items-center">
          <select
            value={project.sim_settings.backend}
            onChange={(e) => onBackendChange(e.target.value as SolverBackendType)}
            className="h-8 appearance-none bg-[var(--bg-subtle)] border border-[var(--border-hairline)] hover:border-[var(--border-input)] rounded-lg pl-2.5 pr-7 text-[12px] font-medium text-[var(--text-primary)] outline-none focus:border-[var(--accent)] cursor-pointer transition-colors"
          >
            {Object.entries(BACKEND_LABELS).map(([k, lbl]) => (
              <option key={k} value={k}>
                {lbl}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-[var(--text-secondary)] pointer-events-none absolute right-2" />
        </div>

        <div
          className="hidden sm:flex h-8 flex-col items-center justify-center bg-[var(--bg-subtle)] border border-[var(--border-hairline)] rounded-lg px-2 leading-tight"
          title="测量采样次数 Shots"
        >
          <span className="text-[8.5px] text-[var(--text-tertiary)] uppercase tracking-wider font-semibold">
            Shots
          </span>
          <select
            value={project.sim_settings.shots}
            onChange={(e) => onShotsChange(Number(e.target.value))}
            className="bg-transparent text-[11px] font-semibold font-mono-num text-[var(--text-primary)] outline-none cursor-pointer -mt-0.5"
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
            title="点击中止当前正在执行的求解 / 优化 / 扫描任务"
            className="relative h-8 px-3.5 rounded-lg bg-red-500 hover:bg-red-600 text-white shadow-xs text-[12.5px] font-semibold flex items-center gap-1.5 cursor-pointer transition-colors"
          >
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <Square className="w-2.5 h-2.5 fill-current" />
            <span>停止</span>
          </button>
        ) : (
          <button
            onClick={onRun}
            disabled={isBusy}
            title={isDirty ? '线路或参数已修改，点击或按 Ctrl+Enter 更新求解结果' : '运行当前模式 (Ctrl+Enter)'}
            className="relative h-8 px-4 rounded-lg studio-btn-primary text-[12.5px] font-semibold flex items-center gap-1.5 disabled:opacity-60 cursor-pointer"
          >
            {isBusy ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            <span>{mode === 'optimize' ? '运行优化' : mode === 'sweep' ? '运行扫描' : '运行'}</span>
            {isDirty && !isBusy && (
              <span
                className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-amber-400 ring-2 ring-[var(--bg-panel)]"
                title="线路或参数已变更，待更新"
              />
            )}
          </button>
        )}

        <div className="relative" ref={shortcutsRef}>
          <button
            onClick={() => setShortcutsOpen((o) => !o)}
            title="快捷键速查表"
            className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
              shortcutsOpen
                ? 'text-[#0A84FF] bg-[#0A84FF]/10'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.05] dark:hover:bg-white/[0.09]'
            }`}
          >
            <Keyboard className="w-4 h-4" />
          </button>

          {shortcutsOpen && (
            <div className="absolute right-0 top-10 w-72 studio-card shadow-xl p-3 z-50 space-y-2">
              <div className="flex items-center justify-between border-b border-[var(--border-hairline)] pb-1.5">
                <span className="text-[12px] font-semibold text-[var(--text-primary)]">
                  工作台快捷键速查
                </span>
                <span className="text-[10px] font-mono-num text-[var(--text-secondary)]">v2.1</span>
              </div>
              <div className="space-y-1.5 text-[11.5px]">
                {[
                  ['Ctrl + K', '唤起 / 关闭 AI 量子线路助手'],
                  ['Ctrl + Enter', '运行当前模式 / 同步底部代码'],
                  ['Space', '播放 / 暂停演化时间轴 (Simulate)'],
                  ['Ctrl + Z / Y', '撤销 / 重做线路与参数编辑'],
                  ['Ctrl + D', '复制当前选中的量子门'],
                  ['Delete', '删除当前选中的量子门'],
                  ['Shift + 拖拽', '框选多个门并封装为复合门'],
                  ['Esc', '取消工具选择 / 清除选中态'],
                ].map(([keyStr, desc]) => (
                  <div key={keyStr} className="flex items-center justify-between gap-2">
                    <span className="text-[var(--text-secondary)] truncate">{desc}</span>
                    <kbd className="shrink-0 px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[10.5px] font-semibold text-[var(--text-primary)]">
                      {keyStr}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <button
          onClick={onToggleDark}
          title={darkMode ? '切换浅色模式' : '切换深色模式'}
          className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.05] dark:hover:bg-white/[0.09] cursor-pointer transition-colors"
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <button
          onClick={onToggleRight}
          title={showRight ? '隐藏检查器' : '显示检查器'}
          className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
            showRight
              ? 'text-[var(--text-primary)] bg-black/[0.04] dark:bg-white/[0.07] hover:bg-black/[0.07] dark:hover:bg-white/[0.11]'
              : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.08]'
          }`}
        >
          <PanelRight className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
