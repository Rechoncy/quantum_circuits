import React, { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Plus,
  Trash2,
  Cpu,
  Check,
  History,
  Box,
  Layers,
  Sparkles,
  FunctionSquare,
  Activity,
  Pencil,
  GripVertical,
  GitCompareArrows,
} from 'lucide-react';
import type {
  CustomGateDef,
  ProjectModel,
  RunHistoryItem,
  SelectionTarget,
} from '../types';
import { BUILTIN_GATES } from '../constants';

interface LeftNavigatorProps {
  project: ProjectModel;
  selection: SelectionTarget;
  activeToolGate: string | null;
  runHistory: RunHistoryItem[];
  activeRunId: string | null;
  compareRunId: string | null;
  onSelect: (sel: SelectionTarget) => void;
  onAddQubit: () => void;
  onResizeRegister?: (targetCount: number) => void;
  onDeleteQubit: (qubitId: number) => void;
  onRenameQubit: (qubitId: number, name: string) => void;
  onReorderQubit: (fromIdx: number, toIdx: number) => void;
  onPickGateTool: (gateName: string) => void;
  onOpenNewGateModal: () => void;
  onEditCustomGate: (gate: CustomGateDef) => void;
  onDeleteCustomGate: (gateId: string) => void;
  onSelectHardware: (hwId: string) => void;
  onSelectRunHistory: (run: RunHistoryItem) => void;
  onToggleCompareRun: (runId: string) => void;
}

export const LeftNavigator: React.FC<LeftNavigatorProps> = ({
  project,
  selection,
  activeToolGate,
  runHistory,
  activeRunId,
  compareRunId,
  onSelect,
  onAddQubit,
  onResizeRegister,
  onDeleteQubit,
  onRenameQubit,
  onReorderQubit,
  onPickGateTool,
  onOpenNewGateModal,
  onEditCustomGate,
  onDeleteCustomGate,
  onSelectHardware,
  onSelectRunHistory,
  onToggleCompareRun,
}) => {
  const [openReg, setOpenReg] = useState(true);
  const [openLib, setOpenLib] = useState(true);
  const [openBuiltin, setOpenBuiltin] = useState(false);
  const [openCustom, setOpenCustom] = useState(true);
  const [openHw, setOpenHw] = useState(true);
  const [openRuns, setOpenRuns] = useState(true);
  const [renamingQ, setRenamingQ] = useState<number | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [dragQIdx, setDragQIdx] = useState<number | null>(null);
  const [qubitFilter, setQubitFilter] = useState('');
  const [showBatchResize, setShowBatchResize] = useState(false);
  const [targetQDraft, setTargetQDraft] = useState('');

  const customGateIcon = (type: CustomGateDef['definition_type']) => {
    if (type === 'matrix') return <Box className="w-3.5 h-3.5 text-purple-500 shrink-0" />;
    if (type === 'expression') return <FunctionSquare className="w-3.5 h-3.5 text-indigo-500 shrink-0" />;
    if (type === 'hamiltonian') return <Activity className="w-3.5 h-3.5 text-blue-500 shrink-0" />;
    return <Layers className="w-3.5 h-3.5 text-violet-500 shrink-0" />;
  };

  return (
    <aside className="w-[244px] shrink-0 bg-[var(--bg-sidebar)] border-r border-[var(--border-hairline)] flex flex-col select-none">
      <div className="flex-1 min-h-0 overflow-y-auto py-2.5 px-2.5 space-y-1.5">
      {/* Section 1: Quantum Register */}
      <div>
        <div className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-colors">
          <button
            onClick={() => setOpenReg((v) => !v)}
            className="flex items-center gap-1.5 flex-1 text-left cursor-pointer"
          >
            {openReg ? <ChevronDown className="w-3.5 h-3.5 text-[var(--text-tertiary)]" /> : <ChevronRight className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />}
            量子寄存器
          </button>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => {
                setTargetQDraft(String(project.qubits.length));
                setShowBatchResize((v) => !v);
              }}
              title="点击批量设置量子比特数量 (1~500Q)"
              className="px-1.5 py-0.25 rounded-md bg-black/[0.045] dark:bg-white/[0.07] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] text-[10px] font-mono-num text-[var(--text-secondary)] cursor-pointer transition-colors"
            >
              {project.qubits.length}Q
            </button>
            <button
              onClick={onAddQubit}
              title="快速添加 1 个比特"
              className="w-5 h-5 rounded-md flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--accent)] hover:bg-[var(--accent-soft)] cursor-pointer transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Batch Qubit Count Popover Strip */}
        {showBatchResize && onResizeRegister && (
          <div className="mx-2 mt-1 mb-1.5 p-2 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-hairline)] shadow-2xs space-y-1.5">
            <div className="flex items-center gap-1">
              <input
                type="number"
                min={1}
                max={500}
                value={targetQDraft}
                onChange={(e) => setTargetQDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const v = Number(targetQDraft);
                    if (v >= 1) onResizeRegister(v);
                    setShowBatchResize(false);
                  }
                }}
                placeholder="1~500"
                className="w-16 px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
              <button
                type="button"
                onClick={() => {
                  const v = Number(targetQDraft);
                  if (v >= 1) onResizeRegister(v);
                  setShowBatchResize(false);
                }}
                className="px-2 py-0.5 rounded bg-[#0A84FF] text-white text-[11px] font-medium cursor-pointer"
              >
                设为 NQ
              </button>
              <button
                type="button"
                onClick={() => onResizeRegister(project.qubits.length + 5)}
                className="px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] hover:border-[#0A84FF] text-[10.5px] font-mono-num cursor-pointer"
              >
                +5
              </button>
            </div>
            <div className="flex items-center gap-1 text-[10px] font-mono-num">
              {[8, 16, 32, 64, 100].map((presetQ) => (
                <button
                  key={presetQ}
                  type="button"
                  onClick={() => {
                    onResizeRegister(presetQ);
                    setShowBatchResize(false);
                  }}
                  className="flex-1 py-0.5 rounded bg-[var(--bg-subtle)] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)] text-[var(--text-secondary)] cursor-pointer transition-colors"
                >
                  {presetQ}Q
                </button>
              ))}
            </div>
          </div>
        )}

        {openReg && (
          <div className="mt-0.5 pl-2 space-y-1">
            {project.qubits.length > 10 && (
              <div className="pr-1">
                <input
                  value={qubitFilter}
                  onChange={(e) => setQubitFilter(e.target.value)}
                  placeholder={`搜索比特 (共 ${project.qubits.length}Q)...`}
                  className="w-full px-2 py-1 rounded-md bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[11px] font-mono-num outline-none focus:border-[#0A84FF]"
                />
              </div>
            )}

            <div className={`space-y-0.5 ${project.qubits.length > 10 ? 'max-h-[228px] overflow-y-auto pr-0.5' : ''}`}>
              {project.qubits
                .map((q, idx) => ({ q, idx }))
                .filter(({ q, idx }) => {
                  const kw = qubitFilter.trim().toLowerCase();
                  if (!kw) return true;
                  return q.name.toLowerCase().includes(kw) || String(idx) === kw;
                })
                .map(({ q, idx }) => {
                  const isSelected = selection.type === 'qubit' && selection.qubitId === q.id;
                  const hasOverride =
                    q.t1_us !== null ||
                    q.t2_us !== null ||
                    q.depol_1q !== null ||
                    q.readout_err !== null;
                  return (
                    <div
                      key={q.id}
                      draggable={project.qubits.length <= 40}
                      onDragStart={() => setDragQIdx(idx)}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={() => {
                        if (dragQIdx !== null && dragQIdx !== idx) {
                          onReorderQubit(dragQIdx, idx);
                        }
                        setDragQIdx(null);
                      }}
                      onClick={() => onSelect({ type: 'qubit', qubitId: q.id })}
                      onDoubleClick={() => {
                        setRenamingQ(q.id);
                        setRenameDraft(q.name);
                      }}
                      className={`group flex items-center justify-between px-2 py-1 rounded-md text-[12.5px] cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-[#0A84FF]/12 text-[#0A84FF] font-medium'
                          : 'text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06]'
                      }`}
                    >
                      <div className="flex items-center gap-1.5 min-w-0">
                        <GripVertical className="w-3 h-3 opacity-0 group-hover:opacity-40 shrink-0 cursor-grab" />
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: q.color }}
                        />
                        {renamingQ === q.id ? (
                          <input
                            value={renameDraft}
                            autoFocus
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => setRenameDraft(e.target.value)}
                            onBlur={() => {
                              if (renameDraft.trim()) onRenameQubit(q.id, renameDraft.trim());
                              setRenamingQ(null);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                if (renameDraft.trim()) onRenameQubit(q.id, renameDraft.trim());
                                setRenamingQ(null);
                              }
                            }}
                            className="w-16 px-1 text-[12px] bg-[var(--bg-panel)] border border-[#0A84FF] rounded outline-none font-mono-num"
                          />
                        ) : (
                          <span className="font-mono-num truncate">{q.name}</span>
                        )}
                        <span className="text-[11px] font-mono-num text-[var(--text-secondary)]">
                          {q.init_state.replace('>', '⟩')}
                        </span>
                        {hasOverride && (
                          <span
                            className="w-1.5 h-1.5 rounded-full bg-amber-500"
                            title="已单独覆盖物理参数"
                          />
                        )}
                      </div>
                      {project.qubits.length > 1 && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteQubit(q.id);
                          }}
                          title="删除此比特"
                          className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-red-500/15 hover:text-red-500 transition-opacity"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  );
                })}
            </div>

            <div className="flex items-center gap-1 pt-0.5">
              <button
                onClick={onAddQubit}
                className="flex-1 flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[12px] text-[var(--text-secondary)] hover:text-[#0A84FF] hover:bg-[#0A84FF]/8 transition-colors cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>添加比特</span>
              </button>
              {onResizeRegister && (
                <button
                  type="button"
                  onClick={() => onResizeRegister(project.qubits.length + 5)}
                  title="一次添加 5 个量子比特"
                  className="px-2 py-1.5 rounded-md text-[11px] font-mono-num text-[var(--text-secondary)] hover:text-[#0A84FF] hover:bg-[#0A84FF]/8 transition-colors cursor-pointer"
                >
                  +5Q
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Section 2: Gate Library */}
      <div className="pt-2 border-t border-[var(--border-hairline)]">
        <button
          onClick={() => setOpenLib((v) => !v)}
          className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] cursor-pointer transition-colors"
        >
          <span className="flex items-center gap-1.5">
            {openLib ? <ChevronDown className="w-3.5 h-3.5 text-[var(--text-tertiary)]" /> : <ChevronRight className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />}
            门库
          </span>
          <span className="px-1.5 py-0.25 rounded-md bg-black/[0.045] dark:bg-white/[0.07] text-[10px] font-mono-num text-[var(--text-secondary)]">
            {BUILTIN_GATES.length + project.custom_gates.length}
          </span>
        </button>

        {openLib && (
          <div className="mt-0.5 pl-2 space-y-1">
            {/* Sub-group: Built-in Gates */}
            <div>
              <button
                onClick={() => setOpenBuiltin((v) => !v)}
                className="w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-[12px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.03]"
              >
                {openBuiltin ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                <span>内置门</span>
              </button>
              {openBuiltin && (
                <div className="grid grid-cols-3 gap-1 px-2 pt-1 pb-1.5">
                  {BUILTIN_GATES.map((g) => {
                    const active = activeToolGate === g.name;
                    return (
                      <button
                        key={g.name}
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData('application/quantum-gate', g.name);
                          onPickGateTool(g.name);
                        }}
                        onClick={() => onPickGateTool(g.name)}
                        title={g.description}
                        className={`px-1.5 py-1 rounded-md text-[11px] font-mono-num font-medium border text-center transition-all cursor-grab active:cursor-grabbing ${
                          active
                            ? 'bg-[#0A84FF]/15 border-[#0A84FF] text-[#0A84FF]'
                            : 'bg-[var(--bg-panel)] border-[var(--border-hairline)] text-[var(--text-primary)] hover:border-[#0A84FF]/50'
                        }`}
                      >
                        {g.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Sub-group: Custom Gates */}
            <div>
              <button
                onClick={() => setOpenCustom((v) => !v)}
                className="w-full flex items-center justify-between px-2 py-1 rounded-md text-[12px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.03]"
              >
                <span className="flex items-center gap-1.5">
                  {openCustom ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                  <span>自定义门</span>
                </span>
                <span className="text-[10px] font-mono-num">{project.custom_gates.length}</span>
              </button>

              {openCustom && (
                <div className="pl-2 mt-0.5 space-y-0.5">
                  {project.custom_gates.map((cg) => {
                    const isActive = activeToolGate === cg.name;
                    return (
                      <div
                        key={cg.id}
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData('application/quantum-gate', cg.name);
                          onPickGateTool(cg.name);
                        }}
                        onClick={() => {
                          onPickGateTool(cg.name);
                          onSelect({ type: 'custom_gate', gateId: cg.id });
                        }}
                        title={`${cg.description || cg.name} (拖到画布或点击放置)`}
                        className={`group flex items-center justify-between px-2 py-1 rounded-md text-[12px] cursor-pointer transition-colors ${
                          isActive
                            ? 'bg-[#0A84FF]/12 text-[#0A84FF] font-medium'
                            : 'text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06]'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 min-w-0">
                          {customGateIcon(cg.definition_type)}
                          <span className="truncate font-mono-num">
                            {cg.display_label || cg.name}
                          </span>
                          {cg.is_native && (
                            <span className="text-[9.5px] px-1 py-0 rounded bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 font-semibold shrink-0">
                              原生
                            </span>
                          )}
                        </div>
                        <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onEditCustomGate(cg);
                            }}
                            title="编辑自定义门"
                            className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10"
                          >
                            <Pencil className="w-3 h-3" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteCustomGate(cg.id);
                            }}
                            title="删除自定义门"
                            className="p-0.5 rounded hover:bg-red-500/15 hover:text-red-500"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    );
                  })}

                  <button
                    onClick={onOpenNewGateModal}
                    className="w-full flex items-center gap-1.5 px-2.5 py-1.5 mt-0.5 rounded-md text-[12px] text-[var(--text-secondary)] hover:text-[#0A84FF] hover:bg-[#0A84FF]/8 transition-colors cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>新建自定义门</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Section 3: Hardware Profiles */}
      <div className="pt-2 border-t border-[var(--border-hairline)]">
        <button
          onClick={() => setOpenHw((v) => !v)}
          className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] cursor-pointer transition-colors"
        >
          <span className="flex items-center gap-1.5">
            {openHw ? <ChevronDown className="w-3.5 h-3.5 text-[var(--text-tertiary)]" /> : <ChevronRight className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />}
            硬件配置
          </span>
        </button>

        {openHw && (
          <div className="mt-1 pl-1 space-y-0.5">
            {project.hardware_profiles.map((hp) => {
              const isActive = hp.id === project.active_hardware_id;
              const isInspected = selection.type === 'hardware' && selection.hardwareId === hp.id;
              return (
                <button
                  key={hp.id}
                  onClick={() => {
                    onSelectHardware(hp.id);
                    onSelect({ type: 'hardware', hardwareId: hp.id });
                  }}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[12px] transition-colors cursor-pointer ${
                    isInspected
                      ? 'bg-[var(--accent-soft)] text-[var(--accent)] font-medium'
                      : isActive
                      ? 'bg-black/[0.045] dark:bg-white/[0.07] text-[var(--text-primary)] font-medium'
                      : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.03] dark:hover:bg-white/[0.05]'
                  }`}
                >
                  <span className="flex items-center gap-1.5 truncate">
                    {hp.platform === 'neutral_atom' ? (
                      <Sparkles className="w-3.5 h-3.5 shrink-0 text-[var(--accent)]" />
                    ) : (
                      <Cpu className="w-3.5 h-3.5 shrink-0" />
                    )}
                    <span className="truncate">{hp.name}</span>
                  </span>
                  {isActive && <Check className="w-3.5 h-3.5 text-[var(--accent)] shrink-0" />}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Section 4: Experiment Run History */}
      <div className="pt-2 border-t border-[var(--border-hairline)]">
        <button
          onClick={() => setOpenRuns((v) => !v)}
          className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] cursor-pointer transition-colors"
        >
          <span className="flex items-center gap-1.5">
            {openRuns ? <ChevronDown className="w-3.5 h-3.5 text-[var(--text-tertiary)]" /> : <ChevronRight className="w-3.5 h-3.5 text-[var(--text-tertiary)]" />}
            实验记录
          </span>
          <span className="px-1.5 py-0.25 rounded-md bg-black/[0.045] dark:bg-white/[0.07] text-[10px] font-mono-num text-[var(--text-secondary)]">
            {runHistory.length}
          </span>
        </button>

        {openRuns && (
          <div className="mt-1 pl-1 space-y-0.5">
            {runHistory.length === 0 ? (
              <div className="px-3 py-1.5 text-[11.5px] text-[var(--text-tertiary)]">
                暂无运行记录
              </div>
            ) : (
              runHistory.map((run) => {
                const isActive = activeRunId === run.id;
                const isCompared = compareRunId === run.id;
                return (
                  <div
                    key={run.id}
                    onClick={() => onSelectRunHistory(run)}
                    className={`group w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[12px] transition-colors cursor-pointer ${
                      isActive
                        ? 'bg-[var(--accent-soft)] text-[var(--accent)] font-medium'
                        : isCompared
                        ? 'bg-purple-500/12 text-purple-600 dark:text-purple-300 font-medium'
                        : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.03] dark:hover:bg-white/[0.05]'
                    }`}
                  >
                    <span className="flex items-center gap-1.5 min-w-0">
                      <History className="w-3.5 h-3.5 shrink-0 opacity-80" />
                      <span className="font-mono-num truncate">{run.label}</span>
                      <span className="text-[9.5px] px-1 py-0 rounded bg-black/[0.04] dark:bg-white/[0.07] text-[var(--text-tertiary)] font-mono-num shrink-0">
                        {run.backend.slice(0, 5)}
                      </span>
                    </span>
                    <div className="flex items-center gap-1 shrink-0">
                      {run.fidelity !== null && (
                        <span className="text-[10.5px] font-mono-num font-semibold opacity-90">
                          {run.fidelity.toFixed(3)}
                        </span>
                      )}
                      {!isActive && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onToggleCompareRun(run.id);
                          }}
                          title={isCompared ? '取消对比此历史运行' : '固定为对比基准 (Diff)'}
                          className={`p-0.5 rounded transition-opacity cursor-pointer ${
                            isCompared
                              ? 'opacity-100 text-purple-600 dark:text-purple-300 bg-purple-500/15'
                              : 'opacity-0 group-hover:opacity-100 text-[var(--text-secondary)] hover:text-purple-600 hover:bg-purple-500/15'
                          }`}
                        >
                          <GitCompareArrows className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
      </div>

      {/* Subtle Bottom Shortcut Hint Strip */}
      <div className="px-3 py-2 border-t border-[var(--border-hairline)] flex items-center justify-between text-[10px] font-mono-num text-[var(--text-tertiary)] shrink-0">
        <span>⌘K AI助手</span>
        <span>⌘↵ 运行</span>
        <span>Space 播放</span>
      </div>
    </aside>
  );
};
