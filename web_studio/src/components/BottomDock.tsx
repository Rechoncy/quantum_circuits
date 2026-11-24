import React, { useState, useEffect, useRef } from 'react';
import {
  ChevronUp,
  ChevronDown,
  RefreshCw,
  Copy,
  Check,
  AlertCircle,
  Download,
} from 'lucide-react';
import type { ProjectModel } from '../types';
import { downloadTextFile } from '../utils/exportCircuitSvg';

interface BottomDockProps {
  project: ProjectModel;
  dslText: string;
  qasmText: string;
  logs: { time: string; level: 'info' | 'warn' | 'error'; text: string }[];
  dslError: string | null;
  onApplyDsl: (dsl: string) => void;
  onUpdateVariable: (name: string, val: number) => void;
  onAddVariable: (name: string, val: number) => void;
}

export const BottomDock: React.FC<BottomDockProps> = ({
  project,
  dslText,
  qasmText,
  logs,
  dslError,
  onApplyDsl,
  onUpdateVariable,
  onAddVariable,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [dockHeight, setDockHeight] = useState(210);
  const [tab, setTab] = useState<'code' | 'params' | 'qasm' | 'logs'>('code');
  const [draft, setDraft] = useState(dslText);
  const [dirty, setDirty] = useState(false);
  const [copied, setCopied] = useState(false);
  const [newVarName, setNewVarName] = useState('');
  const [newVarVal, setNewVarVal] = useState('1.5708');
  const gutterRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setDraft(dslText);
    setDirty(false);
  }, [dslText]);

  const lines = draft.split('\n');
  const varEntries = Object.entries(project.variables);

  const handleCopy = (txt: string) => {
    navigator.clipboard.writeText(txt).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const handleDownloadQasm = () => {
    const base = (project.name || 'circuit').replace(/\.[^.]+$/, '');
    downloadTextFile(`${base}.qasm`, qasmText || '// OPENQASM 3.0;', 'text/plain;charset=utf-8');
  };

  const handleStartResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = dockHeight;
    const onMove = (ev: MouseEvent) => {
      const nextH = Math.max(130, Math.min(460, startH + (startY - ev.clientY)));
      setDockHeight(nextH);
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div
      style={{ height: collapsed ? 36 : dockHeight }}
      className="relative shrink-0 bg-[var(--bg-panel)] border-t border-[var(--border-hairline)] flex flex-col"
    >
      {/* Top Drag-Resize Handle */}
      {!collapsed && (
        <div
          onMouseDown={handleStartResize}
          title="上下拖动调整代码栏高度"
          className="absolute -top-1 left-0 right-0 h-2 cursor-ns-resize hover:bg-[var(--accent)]/25 z-20"
        />
      )}
      {/* Tab Header */}
      <div className="h-9 px-4 border-b border-[var(--border-hairline)] flex items-center justify-between shrink-0 select-none">
        <div className="flex items-center gap-5 h-full">
          {[
            { id: 'code', label: '代码' },
            { id: 'params', label: `参数 (${varEntries.length})` },
            { id: 'qasm', label: 'OpenQASM 3' },
            { id: 'logs', label: `日志 (${logs.length})` },
          ].map((t) => {
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => {
                  setTab(t.id as typeof tab);
                  if (collapsed) setCollapsed(false);
                }}
                className={`h-full text-[12.5px] font-medium border-b-2 transition-colors cursor-pointer ${
                  active
                    ? 'border-[#0A84FF] text-[var(--text-primary)] font-semibold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          {tab === 'code' && dirty && !collapsed && (
            <button
              onClick={() => {
                onApplyDsl(draft);
                setDirty(false);
              }}
              className="px-2.5 py-0.5 rounded-md bg-[#0A84FF] text-white text-[11.5px] font-medium flex items-center gap-1 cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" />
              同步至画布 (Ctrl+Enter)
            </button>
          )}
          {tab === 'qasm' && !collapsed && (
            <button
              onClick={handleDownloadQasm}
              className="px-2 py-0.5 rounded-md border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[11px] text-[var(--text-secondary)] flex items-center gap-1 cursor-pointer"
              title="下载 OpenQASM 3.0 (.qasm) 文件"
            >
              <Download className="w-3 h-3" />
              <span>.qasm</span>
            </button>
          )}
          {(tab === 'code' || tab === 'qasm') && !collapsed && (
            <button
              onClick={() => handleCopy(tab === 'code' ? draft : qasmText)}
              className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 text-[var(--text-secondary)]"
              title="复制代码"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          )}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 text-[var(--text-secondary)]"
            title={collapsed ? '展开底部面板' : '折叠底部面板'}
          >
            {collapsed ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Content */}
      {!collapsed && (
        <div className="flex-1 overflow-auto">
          {tab === 'code' && (
            <div className="flex flex-col h-full">
              {dslError && (
                <div className="px-4 py-1 bg-red-50 dark:bg-red-950/50 text-red-600 dark:text-red-300 text-[11.5px] flex items-center gap-1.5 border-b border-red-200 dark:border-red-800">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{dslError}</span>
                </div>
              )}
              <div className="flex flex-1 min-h-0 overflow-hidden">
                <div
                  ref={gutterRef}
                  className="w-11 py-2.5 pr-2.5 text-right select-none text-[12px] leading-[1.55] font-mono-num text-[var(--text-secondary)]/60 bg-[var(--bg-subtle)] border-r border-[var(--border-hairline)] overflow-hidden shrink-0"
                >
                  {lines.map((_, i) => (
                    <div key={i}>{i + 1}</div>
                  ))}
                </div>
                <textarea
                  value={draft}
                  spellCheck={false}
                  onScroll={(e) => {
                    if (gutterRef.current) {
                      gutterRef.current.scrollTop = e.currentTarget.scrollTop;
                    }
                  }}
                  onChange={(e) => {
                    setDraft(e.target.value);
                    setDirty(true);
                  }}
                  onBlur={() => {
                    if (dirty) {
                      onApplyDsl(draft);
                      setDirty(false);
                    }
                  }}
                  onKeyDown={(e) => {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                      e.preventDefault();
                      onApplyDsl(draft);
                      setDirty(false);
                    }
                  }}
                  className="flex-1 p-2.5 pl-3.5 bg-transparent font-mono-num text-[12.5px] leading-[1.55] text-[var(--text-primary)] outline-none resize-none"
                />
              </div>
            </div>
          )}

          {tab === 'params' && (
            <div className="p-3.5 space-y-3">
              <div className="flex flex-wrap gap-2.5">
                {varEntries.length === 0 ? (
                  <span className="text-[12px] text-[var(--text-secondary)]">
                    当前线路没有符号参数。选中门并在右侧打开「设为符号参数」，或在下方新增。
                  </span>
                ) : (
                  varEntries.map(([name, val]) => (
                    <div
                      key={name}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)]"
                    >
                      <span className="text-[12.5px] font-mono-num font-semibold text-[#0A84FF]">
                        {name}
                      </span>
                      <input
                        type="range"
                        min={-6.2832}
                        max={6.2832}
                        step={0.01}
                        value={val}
                        onChange={(e) => onUpdateVariable(name, Number(e.target.value))}
                        className="mac-slider w-24"
                      />
                      <input
                        type="number"
                        step={0.05}
                        value={val}
                        onChange={(e) => onUpdateVariable(name, Number(e.target.value))}
                        className="w-20 px-1.5 py-0.5 rounded bg-[var(--bg-panel)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                      />
                    </div>
                  ))
                )}
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[var(--border-hairline)]">
                <input
                  placeholder="新变量名 (如 alpha)"
                  value={newVarName}
                  onChange={(e) => setNewVarName(e.target.value)}
                  className="w-36 px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                />
                <input
                  type="number"
                  step={0.1}
                  value={newVarVal}
                  onChange={(e) => setNewVarVal(e.target.value)}
                  className="w-24 px-2 py-1 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                />
                <button
                  onClick={() => {
                    const clean = newVarName.trim();
                    if (clean) {
                      onAddVariable(clean, Number(newVarVal) || 0);
                      setNewVarName('');
                    }
                  }}
                  className="px-3 py-1 rounded-md bg-[#0A84FF]/12 text-[#0A84FF] hover:bg-[#0A84FF]/20 text-[12px] font-medium cursor-pointer"
                >
                  + 添加符号参数
                </button>
              </div>
            </div>
          )}

          {tab === 'qasm' && (
            <pre className="p-3.5 text-[12px] leading-[1.55] font-mono-num text-[var(--text-primary)] whitespace-pre-wrap">
              {qasmText || '// 暂无 OpenQASM 3 输出'}
            </pre>
          )}

          {tab === 'logs' && (
            <div className="p-3 space-y-1 font-mono-num text-[11.5px]">
              {logs.length === 0 ? (
                <div className="text-[var(--text-secondary)]">暂无日志记录</div>
              ) : (
                logs.map((lg, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-[var(--text-secondary)] shrink-0">[{lg.time}]</span>
                    <span
                      className={
                        lg.level === 'error'
                          ? 'text-red-500'
                          : lg.level === 'warn'
                          ? 'text-amber-600'
                          : 'text-[var(--text-primary)]'
                      }
                    >
                      {lg.text}
                    </span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
