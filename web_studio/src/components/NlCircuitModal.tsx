import React, { useEffect, useRef, useState } from 'react';
import {
  Sparkles,
  X,
  Play,
  Send,
  Loader2,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Settings2,
  ArrowRight,
  Code2,
  RotateCcw,
  Copy,
  Check,
  Trash2,
  Layers,
  Cpu,
} from 'lucide-react';
import type { NlCircuitResponse, ProjectModel } from '../types';
import { generateCircuitFromNl } from '../api';
import { CircuitCanvas } from './CircuitCanvas';

interface NlCircuitModalProps {
  open: boolean;
  project: ProjectModel;
  onClose: () => void;
  onApplyResult: (res: NlCircuitResponse, runImmediately: boolean) => void;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  version?: number;
  response?: NlCircuitResponse;
  error?: string;
}

const STARTER_PROMPTS: { label: string; prompt: string }[] = [
  {
    label: '中性原子 4Q GHZ',
    prompt: '用中性原子里德堡阵列生成 4 比特 GHZ 态并加入 1% 退极化噪声后测量',
  },
  {
    label: '50Q 对数深度 GHZ',
    prompt: '生成 50 比特对数深度二叉树 GHZ 纠缠态线路并测量',
  },
  {
    label: '16Q 变分 VQE (态矢量)',
    prompt: '创建 16 比特 2 层硬件高效 VQE 变分线路，方便扫描参数',
  },
  {
    label: '17Q 旋转表面码 QEC',
    prompt: '生成 17 比特旋转表面码稳定子纠错线路并测量',
  },
  {
    label: '8 比特 W 态',
    prompt: '制备 8 比特精确 W 态（单激发对称纠缠态）并测量',
  },
  {
    label: '6 比特 QFT',
    prompt: '生成 6 比特量子傅里叶变换 QFT 线路并测量所有比特',
  },
  {
    label: '5Q Grover 搜索 |10110⟩',
    prompt: '生成 5 比特 Grover 量子搜索线路，目标态 |10110⟩',
  },
  {
    label: '25Q 二维网格簇态',
    prompt: '生成 25 比特二维网格 Cluster 簇态线路',
  },
];

const FOLLOWUP_CHIPS: { label: string; prompt: string }[] = [
  { label: '噪声改 3%', prompt: '把退极化噪声改成 3%' },
  { label: '+1 比特做 CZ', prompt: '再增加 1 个比特并和上一个比特做 CZ 纠缠' },
  { label: '扩容到 20Q', prompt: '扩容到 20 个比特' },
  { label: 'CZ → iSWAP', prompt: '把所有 CZ 换成 iSWAP' },
  { label: '删掉噪声', prompt: '删掉所有噪声' },
  { label: '切超导 CZ', prompt: '切换到超导 CZ 硬件' },
];

const LLM_CFG_KEY = 'quantum_studio_llm_cfg';

function actionModeLabel(mode?: 'replace' | 'refine' | 'append'): { text: string; cls: string } {
  if (mode === 'refine') {
    return {
      text: '对话精调',
      cls: 'bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 border-emerald-500/20',
    };
  }
  if (mode === 'append') {
    return {
      text: '末尾追加',
      cls: 'bg-amber-500/12 text-amber-700 dark:text-amber-300 border-amber-500/20',
    };
  }
  return {
    text: '全新构建',
    cls: 'bg-[var(--accent-soft)] text-[var(--accent)] border-[var(--accent)]/20',
  };
}

export const NlCircuitModal: React.FC<NlCircuitModalProps> = ({
  open,
  project,
  onClose,
  onApplyResult,
}) => {
  const [prompt, setPrompt] = useState('');
  const [genMode, setGenMode] = useState<'auto' | 'refine' | 'replace'>('auto');
  const [engine, setEngine] = useState<'local' | 'llm'>('local');
  const [llmBaseUrl, setLlmBaseUrl] = useState('https://api.openai.com/v1');
  const [llmModel, setLlmModel] = useState('gpt-4o-mini');
  const [llmApiKey, setLlmApiKey] = useState('');

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeVersion, setActiveVersion] = useState<number | null>(null);
  const [activeResult, setActiveResult] = useState<NlCircuitResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const [codeTab, setCodeTab] = useState<'dsl' | 'qasm'>('dsl');
  const [showCode, setShowCode] = useState(true);
  const [copiedCode, setCopiedCode] = useState(false);

  const threadEndRef = useRef<HTMLDivElement>(null);
  const versionCounterRef = useRef(0);

  // Restore saved external LLM settings from localStorage
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(LLM_CFG_KEY);
      if (raw) {
        const cfg = JSON.parse(raw);
        if (cfg.baseUrl) setLlmBaseUrl(cfg.baseUrl);
        if (cfg.model) setLlmModel(cfg.model);
        if (cfg.apiKey) setLlmApiKey(cfg.apiKey);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

  if (!open) return null;

  // Build the working project that the next turn refines from (either active generated version or current workspace)
  const workingProject: ProjectModel = activeResult
    ? {
        ...project,
        name: activeResult.title || project.name,
        qubits: activeResult.qubits,
        operations: activeResult.operations,
        variables: activeResult.variables,
        active_hardware_id: activeResult.recommended_hardware_id || project.active_hardware_id,
      }
    : project;

  const handleClearThread = () => {
    setMessages([]);
    setActiveResult(null);
    setActiveVersion(null);
    versionCounterRef.current = 0;
  };

  const handleSend = async (customText?: string, modeOverride?: 'auto' | 'refine' | 'replace') => {
    const text = (customText ?? prompt).trim();
    if (!text || loading) return;

    if (engine === 'llm') {
      try {
        window.localStorage.setItem(
          LLM_CFG_KEY,
          JSON.stringify({ baseUrl: llmBaseUrl, model: llmModel, apiKey: llmApiKey })
        );
      } catch {
        // ignore
      }
    }

    const userMsg: ChatMessage = {
      id: `u_${Date.now()}`,
      role: 'user',
      content: text,
    };
    const nextHistory = [
      ...messages
        .filter((m) => !m.error)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content })),
    ];

    setMessages((prev) => [...prev, userMsg]);
    if (customText === undefined) setPrompt('');
    setLoading(true);

    try {
      const res = await generateCircuitFromNl({
        prompt: text,
        project: workingProject,
        mode: modeOverride ?? genMode,
        engine,
        history: nextHistory,
        llm_base_url: engine === 'llm' ? llmBaseUrl : undefined,
        llm_api_key: engine === 'llm' ? llmApiKey : undefined,
        llm_model: engine === 'llm' ? llmModel : undefined,
      });
      versionCounterRef.current += 1;
      const ver = versionCounterRef.current;
      const assistantMsg: ChatMessage = {
        id: `a_${Date.now()}`,
        role: 'assistant',
        content: res.explanation,
        version: ver,
        response: res,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setActiveResult(res);
      setActiveVersion(ver);
    } catch (err) {
      const errTxt = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `e_${Date.now()}`,
          role: 'assistant',
          content: '',
          error: errTxt,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = (msg: ChatMessage) => {
    if (!msg.response || msg.version === undefined) return;
    setActiveResult(msg.response);
    setActiveVersion(msg.version);
  };

  const handleCopyCode = () => {
    if (!activeResult) return;
    const text = codeTab === 'dsl' ? activeResult.dsl : activeResult.openqasm3;
    navigator.clipboard.writeText(text).catch(() => {});
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 1200);
  };

  const recHwName =
    project.hardware_profiles.find((h) => h.id === workingProject.active_hardware_id)?.name ??
    '默认硬件';
  const gateCount = workingProject.operations.filter((o) => o.kind === 'gate').length;
  const noiseCount = workingProject.operations.filter((o) => o.kind === 'noise').length;
  const varKeys = Object.keys(workingProject.variables);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/45 backdrop-blur-[4px] studio-modal-backdrop flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-5xl h-[86vh] max-h-[760px] studio-modal-surface flex flex-col overflow-hidden"
      >
        {/* Top Header */}
        <div className="h-13 px-5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded-lg bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center shrink-0">
              <Sparkles className="w-4 h-4" />
            </div>
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-[14px] font-semibold tracking-tight text-[var(--text-primary)] truncate">
                AI 量子线路对话助手
              </span>
              <span className="px-1.5 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[10px] font-mono-num text-[var(--text-secondary)] shrink-0">
                Ctrl+K
              </span>
              {activeVersion !== null && (
                <span className="px-1.5 py-0.5 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] text-[10.5px] font-mono-num font-semibold shrink-0">
                  预览 v{activeVersion}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Engine Switcher */}
            <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex text-[11px]">
              <button
                type="button"
                onClick={() => setEngine('local')}
                className={`px-2.5 py-1 rounded-md cursor-pointer transition-colors ${
                  engine === 'local'
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                内置语义引擎 (免密钥)
              </button>
              <button
                type="button"
                onClick={() => setEngine('llm')}
                className={`px-2.5 py-1 rounded-md cursor-pointer transition-colors flex items-center gap-1 ${
                  engine === 'llm'
                    ? 'bg-[var(--bg-panel)] text-[var(--accent)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                <Settings2 className="w-3 h-3" />
                外部 LLM API
              </button>
            </div>

            {messages.length > 0 && (
              <button
                type="button"
                onClick={handleClearThread}
                title="清空对话并重置预览"
                className="px-2 py-1 rounded-lg text-[11.5px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 flex items-center gap-1 cursor-pointer transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">清空对话</span>
              </button>
            )}

            <button
              type="button"
              onClick={onClose}
              aria-label="关闭窗口"
              className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Optional External LLM Config Bar */}
        {engine === 'llm' && (
          <div className="px-5 py-2.5 bg-[var(--bg-subtle)]/70 border-b border-[var(--border-hairline)] grid grid-cols-3 gap-3 shrink-0">
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                OpenAI 兼容 Base URL (支持 DeepSeek / Ollama)
              </label>
              <input
                value={llmBaseUrl}
                onChange={(e) => setLlmBaseUrl(e.target.value)}
                placeholder="https://api.openai.com/v1"
                className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none focus:border-[var(--accent)]"
              />
            </div>
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                模型名称 (Model)
              </label>
              <input
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                placeholder="gpt-4o-mini / deepseek-chat"
                className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none focus:border-[var(--accent)]"
              />
            </div>
            <div>
              <label className="block text-[10.5px] text-[var(--text-secondary)] mb-0.5">
                API Key (本地 Ollama 可留空)
              </label>
              <input
                type="password"
                value={llmApiKey}
                onChange={(e) => setLlmApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full px-2.5 py-1 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-input)] text-[11.5px] font-mono-num outline-none focus:border-[var(--accent)]"
              />
            </div>
          </div>
        )}

        {/* Two-Pane Body: Left Chat Thread (46%) | Right Live Circuit Workbench (54%) */}
        <div className="flex-1 min-h-0 grid grid-cols-12 divide-x divide-[var(--border-hairline)] overflow-hidden">
          {/* LEFT PANE: Multi-Turn Chat Messaging */}
          <div className="col-span-12 md:col-span-5 flex flex-col min-h-0 bg-[var(--bg-sidebar)]">
            {/* Scrollable Message Thread */}
            <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
              {/* Context Banner: Starting Workspace Circuit */}
              <div className="p-3 rounded-xl bg-[var(--bg-panel)] border border-[var(--border-hairline)] shadow-2xs space-y-1.5">
                <div className="flex items-center justify-between text-[11.5px]">
                  <span className="font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-[var(--accent)]" />
                    当前画布基准
                  </span>
                  <span className="font-mono-num text-[10.5px] text-[var(--text-secondary)]">
                    {project.qubits.length}Q · {project.operations.filter((o) => o.kind === 'gate').length} Gates
                  </span>
                </div>
                <p className="text-[11.5px] text-[var(--text-secondary)] leading-relaxed">
                  直接描述新算法（如 GHZ / QFT / Grover / VQE），或对现有线路进行多轮精调（如“把噪声改 3%”、“把 CZ 换成 iSWAP”、“再加 1 个比特”）。
                </p>
              </div>

              {/* Starter Prompt Cards when empty */}
              {messages.length === 0 && (
                <div className="space-y-2 pt-1">
                  <div className="text-[10.5px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)] px-0.5">
                    快速开始模板
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {STARTER_PROMPTS.map((ex) => (
                      <button
                        key={ex.label}
                        type="button"
                        onClick={() => handleSend(ex.prompt, 'replace')}
                        className="px-2.5 py-1 rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-panel)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[11.5px] text-[var(--text-primary)] shadow-2xs cursor-pointer transition-colors"
                      >
                        {ex.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Chat Bubbles */}
              {messages.map((msg) => {
                if (msg.role === 'user') {
                  return (
                    <div key={msg.id} className="flex justify-end">
                      <div className="max-w-[88%] px-3.5 py-2 rounded-2xl rounded-br-md bg-[var(--accent)] text-white text-[12.5px] leading-relaxed shadow-xs">
                        {msg.content}
                      </div>
                    </div>
                  );
                }

                if (msg.error) {
                  return (
                    <div key={msg.id} className="flex justify-start">
                      <div className="max-w-[94%] p-3 rounded-2xl rounded-bl-md bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/70 text-[12px] text-red-700 dark:text-red-300 flex items-start gap-2">
                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                        <span>{msg.error}</span>
                      </div>
                    </div>
                  );
                }

                const res = msg.response;
                const badge = actionModeLabel(res?.action_mode);
                const isCurrentVer = msg.version !== undefined && msg.version === activeVersion;
                const diff = res?.diff_summary;

                return (
                  <div key={msg.id} className="flex justify-start">
                    <div
                      className={`w-full p-3 rounded-2xl rounded-bl-md border transition-colors space-y-2 ${
                        isCurrentVer
                          ? 'bg-[var(--bg-panel)] border-[var(--accent)]/45 shadow-xs'
                          : 'bg-[var(--bg-panel)]/85 border-[var(--border-hairline)]'
                      }`}
                    >
                      {/* Title & Mode Badge */}
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span
                            className={`px-1.5 py-0.5 rounded-md border text-[10px] font-semibold shrink-0 ${badge.cls}`}
                          >
                            {badge.text}
                          </span>
                          <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                            {res?.title}
                          </span>
                        </div>
                        {msg.version !== undefined && (
                          <span className="text-[10.5px] font-mono-num text-[var(--text-tertiary)] shrink-0">
                            v{msg.version}
                          </span>
                        )}
                      </div>

                      {/* Explanation */}
                      <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">
                        {msg.content}
                      </p>

                      {/* Diff Chips */}
                      {res?.changes && res.changes.length > 0 && (
                        <div className="flex flex-wrap gap-1 pt-0.5">
                          {res.changes.map((c, idx) => (
                            <span
                              key={idx}
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-300 text-[10.5px] font-medium"
                            >
                              <CheckCircle2 className="w-3 h-3 shrink-0" />
                              {c}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Unrecognized Clauses Warning */}
                      {res?.unrecognized_clauses && res.unrecognized_clauses.length > 0 && (
                        <div className="px-2.5 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/25 text-amber-800 dark:text-amber-300 text-[11px] flex items-start gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                          <span>
                            未识别子句：{res.unrecognized_clauses.join('；')}
                          </span>
                        </div>
                      )}

                      {/* Footer: Compact Delta + Version Rollback */}
                      <div className="pt-1.5 border-t border-[var(--border-hairline)] flex items-center justify-between gap-2 text-[10.5px]">
                        <div className="flex items-center gap-2 font-mono-num text-[var(--text-tertiary)]">
                          {diff ? (
                            <>
                              <span>
                                Q:{diff.qubits_before}→{diff.qubits_after}
                              </span>
                              <span>
                                Gates:{diff.gates_before}→{diff.gates_after}
                              </span>
                              {((diff.noise_before ?? 0) > 0 || (diff.noise_after ?? 0) > 0) && (
                                <span>
                                  Noise:{diff.noise_before ?? 0}→{diff.noise_after ?? 0}
                                </span>
                              )}
                            </>
                          ) : (
                            <span>{res?.qubits.length}Q</span>
                          )}
                        </div>

                        {isCurrentVer ? (
                          <span className="inline-flex items-center gap-1 text-[var(--accent)] font-semibold">
                            <Check className="w-3 h-3" />
                            当前预览
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => handleRollback(msg)}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[var(--text-secondary)] hover:text-[var(--accent)] hover:bg-[var(--accent-soft)] font-medium cursor-pointer transition-colors"
                          >
                            <RotateCcw className="w-3 h-3" />
                            回退到此版本
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="flex justify-start">
                  <div className="px-3.5 py-2.5 rounded-2xl rounded-bl-md bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[12px] text-[var(--text-secondary)] flex items-center gap-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent)]" />
                    <span>正在解析指令并编译量子线路...</span>
                  </div>
                </div>
              )}

              <div ref={threadEndRef} />
            </div>

            {/* Bottom Composer Area */}
            <div className="p-3 bg-[var(--bg-panel)] border-t border-[var(--border-hairline)] space-y-2 shrink-0">
              {/* Contextual Follow-Up Chips (after first turn or when workspace already has gates) */}
              {(messages.length > 0 || project.operations.length > 0) && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
                  <span className="text-[10px] text-[var(--text-tertiary)] shrink-0">快捷精调:</span>
                  {FOLLOWUP_CHIPS.map((chip) => (
                    <button
                      key={chip.label}
                      type="button"
                      disabled={loading}
                      onClick={() => handleSend(chip.prompt, 'refine')}
                      className="px-2 py-0.5 rounded-md border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--accent)] hover:text-[var(--accent)] text-[10.5px] text-[var(--text-secondary)] whitespace-nowrap cursor-pointer transition-colors shrink-0 disabled:opacity-50"
                    >
                      {chip.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Input Box + Mode Switch + Send Button */}
              <div className="rounded-xl bg-[var(--bg-subtle)] border border-[var(--border-input)] focus-within:border-[var(--accent)] focus-within:ring-2 focus-within:ring-[var(--accent)]/15 p-2 transition-all">
                <textarea
                  rows={2}
                  value={prompt}
                  autoFocus
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (!e.shiftKey || e.ctrlKey || e.metaKey)) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={
                    messages.length === 0
                      ? '输入量子线路需求（Enter 发送，Shift+Enter 换行）...'
                      : '继续对话精调，如「把退极化噪声改成 3%，再增加 1 个比特并做 CZ 纠缠」...'
                  }
                  className="w-full bg-transparent text-[12.5px] leading-relaxed outline-none resize-none placeholder:text-[var(--text-tertiary)]"
                />

                <div className="flex items-center justify-between gap-2 pt-1.5 mt-1 border-t border-[var(--border-hairline)]">
                  <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex text-[10.5px]">
                    {(
                      [
                        ['auto', '智能'],
                        ['refine', '精调'],
                        ['replace', '新建'],
                      ] as const
                    ).map(([m, lbl]) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setGenMode(m)}
                        className={`px-2 py-0.5 rounded-md cursor-pointer transition-colors ${
                          genMode === m
                            ? 'bg-[var(--bg-panel)] font-semibold text-[var(--text-primary)] shadow-2xs'
                            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                        }`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>

                  <button
                    type="button"
                    onClick={() => handleSend()}
                    disabled={loading || !prompt.trim()}
                    className="h-7 px-3 rounded-lg studio-btn-primary text-[11.5px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-45"
                  >
                    {loading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Send className="w-3.5 h-3.5" />
                    )}
                    <span>发送</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANE: Always-Live Working Circuit Preview + Code Tabs */}
          <div className="col-span-12 md:col-span-7 flex flex-col min-h-0 bg-[var(--bg-app)]">
            {/* Compact KPI Telemetry Strip */}
            <div className="px-4 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2 shrink-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[12.5px] font-semibold text-[var(--text-primary)] truncate">
                  {activeResult ? activeResult.title : `当前画布 · ${project.name}`}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono-num shrink-0">
                <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-primary)]">
                  {workingProject.qubits.length}Q
                </span>
                <span className="px-2 py-0.5 rounded-md bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-primary)]">
                  {gateCount} Gates
                </span>
                {noiseCount > 0 && (
                  <span className="px-2 py-0.5 rounded-md bg-amber-500/12 text-amber-700 dark:text-amber-300">
                    {noiseCount} Noise
                  </span>
                )}
                <span className="px-2 py-0.5 rounded-md bg-[var(--accent-soft)] text-[var(--accent)] font-medium inline-flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {recHwName}
                </span>
                {activeResult?.recommended_backend && (
                  <span className="px-2 py-0.5 rounded-md bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 font-semibold">
                    推荐引擎: {activeResult.recommended_backend.toUpperCase()}
                  </span>
                )}
                {varKeys.length > 0 && (
                  <span className="px-2 py-0.5 rounded-md bg-purple-500/12 text-purple-700 dark:text-purple-300">
                    {varKeys.join(', ')}
                  </span>
                )}
              </div>
            </div>

            {/* Live Circuit Canvas */}
            <div className="flex-1 min-h-[220px] flex flex-col overflow-hidden">
              <CircuitCanvas
                project={workingProject}
                readOnly
                compact
                selection={{ type: 'none' }}
                activeToolGate={null}
                onSelect={() => {}}
                onPickGateTool={() => {}}
                onPlaceGate={() => {}}
                onDeleteOp={() => {}}
                onAddQubit={() => {}}
                titleBadge={
                  activeVersion !== null
                    ? `实时线路预览 · v${activeVersion} · ${workingProject.name}`
                    : `当前画布基准 · ${project.name}`
                }
              />
            </div>

            {/* Collapsible DSL / OpenQASM 3 Code Preview */}
            <div className="border-t border-[var(--border-hairline)] bg-[var(--bg-panel)] shrink-0">
              <div className="h-9 px-4 flex items-center justify-between border-b border-[var(--border-hairline)]">
                <button
                  type="button"
                  onClick={() => setShowCode((v) => !v)}
                  className="text-[11.5px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] flex items-center gap-1.5 cursor-pointer"
                >
                  <Code2 className="w-3.5 h-3.5 text-[var(--accent)]" />
                  <span>{showCode ? '生成代码预览' : '展开生成代码 (DSL / OpenQASM 3)'}</span>
                </button>

                {showCode && (
                  <div className="flex items-center gap-2">
                    <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-md inline-flex text-[10.5px]">
                      <button
                        type="button"
                        onClick={() => setCodeTab('dsl')}
                        className={`px-2 py-0.5 rounded cursor-pointer ${
                          codeTab === 'dsl'
                            ? 'bg-[var(--bg-panel)] font-semibold text-[var(--text-primary)] shadow-2xs'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        DSL
                      </button>
                      <button
                        type="button"
                        onClick={() => setCodeTab('qasm')}
                        className={`px-2 py-0.5 rounded cursor-pointer ${
                          codeTab === 'qasm'
                            ? 'bg-[var(--bg-panel)] font-semibold text-[var(--text-primary)] shadow-2xs'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        OpenQASM 3
                      </button>
                    </div>
                    {activeResult && (
                      <button
                        type="button"
                        onClick={handleCopyCode}
                        title="复制代码"
                        className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
                      >
                        {copiedCode ? (
                          <Check className="w-3.5 h-3.5 text-emerald-500" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    )}
                  </div>
                )}
              </div>

              {showCode && (
                <pre className="h-[132px] p-3 bg-[var(--bg-subtle)]/50 text-[11.5px] font-mono-num overflow-auto leading-relaxed text-[var(--text-primary)]">
                  {activeResult
                    ? codeTab === 'dsl'
                      ? activeResult.dsl
                      : activeResult.openqasm3
                    : '// 发送自然语言指令后，此处实时显示生成的 DSL 与 OpenQASM 3 代码'}
                </pre>
              )}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="h-14 px-5 border-t border-[var(--border-hairline)] bg-[var(--bg-panel)] flex items-center justify-between gap-3 shrink-0">
          <span className="text-[11px] text-[var(--text-secondary)] truncate">
            支持多轮对话精调（改噪声/参数、替换/删除门、增减比特、切硬件）与版本回退
          </span>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-lg border border-[var(--border-input)] hover:bg-black/5 dark:hover:bg-white/10 text-[12px] font-medium cursor-pointer transition-colors"
            >
              取消
            </button>
            <button
              type="button"
              disabled={!activeResult}
              onClick={() => {
                if (activeResult) {
                  onApplyResult(activeResult, false);
                  onClose();
                }
              }}
              className="px-3.5 py-1.5 rounded-lg border border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--accent-soft)] text-[12px] font-semibold flex items-center gap-1 cursor-pointer transition-colors disabled:opacity-40"
            >
              <span>应用到画布</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              disabled={!activeResult}
              onClick={() => {
                if (activeResult) {
                  onApplyResult(activeResult, true);
                  onClose();
                }
              }}
              className="px-4 py-1.5 rounded-lg studio-btn-primary text-[12px] font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>应用并立即模拟</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
