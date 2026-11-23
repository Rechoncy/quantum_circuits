import React, { useState, useEffect } from 'react';
import { X, CheckCircle2, AlertCircle, Sparkles } from 'lucide-react';
import type { CircuitOpModel, CustomGateDef, CustomGateDefinitionType } from '../types';
import { validateCustomGate } from '../api';

interface CustomGateModalProps {
  open: boolean;
  editingGate: CustomGateDef | null;
  macroSeedOps?: CircuitOpModel[] | null;
  macroSeedQubits?: number;
  onClose: () => void;
  onSave: (gate: CustomGateDef) => void;
}

const DEFAULT_MAT_1Q = [
  ['1/sqrt(2)', '1/sqrt(2)'],
  ['1/sqrt(2)', '-1/sqrt(2)'],
];

const DEFAULT_MAT_2Q = [
  ['1', '0', '0', '0'],
  ['0', '1/sqrt(2)', '1j/sqrt(2)', '0'],
  ['0', '1j/sqrt(2)', '1/sqrt(2)', '0'],
  ['0', '0', '0', '1'],
];

export const CustomGateModal: React.FC<CustomGateModalProps> = ({
  open,
  editingGate,
  macroSeedOps,
  macroSeedQubits,
  onClose,
  onSave,
}) => {
  const [defType, setDefType] = useState<CustomGateDefinitionType>('matrix');
  const [name, setName] = useState('MY_GATE');
  const [displayLabel, setDisplayLabel] = useState('U_cust');
  const [numQubits, setNumQubits] = useState<number>(2);
  const [matrixExprs, setMatrixExprs] = useState<string[][]>(DEFAULT_MAT_2Q);
  const [expression, setExpression] = useState('expm(-1j * theta / 2 * ZX)');
  const [paramNamesStr, setParamNamesStr] = useState('theta');
  const [defaultParamsStr, setDefaultParamsStr] = useState('1.5708');
  const [hamiltonianStr, setHamiltonianStr] = useState('0.5*XX + 0.5*YY');
  const [hamiltonianScale, setHamiltonianScale] = useState<number>(0.7854);
  const [pulseShape, setPulseShape] = useState<'cosine' | 'gaussian' | 'rectangular'>('cosine');
  const [subOps, setSubOps] = useState<CircuitOpModel[]>([]);
  const [durationNs, setDurationNs] = useState<number>(40);
  const [infidelity, setInfidelity] = useState<number>(0.003);
  const [isNative, setIsNative] = useState<boolean>(false);
  const [description, setDescription] = useState('');

  const [validation, setValidation] = useState<{
    valid: boolean;
    error?: string;
    unitarity_error: number | null;
    is_clifford: boolean;
    matrix_preview: string[][];
  } | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (editingGate) {
      setDefType(editingGate.definition_type);
      setName(editingGate.name);
      setDisplayLabel(editingGate.display_label || editingGate.name);
      setNumQubits(editingGate.num_qubits);
      setMatrixExprs(
        editingGate.matrix_exprs || (editingGate.num_qubits === 1 ? DEFAULT_MAT_1Q : DEFAULT_MAT_2Q)
      );
      setExpression(editingGate.expression || 'expm(-1j * theta / 2 * ZX)');
      setParamNamesStr(editingGate.param_names.join(', '));
      setDefaultParamsStr(editingGate.default_params.join(', '));
      setHamiltonianStr(editingGate.hamiltonian_str || '0.5*XX + 0.5*YY');
      setHamiltonianScale(editingGate.hamiltonian_scale ?? 0.7854);
      setPulseShape(editingGate.pulse_shape || 'cosine');
      setSubOps(editingGate.sub_operations || []);
      setDurationNs(editingGate.duration_ns);
      setInfidelity(editingGate.infidelity);
      setIsNative(editingGate.is_native);
      setDescription(editingGate.description || '');
    } else if (macroSeedOps && macroSeedOps.length > 0) {
      setDefType('composite');
      setName('CUSTOM_MACRO');
      setDisplayLabel('Macro');
      setNumQubits(macroSeedQubits || 2);
      setSubOps(macroSeedOps);
      setParamNamesStr('');
      setDefaultParamsStr('');
      setIsNative(false);
      setDescription(`封装自画布的 ${macroSeedOps.length} 个门`);
    } else {
      setDefType('matrix');
      setName('MY_ENTANGLER');
      setDisplayLabel('U_exp');
      setNumQubits(2);
      setMatrixExprs(DEFAULT_MAT_2Q.map((r) => [...r]));
      setExpression('expm(-1j * theta / 2 * ZX)');
      setParamNamesStr('theta');
      setDefaultParamsStr('1.5708');
      setHamiltonianStr('0.5*XX + 0.5*YY');
      setHamiltonianScale(0.7854);
      setSubOps([
        { id: 's1', step: 0, kind: 'gate', name: 'H', qubits: [0], params: [] },
        { id: 's2', step: 1, kind: 'gate', name: 'CZ', qubits: [0, 1], params: [] },
      ]);
      setDurationNs(40);
      setInfidelity(0.003);
      setIsNative(false);
      setDescription('');
    }
    setValidation(null);
  }, [open, editingGate, macroSeedOps, macroSeedQubits]);

  const handleNumQubitsChange = (nq: number) => {
    setNumQubits(nq);
    if (nq === 1) {
      setMatrixExprs(DEFAULT_MAT_1Q.map((r) => [...r]));
      if (defType === 'hamiltonian') setHamiltonianStr('0.5*X + 0.5*Z');
      if (defType === 'expression') setExpression('expm(-1j * theta / 2 * (cos(phi)*X + sin(phi)*Y))');
    } else {
      setMatrixExprs(DEFAULT_MAT_2Q.map((r) => [...r]));
      if (defType === 'hamiltonian') setHamiltonianStr('0.5*XX + 0.5*YY');
      if (defType === 'expression') setExpression('expm(-1j * theta / 2 * ZX)');
    }
  };

  const buildCandidateGate = (): CustomGateDef => {
    const pNames =
      defType === 'expression'
        ? paramNamesStr
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
    const pDefs =
      defType === 'expression'
        ? defaultParamsStr
            .split(',')
            .map((s) => Number(s.trim()))
            .filter((n) => !Number.isNaN(n))
        : [];
    return {
      id: editingGate?.id || `cg_${Date.now()}`,
      name: name.trim().toUpperCase().replace(/\s+/g, '_') || 'CUSTOM_GATE',
      display_label: displayLabel.trim() || name.trim(),
      num_qubits: numQubits,
      definition_type: defType,
      matrix_exprs: defType === 'matrix' ? matrixExprs : null,
      expression: defType === 'expression' ? expression : null,
      param_names: pNames,
      default_params: pDefs,
      hamiltonian_str: defType === 'hamiltonian' ? hamiltonianStr : null,
      hamiltonian_scale: hamiltonianScale,
      pulse_shape: pulseShape,
      sub_operations: defType === 'composite' ? subOps : [],
      duration_ns: durationNs,
      infidelity,
      color: '#8B5CF6',
      is_native: isNative,
      description,
    };
  };

  const handleValidate = async () => {
    setChecking(true);
    try {
      const gate = buildCandidateGate();
      const res = await validateCustomGate(gate);
      setValidation(res);
    } catch (err) {
      setValidation({
        valid: false,
        error: String(err),
        unitarity_error: null,
        is_clifford: false,
        matrix_preview: [],
      });
    } finally {
      setChecking(false);
    }
  };

  const handleSave = async () => {
    const gate = buildCandidateGate();
    const res = await validateCustomGate(gate);
    setValidation(res);
    if (res.valid) {
      onSave(gate);
      onClose();
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/35 backdrop-blur-[3px] studio-modal-backdrop flex items-center justify-center p-4">
      <div className="w-full max-w-[680px] max-h-[88vh] studio-modal-surface flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between">
          <div>
            <h2 className="text-[15px] font-semibold text-[var(--text-primary)]">
              {editingGate ? `编辑自定义门 · ${editingGate.display_label}` : '新建自定义量子门'}
            </h2>
            <p className="text-[11.5px] text-[var(--text-secondary)]">
              支持 4 种定义方式，校验幺正性后可直接拖放至画布，或作为实验原生门接入优化器
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[var(--text-secondary)] hover:bg-black/5 dark:hover:bg-white/10"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* 4-Mode Segmented Control */}
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg grid grid-cols-4 gap-1">
            {[
              { id: 'matrix', label: '① 酉矩阵' },
              { id: 'expression', label: '② 参数化表达式' },
              { id: 'hamiltonian', label: '③ 哈密顿量 + 时长' },
              { id: 'composite', label: '④ 复合门 (宏)' },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setDefType(item.id as CustomGateDefinitionType);
                  setValidation(null);
                }}
                className={`py-1.5 rounded-[7px] text-[12px] font-medium transition-all cursor-pointer ${
                  defType === item.id
                    ? 'bg-[var(--bg-panel)] text-[#0A84FF] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* Basic Metadata Row */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                指令标识 (DSL 名称)
              </label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12.5px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                画布显示名
              </label>
              <input
                value={displayLabel}
                onChange={(e) => setDisplayLabel(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12.5px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                作用比特数
              </label>
              <div className="grid grid-cols-2 gap-1.5">
                {[1, 2].map((nq) => (
                  <button
                    key={nq}
                    onClick={() => handleNumQubitsChange(nq)}
                    className={`py-1.5 rounded-lg border text-[12px] font-medium cursor-pointer ${
                      numQubits === nq
                        ? 'bg-[#0A84FF]/12 border-[#0A84FF] text-[#0A84FF]'
                        : 'bg-[var(--bg-subtle)] border-[var(--border-hairline)] text-[var(--text-secondary)]'
                    }`}
                  >
                    {nq} Qubit ({1 << nq}×{1 << nq})
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Mode-Specific Editor */}
          {defType === 'matrix' && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label className="text-[11.5px] font-medium text-[var(--text-secondary)]">
                  复数酉矩阵条目 (支持 1/sqrt(2), 1j, exp(i*pi/4))
                </label>
                <div className="flex items-center gap-1">
                  {(numQubits === 1
                    ? [
                        {
                          lbl: 'H',
                          m: [
                            ['1/sqrt(2)', '1/sqrt(2)'],
                            ['1/sqrt(2)', '-1/sqrt(2)'],
                          ],
                        },
                        {
                          lbl: 'T',
                          m: [
                            ['1', '0'],
                            ['0', 'exp(i*pi/4)'],
                          ],
                        },
                        {
                          lbl: '√Y',
                          m: [
                            ['0.5+0.5j', '-0.5-0.5j'],
                            ['0.5+0.5j', '0.5+0.5j'],
                          ],
                        },
                      ]
                    : [
                        { lbl: '√iSWAP', m: DEFAULT_MAT_2Q },
                        {
                          lbl: 'fSim(π/4,π/6)',
                          m: [
                            ['1', '0', '0', '0'],
                            ['0', 'cos(pi/4)', '-1j*sin(pi/4)', '0'],
                            ['0', '-1j*sin(pi/4)', 'cos(pi/4)', '0'],
                            ['0', '0', '0', 'exp(-1j*pi/6)'],
                          ],
                        },
                        {
                          lbl: 'B-Gate',
                          m: [
                            ['cos(pi/8)', '0', '0', '1j*sin(pi/8)'],
                            ['0', 'cos(3*pi/8)', '1j*sin(3*pi/8)', '0'],
                            ['0', '1j*sin(3*pi/8)', 'cos(3*pi/8)', '0'],
                            ['1j*sin(pi/8)', '0', '0', 'cos(pi/8)'],
                          ],
                        },
                      ]
                  ).map((pre) => (
                    <button
                      key={pre.lbl}
                      type="button"
                      onClick={() => setMatrixExprs(pre.m.map((r) => [...r]))}
                      className="px-2 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] hover:border-[#0A84FF] hover:text-[#0A84FF] text-[10.5px] font-mono-num cursor-pointer"
                    >
                      {pre.lbl}
                    </button>
                  ))}
                </div>
              </div>
              <div
                className="grid gap-1.5"
                style={{ gridTemplateColumns: `repeat(${1 << numQubits}, minmax(0, 1fr))` }}
              >
                {matrixExprs.map((row, r) =>
                  row.map((cell, c) => (
                    <input
                      key={`${r}_${c}`}
                      value={cell}
                      onChange={(e) => {
                        const copy = matrixExprs.map((rw) => [...rw]);
                        copy[r][c] = e.target.value;
                        setMatrixExprs(copy);
                      }}
                      className="px-2 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num text-center outline-none focus:border-[#0A84FF]"
                    />
                  ))
                )}
              </div>
            </div>
          )}

          {defType === 'expression' && (
            <div className="space-y-3">
              <div>
                <label className="block text-[11.5px] font-medium text-[var(--text-secondary)] mb-1">
                  参数化酉矩阵表达式（内置变量: I, X, Y, Z, XX..ZZ, expm, cos, sin, pi, 1j）
                </label>
                <input
                  value={expression}
                  onChange={(e) => setExpression(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12.5px] font-mono-num outline-none focus:border-[#0A84FF]"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                    参数名列表 (逗号分隔)
                  </label>
                  <input
                    value={paramNamesStr}
                    onChange={(e) => setParamNamesStr(e.target.value)}
                    placeholder="theta, phi"
                    className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                    默认参数值 (逗号分隔)
                  </label>
                  <input
                    value={defaultParamsStr}
                    onChange={(e) => setDefaultParamsStr(e.target.value)}
                    placeholder="1.5708, 0"
                    className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                  />
                </div>
              </div>
            </div>
          )}

          {defType === 'hamiltonian' && (
            <div className="space-y-3">
              <div>
                <label className="block text-[11.5px] font-medium text-[var(--text-secondary)] mb-1">
                  生成元哈密顿量 H（泡利字符串线性组合，如 0.5*XX + 0.5*YY - 0.2*ZZ）
                </label>
                <input
                  value={hamiltonianStr}
                  onChange={(e) => setHamiltonianStr(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12.5px] font-mono-num outline-none focus:border-[#0A84FF]"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                    演化积分相位 θ = ∫Ω(t)dt (rad)
                  </label>
                  <input
                    type="number"
                    step={0.05}
                    value={hamiltonianScale}
                    onChange={(e) => setHamiltonianScale(Number(e.target.value))}
                    className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                    QuTiP 脉冲包络
                  </label>
                  <select
                    value={pulseShape}
                    onChange={(e) => setPulseShape(e.target.value as 'cosine' | 'gaussian' | 'rectangular')}
                    className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] outline-none"
                  >
                    <option value="cosine">平滑 Hann 余弦包络</option>
                    <option value="gaussian">高斯包络 (Gaussian)</option>
                    <option value="rectangular">矩形恒定脉冲</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {defType === 'composite' && (
            <div className="space-y-2">
              <div className="text-[11.5px] text-[var(--text-secondary)]">
                内部包含的子线路操作（可在画布按住 Shift 多选多个门后点击「封装为自定义门」自动导入）：
              </div>
              <div className="p-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono-num text-[12px] space-y-1 max-h-32 overflow-y-auto">
                {subOps.length === 0 ? (
                  <div className="text-[var(--text-secondary)]">暂无子操作</div>
                ) : (
                  subOps.map((so, i) => (
                    <div key={i}>
                      Step {so.step + 1}: {so.name}
                      {so.params.length > 0 ? `(${so.params.join(', ')})` : ''} @ q[{so.qubits.join(', ')}]
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Hardware Properties & Promote to Native Gate Switch */}
          <div className="pt-3 border-t border-[var(--border-hairline)] grid grid-cols-3 gap-3 items-end">
            <div>
              <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                物理门时长 (ns)
              </label>
              <input
                type="number"
                value={durationNs}
                onChange={(e) => setDurationNs(Number(e.target.value))}
                className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[var(--text-secondary)] mb-1">
                门不保真度 (Error)
              </label>
              <input
                type="number"
                step={0.0005}
                value={infidelity}
                onChange={(e) => setInfidelity(Number(e.target.value))}
                className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-input)] text-[12px] font-mono-num outline-none focus:border-[#0A84FF]"
              />
            </div>
            <label className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-[#0A84FF]/8 border border-[#0A84FF]/25 cursor-pointer">
              <span className="flex items-center gap-1.5 text-[12px] font-medium text-[#0A84FF]">
                <Sparkles className="w-3.5 h-3.5" />
                作为原生门
              </span>
              <input
                type="checkbox"
                checked={isNative}
                onChange={(e) => setIsNative(e.target.checked)}
                className="accent-[#0A84FF] w-4 h-4 cursor-pointer"
              />
            </label>
          </div>

          {/* Live Validation Result Card */}
          {validation && (
            <div
              className={`p-3 rounded-xl border text-[12px] space-y-2 ${
                validation.valid
                  ? 'bg-emerald-50/70 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
                  : 'bg-red-50/80 dark:bg-red-950/30 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 font-semibold">
                  {validation.valid ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-600" />
                  )}
                  {validation.valid
                    ? `幺正性校验通过 (‖U†U − I‖ = ${validation.unitarity_error?.toExponential(1)})`
                    : `校验未通过: ${validation.error || `非酉矩阵 (误差 ${validation.unitarity_error?.toFixed(4)})`}`}
                </span>
                {validation.valid && (
                  <span className="text-[11px] font-mono-num px-2 py-0.5 rounded bg-white/70 dark:bg-black/25">
                    Clifford: {validation.is_clifford ? '是 (支持 Stim)' : '否 (QuTiP/Kraus)'}
                  </span>
                )}
              </div>
              {validation.matrix_preview.length > 0 && (
                <div
                  className="grid gap-1 font-mono-num text-[11px] pt-1"
                  style={{
                    gridTemplateColumns: `repeat(${validation.matrix_preview.length}, minmax(0, 1fr))`,
                  }}
                >
                  {validation.matrix_preview.flatMap((row, r) =>
                    row.map((val, c) => (
                      <div
                        key={`p_${r}_${c}`}
                        className="px-1.5 py-0.5 rounded bg-white/75 dark:bg-black/25 text-center"
                      >
                        {val}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] flex items-center justify-between">
          <button
            onClick={handleValidate}
            disabled={checking}
            className="px-3.5 py-1.5 rounded-lg border border-[var(--border-input)] bg-[var(--bg-panel)] text-[12.5px] font-medium hover:border-[#0A84FF] hover:text-[#0A84FF] cursor-pointer"
          >
            {checking ? '校验中...' : '校验幺正性 & 预览矩阵'}
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-lg text-[12.5px] text-[var(--text-secondary)] hover:bg-black/5 cursor-pointer"
            >
              取消
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 rounded-lg bg-[#0A84FF] hover:bg-[#0070E0] text-white text-[12.5px] font-semibold cursor-pointer"
            >
              保存至门库
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
