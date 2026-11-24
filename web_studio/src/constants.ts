export interface BuiltinGateInfo {
  name: string;
  label: string;
  category: '1q_clifford' | '1q_rot' | '2q' | '3q' | 'noise' | 'measure';
  numQubits: number;
  paramNames: string[];
  defaultParams: number[];
  tint: 'blue' | 'teal' | 'indigo' | 'amber' | 'slate';
  description: string;
}

export const BUILTIN_GATES: BuiltinGateInfo[] = [
  { name: 'H', label: 'H', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'blue', description: 'Hadamard 门' },
  { name: 'X', label: 'X', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'teal', description: 'Pauli-X 翻转门' },
  { name: 'Y', label: 'Y', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'teal', description: 'Pauli-Y 门' },
  { name: 'Z', label: 'Z', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'teal', description: 'Pauli-Z 相位门' },
  { name: 'SX', label: '√X', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'blue', description: '超导原生 π/2 脉冲' },
  { name: 'S', label: 'S', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'teal', description: 'π/2 相位门' },
  { name: 'T', label: 'T', category: '1q_clifford', numQubits: 1, paramNames: [], defaultParams: [], tint: 'teal', description: 'π/4 非 Clifford 门' },
  { name: 'RXY', label: 'RXY', category: '1q_rot', numQubits: 1, paramNames: ['θ', 'φ'], defaultParams: [1.5708, 0], tint: 'blue', description: '中性原子/离子阱拉曼赤道旋转 RXY(θ, φ)' },
  { name: 'RX', label: 'RX', category: '1q_rot', numQubits: 1, paramNames: ['θ'], defaultParams: [1.5708], tint: 'blue', description: '绕 X 轴旋转 RX(θ)' },
  { name: 'RY', label: 'RY', category: '1q_rot', numQubits: 1, paramNames: ['θ'], defaultParams: [1.5708], tint: 'blue', description: '绕 Y 轴旋转 RY(θ)' },
  { name: 'RZ', label: 'RZ', category: '1q_rot', numQubits: 1, paramNames: ['φ'], defaultParams: [0.7854], tint: 'teal', description: 'Virtual-Z 旋转 RZ(φ)' },
  { name: 'CZ', label: 'CZ', category: '2q', numQubits: 2, paramNames: [], defaultParams: [], tint: 'indigo', description: '里德堡/超导原生受控 Z 纠缠门' },
  { name: 'CNOT', label: 'CNOT', category: '2q', numQubits: 2, paramNames: [], defaultParams: [], tint: 'indigo', description: '受控非门 CNOT' },
  { name: 'ISWAP', label: 'iSWAP', category: '2q', numQubits: 2, paramNames: [], defaultParams: [], tint: 'indigo', description: '电容耦合 iSWAP 纠缠门' },
  { name: 'MS', label: 'MS', category: '2q', numQubits: 2, paramNames: ['θ'], defaultParams: [1.5708], tint: 'indigo', description: '离子阱 Mølmer-Sørensen XX 纠缠门' },
  { name: 'RZZ', label: 'RZZ', category: '2q', numQubits: 2, paramNames: ['θ'], defaultParams: [0.7854], tint: 'indigo', description: '两比特 ZZ 耦合旋转' },
  { name: 'SWAP', label: 'SWAP', category: '2q', numQubits: 2, paramNames: [], defaultParams: [], tint: 'indigo', description: '两比特状态交换门' },
  { name: 'CCZ', label: 'CCZ', category: '3q', numQubits: 3, paramNames: [], defaultParams: [], tint: 'indigo', description: '中性原子三比特里德堡阻塞 CCZ 门' },
  { name: 'CCX', label: 'CCX', category: '3q', numQubits: 3, paramNames: [], defaultParams: [], tint: 'indigo', description: 'Toffoli 三比特门' },
  { name: 'DEPOLARIZE', label: '退极化', category: 'noise', numQubits: 1, paramNames: ['p'], defaultParams: [0.01], tint: 'amber', description: '退极化噪声通道' },
  { name: 'PHASE_FLIP', label: '相位翻转', category: 'noise', numQubits: 1, paramNames: ['p'], defaultParams: [0.01], tint: 'amber', description: 'Z 泡利相位翻转噪声' },
  { name: 'MEASURE', label: 'M', category: 'measure', numQubits: 1, paramNames: [], defaultParams: [], tint: 'slate', description: '计算基测量' },
];

export const PALETTE_QUICK_GATES = ['H', 'X', 'Y', 'Z', 'RXY', 'CZ', 'CNOT', 'MEASURE'];

export const QUBIT_DOT_PALETTE = [
  '#FF6B6B',
  '#0A84FF',
  '#34C759',
  '#FF9F0A',
  '#AF52DE',
  '#00C7BE',
  '#FF2D55',
  '#5856D6',
];

export const BACKEND_LABELS: Record<string, string> = {
  qutip: 'QuTiP · 主方程 (≤9Q)',
  kraus: 'Kraus · 密度矩阵 (≤14Q)',
  trajectory: 'Trajectory · 态矢量轨迹 (≤24Q)',
  stim: 'Stim · 稳定子 (500+Q)',
  stim_qutip: 'Stim + QuTiP 标定',
  compare: '多后端对比',
};
