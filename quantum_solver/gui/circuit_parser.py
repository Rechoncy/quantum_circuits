"""
Parameterized Quantum Circuit DSL Parser & Graphical Circuit Diagram Renderer.
Allows users to write clean, human-readable quantum circuits with symbolic parameters
(e.g. `theta`, `phi`, `p_err`, `pi/4`) in the GUI and compile them to `QuantumCircuit`.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.circuit import GateInstruction, MeasureInstruction, NoiseInstruction, QuantumCircuit
from ..core.noise import NoiseChannel, ReadoutError


SAFE_MATH_NAMESPACE: Dict[str, float] = {
    "pi": math.pi,
    "PI": math.pi,
    "e": math.e,
    "sqrt2": math.sqrt(2.0),
}


def eval_math_expr(expr_str: str, variables: Optional[Dict[str, float]] = None) -> float:
    """
    Safely evaluate a mathematical expression like 'pi/2', '2*theta + 0.1', '-pi/4'
    using the provided symbolic variable dictionary.
    """
    env = dict(SAFE_MATH_NAMESPACE)
    if variables:
        env.update(variables)

    expr_str = expr_str.strip()
    if not expr_str:
        return 0.0

    tree = ast.parse(expr_str, mode="eval")

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return float(env[node.id])
            raise ValueError(f"未定义的参数变量 '{node.id}'（请在参数栏或扫描器中设置）")
        if isinstance(node, ast.UnaryOp):
            val = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id.lower()
            args = [_eval_node(a) for a in node.args]
            if fname == "sin":
                return math.sin(args[0])
            if fname == "cos":
                return math.cos(args[0])
            if fname == "tan":
                return math.tan(args[0])
            if fname == "sqrt":
                return math.sqrt(args[0])
            if fname == "exp":
                return math.exp(args[0])
        raise ValueError(f"不支持的表达式语法: '{expr_str}'")

    return float(_eval_node(tree))


def extract_symbolic_parameters(script: str) -> List[str]:
    """
    Scan a quantum circuit script for symbolic parameter names inside parenthesized
    expressions, e.g. `RX(theta) 0`, `RZ(2*phi) 1`, `DEPOLARIZE(p_noise) 0`.
    Returns a sorted list of unique variable names (excluding constants like 'pi').
    """
    found: List[str] = []
    reserved = set(SAFE_MATH_NAMESPACE.keys()) | {"sin", "cos", "tan", "sqrt", "exp"}
    for raw_line in script.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for match in re.finditer(r"\(([^)]*)\)", line):
            inner = match.group(1)
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", inner):
                if token not in reserved and token not in found:
                    found.append(token)
    return found


def parse_circuit_script(
    script: str,
    variables: Optional[Dict[str, float]] = None,
    default_num_qubits: int = 2,
    circuit_name: str = "UserCircuit",
) -> QuantumCircuit:
    """
    Parse a quantum circuit DSL script into a `QuantumCircuit`.

    Supported syntax examples:
        qubits 3
        H 0
        RX(theta) 0
        CNOT 0 1
        CZ 1 2
        RZZ(pi/4) 0 1
        DEPOLARIZE(0.01) 0
        DEPOLARIZE(0.02) 0 1
        AMPLITUDE_DAMPING(0.015) 0
        PHASE_DAMPING(0.01) 1
        PAULI_ERROR(0.01, 0.0, 0.02) 0
        MEASURE 0 1 2
    """
    lines = [raw.split("#", 1)[0].strip() for raw in script.splitlines()]
    lines = [ln for ln in lines if ln]

    num_qubits: Optional[int] = None
    max_q_seen = -1

    # First pass: determine num_qubits
    inst_pattern = re.compile(r"^([A-Za-z0-9_]+)(?:\(([^)]*)\))?\s+(.*)$")
    for ln in lines:
        parts = ln.split()
        if parts[0].lower() in ("qubits", "num_qubits", "n_qubits") and len(parts) >= 2:
            num_qubits = int(parts[1])
            continue
        m = inst_pattern.match(ln)
        if m:
            q_str = m.group(3).replace(",", " ")
            q_list = [int(tok.lstrip("qQ")) for tok in q_str.split() if tok.strip()]
            if q_list:
                max_q_seen = max(max_q_seen, max(q_list))

    if num_qubits is None:
        num_qubits = max(default_num_qubits, max_q_seen + 1, 1)
    else:
        num_qubits = max(num_qubits, max_q_seen + 1, 1)

    circ = QuantumCircuit(num_qubits=num_qubits, name=circuit_name)

    for line_no, ln in enumerate(lines, start=1):
        parts = ln.split()
        cmd_lower = parts[0].lower()
        if cmd_lower in ("qubits", "num_qubits", "n_qubits"):
            continue
        if cmd_lower in ("measure_all", "measureall", "m_all"):
            circ.measure_all()
            continue

        m = inst_pattern.match(ln)
        if not m:
            raise ValueError(f"第 {line_no} 行语法无法解析: '{ln}'")

        op_name = m.group(1).upper()
        param_str = m.group(2)
        qubit_str = m.group(3).replace(",", " ")

        params: Tuple[float, ...] = ()
        if param_str is not None and param_str.strip():
            params = tuple(
                eval_math_expr(p_item, variables)
                for p_item in param_str.split(",")
                if p_item.strip()
            )

        qubits = tuple(int(tok.lstrip("qQ")) for tok in qubit_str.split() if tok.strip())
        if not qubits:
            raise ValueError(f"第 {line_no} 行缺少目标比特编号: '{ln}'")

        # Handle measurement instructions
        if op_name in ("M", "MEASURE", "MEAS"):
            if len(params) == 1:
                p_err = params[0]
                circ.measure(qubits, readout_error=ReadoutError(p_err, p_err))
            elif len(params) >= 2:
                circ.measure(qubits, readout_error=ReadoutError(params[0], params[1]))
            else:
                circ.measure(qubits)
            continue

        # Handle inline noise channels
        if op_name in ("DEPOLARIZE", "DEPOL", "DEPOLARIZING", "DEPOLARIZE1", "DEPOLARIZE2"):
            p = params[0] if params else 0.01
            if len(qubits) == 1:
                circ.depolarize(qubits[0], p)
            elif len(qubits) == 2:
                circ.depolarize(qubits, p)
            else:
                for q in qubits:
                    circ.depolarize(q, p)
            continue

        if op_name in ("AMPLITUDE_DAMPING", "AD", "T1_DECAY"):
            gamma = params[0] if params else 0.01
            for q in qubits:
                circ.amplitude_damping(q, gamma)
            continue

        if op_name in ("PHASE_DAMPING", "PD", "DEPHASING"):
            lam = params[0] if params else 0.01
            for q in qubits:
                circ.phase_damping(q, lam)
            continue

        if op_name in ("BIT_FLIP", "X_ERROR"):
            p = params[0] if params else 0.01
            for q in qubits:
                circ.add_noise(NoiseChannel.bit_flip(q, p))
            continue

        if op_name in ("PHASE_FLIP", "Z_ERROR"):
            p = params[0] if params else 0.01
            for q in qubits:
                circ.add_noise(NoiseChannel.phase_flip(q, p))
            continue

        if op_name in ("PAULI_ERROR", "PAULI_CHANNEL_1", "PAULI1Q"):
            if len(params) != 3:
                raise ValueError(f"第 {line_no} 行 PAULI_ERROR 需要 3 个参数 (px, py, pz)")
            for q in qubits:
                circ.pauli_error(q, params[0], params[1], params[2])
            continue

        if op_name in ("THERMAL", "THERMAL_RELAXATION"):
            if len(params) < 3:
                raise ValueError(f"第 {line_no} 行 THERMAL 需要 3 个参数 (t1_ns, t2_ns, duration_ns)")
            for q in qubits:
                circ.thermal_relaxation(q, params[0], params[1], params[2])
            continue

        # Standard or experimental quantum gate
        if len(qubits) > 1 and op_name in ("H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX", "SXDG", "RX", "RY", "RZ", "RXY", "U3"):
            # Broadcast 1-qubit gate across listed qubits (e.g. `H 0 1 2`)
            for q in qubits:
                circ.add_gate(op_name, (q,), params=params)
        else:
            circ.add_gate(op_name, qubits, params=params)

    return circ


PRESET_CIRCUIT_TEMPLATES: Dict[str, Tuple[str, Dict[str, float]]] = {
    "1. 单比特 Rabi / Ramsey 连续旋转": (
        """# 单比特旋转与相干演化 (可扫描 theta 观察 Rabi 振荡)
qubits 1
RX(theta) 0
RZ(phi) 0
SX 0
MEASURE 0
""",
        {"theta": round(math.pi / 2.0, 4), "phi": 0.4},
    ),
    "2. 两比特 Bell 纠缠态 + 可调噪声": (
        """# 两比特 Bell 态制备与退极化噪声实验
qubits 2
H 0
CNOT 0 1
DEPOLARIZE(p_noise) 0 1
MEASURE 0 1
""",
        {"p_noise": 0.02},
    ),
    "3. 三比特 GHZ 纠缠态制备与演化": (
        """# 三比特 GHZ 态制备 (|000> + |111>)/sqrt(2)
qubits 3
H 0
CNOT 0 1
CNOT 1 2
PHASE_FLIP(p_dephase) 2
MEASURE 0 1 2
""",
        {"p_dephase": 0.015},
    ),
    "4. 两比特 QFT + SWAP (最优门综合演示)": (
        """# 1xCZ + 3xCNOT(SWAP) 复合线路
# 点击【最优综合】可压缩为 <=2 个原生 CZ 门
qubits 2
H 0
CZ 0 1
S 0
H 1
CNOT 0 1
CNOT 1 0
CNOT 0 1
RY(theta) 0
RX(phi) 1
MEASURE 0 1
""",
        {"theta": round(math.pi / 3.0, 4), "phi": round(math.pi / 5.0, 4)},
    ),
    "5. 变分量子本征求解 (2Q VQE Ansatz)": (
        """# 含参双比特变分线路 (支持 theta / J_zz 1D/2D 扫描)
qubits 2
RY(theta) 0
RY(theta) 1
CZ 0 1
RX(J_zz) 0
RX(J_zz) 1
CZ 0 1
MEASURE 0 1
""",
        {"theta": 1.05, "J_zz": 0.8},
    ),
    "6. 中性原子里德堡阵列 (RXY + Rydberg CZ)": (
        """# 中性原子光镊阵列: 拉曼 RXY + 里德堡阻塞 CZ 门
qubits 3
RXY(theta, 0) 0
RXY(pi/2, -pi/2) 1
CZ 0 1
RXY(pi/2, -pi/2) 1
RXY(pi/2, -pi/2) 2
CZ 1 2
RXY(pi/2, pi/2) 2
RZ(phi) 0
MEASURE 0 1 2
""",
        {"theta": round(math.pi / 2.0, 4), "phi": 0.5},
    ),
}
