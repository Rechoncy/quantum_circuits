"""
Natural Language -> Quantum Circuit Compiler & Multi-Turn Refinement Engine for Quantum Studio v2.

Dual-Engine Architecture:
  1. Built-in Zero-Config Bilingual Quantum Semantic Compiler & Multi-Turn Editor (`local`):
     - Canonical Quantum Algorithms & Entangled States: Bell (Φ±/Ψ±), N-qubit GHZ (standard &
       Neutral Atom RXY+CZ), 3Q/4Q exact W-state, N-qubit QFT & IQFT, 2Q/3Q Grover search,
       Quantum Teleportation, Bernstein-Vazirani, Hardware-Efficient VQE Ansatz, QAOA MaxCut,
       5Q Bit/Phase-Flip QEC, 1D Cluster/Graph state, SWAP Test, Rabi/Ramsey/Hahn Echo.
     - Compound Prompt Parsing: combines a named algorithm with extra follow-up clauses in the
       same sentence (e.g. "生成 3Q GHZ 态，再在 q0 上加 RZ(phi)，最后加 2% 退极化噪声").
     - Multi-Turn Conversational Refinement (`refine` / `auto`): directly edits the active
       working circuit in place — replace gates (all / ordinal / per-qubit), delete gates or
       noise, tune noise probabilities or symbolic/gate angles, add/remove qubits (with optional
       entanglement), change initial states, switch hardware platforms, or append new gates.
     - Clean Layer Alignment: synchronizes trailing noise layers into a single crisp parallel
       column right before final measurements (no diagonal staggered DEPOL artifacts!).
  2. Optional External OpenAI-Compatible LLM Engine (`llm`):
     - Passes multi-turn conversation history & current working circuit DSL to any OpenAI-compatible
       endpoint (OpenAI / DeepSeek / Ollama / vLLM) and compiles & validates the returned DSL.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import CircuitOpModel, NlCircuitRequest, ProjectModel, QubitSpec
from .project_engine import (
    QUBIT_COLORS,
    dsl_to_project_ops,
    project_to_dsl,
    project_to_openqasm3,
)

# ---------------------------------------------------------------------------
# Helpers: Bilingual Numbers, Angles, Qubits, Gate Tokens, Hardware & Noise
# ---------------------------------------------------------------------------

_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_CLAUSE_SPLIT_RE = re.compile(
    r"[，。；;\n]+|(?:\s*(?:然后|接着|随后|再|并且|之后|最后|同时|顺便|and then|then)\s*)"
    r"|(?:并(?=(?:测量|读出|加入|加上|添加|施加|执行|做|对|在|与|和|把|将|改|换|删|移|切|设)))"
)

_KNOWN_GATE_TOKENS: List[Tuple[str, str]] = [
    ("SQISWAP", r"(?:\bsqiswap\b|√iswap|根号iswap)"),
    ("ISWAP", r"\biswap\b"),
    ("CNOT", r"(?:\bcnot\b|\bcx\b|受控非)"),
    ("SWAP", r"(?:\bswap\b|交换门)"),
    ("CCZ", r"(?:\bccz\b|三比特\s*cz)"),
    ("CCX", r"(?:\bccx\b|\btoffoli\b|托佛利)"),
    ("CZ", r"(?:\bcz\b|受控z|里德堡\s*cz)"),
    ("RZZ", r"\brzz\b"),
    ("MS", r"(?:\bms\b|\brxx\b)"),
    ("RXY", r"(?:\brxy\b|拉曼)"),
    ("RX", r"(?:\brx\b|rx\s*旋转)"),
    ("RY", r"(?:\bry\b|ry\s*旋转)"),
    ("RZ", r"(?:\brz\b|rz\s*旋转|虚拟z|virtual-?z)"),
    ("SX", r"(?:\bsx\b|√x)"),
    ("H", r"(?:\bhadamard\b|哈达玛|\bh\s*门|\bh\b)"),
    ("X", r"(?:\bx\s*门|\bx\b)"),
    ("Y", r"(?:\by\s*门|\by\b)"),
    ("Z", r"(?:\bz\s*门|\bz\b)"),
    ("S", r"(?:\bs\s*门|\bs\b)"),
    ("T", r"(?:\bt\s*门|\bt\b)"),
]

_GATE_DEFAULT_PARAMS: Dict[str, List[Any]] = {
    "RX": [round(math.pi / 2.0, 4)],
    "RY": [round(math.pi / 2.0, 4)],
    "RZ": [round(math.pi / 2.0, 4)],
    "RXY": ["theta", 0.0],
    "RZZ": [round(math.pi / 2.0, 4)],
    "MS": [round(math.pi / 2.0, 4)],
}


def _extract_qubit_count(text: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(
        r"(\d+)\s*(?:个)?\s*(?:量子)?(?:比特|qubit|qubits|q\b)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return max(1, min(12, int(m.group(1))))

    m_cn = re.search(r"([一二两三四五六七八九十])\s*(?:个)?\s*(?:量子)?比特", text)
    if m_cn:
        return max(1, min(12, _CN_NUM[m_cn.group(1)]))

    q_ids = [int(x) for x in re.findall(r"\bq(\d+)\b", text, flags=re.IGNORECASE)]
    if q_ids:
        return max(1, min(12, max(q_ids) + 1))

    return default


def _extract_layers(text: str, default: int = 2) -> int:
    m = re.search(r"(\d+)\s*(?:层|layers?|reps?|步重复|轮)", text, flags=re.IGNORECASE)
    if m:
        return max(1, min(6, int(m.group(1))))
    m_cn = re.search(r"([一二两三四五六])\s*(?:层|轮)", text)
    if m_cn:
        return max(1, min(6, _CN_NUM[m_cn.group(1)]))
    return default


def _extract_bitstring(text: str, length: Optional[int] = None) -> Optional[str]:
    m_ket = re.search(r"\|([01]{1,8})[>⟩]", text)
    if m_ket:
        bs = m_ket.group(1)
        if length is None or len(bs) == length:
            return bs
    m_kw = re.search(
        r"(?:目标态|目标|搜索|查找|隐藏串|秘密串|secret|target|marked)\D{0,8}([01]{2,8})",
        text,
        flags=re.IGNORECASE,
    )
    if m_kw:
        bs = m_kw.group(1)
        if length is None or len(bs) == length:
            return bs
    return None


def _detect_hardware_platform(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in ("中性原子", "里德堡", "光镊", "neutral atom", "rydberg")):
        return "hw_neutral_atom"
    if any(k in low for k in ("离子阱", "trapped ion", "molmer", "sorensen")):
        return "hw_trapped_ion"
    if any(k in low for k in ("超导iswap", "超导 iswap", "电容耦合", "iswap硬件", "iswap 硬件", "iswap平台")):
        return "hw_sc_iswap"
    if any(k in low for k in ("超导", "transmon", "superconducting")):
        return "hw_sc_cz"
    return None


def _extract_noise_clause(text: str) -> Optional[Tuple[str, float]]:
    """Detect noise kind & rate such as '3% 退极化噪声' or '噪声改为 0.05'."""
    low = text.lower()
    kind: Optional[str] = None
    if any(k in low for k in ("退极化", "depolariz", "depol")):
        kind = "DEPOLARIZE"
    elif any(k in low for k in ("振幅阻尼", "能量弛豫", "amplitude damping", "t1衰减", "t1 衰减")):
        kind = "AMPLITUDE_DAMPING"
    elif any(k in low for k in ("相位阻尼", "纯退相干", "phase damping", "dephasing")):
        kind = "PHASE_DAMPING"
    elif any(k in low for k in ("比特翻转", "bit flip", "bit-flip", "x噪声", "x 误差")):
        kind = "BIT_FLIP"
    elif any(k in low for k in ("相位翻转", "phase flip", "phase-flip", "z噪声", "z 误差")):
        kind = "PHASE_FLIP"
    elif any(k in low for k in ("噪声", "噪音", "noise")):
        kind = "DEPOLARIZE"

    if kind is None:
        return None

    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m_pct:
        p = max(0.0001, min(0.5, float(m_pct.group(1)) / 100.0))
        return kind, round(p, 5)
    m_dec = re.search(r"(?:概率|噪声|噪音|误差|强度|改成|改为|调到|调成|设为|为|p\s*=|rate)\D{0,6}(0?\.\d+)", low)
    if m_dec:
        p = max(0.0001, min(0.5, float(m_dec.group(1))))
        return kind, round(p, 5)
    return kind, 0.01


def _extract_angle_token(clause: str) -> Optional[str]:
    """Extract angle expression such as 'pi/2', '3*pi/4', '60度', 'theta', '0.5'."""
    m_sym = re.search(
        r"\b(theta(?:_\d+)?|phi(?:_\d+)?|gamma(?:_\d+)?|beta(?:_\d+)?|alpha(?:_\d+)?)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if m_sym:
        return m_sym.group(1).lower()
    if "θ" in clause:
        return "theta"
    if "φ" in clause:
        return "phi"
    if "γ" in clause:
        return "gamma"
    if "β" in clause:
        return "beta"

    norm = clause.replace("π", "pi")
    m_pi = re.search(
        r"([+-]?\s*(?:\d+(?:\.\d+)?\s*\*?\s*)?pi(?:\s*/\s*\d+(?:\.\d+)?)?)",
        norm,
        flags=re.IGNORECASE,
    )
    if m_pi:
        expr = re.sub(r"\s+", "", m_pi.group(1)).lower()
        expr = re.sub(r"(\d)pi", r"\1*pi", expr)
        try:
            val = float(eval(expr, {"__builtins__": {}, "pi": math.pi}, {}))
            return f"{val:.4f}"
        except Exception:
            return expr

    m_deg = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:度|deg|°)", clause, flags=re.IGNORECASE)
    if m_deg:
        rad = float(m_deg.group(1)) * math.pi / 180.0
        return f"{rad:.4f}"

    m_par = re.search(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*\)", clause)
    if m_par:
        return m_par.group(1)
    m_num = re.search(r"(?:角度|相位|改成|改为|设为|调到|调成|为|rad)\s*([+-]?\d+(?:\.\d+)?)", clause, flags=re.IGNORECASE)
    if m_num:
        return m_num.group(1)

    return None


def _extract_numeric_angle(clause: str) -> Optional[float]:
    """Extract a concrete float value (rad) from expressions like '60度', 'pi/3', '0.8'."""
    norm = clause.replace("π", "pi")
    m_deg = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:度|deg|°)", norm, flags=re.IGNORECASE)
    if m_deg:
        return round(float(m_deg.group(1)) * math.pi / 180.0, 4)
    m_pi = re.search(
        r"([+-]?\s*(?:\d+(?:\.\d+)?\s*\*?\s*)?pi(?:\s*/\s*\d+(?:\.\d+)?)?)",
        norm,
        flags=re.IGNORECASE,
    )
    if m_pi:
        expr = re.sub(r"\s+", "", m_pi.group(1)).lower()
        expr = re.sub(r"(\d)pi", r"\1*pi", expr)
        try:
            return round(float(eval(expr, {"__builtins__": {}, "pi": math.pi}, {})), 4)
        except Exception:
            pass
    m_num = re.search(r"(?:改成|改为|设为|调成|调到|变成|=|为|to)\s*([+-]?\d+(?:\.\d+)?)", norm, flags=re.IGNORECASE)
    if m_num:
        return round(float(m_num.group(1)), 4)
    return None


def _extract_clause_qubits(clause: str, n_qubits: int) -> List[int]:
    low = clause.lower()
    if any(k in low for k in ("所有比特", "全部比特", "每个比特", "全部量子比特", "全比特", "all qubits", "every qubit")):
        return list(range(n_qubits))

    m_rng = re.search(r"q?(\d+)\s*(?:到|至|-|~|to)\s*q?(\d+)", low)
    if m_rng and ("q" in m_rng.group(0) or "比特" in clause):
        a, b = int(m_rng.group(1)), int(m_rng.group(2))
        lo, hi = min(a, b), min(n_qubits - 1, max(a, b))
        return list(range(max(0, lo), hi + 1))

    explicit = [int(x) for x in re.findall(r"\bq(\d+)\b", low)]
    if explicit:
        seen: List[int] = []
        for q in explicit:
            if 0 <= q < n_qubits and q not in seen:
                seen.append(q)
        return seen

    cn_matches = re.findall(r"(?:比特|第)\s*(\d+)\s*(?:个?比特)?", clause)
    if cn_matches and not re.search(r"第\s*\d+\s*个\s*(?:门|[a-zA-Z]+)", clause):
        seen = []
        for x in cn_matches:
            q = int(x)
            if 0 <= q < n_qubits and q not in seen:
                seen.append(q)
        return seen

    m_pair = re.search(r"(?:在|对|连接|控制)\s*(\d+)\s*(?:和|与|,|->|→)\s*(\d+)", clause)
    if m_pair:
        q0, q1 = int(m_pair.group(1)), int(m_pair.group(2))
        if 0 <= q0 < n_qubits and 0 <= q1 < n_qubits and q0 != q1:
            return [q0, q1]

    return []


def _extract_init_states(text: str, n_qubits: int) -> Dict[int, str]:
    inits: Dict[int, str] = {}
    state_map = {
        "|0>": "|0>", "|0⟩": "|0>",
        "|1>": "|1>", "|1⟩": "|1>",
        "|+>": "|+>", "|+⟩": "|+>",
        "|->": "|->", "|-⟩": "|->",
        "|+i>": "|+i>", "|+i⟩": "|+i>",
        "|-i>": "|-i>", "|-i⟩": "|-i>",
    }
    # Broadcast init state: "所有比特初始化为 |+⟩" or "重置初态为 |0⟩"
    m_all = re.search(r"(?:所有|全部|重置)[^，。；;\n]{0,12}(?:初态|初始化)[^，。；;\n]{0,10}(\|[01+\-i]+[>⟩])", text, flags=re.IGNORECASE)
    if m_all:
        st = state_map.get(m_all.group(1).lower().replace("⟩", ">"))
        if st:
            for q in range(n_qubits):
                inits[q] = st

    for m in re.finditer(r"q(\d+)[^，。；;\n]{0,18}(\|[01+\-i]+[>⟩])", text, flags=re.IGNORECASE):
        q = int(m.group(1))
        st = state_map.get(m.group(2).lower().replace("⟩", ">"))
        if 0 <= q < n_qubits and st:
            inits[q] = st
    return inits


def _match_gate_name_in_fragment(fragment: str, custom_gates: List[Any]) -> Optional[str]:
    """Identify a gate name (custom or built-in) mentioned in a short text fragment."""
    low = fragment.lower()
    for cg in custom_gates:
        if re.search(rf"\b{re.escape(cg.name)}\b", fragment, flags=re.IGNORECASE):
            return cg.name
        if cg.display_label and cg.display_label.lower() in low:
            return cg.name
    for canonical, pat in _KNOWN_GATE_TOKENS:
        if re.search(pat, fragment, flags=re.IGNORECASE):
            return canonical
    return None


def _align_circuit_layers(ops: List[CircuitOpModel], n_qubits: int) -> List[CircuitOpModel]:
    """
    Re-schedule operations cleanly so:
      1. Regular gates pack compactly (ASAP) with zero overlap on any qubit wire.
      2. Trailing noise ops (those placed after all gates) synchronize into ONE crisp vertical
         column (`max_gate_step + 1`) instead of staggering diagonally across columns!
      3. Final measurement ops synchronize cleanly in the column right after (`+ 1`).
    """
    if not ops:
        return []

    ordered = [op.model_copy(deep=True) for _, op in sorted(enumerate(ops), key=lambda p: (p[1].step, p[0]))]

    # Identify index of the last actual quantum gate
    last_gate_idx = -1
    for i, op in enumerate(ordered):
        if op.kind == "gate":
            last_gate_idx = i

    qubit_col = [0] * max(1, n_qubits)
    scheduled: List[CircuitOpModel] = []
    trailing_noise: List[CircuitOpModel] = []
    trailing_meas: List[CircuitOpModel] = []

    for idx, op in enumerate(ordered):
        valid_q = [q for q in op.qubits if 0 <= q < n_qubits]
        if not valid_q:
            continue
        op.qubits = valid_q

        if idx > last_gate_idx and op.kind == "noise":
            trailing_noise.append(op)
            continue
        if idx > last_gate_idx and op.kind == "measure":
            trailing_meas.append(op)
            continue

        span = range(min(valid_q), max(valid_q) + 1) if len(valid_q) > 1 and op.kind != "measure" else valid_q
        col = max(qubit_col[q] for q in span)
        op.step = col
        scheduled.append(op)
        for q in span:
            qubit_col[q] = col + 1

    next_col = max(qubit_col) if scheduled else 0

    # Place all trailing noise channels cleanly aligned in ONE parallel layer (unless duplicate qubits)
    if trailing_noise:
        noise_qubit_col = [next_col] * max(1, n_qubits)
        for op in trailing_noise:
            span = op.qubits
            col = max(noise_qubit_col[q] for q in span)
            op.step = col
            scheduled.append(op)
            for q in span:
                noise_qubit_col[q] = col + 1
        next_col = max(noise_qubit_col)

    # Merge & align trailing measurement instructions cleanly into the final column
    if trailing_meas:
        merged_q: List[int] = []
        for op in trailing_meas:
            for q in op.qubits:
                if q not in merged_q:
                    merged_q.append(q)
        merged_q.sort()
        scheduled.append(
            CircuitOpModel(
                id=trailing_meas[0].id,
                step=next_col,
                kind="measure",
                name="MEASURE",
                qubits=merged_q,
            )
        )

    return scheduled


# ---------------------------------------------------------------------------
# Canonical Quantum Algorithm & State Template Generators
# ---------------------------------------------------------------------------

def _gen_bell(text: str, hw_id: Optional[str]) -> Tuple[str, str, List[str], Dict[str, float], int]:
    low = text.lower()
    lines = ["qubits 2"]
    variant = "|Φ+⟩ = (|00⟩ + |11⟩)/√2"
    if any(k in low for k in ("ψ-", "psi-", "psi_-", "单态", "singlet")):
        lines.extend(["X 0", "X 1", "H 0", "CNOT 0 1"])
        variant = "|Ψ-⟩ = (|01⟩ - |10⟩)/√2（自旋单态）"
    elif any(k in low for k in ("ψ+", "psi+", "psi_+")):
        lines.extend(["X 1", "H 0", "CNOT 0 1"])
        variant = "|Ψ+⟩ = (|01⟩ + |10⟩)/√2"
    elif any(k in low for k in ("φ-", "phi-", "phi_-")):
        lines.extend(["X 0", "H 0", "CNOT 0 1"])
        variant = "|Φ-⟩ = (|00⟩ - |11⟩)/√2"
    elif hw_id == "hw_neutral_atom":
        lines.extend(["RXY(1.5708, -1.5708) 0", "RXY(1.5708, -1.5708) 1", "CZ 0 1", "RXY(1.5708, 1.5708) 1"])
        variant = "|Φ+⟩（中性原子 RXY + 里德堡 CZ 原生门实现）"
    else:
        lines.extend(["H 0", "CNOT 0 1"])
    return "NL_Bell_State.qproj", f"已生成 2 比特最大纠缠 Bell 态 {variant} 量子线路。", lines, {}, 2


def _gen_ghz(text: str, hw_id: Optional[str]) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=4) or 4
    n = max(2, min(10, n))
    lines = [f"qubits {n}"]
    if hw_id == "hw_neutral_atom":
        lines.append("RXY(1.5708, -1.5708) 0")
        for q in range(1, n):
            lines.append(f"RXY(1.5708, -1.5708) {q}")
        for q in range(n - 1):
            lines.append(f"CZ {q} {q + 1}")
            lines.append(f"RXY(1.5708, 1.5708) {q + 1}")
        desc = f"已生成 {n} 比特中性原子里德堡阵列 GHZ 纠缠态线路（原生拉曼 RXY + 里德堡阻塞 CZ 门），理想输出 (|{'0'*n}⟩ + |{'1'*n}⟩)/√2。"
    else:
        lines.append("H 0")
        for q in range(n - 1):
            lines.append(f"CNOT {q} {q + 1}")
        desc = f"已生成 {n} 比特 GHZ 最大纠缠态线路（H + 链式 CNOT），理想输出 (|{'0'*n}⟩ + |{'1'*n}⟩)/√2。"
    return f"NL_GHZ_{n}Q.qproj", desc, lines, {}, n


def _gen_w_state(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=3) or 3
    n = max(3, min(4, n))
    lines = [f"qubits {n}"]
    if n == 3:
        theta0 = 2.0 * math.acos(1.0 / math.sqrt(3.0))
        lines.extend([
            f"RY({theta0:.4f}) 0",
            "RY(0.7854) 1",
            "CZ 0 1",
            "RY(-0.7854) 1",
            "CNOT 1 2",
            "CNOT 0 1",
            "X 0",
        ])
        desc = "已生成 3 比特精确 W 态线路，理想输出 (|100⟩ + |010⟩ + |001⟩)/√3（单激发对称纠缠态）。"
    else:
        lines.extend([
            "X 0",
            "RY(0.7854) 2",
            "CNOT 0 2",
            "RY(-0.7854) 2",
            "CNOT 2 0",
            "RY(0.7854) 1",
            "CNOT 0 1",
            "RY(-0.7854) 1",
            "CNOT 1 0",
            "RY(0.7854) 3",
            "CNOT 2 3",
            "RY(-0.7854) 3",
            "CNOT 3 2",
        ])
        desc = "已生成 4 比特二叉树分解精确 W 态线路，理想输出 (|1000⟩ + |0100⟩ + |0010⟩ + |0001⟩)/2。"
    return f"NL_W_State_{n}Q.qproj", desc, lines, {}, n


def _gen_qft(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=3) or 3
    n = max(2, min(6, n))
    is_inv = any(k in text.lower() for k in ("逆", "inverse", "iqft", "qft†", "qft_dag"))
    lines = [f"qubits {n}"]

    def _cphase(c: int, t: int, angle: float) -> None:
        half = angle / 2.0
        lines.append(f"RZ({half:.4f}) {c}")
        lines.append(f"RZ({half:.4f}) {t}")
        lines.append(f"CNOT {c} {t}")
        lines.append(f"RZ({-half:.4f}) {t}")
        lines.append(f"CNOT {c} {t}")

    if not is_inv:
        for j in range(n):
            lines.append(f"H {j}")
            for k in range(j + 1, n):
                _cphase(k, j, math.pi / (1 << (k - j)))
        for j in range(n // 2):
            lines.append(f"SWAP {j} {n - 1 - j}")
        desc = f"已生成 {n} 比特量子傅里叶变换（QFT）线路，包含 Hadamard、精确分解的受控旋转相位门 CPhase 及末端 SWAP。"
        title = f"NL_QFT_{n}Q.qproj"
    else:
        for j in range(n // 2):
            lines.append(f"SWAP {j} {n - 1 - j}")
        for j in range(n - 1, -1, -1):
            for k in range(n - 1, j, -1):
                _cphase(k, j, -math.pi / (1 << (k - j)))
            lines.append(f"H {j}")
        desc = f"已生成 {n} 比特逆量子傅里叶变换（IQFT）线路。"
        title = f"NL_IQFT_{n}Q.qproj"
    return title, desc, lines, {}, n


def _gen_grover(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    bs = _extract_bitstring(text)
    n = _extract_qubit_count(text, default=len(bs) if bs else 2) or 2
    n = max(2, min(3, n))
    target = bs if (bs and len(bs) == n) else ("1" * n)

    lines = [f"qubits {n}"]
    for q in range(n):
        lines.append(f"H {q}")

    n_iters = 1 if n == 2 else 2
    for _ in range(n_iters):
        for q, bit in enumerate(target):
            if bit == "0":
                lines.append(f"X {q}")
        lines.append("CZ 0 1" if n == 2 else "CCZ 0 1 2")
        for q, bit in enumerate(target):
            if bit == "0":
                lines.append(f"X {q}")

        for q in range(n):
            lines.append(f"H {q}")
        for q in range(n):
            lines.append(f"X {q}")
        lines.append("CZ 0 1" if n == 2 else "CCZ 0 1 2")
        for q in range(n):
            lines.append(f"X {q}")
        for q in range(n):
            lines.append(f"H {q}")

    desc = f"已生成 {n} 比特 Grover 量子搜索线路（目标标记态 |{target}⟩，执行 {n_iters} 轮 Oracle + Diffusion 振幅放大）。"
    return f"NL_Grover_{n}Q_{target}.qproj", desc, lines, {}, n


def _gen_teleportation(_text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    lines = [
        "qubits 3",
        "RY(theta) 0",
        "RZ(phi) 0",
        "H 1",
        "CNOT 1 2",
        "CNOT 0 1",
        "H 0",
        "CNOT 1 2",
        "CZ 0 2",
    ]
    variables = {"theta": 1.0472, "phi": 0.5236}
    desc = "已生成 3 比特量子隐形传态线路（q0 制备含参态 RY(θ)RZ(φ)|0⟩，通过 q1-q2 Bell 通道与相干反馈门将态精确传输至 q2）。"
    return "NL_Teleportation_3Q.qproj", desc, lines, variables, 3


def _gen_bv_or_dj(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    secret = _extract_bitstring(text)
    if secret is None:
        n_data = max(2, min(5, (_extract_qubit_count(text, default=4) or 4) - 1))
        secret = "10110"[:n_data]
    n_data = len(secret)
    n = n_data + 1
    anc = n - 1
    lines = [f"qubits {n}", f"X {anc}"]
    for q in range(n):
        lines.append(f"H {q}")
    for q, bit in enumerate(secret):
        if bit == "1":
            lines.append(f"CNOT {q} {anc}")
    for q in range(n_data):
        lines.append(f"H {q}")
    desc = f"已生成 {n} 比特 Bernstein-Vazirani 隐藏比特串线路（隐藏串 s = {secret}，q{anc} 为相位反冲辅助比特）。"
    return f"NL_BV_{secret}.qproj", desc, lines, {}, n


def _gen_vqe_ansatz(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=4) or 4
    n = max(2, min(8, n))
    layers = _extract_layers(text, default=2)
    lines = [f"qubits {n}"]
    variables: Dict[str, float] = {}
    p_idx = 0
    for l in range(layers):
        for q in range(n):
            v_y = f"theta_{p_idx}"
            variables[v_y] = round(0.4 + 0.35 * ((l + q) % 4), 4)
            lines.append(f"RY({v_y}) {q}")
            p_idx += 1
        for q in range(n - 1):
            lines.append(f"CZ {q} {q + 1}")
    for q in range(n):
        v_z = f"phi_{q}"
        variables[v_z] = round(0.25 * (q + 1), 4)
        lines.append(f"RZ({v_z}) {q}")
    desc = f"已生成 {n} 比特 {layers} 层硬件高效变分线路（Hardware-Efficient VQE Ansatz），包含 {len(variables)} 个可扫描符号参数。"
    return f"NL_VQE_{n}Q_L{layers}.qproj", desc, lines, variables, n


def _gen_qaoa(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=4) or 4
    n = max(2, min(8, n))
    p_layers = _extract_layers(text, default=1)
    lines = [f"qubits {n}"]
    variables: Dict[str, float] = {}
    for q in range(n):
        lines.append(f"H {q}")
    for l in range(p_layers):
        g_name = "gamma" if p_layers == 1 else f"gamma_{l}"
        b_name = "beta" if p_layers == 1 else f"beta_{l}"
        variables[g_name] = round(0.7 + 0.2 * l, 4)
        variables[b_name] = round(0.4 + 0.15 * l, 4)
        edges = [(q, q + 1) for q in range(n - 1)]
        if n >= 3:
            edges.append((n - 1, 0))
        for u, v in edges:
            lines.append(f"RZZ({g_name}) {u} {v}")
        for q in range(n):
            lines.append(f"RX({b_name}) {q}")
    desc = f"已生成 {n} 比特 p={p_layers} 层环形图 MaxCut QAOA 线路（RZZ 代价层 + RX 混合层）。"
    return f"NL_QAOA_{n}Q_p{p_layers}.qproj", desc, lines, variables, n


def _gen_qec_3q(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    is_phase = any(k in text.lower() for k in ("相位翻转", "phase flip", "phase-flip", "z纠错"))
    lines = ["qubits 5", "RY(theta) 0", "CNOT 0 1", "CNOT 0 2"]
    if is_phase:
        lines.extend(["H 0", "H 1", "H 2", "PHASE_FLIP(p_err) 1", "H 0", "H 1", "H 2"])
        title = "NL_QEC_PhaseFlip_5Q.qproj"
        desc = "已生成 3 比特相位翻转纠错编码 + 双校验子（q3, q4）稳定子测量线路。"
    else:
        lines.extend(["BIT_FLIP(p_err) 1"])
        title = "NL_QEC_BitFlip_5Q.qproj"
        desc = "已生成 3 比特比特翻转重复码编码 + 双校验子（q3, q4）纠错线路。"
    lines.extend(["CNOT 0 3", "CNOT 1 3", "CNOT 1 4", "CNOT 2 4"])
    return title, desc, lines, {"theta": 1.0472, "p_err": 0.05}, 5


def _gen_cluster_state(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    n = _extract_qubit_count(text, default=4) or 4
    n = max(2, min(10, n))
    lines = [f"qubits {n}"]
    for q in range(n):
        lines.append(f"H {q}")
    for q in range(n - 1):
        lines.append(f"CZ {q} {q + 1}")
    desc = f"已生成 {n} 比特一维线性 Cluster 簇态（图态）线路。"
    return f"NL_Cluster_{n}Q.qproj", desc, lines, {}, n


def _gen_swap_test(_text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    lines = [
        "qubits 3",
        "RY(theta) 1",
        "RY(phi) 2",
        "H 0",
        "CNOT 2 1",
        "CCX 0 1 2",
        "CNOT 2 1",
        "H 0",
    ]
    desc = "已生成 3 比特 SWAP Test 量子态内积比对线路，q0 测得 |0⟩ 概率为 (1 + |⟨ψ₁|ψ₂⟩|²)/2。"
    return "NL_SWAP_Test_3Q.qproj", desc, lines, {"theta": 1.5708, "phi": 0.5236}, 3


def _gen_ramsey_or_rabi(text: str) -> Tuple[str, str, List[str], Dict[str, float], int]:
    low = text.lower()
    if any(k in low for k in ("echo", "回波", "hahn")):
        return (
            "NL_Hahn_Echo_1Q.qproj",
            "已生成单比特 Hahn 自旋回波去相干抑制线路。",
            ["qubits 1", "RX(1.5708) 0", "RZ(phi) 0", "X 0", "RZ(phi) 0", "RX(1.5708) 0"],
            {"phi": 0.6},
            1,
        )
    if any(k in low for k in ("ramsey", "拉姆齐", "干涉")):
        return (
            "NL_Ramsey_1Q.qproj",
            "已生成单比特 Ramsey 干涉测量线路（SX → RZ(phi) → SX）。",
            ["qubits 1", "SX 0", "RZ(phi) 0", "SX 0"],
            {"phi": 0.7854},
            1,
        )
    return (
        "NL_Rabi_1Q.qproj",
        "已生成单比特 Rabi 振荡实验线路 RX(theta)。",
        ["qubits 1", "RX(theta) 0"],
        {"theta": 1.5708},
        1,
    )


def _match_canonical_algorithm(clause: str, hw_id: Optional[str]) -> Optional[Tuple[str, str, List[str], Dict[str, float], int]]:
    low = clause.lower()
    if any(k in low for k in ("隐形传态", "teleport")):
        return _gen_teleportation(clause)
    if any(k in low for k in ("grover", "格罗弗", "量子搜索", "振幅放大")):
        return _gen_grover(clause)
    if any(k in low for k in ("qft", "傅里叶", "fourier")):
        return _gen_qft(clause)
    if any(k in low for k in ("bernstein", "vazirani", "deutsch", "隐藏串", "秘密串")):
        return _gen_bv_or_dj(clause)
    if any(k in low for k in ("qaoa", "maxcut", "最大割")):
        return _gen_qaoa(clause)
    if any(k in low for k in ("vqe", "变分", "ansatz", "硬件高效")):
        return _gen_vqe_ansatz(clause)
    if any(k in low for k in ("纠错", "qec", "重复码", "校验子", "syndrome")):
        return _gen_qec_3q(clause)
    if any(k in low for k in ("swap test", "swap测试", "内积测试")):
        return _gen_swap_test(clause)
    if any(k in low for k in ("cluster", "簇态", "图态", "graph state")):
        return _gen_cluster_state(clause)
    if re.search(r"\bw\s*态|\bw[\s-]?state\b", low):
        return _gen_w_state(clause)
    if "ghz" in low:
        return _gen_ghz(clause, hw_id)
    if any(k in low for k in ("bell", "贝尔态", "最大纠缠态", "单态", "singlet")):
        return _gen_bell(clause, hw_id)
    if any(k in low for k in ("ramsey", "拉姆齐", "rabi", "拉比", "hahn", "自旋回波", "spin echo")):
        return _gen_ramsey_or_rabi(clause)
    return None


# ---------------------------------------------------------------------------
# Clause-Level Gate / Noise / Measure Instruction Emitter
# ---------------------------------------------------------------------------

def _parse_single_gate_or_noise_clause(
    clause: str,
    n_qubits: int,
    project: ProjectModel,
    variables: Dict[str, float],
) -> Tuple[List[CircuitOpModel], bool, bool]:
    """
    Parse one clause into CircuitOpModel(s) to append.
    Returns: (emitted_ops, saw_measure, recognized)
    """
    low = clause.lower()
    qs = _extract_clause_qubits(clause, n_qubits)
    emitted: List[CircuitOpModel] = []

    def _reg_var(ang: str) -> Any:
        try:
            return round(float(ang), 4)
        except ValueError:
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", ang) and ang not in variables:
                variables[ang] = 0.02 if ang.startswith("p_") else round(math.pi / 2.0, 4)
            return ang

    def _new_op(kind: str, name: str, qubits: List[int], params: Optional[List[Any]] = None) -> CircuitOpModel:
        return CircuitOpModel(
            id=f"nl_{uuid.uuid4().hex[:7]}",
            step=999,  # will be packed by _align_circuit_layers
            kind=kind,  # type: ignore[arg-type]
            name=name,
            qubits=qubits,
            params=params or [],
        )

    # Pure measurement clause (make sure no gate/noise is hidden in the same clause)
    if any(k in low for k in ("测量", "measure", "读出")) and not any(k in low for k in ("不测量", "无需测量", "取消测量", "删", "去掉")):
        has_gate_or_noise = (
            _match_gate_name_in_fragment(clause, project.custom_gates) is not None
            or _extract_noise_clause(clause) is not None
        )
        if not has_gate_or_noise:
            tq = qs if qs else list(range(n_qubits))
            emitted.append(_new_op("measure", "MEASURE", tq))
            return emitted, True, True

    # Noise clause
    noise_info = _extract_noise_clause(clause)
    if noise_info is not None and not any(k in low for k in ("删", "去掉", "移除", "清除", "无噪声")):
        n_kind, n_p = noise_info
        tq = qs if qs else list(range(n_qubits))
        if n_kind == "DEPOLARIZE" and len(tq) == 2 and len(qs) == 2:
            emitted.append(_new_op("noise", "DEPOLARIZE", [tq[0], tq[1]], [n_p]))
        else:
            for q in tq:
                emitted.append(_new_op("noise", n_kind, [q], [n_p]))
        return emitted, False, True

    # Custom gate from project library
    for cg in project.custom_gates:
        if re.search(rf"\b{re.escape(cg.name)}\b", clause, flags=re.IGNORECASE) or (
            cg.display_label and cg.display_label.lower() in low
        ):
            need = cg.num_qubits
            tq = qs if len(qs) >= need else list(range(min(need, n_qubits)))
            if len(tq) >= need:
                params: List[Any] = []
                if cg.param_names:
                    ang = _extract_angle_token(clause)
                    params = [_reg_var(ang)] if ang else list(cg.default_params)
                emitted.append(_new_op("gate", cg.name, tq[:need], params))
                return emitted, False, True

    # 3-qubit gates
    if any(k in low for k in ("ccz", "三比特cz", "三比特 cz")):
        tq = qs[:3] if len(qs) >= 3 else ([0, 1, 2] if n_qubits >= 3 else [])
        if len(tq) == 3:
            emitted.append(_new_op("gate", "CCZ", tq))
            return emitted, False, True
    if any(k in low for k in ("ccx", "toffoli", "托佛利")):
        tq = qs[:3] if len(qs) >= 3 else ([0, 1, 2] if n_qubits >= 3 else [])
        if len(tq) == 3:
            emitted.append(_new_op("gate", "CCX", tq))
            return emitted, False, True

    # 2-qubit gates
    matched_2q: Optional[str] = None
    if re.search(r"\b(?:cnot|cx)\b", low) or "受控非" in clause or ("纠缠" in clause and not re.search(r"\b(?:cz|iswap|swap|rzz|ms)\b", low)):
        matched_2q = "CNOT"
    elif re.search(r"\bsqiswap\b", low) or "√iswap" in low:
        matched_2q = "SQISWAP"
    elif re.search(r"\biswap\b", low):
        matched_2q = "ISWAP"
    elif re.search(r"\bswap\b", low) or "交换门" in clause:
        matched_2q = "SWAP"
    elif re.search(r"\bcz\b", low) or "受控z" in low or "里德堡cz" in low:
        matched_2q = "CZ"
    elif re.search(r"\brzz\b", low):
        matched_2q = "RZZ"
    elif re.search(r"\b(?:rxx|ms)\b", low):
        matched_2q = "MS"

    if matched_2q is not None and n_qubits >= 2:
        if len(qs) >= 2:
            pairs = [(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
        elif any(k in low for k in ("相邻", "链式", "依次", "最近邻", "所有")):
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        elif len(qs) == 1:
            # e.g. "和 q3 做 CZ 纠缠" right after adding a qubit q4 -> connect (q3, last_qubit)
            other = n_qubits - 1 if qs[0] != n_qubits - 1 else max(0, qs[0] - 1)
            pairs = [(qs[0], other)]
        else:
            pairs = [(max(0, n_qubits - 2), n_qubits - 1)]
        ang = _extract_angle_token(clause)
        for qa, qb in pairs:
            if matched_2q in ("RZZ", "MS"):
                emitted.append(_new_op("gate", matched_2q, [qa, qb], [_reg_var(ang or "1.5708")]))
            else:
                emitted.append(_new_op("gate", matched_2q, [qa, qb]))
        return emitted, False, True

    # 1-qubit gates
    tq = qs if qs else [0]
    if re.search(r"\brxy\b", low) or "拉曼" in clause:
        ang = _extract_angle_token(clause) or "theta"
        p0 = _reg_var(ang)
        for q in tq:
            emitted.append(_new_op("gate", "RXY", [q], [p0, 0.0]))
        return emitted, False, True

    for rot in ("RX", "RY", "RZ"):
        if re.search(rf"\b{rot.lower()}\b", low) or f"{rot.lower()}旋转" in low or f"{rot.lower()} 旋转" in low or ("相位门" in clause and rot == "RZ"):
            ang = _extract_angle_token(clause) or ("phi" if rot == "RZ" else "1.5708")
            p0 = _reg_var(ang)
            for q in tq:
                emitted.append(_new_op("gate", rot, [q], [p0]))
            return emitted, False, True

    fixed_1q: Optional[str] = None
    if "hadamard" in low or "哈达玛" in clause or re.search(r"\bh\s*门|\bh\b", low) or "叠加态" in clause:
        fixed_1q = "H"
    elif re.search(r"\bsx\b", low) or "√x" in low:
        fixed_1q = "SX"
    elif re.search(r"\bx\s*门|\bx\b", low) or "翻转" in clause:
        fixed_1q = "X"
    elif re.search(r"\by\s*门|\by\b", low):
        fixed_1q = "Y"
    elif re.search(r"\bz\s*门|\bz\b", low):
        fixed_1q = "Z"
    elif re.search(r"\bs\s*门|\bs\b", low):
        fixed_1q = "S"
    elif re.search(r"\bt\s*门|\bt\b", low):
        fixed_1q = "T"

    if fixed_1q is not None:
        for q in tq:
            emitted.append(_new_op("gate", fixed_1q, [q]))
        return emitted, False, True

    return [], False, False


# ---------------------------------------------------------------------------
# Multi-Turn Conversational Refinement Engine (`refine` & `auto`)
# ---------------------------------------------------------------------------

def _is_refinement_intent(prompt: str, project: ProjectModel) -> bool:
    """Heuristically decide in `auto` mode whether prompt refines the current circuit or builds a new one."""
    if not project.operations:
        return False
    low = prompt.lower()

    # Strong "create new" signals with a named canonical algorithm
    explicit_new = any(k in low for k in ("新建", "重新生成", "重新创建", "从头", "帮我生成", "生成一个", "构建一个", "制备"))
    has_canonical = _match_canonical_algorithm(prompt, None) is not None
    if has_canonical and (explicit_new or not any(k in low for k in ("把", "将", "改", "换", "删", "去掉", "移除", "增加", "追加", "调"))):
        return False

    # Strong refinement verbs / targets
    refine_keywords = (
        "改成", "改为", "换成", "替换为", "替换成", "调成", "调到", "设为", "修改",
        "删掉", "删除", "去掉", "移除", "取消", "清除", "撤销",
        "增加一个", "增加1个", "加一个", "加1个", "加两个", "加2个", "新增比特", "添加比特", "再加", "末尾", "追加",
        "切换", "改用", "翻转控制", "互换", "调大", "调小",
        "replace", "change", "remove", "delete", "add a qubit", "append", "switch to", "set ",
    )
    if any(k in low for k in refine_keywords):
        return True

    return False


def _apply_refinement_commands(
    prompt: str,
    project: ProjectModel,
) -> Tuple[ProjectModel, List[str], List[str], Optional[str]]:
    """
    Apply one or more natural-language refinement clauses directly onto a deep copy of `project`.
    Returns: (updated_project, changes_list, unrecognized_clauses, recommended_hw_id)
    """
    proj = project.model_copy(deep=True)
    changes: List[str] = []
    unrecognized: List[str] = []
    rec_hw: Optional[str] = None

    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(prompt) if c.strip()]
    if not clauses:
        raise ValueError("请输入具体的修改指令，例如：「把退极化噪声改成 3%」「把 CZ 换成 iSWAP」「增加 1 个比特」。")

    for clause in clauses:
        low = clause.lower()
        handled = False

        # ------------------------------------------------------------------
        # 1. Switch Hardware Platform (e.g. "切换到超导 CZ 硬件", "改用中性原子")
        # ------------------------------------------------------------------
        if any(k in low for k in ("切换", "改用", "换成", "使用", "硬件", "平台")) and _detect_hardware_platform(clause) is not None:
            # Make sure it's not actually "把 CZ 换成 iSWAP" (gate replacement)
            if not re.search(r"把[^，。]+(?:换成|改成|替换为)", clause) or any(k in low for k in ("硬件", "平台", "中性原子", "里德堡", "离子阱", "超导")):
                hw_id = _detect_hardware_platform(clause)
                if hw_id:
                    proj.active_hardware_id = hw_id
                    rec_hw = hw_id
                    hw_obj = next((h for h in proj.hardware_profiles if h.id == hw_id), None)
                    changes.append(f"切换硬件平台 → {hw_obj.name if hw_obj else hw_id}")
                    handled = True

        # ------------------------------------------------------------------
        # 2. Add Qubit(s) (e.g. "增加 1 个比特", "添加比特 q4")
        # ------------------------------------------------------------------
        m_add_q = re.search(
            r"(?:增加|添加|新增|加上|加|add)\s*(?:(\d+|[一二两三四])\s*个?)?\s*(?:量子)?(?:比特|qubit)",
            clause,
            flags=re.IGNORECASE,
        )
        if m_add_q:
            raw_cnt = m_add_q.group(1)
            add_cnt = int(raw_cnt) if (raw_cnt and raw_cnt.isdigit()) else _CN_NUM.get(raw_cnt or "一", 1)
            add_cnt = max(1, min(6, add_cnt))
            added_names: List[str] = []
            for _ in range(add_cnt):
                next_id = len(proj.qubits)
                max_x = max((q.pos_x for q in proj.qubits), default=-4.5)
                proj.qubits.append(
                    QubitSpec(
                        id=next_id,
                        name=f"q{next_id}",
                        color=QUBIT_COLORS[next_id % len(QUBIT_COLORS)],
                        pos_x=round(max_x + 4.5, 1),
                        pos_y=0.0,
                    )
                )
                added_names.append(f"q{next_id}")
            # Extend trailing uniform noise layer (if present on all old qubits) & trailing MEASURE
            old_n = len(proj.qubits) - add_cnt
            last_gate_step = max((o.step for o in proj.operations if o.kind == "gate"), default=-1)
            trailing_noises = [o for o in proj.operations if o.kind == "noise" and o.step > last_gate_step]
            if trailing_noises and len({q for o in trailing_noises for q in o.qubits}) >= old_n:
                ref_n = trailing_noises[0]
                for new_q in range(old_n, len(proj.qubits)):
                    proj.operations.append(
                        CircuitOpModel(
                            id=f"nl_{uuid.uuid4().hex[:7]}",
                            step=ref_n.step,
                            kind="noise",
                            name=ref_n.name,
                            qubits=[new_q],
                            params=list(ref_n.params),
                        )
                    )
            for op in proj.operations:
                if op.kind == "measure":
                    op.qubits = list(range(len(proj.qubits)))
            changes.append(f"新增 {add_cnt} 个比特 ({', '.join(added_names)})")
            handled = True
            # Note: If the same clause ALSO specifies a gate on the new qubit (e.g. "增加比特并与q3做CZ纠缠"), don't skip gate parsing!
            if _match_gate_name_in_fragment(clause, proj.custom_gates) is None and "纠缠" not in clause:
                continue

        # ------------------------------------------------------------------
        # 3. Delete Qubit (e.g. "删掉最后一个比特", "删除比特 q3")
        # ------------------------------------------------------------------
        if any(k in low for k in ("删", "移除", "去掉", "delete", "remove")) and any(k in low for k in ("比特", "qubit")):
            if len(proj.qubits) <= 1:
                changes.append("至少需保留 1 个量子比特（未删除）")
            else:
                qs_del = _extract_clause_qubits(clause, len(proj.qubits))
                del_idx = qs_del[0] if qs_del else (len(proj.qubits) - 1)
                del_name = proj.qubits[del_idx].name
                proj.qubits = [
                    q.model_copy(update={"id": new_i})
                    for new_i, q in enumerate(q for i, q in enumerate(proj.qubits) if i != del_idx)
                ]
                next_ops: List[CircuitOpModel] = []
                for op in proj.operations:
                    if op.kind == "measure":
                        kept = [q - 1 if q > del_idx else q for q in op.qubits if q != del_idx]
                        if kept:
                            next_ops.append(op.model_copy(update={"qubits": kept}))
                    elif del_idx not in op.qubits:
                        remapped = [q - 1 if q > del_idx else q for q in op.qubits]
                        next_ops.append(op.model_copy(update={"qubits": remapped}))
                proj.operations = next_ops
                changes.append(f"删除比特 {del_name}（剩余 {len(proj.qubits)}Q）")
            continue

        # ------------------------------------------------------------------
        # 4. Update Qubit Initial State (e.g. "把 q0 初态设为 |1⟩", "所有比特初始化为 |+⟩")
        # ------------------------------------------------------------------
        init_updates = _extract_init_states(clause, len(proj.qubits))
        if init_updates:
            for q_idx, st in init_updates.items():
                proj.qubits[q_idx].init_state = st  # type: ignore[assignment]
            st_desc = ", ".join(f"{proj.qubits[q].name}={st.replace('>', '⟩')}" for q, st in init_updates.items())
            changes.append(f"更新初态: {st_desc}")
            handled = True
            if _match_gate_name_in_fragment(clause, proj.custom_gates) is None:
                continue

        # ------------------------------------------------------------------
        # 5. Delete Noise / Measurements / Specific Gate
        # ------------------------------------------------------------------
        if any(k in low for k in ("删掉", "删除", "去掉", "移除", "清除", "取消", "无噪声", "delete", "remove")):
            # 5a. Remove noise
            if any(k in low for k in ("噪声", "噪音", "退极化", "阻尼", "noise", "depol")):
                before = len(proj.operations)
                proj.operations = [o for o in proj.operations if o.kind != "noise"]
                removed = before - len(proj.operations)
                changes.append(f"移除 {removed} 个噪声操作" if removed else "当前线路无显式噪声操作")
                continue

            # 5b. Remove measurement
            if any(k in low for k in ("测量", "读出", "measure")):
                before = len(proj.operations)
                proj.operations = [o for o in proj.operations if o.kind != "measure"]
                removed = before - len(proj.operations)
                changes.append(f"移除测量指令 ({removed})")
                continue

            # 5c. Remove last gate ("删掉最后一个门")
            if any(k in low for k in ("最后一个", "最后一步", "末尾门", "last gate")):
                gate_indices = [i for i, o in enumerate(proj.operations) if o.kind == "gate"]
                if gate_indices:
                    popped = proj.operations.pop(gate_indices[-1])
                    changes.append(f"删除最后一个门 {popped.name}")
                continue

            # 5d. Remove ordinal gate ("删除第 2 个门" or "删除第 2 个 CZ")
            m_ord = re.search(r"第\s*(\d+|[一二两三四五六七八九十])\s*个\s*(.*)", clause)
            if m_ord:
                raw_n = m_ord.group(1)
                ord_idx = (int(raw_n) if raw_n.isdigit() else _CN_NUM.get(raw_n, 1)) - 1
                g_filter = _match_gate_name_in_fragment(m_ord.group(2), proj.custom_gates)
                candidates = [
                    i
                    for i, o in enumerate(proj.operations)
                    if o.kind == "gate" and (g_filter is None or o.name.upper() == g_filter.upper())
                ]
                if 0 <= ord_idx < len(candidates):
                    popped = proj.operations.pop(candidates[ord_idx])
                    changes.append(f"删除第 {ord_idx + 1} 个 {popped.name} 门")
                    continue

            # 5e. Remove gates by name (optionally filtered by qubit)
            g_del = _match_gate_name_in_fragment(clause, proj.custom_gates)
            if g_del:
                qs_f = _extract_clause_qubits(clause, len(proj.qubits))
                before = len(proj.operations)
                proj.operations = [
                    o
                    for o in proj.operations
                    if not (
                        o.kind == "gate"
                        and o.name.upper() == g_del.upper()
                        and (not qs_f or any(q in o.qubits for q in qs_f))
                    )
                ]
                removed = before - len(proj.operations)
                changes.append(f"删除 {removed} 个 {g_del} 门")
                continue

        # ------------------------------------------------------------------
        # 6. Replace Gate (e.g. "把所有 CZ 换成 iSWAP", "把第 2 个 CZ 改成 CNOT")
        # ------------------------------------------------------------------
        m_rep = re.search(r"(?:把|将)?(.+?)(?:换成|替换成|替换为|改成|改为|replace\s+with|to)\s*(.+)", clause, flags=re.IGNORECASE)
        if m_rep:
            lhs, rhs = m_rep.group(1).strip(), m_rep.group(2).strip()

            # 6a. Noise probability update ("把退极化噪声改成 3%")
            if any(k in lhs.lower() for k in ("噪声", "噪音", "退极化", "阻尼", "noise", "depol")):
                n_info = _extract_noise_clause(clause)
                if n_info is not None:
                    n_kind, n_p = n_info
                    existing_noise = [o for o in proj.operations if o.kind == "noise"]
                    if existing_noise:
                        old_p = existing_noise[0].params[0] if existing_noise[0].params else "?"
                        for o in existing_noise:
                            if any(k in rhs.lower() for k in ("振幅", "相位", "比特翻转", "相位翻转", "退极化")):
                                o.name = n_kind
                            o.params = [n_p]
                        changes.append(f"更新 {len(existing_noise)} 个噪声层参数: {old_p} → {n_p}")
                    else:
                        for q in range(len(proj.qubits)):
                            proj.operations.append(
                                CircuitOpModel(
                                    id=f"nl_{uuid.uuid4().hex[:7]}",
                                    step=999,
                                    kind="noise",
                                    name=n_kind,
                                    qubits=[q],
                                    params=[n_p],
                                )
                            )
                        changes.append(f"新增全比特 {n_kind}({n_p}) 噪声层")
                    continue

            # 6b. Symbolic variable value update ("把 theta 改成 60 度", "把 phi 设为 pi/4")
            m_var = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", lhs)
            if m_var and m_var.group(1) in proj.variables:
                vname = m_var.group(1)
                new_val = _extract_numeric_angle(rhs)
                if new_val is not None:
                    old_val = proj.variables[vname]
                    proj.variables[vname] = new_val
                    changes.append(f"更新符号参数 {vname}: {old_val} → {new_val}")
                    continue

            # 6c. Hardware parameter update ("把 T1 改成 100 us", "把阻塞半径改成 8 um")
            active_hw = next((h for h in proj.hardware_profiles if h.id == proj.active_hardware_id), None)
            if active_hw is not None:
                m_val_num = re.search(r"(\d+(?:\.\d+)?)", rhs)
                if m_val_num:
                    val_f = float(m_val_num.group(1))
                    if "t1" in lhs.lower():
                        active_hw.t1_us = val_f
                        changes.append(f"更新硬件 T1 → {val_f} μs")
                        continue
                    if "t2" in lhs.lower():
                        active_hw.t2_us = val_f
                        changes.append(f"更新硬件 T2* → {val_f} μs")
                        continue
                    if "阻塞半径" in lhs or "r_b" in lhs.lower() or "rb" in lhs.lower():
                        active_hw.blockade_radius_um = val_f
                        changes.append(f"更新里德堡阻塞半径 R_b → {val_f} μm")
                        continue

            # 6d. Gate angle update ("把 q1 上的 RY 角度改成 pi/4")
            src_gate = _match_gate_name_in_fragment(lhs, proj.custom_gates)
            dst_gate = _match_gate_name_in_fragment(rhs, proj.custom_gates)
            if src_gate and dst_gate is None and any(k in lhs for k in ("角度", "相位", "参数")) or (
                src_gate and dst_gate is None and _extract_angle_token(rhs) is not None
            ):
                ang = _extract_angle_token(rhs)
                if ang is not None:
                    qs_f = _extract_clause_qubits(lhs, len(proj.qubits))
                    updated = 0
                    for o in proj.operations:
                        if o.kind == "gate" and o.name.upper() == src_gate.upper():
                            if qs_f and not any(q in o.qubits for q in qs_f):
                                continue
                            val_p: Any = round(float(ang), 4) if re.match(r"^[+-]?\d", ang) else ang
                            if isinstance(val_p, str) and val_p not in proj.variables:
                                proj.variables[val_p] = round(math.pi / 2.0, 4)
                            if o.params:
                                o.params[0] = val_p
                            else:
                                o.params = [val_p]
                            updated += 1
                    if updated:
                        changes.append(f"更新 {updated} 个 {src_gate} 门角度 → {ang}")
                        continue

            # 6e. Gate-to-Gate replacement ("把所有 CZ 换成 iSWAP" / "把第 2 个 CZ 换成 CNOT")
            if src_gate and dst_gate:
                qs_f = _extract_clause_qubits(lhs, len(proj.qubits))
                m_ord = re.search(r"第\s*(\d+|[一二两三四五六七八九十])\s*个", lhs)
                ord_target = None
                if m_ord:
                    raw_n = m_ord.group(1)
                    ord_target = (int(raw_n) if raw_n.isdigit() else _CN_NUM.get(raw_n, 1)) - 1

                matches = [
                    o
                    for o in proj.operations
                    if o.kind == "gate"
                    and o.name.upper() == src_gate.upper()
                    and (not qs_f or any(q in o.qubits for q in qs_f))
                ]
                targets = (
                    [matches[ord_target]]
                    if (ord_target is not None and 0 <= ord_target < len(matches))
                    else matches
                )
                rhs_ang = _extract_angle_token(rhs)
                if targets:
                    for o in targets:
                        o.name = dst_gate
                        if rhs_ang is not None:
                            val_p = round(float(rhs_ang), 4) if re.match(r"^[+-]?\d", rhs_ang) else rhs_ang
                            o.params = [val_p, 0.0] if dst_gate == "RXY" else [val_p]
                        elif dst_gate in _GATE_DEFAULT_PARAMS:
                            o.params = list(_GATE_DEFAULT_PARAMS[dst_gate])
                            if dst_gate == "RXY" and "theta" not in proj.variables:
                                proj.variables["theta"] = round(math.pi / 2.0, 4)
                        else:
                            o.params = []
                    changes.append(f"替换 {len(targets)} 个 {src_gate} → {dst_gate}")
                    continue

        # ------------------------------------------------------------------
        # 7. Flip CNOT / 2Q Control & Target ("翻转所有 CNOT 控制目标")
        # ------------------------------------------------------------------
        if any(k in low for k in ("翻转控制", "互换控制", "反转cnot", "翻转cnot")):
            cnt = 0
            for o in proj.operations:
                if o.kind == "gate" and len(o.qubits) == 2:
                    o.qubits = [o.qubits[1], o.qubits[0]]
                    cnt += 1
            if cnt:
                changes.append(f"翻转 {cnt} 个两比特门的控制/目标比特")
                continue

        # ------------------------------------------------------------------
        # 8. Fallback: Append Gate / Noise / Measure Clause onto Current Circuit
        # ------------------------------------------------------------------
        new_ops, saw_meas, recognized = _parse_single_gate_or_noise_clause(
            clause, len(proj.qubits), proj, proj.variables
        )
        if recognized and new_ops:
            # Insert new gate(s) right before trailing noise/measure with monotonically ordered steps
            if new_ops[0].kind == "gate":
                insert_pos = len(proj.operations)
                while insert_pos > 0 and proj.operations[insert_pos - 1].kind in ("noise", "measure"):
                    insert_pos -= 1
                base_step = max((o.step for o in proj.operations[:insert_pos]), default=-1) + 1
                for k_i, no in enumerate(new_ops):
                    no.step = base_step + k_i
                after_step = base_step + len(new_ops)
                for tail_op in proj.operations[insert_pos:]:
                    tail_op.step = max(tail_op.step, after_step + (1 if tail_op.kind == "measure" else 0))
                proj.operations[insert_pos:insert_pos] = new_ops
                gate_summary = ", ".join(f"{o.name}({','.join(f'q{q}' for q in o.qubits)})" for o in new_ops[:3])
                if len(new_ops) > 3:
                    gate_summary += f" 等 {len(new_ops)} 门"
                changes.append(f"追加门: {gate_summary}")
            elif new_ops[0].kind == "noise":
                # Replace trailing noise or append before measure
                insert_pos = len(proj.operations)
                while insert_pos > 0 and proj.operations[insert_pos - 1].kind == "measure":
                    insert_pos -= 1
                base_step = max((o.step for o in proj.operations[:insert_pos]), default=-1) + 1
                for no in new_ops:
                    no.step = base_step
                for tail_op in proj.operations[insert_pos:]:
                    tail_op.step = max(tail_op.step, base_step + 1)
                proj.operations[insert_pos:insert_pos] = new_ops
                changes.append(f"追加噪声层: {new_ops[0].name}({new_ops[0].params[0]})")
            elif saw_meas:
                proj.operations = [o for o in proj.operations if o.kind != "measure"] + new_ops
                changes.append("添加全比特测量层")
            handled = True

        if not handled:
            unrecognized.append(clause)

    if not changes:
        raise ValueError(
            "未能识别出明确的精调操作。你可以说例如：「把退极化噪声改成 3%」、「把所有 CZ 换成 iSWAP」、「增加 1 个比特并和 q3 纠缠」、「删掉所有噪声」、「把 theta 改成 60 度」。"
        )

    proj.operations = _align_circuit_layers(proj.operations, len(proj.qubits))
    return proj, changes, unrecognized, rec_hw


# ---------------------------------------------------------------------------
# Optional External OpenAI-Compatible LLM Engine (Multi-Turn History Aware)
# ---------------------------------------------------------------------------

def _call_external_llm(req: NlCircuitRequest) -> Tuple[str, str, str, Dict[str, float], Optional[str], List[str]]:
    api_key = (req.llm_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (req.llm_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = (req.llm_model or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()

    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise ValueError(
            "启用外部 LLM 模式需要填写 API Key（或配置环境变量 OPENAI_API_KEY），也可切换回免密钥的「内置语义引擎」。"
        )

    custom_gate_list = ", ".join(
        f"{cg.name}({cg.num_qubits}Q)" for cg in req.project.custom_gates
    ) or "none"

    system_prompt = (
        "You are an expert quantum circuit designer inside Quantum Studio.\n"
        "Convert the user's natural language request (new generation or multi-turn refinement) "
        "into a strict JSON object with exact keys:\n"
        '  {"title": "Name.qproj", "explanation": "Chinese concise explanation", '
        '"changes": ["change 1", "change 2"], '
        '"dsl": "qubits N\\n...", "variables": {"theta": 1.5708}, '
        '"recommended_hardware_id": null or one of ["hw_neutral_atom","hw_sc_cz","hw_sc_iswap","hw_trapped_ion"]}\n'
        "Allowed DSL instructions (one per line):\n"
        "  qubits N\n"
        "  1Q gates: H, X, Y, Z, S, SDG, T, TDG, SX, RX(angle), RY(angle), RZ(angle), RXY(theta, phi)\n"
        "  2Q gates: CNOT q0 q1, CZ q0 q1, SWAP q0 q1, ISWAP q0 q1, SQISWAP q0 q1, RZZ(angle) q0 q1, MS(angle) q0 q1\n"
        "  3Q gates: CCX q0 q1 q2, CCZ q0 q1 q2\n"
        f"  Project custom gates: {custom_gate_list}\n"
        "  Noise: DEPOLARIZE(p) q0 [q1], AMPLITUDE_DAMPING(p) q0, PHASE_DAMPING(p) q0, BIT_FLIP(p) q0, PHASE_FLIP(p) q0\n"
        "  Measurement: MEASURE 0 1 ...\n"
        "Return ONLY valid JSON without markdown fences."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for h in req.history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    current_dsl = project_to_dsl(req.project)
    user_msg = (
        f"Current working circuit ({req.project.name}):\n{current_dsl}\n"
        f"Variables: {json.dumps(req.project.variables)}\n"
        f"Requested mode: {req.mode}\n\n"
        f"User instruction:\n{req.prompt.strip()}"
    )
    messages.append({"role": "user", "content": user_msg})

    payload = {"model": model, "temperature": 0.1, "messages": messages}
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    http_req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(http_req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:300]
        raise ValueError(f"LLM API 返回 HTTP {e.code}: {body}") from e
    except Exception as e:
        raise ValueError(f"无法连接 LLM API ({url}): {e}") from e

    content = raw["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)

    title = str(parsed.get("title") or req.project.name or "AI_Generated.qproj")
    explanation = str(parsed.get("explanation") or "已通过外部 LLM 更新量子线路。")
    dsl = str(parsed.get("dsl") or "").strip()
    variables = {str(k): float(v) for k, v in (parsed.get("variables") or {}).items()}
    hw_id = parsed.get("recommended_hardware_id")
    changes = [str(c) for c in (parsed.get("changes") or [])]
    if not dsl:
        raise ValueError("LLM 返回结果中缺少 dsl 线路代码。")
    return title, explanation, dsl, variables, hw_id, changes


# ---------------------------------------------------------------------------
# Main Public Entrypoint
# ---------------------------------------------------------------------------

def generate_circuit_from_nl(req: NlCircuitRequest) -> Dict[str, Any]:
    prompt = req.prompt.strip()
    if not prompt:
        raise ValueError("请输入自然语言描述或修改指令，例如：「生成 4 比特中性原子 GHZ 态」或「把噪声改成 3%」。")

    before_q = len(req.project.qubits)
    before_g = sum(1 for o in req.project.operations if o.kind == "gate")

    detected_hw = _detect_hardware_platform(prompt)
    engine_used = "local_quantum_compiler"
    unrecognized: List[str] = []
    changes: List[str] = []
    action_mode = req.mode

    # ---- Path A: External LLM API ----
    if req.engine == "llm":
        title, explanation, dsl, extra_vars, llm_hw, llm_changes = _call_external_llm(req)
        recommended_hw = llm_hw or detected_hw
        engine_used = f"llm ({req.llm_model or 'openai-compatible'})"
        base_proj = req.project.model_copy(deep=True)
        m_q = re.search(r"^\s*qubits\s+(\d+)", dsl, flags=re.IGNORECASE | re.MULTILINE)
        n_target = int(m_q.group(1)) if m_q else len(base_proj.qubits)
        base_proj.qubits = [
            base_proj.qubits[i].model_copy(update={"id": i})
            if i < len(base_proj.qubits)
            else QubitSpec(
                id=i,
                name=f"q{i}",
                color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
                pos_x=round(i * 4.5, 1),
                pos_y=0.0,
            )
            for i in range(n_target)
        ]
        new_qubits, new_ops, merged_vars = dsl_to_project_ops(dsl, base_proj)
        merged_vars.update(extra_vars)
        aligned_ops = _align_circuit_layers(new_ops, len(new_qubits))

        preview_proj = base_proj.model_copy(deep=True)
        preview_proj.name = title
        preview_proj.qubits = new_qubits
        preview_proj.operations = aligned_ops
        preview_proj.variables = merged_vars
        if recommended_hw:
            preview_proj.active_hardware_id = recommended_hw
        changes = llm_changes or ["由外部 LLM 更新线路"]
        action_mode = "llm"

    # ---- Path B: Local Multi-Turn Refinement (`refine`, `append`, or `auto` -> refine) ----
    elif req.mode in ("refine", "append") or (req.mode == "auto" and _is_refinement_intent(prompt, req.project)):
        action_mode = "refine"
        preview_proj, changes, unrecognized, recommended_hw = _apply_refinement_commands(prompt, req.project)
        title = preview_proj.name
        explanation = "已基于当前线路完成精调：" + "；".join(changes) + "。"

    # ---- Path C: Local Fresh Generation (Canonical Algorithm + Compound Extra Clauses OR Free-Form) ----
    else:
        action_mode = "replace"
        recommended_hw = detected_hw
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(prompt) if c.strip()]

        # Try matching canonical algorithm in the first clause or entire prompt
        canonical_match = None
        matched_clause_idx = -1
        for idx, c in enumerate(clauses):
            m_alg = _match_canonical_algorithm(c, detected_hw)
            if m_alg is not None:
                canonical_match = m_alg
                matched_clause_idx = idx
                break
        if canonical_match is None:
            canonical_match = _match_canonical_algorithm(prompt, detected_hw)
            matched_clause_idx = 0

        if canonical_match is not None:
            title, explanation, dsl_lines, extra_vars, n_alg = canonical_match
            # Allow qubit count mentioned in prompt to expand if free-form clauses reference higher qubits
            n_target = max(n_alg, _extract_qubit_count(prompt, default=n_alg) or n_alg)
            if n_target != n_alg and dsl_lines and dsl_lines[0].startswith("qubits "):
                dsl_lines[0] = f"qubits {n_target}"

            base_proj = req.project.model_copy(deep=True)
            base_proj.qubits = [
                QubitSpec(
                    id=i,
                    name=f"q{i}",
                    color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
                    pos_x=round(i * 4.5, 1),
                    pos_y=0.0,
                )
                for i in range(n_target)
            ]
            base_proj.variables = {}
            new_qubits, new_ops, merged_vars = dsl_to_project_ops("\n".join(dsl_lines), base_proj)
            merged_vars.update(extra_vars)
            changes.append(f"生成 {title.replace('.qproj', '')}")

            # Process any additional compound clauses (extra gates, noise, init states, measure)!
            saw_explicit_measure = False
            saw_noise = False
            for idx, c in enumerate(clauses):
                if idx == matched_clause_idx:
                    # Still check if the matched clause itself requested noise (e.g. "生成 4Q GHZ 并加 1% 噪声")
                    n_info = _extract_noise_clause(c)
                    if n_info is not None:
                        nk, np_val = n_info
                        for q in range(len(new_qubits)):
                            new_ops.append(
                                CircuitOpModel(
                                    id=f"nl_{uuid.uuid4().hex[:7]}",
                                    step=999,
                                    kind="noise",
                                    name=nk,
                                    qubits=[q],
                                    params=[np_val],
                                )
                            )
                        saw_noise = True
                        changes.append(f"添加同步 {nk}({np_val}) 噪声层")
                    continue

                # Check initial state overrides in extra clauses
                inits_c = _extract_init_states(c, len(new_qubits))
                if inits_c:
                    for qi, st in inits_c.items():
                        new_qubits[qi].init_state = st  # type: ignore[assignment]
                    changes.append("设置初态 " + ", ".join(f"q{q}={s.replace('>', '⟩')}" for q, s in inits_c.items()))
                    if _match_gate_name_in_fragment(c, base_proj.custom_gates) is None:
                        continue

                extra_ops, saw_m, recognized = _parse_single_gate_or_noise_clause(
                    c, len(new_qubits), base_proj, merged_vars
                )
                if recognized and extra_ops:
                    if extra_ops[0].kind == "noise":
                        saw_noise = True
                        changes.append(f"添加同步 {extra_ops[0].name}({extra_ops[0].params[0]}) 噪声层")
                    elif saw_m:
                        saw_explicit_measure = True
                    else:
                        changes.append(f"附加门: {', '.join(o.name for o in extra_ops[:3])}")
                    new_ops.extend(extra_ops)

            # Check global init states anywhere in prompt
            for qi, st in _extract_init_states(prompt, len(new_qubits)).items():
                new_qubits[qi].init_state = st  # type: ignore[assignment]

            # Check global noise if not already added in clauses
            if not saw_noise:
                global_noise = _extract_noise_clause(prompt)
                if global_noise is not None and not any(o.kind == "noise" for o in new_ops):
                    nk, np_val = global_noise
                    for q in range(len(new_qubits)):
                        new_ops.append(
                            CircuitOpModel(
                                id=f"nl_{uuid.uuid4().hex[:7]}",
                                step=999,
                                kind="noise",
                                name=nk,
                                qubits=[q],
                                params=[np_val],
                            )
                        )
                    changes.append(f"添加同步 {nk}({np_val}) 噪声层")

            no_meas = any(k in prompt.lower() for k in ("不测量", "无需测量", "without measurement", "no measure"))
            if not no_meas and not saw_explicit_measure and not any(o.kind == "measure" for o in new_ops):
                new_ops.append(
                    CircuitOpModel(
                        id=f"nl_m_{uuid.uuid4().hex[:6]}",
                        step=999,
                        kind="measure",
                        name="MEASURE",
                        qubits=list(range(len(new_qubits))),
                    )
                )

            preview_proj = base_proj.model_copy(deep=True)
            preview_proj.name = title
            preview_proj.qubits = new_qubits
            preview_proj.operations = _align_circuit_layers(new_ops, len(new_qubits))
            preview_proj.variables = merged_vars
            if recommended_hw:
                preview_proj.active_hardware_id = recommended_hw
        else:
            # Free-form clause-by-clause circuit construction from scratch
            n_target = _extract_qubit_count(prompt, default=2) or 2
            init_states = _extract_init_states(prompt, n_target)
            base_proj = req.project.model_copy(deep=True)
            base_proj.qubits = [
                QubitSpec(
                    id=i,
                    name=f"q{i}",
                    color=QUBIT_COLORS[i % len(QUBIT_COLORS)],
                    init_state=init_states.get(i, "|0>"),  # type: ignore[arg-type]
                    pos_x=round(i * 4.5, 1),
                    pos_y=0.0,
                )
                for i in range(n_target)
            ]
            base_proj.variables = {}
            new_ops_list: List[CircuitOpModel] = []
            saw_meas = False

            for c in clauses:
                low_c = c.lower()
                if re.fullmatch(r"(?:创建|新建|使用|设置|准备)?\s*(?:\d+|[一二两三四五六七八九十])\s*(?:个)?(?:量子)?(?:比特|qubits?)", low_c):
                    continue
                if ("初态" in c or "初始化" in c) and _match_gate_name_in_fragment(c, base_proj.custom_gates) is None:
                    continue
                emitted, sm, rec = _parse_single_gate_or_noise_clause(
                    c, n_target, base_proj, base_proj.variables
                )
                if rec and emitted:
                    new_ops_list.extend(emitted)
                    if sm:
                        saw_meas = True
                else:
                    unrecognized.append(c)

            if not new_ops_list:
                raise ValueError(
                    "未能从描述中识别出量子算法或门指令。可尝试：「生成 4 比特中性原子 GHZ 态」、「3 比特 QFT 线路」，或分步描述门操作如「对 q0 做 H，q0 和 q1 做 CNOT，最后测量」。"
                )

            no_meas = any(k in prompt.lower() for k in ("不测量", "无需测量", "no measure"))
            if not no_meas and not saw_meas:
                new_ops_list.append(
                    CircuitOpModel(
                        id=f"nl_m_{uuid.uuid4().hex[:6]}",
                        step=999,
                        kind="measure",
                        name="MEASURE",
                        qubits=list(range(n_target)),
                    )
                )

            title = f"NL_Custom_{n_target}Q.qproj"
            preview_proj = base_proj.model_copy(deep=True)
            preview_proj.name = title
            preview_proj.operations = _align_circuit_layers(new_ops_list, n_target)
            if recommended_hw:
                preview_proj.active_hardware_id = recommended_hw
            gate_cnt = sum(1 for o in preview_proj.operations if o.kind == "gate")
            changes.append(f"新建 {n_target}Q 线路 ({gate_cnt} 门)")
            explanation = f"已根据自然语言逐句解析并构建 {n_target} 比特量子线路（共 {gate_cnt} 个量子门）。"

    clean_dsl = project_to_dsl(preview_proj)
    clean_qasm = project_to_openqasm3(preview_proj)
    after_q = len(preview_proj.qubits)
    after_g = sum(1 for o in preview_proj.operations if o.kind == "gate")
    rec_backend = "qutip" if after_q <= 5 else ("kraus" if after_q <= 8 else "stim")

    return {
        "title": preview_proj.name,
        "explanation": explanation,
        "engine_used": engine_used,
        "action_mode": action_mode,
        "changes": changes,
        "unrecognized_clauses": unrecognized,
        "diff_summary": {
            "qubits_before": before_q,
            "qubits_after": after_q,
            "gates_before": before_g,
            "gates_after": after_g,
        },
        "dsl": clean_dsl,
        "openqasm3": clean_qasm,
        "qubits": [q.model_dump() for q in preview_proj.qubits],
        "operations": [op.model_dump() for op in preview_proj.operations],
        "variables": preview_proj.variables,
        "recommended_hardware_id": recommended_hw,
        "recommended_backend": rec_backend,
    }
