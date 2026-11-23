"""
Custom Gate Evaluation, Unitarity Validation, Clifford Detection,
Pauli-String Hamiltonian Builder, Parametric Matrix Evaluator, and Composite Expander.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from ..core.gates import I2, PAULI_X, PAULI_Y, PAULI_Z, PAULIS_1Q
from .models import CircuitOpModel, CustomGateDef


def _safe_complex_eval(expr: str, extra_vars: Optional[Dict[str, Any]] = None) -> complex:
    """Safely evaluate a complex scalar math expression such as '1/sqrt(2)', 'exp(i*pi/4)', '-1j'."""
    s = str(expr).strip()
    if not s:
        return 0.0 + 0.0j
    # Allow 'i' as imaginary unit when standing alone (e.g. '1i', 'i', '-i', 'i*pi')
    ns: Dict[str, Any] = {
        "__builtins__": {},
        "i": 1j,
        "j": 1j,
        "pi": math.pi,
        "e": math.e,
        "sqrt": lambda x: np.sqrt(complex(x)),
        "exp": lambda x: np.exp(complex(x)),
        "sin": lambda x: np.sin(complex(x)),
        "cos": lambda x: np.cos(complex(x)),
        "tan": lambda x: np.tan(complex(x)),
        "abs": abs,
    }
    if extra_vars:
        ns.update(extra_vars)
    # Replace shorthand '1i' or '0.5i' -> '1*1j'
    s = re.sub(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)i\b", r"\1j", s)
    val = eval(s, ns, {})
    return complex(val)


def parse_pauli_hamiltonian(hamiltonian_str: str, num_qubits: int) -> np.ndarray:
    """
    Parse a Pauli-string Hamiltonian such as:
      '0.5*XX + 0.5*YY'
      '0.25*ZZ - 0.5*XI + 0.5*IX'
      'X + Z' (for 1 qubit)
    into an exact 2^n x 2^n Hermitian numpy matrix.
    """
    dim = 1 << num_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    cleaned = hamiltonian_str.strip()
    if not cleaned:
        return H

    # Split into signed terms on '+' or '-' outside parentheses
    terms = re.split(r"(?=[+-])", cleaned.replace(" ", ""))
    for raw_term in terms:
        if not raw_term:
            continue
        # Match optional coefficient * Pauli string (e.g. +0.5*XX, -YY, 1.2*ZI)
        m = re.match(r"^([+-]?[^*]*)\*?([IXYZ]+)$", raw_term)
        if not m:
            # Try Pauli string alone with optional sign
            m2 = re.match(r"^([+-]?)([IXYZ]+)$", raw_term)
            if not m2:
                raise ValueError(f"无法解析哈密顿量项 '{raw_term}'，请使用如 '0.5*XX + 0.5*YY' 格式")
            coeff_str, p_str = m2.group(1), m2.group(2)
        else:
            coeff_str, p_str = m.group(1), m.group(2)

        if len(p_str) != num_qubits:
            raise ValueError(
                f"泡利项 '{p_str}' 的比特数 ({len(p_str)}) 与门比特数 ({num_qubits}) 不一致"
            )

        if coeff_str in ("", "+"):
            coeff = 1.0
        elif coeff_str == "-":
            coeff = -1.0
        else:
            coeff = float(np.real(_safe_complex_eval(coeff_str)))

        mat = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for ch in p_str:
            mat = np.kron(mat, PAULIS_1Q[ch])
        H += coeff * mat

    # Enforce exact Hermiticity
    return 0.5 * (H + H.conj().T)


def eval_parametric_gate_expr(
    expression: str,
    num_qubits: int,
    param_values: Dict[str, float],
) -> np.ndarray:
    """
    Evaluate a parametric matrix expression such as:
      'expm(-1j * theta / 2 * (cos(phi)*X + sin(phi)*Y))'
      'expm(-1j * theta / 2 * ZX)'
      '[[cos(theta/2), -1j*sin(theta/2)], [-1j*sin(theta/2), cos(theta/2)]]'
    """
    dim = 1 << num_qubits
    ns: Dict[str, Any] = {
        "__builtins__": {},
        "np": np,
        "pi": math.pi,
        "e": math.e,
        "i": 1j,
        "j": 1j,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "expm": expm,
        "kron": np.kron,
        "eye": np.eye(dim, dtype=np.complex128),
        "I": I2,
        "X": PAULI_X,
        "Y": PAULI_Y,
        "Z": PAULI_Z,
    }
    # Add 2-qubit Pauli products II..ZZ for convenience
    if num_qubits == 2:
        for a, ma in PAULIS_1Q.items():
            for b, mb in PAULIS_1Q.items():
                ns[a + b] = np.kron(ma, mb)
    elif num_qubits == 3:
        for a, ma in PAULIS_1Q.items():
            for b, mb in PAULIS_1Q.items():
                ab = np.kron(ma, mb)
                for c, mc in PAULIS_1Q.items():
                    ns[a + b + c] = np.kron(ab, mc)

    for k, v in param_values.items():
        ns[k] = float(v)

    expr = expression.strip()
    # Allow exp(...) on matrix by routing exp(matrix) to expm
    def _smart_exp(x: Any) -> Any:
        arr = np.asarray(x)
        if arr.ndim == 2:
            return expm(arr)
        return np.exp(x)

    ns["exp"] = _smart_exp
    expr = re.sub(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)i\b", r"\1j", expr)
    res = eval(expr, ns, {})
    mat = np.asarray(res, dtype=np.complex128)
    if mat.shape != (dim, dim):
        raise ValueError(f"表达式结果维度 {mat.shape} 与 {num_qubits} 比特期望维度 ({dim}, {dim}) 不符")
    return mat


def validate_unitary_matrix(U: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float, bool]:
    """
    Validate whether U is unitary (||U^† U - I||_F < tol) and check whether it is a Clifford gate.
    Returns (is_unitary, frobenius_error, is_clifford).
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False, float("inf"), False
    dim = U.shape[0]
    err = float(np.linalg.norm(U.conj().T @ U - np.eye(dim, dtype=np.complex128)))
    is_unitary = err < tol
    if not is_unitary:
        return False, err, False

    # Check Clifford property for 1Q or 2Q gates: U P U^† must be ±P' (single Pauli up to phase)
    n_q = int(round(math.log2(dim)))
    is_cliff = False
    if (1 << n_q) == dim and n_q in (1, 2):
        is_cliff = True
        paulis = (
            [PAULI_X, PAULI_Z]
            if n_q == 1
            else [
                np.kron(PAULI_X, I2),
                np.kron(PAULI_Z, I2),
                np.kron(I2, PAULI_X),
                np.kron(I2, PAULI_Z),
            ]
        )
        all_paulis = (
            list(PAULIS_1Q.values())
            if n_q == 1
            else [np.kron(a, b) for a in PAULIS_1Q.values() for b in PAULIS_1Q.values()]
        )
        u_dag = U.conj().T
        for gen in paulis:
            conj = U @ gen @ u_dag
            matched = any(
                abs(abs(np.trace(conj @ P)) / dim - 1.0) < 1e-5
                for P in all_paulis
            )
            if not matched:
                is_cliff = False
                break

    return True, err, is_cliff


def evaluate_custom_gate_unitary(
    gate_def: CustomGateDef,
    param_values: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute (unitary_matrix, optional_hamiltonian_generator) for any of the 4 CustomGateDef modes.
    For 'hamiltonian' mode, returns (expm(-i * scale * H), 2 * scale * H) so QuTiP integrates H directly!
    """
    n_q = gate_def.num_qubits
    dim = 1 << n_q
    p_vals = list(param_values) if param_values is not None else list(gate_def.default_params)
    p_dict = {
        pname: float(p_vals[i]) if i < len(p_vals) else (
            float(gate_def.default_params[i]) if i < len(gate_def.default_params) else 0.0
        )
        for i, pname in enumerate(gate_def.param_names)
    }

    dtype = gate_def.definition_type
    if dtype == "matrix":
        if not gate_def.matrix_exprs or len(gate_def.matrix_exprs) != dim:
            raise ValueError(f"矩阵维度必须为 {dim}×{dim}")
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for r in range(dim):
            if len(gate_def.matrix_exprs[r]) != dim:
                raise ValueError(f"第 {r + 1} 行长度必须为 {dim}")
            for c in range(dim):
                mat[r, c] = _safe_complex_eval(gate_def.matrix_exprs[r][c], p_dict)
        return mat, None

    if dtype == "expression":
        if not gate_def.expression:
            raise ValueError("请输入参数化矩阵表达式")
        mat = eval_parametric_gate_expr(gate_def.expression, n_q, p_dict)
        return mat, None

    if dtype == "hamiltonian":
        if not gate_def.hamiltonian_str:
            raise ValueError("请输入泡利字符串哈密顿量，例如 '0.5*XX + 0.5*YY'")
        H = parse_pauli_hamiltonian(gate_def.hamiltonian_str, n_q)
        scale = float(p_vals[0]) if (gate_def.param_names and p_vals) else float(gate_def.hamiltonian_scale)
        U = expm(-1.0j * scale * H)
        return U, H

    if dtype == "composite":
        from ..core.circuit import embed_gate_unitary
        from ..core.gates import gate_unitary

        U = np.eye(dim, dtype=np.complex128)
        for sub in sorted(gate_def.sub_operations, key=lambda op: op.step):
            if sub.kind != "gate":
                continue
            sub_params = [
                float(np.real(_safe_complex_eval(str(p), p_dict)))
                for p in sub.params
            ]
            u_sub = gate_unitary(sub.name, sub_params)
            U = embed_gate_unitary(u_sub, sub.qubits, n_q) @ U
        return U, None

    raise ValueError(f"不支持的自定义门类型: {dtype}")


def format_matrix_preview(U: np.ndarray) -> List[List[str]]:
    """Format a complex matrix cleanly for UI preview."""
    rows: List[List[str]] = []
    for r in range(U.shape[0]):
        row: List[str] = []
        for c in range(U.shape[1]):
            z = U[r, c]
            re_v = 0.0 if abs(z.real) < 1e-6 else z.real
            im_v = 0.0 if abs(z.imag) < 1e-6 else z.imag
            if im_v == 0.0:
                row.append(f"{re_v:.3f}")
            elif re_v == 0.0:
                row.append(f"{im_v:.3f}i")
            else:
                sign = "+" if im_v >= 0 else "-"
                row.append(f"{re_v:.2f}{sign}{abs(im_v):.2f}i")
        rows.append(row)
    return rows
