"""
macOS-Style Vector 2D Quantum Circuit Diagram Renderer for Matplotlib Canvases.
Enforces equal-aspect geometry so quantum gate boxes, control dots, CNOT targets,
and measurement meters never stretch or distort regardless of qubit count or window size.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
from matplotlib.axes import Axes

from ..core.circuit import GateInstruction, MeasureInstruction, NoiseInstruction, QuantumCircuit


def draw_quantum_circuit(
    circuit: QuantumCircuit,
    ax: Axes,
    title: Optional[str] = None,
) -> None:
    """
    Draw a clean, macOS-styled 2D quantum circuit schematic with locked proportional
    aspect ratio onto a Matplotlib Axes.
    """
    ax.clear()
    ax.set_facecolor("#FFFFFF")
    n_q = circuit.num_qubits

    # Assign column layers (ASAP scheduling) so parallel gates align vertically
    qubit_col = [0] * n_q
    placed_ops: List[Tuple[int, object]] = []

    for op in circuit.operations:
        if isinstance(op, GateInstruction):
            q_span = range(min(op.qubits), max(op.qubits) + 1)
            c = max(qubit_col[q] for q in q_span)
            placed_ops.append((c, op))
            for q in q_span:
                qubit_col[q] = c + 1
        elif isinstance(op, NoiseInstruction):
            q_span = range(min(op.channel.qubits), max(op.channel.qubits) + 1)
            c = max(qubit_col[q] for q in q_span)
            placed_ops.append((c, op))
            for q in q_span:
                qubit_col[q] = c + 1
        elif isinstance(op, MeasureInstruction):
            c = max(qubit_col[q] for q in op.qubits)
            placed_ops.append((c, op))
            for q in op.qubits:
                qubit_col[q] = c + 1

    total_cols = max(max(qubit_col, default=1), 1)
    wire_spacing = 1.05  # Vertical distance between adjacent qubit wires

    # Coordinate convention: qubit q is at y = -q * wire_spacing, column c is at x = c + 1
    x_start = 0.25
    x_end = max(total_cols + 0.75, 4.5)

    for q in range(n_q):
        y = -float(q) * wire_spacing
        ax.plot([x_start, x_end], [y, y], color="#C7C7CC", lw=1.6, zorder=1, solid_capstyle="round")
        ax.text(
            x_start - 0.15,
            y,
            f"q{q}  |0>",
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#1D1D1F",
        )

    for col, op in placed_ops:
        x = float(col) + 1.0
        if isinstance(op, GateInstruction):
            _draw_gate_op(ax, x, op, wire_spacing)
        elif isinstance(op, NoiseInstruction):
            _draw_noise_op(ax, x, op, wire_spacing)
        elif isinstance(op, MeasureInstruction):
            _draw_measure_op(ax, x, op, wire_spacing)

    # Lock aspect ratio to 1:1 in data units so gates are always crisp squares/pills, never stretched!
    x_min_view = -0.85
    x_max_view = max(x_end + 0.35, 9.5)
    y_center = -0.5 * float(n_q - 1) * wire_spacing
    y_half_span = max(0.5 * float(n_q - 1) * wire_spacing + 0.65, 0.85)

    ax.set_xlim(x_min_view, x_max_view)
    ax.set_ylim(y_center - y_half_span, y_center + y_half_span)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if title:
        ax.set_title(
            title,
            fontsize=10.0,
            fontweight="bold",
            color="#1D1D1F",
            loc="left",
            pad=4,
        )


def _format_short_gate_text(op: GateInstruction) -> Tuple[str, str]:
    """Return (main_symbol, subtitle_param) for compact visual gate box rendering."""
    uname = op.name.upper()
    if not op.params:
        return uname, ""
    p_strs = []
    for p in op.params:
        ratio = p / math.pi
        if abs(ratio - 1.0) < 1e-3:
            p_strs.append("π")
        elif abs(ratio + 1.0) < 1e-3:
            p_strs.append("-π")
        elif abs(ratio - 0.5) < 1e-3:
            p_strs.append("π/2")
        elif abs(ratio + 0.5) < 1e-3:
            p_strs.append("-π/2")
        elif abs(ratio - 0.25) < 1e-3:
            p_strs.append("π/4")
        elif abs(ratio + 0.25) < 1e-3:
            p_strs.append("-π/4")
        else:
            p_strs.append(f"{p:.2f}")
    return uname, ",".join(p_strs)


def _draw_gate_op(ax: Axes, x: float, op: GateInstruction, wire_spacing: float) -> None:
    uname = op.name.upper()
    main_lbl, sub_lbl = _format_short_gate_text(op)

    if len(op.qubits) == 1:
        q = op.qubits[0]
        y = -float(q) * wire_spacing
        is_virtual = uname == "RZ" and (op.duration_ns is not None and op.duration_ns == 0.0)
        if is_virtual or uname in ("RZ", "Z", "S", "SDG", "T", "TDG"):
            face = "#ECFDF5"
            edge = "#34C759"
            text_col = "#065F46"
            ls = "--" if is_virtual else "-"
        elif uname in ("SX", "SXDG", "X", "Y", "RX", "RY", "RXY", "U3"):
            face = "#EFF6FF"
            edge = "#007AFF"
            text_col = "#0040DD"
            ls = "-"
        else:
            face = "#F0F9FF"
            edge = "#0284C7"
            text_col = "#0C4A6E"
            ls = "-"

        w = 0.78 if len(sub_lbl) >= 6 else 0.62
        h = 0.56
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
            linestyle=ls,
            zorder=3,
        )
        ax.add_patch(box)
        if sub_lbl:
            sub_fs = 6.5 if len(sub_lbl) >= 6 else 7.2
            ax.text(x, y + 0.10, main_lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=text_col, zorder=4)
            ax.text(x, y - 0.12, sub_lbl, ha="center", va="center", fontsize=sub_fs, color=text_col, zorder=4)
        else:
            ax.text(x, y, main_lbl, ha="center", va="center", fontsize=9.2, fontweight="bold", color=text_col, zorder=4)
        return

    if len(op.qubits) == 2:
        q0, q1 = op.qubits
        y0, y1 = -float(q0) * wire_spacing, -float(q1) * wire_spacing
        ax.plot([x, x], [y0, y1], color="#5856D6", lw=2.0, zorder=2)

        if uname in ("CNOT", "CX"):
            ax.scatter([x], [y0], s=48, color="#5856D6", zorder=4)
            circ = mpatches.Circle((x, y1), radius=0.19, facecolor="#F5F3FF", edgecolor="#5856D6", lw=1.7, zorder=3)
            ax.add_patch(circ)
            ax.plot([x - 0.19, x + 0.19], [y1, y1], color="#5856D6", lw=1.5, zorder=4)
            ax.plot([x, x], [y1 - 0.19, y1 + 0.19], color="#5856D6", lw=1.5, zorder=4)
            return

        if uname == "CZ":
            ax.scatter([x, x], [y0, y1], s=52, color="#5856D6", zorder=4)
            ymid = 0.5 * (y0 + y1)
            ax.text(
                x + 0.14,
                ymid,
                "CZ",
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#5856D6",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#F5F3FF", edgecolor="#C4B5FD", lw=0.9),
                zorder=5,
            )
            return

        if uname == "SWAP":
            for y_pt in (y0, y1):
                d = 0.13
                ax.plot([x - d, x + d], [y_pt - d, y_pt + d], color="#5856D6", lw=1.9, zorder=4)
                ax.plot([x - d, x + d], [y_pt + d, y_pt - d], color="#5856D6", lw=1.9, zorder=4)
            return

        for y_pt in (y0, y1):
            w, h = 0.62, 0.50
            box = mpatches.FancyBboxPatch(
                (x - w / 2, y_pt - h / 2),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.09",
                facecolor="#F5F3FF",
                edgecolor="#5856D6",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(box)
            txt = f"{main_lbl}" if not sub_lbl else f"{main_lbl}\n{sub_lbl}"
            ax.text(x, y_pt, txt, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#312E81", zorder=4)
        return

    if len(op.qubits) >= 3:
        ys = [-float(q) * wire_spacing for q in op.qubits]
        ax.plot([x, x], [min(ys), max(ys)], color="#5856D6", lw=2.0, zorder=2)
        if uname == "CCZ":
            ax.scatter([x] * len(ys), ys, s=50, color="#5856D6", zorder=4)
            ax.text(
                x + 0.14,
                sum(ys) / len(ys),
                "CCZ",
                ha="left",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color="#5856D6",
                bbox=dict(boxstyle="round,pad=0.13", facecolor="#F5F3FF", edgecolor="#C4B5FD", lw=0.9),
                zorder=5,
            )
        else:
            # Control dots on all but last qubit, target ⊕ on last qubit
            ax.scatter([x] * (len(ys) - 1), ys[:-1], s=48, color="#5856D6", zorder=4)
            yt = ys[-1]
            circ = mpatches.Circle((x, yt), radius=0.19, facecolor="#F5F3FF", edgecolor="#5856D6", lw=1.7, zorder=3)
            ax.add_patch(circ)
            ax.plot([x - 0.19, x + 0.19], [yt, yt], color="#5856D6", lw=1.5, zorder=4)
            ax.plot([x, x], [yt - 0.19, yt + 0.19], color="#5856D6", lw=1.5, zorder=4)


def _draw_noise_op(ax: Axes, x: float, op: NoiseInstruction, wire_spacing: float) -> None:
    for q in op.channel.qubits:
        y = -float(q) * wire_spacing
        w, h = 0.60, 0.46
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor="#FFFBEB",
            edgecolor="#FF9500",
            linewidth=1.4,
            linestyle=":",
            zorder=3,
        )
        ax.add_patch(box)
        short_k = op.channel.kind[:5].upper()
        ax.text(x, y, f"~{short_k}~", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#B45309", zorder=4)


def _draw_measure_op(ax: Axes, x: float, op: MeasureInstruction, wire_spacing: float) -> None:
    for q in op.qubits:
        y = -float(q) * wire_spacing
        w, h = 0.56, 0.52
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            facecolor="#F2F2F7",
            edgecolor="#636366",
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, "M", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1D1D1F", zorder=4)
