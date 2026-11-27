"""
macOS Sonoma / Xcode-Style Multi-Partition GUI Workbench for Quantum Circuit Solver & Optimizer.

Design Highlights:
  - Top Unified macOS Toolbar:
      * App Brand & Status Badge on the left
      * macOS Segmented Capsule Control in the center to switch the 4 functional workspaces
      * Solver Backend selector & Primary Action Buttons (▶ 运行求解, ⚡ 最优门综合, 📈 参数扫描) in the TOP-RIGHT corner
  - Left macOS Inspector Sidebar (#F5F5F7):
      * Clean rounded white cards without cluttered Windows spinbox arrows
      * Circuit Editor with pill gate palette & symbolic parameter table
      * Hardware & Noise physical parameter inspector
  - Golden-Ratio 2x2 Balanced Scientific Plot Layouts:
      * Equal-aspect 2D quantum circuit schematics (gates never stretch or distort)
      * Landscape 16:9 line plots paired with 1:1 square 3D Bloch sphere & density matrix heatmaps
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
from typing import Dict, List, Optional

# Ensure Conda Library/bin and Qt6/bin are on DLL search path on Windows
if sys.platform == "win32":
    for _sub in ("Library/bin", "Lib/site-packages/PyQt6/Qt6/bin"):
        _p = pathlib.Path(sys.prefix) / _sub
        if _p.exists():
            os.environ["PATH"] = str(_p) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(_p))
            except Exception:
                pass

os.environ.setdefault("MPMATH_NOGMPY", "1")

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.circuit import QuantumCircuit
from ..core.gates import NativeGateSet
from ..core.hardware import HardwareModel
from ..optimizer.compiler import OptimizationReport, QuantumCircuitOptimizer
from ..solver.base import EvolutionResult
from ..solver.unified_solver import QuantumSolver
from .circuit_drawer import draw_quantum_circuit
from .circuit_parser import (
    PRESET_CIRCUIT_TEMPLATES,
    extract_symbolic_parameters,
    parse_circuit_script,
)
from .sweep_engine import (
    HARDWARE_SWEEP_PARAMS,
    ParameterSweepEngine,
    Sweep1DResult,
    Sweep2DResult,
)

# Configure clean fonts for Matplotlib
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "PingFang SC", "SimHei", "Segoe UI", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#D1D1D6"
plt.rcParams["axes.linewidth"] = 0.9


MACOS_STYLESHEET = """
QMainWindow, QWidget#rootWidget {
    background-color: #F5F5F7;
    color: #1D1D1F;
    font-family: "Microsoft YaHei UI", "PingFang SC", "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 12.5px;
}

/* Top macOS Unified Toolbar */
QFrame#macToolbar {
    background-color: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
}

/* Segmented Capsule Bar */
QFrame#segmentedBar {
    background-color: #E5E5EA;
    border-radius: 9px;
    padding: 2px;
}
QPushButton.segBtn {
    background-color: transparent;
    border: none;
    border-radius: 7px;
    padding: 5px 12px;
    color: #515154;
    font-weight: 600;
    font-size: 12px;
}
QPushButton.segBtn:hover {
    color: #1D1D1F;
}
QPushButton.segBtn:checked {
    background-color: #FFFFFF;
    color: #007AFF;
    font-weight: bold;
}

/* macOS White Cards */
QFrame.macCard {
    background-color: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
}

QLabel.cardTitle {
    font-size: 13px;
    font-weight: bold;
    color: #1D1D1F;
    padding: 0px;
}
QLabel.cardSubtitle {
    font-size: 11.5px;
    color: #86868B;
}
QLabel.fieldLabel {
    font-size: 12px;
    color: #3A3A3C;
    font-weight: 500;
}
QLabel.badgeBlue {
    background-color: #EBF5FF;
    color: #007AFF;
    border: 1px solid #BFE0FF;
    border-radius: 6px;
    padding: 3px 8px;
    font-weight: bold;
    font-size: 11.5px;
}
QLabel.badgeGreen {
    background-color: #ECFDF5;
    color: #059669;
    border: 1px solid #A7F3D0;
    border-radius: 6px;
    padding: 3px 8px;
    font-weight: bold;
    font-size: 11.5px;
}
QLabel.platformBanner {
    background-color: #F2F2F7;
    color: #48484A;
    border: 1px solid #E5E5EA;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
}
QLabel.sectionHeader {
    font-size: 11px;
    font-weight: bold;
    color: #86868B;
    padding-top: 2px;
}
QFrame.paramChip {
    background-color: #F5F5F7;
    border: 1px solid #D1D1D6;
    border-radius: 7px;
}

/* Inputs & Dropdowns (Clean macOS borderless-arrow style) */
QPlainTextEdit {
    background-color: #FAFAFC;
    border: 1px solid #D1D1D6;
    border-radius: 8px;
    padding: 6px 8px;
    color: #1D1D1F;
    selection-background-color: #007AFF;
    selection-color: #FFFFFF;
}
QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1.5px solid #007AFF;
    background-color: #FFFFFF;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #F5F5F7;
    border: 1px solid #D1D1D6;
    border-radius: 7px;
    padding: 4px 8px;
    color: #1D1D1F;
    min-height: 20px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}

/* Quick Gate Pill Buttons */
QPushButton.gatePill {
    background-color: #F2F2F7;
    border: 1px solid #D1D1D6;
    border-radius: 6px;
    padding: 3px 7px;
    color: #1D1D1F;
    font-weight: 600;
    font-size: 11.5px;
}
QPushButton.gatePill:hover {
    background-color: #E5E5EA;
    border-color: #007AFF;
    color: #007AFF;
}

/* Standard Secondary Button */
QPushButton {
    background-color: #F2F2F7;
    border: 1px solid #D1D1D6;
    border-radius: 7px;
    padding: 5px 12px;
    color: #1D1D1F;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #E5E5EA;
}

/* Top-Right macOS Action Pill Buttons */
QPushButton#runBtn {
    background-color: #007AFF;
    border: 1px solid #0062CC;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#runBtn:hover {
    background-color: #0062CC;
}
QPushButton#optBtn {
    background-color: #34C759;
    border: 1px solid #28A745;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 6px 14px;
    font-size: 12.5px;
    font-weight: bold;
}
QPushButton#optBtn:hover {
    background-color: #28A745;
}
QPushButton#sweepTopBtn {
    background-color: #5856D6;
    border: 1px solid #4644B5;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 6px 14px;
    font-size: 12.5px;
    font-weight: bold;
}
QPushButton#sweepTopBtn:hover {
    background-color: #4644B5;
}

/* Tables */
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #E5E5EA;
    border-radius: 8px;
    gridline-color: #F2F2F7;
    selection-background-color: #EBF5FF;
    selection-color: #007AFF;
}
QHeaderView::section {
    background-color: #F9F9FB;
    color: #636366;
    font-weight: 600;
    font-size: 11.5px;
    border: none;
    border-bottom: 1px solid #E5E5EA;
    border-right: 1px solid #F2F2F7;
    padding: 5px 6px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 5px;
    background: #E5E5EA;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #007AFF;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1.5px solid #007AFF;
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 8px;
}

QSplitter::handle {
    background-color: transparent;
    width: 6px;
    height: 6px;
}

QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E5EA;
    color: #636366;
    font-size: 12px;
}
"""


def _make_card() -> QFrame:
    card = QFrame()
    card.setProperty("class", "macCard")
    return card


def _clean_spinbox(spin: QAbstractSpinBox) -> None:
    """Remove cluttered Windows up/down arrows for a clean macOS Inspector field."""
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)


class QuantumWorkbenchWindow(QMainWindow):
    """
    macOS-Style Quantum Circuit Solver & Hardware-Aware Optimal Synthesizer Workbench.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantum Studio — 多后端量子线路演化求解与最优门综合工作台")
        self.resize(1560, 950)
        self.setStyleSheet(MACOS_STYLESHEET)

        # Internal state
        self.current_variables: Dict[str, float] = {}
        self.param_spinboxes: Dict[str, QDoubleSpinBox] = {}
        self.current_circuit: Optional[QuantumCircuit] = None
        self.current_opt_report: Optional[OptimizationReport] = None
        self.current_results: Dict[str, EvolutionResult] = {}
        self.primary_result: Optional[EvolutionResult] = None
        self.sweep_engine = ParameterSweepEngine()

        self._build_ui()

        # Load Template 4 by default (2Q QFT + SWAP optimization & multi-backend showcase)
        self.template_combo.setCurrentIndex(3)
        self._on_template_selected(3)
        self.run_simulation()
        self.run_optimal_synthesis(silent=True)
        self.run_parameter_sweep()

    # =========================================================================
    # Top-Level macOS Layout Construction
    # =========================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("rootWidget")
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        # 1. Top Unified macOS Toolbar (Title Left | Segmented Nav Center | Run Controls Top-Right)
        root_layout.addWidget(self._build_top_macos_toolbar())

        # 2. Main Body Splitter (Left macOS Inspector Sidebar | Right Stacked Workspace)
        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(body_splitter, stretch=1)

        left_sidebar = self._build_left_inspector_sidebar()
        self.workspace_stack = QStackedWidget()

        self.page_circuit = self._build_page_circuit_and_synthesis()
        self.page_evolution = self._build_page_evolution_and_bloch()
        self.page_populations = self._build_page_state_populations()
        self.page_sweep = self._build_page_parameter_sweep()

        self.workspace_stack.addWidget(self.page_circuit)
        self.workspace_stack.addWidget(self.page_evolution)
        self.workspace_stack.addWidget(self.page_populations)
        self.workspace_stack.addWidget(self.page_sweep)

        body_splitter.addWidget(left_sidebar)
        body_splitter.addWidget(self.workspace_stack)
        body_splitter.setStretchFactor(0, 28)
        body_splitter.setStretchFactor(1, 72)
        body_splitter.setSizes([430, 1100])

        # 3. Bottom Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(160)
        self.progress_bar.setMaximumHeight(12)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _build_top_macos_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("macToolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        # Left: Traffic-light style dots + App Title
        dots_lbl = QLabel(
            "<span style='color:#FF5F56;'>●</span> "
            "<span style='color:#FFBD2E;'>●</span> "
            "<span style='color:#27C93F;'>●</span>"
        )
        dots_lbl.setStyleSheet("font-size: 14px;")
        brand_lbl = QLabel("⚛ Quantum Studio")
        brand_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #1D1D1F;")
        layout.addWidget(dots_lbl)
        layout.addWidget(brand_lbl)

        layout.addStretch(1)

        # Center: macOS Segmented Capsule Navigation
        seg_frame = QFrame()
        seg_frame.setObjectName("segmentedBar")
        seg_layout = QHBoxLayout(seg_frame)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(2)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        nav_items = [
            "线路与最优综合",
            "逐步演化 & Bloch",
            "态布居与测量",
            "参数扫描 (1D/2D)",
        ]
        for idx, text in enumerate(nav_items):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("class", "segBtn")
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked=False, i=idx: self._switch_workspace_page(i))
            self.nav_group.addButton(btn, idx)
            seg_layout.addWidget(btn)

        layout.addWidget(seg_frame)
        layout.addStretch(1)

        # Top-Right: Solver Backend Selector + Primary Run & Optimize Buttons
        backend_lbl = QLabel("求解引擎:")
        backend_lbl.setProperty("class", "fieldLabel")
        layout.addWidget(backend_lbl)

        self.backend_combo = QComboBox()
        self.backend_combo.setMinimumWidth(165)
        self.backend_combo.setMaximumWidth(215)
        self.backend_combo.addItems(
            [
                "全后端对比 (QuTiP+Kraus+Stim)",
                "QuTiP (脉冲/林德布拉德主方程)",
                "Stim (稳定子/泡利框架采样)",
                "Stim + QuTiP (物理门标定桥接)",
                "Kraus (通用门级密度矩阵)",
            ]
        )
        layout.addWidget(self.backend_combo)

        self.opt_btn = QPushButton("⚡ 最优综合")
        self.opt_btn.setObjectName("optBtn")
        self.opt_btn.setToolTip("根据左侧设置的实验原生门集，计算当前线路的最优量子门组合")
        self.opt_btn.clicked.connect(lambda: self.run_optimal_synthesis(silent=False))
        layout.addWidget(self.opt_btn)

        self.sweep_top_btn = QPushButton("📈 参数扫描")
        self.sweep_top_btn.setObjectName("sweepTopBtn")
        self.sweep_top_btn.setToolTip("执行参数扫描并跳转至参数扫描实验室")
        self.sweep_top_btn.clicked.connect(self._jump_to_and_run_sweep)
        layout.addWidget(self.sweep_top_btn)

        self.run_btn = QPushButton("▶ 运行求解")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setToolTip("运行量子线路演化、态布居与含噪测量求解")
        self.run_btn.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_btn)

        return toolbar

    def _switch_workspace_page(self, index: int) -> None:
        btn = self.nav_group.button(index)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self.workspace_stack.setCurrentIndex(index)

    # =========================================================================
    # Left macOS Inspector Sidebar (Optimized Proportion & Neutral Atom Support)
    # =========================================================================

    def _build_left_inspector_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(405)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(10)

        # ---- Card 1: Circuit Editor & Inline Symbolic Parameter Chips (Expands Vertically) ----
        card_editor = _make_card()
        card_editor.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        ce_lay = QVBoxLayout(card_editor)
        ce_lay.setContentsMargins(12, 10, 12, 10)
        ce_lay.setSpacing(7)

        hdr1 = QHBoxLayout()
        t1 = QLabel("量子线路编写 (Circuit Studio)")
        t1.setProperty("class", "cardTitle")
        hdr1.addWidget(t1)
        hdr1.addStretch()
        ce_lay.addLayout(hdr1)

        # Preset Template Selector
        tpl_row = QHBoxLayout()
        tpl_row.setSpacing(6)
        tpl_lbl = QLabel("预设模板:")
        tpl_lbl.setProperty("class", "fieldLabel")
        tpl_lbl.setFixedWidth(56)
        tpl_row.addWidget(tpl_lbl)
        self.template_combo = QComboBox()
        for tpl_name in PRESET_CIRCUIT_TEMPLATES.keys():
            self.template_combo.addItem(tpl_name)
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        tpl_row.addWidget(self.template_combo, stretch=1)
        ce_lay.addLayout(tpl_row)

        # 2-Row Quick Gate Insertion Pill Palette (including RXY & CCZ for Neutral Atom)
        gate_grid = QGridLayout()
        gate_grid.setHorizontalSpacing(4)
        gate_grid.setVerticalSpacing(4)
        quick_gates = [
            ("H", "H 0"),
            ("X", "X 0"),
            ("SX", "SX 0"),
            ("RX(θ)", "RX(theta) 0"),
            ("RY(θ)", "RY(theta) 0"),
            ("RZ(φ)", "RZ(phi) 0"),
            ("RXY", "RXY(theta, 0) 0"),
            ("CNOT", "CNOT 0 1"),
            ("CZ", "CZ 0 1"),
            ("iSWAP", "ISWAP 0 1"),
            ("CCZ", "CCZ 0 1 2"),
            ("+测量", "MEASURE 0 1"),
        ]
        for idx, (btn_lbl, snippet) in enumerate(quick_gates):
            btn = QPushButton(btn_lbl)
            btn.setProperty("class", "gatePill")
            btn.setToolTip(f"点击插入指令: {snippet}")
            btn.clicked.connect(lambda _checked=False, s=snippet: self._insert_gate_snippet(s))
            gate_grid.addWidget(btn, idx // 6, idx % 6)
        ce_lay.addLayout(gate_grid)

        # Spacious Monospace Code Editor
        self.script_editor = QPlainTextEdit()
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.script_editor.setFont(mono)
        self.script_editor.setMinimumHeight(210)
        self.script_editor.textChanged.connect(self._on_script_text_changed)
        ce_lay.addWidget(self.script_editor, stretch=1)

        # Compact Inline Symbolic Parameter Chips (Replaces bulky QTableWidget)
        param_bar_hdr = QHBoxLayout()
        p_lbl = QLabel("线路符号参数 (实时联动求解与扫描):")
        p_lbl.setProperty("class", "sectionHeader")
        param_bar_hdr.addWidget(p_lbl)
        param_bar_hdr.addStretch()
        ce_lay.addLayout(param_bar_hdr)

        self.param_chips_container = QWidget()
        self.param_chips_grid = QGridLayout(self.param_chips_container)
        self.param_chips_grid.setContentsMargins(0, 0, 0, 0)
        self.param_chips_grid.setHorizontalSpacing(6)
        self.param_chips_grid.setVerticalSpacing(4)
        ce_lay.addWidget(self.param_chips_container)

        layout.addWidget(card_editor, stretch=1)

        # ---- Card 2: Hardware & Noise Inspector (Content-Fitted, Zero Wasted Blank Space) ----
        card_hw = _make_card()
        card_hw.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        hw_lay = QVBoxLayout(card_hw)
        hw_lay.setContentsMargins(12, 10, 12, 10)
        hw_lay.setSpacing(6)

        t2 = QLabel("实验硬件与物理噪声参数 (Hardware Inspector)")
        t2.setProperty("class", "cardTitle")
        hw_lay.addWidget(t2)

        plat_row = QHBoxLayout()
        plat_row.setSpacing(6)
        lbl_gset = QLabel("物理平台:")
        lbl_gset.setProperty("class", "fieldLabel")
        lbl_gset.setFixedWidth(56)
        plat_row.addWidget(lbl_gset)
        self.gateset_combo = QComboBox()
        self.gateset_combo.addItems(
            [
                "超导 CZ 体系 {Virtual-RZ, SX, X, CZ}",
                "中性原子里德堡 {RXY, Virtual-RZ, CZ}",
                "超导 iSWAP 体系 {Virtual-RZ, SX, iSWAP}",
                "超导 CNOT 体系 {Virtual-RZ, SX, CNOT}",
                "离子阱 MS 体系 {RXY, Virtual-RZ, MS}",
            ]
        )
        self.gateset_combo.currentIndexChanged.connect(self._on_platform_preset_changed)
        plat_row.addWidget(self.gateset_combo, stretch=1)
        hw_lay.addLayout(plat_row)

        self.platform_info_badge = QLabel(
            "⚡ 超导 Transmon · 零耗时 Virtual-Z + 微波 SX(20ns) + 可调耦合器 CZ(40ns)"
        )
        self.platform_info_badge.setProperty("class", "platformBanner")
        self.platform_info_badge.setWordWrap(True)
        hw_lay.addWidget(self.platform_info_badge)

        # Section 1: Coherence & Pulse Timing
        sec1 = QLabel("相干时间与门脉冲时序")
        sec1.setProperty("class", "sectionHeader")
        hw_lay.addWidget(sec1)

        hw_grid = QGridLayout()
        hw_grid.setContentsMargins(0, 0, 0, 0)
        hw_grid.setHorizontalSpacing(8)
        hw_grid.setVerticalSpacing(5)
        hw_grid.setColumnStretch(1, 1)
        hw_grid.setColumnStretch(3, 1)

        # Row 0: T1 & T2*
        hw_grid.addWidget(self._field_lbl("T1 弛豫:", 62), 0, 0)
        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(0.1, 100000.0)
        self.t1_spin.setDecimals(1)
        self.t1_spin.setSuffix(" μs")
        self.t1_spin.setValue(35.0)
        _clean_spinbox(self.t1_spin)
        hw_grid.addWidget(self.t1_spin, 0, 1)

        hw_grid.addWidget(self._field_lbl("T2* 退相干:", 68), 0, 2)
        self.t2_spin = QDoubleSpinBox()
        self.t2_spin.setRange(0.1, 100000.0)
        self.t2_spin.setDecimals(1)
        self.t2_spin.setSuffix(" μs")
        self.t2_spin.setValue(25.0)
        _clean_spinbox(self.t2_spin)
        hw_grid.addWidget(self.t2_spin, 0, 3)

        # Row 1: 1Q & 2Q Gate Duration
        hw_grid.addWidget(self._field_lbl("1Q 门时长:", 62), 1, 0)
        self.t1q_spin = QDoubleSpinBox()
        self.t1q_spin.setRange(1.0, 500000.0)
        self.t1q_spin.setDecimals(1)
        self.t1q_spin.setSuffix(" ns")
        self.t1q_spin.setValue(20.0)
        _clean_spinbox(self.t1q_spin)
        hw_grid.addWidget(self.t1q_spin, 1, 1)

        hw_grid.addWidget(self._field_lbl("2Q 门时长:", 68), 1, 2)
        self.t2q_spin = QDoubleSpinBox()
        self.t2q_spin.setRange(1.0, 500000.0)
        self.t2q_spin.setDecimals(1)
        self.t2q_spin.setSuffix(" ns")
        self.t2q_spin.setValue(40.0)
        _clean_spinbox(self.t2q_spin)
        hw_grid.addWidget(self.t2q_spin, 1, 3)

        hw_lay.addLayout(hw_grid)

        # Section 2: Gate Errors, Crosstalk & Readout
        sec2 = QLabel("控制误差、串扰与测量拓扑")
        sec2.setProperty("class", "sectionHeader")
        hw_lay.addWidget(sec2)

        err_grid = QGridLayout()
        err_grid.setContentsMargins(0, 0, 0, 0)
        err_grid.setHorizontalSpacing(8)
        err_grid.setVerticalSpacing(5)
        err_grid.setColumnStretch(1, 1)
        err_grid.setColumnStretch(3, 1)

        # Row 0: 1Q & 2Q Depolarizing Error
        err_grid.addWidget(self._field_lbl("1Q 退极化:", 62), 0, 0)
        self.dep1_spin = QDoubleSpinBox()
        self.dep1_spin.setRange(0.0, 0.5)
        self.dep1_spin.setDecimals(4)
        self.dep1_spin.setValue(0.0015)
        _clean_spinbox(self.dep1_spin)
        err_grid.addWidget(self.dep1_spin, 0, 1)

        err_grid.addWidget(self._field_lbl("2Q 退极化:", 68), 0, 2)
        self.dep2_spin = QDoubleSpinBox()
        self.dep2_spin.setRange(0.0, 0.5)
        self.dep2_spin.setDecimals(4)
        self.dep2_spin.setValue(0.0120)
        _clean_spinbox(self.dep2_spin)
        err_grid.addWidget(self.dep2_spin, 0, 3)

        # Row 1: ZZ Crosstalk & Readout Error
        err_grid.addWidget(self._field_lbl("ZZ 串扰:", 62), 1, 0)
        self.zz_spin = QDoubleSpinBox()
        self.zz_spin.setRange(0.0, 5000.0)
        self.zz_spin.setDecimals(1)
        self.zz_spin.setSuffix(" kHz")
        self.zz_spin.setValue(50.0)
        _clean_spinbox(self.zz_spin)
        err_grid.addWidget(self.zz_spin, 1, 1)

        err_grid.addWidget(self._field_lbl("读出误差:", 68), 1, 2)
        self.ro_spin = QDoubleSpinBox()
        self.ro_spin.setRange(0.0, 0.4)
        self.ro_spin.setDecimals(4)
        self.ro_spin.setValue(0.0100)
        _clean_spinbox(self.ro_spin)
        err_grid.addWidget(self.ro_spin, 1, 3)

        # Row 2: Shots & Coupling Topology
        err_grid.addWidget(self._field_lbl("采样次数:", 62), 2, 0)
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(100, 1000000)
        self.shots_spin.setValue(20000)
        _clean_spinbox(self.shots_spin)
        err_grid.addWidget(self.shots_spin, 2, 1)

        err_grid.addWidget(self._field_lbl("耦合拓扑:", 68), 2, 2)
        self.topo_combo = QComboBox()
        self.topo_combo.addItems(["全连接 / 可重构阵列", "一维链 (Linear Chain)"])
        err_grid.addWidget(self.topo_combo, 2, 3)

        hw_lay.addLayout(err_grid)
        layout.addWidget(card_hw, stretch=0)
        return sidebar

    @staticmethod
    def _field_lbl(text: str, width: Optional[int] = None) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "fieldLabel")
        if width is not None:
            lbl.setFixedWidth(width)
        return lbl

    # =========================================================================
    # Workspace Page 1: Circuit Schematics & Optimal Gate Synthesis
    # =========================================================================

    def _build_page_circuit_and_synthesis(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        # Top Card: Equal-Aspect 2D Circuit Schematics (Original vs Optimal Hardware-Native)
        card_canvas = _make_card()
        cc_lay = QVBoxLayout(card_canvas)
        cc_lay.setContentsMargins(14, 10, 14, 10)
        cc_lay.setSpacing(4)

        hdr = QHBoxLayout()
        t_lbl = QLabel("量子线路图与最优硬件原生门综合对比")
        t_lbl.setProperty("class", "cardTitle")
        hdr.addWidget(t_lbl)
        sub_lbl = QLabel("（蓝框: 物理单比特脉冲  |  虚线绿框: 零耗时 Virtual-Z  |  紫框: 原生两比特纠缠门）")
        sub_lbl.setProperty("class", "cardSubtitle")
        hdr.addWidget(sub_lbl)
        hdr.addStretch()

        self.badge_unitary_fid = QLabel("酉等价保真度 F = 1.000000")
        self.badge_unitary_fid.setProperty("class", "badgeBlue")
        hdr.addWidget(self.badge_unitary_fid)

        self.badge_noisy_fid = QLabel("QuTiP 含噪保真度: —")
        self.badge_noisy_fid.setProperty("class", "badgeGreen")
        hdr.addWidget(self.badge_noisy_fid)

        adopt_btn = QPushButton("回填最优门序列至编辑器")
        adopt_btn.clicked.connect(self._adopt_optimized_circuit_to_editor)
        hdr.addWidget(adopt_btn)
        cc_lay.addLayout(hdr)

        self.fig_circuits = Figure(figsize=(10.5, 4.4), dpi=110, facecolor="#FFFFFF")
        self.canvas_circuits = FigureCanvasQTAgg(self.fig_circuits)
        cc_lay.addWidget(self.canvas_circuits, stretch=1)
        splitter.addWidget(card_canvas)

        # Bottom Row: 2 Side-by-Side Cards (Left: Clean Structured Metrics Table | Right: Stim & Native Code)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        card_metrics = _make_card()
        cm_lay = QVBoxLayout(card_metrics)
        cm_lay.setContentsMargins(14, 10, 14, 10)
        cm_title = QLabel("硬件感知最优门综合指标对比表 (Euler + KAK Cartan 分解)")
        cm_title.setProperty("class", "cardTitle")
        cm_lay.addWidget(cm_title)

        self.metrics_table = QTableWidget(6, 4)
        self.metrics_table.setHorizontalHeaderLabels(["评估维度", "原始输入线路", "最优原生门线路", "优化收益 / 状态"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        cm_lay.addWidget(self.metrics_table, stretch=1)
        bottom_splitter.addWidget(card_metrics)

        card_code = _make_card()
        cd_lay = QVBoxLayout(card_code)
        cd_lay.setContentsMargins(14, 10, 14, 10)
        cd_title = QLabel("编译输出：最优原生门序列 & Stim 含噪采样指令")
        cd_title.setProperty("class", "cardTitle")
        cd_lay.addWidget(cd_title)

        self.stim_code_text = QPlainTextEdit()
        self.stim_code_text.setReadOnly(True)
        self.stim_code_text.setFont(QFont("Consolas", 9))
        cd_lay.addWidget(self.stim_code_text, stretch=1)
        bottom_splitter.addWidget(card_code)

        bottom_splitter.setSizes([560, 560])
        splitter.addWidget(bottom_splitter)
        splitter.setSizes([510, 310])
        return page

    # =========================================================================
    # Workspace Page 2: Step-by-Step Evolution & 3D Bloch Dynamics (2x2 Balanced)
    # =========================================================================

    def _build_page_evolution_and_bloch(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)

        # Top Interactive Step Inspector Bar
        bar_card = _make_card()
        b_lay = QHBoxLayout(bar_card)
        b_lay.setContentsMargins(14, 8, 14, 8)
        b_lay.setSpacing(12)

        b_lay.addWidget(self._field_lbl("观测比特:"))
        self.evo_qubit_combo = QComboBox()
        self.evo_qubit_combo.setMinimumWidth(75)
        self.evo_qubit_combo.addItems(["q0", "q1"])
        self.evo_qubit_combo.currentIndexChanged.connect(self._refresh_evolution_plots)
        b_lay.addWidget(self.evo_qubit_combo)

        b_lay.addWidget(self._field_lbl("演化步进检查器 (拖动滑块或点击右下表查看任意时刻 ρ_k 与 Bloch 指针):"))
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setMinimum(0)
        self.step_slider.setMaximum(1)
        self.step_slider.valueChanged.connect(self._on_step_slider_changed)
        b_lay.addWidget(self.step_slider, stretch=1)

        self.step_info_label = QLabel("Step 0: INIT")
        self.step_info_label.setProperty("class", "badgeBlue")
        b_lay.addWidget(self.step_info_label)
        layout.addWidget(bar_card)

        # 2x2 Balanced Scientific Layout:
        #   Top Card: 2 wide landscape plots (Left: Continuous Pulse Bloch Trajectory | Right: Fidelity & Purity Decay)
        #   Bottom Row: Left Card (1:1 3D Bloch Sphere + 1:1 Density Matrix Heatmap) | Right Card (Step-by-Step Table)
        v_split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(v_split, stretch=1)

        card_top_plots = _make_card()
        ctp_lay = QVBoxLayout(card_top_plots)
        ctp_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_evo_top = Figure(figsize=(10.5, 3.3), dpi=110, facecolor="#FFFFFF")
        self.canvas_evo_top = FigureCanvasQTAgg(self.fig_evo_top)
        ctp_lay.addWidget(self.canvas_evo_top)
        v_split.addWidget(card_top_plots)

        bot_split = QSplitter(Qt.Orientation.Horizontal)

        card_3d_rho = _make_card()
        c3r_lay = QVBoxLayout(card_3d_rho)
        c3r_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_evo_bot = Figure(figsize=(5.4, 3.4), dpi=110, facecolor="#FFFFFF")
        self.canvas_evo_bot = FigureCanvasQTAgg(self.fig_evo_bot)
        c3r_lay.addWidget(self.canvas_evo_bot)
        bot_split.addWidget(card_3d_rho)

        card_evo_tbl = _make_card()
        cet_lay = QVBoxLayout(card_evo_tbl)
        cet_lay.setContentsMargins(12, 10, 12, 10)
        tbl_title = QLabel("逐步演化轨迹与稳定子表 (点击任意行可定位至该步骤)")
        tbl_title.setProperty("class", "cardTitle")
        cet_lay.addWidget(tbl_title)

        self.evo_table = QTableWidget(0, 6)
        self.evo_table.setHorizontalHeaderLabels(
            ["步骤", "量子门 / 噪声", "时间 (ns)", "保真度 F", "纯度 Tr(ρ²)", "Bloch 矢量 (<X>,<Y>,<Z>) & 稳定子"]
        )
        self.evo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.evo_table.setColumnWidth(0, 48)
        self.evo_table.setColumnWidth(1, 120)
        self.evo_table.setColumnWidth(2, 78)
        self.evo_table.setColumnWidth(3, 82)
        self.evo_table.setColumnWidth(4, 82)
        self.evo_table.horizontalHeader().setStretchLastSection(True)
        self.evo_table.verticalHeader().setVisible(False)
        self.evo_table.cellClicked.connect(lambda row, _col: self.step_slider.setValue(row))
        cet_lay.addWidget(self.evo_table, stretch=1)
        bot_split.addWidget(card_evo_tbl)

        bot_split.setSizes([540, 580])
        v_split.addWidget(bot_split)
        v_split.setSizes([380, 390])
        return page

    # =========================================================================
    # Workspace Page 3: State Populations & Multi-Backend Measurements (2x2 Balanced)
    # =========================================================================

    def _build_page_state_populations(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)

        v_split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(v_split, stretch=1)

        # Top Card: 2 Wide Landscape Plots (Left: Step-by-Step Population Curves | Right: Multi-Backend Final Bar Chart)
        card_pop_top = _make_card()
        cpt_lay = QVBoxLayout(card_pop_top)
        cpt_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_pop_top = Figure(figsize=(10.5, 3.6), dpi=110, facecolor="#FFFFFF")
        self.canvas_pop_top = FigureCanvasQTAgg(self.fig_pop_top)
        cpt_lay.addWidget(self.canvas_pop_top)
        v_split.addWidget(card_pop_top)

        # Bottom Row: Left Card (Spatio-Temporal Population Heatmap) | Right Card (Multi-Backend Comparison Table)
        bot_split = QSplitter(Qt.Orientation.Horizontal)

        card_pop_heat = _make_card()
        cph_lay = QVBoxLayout(card_pop_heat)
        cph_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_pop_bot = Figure(figsize=(5.2, 3.3), dpi=110, facecolor="#FFFFFF")
        self.canvas_pop_bot = FigureCanvasQTAgg(self.fig_pop_bot)
        cph_lay.addWidget(self.canvas_pop_bot)
        bot_split.addWidget(card_pop_heat)

        card_pop_tbl = _make_card()
        ctb_lay = QVBoxLayout(card_pop_tbl)
        ctb_lay.setContentsMargins(12, 10, 12, 10)
        pt_title = QLabel("最终含噪测量态布居与多后端交叉对比表")
        pt_title.setProperty("class", "cardTitle")
        ctb_lay.addWidget(pt_title)

        self.pop_table = QTableWidget(0, 6)
        self.pop_table.setHorizontalHeaderLabels(
            ["计算基态", "QuTiP 主方程", "Kraus 密度矩阵", "Stim 泡利采样", "Stim+QuTiP 桥接", "采样计数 (Shots)"]
        )
        self.pop_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pop_table.verticalHeader().setVisible(False)
        ctb_lay.addWidget(self.pop_table, stretch=1)
        bot_split.addWidget(card_pop_tbl)

        bot_split.setSizes([520, 600])
        v_split.addWidget(bot_split)
        v_split.setSizes([410, 380])
        return page

    # =========================================================================
    # Workspace Page 4: Parameter Sweep Studio (2x2 Balanced Layout)
    # =========================================================================

    def _build_page_parameter_sweep(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)

        # Top Compact macOS Sweep Inspector Card
        ctrl_card = _make_card()
        sc_grid = QGridLayout(ctrl_card)
        sc_grid.setContentsMargins(14, 10, 14, 10)
        sc_grid.setHorizontalSpacing(10)
        sc_grid.setVerticalSpacing(6)

        sc_grid.addWidget(self._field_lbl("扫描模式:"), 0, 0)
        self.sweep_mode_combo = QComboBox()
        self.sweep_mode_combo.addItems(["一维参数扫描 (1D Sweep)", "二维双参数相图 (2D Heatmap)"])
        self.sweep_mode_combo.currentIndexChanged.connect(self._on_sweep_mode_changed)
        sc_grid.addWidget(self.sweep_mode_combo, 0, 1)

        sc_grid.addWidget(self._field_lbl("扫描变量 X:"), 0, 2)
        self.sweep_param_x = QComboBox()
        self.sweep_param_x.setMinimumWidth(165)
        self.sweep_param_x.currentIndexChanged.connect(self._on_sweep_param_x_changed)
        sc_grid.addWidget(self.sweep_param_x, 0, 3)

        sc_grid.addWidget(self._field_lbl("范围 [起, 止, 点数]:"), 0, 4)
        x_range_box = QHBoxLayout()
        self.sweep_x_start = QDoubleSpinBox()
        self.sweep_x_start.setRange(-10000.0, 100000.0)
        self.sweep_x_start.setDecimals(3)
        self.sweep_x_start.setValue(0.0)
        _clean_spinbox(self.sweep_x_start)
        self.sweep_x_stop = QDoubleSpinBox()
        self.sweep_x_stop.setRange(-10000.0, 100000.0)
        self.sweep_x_stop.setDecimals(3)
        self.sweep_x_stop.setValue(round(2.0 * math.pi, 3))
        _clean_spinbox(self.sweep_x_stop)
        self.sweep_x_pts = QSpinBox()
        self.sweep_x_pts.setRange(3, 200)
        self.sweep_x_pts.setValue(25)
        self.sweep_x_pts.setMinimumWidth(56)
        _clean_spinbox(self.sweep_x_pts)
        x_range_box.addWidget(self.sweep_x_start)
        x_range_box.addWidget(self.sweep_x_stop)
        x_range_box.addWidget(self.sweep_x_pts)
        sc_grid.addLayout(x_range_box, 0, 5)

        sc_grid.addWidget(self._field_lbl("后端:"), 0, 6)
        self.sweep_backend_combo = QComboBox()
        self.sweep_backend_combo.addItems(["qutip", "kraus", "stim", "stim_qutip"])
        sc_grid.addWidget(self.sweep_backend_combo, 0, 7)

        run_sweep_btn = QPushButton("▶ 开始扫描")
        run_sweep_btn.setObjectName("sweepTopBtn")
        run_sweep_btn.clicked.connect(self.run_parameter_sweep)
        sc_grid.addWidget(run_sweep_btn, 0, 8)

        # Optional Row 1 for 2D secondary parameter Y
        self.lbl_param_y = self._field_lbl("第二变量 Y:")
        sc_grid.addWidget(self.lbl_param_y, 1, 2)
        self.sweep_param_y = QComboBox()
        sc_grid.addWidget(self.sweep_param_y, 1, 3)

        self.lbl_y_range = self._field_lbl("Y 范围 [起, 止, 点数]:")
        sc_grid.addWidget(self.lbl_y_range, 1, 4)
        y_box = QHBoxLayout()
        self.sweep_y_start = QDoubleSpinBox()
        self.sweep_y_start.setRange(-10000.0, 100000.0)
        self.sweep_y_start.setDecimals(3)
        self.sweep_y_start.setValue(0.0)
        _clean_spinbox(self.sweep_y_start)
        self.sweep_y_stop = QDoubleSpinBox()
        self.sweep_y_stop.setRange(-10000.0, 100000.0)
        self.sweep_y_stop.setDecimals(3)
        self.sweep_y_stop.setValue(3.142)
        _clean_spinbox(self.sweep_y_stop)
        self.sweep_y_pts = QSpinBox()
        self.sweep_y_pts.setRange(3, 50)
        self.sweep_y_pts.setValue(12)
        self.sweep_y_pts.setMinimumWidth(56)
        _clean_spinbox(self.sweep_y_pts)
        y_box.addWidget(self.sweep_y_start)
        y_box.addWidget(self.sweep_y_stop)
        y_box.addWidget(self.sweep_y_pts)
        sc_grid.addLayout(y_box, 1, 5)

        layout.addWidget(ctrl_card)

        # 2x2 Balanced Grid for Sweep Plots & Table
        v_split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(v_split, stretch=1)

        card_sw_top = _make_card()
        cst_lay = QVBoxLayout(card_sw_top)
        cst_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_sweep_top = Figure(figsize=(10.5, 3.5), dpi=110, facecolor="#FFFFFF")
        self.canvas_sweep_top = FigureCanvasQTAgg(self.fig_sweep_top)
        cst_lay.addWidget(self.canvas_sweep_top)
        v_split.addWidget(card_sw_top)

        bot_split = QSplitter(Qt.Orientation.Horizontal)

        card_sw_bot_left = _make_card()
        csbl_lay = QVBoxLayout(card_sw_bot_left)
        csbl_lay.setContentsMargins(10, 8, 10, 8)
        self.fig_sweep_bot = Figure(figsize=(5.2, 3.2), dpi=110, facecolor="#FFFFFF")
        self.canvas_sweep_bot = FigureCanvasQTAgg(self.fig_sweep_bot)
        csbl_lay.addWidget(self.canvas_sweep_bot)
        bot_split.addWidget(card_sw_bot_left)

        card_sw_tbl = _make_card()
        cstb_lay = QVBoxLayout(card_sw_tbl)
        cstb_lay.setContentsMargins(12, 10, 12, 10)
        st_title = QLabel("参数扫描采样点数值详情表")
        st_title.setProperty("class", "cardTitle")
        cstb_lay.addWidget(st_title)

        self.sweep_table = QTableWidget(0, 6)
        self.sweep_table.setHorizontalHeaderLabels(
            ["点序号", "参数值", "保真度 F", "纯度 Tr(ρ²)", "<Z_0> 期望值", "各计算基态布居分布"]
        )
        self.sweep_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sweep_table.verticalHeader().setVisible(False)
        cstb_lay.addWidget(self.sweep_table, stretch=1)
        bot_split.addWidget(card_sw_tbl)

        bot_split.setSizes([520, 600])
        v_split.addWidget(bot_split)
        v_split.setSizes([400, 350])

        self._on_sweep_mode_changed(0)
        return page

    # =========================================================================
    # Event Handlers & Data Binding
    # =========================================================================

    def _on_template_selected(self, index: int) -> None:
        tpl_names = list(PRESET_CIRCUIT_TEMPLATES.keys())
        if 0 <= index < len(tpl_names):
            script_text, default_vars = PRESET_CIRCUIT_TEMPLATES[tpl_names[index]]
            self.current_variables = dict(default_vars)
            self.script_editor.blockSignals(True)
            self.script_editor.setPlainText(script_text.strip())
            self.script_editor.blockSignals(False)
            self._sync_symbolic_parameters_from_script()
            if "中性原子" in tpl_names[index] and hasattr(self, "gateset_combo"):
                self.gateset_combo.setCurrentIndex(1)

    def _insert_gate_snippet(self, snippet: str) -> None:
        cursor = self.script_editor.textCursor()
        text = self.script_editor.toPlainText()
        if text and not text.endswith("\n"):
            self.script_editor.appendPlainText(snippet)
        else:
            cursor.insertText(snippet + "\n")
        self._sync_symbolic_parameters_from_script()

    def _on_script_text_changed(self) -> None:
        self._sync_symbolic_parameters_from_script()

    def _sync_symbolic_parameters_from_script(self) -> None:
        script = self.script_editor.toPlainText()
        sym_names = extract_symbolic_parameters(script)
        for name in sym_names:
            if name not in self.current_variables:
                if name.startswith("p_"):
                    self.current_variables[name] = 0.015
                else:
                    self.current_variables[name] = round(math.pi / 4.0, 4)

        # Clear existing parameter chips
        while self.param_chips_grid.count():
            item = self.param_chips_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.param_spinboxes.clear()

        if not sym_names:
            empty_lbl = QLabel("当前线路无符号变量 (可在门括号内写如 RX(theta) 0)")
            empty_lbl.setProperty("class", "cardSubtitle")
            self.param_chips_grid.addWidget(empty_lbl, 0, 0)
        else:
            for idx, name in enumerate(sym_names):
                chip = QFrame()
                chip.setProperty("class", "paramChip")
                c_lay = QHBoxLayout(chip)
                c_lay.setContentsMargins(8, 3, 6, 3)
                c_lay.setSpacing(6)
                lbl = QLabel(f"{name} =")
                lbl.setStyleSheet("font-weight: bold; color: #007AFF; font-family: Consolas;")
                spin = QDoubleSpinBox()
                spin.setRange(-10000.0, 10000.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.05)
                spin.setValue(float(self.current_variables.get(name, 0.0)))
                spin.setStyleSheet("background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 5px; padding: 2px 6px;")
                _clean_spinbox(spin)
                spin.valueChanged.connect(lambda val, n=name: self._on_param_chip_changed(n, val))
                c_lay.addWidget(lbl)
                c_lay.addWidget(spin, stretch=1)
                self.param_spinboxes[name] = spin
                self.param_chips_grid.addWidget(chip, idx // 2, idx % 2)

        prev_x = self.sweep_param_x.currentData()
        prev_y = self.sweep_param_y.currentData()
        self.sweep_param_x.blockSignals(True)
        self.sweep_param_y.blockSignals(True)
        self.sweep_param_x.clear()
        self.sweep_param_y.clear()

        for sname in sym_names:
            self.sweep_param_x.addItem(f"线路符号参数: {sname}", userData=sname)
            self.sweep_param_y.addItem(f"线路符号参数: {sname}", userData=sname)
        for hw_key, hw_desc in HARDWARE_SWEEP_PARAMS.items():
            self.sweep_param_x.addItem(hw_desc, userData=hw_key)
            self.sweep_param_y.addItem(hw_desc, userData=hw_key)

        if prev_x is not None:
            idx_x = self.sweep_param_x.findData(prev_x)
            if idx_x >= 0:
                self.sweep_param_x.setCurrentIndex(idx_x)
        if prev_y is not None:
            idx_y = self.sweep_param_y.findData(prev_y)
            if idx_y >= 0:
                self.sweep_param_y.setCurrentIndex(idx_y)
        elif self.sweep_param_y.count() > 1:
            self.sweep_param_y.setCurrentIndex(1)

        self.sweep_param_x.blockSignals(False)
        self.sweep_param_y.blockSignals(False)
        self._on_sweep_param_x_changed()

    def _on_param_chip_changed(self, name: str, val: float) -> None:
        self.current_variables[name] = float(val)

    def _on_platform_preset_changed(self, index: int) -> None:
        """Populate realistic physical parameters when switching experimental hardware platform."""
        if index == 0:
            # Superconducting CZ
            self.platform_info_badge.setText("⚡ 超导 Transmon · 零耗时 Virtual-Z + 微波 SX(20ns) + 可调耦合器 CZ(40ns)")
            self.t1_spin.setValue(35.0)
            self.t2_spin.setValue(25.0)
            self.t1q_spin.setValue(20.0)
            self.t2q_spin.setValue(40.0)
            self.dep1_spin.setValue(0.0015)
            self.dep2_spin.setValue(0.0120)
            self.zz_spin.setValue(50.0)
            self.ro_spin.setValue(0.0100)
        elif index == 1:
            # Neutral Atom Rydberg Array
            self.platform_info_badge.setText("⚛ 中性原子光镊阵列 (Rb-87/Yb-171) · 拉曼激光 RXY(400ns) + 里德堡阻塞 CZ(240ns)")
            self.t1_spin.setValue(1000.0)
            self.t2_spin.setValue(800.0)
            self.t1q_spin.setValue(400.0)
            self.t2q_spin.setValue(240.0)
            self.dep1_spin.setValue(0.0008)
            self.dep2_spin.setValue(0.0045)
            self.zz_spin.setValue(2.0)
            self.ro_spin.setValue(0.0080)
        elif index == 2:
            # Superconducting iSWAP
            self.platform_info_badge.setText("⚡ 超导电容耦合体系 · 零耗时 Virtual-Z + 微波 SX(20ns) + XY 交换 iSWAP(32ns)")
            self.t1_spin.setValue(40.0)
            self.t2_spin.setValue(28.0)
            self.t1q_spin.setValue(20.0)
            self.t2q_spin.setValue(32.0)
            self.dep1_spin.setValue(0.0012)
            self.dep2_spin.setValue(0.0100)
            self.zz_spin.setValue(35.0)
            self.ro_spin.setValue(0.0100)
        elif index == 3:
            # Superconducting CNOT
            self.platform_info_badge.setText("⚡ 超导交叉共振体系 · 零耗时 Virtual-Z + 微波 SX(20ns) + 受控反转 CNOT(60ns)")
            self.t1_spin.setValue(45.0)
            self.t2_spin.setValue(30.0)
            self.t1q_spin.setValue(20.0)
            self.t2q_spin.setValue(60.0)
            self.dep1_spin.setValue(0.0010)
            self.dep2_spin.setValue(0.0110)
            self.zz_spin.setValue(40.0)
            self.ro_spin.setValue(0.0100)
        else:
            # Trapped Ion MS
            self.platform_info_badge.setText("✦ 线性保罗阱离子链 (Yb+/Ca+) · 激光赤道旋转 RXY + 声子总线 Mølmer-Sørensen 门")
            self.t1_spin.setValue(5000.0)
            self.t2_spin.setValue(2000.0)
            self.t1q_spin.setValue(2000.0)
            self.t2q_spin.setValue(40000.0)
            self.dep1_spin.setValue(0.0003)
            self.dep2_spin.setValue(0.0030)
            self.zz_spin.setValue(0.1)
            self.ro_spin.setValue(0.0030)

    def _on_sweep_mode_changed(self, index: int) -> None:
        is_2d = index == 1
        self.lbl_param_y.setVisible(is_2d)
        self.sweep_param_y.setVisible(is_2d)
        self.lbl_y_range.setVisible(is_2d)
        self.sweep_y_start.setVisible(is_2d)
        self.sweep_y_stop.setVisible(is_2d)
        self.sweep_y_pts.setVisible(is_2d)

    def _on_sweep_param_x_changed(self) -> None:
        key = self.sweep_param_x.currentData()
        if not key:
            return
        if key in ("hw:t1_us", "hw:t2_us"):
            self.sweep_x_start.setValue(2.0)
            self.sweep_x_stop.setValue(80.0)
        elif key in ("hw:depol_1q", "hw:depol_2q", "hw:readout_err") or str(key).startswith("p_"):
            self.sweep_x_start.setValue(0.0)
            self.sweep_x_stop.setValue(0.08)
        elif key in ("hw:t_1q_ns", "hw:t_2q_ns"):
            self.sweep_x_start.setValue(10.0)
            self.sweep_x_stop.setValue(200.0)
        elif key == "hw:zz_khz":
            self.sweep_x_start.setValue(0.0)
            self.sweep_x_stop.setValue(500.0)
        else:
            self.sweep_x_start.setValue(0.0)
            self.sweep_x_stop.setValue(round(2.0 * math.pi, 3))

    def _build_current_hardware_model(self, num_qubits: int) -> HardwareModel:
        gset_idx = self.gateset_combo.currentIndex()
        t1_us = self.t1_spin.value()
        t2_us = self.t2_spin.value()
        t1q = self.t1q_spin.value()
        t2q = self.t2q_spin.value()
        dep1 = self.dep1_spin.value()
        dep2 = self.dep2_spin.value()
        zz_khz = self.zz_spin.value()
        ro_err = self.ro_spin.value()

        if self.topo_combo.currentIndex() == 1 and num_qubits > 1:
            topo = [(i, i + 1) for i in range(num_qubits - 1)]
        else:
            topo = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

        if gset_idx == 1:
            return HardwareModel.neutral_atom_processor(
                num_qubits=num_qubits,
                t1_us=t1_us,
                t2_us=t2_us,
                t_1q_ns=t1q,
                t_2q_ns=t2q,
                depol_1q=dep1,
                depol_2q=dep2,
                zz_crosstalk_khz=zz_khz,
                readout_err=ro_err,
                topology=topo,
            )

        if gset_idx == 0:
            p2q = "CZ"
            gset = NativeGateSet.superconducting_cz(t1q, t2q, dep1, dep2)
        elif gset_idx == 2:
            p2q = "ISWAP"
            gset = NativeGateSet.superconducting_iswap(t1q, t2q, dep1, dep2)
        elif gset_idx == 3:
            p2q = "CNOT"
            gset = NativeGateSet.superconducting_cnot(t1q, t2q, dep1, dep2)
        else:
            p2q = "MS"
            gset = NativeGateSet.trapped_ion(t1q, t2q, dep1, dep2)

        hw = HardwareModel.superconducting_processor(
            num_qubits=num_qubits,
            t1_us=t1_us,
            t2_us=t2_us,
            t_1q_ns=t1q,
            t_2q_ns=t2q,
            depol_1q=dep1,
            depol_2q=dep2,
            zz_crosstalk_khz=zz_khz,
            readout_err=ro_err,
            primary_2q_gate=p2q,
            topology=topo,
        )
        hw.native_gate_set = gset
        hw.name = gset.name
        return hw

    # =========================================================================
    # Core Actions: Run Simulation, Optimal Synthesis, Parameter Sweep
    # =========================================================================

    def run_simulation(self) -> None:
        try:
            circ = parse_circuit_script(
                self.script_editor.toPlainText(),
                variables=self.current_variables,
                circuit_name="UserCircuit",
            )
        except Exception as exc:
            QMessageBox.warning(self, "量子线路语法解析错误", str(exc))
            return

        self.current_circuit = circ
        hw = self._build_current_hardware_model(circ.num_qubits)
        shots = self.shots_spin.value()
        solver = QuantumSolver(hardware=hw, intra_gate_steps=15)

        b_idx = self.backend_combo.currentIndex()
        try:
            if b_idx == 0:
                self.current_results = solver.compare_backends(circ, hardware=hw, shots=shots)
                self.primary_result = (
                    self.current_results.get("qutip")
                    or self.current_results.get("kraus")
                    or next(iter(self.current_results.values()))
                )
            elif b_idx == 1:
                res = solver.solve(circ, backend="qutip", hardware=hw, shots=shots)
                self.current_results = {"qutip": res}
                self.primary_result = res
            elif b_idx == 2:
                res = solver.solve(circ, backend="stim", hardware=hw, shots=shots)
                self.current_results = {"stim": res}
                self.primary_result = res
            elif b_idx == 3:
                res = solver.solve(circ, backend="stim_qutip", hardware=hw, shots=shots)
                self.current_results = {"stim_qutip": res}
                self.primary_result = res
            else:
                res = solver.solve(circ, backend="kraus", hardware=hw, shots=shots)
                self.current_results = {"kraus": res}
                self.primary_result = res
        except Exception as exc:
            QMessageBox.critical(self, "求解器执行异常", str(exc))
            return

        # Update qubit selector in Evolution Page
        self.evo_qubit_combo.blockSignals(True)
        prev_q = max(0, self.evo_qubit_combo.currentIndex())
        self.evo_qubit_combo.clear()
        for q in range(circ.num_qubits):
            self.evo_qubit_combo.addItem(f"q{q}")
        self.evo_qubit_combo.setCurrentIndex(min(prev_q, circ.num_qubits - 1))
        self.evo_qubit_combo.blockSignals(False)

        n_steps = len(self.primary_result.steps)
        self.step_slider.blockSignals(True)
        self.step_slider.setMaximum(max(0, n_steps - 1))
        self.step_slider.setValue(max(0, n_steps - 1))
        self.step_slider.blockSignals(False)

        # Also refresh optimal synthesis silently so Page 1 stays in sync
        self.run_optimal_synthesis(silent=True)
        self._refresh_evolution_page()
        self._refresh_populations_page()

        fid_val = self.primary_result.final_fidelity
        fid_txt = f"{fid_val:.5f}" if fid_val is not None else "N/A"
        self.status_bar.showMessage(
            f"✓ 演化求解完成 [{self.primary_result.backend_name}]  |  比特数: {circ.num_qubits}  |  "
            f"脉冲总时长: {self.primary_result.total_duration_ns:.1f} ns  |  最终含噪态保真度: {fid_txt}"
        )

    def run_optimal_synthesis(self, silent: bool = False) -> None:
        try:
            circ = parse_circuit_script(
                self.script_editor.toPlainText(),
                variables=self.current_variables,
                circuit_name="TargetCircuit",
            )
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "量子线路语法解析错误", str(exc))
            return

        self.current_circuit = circ
        hw = self._build_current_hardware_model(circ.num_qubits)
        optimizer = QuantumCircuitOptimizer(hardware=hw)

        try:
            report = optimizer.optimize_circuit(
                circ,
                hardware=hw,
                evaluate_noisy_fidelity=(circ.num_qubits <= 4),
            )
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "量子线路综合优化失败", str(exc))
            return

        self.current_opt_report = report
        self.badge_unitary_fid.setText(f"酉等价保真度 F = {report.unitary_fidelity:.8f}")
        if report.noisy_fidelity_optimized is not None and report.noisy_fidelity_original is not None:
            delta_f = report.noisy_fidelity_optimized - report.noisy_fidelity_original
            sign = "+" if delta_f >= 0 else ""
            self.badge_noisy_fid.setText(
                f"QuTiP 含噪保真度: {report.noisy_fidelity_original:.4f} → {report.noisy_fidelity_optimized:.4f} ({sign}{delta_f:.4f})"
            )

        # Populate Clean Structured Metrics Comparison Table
        m0 = report.original_metrics
        m1 = report.optimized_metrics
        if m0 is not None:
            pulse_kind = "RXY/Raman" if hw.native_gate_set.decomposition_mode_1q == "rxy_rz" else "SX/X"
            rows = [
                ("两比特纠缠门数量 (2Q Gates)", str(m0.num_2q_gates), str(m1.num_2q_gates), f"减少 {m0.num_2q_gates - m1.num_2q_gates} 个" if m1.num_2q_gates <= m0.num_2q_gates else "已转原生基"),
                ("单比特物理脉冲数 (1Q Pulses)", str(m0.num_1q_physical_pulses), str(m1.num_1q_physical_pulses), f"{m1.num_1q_physical_pulses} 个 {pulse_kind} 脉冲"),
                ("零耗时 Virtual-Z 门 (0 ns)", str(m0.num_1q_virtual_gates), str(m1.num_1q_virtual_gates), "软件相位帧零误差"),
                ("总脉冲执行时长 (Duration)", f"{m0.duration_ns:.1f} ns", f"{m1.duration_ns:.1f} ns", f"缩短 {max(0.0, m0.duration_ns - m1.duration_ns):.1f} ns"),
                ("线路逻辑深度 (Circuit Depth)", str(m0.depth), str(m1.depth), f"总指令数: {m1.total_gates}"),
                ("量子门组成明细 (Breakdown)", str(m0.gate_breakdown), str(m1.gate_breakdown), report.gate_set_name),
            ]
            self.metrics_table.setRowCount(len(rows))
            for r_idx, r_tuple in enumerate(rows):
                for c_idx, val in enumerate(r_tuple):
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.metrics_table.setItem(r_idx, c_idx, item)

        # Populate Right Code Box (Optimal Native DSL + Stim Noisy Circuit)
        opt_c = report.optimized_circuit
        dsl_lines = [f"# 最优硬件原生门组合 [{report.gate_set_name}]", f"qubits {opt_c.num_qubits}"]
        for g in opt_c.gate_instructions:
            q_s = " ".join(str(q) for q in g.qubits)
            if g.params:
                p_s = ", ".join(f"{p:.4g}" for p in g.params)
                dsl_lines.append(f"{g.name}({p_s}) {q_s}")
            else:
                dsl_lines.append(f"{g.name} {q_s}")

        stim_res = self.current_results.get("stim_qutip") or self.current_results.get("stim")
        if stim_res and stim_res.stim_circuit_str:
            dsl_lines.extend(["", f"# 编译生成的 Stim 含噪量子线路 ({stim_res.backend_name})", stim_res.stim_circuit_str])
        self.stim_code_text.setPlainText("\n".join(dsl_lines))

        self._refresh_circuit_diagrams()
        if not silent:
            self._switch_workspace_page(0)

    def _adopt_optimized_circuit_to_editor(self) -> None:
        if self.current_opt_report is None:
            self.run_optimal_synthesis(silent=False)
            if self.current_opt_report is None:
                return
        opt_c = self.current_opt_report.optimized_circuit
        lines = [
            f"# 由优化器针对 [{self.current_opt_report.gate_set_name}] 生成的最优量子门序列",
            f"qubits {opt_c.num_qubits}",
        ]
        for g in opt_c.gate_instructions:
            q_str = " ".join(str(q) for q in g.qubits)
            if g.params:
                p_str = ", ".join(f"{p:.6g}" for p in g.params)
                lines.append(f"{g.name}({p_str}) {q_str}")
            else:
                lines.append(f"{g.name} {q_str}")
        lines.append("MEASURE " + " ".join(str(q) for q in range(opt_c.num_qubits)))
        self.script_editor.setPlainText("\n".join(lines))
        self.run_simulation()

    def _jump_to_and_run_sweep(self) -> None:
        self._switch_workspace_page(3)
        self.run_parameter_sweep()

    def run_parameter_sweep(self) -> None:
        script = self.script_editor.toPlainText()
        try:
            probe_circ = parse_circuit_script(script, self.current_variables)
        except Exception as exc:
            QMessageBox.warning(self, "量子线路解析错误", str(exc))
            return

        hw = self._build_current_hardware_model(probe_circ.num_qubits)
        backend = self.sweep_backend_combo.currentText().strip()
        param_x = self.sweep_param_x.currentData()
        if not param_x:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        def _prog(done: int, total: int) -> None:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(done)
            QApplication.processEvents()

        try:
            if self.sweep_mode_combo.currentIndex() == 0:
                res_1d = self.sweep_engine.run_1d_sweep(
                    script=script,
                    base_variables=self.current_variables,
                    base_hardware=hw,
                    param_name=str(param_x),
                    start_val=self.sweep_x_start.value(),
                    stop_val=self.sweep_x_stop.value(),
                    num_points=self.sweep_x_pts.value(),
                    backend=backend,
                    shots=self.shots_spin.value(),
                    progress_cb=_prog,
                )
                self._plot_sweep_1d(res_1d)
            else:
                param_y = self.sweep_param_y.currentData() or "hw:depol_2q"
                res_2d = self.sweep_engine.run_2d_sweep(
                    script=script,
                    base_variables=self.current_variables,
                    base_hardware=hw,
                    param_x=str(param_x),
                    x_range=(self.sweep_x_start.value(), self.sweep_x_stop.value(), self.sweep_x_pts.value()),
                    param_y=str(param_y),
                    y_range=(self.sweep_y_start.value(), self.sweep_y_stop.value(), self.sweep_y_pts.value()),
                    target_bitstring="0" * probe_circ.num_qubits,
                    backend=backend,
                    shots=max(1024, self.shots_spin.value() // 4),
                )
                self._plot_sweep_2d(res_2d)
        except Exception as exc:
            QMessageBox.critical(self, "参数扫描异常", str(exc))
        finally:
            self.progress_bar.setVisible(False)

    # =========================================================================
    # Plot Renderers (Golden-Ratio 2x2 Balanced Proportions)
    # =========================================================================

    def _refresh_circuit_diagrams(self) -> None:
        self.fig_circuits.clear()
        if self.current_circuit is None:
            self.canvas_circuits.draw()
            return

        if self.current_opt_report is not None:
            ax1 = self.fig_circuits.add_subplot(2, 1, 1)
            ax2 = self.fig_circuits.add_subplot(2, 1, 2)
            orig_c = self.current_circuit
            opt_c = self.current_opt_report.optimized_circuit
            m_opt = self.current_opt_report.optimized_metrics

            t1 = f"① 原始输入量子线路 ({orig_c.num_qubits} Qubits  |  总门数: {len(orig_c.gate_instructions)}  |  两比特门: {orig_c.num_2q_gates()}  |  深度: {orig_c.depth()})"
            t2 = (
                f"② 最优硬件原生门组合 [{self.current_opt_report.gate_set_name}] "
                f"(两比特门: {m_opt.num_2q_gates}  |  物理1Q脉冲: {m_opt.num_1q_physical_pulses}  |  "
                f"Virtual-Z: {m_opt.num_1q_virtual_gates}  |  总时长: {m_opt.duration_ns:.0f} ns)"
            )
            draw_quantum_circuit(orig_c, ax1, title=t1)
            draw_quantum_circuit(opt_c, ax2, title=t2)
        else:
            ax1 = self.fig_circuits.add_subplot(1, 1, 1)
            draw_quantum_circuit(self.current_circuit, ax1, title="当前编写量子线路图")

        self.fig_circuits.tight_layout(pad=0.8)
        self.canvas_circuits.draw()

    def _refresh_evolution_page(self) -> None:
        if self.primary_result is None:
            return
        res = self.primary_result

        self.evo_table.setRowCount(len(res.steps))
        for r, s in enumerate(res.steps):
            fid_str = f"{s.state_fidelity:.5f}" if s.state_fidelity is not None else "N/A"
            pur_str = f"{s.purity:.5f}" if s.purity is not None else "N/A"
            b_parts = [
                f"q{q}:({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})"
                for q, bv in sorted(s.bloch_vectors.items())
            ]
            if s.stabilizers:
                b_parts.append(f"Stab:[{', '.join(s.stabilizers)}]")
            row_vals = [
                str(s.step_index),
                s.operation_label,
                f"{s.elapsed_time_ns:.1f}",
                fid_str,
                pur_str,
                " | ".join(b_parts),
            ]
            for c, txt in enumerate(row_vals):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.evo_table.setItem(r, c, item)

        self._refresh_evolution_plots()

    def _on_step_slider_changed(self, _step_idx: int) -> None:
        self._refresh_evolution_plots()

    def _refresh_evolution_plots(self) -> None:
        if self.primary_result is None:
            return
        res = self.primary_result
        sel_q = max(0, self.evo_qubit_combo.currentIndex())
        step_k = min(max(0, self.step_slider.value()), len(res.steps) - 1)
        snap = res.steps[step_k]

        fid_s = f"{snap.state_fidelity:.4f}" if snap.state_fidelity is not None else "N/A"
        self.step_info_label.setText(
            f"Step {snap.step_index}: {snap.operation_label} (t={snap.elapsed_time_ns:.1f} ns, F={fid_s})"
        )

        # ---- Top Canvas: 2 Wide Landscape Plots (1x2) ----
        self.fig_evo_top.clear()
        ax_bloch_t = self.fig_evo_top.add_subplot(1, 2, 1)
        ax_fid = self.fig_evo_top.add_subplot(1, 2, 2)

        has_pulse = any(
            s.pulse_times_ns is not None and len(s.pulse_times_ns) > 1
            for s in res.steps
        )
        t_traj: List[float] = []
        bx_list: List[float] = []
        by_list: List[float] = []
        bz_list: List[float] = []

        if has_pulse:
            for s in res.steps:
                if s.pulse_times_ns is not None and s.pulse_bloch_trajectories and sel_q in s.pulse_bloch_trajectories:
                    t_traj.extend(s.pulse_times_ns.tolist())
                    tr = s.pulse_bloch_trajectories[sel_q]
                    bx_list.extend(tr[:, 0].tolist())
                    by_list.extend(tr[:, 1].tolist())
                    bz_list.extend(tr[:, 2].tolist())
            ax_bloch_t.plot(t_traj, bx_list, lw=2.0, color="#007AFF", label=f"<X_{sel_q}(t)>")
            ax_bloch_t.plot(t_traj, by_list, lw=2.0, color="#FF9500", label=f"<Y_{sel_q}(t)>")
            ax_bloch_t.plot(t_traj, bz_list, lw=2.0, color="#34C759", label=f"<Z_{sel_q}(t)>")
            ax_bloch_t.axvline(snap.elapsed_time_ns, color="#FF3B30", ls="--", lw=1.3)
            ax_bloch_t.set_xlabel("物理演化时间 t (ns)")
            ax_bloch_t.set_title(f"比特 q{sel_q} 脉冲级连续 Bloch 演化轨迹", fontsize=10.5, fontweight="bold")
        else:
            s_ids = [s.step_index for s in res.steps]
            bx_list = [s.bloch_vectors.get(sel_q, (0.0, 0.0, 0.0))[0] for s in res.steps]
            by_list = [s.bloch_vectors.get(sel_q, (0.0, 0.0, 0.0))[1] for s in res.steps]
            bz_list = [s.bloch_vectors.get(sel_q, (0.0, 0.0, 0.0))[2] for s in res.steps]
            ax_bloch_t.plot(s_ids, bx_list, "o-", lw=2.0, color="#007AFF", label=f"<X_{sel_q}>")
            ax_bloch_t.plot(s_ids, by_list, "s-", lw=2.0, color="#FF9500", label=f"<Y_{sel_q}>")
            ax_bloch_t.plot(s_ids, bz_list, "^-", lw=2.0, color="#34C759", label=f"<Z_{sel_q}>")
            ax_bloch_t.axvline(step_k, color="#FF3B30", ls="--", lw=1.3)
            ax_bloch_t.set_xlabel("线路演化步骤 Step")
            ax_bloch_t.set_title(f"比特 q{sel_q} 逐步 Bloch 分量演化", fontsize=10.5, fontweight="bold")

        ax_bloch_t.set_ylabel("Bloch 分量期望值")
        ax_bloch_t.set_ylim(-1.08, 1.08)
        ax_bloch_t.grid(True, alpha=0.25)
        ax_bloch_t.legend(loc="best", fontsize=8.5)

        for bname, r_obj in self.current_results.items():
            if bname in ("qutip", "kraus"):
                sx = [s.step_index for s in r_obj.steps]
                f_y = [s.state_fidelity if s.state_fidelity is not None else 1.0 for s in r_obj.steps]
                p_y = [s.purity if s.purity is not None else 1.0 for s in r_obj.steps]
                col = "#007AFF" if bname == "qutip" else "#34C759"
                ax_fid.plot(sx, f_y, "o-", lw=2.0, color=col, label=f"{bname.upper()} 保真度 F")
                ax_fid.plot(sx, p_y, "s--", lw=1.6, color=col, alpha=0.65, label=f"{bname.upper()} 纯度 Tr(ρ²)")
        ax_fid.axvline(step_k, color="#FF3B30", ls="--", lw=1.3)
        ax_fid.set_xlabel("线路演化步骤 Step")
        ax_fid.set_ylabel("Fidelity / Purity")
        ax_fid.set_title("逐步含噪态保真度与纯度衰减曲线", fontsize=10.5, fontweight="bold")
        ax_fid.grid(True, alpha=0.25)
        ax_fid.legend(loc="best", fontsize=8.5)

        self.fig_evo_top.tight_layout(pad=1.0, w_pad=2.0)
        self.canvas_evo_top.draw()

        # ---- Bottom-Left Canvas: Two 1:1 Square Plots (3D Bloch Sphere + Density Matrix Heatmap) ----
        self.fig_evo_bot.clear()
        ax_3d = self.fig_evo_bot.add_subplot(1, 2, 1, projection="3d")
        ax_rho = self.fig_evo_bot.add_subplot(1, 2, 2)

        u_sph = np.linspace(0, 2 * np.pi, 24)
        v_sph = np.linspace(0, np.pi, 14)
        xs = np.outer(np.cos(u_sph), np.sin(v_sph))
        ys = np.outer(np.sin(u_sph), np.sin(v_sph))
        zs = np.outer(np.ones_like(u_sph), np.cos(v_sph))
        ax_3d.plot_wireframe(xs, ys, zs, color="#D1D1D6", alpha=0.4, lw=0.6)
        ax_3d.plot([-1, 1], [0, 0], [0, 0], color="#AEAEB2", lw=0.8, ls="--")
        ax_3d.plot([0, 0], [-1, 1], [0, 0], color="#AEAEB2", lw=0.8, ls="--")
        ax_3d.plot([0, 0], [0, 0], [-1, 1], color="#AEAEB2", lw=0.8, ls="--")
        ax_3d.text(0, 0, 1.15, "|0>", ha="center", fontsize=8.5, fontweight="bold")
        ax_3d.text(0, 0, -1.22, "|1>", ha="center", fontsize=8.5, fontweight="bold")
        ax_3d.text(1.18, 0, 0, "+X", ha="center", fontsize=8)

        if bx_list:
            ax_3d.plot(bx_list, by_list, bz_list, color="#007AFF", lw=2.0, alpha=0.85)
            cur_bv = snap.bloch_vectors.get(sel_q, (0.0, 0.0, 1.0))
            ax_3d.scatter([cur_bv[0]], [cur_bv[1]], [cur_bv[2]], color="#FF3B30", s=50, zorder=10)
            ax_3d.plot([0, cur_bv[0]], [0, cur_bv[1]], [0, cur_bv[2]], color="#FF3B30", lw=2.2)

        ax_3d.set_xlim([-1.05, 1.05])
        ax_3d.set_ylim([-1.05, 1.05])
        ax_3d.set_zlim([-1.05, 1.05])
        ax_3d.set_box_aspect((1, 1, 1))
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        ax_3d.set_title(f"q{sel_q} 3D Bloch 球轨迹 (Step {step_k})", fontsize=10, fontweight="bold")

        if snap.density_matrix is not None:
            rho_abs = np.abs(snap.density_matrix)
            im = ax_rho.imshow(rho_abs, cmap="Blues", vmin=0.0, vmax=max(0.5, float(np.max(rho_abs))))
            dim = rho_abs.shape[0]
            if dim <= 8:
                b_labels = [format(i, f"0{res.num_qubits}b") for i in range(dim)]
                ax_rho.set_xticks(range(dim))
                ax_rho.set_yticks(range(dim))
                ax_rho.set_xticklabels(b_labels, rotation=35, fontsize=8)
                ax_rho.set_yticklabels(b_labels, fontsize=8)
                if dim <= 4:
                    for i in range(dim):
                        for j in range(dim):
                            ax_rho.text(
                                j,
                                i,
                                f"{rho_abs[i, j]:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7.8,
                                color="white" if rho_abs[i, j] > 0.3 else "#1D1D1F",
                            )
            self.fig_evo_bot.colorbar(im, ax=ax_rho, fraction=0.046, pad=0.04)
            ax_rho.set_title(f"Step {step_k} 密度矩阵模 |ρ_k|", fontsize=10, fontweight="bold")
        else:
            ax_rho.text(0.5, 0.5, "稳定子模式\n请查看右侧稳定子表", ha="center", va="center")
            ax_rho.axis("off")

        self.fig_evo_bot.tight_layout(pad=0.9)
        self.canvas_evo_bot.draw()

    def _refresh_populations_page(self) -> None:
        if self.primary_result is None:
            return
        res = self.primary_result
        n_q = res.num_qubits
        all_bstrs = [format(i, f"0{n_q}b") for i in range(1 << min(n_q, 5))]

        # ---- Top Canvas: 2 Wide Landscape Plots (Stepwise Population Curves + Multi-Backend Final Bar Chart) ----
        self.fig_pop_top.clear()
        ax_line = self.fig_pop_top.add_subplot(1, 2, 1)
        ax_bar = self.fig_pop_top.add_subplot(1, 2, 2)

        step_ids = [s.step_index for s in res.steps]
        pop_matrix = np.zeros((len(all_bstrs), len(res.steps)), dtype=np.float64)
        for col_idx, s in enumerate(res.steps):
            for row_idx, b in enumerate(all_bstrs):
                pop_matrix[row_idx, col_idx] = s.populations.get(b, 0.0)

        palette = ["#007AFF", "#FF9500", "#34C759", "#AF52DE", "#FF2D55", "#5856D6", "#00C7BE", "#A2845E"]
        for row_idx, b in enumerate(all_bstrs):
            if np.max(pop_matrix[row_idx]) > 0.01 or len(all_bstrs) <= 4:
                ax_line.plot(
                    step_ids,
                    pop_matrix[row_idx],
                    "o-",
                    lw=2.0,
                    color=palette[row_idx % len(palette)],
                    label=f"|{b}>",
                )
        ax_line.set_xlabel("线路演化步骤 Step")
        ax_line.set_ylabel("态布居概率 P(|x>)")
        ax_line.set_ylim(-0.04, 1.05)
        ax_line.set_title("各计算基态布居随步骤演化动态曲线", fontsize=10.5, fontweight="bold")
        ax_line.grid(True, alpha=0.25)
        ax_line.legend(loc="best", fontsize=8.5)

        x_pos = np.arange(len(all_bstrs))
        width = 0.80 / max(1, len(self.current_results))
        b_colors = ["#007AFF", "#34C759", "#FF9500", "#5856D6"]
        for idx, (_bname, r_obj) in enumerate(self.current_results.items()):
            probs = [r_obj.final_probabilities.get(b, 0.0) for b in all_bstrs]
            offset = (idx - 0.5 * (len(self.current_results) - 1)) * width
            ax_bar.bar(
                x_pos + offset,
                probs,
                width=width,
                color=b_colors[idx % len(b_colors)],
                alpha=0.92,
                label=r_obj.backend_name,
            )
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f"|{b}>" for b in all_bstrs], fontsize=9)
        ax_bar.set_ylabel("最终含噪测量概率")
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_title("最终测量态布居分布与多后端对比", fontsize=10.5, fontweight="bold")
        ax_bar.grid(True, axis="y", alpha=0.25)
        ax_bar.legend(loc="best", fontsize=8.5)

        self.fig_pop_top.tight_layout(pad=1.0, w_pad=2.0)
        self.canvas_pop_top.draw()

        # ---- Bottom-Left Canvas: Spatio-Temporal Population Evolution Heatmap ----
        self.fig_pop_bot.clear()
        ax_heat = self.fig_pop_bot.add_subplot(1, 1, 1)
        im = ax_heat.imshow(pop_matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax_heat.set_xticks(step_ids)
        ax_heat.set_xticklabels([f"Step {i}" for i in step_ids], fontsize=8.5)
        ax_heat.set_yticks(range(len(all_bstrs)))
        ax_heat.set_yticklabels([f"|{b}>" for b in all_bstrs], fontsize=8.5)
        ax_heat.set_xlabel("演化步骤 Step")
        ax_heat.set_ylabel("计算基态")
        ax_heat.set_title("态布居随步骤演化时空热力图", fontsize=10.5, fontweight="bold")
        self.fig_pop_bot.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        self.fig_pop_bot.tight_layout(pad=1.0)
        self.canvas_pop_bot.draw()

        # ---- Bottom-Right Table: Multi-Backend Population & Shot Counts ----
        self.pop_table.setRowCount(len(all_bstrs))
        for r_idx, b in enumerate(all_bstrs):
            p_qt = self.current_results["qutip"].final_probabilities.get(b, 0.0) if "qutip" in self.current_results else None
            p_kr = self.current_results["kraus"].final_probabilities.get(b, 0.0) if "kraus" in self.current_results else None
            p_st = self.current_results["stim"].final_probabilities.get(b, 0.0) if "stim" in self.current_results else None
            p_hy = self.current_results["stim_qutip"].final_probabilities.get(b, 0.0) if "stim_qutip" in self.current_results else None
            cnt = res.measurement_counts.get(b, 0)

            vals = [
                f"|{b}>",
                f"{p_qt:.5f}" if p_qt is not None else "—",
                f"{p_kr:.5f}" if p_kr is not None else "—",
                f"{p_st:.5f}" if p_st is not None else "—",
                f"{p_hy:.5f}" if p_hy is not None else "—",
                f"{cnt} / {res.shots} ({cnt / max(1, res.shots):.4f})",
            ]
            for c_idx, txt in enumerate(vals):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.pop_table.setItem(r_idx, c_idx, item)

    def _plot_sweep_1d(self, res: Sweep1DResult) -> None:
        # ---- Top Canvas: 2 Wide Landscape Plots (State Populations + Pauli Expectations) ----
        self.fig_sweep_top.clear()
        ax0 = self.fig_sweep_top.add_subplot(1, 2, 1)
        ax1 = self.fig_sweep_top.add_subplot(1, 2, 2)

        x = res.x_values
        palette = ["#007AFF", "#FF9500", "#34C759", "#AF52DE", "#FF2D55", "#5856D6"]
        for idx, (bstr, p_arr) in enumerate(res.populations.items()):
            if np.max(p_arr) > 0.01 or len(res.populations) <= 4:
                ax0.plot(x, p_arr, "o-", lw=2.0, ms=4, color=palette[idx % len(palette)], label=f"P(|{bstr}>)")
        ax0.set_xlabel(res.param_label)
        ax0.set_ylabel("态布居概率 P(|x>)")
        ax0.set_ylim(-0.04, 1.05)
        ax0.set_title(f"各计算基态布居随 [{res.param_name}] 扫描曲线", fontsize=10.5, fontweight="bold")
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc="best", fontsize=8.5)

        if 0 in res.bloch_expectations:
            b0 = res.bloch_expectations[0]
            ax1.plot(x, b0[:, 0], "o-", lw=2.0, ms=4, color="#007AFF", label="<X_0>")
            ax1.plot(x, b0[:, 1], "s-", lw=2.0, ms=4, color="#FF9500", label="<Y_0>")
            ax1.plot(x, b0[:, 2], "^-", lw=2.0, ms=4, color="#34C759", label="<Z_0>")
        ax1.set_xlabel(res.param_label)
        ax1.set_ylabel("泡利算符期望值")
        ax1.set_ylim(-1.08, 1.08)
        ax1.set_title(f"比特 q0 Bloch 分量随 [{res.param_name}] 扫描曲线", fontsize=10.5, fontweight="bold")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best", fontsize=8.5)

        self.fig_sweep_top.tight_layout(pad=1.0, w_pad=2.0)
        self.canvas_sweep_top.draw()

        # ---- Bottom-Left Canvas: Fidelity & Purity vs Swept Parameter ----
        self.fig_sweep_bot.clear()
        ax2 = self.fig_sweep_bot.add_subplot(1, 1, 1)
        ax2.plot(x, res.fidelities, "o-", lw=2.2, ms=4, color="#5856D6", label="含噪态保真度 F")
        ax2.plot(x, res.purities, "s--", lw=1.8, ms=4, color="#007AFF", label="量子态纯度 Tr(ρ²)")
        ax2.set_xlabel(res.param_label)
        ax2.set_ylabel("Fidelity / Purity")
        ax2.set_ylim(max(0.0, float(np.min(res.purities)) - 0.06), 1.03)
        ax2.set_title(f"保真度与纯度随 [{res.param_name}] 变化响应", fontsize=10.5, fontweight="bold")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best", fontsize=8.5)
        self.fig_sweep_bot.tight_layout(pad=1.0)
        self.canvas_sweep_bot.draw()

        # ---- Bottom-Right Table ----
        n_pts = len(x)
        self.sweep_table.setRowCount(n_pts)
        b0_arr = res.bloch_expectations.get(0, np.zeros((n_pts, 3)))
        for idx in range(n_pts):
            top_pops = ", ".join(
                f"|{b}>:{arr[idx]:.3f}"
                for b, arr in list(res.populations.items())[:4]
            )
            row_vals = [
                str(idx + 1),
                f"{x[idx]:.5g}",
                f"{res.fidelities[idx]:.5f}",
                f"{res.purities[idx]:.5f}",
                f"{b0_arr[idx, 2]:+.4f}",
                top_pops,
            ]
            for c_idx, txt in enumerate(row_vals):
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.sweep_table.setItem(idx, c_idx, item)

        self.status_bar.showMessage(
            f"✓ 一维参数扫描完成 [{res.backend_name}]  |  扫描参数: {res.param_label}  |  采样点数: {n_pts}"
        )

    def _plot_sweep_2d(self, res: Sweep2DResult) -> None:
        self.fig_sweep_top.clear()
        ax0 = self.fig_sweep_top.add_subplot(1, 2, 1)
        ax1 = self.fig_sweep_top.add_subplot(1, 2, 2)

        extent = [
            float(res.x_values[0]),
            float(res.x_values[-1]),
            float(res.y_values[0]),
            float(res.y_values[-1]),
        ]

        im0 = ax0.imshow(res.population_grid, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.set_xlabel(res.param_x_name)
        ax0.set_ylabel(res.param_y_name)
        ax0.set_title(f"基态 P(|{res.target_bitstring}>) 二维相图", fontsize=10.5, fontweight="bold")
        self.fig_sweep_top.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(res.fidelity_grid, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax1.set_xlabel(res.param_x_name)
        ax1.set_ylabel(res.param_y_name)
        ax1.set_title("含噪态保真度 F(x, y) 二维相图", fontsize=10.5, fontweight="bold")
        self.fig_sweep_top.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        self.fig_sweep_top.tight_layout(pad=1.0, w_pad=2.0)
        self.canvas_sweep_top.draw()

        self.fig_sweep_bot.clear()
        ax2 = self.fig_sweep_bot.add_subplot(1, 1, 1)
        im2 = ax2.imshow(res.z0_grid, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax2.set_xlabel(res.param_x_name)
        ax2.set_ylabel(res.param_y_name)
        ax2.set_title("<Z_0>(x, y) 期望值二维相图", fontsize=10.5, fontweight="bold")
        self.fig_sweep_bot.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        self.fig_sweep_bot.tight_layout(pad=1.0)
        self.canvas_sweep_bot.draw()

        self.status_bar.showMessage(
            f"✓ 二维参数相图扫描完成 [{res.backend_name}]  |  ({res.param_x_name} × {res.param_y_name})"
        )


def launch_gui(screenshot_dir: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = QuantumWorkbenchWindow()
    win.show()

    if screenshot_dir is not None:
        out_dir = pathlib.Path(screenshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _capture_all_pages() -> None:
            QApplication.processEvents()
            page_filenames = [
                "gui_mac_page1_circuit_and_optimizer.png",
                "gui_mac_page2_step_evolution_bloch.png",
                "gui_mac_page3_state_populations.png",
                "gui_mac_page4_parameter_sweep.png",
            ]
            for idx, fname in enumerate(page_filenames):
                win._switch_workspace_page(idx)
                QApplication.processEvents()
                pix = win.grab()
                pix.save(str(out_dir / fname), "PNG")

            # Also capture Template 6 (Neutral Atom Rydberg Array) to verify layout & platform support!
            win.template_combo.setCurrentIndex(5)
            win.run_simulation()
            win._switch_workspace_page(0)
            QApplication.processEvents()
            win.grab().save(str(out_dir / "gui_mac_neutral_atom_page1.png"), "PNG")
            win._switch_workspace_page(1)
            QApplication.processEvents()
            win.grab().save(str(out_dir / "gui_mac_neutral_atom_page2.png"), "PNG")
            app.quit()

        QTimer.singleShot(450, _capture_all_pages)

    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_gui())
