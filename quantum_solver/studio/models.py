"""
Unified Pydantic Data Models for Quantum Studio v2 (Single Source of Truth).
Covers Quantum Registers, Custom Gate Library (4 definition types), Visual Circuit
Operations, Hardware Profiles (including Neutral Atom 2D coordinates & Rydberg
blockade radius), Simulation Settings, Optimization, and Parameter Sweeps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field


class QubitSpec(BaseModel):
    """First-class quantum register entry with optional per-qubit physical overrides."""

    id: int
    name: str = "q0"
    color: str = "#0A84FF"
    init_state: Literal["|0>", "|1>", "|+>", "|->", "|+i>", "|-i>", "custom"] = "|0>"
    init_theta: float = 0.0
    init_phi: float = 0.0
    # Per-qubit overrides (None = inherit from active HardwareProfile)
    t1_us: Optional[float] = None
    t2_us: Optional[float] = None
    freq_ghz: Optional[float] = None
    depol_1q: Optional[float] = None
    readout_err: Optional[float] = None
    rabi_mhz: Optional[float] = None  # Per-qubit Rabi frequency override (MHz)
    # 2D spatial position (in micrometers, used for Neutral Atom Rydberg blockade map & topology)
    pos_x: float = 0.0
    pos_y: float = 0.0


class CircuitOpModel(BaseModel):
    """Single operation on the visual circuit canvas (and synchronized with DSL code)."""

    id: str
    step: int = 0
    kind: Literal["gate", "noise", "measure"] = "gate"
    name: str
    qubits: List[int]
    # Parameters can be concrete floats or symbolic expressions like "theta", "pi/2", "2*phi"
    params: List[Union[float, str]] = Field(default_factory=list)
    duration_ns: Optional[float] = None
    noise_mode: Literal["inherit", "none", "custom"] = "inherit"
    custom_depol: Optional[float] = None
    rabi_mhz: Optional[float] = None  # Per-gate Rabi frequency override (single-qubit drive gates)


class CustomGateDef(BaseModel):
    """
    User-defined custom quantum gate supporting 4 creation modes:
      1. 'matrix': explicit 2^n x 2^n complex matrix (stored as string expressions e.g. '0', '1', '1/sqrt(2)', '1j')
      2. 'expression': parametric matrix expression e.g. 'expm(-1j * theta / 2 * (cos(phi)*X + sin(phi)*Y))'
      3. 'hamiltonian': Pauli-string generator e.g. '0.5*XX + 0.5*YY' + duration_ns (QuTiP pulse integration)
      4. 'composite': reusable sub-circuit macro encapsulated from canvas operations
    """

    id: str
    name: str
    display_label: str = ""
    num_qubits: int = 1
    definition_type: Literal["matrix", "expression", "hamiltonian", "composite"] = "matrix"
    # Mode 1: matrix entries (2^n x 2^n of string complex expressions)
    matrix_exprs: Optional[List[List[str]]] = None
    # Mode 2: parametric expression
    expression: Optional[str] = None
    param_names: List[str] = Field(default_factory=list)
    default_params: List[float] = Field(default_factory=list)
    # Mode 3: Pauli-string Hamiltonian + pulse
    hamiltonian_str: Optional[str] = None
    hamiltonian_scale: float = 1.0  # angle or strength factor (rad)
    pulse_shape: Literal["cosine", "gaussian", "rectangular", "trapezoid"] = "cosine"
    # Mode 4: composite macro
    sub_operations: List[CircuitOpModel] = Field(default_factory=list)
    # Hardware & visual metadata
    duration_ns: float = 40.0
    infidelity: float = 0.002
    color: str = "#8B5CF6"
    is_native: bool = False  # If True, promoted into active HardwareProfile native gate set for optimizer!
    description: str = ""


class HardwareProfileModel(BaseModel):
    """Experimental quantum hardware platform configuration."""

    id: str
    name: str
    platform: Literal[
        "neutral_atom",
        "superconducting_cz",
        "superconducting_iswap",
        "superconducting_cnot",
        "trapped_ion",
    ] = "neutral_atom"
    t1_us: float = 1000.0
    t2_us: float = 800.0
    t_1q_ns: float = 400.0
    t_2q_ns: float = 240.0
    depol_1q: float = 0.0008
    depol_2q: float = 0.0045
    zz_khz: float = 2.0
    readout_err: float = 0.008
    topology_mode: Literal["all", "linear", "custom", "rydberg_radius"] = "rydberg_radius"
    edges: List[Tuple[int, int]] = Field(default_factory=list)
    blockade_radius_um: float = 6.5  # Neutral atom Rydberg blockade radius in μm
    pulse_shape: Literal["cosine", "gaussian", "rectangular", "trapezoid"] = "cosine"
    # Single-qubit drive model
    #   fixed_duration: every physical 1Q pulse lasts t_1q_ns, amplitude scales with rotation angle
    #   fixed_rabi:     peak Rabi frequency fixed at rabi_mhz, pulse length scales with rotation angle
    drive_mode: Literal["fixed_duration", "fixed_rabi"] = "fixed_duration"
    rabi_mhz: float = 1.25
    # Rise / fall edges of the rectangular (flat-top) envelope, ns
    rise_ns: float = 0.0
    fall_ns: float = 0.0


class SimulationSettingsModel(BaseModel):
    """Simulation execution & noise toggle controls."""

    backend: Literal["qutip", "kraus", "trajectory", "stim", "stim_qutip", "compare"] = "qutip"
    shots: int = 20000
    enable_t1_t2: bool = True
    enable_depol: bool = True
    enable_zz: bool = True
    enable_readout: bool = True
    intra_gate_steps: int = 12


class ProjectModel(BaseModel):
    """Single source of truth for an entire Quantum Studio project."""

    name: str = "Rydberg_GHZ.qproj"
    qubits: List[QubitSpec] = Field(default_factory=list)
    custom_gates: List[CustomGateDef] = Field(default_factory=list)
    operations: List[CircuitOpModel] = Field(default_factory=list)
    variables: Dict[str, float] = Field(default_factory=dict)
    hardware_profiles: List[HardwareProfileModel] = Field(default_factory=list)
    active_hardware_id: str = "hw_neutral_atom"
    sim_settings: SimulationSettingsModel = Field(default_factory=SimulationSettingsModel)


# ---- API Request / Response Models ----

class ValidateCustomGateRequest(BaseModel):
    gate: CustomGateDef
    sample_params: Optional[List[float]] = None
    existing_custom_gates: List[CustomGateDef] = Field(default_factory=list)


class DslSyncRequest(BaseModel):
    dsl: str
    project: ProjectModel


class QasmSyncRequest(BaseModel):
    qasm: str
    project: ProjectModel


class OptimizeRequest(BaseModel):
    project: ProjectModel
    evaluate_noisy_fidelity: bool = True
    preferred_native_2q: Optional[str] = None


class BenchmarkHardwareRequest(BaseModel):
    project: ProjectModel


class SynthesizeUnitaryRequest(BaseModel):
    """Direct target unitary / Pauli Hamiltonian optimal native gate synthesis request."""

    project: ProjectModel
    mode: Literal["matrix", "hamiltonian"] = "matrix"
    num_qubits: int = 2
    matrix_exprs: Optional[List[List[str]]] = None
    hamiltonian_str: Optional[str] = None
    hamiltonian_scale: float = 1.5708
    evaluate_noisy_fidelity: bool = True
    preferred_native_2q: Optional[str] = None


class SweepRequest(BaseModel):
    project: ProjectModel
    mode: Literal["1d", "2d"] = "1d"
    param_x: str
    x_start: float
    x_stop: float
    x_points: int = 25
    param_y: Optional[str] = None
    y_start: float = 0.0
    y_stop: float = 3.1416
    y_points: int = 10
    target_bitstring: Optional[str] = None
    observed_qubit: int = 0
    hamiltonian_str: Optional[str] = None
    backend: Literal["qutip", "kraus", "trajectory", "stim", "stim_qutip"] = "qutip"


class VariationalOptimizeRequest(BaseModel):
    """Closed-loop VQE / QAOA parameter optimization minimizing <psi(theta)|H|psi(theta)>."""

    project: ProjectModel
    hamiltonian_str: str
    param_names: List[str] = Field(default_factory=list)
    method: Literal["Parameter-Shift", "COBYLA", "Powell", "Nelder-Mead", "L-BFGS-B"] = "COBYLA"
    max_iter: int = 50
    backend: Literal["qutip", "kraus", "trajectory", "stim", "stim_qutip"] = "trajectory"


class ZneRequest(BaseModel):
    """Zero-Noise Extrapolation (ZNE) quantum error mitigation request."""

    project: ProjectModel
    scale_factors: List[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    observed_qubit: int = 0
    hamiltonian_str: Optional[str] = None
    backend: Literal["qutip", "kraus", "trajectory", "stim", "stim_qutip"] = "kraus"


class NoiseBudgetRequest(BaseModel):
    """Channel-by-channel Noise Error Budget ablation request."""

    project: ProjectModel
    backend: Optional[Literal["qutip", "kraus", "trajectory", "stim", "stim_qutip"]] = None


class GateBenchmarkRequest(BaseModel):
    """Quantum Gate Characterization & Benchmarking request (RB / Interleaved RB, AllXY, GST, XEB)."""

    project: ProjectModel
    protocol: Literal["rb", "allxy", "gst", "xeb", "rabi", "ramsey", "t1"] = "rb"
    # Basic calibration experiments (Rabi / Ramsey / T1)
    cal_points: int = 61
    cal_max_ns: Optional[float] = None
    ramsey_detuning_mhz: float = 0.5
    target_qubit: int = 0
    second_qubit: int = 1
    rb_num_qubits: Literal[1, 2] = 1
    rb_lengths: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    rb_num_seeds: int = 20
    interleaved_gate: Optional[str] = None
    xeb_entangler: str = "SQISWAP"
    gst_max_length: int = 4
    shots: int = 4096
    # Optional coherent calibration error injection for realistic experimental diagnostics
    amp_error_pct: float = 0.0
    detuning_mhz: float = 0.0
    phase_skew_deg: float = 0.0
    seed: int = 42


class NlCircuitRequest(BaseModel):
    """Natural language -> quantum circuit generation & multi-turn refinement request."""

    prompt: str
    project: ProjectModel
    mode: Literal["auto", "replace", "refine", "append"] = "auto"
    engine: Literal["local", "llm"] = "local"
    history: List[Dict[str, str]] = Field(default_factory=list)
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
