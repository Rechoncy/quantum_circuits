"""
Experimental quantum hardware model: physical parameters (T1, T2, ZZ crosstalk,
qubit frequencies, readout errors), coupling topology, pulse envelopes, and
native gate sets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .gates import NativeGateSet
from .noise import NoiseChannel, ReadoutError


@dataclass
class QubitPhysicalParams:
    """Physical characterization parameters of a single experimental qubit."""

    index: int
    t1_ns: float = 40_000.0          # Energy relaxation time T1 (default 40 us)
    t2_ns: float = 30_000.0          # Coherence time T2* / T2e (default 30 us)
    freq_ghz: float = 5.0            # Qubit transition frequency omega_q / 2pi (GHz)
    anharmonicity_ghz: float = -0.25 # Transmon anharmonicity alpha / 2pi (GHz)
    depol_1q: float = 0.0            # Additional depolarizing control error per 1Q gate
    readout_error: ReadoutError = field(default_factory=lambda: ReadoutError(0.01, 0.02))

    @property
    def t_phi_ns(self) -> float:
        """Pure dephasing time T_phi derived from 1/T2 = 1/(2*T1) + 1/T_phi."""
        t2_eff = min(self.t2_ns, 2.0 * self.t1_ns)
        if t2_eff <= 0 or not math.isfinite(t2_eff):
            return float("inf")
        rate = (1.0 / t2_eff) - (0.5 / self.t1_ns if (self.t1_ns > 0 and math.isfinite(self.t1_ns)) else 0.0)
        if rate <= 1e-15:
            return float("inf")
        return 1.0 / rate


@dataclass
class CouplerPhysicalParams:
    """Physical characterization of a two-qubit coupling link (qi, qj)."""

    qubits: Tuple[int, int]
    zz_crosstalk_mhz: float = 0.05   # Residual static ZZ coupling in MHz (e.g. 50 kHz)
    depol_2q: float = 0.0            # Additional control depolarizing error per 2Q gate
    gate_duration_ns: float = 40.0


@dataclass
class HardwareModel:
    """
    Complete experimental quantum device model.
    Used by:
      1. QuTiPBackend: constructs time-dependent Hamiltonians H(t), ZZ crosstalk, and T1/T2 collapse operators.
      2. KrausBackend & StimBackend: attaches gate-duration thermal relaxation + control depolarizing channels.
      3. QuantumCompiler / Optimizer: provides native gate set, coupling topology, and physical error weights.
    """

    name: str
    num_qubits: int
    native_gate_set: NativeGateSet = field(default_factory=NativeGateSet.superconducting_cz)
    qubit_params: Dict[int, QubitPhysicalParams] = field(default_factory=dict)
    coupler_params: Dict[Tuple[int, int], CouplerPhysicalParams] = field(default_factory=dict)
    pulse_shape: str = "cosine"  # 'rectangular' or 'cosine' (smooth Hann window)
    rise_ns: float = 0.0  # rectangular (flat-top) envelope rise edge
    fall_ns: float = 0.0  # rectangular (flat-top) envelope fall edge
    include_idle_noise: bool = True

    def __post_init__(self) -> None:
        for q in range(self.num_qubits):
            if q not in self.qubit_params:
                self.qubit_params[q] = QubitPhysicalParams(
                    index=q,
                    freq_ghz=5.0 + 0.12 * q,
                )
        # Default to all-to-all for <= 32 qubits, or nearest-neighbor linear chain for large arrays
        if not self.coupler_params and self.num_qubits > 1:
            if self.num_qubits <= 32:
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        self.coupler_params[(i, j)] = CouplerPhysicalParams(qubits=(i, j))
            else:
                for i in range(self.num_qubits - 1):
                    self.coupler_params[(i, i + 1)] = CouplerPhysicalParams(qubits=(i, i + 1))

    # ---- Preset Experimental Platforms ----

    @classmethod
    def superconducting_processor(
        cls,
        num_qubits: int = 3,
        t1_us: float = 40.0,
        t2_us: float = 30.0,
        t_1q_ns: float = 20.0,
        t_2q_ns: float = 40.0,
        depol_1q: float = 5e-4,
        depol_2q: float = 4e-3,
        zz_crosstalk_khz: float = 50.0,
        readout_err: float = 0.015,
        primary_2q_gate: str = "CZ",
        topology: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> "HardwareModel":
        """Create a realistic superconducting transmon processor model."""
        if primary_2q_gate.upper() == "CZ":
            gate_set = NativeGateSet.superconducting_cz(t_1q_ns, t_2q_ns, depol_1q, depol_2q)
        elif primary_2q_gate.upper() == "ISWAP":
            gate_set = NativeGateSet.superconducting_iswap(t_1q_ns, t_2q_ns, depol_1q, depol_2q)
        else:
            gate_set = NativeGateSet.superconducting_cnot(t_1q_ns, t_2q_ns, depol_1q, depol_2q)

        q_params = {
            q: QubitPhysicalParams(
                index=q,
                t1_ns=t1_us * 1000.0,
                t2_ns=t2_us * 1000.0,
                freq_ghz=5.0 + 0.15 * q,
                depol_1q=depol_1q,
                readout_error=ReadoutError(readout_err, readout_err * 1.2),
            )
            for q in range(num_qubits)
        }
        edges = (
            list(topology)
            if topology is not None
            else (
                [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
                if num_qubits <= 32
                else [(i, i + 1) for i in range(num_qubits - 1)]
            )
        )
        c_params = {}
        for u, v in edges:
            pair = (min(u, v), max(u, v))
            c_params[pair] = CouplerPhysicalParams(
                qubits=pair,
                zz_crosstalk_mhz=zz_crosstalk_khz / 1000.0,
                depol_2q=depol_2q,
                gate_duration_ns=t_2q_ns,
            )
        return cls(
            name=f"Superconducting-{num_qubits}Q ({primary_2q_gate})",
            num_qubits=num_qubits,
            native_gate_set=gate_set,
            qubit_params=q_params,
            coupler_params=c_params,
            pulse_shape="cosine",
        )

    @classmethod
    def neutral_atom_processor(
        cls,
        num_qubits: int = 3,
        t1_us: float = 1000.0,
        t2_us: float = 800.0,
        t_1q_ns: float = 400.0,
        t_2q_ns: float = 240.0,
        depol_1q: float = 8e-4,
        depol_2q: float = 4.5e-3,
        zz_crosstalk_khz: float = 2.0,
        readout_err: float = 0.008,
        topology: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> "HardwareModel":
        """
        Create a Neutral-Atom Rydberg Array quantum processor model (optical tweezers + Rydberg blockade).
        Features long hyperfine coherence times (ms scale), Raman RXY(theta, phi) + Virtual-RZ 1Q gates,
        and high-speed Levine-Pichler Rydberg blockade CZ gates (~240 ns).
        """
        gate_set = NativeGateSet.neutral_atom_rydberg(t_1q_ns, t_2q_ns, depol_1q, depol_2q)
        q_params = {
            q: QubitPhysicalParams(
                index=q,
                t1_ns=t1_us * 1000.0,
                t2_ns=t2_us * 1000.0,
                freq_ghz=6.834,  # Rb-87 hyperfine clock transition
                anharmonicity_ghz=0.0,
                depol_1q=depol_1q,
                readout_error=ReadoutError(readout_err, readout_err * 1.1),
            )
            for q in range(num_qubits)
        }
        edges = (
            list(topology)
            if topology is not None
            else (
                [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
                if num_qubits <= 32
                else [(i, i + 1) for i in range(num_qubits - 1)]
            )
        )
        c_params = {}
        for u, v in edges:
            pair = (min(u, v), max(u, v))
            c_params[pair] = CouplerPhysicalParams(
                qubits=pair,
                zz_crosstalk_mhz=zz_crosstalk_khz / 1000.0,
                depol_2q=depol_2q,
                gate_duration_ns=t_2q_ns,
            )
        return cls(
            name=f"NeutralAtom-Rydberg-{num_qubits}Q",
            num_qubits=num_qubits,
            native_gate_set=gate_set,
            qubit_params=q_params,
            coupler_params=c_params,
            pulse_shape="cosine",
        )

    @classmethod
    def ideal_processor(
        cls,
        num_qubits: int = 3,
        native_gate_set: Optional[NativeGateSet] = None,
    ) -> "HardwareModel":
        """Noise-free hardware model for ideal verification."""
        gset = native_gate_set or NativeGateSet.superconducting_cz(err_1q=0.0, err_2q=0.0)
        q_params = {
            q: QubitPhysicalParams(
                index=q,
                t1_ns=float("inf"),
                t2_ns=float("inf"),
                depol_1q=0.0,
                readout_error=ReadoutError(0.0, 0.0),
            )
            for q in range(num_qubits)
        }
        c_params = {
            (i, j): CouplerPhysicalParams(
                qubits=(i, j),
                zz_crosstalk_mhz=0.0,
                depol_2q=0.0,
            )
            for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
        }
        return cls(
            name=f"Ideal-{num_qubits}Q",
            num_qubits=num_qubits,
            native_gate_set=gset,
            qubit_params=q_params,
            coupler_params=c_params,
        )

    # ---- Helper Queries ----

    def is_connected(self, q0: int, q1: int) -> bool:
        pair = (min(q0, q1), max(q0, q1))
        return pair in self.coupler_params

    def gate_duration_ns(self, gate_name: str, qubits: Sequence[int]) -> float:
        uname = gate_name.upper()
        spec = self.native_gate_set.get_gate_spec(uname)
        if spec is not None:
            if spec.is_virtual:
                return 0.0
            return spec.duration_ns
        if uname in ("RZ", "P", "PHASE", "Z", "S", "SDG", "T", "TDG", "I"):
            # Check if RZ is virtual in this hardware
            rz_spec = self.native_gate_set.get_gate_spec("RZ")
            if rz_spec is not None and rz_spec.is_virtual:
                return 0.0
            return 10.0
        if len(qubits) == 1:
            return 20.0
        pair = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
        if pair in self.coupler_params:
            return self.coupler_params[pair].gate_duration_ns
        return 40.0

    def gate_noise_channels(
        self,
        gate_name: str,
        qubits: Tuple[int, ...],
        duration_ns: Optional[float] = None,
    ) -> List[NoiseChannel]:
        """
        Generate the physical noise channels induced during execution of a gate on this hardware:
          1. Control depolarizing error on active qubit(s)
          2. T1/T2 thermal relaxation on active (and optionally idle) qubits over duration_ns
        """
        dt = self.gate_duration_ns(gate_name, qubits) if duration_ns is None else duration_ns
        if dt <= 0.0:
            return []

        channels: List[NoiseChannel] = []
        # 1. Control error
        if len(qubits) == 1:
            q = qubits[0]
            p_dep = self.qubit_params[q].depol_1q
            if p_dep > 0:
                channels.append(NoiseChannel.depolarizing_1q(q, p_dep))
        elif len(qubits) == 2:
            pair = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
            p_dep2 = self.coupler_params.get(pair, CouplerPhysicalParams(pair)).depol_2q
            if p_dep2 > 0:
                channels.append(NoiseChannel.depolarizing_2q(qubits[0], qubits[1], p_dep2))

        # 2. T1 / T2 decoherence during gate duration
        target_qubits = range(self.num_qubits) if self.include_idle_noise else qubits
        for q in target_qubits:
            qp = self.qubit_params[q]
            if math.isfinite(qp.t1_ns) or math.isfinite(qp.t2_ns):
                channels.append(
                    NoiseChannel.thermal_relaxation(
                        qubit=q,
                        t1_ns=qp.t1_ns,
                        t2_ns=qp.t2_ns,
                        duration_ns=dt,
                    )
                )
        return channels
