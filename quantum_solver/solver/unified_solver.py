"""
Unified Multi-Backend Quantum Solver entry point.
Seamlessly dispatches between:
  - 'qutip': QuTiP physical Hamiltonian & Lindblad master equation solver
  - 'stim': Stim stabilizer tableau & high-speed Pauli noise frame solver
  - 'stim_qutip': Hybrid solver where QuTiP extracts gate Pauli error rates and Stim runs fast sampling
  - 'kraus': Exact gate-level density matrix & Kraus operator solver
  - 'auto': Automatic selection based on circuit Cliffordness, qubit count, and hardware specification
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..core.circuit import QuantumCircuit
from ..core.hardware import HardwareModel
from .base import EvolutionResult
from .kraus_backend import KrausBackend
from .qutip_backend import QuTiPBackend
from .stim_backend import StimBackend


class QuantumSolver:
    """
    Unified Quantum Circuit Evolution & Noisy Measurement Solver.
    """

    def __init__(
        self,
        hardware: Optional[HardwareModel] = None,
        intra_gate_steps: int = 15,
    ) -> None:
        self.hardware = hardware
        self.qutip_backend = QuTiPBackend(intra_gate_steps=intra_gate_steps)
        self.kraus_backend = KrausBackend()
        self.stim_backend = StimBackend(non_clifford_policy="twirl", use_qutip_calibration=False)
        self.stim_qutip_backend = StimBackend(non_clifford_policy="twirl", use_qutip_calibration=True)

    def solve(
        self,
        circuit: QuantumCircuit,
        backend: str = "auto",
        hardware: Optional[HardwareModel] = None,
        shots: int = 4096,
        seed: Optional[int] = 42,
        initial_state: Optional[np.ndarray] = None,
        initial_state_labels: Optional[Sequence[str]] = None,
    ) -> EvolutionResult:
        """
        Execute a quantum circuit and return step-by-step state evolution and measurement results.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to simulate.
        backend : str
            One of:
              - 'qutip': Continuous-time pulse & Lindblad master equation solver (QuTiP)
              - 'stim': High-speed stabilizer & Pauli-frame solver (Stim)
              - 'stim_qutip': Hybrid QuTiP-calibrated Pauli-twirled Stim solver
              - 'kraus': Gate-level density matrix & Kraus operator solver
              - 'auto': Automatically picks 'stim' for large Clifford circuits, 'qutip' when
                experimental hardware model is active on <= 6 qubits, or 'kraus' otherwise.
        hardware : Optional[HardwareModel]
            Experimental hardware model (overrides self.hardware if provided).
        shots : int
            Number of measurement shots to sample.
        seed : Optional[int]
            Random seed for reproducible measurement sampling.
        """
        hw = hardware if hardware is not None else self.hardware
        b = backend.lower().strip()

        if b == "auto":
            if circuit.num_qubits > 10 and circuit.is_clifford():
                b = "stim"
            elif hw is not None and circuit.num_qubits <= 6:
                b = "qutip"
            else:
                b = "kraus"

        if b == "qutip":
            return self.qutip_backend.run(
                circuit, hardware=hw, shots=shots, seed=seed, initial_state=initial_state
            )
        if b == "kraus":
            return self.kraus_backend.run(
                circuit, hardware=hw, shots=shots, seed=seed, initial_state=initial_state
            )
        if b == "stim":
            return self.stim_backend.run(
                circuit,
                hardware=hw,
                shots=shots,
                seed=seed,
                initial_state_labels=initial_state_labels,
            )
        if b in ("stim_qutip", "hybrid"):
            return self.stim_qutip_backend.run(
                circuit,
                hardware=hw,
                shots=shots,
                seed=seed,
                initial_state_labels=initial_state_labels,
            )

        raise ValueError(
            f"Unknown backend '{backend}'. Choose from 'qutip', 'stim', 'stim_qutip', 'kraus', or 'auto'."
        )

    def compare_backends(
        self,
        circuit: QuantumCircuit,
        hardware: Optional[HardwareModel] = None,
        shots: int = 10000,
        seed: int = 42,
    ) -> Dict[str, EvolutionResult]:
        """
        Run the same circuit across QuTiP, Kraus, and Stim backends for direct cross-validation.
        """
        hw = hardware if hardware is not None else self.hardware
        results: Dict[str, EvolutionResult] = {}
        if circuit.num_qubits <= 8:
            results["qutip"] = self.solve(circuit, backend="qutip", hardware=hw, shots=shots, seed=seed)
            results["kraus"] = self.solve(circuit, backend="kraus", hardware=hw, shots=shots, seed=seed)
        results["stim"] = self.solve(circuit, backend="stim", hardware=hw, shots=shots, seed=seed)
        if hw is not None and circuit.num_qubits <= 8:
            results["stim_qutip"] = self.solve(
                circuit, backend="stim_qutip", hardware=hw, shots=shots, seed=seed
            )
        return results
