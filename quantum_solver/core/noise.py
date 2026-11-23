"""
Quantum noise channel models, Kraus operator representations, Lindblad collapse
operator builders, and Pauli Twirling bridges (for converting QuTiP/Kraus channels
into Stim-compatible Pauli error probabilities).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gates import I2, PAULI_X, PAULI_Y, PAULI_Z, PAULIS_1Q


PAULI_LABELS_1Q = ("I", "X", "Y", "Z")
PAULI_LABELS_2Q = tuple(
    p1 + p2 for p1 in PAULI_LABELS_1Q for p2 in PAULI_LABELS_1Q
)  # 16 labels starting with 'II'


@dataclass
class NoiseChannel:
    """
    Unified quantum noise channel representation.
    Can be attached to specific instructions or generated automatically by a HardwareModel.
    """

    kind: str  # 'depolarizing1q', 'depolarizing2q', 'pauli1q', 'pauli2q', 'amplitude_damping', 'phase_damping', 'thermal_relaxation'
    qubits: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)
    custom_kraus: Optional[List[np.ndarray]] = None

    # ---- Factory constructors ----

    @classmethod
    def depolarizing_1q(cls, qubit: int, p: float) -> "NoiseChannel":
        """1-qubit symmetric depolarizing channel with total error probability p."""
        return cls("depolarizing1q", (qubit,), {"p": float(p)})

    @classmethod
    def depolarizing_2q(cls, q0: int, q1: int, p: float) -> "NoiseChannel":
        """2-qubit symmetric depolarizing channel with total error probability p across 15 non-identity Paulis."""
        return cls("depolarizing2q", (q0, q1), {"p": float(p)})

    @classmethod
    def bit_flip(cls, qubit: int, p: float) -> "NoiseChannel":
        return cls("pauli1q", (qubit,), {"px": float(p), "py": 0.0, "pz": 0.0})

    @classmethod
    def phase_flip(cls, qubit: int, p: float) -> "NoiseChannel":
        return cls("pauli1q", (qubit,), {"px": 0.0, "py": 0.0, "pz": float(p)})

    @classmethod
    def pauli_1q(cls, qubit: int, px: float, py: float, pz: float) -> "NoiseChannel":
        return cls(
            "pauli1q",
            (qubit,),
            {"px": float(px), "py": float(py), "pz": float(pz)},
        )

    @classmethod
    def pauli_2q(
        cls,
        q0: int,
        q1: int,
        probs: Dict[str, float],
    ) -> "NoiseChannel":
        """
        2-qubit Pauli channel specified by a dict mapping 2-letter Pauli strings
        ('IX', 'IY', ..., 'ZZ') to their probabilities.
        """
        clean_probs = {
            lbl: float(probs.get(lbl, 0.0))
            for lbl in PAULI_LABELS_2Q
            if lbl != "II"
        }
        return cls("pauli2q", (q0, q1), clean_probs)

    @classmethod
    def amplitude_damping(cls, qubit: int, gamma: float) -> "NoiseChannel":
        """Energy relaxation (T1 decay) with damping probability gamma = 1 - exp(-t/T1)."""
        return cls("amplitude_damping", (qubit,), {"gamma": float(gamma)})

    @classmethod
    def phase_damping(cls, qubit: int, lam: float) -> "NoiseChannel":
        """Pure dephasing with probability parameter lam = 1 - exp(-2*t/T_phi)."""
        return cls("phase_damping", (qubit,), {"lam": float(lam)})

    @classmethod
    def thermal_relaxation(
        cls,
        qubit: int,
        t1_ns: float,
        t2_ns: float,
        duration_ns: float,
    ) -> "NoiseChannel":
        """
        Combined T1 & T2 thermal relaxation channel over a gate/idle duration_ns.
        Enforces physical bound T2 <= 2 * T1.
        """
        return cls(
            "thermal_relaxation",
            (qubit,),
            {
                "t1_ns": float(t1_ns),
                "t2_ns": float(min(t2_ns, 2.0 * t1_ns)),
                "duration_ns": float(duration_ns),
            },
        )

    # ---- Kraus Operator Generation ----

    def kraus_operators(self) -> List[np.ndarray]:
        """Return exact Kraus operators {K_m} satisfying sum_m K_m^† K_m = I."""
        if self.custom_kraus is not None:
            return [ np.asarray(k, dtype=np.complex128) for k in self.custom_kraus ]

        kind = self.kind.lower()
        if kind == "depolarizing1q":
            p = max(0.0, min(1.0, self.params["p"]))
            return [
                math.sqrt(max(0.0, 1.0 - p)) * I2,
                math.sqrt(p / 3.0) * PAULI_X,
                math.sqrt(p / 3.0) * PAULI_Y,
                math.sqrt(p / 3.0) * PAULI_Z,
            ]

        if kind == "pauli1q":
            px = max(0.0, self.params.get("px", 0.0))
            py = max(0.0, self.params.get("py", 0.0))
            pz = max(0.0, self.params.get("pz", 0.0))
            pi = max(0.0, 1.0 - (px + py + pz))
            ops = [math.sqrt(pi) * I2]
            if px > 0:
                ops.append(math.sqrt(px) * PAULI_X)
            if py > 0:
                ops.append(math.sqrt(py) * PAULI_Y)
            if pz > 0:
                ops.append(math.sqrt(pz) * PAULI_Z)
            return ops

        if kind == "amplitude_damping":
            gamma = max(0.0, min(1.0, self.params["gamma"]))
            k0 = np.array(
                [[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]],
                dtype=np.complex128,
            )
            k1 = np.array(
                [[0.0, math.sqrt(gamma)], [0.0, 0.0]],
                dtype=np.complex128,
            )
            return [k0, k1]

        if kind == "phase_damping":
            lam = max(0.0, min(1.0, self.params["lam"]))
            k0 = np.array(
                [[1.0, 0.0], [0.0, math.sqrt(1.0 - lam)]],
                dtype=np.complex128,
            )
            k1 = np.array(
                [[0.0, 0.0], [0.0, math.sqrt(lam)]],
                dtype=np.complex128,
            )
            return [k0, k1]

        if kind == "thermal_relaxation":
            t1 = self.params["t1_ns"]
            t2 = min(self.params["t2_ns"], 2.0 * t1)
            dt = self.params["duration_ns"]
            gamma_1 = 1.0 - math.exp(-dt / t1) if t1 > 0 and math.isfinite(t1) else 0.0
            # Pure dephasing rate 1/T_phi = 1/T2 - 1/(2*T1)
            if t2 > 0 and math.isfinite(t2):
                rate_phi = max(0.0, (1.0 / t2) - (0.5 / t1 if (t1 > 0 and math.isfinite(t1)) else 0.0))
                lam_phi = 1.0 - math.exp(-2.0 * dt * rate_phi)
            else:
                lam_phi = 0.0

            ad_kraus = NoiseChannel.amplitude_damping(self.qubits[0], gamma_1).kraus_operators()
            pd_kraus = NoiseChannel.phase_damping(self.qubits[0], lam_phi).kraus_operators()
            return [kp @ ka for kp in pd_kraus for ka in ad_kraus if np.linalg.norm(kp @ ka) > 1e-15]

        if kind == "depolarizing2q":
            p = max(0.0, min(1.0, self.params["p"]))
            ops = [math.sqrt(max(0.0, 1.0 - p)) * np.eye(4, dtype=np.complex128)]
            weight = math.sqrt(p / 15.0)
            for lbl in PAULI_LABELS_2Q:
                if lbl == "II":
                    continue
                p_mat = np.kron(PAULIS_1Q[lbl[0]], PAULIS_1Q[lbl[1]])
                ops.append(weight * p_mat)
            return ops

        if kind == "pauli2q":
            total_err = sum(max(0.0, self.params.get(lbl, 0.0)) for lbl in PAULI_LABELS_2Q if lbl != "II")
            p_ii = max(0.0, 1.0 - total_err)
            ops = [math.sqrt(p_ii) * np.eye(4, dtype=np.complex128)]
            for lbl in PAULI_LABELS_2Q:
                if lbl == "II":
                    continue
                prob = max(0.0, self.params.get(lbl, 0.0))
                if prob > 0:
                    p_mat = np.kron(PAULIS_1Q[lbl[0]], PAULIS_1Q[lbl[1]])
                    ops.append(math.sqrt(prob) * p_mat)
            return ops

        raise ValueError(f"Unsupported NoiseChannel kind: {self.kind}")

    # ---- Pauli Twirling (Kraus / Physical -> Stim Pauli Probabilities) ----

    def to_pauli_probabilities(self) -> Dict[str, float]:
        """
        Project the noise channel onto an exact Pauli channel via Pauli Twirling:
          E_twirl(rho) = (1/4^n) sum_Q Q E(Q rho Q) Q = sum_P p_P P rho P.
        Returns a dictionary of Pauli probabilities:
          - 1Q: {'I': pI, 'X': pX, 'Y': pY, 'Z': pZ}
          - 2Q: {'II': pII, 'IX': pIX, ..., 'ZZ': pZZ}
        """
        return pauli_twirl_kraus(self.kraus_operators(), num_qubits=len(self.qubits))


def pauli_twirl_kraus(
    kraus_ops: Sequence[np.ndarray],
    num_qubits: int = 1,
) -> Dict[str, float]:
    """
    Compute the exact Pauli-twirled error probabilities from any list of Kraus
    operators {K_m} on 1 or 2 qubits:
        p_P = sum_m | (1 / 2^n) * Tr(P * K_m) |^2
    """
    if num_qubits == 1:
        probs: Dict[str, float] = {}
        for lbl in PAULI_LABELS_1Q:
            P = PAULIS_1Q[lbl]
            val = sum(abs(0.5 * np.trace(P @ K)) ** 2 for K in kraus_ops)
            probs[lbl] = float(max(0.0, val))
        total = sum(probs.values())
        if total > 0:
            for k in probs:
                probs[k] /= total
        return probs

    if num_qubits == 2:
        probs_2q: Dict[str, float] = {}
        for lbl in PAULI_LABELS_2Q:
            P = np.kron(PAULIS_1Q[lbl[0]], PAULIS_1Q[lbl[1]])
            val = sum(abs(0.25 * np.trace(P @ K)) ** 2 for K in kraus_ops)
            probs_2q[lbl] = float(max(0.0, val))
        total = sum(probs_2q.values())
        if total > 0:
            for k in probs_2q:
                probs_2q[k] /= total
        return probs_2q

    raise ValueError(f"pauli_twirl_kraus supports 1 or 2 qubits, got {num_qubits}")


def pauli_twirl_superoperator(
    choi_or_kraus: Sequence[np.ndarray],
    ideal_unitary: np.ndarray,
) -> Dict[str, float]:
    """
    Given the noisy evolution Kraus operators {E_m} of a gate and its ideal unitary U,
    extract the post-gate error channel E_err = E ∘ U^{-1} (with Kraus {E_m U^†})
    and return its Pauli-twirled probabilities for Stim injection.
    """
    n_qubits = int(round(math.log2(ideal_unitary.shape[0])))
    u_dag = ideal_unitary.conj().T
    err_kraus = [K @ u_dag for K in choi_or_kraus]
    return pauli_twirl_kraus(err_kraus, num_qubits=n_qubits)


@dataclass
class ReadoutError:
    """
    Single-qubit measurement assignment error model:
      - p1_given_0: probability of measuring '1' when state is |0>
      - p0_given_1: probability of measuring '0' when state is |1>
    """

    p1_given_0: float = 0.01
    p0_given_1: float = 0.02

    def confusion_matrix(self) -> np.ndarray:
        """2x2 column-stochastic readout confusion matrix M where p_noisy = M @ p_ideal."""
        return np.array(
            [
                [1.0 - self.p1_given_0, self.p0_given_1],
                [self.p1_given_0, 1.0 - self.p0_given_1],
            ],
            dtype=np.float64,
        )

    def symmetric_stim_probability(self) -> float:
        """Effective bit-flip readout probability for Stim M(p) instruction."""
        return 0.5 * (self.p1_given_0 + self.p0_given_1)
