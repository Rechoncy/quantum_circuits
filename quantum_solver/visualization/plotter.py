"""
Visualization tools for quantum state evolution trajectories, Bloch vectors,
fidelity/purity curves, and multi-backend measurement comparison plots.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..solver.base import EvolutionResult


def plot_evolution_and_measurements(
    results: Dict[str, EvolutionResult],
    save_path: str,
    title: str = "Quantum Circuit Step-by-Step Evolution & Noisy Measurement Comparison",
) -> str:
    """
    Generate a 3-panel summary figure comparing step-by-step evolution and final
    measurement probabilities across backends (e.g. QuTiP vs Kraus vs Stim).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: Qubit-0 Bloch Vector Evolution (<X>, <Y>, <Z>)
    ax0 = axes[0]
    primary_key = "qutip" if "qutip" in results else next(iter(results.keys()))
    primary_res = results[primary_key]

    has_pulse_traj = any(
        s.pulse_times_ns is not None and len(s.pulse_times_ns) > 1
        for s in primary_res.steps
    )
    if has_pulse_traj:
        t_all = []
        bx_all, by_all, bz_all = [], [], []
        for s in primary_res.steps:
            if s.pulse_times_ns is not None and s.pulse_bloch_trajectories is not None and 0 in s.pulse_bloch_trajectories:
                t_all.extend(s.pulse_times_ns.tolist())
                traj = s.pulse_bloch_trajectories[0]
                bx_all.extend(traj[:, 0].tolist())
                by_all.extend(traj[:, 1].tolist())
                bz_all.extend(traj[:, 2].tolist())
        ax0.plot(t_all, bx_all, label=r"$\langle X_0(t) \rangle$", lw=2.0, color="#1f77b4")
        ax0.plot(t_all, by_all, label=r"$\langle Y_0(t) \rangle$", lw=2.0, color="#ff7f0e")
        ax0.plot(t_all, bz_all, label=r"$\langle Z_0(t) \rangle$", lw=2.0, color="#2ca02c")
        ax0.set_xlabel("Physical Time (ns)")
        ax0.set_title(f"Qubit 0 Continuous Pulse Trajectory [{primary_res.backend_name}]")
    else:
        step_ids = [s.step_index for s in primary_res.steps]
        bx = [s.bloch_vectors.get(0, (0, 0, 0))[0] for s in primary_res.steps]
        by = [s.bloch_vectors.get(0, (0, 0, 0))[1] for s in primary_res.steps]
        bz = [s.bloch_vectors.get(0, (0, 0, 0))[2] for s in primary_res.steps]
        ax0.plot(step_ids, bx, "o-", label=r"$\langle X_0 \rangle$", lw=2.0)
        ax0.plot(step_ids, by, "s-", label=r"$\langle Y_0 \rangle$", lw=2.0)
        ax0.plot(step_ids, bz, "^-", label=r"$\langle Z_0 \rangle$", lw=2.0)
        ax0.set_xlabel("Circuit Step Index")
        ax0.set_title(f"Qubit 0 Stepwise Bloch Vector [{primary_res.backend_name}]")

    ax0.set_ylim(-1.08, 1.08)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    # Panel 2: Step-by-Step Fidelity & Purity Decay
    ax1 = axes[1]
    for name, res in results.items():
        steps_x = [s.step_index for s in res.steps if s.state_fidelity is not None]
        fids = [s.state_fidelity for s in res.steps if s.state_fidelity is not None]
        purs = [s.purity for s in res.steps if s.purity is not None]
        if fids and name in ("qutip", "kraus"):
            ax1.plot(steps_x, fids, "o-", lw=2.0, label=f"{name.upper()} Fidelity")
        if purs and name in ("qutip", "kraus"):
            ax1.plot(steps_x, purs, "s--", lw=1.6, alpha=0.8, label=f"{name.upper()} Purity")
    ax1.set_xlabel("Circuit Step Index")
    ax1.set_ylabel("Fidelity / Purity")
    ax1.set_title("Step-by-Step Fidelity & Purity under Noise")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Panel 3: Final Measurement Distribution across Backends
    ax2 = axes[2]
    all_bitstrings = sorted(
        {
            bstr
            for res in results.values()
            for bstr, p in res.final_probabilities.items()
            if p > 1e-4
        }
    )
    x = np.arange(len(all_bitstrings))
    width = 0.8 / max(1, len(results))
    for idx, (name, res) in enumerate(results.items()):
        probs = [res.final_probabilities.get(b, 0.0) for b in all_bitstrings]
        offset = (idx - 0.5 * (len(results) - 1)) * width
        ax2.bar(x + offset, probs, width=width, label=res.backend_name, alpha=0.88)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"|{b}>" for b in all_bitstrings], rotation=25)
    ax2.set_ylabel("Measurement Probability")
    ax2.set_title("Final Noisy Measurement Probabilities")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="best", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
