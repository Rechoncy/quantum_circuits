"""
Top-level Launcher for the Quantum Circuit Solver & Optimal Synthesizer GUI Workbench.

Usage:
  1. Interactive GUI Mode:
       C:\\Software\\miniconda3\\envs\\qutip\\python.exe run_gui.py
  2. Screenshot Verification Mode:
       C:\\Software\\miniconda3\\envs\\qutip\\python.exe run_gui.py --screenshot ./screenshots
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Ensure Windows Conda DLL paths are registered before importing Qt/QuTiP
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

from quantum_solver.gui.app import launch_gui


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Quantum Circuit Solver & Optimizer GUI Workbench")
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Optional output directory to automatically capture screenshots of all 4 GUI partitions and exit.",
    )
    args = parser.parse_args()
    return launch_gui(screenshot_dir=args.screenshot)


if __name__ == "__main__":
    sys.exit(main())
