"""
Build self-contained Windows Setup executable: `release/QuantumStudio_v2.0.0_Setup.exe`.

Combines:
  1. Compiled C# WinForms Setup stub (`InstallerStub.exe`, with custom icon & GUI/silent installer)
  2. Compressed ZIP archive of Quantum Studio v2 (including embedded `python_env/`, `web_studio/dist/`,
     `quantum_solver/`, `assets/`, `examples/`, and `QuantumStudio.exe`)
  3. 24-byte QSTUDIO2 trailer (`<QQ8s`: payload_offset, payload_length, b"QSTUDIO2")
"""

from __future__ import annotations

import os
import struct
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = ROOT / "release"
STUB_EXE = RELEASE_DIR / "InstallerStub.exe"
PAYLOAD_ZIP = RELEASE_DIR / "payload_temp.zip"
FINAL_INSTALLER = RELEASE_DIR / "QuantumStudio_v2.0.0_Setup.exe"

CSC_EXE = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")


def build_installer() -> None:
    t0 = time.time()
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compile C# WinForms Setup Stub with embedded icon
    print("[1/3] Compiling C# InstallerStub.exe...")
    subprocess.run(
        [
            str(CSC_EXE),
            "/nologo",
            "/target:winexe",
            "/optimize+",
            f"/win32icon:{ROOT / 'assets' / 'quantum_studio.ico'}",
            "/r:System.IO.Compression.dll",
            "/r:System.IO.Compression.FileSystem.dll",
            f"/out:{STUB_EXE}",
            str(ROOT / "assets" / "QuantumStudioInstaller.cs"),
        ],
        check=True,
    )

    # 2. Pack release files into compressed payload ZIP
    print("[2/3] Packing project + embedded python_env into compressed payload...")
    include_files = [
        "QuantumStudio.exe",
        "run_studio.py",
        "run_gui.py",
        "start_studio.bat",
        "start_gui.bat",
        "pyproject.toml",
        "README.md",
    ]
    include_dirs = [
        "assets",
        "quantum_solver",
        "examples",
        "web_studio/dist",
        "python_env",
    ]

    file_count = 0
    with zipfile.ZipFile(PAYLOAD_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fname in include_files:
            fpath = ROOT / fname
            if fpath.is_file():
                zf.write(fpath, fname)
                file_count += 1

        for dname in include_dirs:
            dpath = ROOT / dname
            if not dpath.is_dir():
                continue
            for dp, dirnames, filenames in os.walk(dpath):
                # Skip __pycache__
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in filenames:
                    if fn.endswith((".pyc", ".pdb")):
                        continue
                    full_p = Path(dp) / fn
                    rel_p = full_p.relative_to(ROOT).as_posix()
                    zf.write(full_p, rel_p)
                    file_count += 1

    # 3. Concatenate Stub + Payload + 24-byte QSTUDIO2 Footer into QuantumStudio_v2.0.0_Setup.exe
    print(f"[3/3] Assembling single-file installer ({file_count} files)...")
    stub_bytes = STUB_EXE.read_bytes()
    payload_offset = len(stub_bytes)
    payload_length = PAYLOAD_ZIP.stat().st_size

    with open(FINAL_INSTALLER, "wb") as out_f:
        out_f.write(stub_bytes)
        with open(PAYLOAD_ZIP, "rb") as pf:
            while True:
                chunk = pf.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
        footer = struct.pack("<QQ8s", payload_offset, payload_length, b"QSTUDIO2")
        out_f.write(footer)

    STUB_EXE.unlink(missing_ok=True)
    PAYLOAD_ZIP.unlink(missing_ok=True)

    size_mb = FINAL_INSTALLER.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"Done! Built {FINAL_INSTALLER} ({size_mb:.1f} MB, {file_count} files) in {elapsed:.1f}s.")


if __name__ == "__main__":
    build_installer()
