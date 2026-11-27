"""
FastAPI Backend Server for Quantum Studio v2.
Serves both the REST API (/api/*) and the built React static bundle (web_studio/dist/).
"""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import os
import pathlib
import sys
from typing import Any, Dict, Optional
import json
import re

from pydantic import BaseModel

os.environ.setdefault("MPMATH_NOGMPY", "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..gui.sweep_engine import HARDWARE_SWEEP_PARAMS
from .custom_gates import (
    evaluate_custom_gate_unitary,
    format_matrix_preview,
    validate_unitary_matrix,
)
from .benchmarking import execute_gate_benchmark
from .models import (
    BenchmarkHardwareRequest,
    DslSyncRequest,
    GateBenchmarkRequest,
    NlCircuitRequest,
    NoiseBudgetRequest,
    OptimizeRequest,
    ProjectModel,
    QasmSyncRequest,
    SweepRequest,
    SynthesizeUnitaryRequest,
    ValidateCustomGateRequest,
    VariationalOptimizeRequest,
    ZneRequest,
)
from .nl_circuit import generate_circuit_from_nl
from .project_engine import (
    build_preset_projects,
    dsl_to_project_ops,
    execute_cross_hardware_benchmark,
    execute_noise_budget,
    execute_project_optimization,
    execute_project_simulation,
    execute_project_sweep,
    execute_unitary_synthesis,
    execute_variational_optimization,
    execute_zne_mitigation,
    openqasm_to_project_ops,
    project_to_dsl,
    project_to_openqasm3,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "web_studio" / "dist"


class SaveFilePayload(BaseModel):
    filename: str
    content: str
    is_base64: bool = False


class ImportProjectPayload(BaseModel):
    filename: str = "imported.qproj"
    content: str
    base_project: Optional[ProjectModel] = None


class _OPENFILENAMEW(ctypes.Structure):
    _fields_ = [
        ("lStructSize", wintypes.DWORD),
        ("hwndOwner", wintypes.HWND),
        ("hInstance", wintypes.HINSTANCE),
        ("lpstrFilter", ctypes.c_void_p),
        ("lpstrCustomFilter", ctypes.c_void_p),
        ("nMaxCustFilter", wintypes.DWORD),
        ("nFilterIndex", wintypes.DWORD),
        ("lpstrFile", ctypes.c_void_p),
        ("nMaxFile", wintypes.DWORD),
        ("lpstrFileTitle", ctypes.c_void_p),
        ("nMaxFileTitle", wintypes.DWORD),
        ("lpstrInitialDir", wintypes.LPCWSTR),
        ("lpstrTitle", wintypes.LPCWSTR),
        ("Flags", wintypes.DWORD),
        ("nFileOffset", wintypes.WORD),
        ("nFileExtension", wintypes.WORD),
        ("lpstrDefExt", wintypes.LPCWSTR),
        ("lCustData", wintypes.LPARAM),
        ("lpfnHook", ctypes.c_void_p),
        ("lpTemplateName", wintypes.LPCWSTR),
        ("pvReserved", ctypes.c_void_p),
        ("dwReserved", wintypes.DWORD),
        ("FlagsEx", wintypes.DWORD),
    ]


def _auto_guard_backend(n_qubits: int, current_backend: str) -> str:
    if n_qubits > 24 and current_backend in ("qutip", "kraus", "trajectory"):
        return "stim"
    if n_qubits > 14 and current_backend in ("qutip", "kraus"):
        return "trajectory"
    if n_qubits > 9 and current_backend == "qutip":
        return "kraus"
    return current_backend


def parse_and_normalize_imported_project(
    filename: str,
    content: str,
    base_project: Optional[ProjectModel] = None,
) -> tuple[str, ProjectModel]:
    """
    Parse and schema-normalize an imported file into a valid `ProjectModel`.
    Supports:
      1. `.qproj` / `.json` (both versioned bundle `{ "project": {...} }` and direct/legacy `ProjectModel` JSON)
      2. `.qasm` (OpenQASM 2.0 / 3.0)
      3. `.dsl` / `.txt` (Quantum Studio DSL script)
    """
    raw = (content or "").lstrip("\ufeff").strip()
    if not raw:
        raise ValueError("导入文件内容为空")

    fallback = (base_project or build_preset_projects()[0]).model_copy(deep=True)
    fb_dump = fallback.model_dump()
    safe_stem = re.sub(
        r"\.(qproj\.json|qproj|json|qasm|dsl|txt)$",
        "",
        pathlib.Path(filename or "Imported").name,
        flags=re.IGNORECASE,
    ) or "Imported_Circuit"

    # 1. JSON / .qproj project file
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"JSON 语法解析失败: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("工程 JSON 顶层结构必须为对象")

        proj_dict = parsed.get("project") if isinstance(parsed.get("project"), dict) else parsed
        raw_qubits = proj_dict.get("qubits")
        if not isinstance(raw_qubits, list) or len(raw_qubits) == 0:
            raise ValueError("工程文件缺少有效的量子比特寄存器 (qubits)")

        palette = ["#FF6B6B", "#0A84FF", "#34C759", "#FF9F0A", "#AF52DE", "#00C7BE", "#FF2D55", "#5856D6"]
        norm_qubits = []
        for idx, q in enumerate(raw_qubits):
            qd = dict(q) if isinstance(q, dict) else {}
            qd.setdefault("id", idx)
            qd["id"] = idx
            qd.setdefault("name", f"q{idx}")
            qd.setdefault("color", palette[idx % len(palette)])
            qd.setdefault("init_state", "|0>")
            qd.setdefault("init_theta", 0.0)
            qd.setdefault("init_phi", 0.0)
            qd.setdefault("pos_x", round((idx % 8) * 4.5, 2))
            qd.setdefault("pos_y", round((idx // 8) * 4.5, 2))
            norm_qubits.append(qd)

        raw_ops = proj_dict.get("operations") if isinstance(proj_dict.get("operations"), list) else []
        norm_ops = []
        for idx, op in enumerate(raw_ops):
            if not isinstance(op, dict):
                continue
            od = dict(op)
            od.setdefault("id", f"imp_op_{idx}")
            od.setdefault("step", idx)
            od.setdefault("kind", "measure" if str(od.get("name", "")).upper() == "MEASURE" else "gate")
            od.setdefault("params", [])
            norm_ops.append(od)

        hw_list = proj_dict.get("hardware_profiles")
        if not isinstance(hw_list, list) or len(hw_list) == 0:
            hw_list = fb_dump["hardware_profiles"]
        hw_ids = {h.get("id") for h in hw_list if isinstance(h, dict)}
        active_hw = proj_dict.get("active_hardware_id")
        if not active_hw or active_hw not in hw_ids:
            active_hw = hw_list[0].get("id", fb_dump["active_hardware_id"])

        sim_merged = dict(fb_dump["sim_settings"])
        if isinstance(proj_dict.get("sim_settings"), dict):
            sim_merged.update(proj_dict["sim_settings"])
        sim_merged["backend"] = _auto_guard_backend(len(norm_qubits), str(sim_merged.get("backend", "kraus")))

        raw_name = str(proj_dict.get("name") or f"{safe_stem}.qproj").strip()
        if not raw_name.endswith(".qproj"):
            raw_name = re.sub(r"\.(json|qproj\.json)$", "", raw_name, flags=re.IGNORECASE) + ".qproj"

        merged = {
            "name": raw_name,
            "qubits": norm_qubits,
            "custom_gates": proj_dict.get("custom_gates") if isinstance(proj_dict.get("custom_gates"), list) else [],
            "operations": norm_ops,
            "variables": proj_dict.get("variables") if isinstance(proj_dict.get("variables"), dict) else {},
            "hardware_profiles": hw_list,
            "active_hardware_id": active_hw,
            "sim_settings": sim_merged,
        }
        return "qproj", ProjectModel.model_validate(merged)

    # 2. OpenQASM 2.0 / 3.0 (.qasm)
    lower_fn = (filename or "").lower()
    is_qasm = lower_fn.endswith(".qasm") or bool(
        re.search(r"^\s*(OPENQASM|include\s+\"|qreg\s+\w+\s*\[|qubit\s*\[)", raw, re.IGNORECASE | re.MULTILINE)
    )
    if is_qasm:
        qubits, ops, variables = openqasm_to_project_ops(raw, fallback)
        fallback.name = f"{safe_stem}.qproj"
        fallback.qubits = qubits
        fallback.operations = ops
        fallback.variables = variables
        fallback.sim_settings.backend = _auto_guard_backend(len(qubits), fallback.sim_settings.backend)  # type: ignore[assignment]
        return "qasm", fallback

    # 3. Quantum Studio DSL (.dsl / .txt)
    qubits, ops, variables = dsl_to_project_ops(raw, fallback)
    has_qubits_decl = bool(re.search(r"^\s*qubits\s+\d+", raw, re.IGNORECASE | re.MULTILINE))
    if len(ops) == 0 and not has_qubits_decl:
        raise ValueError("无法识别的文件内容：支持导入 .qproj / .json 工程文件、OpenQASM (.qasm) 或 Quantum DSL (.dsl/.txt)")
    fallback.name = f"{safe_stem}.qproj"
    fallback.qubits = qubits
    fallback.operations = ops
    fallback.variables = variables
    fallback.sim_settings.backend = _auto_guard_backend(len(qubits), fallback.sim_settings.backend)  # type: ignore[assignment]
    return "dsl", fallback


def create_app() -> FastAPI:
    app = FastAPI(title="Quantum Studio API", version="2.8.1")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": "2.8.1"}

    @app.get("/api/presets")
    def get_presets() -> Dict[str, Any]:
        presets = build_preset_projects()
        return {
            "presets": [p.model_dump() for p in presets],
            "hardware_sweep_params": HARDWARE_SWEEP_PARAMS,
        }

    @app.post("/api/sync/project-to-dsl")
    def api_project_to_dsl(project: ProjectModel) -> Dict[str, Any]:
        try:
            return {
                "dsl": project_to_dsl(project),
                "openqasm3": project_to_openqasm3(project),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sync/dsl-to-project")
    def api_dsl_to_project(req: DslSyncRequest) -> Dict[str, Any]:
        try:
            qubits, ops, variables = dsl_to_project_ops(req.dsl, req.project)
            return {
                "qubits": [q.model_dump() for q in qubits],
                "operations": [op.model_dump() for op in ops],
                "variables": variables,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sync/qasm-to-project")
    def api_qasm_to_project(req: QasmSyncRequest) -> Dict[str, Any]:
        try:
            qubits, ops, variables = openqasm_to_project_ops(req.qasm, req.project)
            return {
                "qubits": [q.model_dump() for q in qubits],
                "operations": [op.model_dump() for op in ops],
                "variables": variables,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/custom-gate/validate")
    def api_validate_custom_gate(req: ValidateCustomGateRequest) -> Dict[str, Any]:
        try:
            cg_map = {cg.name.upper(): cg for cg in req.existing_custom_gates}
            U, H = evaluate_custom_gate_unitary(req.gate, req.sample_params, custom_gate_map=cg_map)
            is_unitary, err, is_cliff = validate_unitary_matrix(U)
            return {
                "valid": is_unitary,
                "unitarity_error": err,
                "is_clifford": is_cliff,
                "has_hamiltonian": H is not None,
                "matrix_preview": format_matrix_preview(U),
            }
        except Exception as exc:
            return {
                "valid": False,
                "error": str(exc),
                "unitarity_error": None,
                "is_clifford": False,
                "matrix_preview": [],
            }

    @app.post("/api/simulate")
    def api_simulate(project: ProjectModel) -> Dict[str, Any]:
        try:
            return execute_project_simulation(project)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/optimize")
    def api_optimize(req: OptimizeRequest) -> Dict[str, Any]:
        try:
            return execute_project_optimization(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/optimize/synthesize")
    def api_optimize_synthesize(req: SynthesizeUnitaryRequest) -> Dict[str, Any]:
        try:
            return execute_unitary_synthesis(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/optimize/benchmark")
    def api_optimize_benchmark(req: BenchmarkHardwareRequest) -> Dict[str, Any]:
        try:
            return execute_cross_hardware_benchmark(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sweep")
    def api_sweep(req: SweepRequest) -> Dict[str, Any]:
        try:
            return execute_project_sweep(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/vqe-optimize")
    def api_vqe_optimize(req: VariationalOptimizeRequest) -> Dict[str, Any]:
        try:
            return execute_variational_optimization(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/zne")
    def api_zne_mitigation(req: ZneRequest) -> Dict[str, Any]:
        try:
            return execute_zne_mitigation(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/noise-budget")
    def api_noise_budget(req: NoiseBudgetRequest) -> Dict[str, Any]:
        try:
            return execute_noise_budget(req.project, req.backend)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/benchmark-gates")
    def api_benchmark_gates(req: GateBenchmarkRequest) -> Dict[str, Any]:
        try:
            return execute_gate_benchmark(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/nl-to-circuit")
    def api_nl_to_circuit(req: NlCircuitRequest) -> Dict[str, Any]:
        try:
            return generate_circuit_from_nl(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/project/import")
    def api_project_import(req: ImportProjectPayload) -> Dict[str, Any]:
        try:
            fmt, normalized_proj = parse_and_normalize_imported_project(
                req.filename, req.content, req.base_project
            )
            return {
                "format_detected": fmt,
                "project": normalized_proj.model_dump(),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/system/open-file")
    def api_system_open_file() -> Dict[str, Any]:
        """Thread-safe native Win32 Open-File dialog for desktop mode (`QuantumStudio.exe`)."""
        if sys.platform != "win32" or os.environ.get("QUANTUM_STUDIO_DESKTOP") != "1":
            return {"desktop": False}

        raw_filter = (
            "所有支持的量子工程与线路 (*.qproj;*.json;*.qasm;*.dsl;*.txt)\0*.qproj;*.json;*.qasm;*.dsl;*.txt\0"
            "Quantum Studio 工程文件 (*.qproj;*.json)\0*.qproj;*.json\0"
            "OpenQASM 2.0/3.0 量子线路 (*.qasm)\0*.qasm\0"
            "Quantum DSL 线路脚本 (*.dsl;*.txt)\0*.dsl;*.txt\0"
            "所有文件 (*.*)\0*.*\0\0"
        )
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 2)
        except Exception:
            pass

        file_buf = ctypes.create_unicode_buffer("", 4096)
        flt_buf = ctypes.create_unicode_buffer(raw_filter)
        downloads_dir = pathlib.Path.home() / "Downloads"
        init_dir: Optional[str] = str(downloads_dir) if downloads_dir.is_dir() else None

        ofn = _OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
        ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
        ofn.lpstrFilter = ctypes.cast(flt_buf, ctypes.c_void_p)
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.cast(file_buf, ctypes.c_void_p)
        ofn.nMaxFile = 4096
        ofn.lpstrInitialDir = init_dir
        ofn.lpstrTitle = "导入工程或量子线路 — Quantum Studio EDA"
        # OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER
        ofn.Flags = 0x00000800 | 0x00001000 | 0x00000008 | 0x00080000

        ok = ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn))
        if not ok:
            return {"desktop": True, "opened": False, "cancelled": True}

        chosen_path = pathlib.Path(file_buf.value)
        if not chosen_path.is_file():
            return {"desktop": True, "opened": False, "cancelled": True}

        try:
            text = chosen_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = chosen_path.read_text(encoding="gbk", errors="replace")

        return {
            "desktop": True,
            "opened": True,
            "filename": chosen_path.name,
            "path": str(chosen_path),
            "content": text,
        }

    @app.post("/api/system/save-file")
    def api_system_save_file(req: SaveFilePayload) -> Dict[str, Any]:
        """Thread-safe native Win32 Save-As dialog for desktop mode (avoids pywebview js_api deadlocks)."""
        if sys.platform != "win32" or os.environ.get("QUANTUM_STUDIO_DESKTOP") != "1":
            return {"desktop": False}

        safe_name = pathlib.Path(req.filename).name or "export.qproj"
        ext = pathlib.Path(safe_name).suffix.lower()
        filter_labels = {
            ".qproj": ("Quantum Studio 工程文件 (*.qproj)", "*.qproj"),
            ".json": ("Quantum Studio JSON 工程 (*.json)", "*.json"),
            ".qasm": ("OpenQASM 3.0 量子线路 (*.qasm)", "*.qasm"),
            ".dsl": ("Quantum Studio DSL 脚本 (*.dsl)", "*.dsl"),
            ".svg": ("SVG 矢量原理图/图表 (*.svg)", "*.svg"),
            ".tex": ("LaTeX Quantikz 论文源码 (*.tex)", "*.tex"),
            ".csv": ("CSV 实验数据表 (*.csv)", "*.csv"),
            ".md": ("Markdown 标定报告 (*.md)", "*.md"),
            ".png": ("PNG 高清图像 (*.png)", "*.png"),
        }
        desc, pat = filter_labels.get(ext, ("导出文件 (*.*)", "*.*"))
        raw_filter = f"{desc}\0{pat}\0所有文件 (*.*)\0*.*\0\0"

        try:
            ctypes.windll.ole32.CoInitializeEx(None, 2)
        except Exception:
            pass

        file_buf = ctypes.create_unicode_buffer(safe_name, 4096)
        flt_buf = ctypes.create_unicode_buffer(raw_filter)
        downloads_dir = pathlib.Path.home() / "Downloads"
        init_dir: Optional[str] = str(downloads_dir) if downloads_dir.is_dir() else None

        ofn = _OPENFILENAMEW()
        ofn.lStructSize = ctypes.sizeof(_OPENFILENAMEW)
        ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
        ofn.lpstrFilter = ctypes.cast(flt_buf, ctypes.c_void_p)
        ofn.nFilterIndex = 1
        ofn.lpstrFile = ctypes.cast(file_buf, ctypes.c_void_p)
        ofn.nMaxFile = 4096
        ofn.lpstrInitialDir = init_dir
        ofn.lpstrTitle = "导出保存文件 — Quantum Studio EDA"
        ofn.lpstrDefExt = ext.lstrip(".") if ext else None
        # OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER
        ofn.Flags = 0x00000002 | 0x00000800 | 0x00000008 | 0x00080000

        ok = ctypes.windll.comdlg32.GetSaveFileNameW(ctypes.byref(ofn))
        if not ok:
            return {"desktop": True, "saved": False, "cancelled": True}

        chosen_path = pathlib.Path(file_buf.value)
        if ext and not chosen_path.suffix:
            chosen_path = chosen_path.with_suffix(ext)
        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        if req.is_base64:
            chosen_path.write_bytes(base64.b64decode(req.content))
        else:
            chosen_path.write_text(req.content, encoding="utf-8")
        return {"desktop": True, "saved": True, "path": str(chosen_path)}

    # Serve built React frontend if available
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        def serve_spa(full_path: str) -> FileResponse:
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(FRONTEND_DIST / "index.html"))

    return app


app = create_app()
