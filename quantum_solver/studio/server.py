"""
FastAPI Backend Server for Quantum Studio v2.
Serves both the REST API (/api/*) and the built React static bundle (web_studio/dist/).
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict

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


def create_app() -> FastAPI:
    app = FastAPI(title="Quantum Studio API", version="2.7.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": "2.7.0"}

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
