"""Quantum Studio v2 unified project data model, custom gate engine, and FastAPI backend."""

from .models import (
    CircuitOpModel,
    CustomGateDef,
    HardwareProfileModel,
    NlCircuitRequest,
    OptimizeRequest,
    ProjectModel,
    QubitSpec,
    SimulationSettingsModel,
    SweepRequest,
    SynthesizeUnitaryRequest,
)
from .nl_circuit import generate_circuit_from_nl
from .project_engine import (
    build_preset_projects,
    compile_project_circuit,
    dsl_to_project_ops,
    execute_project_optimization,
    execute_project_simulation,
    execute_project_sweep,
    execute_unitary_synthesis,
    project_to_dsl,
    project_to_openqasm3,
)
from .server import app, create_app

__all__ = [
    "ProjectModel",
    "QubitSpec",
    "CircuitOpModel",
    "CustomGateDef",
    "HardwareProfileModel",
    "SimulationSettingsModel",
    "OptimizeRequest",
    "SynthesizeUnitaryRequest",
    "SweepRequest",
    "NlCircuitRequest",
    "generate_circuit_from_nl",
    "build_preset_projects",
    "compile_project_circuit",
    "project_to_dsl",
    "dsl_to_project_ops",
    "project_to_openqasm3",
    "execute_project_simulation",
    "execute_project_optimization",
    "execute_unitary_synthesis",
    "execute_project_sweep",
    "create_app",
    "app",
]
