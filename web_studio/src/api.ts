import type {
  CircuitOpModel,
  CustomGateDef,
  GateBenchmarkResponse,
  HardwareBenchmarkResponse,
  NlCircuitResponse,
  OptimizeResponse,
  ProjectModel,
  QubitSpec,
  SimulationResponse,
  SweepResponse,
  VariationalOptimizeResponse,
  ZneResponse,
} from './types';

async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchPresets(): Promise<{
  presets: ProjectModel[];
  hardware_sweep_params: Record<string, string>;
}> {
  const res = await fetch('/api/presets');
  if (!res.ok) throw new Error(`加载预设失败: HTTP ${res.status}`);
  return res.json();
}

export async function syncProjectToDsl(project: ProjectModel): Promise<{
  dsl: string;
  openqasm3: string;
}> {
  return postJson('/api/sync/project-to-dsl', project);
}

export async function syncDslToProject(
  dsl: string,
  project: ProjectModel
): Promise<{
  qubits: QubitSpec[];
  operations: CircuitOpModel[];
  variables: Record<string, number>;
}> {
  return postJson('/api/sync/dsl-to-project', { dsl, project });
}

export async function syncQasmToProject(
  qasm: string,
  project: ProjectModel
): Promise<{
  qubits: QubitSpec[];
  operations: CircuitOpModel[];
  variables: Record<string, number>;
}> {
  return postJson('/api/sync/qasm-to-project', { qasm, project });
}

export async function validateCustomGate(
  gate: CustomGateDef,
  sample_params?: number[],
  existing_custom_gates?: CustomGateDef[]
): Promise<{
  valid: boolean;
  error?: string;
  unitarity_error: number | null;
  is_clifford: boolean;
  has_hamiltonian?: boolean;
  matrix_preview: string[][];
}> {
  return postJson('/api/custom-gate/validate', {
    gate,
    sample_params,
    existing_custom_gates: existing_custom_gates || [],
  });
}

export async function runSimulation(
  project: ProjectModel,
  signal?: AbortSignal
): Promise<SimulationResponse> {
  return postJson('/api/simulate', project, signal);
}

export async function runOptimization(
  project: ProjectModel,
  evaluate_noisy_fidelity = true,
  preferred_native_2q?: string | null,
  signal?: AbortSignal
): Promise<OptimizeResponse> {
  return postJson(
    '/api/optimize',
    {
      project,
      evaluate_noisy_fidelity,
      preferred_native_2q: preferred_native_2q || null,
    },
    signal
  );
}

export async function runUnitarySynthesis(
  payload: {
    project: ProjectModel;
    mode: 'matrix' | 'hamiltonian';
    num_qubits: number;
    matrix_exprs?: string[][];
    hamiltonian_str?: string;
    hamiltonian_scale?: number;
    evaluate_noisy_fidelity?: boolean;
    preferred_native_2q?: string | null;
  },
  signal?: AbortSignal
): Promise<OptimizeResponse> {
  return postJson('/api/optimize/synthesize', payload, signal);
}

export async function runCrossHardwareBenchmark(
  project: ProjectModel,
  signal?: AbortSignal
): Promise<HardwareBenchmarkResponse> {
  return postJson('/api/optimize/benchmark', { project }, signal);
}

export async function runVariationalOptimization(
  payload: {
    project: ProjectModel;
    hamiltonian_str: string;
    param_names?: string[];
    method: 'Parameter-Shift' | 'COBYLA' | 'Powell' | 'Nelder-Mead' | 'L-BFGS-B';
    max_iter: number;
    backend: 'qutip' | 'kraus' | 'trajectory' | 'stim' | 'stim_qutip';
  },
  signal?: AbortSignal
): Promise<VariationalOptimizeResponse> {
  return postJson('/api/vqe-optimize', payload, signal);
}

export async function runZneMitigation(
  payload: {
    project: ProjectModel;
    scale_factors?: number[];
    observed_qubit?: number;
    hamiltonian_str?: string;
    backend?: 'qutip' | 'kraus' | 'trajectory' | 'stim' | 'stim_qutip';
  },
  signal?: AbortSignal
): Promise<ZneResponse> {
  return postJson('/api/zne', payload, signal);
}

export async function runGateBenchmark(
  payload: {
    project: ProjectModel;
    protocol: 'rb' | 'allxy' | 'gst';
    target_qubit: number;
    second_qubit?: number;
    rb_num_qubits?: 1 | 2;
    rb_lengths?: number[];
    rb_num_seeds?: number;
    interleaved_gate?: string | null;
    gst_max_length?: number;
    shots?: number;
    amp_error_pct?: number;
    detuning_mhz?: number;
    phase_skew_deg?: number;
    seed?: number;
  },
  signal?: AbortSignal
): Promise<GateBenchmarkResponse> {
  return postJson('/api/benchmark-gates', payload, signal);
}

export async function runSweep(
  payload: {
    project: ProjectModel;
    mode: '1d' | '2d';
    param_x: string;
    x_start: number;
    x_stop: number;
    x_points: number;
    param_y?: string;
    y_start?: number;
    y_stop?: number;
    y_points?: number;
    target_bitstring?: string;
    observed_qubit?: number;
    hamiltonian_str?: string;
    backend: 'qutip' | 'kraus' | 'trajectory' | 'stim' | 'stim_qutip';
  },
  signal?: AbortSignal
): Promise<SweepResponse> {
  return postJson('/api/sweep', payload, signal);
}

export async function generateCircuitFromNl(payload: {
  prompt: string;
  project: ProjectModel;
  mode: 'auto' | 'replace' | 'refine' | 'append';
  engine: 'local' | 'llm';
  history?: { role: string; content: string }[];
  llm_base_url?: string;
  llm_api_key?: string;
  llm_model?: string;
}): Promise<NlCircuitResponse> {
  return postJson('/api/nl-to-circuit', payload);
}
