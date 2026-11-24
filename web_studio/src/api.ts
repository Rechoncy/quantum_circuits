import type {
  CircuitOpModel,
  CustomGateDef,
  NlCircuitResponse,
  OptimizeResponse,
  ProjectModel,
  QubitSpec,
  SimulationResponse,
  SweepResponse,
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

export async function validateCustomGate(
  gate: CustomGateDef,
  sample_params?: number[]
): Promise<{
  valid: boolean;
  error?: string;
  unitarity_error: number | null;
  is_clifford: boolean;
  has_hamiltonian?: boolean;
  matrix_preview: string[][];
}> {
  return postJson('/api/custom-gate/validate', { gate, sample_params });
}

export async function runSimulation(project: ProjectModel): Promise<SimulationResponse> {
  return postJson('/api/simulate', project);
}

export async function runOptimization(
  project: ProjectModel,
  evaluate_noisy_fidelity = true,
  preferred_native_2q?: string | null
): Promise<OptimizeResponse> {
  return postJson('/api/optimize', {
    project,
    evaluate_noisy_fidelity,
    preferred_native_2q: preferred_native_2q || null,
  });
}

export async function runUnitarySynthesis(payload: {
  project: ProjectModel;
  mode: 'matrix' | 'hamiltonian';
  num_qubits: number;
  matrix_exprs?: string[][];
  hamiltonian_str?: string;
  hamiltonian_scale?: number;
  evaluate_noisy_fidelity?: boolean;
  preferred_native_2q?: string | null;
}): Promise<OptimizeResponse> {
  return postJson('/api/optimize/synthesize', payload);
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
