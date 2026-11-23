export type WorkMode = 'design' | 'simulate' | 'optimize' | 'sweep';

export type InitStateType = '|0>' | '|1>' | '|+>' | '|->' | '|+i>' | '|-i>' | 'custom';

export interface QubitSpec {
  id: number;
  name: string;
  color: string;
  init_state: InitStateType;
  init_theta: number;
  init_phi: number;
  t1_us: number | null;
  t2_us: number | null;
  freq_ghz: number | null;
  depol_1q: number | null;
  readout_err: number | null;
  pos_x: number;
  pos_y: number;
}

export interface CircuitOpModel {
  id: string;
  step: number;
  kind: 'gate' | 'noise' | 'measure';
  name: string;
  qubits: number[];
  params: (number | string)[];
  duration_ns?: number | null;
  noise_mode?: 'inherit' | 'none' | 'custom';
  custom_depol?: number | null;
}

export type CustomGateDefinitionType = 'matrix' | 'expression' | 'hamiltonian' | 'composite';

export interface CustomGateDef {
  id: string;
  name: string;
  display_label: string;
  num_qubits: number;
  definition_type: CustomGateDefinitionType;
  matrix_exprs?: string[][] | null;
  expression?: string | null;
  param_names: string[];
  default_params: number[];
  hamiltonian_str?: string | null;
  hamiltonian_scale: number;
  pulse_shape: 'cosine' | 'gaussian' | 'rectangular';
  sub_operations: CircuitOpModel[];
  duration_ns: number;
  infidelity: number;
  color: string;
  is_native: boolean;
  description: string;
}

export type HardwarePlatformType =
  | 'neutral_atom'
  | 'superconducting_cz'
  | 'superconducting_iswap'
  | 'superconducting_cnot'
  | 'trapped_ion';

export interface HardwareProfileModel {
  id: string;
  name: string;
  platform: HardwarePlatformType;
  t1_us: number;
  t2_us: number;
  t_1q_ns: number;
  t_2q_ns: number;
  depol_1q: number;
  depol_2q: number;
  zz_khz: number;
  readout_err: number;
  topology_mode: 'all' | 'linear' | 'custom' | 'rydberg_radius';
  edges: [number, number][];
  blockade_radius_um: number;
  pulse_shape: 'cosine' | 'gaussian' | 'rectangular';
}

export type SolverBackendType = 'qutip' | 'kraus' | 'stim' | 'stim_qutip' | 'compare';

export interface SimulationSettingsModel {
  backend: SolverBackendType;
  shots: number;
  enable_t1_t2: boolean;
  enable_depol: boolean;
  enable_zz: boolean;
  enable_readout: boolean;
  intra_gate_steps: number;
}

export interface ProjectModel {
  name: string;
  qubits: QubitSpec[];
  custom_gates: CustomGateDef[];
  operations: CircuitOpModel[];
  variables: Record<string, number>;
  hardware_profiles: HardwareProfileModel[];
  active_hardware_id: string;
  sim_settings: SimulationSettingsModel;
}

export interface StepSnapshotJson {
  step_index: number;
  operation_label: string;
  elapsed_time_ns: number;
  state_fidelity: number | null;
  purity: number | null;
  bloch_vectors: Record<string, [number, number, number]>;
  populations: Record<string, number>;
  stabilizers: string[] | null;
  pulse_times_ns: number[] | null;
  pulse_bloch_trajectories: Record<string, [number, number, number][]> | null;
  density_matrix_abs: number[][] | null;
  density_matrix_real: number[][] | null;
  density_matrix_imag: number[][] | null;
}

export interface EvolutionResultJson {
  backend_name: string;
  num_qubits: number;
  total_duration_ns: number;
  shots: number;
  final_fidelity: number | null;
  final_purity: number | null;
  final_probabilities: Record<string, number>;
  measurement_counts: Record<string, number>;
  stim_circuit_str: string | null;
  steps: StepSnapshotJson[];
}

export interface SimulationResponse {
  primary_backend: string;
  warnings: string[];
  hardware_name: string;
  topology_edges: [number, number][];
  results: Record<string, EvolutionResultJson>;
}

export interface CircuitMetricsJson {
  total_gates: number;
  num_2q_gates: number;
  num_1q_physical_pulses: number;
  num_1q_virtual_gates: number;
  depth: number;
  duration_ns: number;
  expected_gate_fidelity: number;
  gate_breakdown: Record<string, number>;
}

export interface OptimizeResponse {
  gate_set_name: string;
  primary_2q_gate: string;
  unitary_fidelity: number;
  noisy_fidelity_original: number | null;
  noisy_fidelity_optimized: number | null;
  original_metrics: CircuitMetricsJson | null;
  optimized_metrics: CircuitMetricsJson;
  optimized_operations: CircuitOpModel[];
  optimized_dsl: string;
  optimized_qasm: string;
  summary_text: string;
  target_num_qubits?: number;
}

export interface Sweep1DResponse {
  mode: '1d';
  param_x: string;
  param_x_label: string;
  backend: string;
  observed_qubit?: number;
  observed_qubit_name?: string;
  x_values: number[];
  populations: Record<string, number[]>;
  fidelities: number[];
  purities: number[];
  bloch_q0: [number, number, number][];
}

export interface Sweep2DResponse {
  mode: '2d';
  param_x: string;
  param_x_label: string;
  param_y: string;
  param_y_label: string;
  backend: string;
  observed_qubit?: number;
  observed_qubit_name?: string;
  target_bitstring: string;
  x_values: number[];
  y_values: number[];
  population_grid: number[][];
  fidelity_grid: number[][];
  z0_grid: number[][];
}

export type SweepResponse = Sweep1DResponse | Sweep2DResponse;

export interface RunHistoryItem {
  id: string;
  label: string;
  timestamp: string;
  backend: string;
  num_qubits: number;
  fidelity: number | null;
  duration_ns: number;
  response: SimulationResponse;
}

export type SelectionTarget =
  | { type: 'none' }
  | { type: 'op'; opId: string }
  | { type: 'qubit'; qubitId: number }
  | { type: 'hardware'; hardwareId: string }
  | { type: 'custom_gate'; gateId: string };

export interface NlCircuitDiffSummary {
  qubits_before: number;
  qubits_after: number;
  gates_before: number;
  gates_after: number;
  noise_before: number;
  noise_after: number;
}

export interface NlCircuitResponse {
  title: string;
  explanation: string;
  engine_used: string;
  action_mode: 'replace' | 'refine' | 'append';
  changes: string[];
  unrecognized_clauses: string[];
  diff_summary: NlCircuitDiffSummary | null;
  dsl: string;
  openqasm3: string;
  qubits: QubitSpec[];
  operations: CircuitOpModel[];
  variables: Record<string, number>;
  recommended_hardware_id: string | null;
  recommended_backend: SolverBackendType;
}
