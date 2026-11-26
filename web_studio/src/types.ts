export type PrimaryDomain = 'circuits' | 'characterization' | 'device';
export type CircuitSubMode = 'design' | 'simulate' | 'optimize' | 'sweep';
export type DeviceSubTab = 'drive' | 'qubits' | 'topology' | 'platforms';
export type BenchmarkProtocol = 'rabi' | 'ramsey' | 't1' | 'rb' | 'allxy' | 'gst' | 'xeb';

export type WorkMode = 'design' | 'simulate' | 'optimize' | 'sweep' | 'benchmark' | 'device';

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
  rabi_mhz?: number | null;
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
  rabi_mhz?: number | null;
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
  pulse_shape: 'cosine' | 'gaussian' | 'rectangular' | 'trapezoid';
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
  pulse_shape: 'cosine' | 'gaussian' | 'rectangular' | 'trapezoid';
  drive_mode?: 'fixed_duration' | 'fixed_rabi';
  rabi_mhz?: number;
  rise_ns?: number;
  fall_ns?: number;
}

export type SolverBackendType = 'qutip' | 'kraus' | 'trajectory' | 'stim' | 'stim_qutip' | 'compare';

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
  bipartite_entropy?: number | null;
  mutual_information?: number[][] | null;
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
  hamiltonian_str?: string | null;
  x_values: number[];
  populations: Record<string, number[]>;
  fidelities: number[];
  purities: number[];
  bloch_q0: [number, number, number][];
  energies?: number[] | null;
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
  hamiltonian_str?: string | null;
  x_values: number[];
  y_values: number[];
  population_grid: number[][];
  fidelity_grid: number[][];
  z0_grid: number[][];
  energy_grid?: number[][] | null;
}

export type SweepResponse = Sweep1DResponse | Sweep2DResponse;

export interface VariationalOptimizeResponse {
  method: 'Parameter-Shift' | 'COBYLA' | 'Powell' | 'Nelder-Mead' | 'L-BFGS-B';
  backend: string;
  hamiltonian_str: string;
  param_names: string[];
  initial_params: Record<string, number>;
  optimal_params: Record<string, number>;
  initial_energy: number;
  optimal_energy: number;
  energy_improvement: number;
  optimal_fidelity: number;
  num_evaluations: number;
  energy_trace: number[];
  best_so_far_trace: number[];
  dominant_states: { bitstring: string; probability: number }[];
}

export interface ZnePerQubitRow {
  qubit: number;
  name: string;
  noisy: number;
  zne_linear: number;
  zne_richardson: number;
  ideal: number;
}

export interface ZneResponse {
  backend: string;
  scale_factors: number[];
  fit_x: number[];
  observed_qubit: number;
  observed_qubit_name: string;
  fidelities_by_scale: number[];
  z_by_scale: number[];
  z_unmitigated: number;
  z_zne_linear: number;
  z_zne_richardson: number;
  z_ideal: number;
  z_linear_curve: number[];
  z_richardson_curve: number[];
  hamiltonian_str?: string | null;
  energy_by_scale?: number[] | null;
  energy_unmitigated?: number | null;
  energy_zne_linear?: number | null;
  energy_zne_richardson?: number | null;
  energy_ideal?: number | null;
  energy_linear_curve?: number[] | null;
  energy_richardson_curve?: number[] | null;
  per_qubit_z: ZnePerQubitRow[];
}

export interface NoiseBudgetItem {
  id: string;
  label: string;
  infidelity: number;
  share_pct: number;
  detail: string;
}

export interface NoiseBudgetResponse {
  backend: string;
  hardware_name: string;
  num_qubits: number;
  full_fidelity: number;
  total_state_infidelity: number;
  readout_tvd: number;
  sum_isolated_errors: number;
  bottleneck_summary: string;
  contributions: NoiseBudgetItem[];
}

export interface RbInterleavedResult {
  gate_name: string;
  mean_survival: number[];
  std_survival: number[];
  raw_survival: number[][];
  fit_y: number[];
  p_irb: number;
  a_irb: number;
  b_irb: number;
  gate_error: number;
  gate_fidelity: number;
  systematic_bound: number;
}

export interface CharacterizationCircuitSample {
  id: string;
  label: string;
  subtitle?: string;
  gate_id?: string;
  operations: CircuitOpModel[];
}

export interface RbBenchmarkResponse {
  protocol: 'rb';
  hardware_name: string;
  target_label: string;
  num_qubits: 1 | 2;
  dimension: number;
  lengths: number[];
  num_seeds: number;
  shots: number;
  fit_x: number[];
  mean_survival: number[];
  std_survival: number[];
  raw_survival: number[][];
  fit_y: number[];
  a_spam: number;
  b_offset: number;
  depolarizing_p: number;
  epc: number;
  clifford_fidelity: number;
  coherence_epc_floor: number;
  interleaved: RbInterleavedResult | null;
  sample_operations: CircuitOpModel[];
  circuit_samples?: CharacterizationCircuitSample[];
}

export interface AllXySequenceRow {
  index: number;
  label: string;
  gate_first: string;
  gate_second: string;
  plateau: 'ground' | 'equator' | 'excited';
  ideal_z: number;
  measured_z: number;
  std_z: number;
  ideal_p1: number;
  measured_p1: number;
  residual_z: number;
  operations: CircuitOpModel[];
}

export interface AllXyBenchmarkResponse {
  protocol: 'allxy';
  hardware_name: string;
  target_label: string;
  shots: number;
  rms_error_z: number;
  estimated_amp_error_pct: number;
  phase_detuning_metric: number;
  diagnosis: string;
  severity: 'good' | 'warn' | 'info';
  sequences: AllXySequenceRow[];
}

export interface GstGateReport {
  gate_id: string;
  gate_token: string;
  label: string;
  process_fidelity: number;
  avg_gate_fidelity: number;
  gate_infidelity: number;
  diamond_distance: number;
  unitarity: number;
  incoherent_infidelity: number;
  coherent_ratio: number;
  ptm_ideal: number[][];
  ptm_reconstructed: number[][];
  ptm_error: number[][];
}

export interface GstBenchmarkResponse {
  protocol: 'gst';
  hardware_name: string;
  target_label: string;
  pauli_labels: string[];
  shots: number;
  max_germ_power: number;
  germ_powers: number[];
  germ_curves: Record<string, number[]>;
  num_experiments: number;
  mean_gate_fidelity: number;
  max_diamond_distance: number;
  mean_unitarity: number;
  spam_prep_bloch: [number, number, number];
  spam_prep_fidelity: number;
  spam_readout_fidelity: number;
  gram_matrix: number[][];
  gates: GstGateReport[];
  sample_operations: CircuitOpModel[];
  circuit_samples?: CharacterizationCircuitSample[];
}

export interface XebBenchmarkResponse {
  protocol: 'xeb';
  hardware_name: string;
  target_label: string;
  entangler_name: string;
  lengths: number[];
  num_seeds: number;
  shots: number;
  fit_x: number[];
  fit_y: number[];
  mean_xeb: number[];
  std_xeb: number[];
  a_spam: number;
  cycle_polarization: number;
  cycle_pauli_error: number;
  entangler_fidelity: number;
  speckle_ideal: number[];
  speckle_measured: number[];
  sample_operations: CircuitOpModel[];
  circuit_samples?: CharacterizationCircuitSample[];
}

export interface BasicCalibrationParam {
  key: string;
  label: string;
  value: number;
  unit: string;
}

export interface CalibrationWritebackItem {
  scope: 'qubit' | 'hardware';
  qubit?: number;
  field: 'rabi_mhz' | 't1_us' | 't2_us' | 'freq_ghz' | 't_1q_ns';
  value: number;
  label: string;
}

export interface BasicCalibrationResponse {
  protocol: 'rabi' | 'ramsey' | 't1';
  hardware_name: string;
  target_label: string;
  qubit: number;
  shots: number;
  x: number[];
  measured: number[];
  ideal: number[];
  fit_x: number[];
  fit_y: number[];
  x_label: string;
  y_label: string;
  params: BasicCalibrationParam[];
  writeback: CalibrationWritebackItem[];
  circuit_samples?: CharacterizationCircuitSample[];
  summary: string;
}

export type GateBenchmarkResponse =
  | RbBenchmarkResponse
  | AllXyBenchmarkResponse
  | GstBenchmarkResponse
  | XebBenchmarkResponse
  | BasicCalibrationResponse;

export interface HardwareBenchmarkEntry {
  hardware_id: string;
  hardware_name: string;
  platform: HardwarePlatformType;
  gate_set_name: string;
  primary_2q_gate: string;
  unitary_fidelity: number;
  total_gates: number;
  num_2q_gates: number;
  num_1q_physical_pulses: number;
  num_1q_virtual_gates: number;
  depth: number;
  duration_ns: number;
  expected_gate_fidelity: number;
  gate_breakdown: Record<string, number>;
  optimized_operations: CircuitOpModel[];
}

export interface HardwareBenchmarkResponse {
  num_qubits: number;
  original_gate_count: number;
  recommended_hardware_id: string;
  entries: HardwareBenchmarkEntry[];
}

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
  noise_before?: number;
  noise_after?: number;
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
