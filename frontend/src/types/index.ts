export type JobStatus = 'pending' | 'running' | 'success' | 'failed'

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface User {
  id: number
  email: string
  name: string
  created_at: string
}

export interface OptimizeRequest {
  stations: Record<string, unknown>[]
  terminal: Record<string, unknown>
  FLOW: number
  KV_list: number[]
  rho_list: number[]
  segment_slices?: Record<string, unknown>[][] | null
  RateDRA: number
  Price_HSD: number
  Fuel_density: number
  Ambient_temp: number
  linefill?: Record<string, unknown>[] | null
  dra_reach_km?: number
  mop_kgcm2?: number | null
  hours?: number
  start_time?: string
  pump_shear_rate?: number
  branches?: BranchInput[] | null
  rpm_step?: number | null
  dra_step?: number | null
  priority_feasibility?: boolean
}

export interface BranchInput {
  flow_m3h: number
  stations: Record<string, unknown>[]
  terminal: Record<string, unknown>
  uid?: string
  name?: string
  KV?: number
  rho?: number
}

export interface OptimizeResponse {
  job_id: string
  status: JobStatus
}

export interface JobResult {
  job_id: string
  status: JobStatus
  result?: SolverResult | null
  error?: string | null
  created_at: string
  finished_at?: string | null
}

export interface SolverResult {
  mainline: MainlineResult
  branches: BranchResult[]
}

export interface MainlineResult {
  error?: boolean | string
  total_cost?: number
  dra_front_km?: number
  dra_segments?: DraSegment[]
  stations_used?: StationInput[]
  executed_passes?: string[]
  [key: string]: unknown  // per-station keys like residual_head_in_ps1
}

export interface BranchResult {
  branch_uid: string
  branch_name: string
  junction_head: number
  total_cost?: number
  error?: boolean
  [key: string]: unknown
}

export interface DraSegment {
  length_km: number
  dra_ppm: number
}

export interface StationInput {
  name: string
  elev?: number
  km_from_origin?: number
  [key: string]: unknown
}

export interface Pipeline {
  id: number
  name: string
  description: string
  config: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface PipelineCreate {
  name: string
  description?: string
  config: Record<string, unknown>
}

export interface JobSummary {
  job_id: string
  status: JobStatus
  created_at: string
  finished_at?: string | null
}

export interface JobListResponse {
  jobs: JobSummary[]
  total: number
}

// Derived types for chart data

export interface StationPoint {
  name: string
  km: number
  arrival_head: number    // residual_head_in_{name}
  discharge_head: number  // sdh_{name}
  maop: number            // maop_{name}
  elev: number
  num_pumps: number
  speed: number
  efficiency: number
  motor_kw: number
  power_cost: number
  dra_cost: number
  dra_ppm: number
  drag_reduction: number
}
