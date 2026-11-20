import { create } from 'zustand';
import { computePumpCoeffs, computeEffCoeffs } from '../utils/polyfit';

export interface QHPoint  { flow: number; head: number; }
export interface EffPoint { flow: number; eff: number; }
export interface PeakPoint { location_km: number; elevation_m: number; }
export interface TimeSlot { start: string; end: string; rate: number; }
export interface LinefillRow { product: string; volume: number; visc: number; density: number; init_dra: number; }
export interface DayPlanRow  { product: string; volume: number; visc: number; density: number; init_dra: number; }
export interface ProjFlowRow { start: string; end: string; flow: number; }
export interface ProjPlanRow { product: string; volume: number; visc: number; density: number; }

export interface PumpConfig {
  enabled: boolean;
  available: number;
  name: string;
  qh_table: QHPoint[];
  eff_table: EffPoint[];
  power_source: 'Grid' | 'Diesel';
  min_rpm: number;
  dol_rpm: number;
  tariff_type: 'Fixed' | 'Varying';
  fixed_rate: number;
  time_slots: TimeSlot[];
  sfc_mode: 'manual' | 'iso3046';
  sfc: number;
  engine_make: string;
  engine_model: string;
  rated_power: number;
  sfc50: number;
  sfc75: number;
  sfc100: number;
}

export interface BranchStationInput {
  name: string;
  elev: number;
  L: number;
  OD: number;
  wall: number;
  SMYS: number;
  roughness: number;
  min_residual: number;
  max_dra: number;
  is_pump: boolean;
  pumpA: PumpConfig;
  pumpB: PumpConfig;
}

export interface BranchConfig {
  id: string;
  branch_name: string;
  flow_m3h: number;
  terminal_name: string;
  terminal_elev: number;
  terminal_min_pressure: number;
  stations: BranchStationInput[];
}

export interface StationData {
  id: string;
  name: string;
  elevation: number;
  is_pump: boolean;
  length_km: number;
  max_dra_pct: number;
  min_suction_head: number;
  OD_in: number;
  wall_thk_in: number;
  SMYS_psi: number;
  roughness_m: number;
  max_pumps: number;
  delivery: number;
  supply: number;
  pumpA: PumpConfig;
  pumpB: PumpConfig;
  peaks_table: PeakPoint[];
  branches: BranchConfig[];
}

export interface LooplineConfig {
  enabled: boolean;
  name: string;
  start_km: number;
  end_km: number;
  L_km: number;
  D_in: number;
  t_in: number;
  SMYS_psi: number;
  roughness_m: number;
  max_dra_pct: number;
  elev: number;
  peak_profile: PeakPoint[];
}

export interface GlobalParams {
  FLOW: number;
  op_mode: 'flow_rate' | 'daily_schedule' | 'pumping_planner';
  pipeline_category: string;
  pipeline_code: string;
  RateDRA: number;
  Price_HSD: number;
  Fuel_density: number;
  Ambient_temp: number;
  pump_shear_rate: number;
  MOP_kgcm2: number;
  dra_lacing_mode: 'auto' | 'manual';
  laced_flow: number;
  laced_visc: number;
  laced_suction: number;
  laced_density: number;
  baseline_ppm_table: Array<{segment: string; ppm: number}>;
  rpm_step: number;
  dra_step: number;
  priority_feasibility: boolean;
  collect_state_audit: boolean;
  kv_list: number[];
  rho_list: number[];
  planner_days: number;
  flow_mode: 'fixed' | 'variable';
  hourly_flow: number;
}

const defaultPumpConfig = (): PumpConfig => ({
  enabled: false,
  available: 2,
  name: '',
  qh_table: [{flow: 0, head: 150}, {flow: 250, head: 120}, {flow: 500, head: 80}],
  eff_table: [{flow: 0, eff: 0}, {flow: 100, eff: 50}, {flow: 300, eff: 75}, {flow: 450, eff: 70}, {flow: 550, eff: 55}],
  power_source: 'Grid',
  min_rpm: 1000,
  dol_rpm: 1480,
  tariff_type: 'Fixed',
  fixed_rate: 7.5,
  time_slots: [
    {start: '00:00', end: '06:00', rate: 5.0},
    {start: '06:00', end: '10:00', rate: 8.5},
    {start: '10:00', end: '18:00', rate: 9.0},
    {start: '18:00', end: '24:00', rate: 8.0},
  ],
  sfc_mode: 'manual',
  sfc: 220,
  engine_make: '',
  engine_model: '',
  rated_power: 1000,
  sfc50: 220,
  sfc75: 210,
  sfc100: 215,
});

const defaultStation = (index: number): StationData => ({
  id: crypto.randomUUID(),
  name: index === 0 ? 'Origin' : `Station ${index}`,
  elevation: 0,
  is_pump: index > 0,
  length_km: 50,
  max_dra_pct: 20,
  min_suction_head: 50,
  OD_in: 20,
  wall_thk_in: 0.375,
  SMYS_psi: 52000,
  roughness_m: 0.000046,
  max_pumps: 3,
  delivery: 0,
  supply: 0,
  pumpA: defaultPumpConfig(),
  pumpB: defaultPumpConfig(),
  peaks_table: [],
  branches: [],
});

interface PipelineStore {
  global: GlobalParams;
  stations: StationData[];
  terminal: {name: string; elevation: number; min_residual: number};
  loopline: LooplineConfig;
  linefill_table: LinefillRow[];
  day_plan: DayPlanRow[];
  proj_flow: ProjFlowRow[];
  proj_plan: ProjPlanRow[];

  setGlobal: (patch: Partial<GlobalParams>) => void;
  addStation: () => void;
  removeStation: (id: string) => void;
  updateStation: (id: string, patch: Partial<StationData>) => void;
  updatePump: (stationId: string, type: 'A' | 'B', patch: Partial<PumpConfig>) => void;
  moveStation: (id: string, dir: 'up' | 'down') => void;
  setTerminal: (t: Partial<{name: string; elevation: number; min_residual: number}>) => void;
  setLoopline: (patch: Partial<LooplineConfig>) => void;
  setLinefill: (rows: LinefillRow[]) => void;
  setDayPlan: (rows: DayPlanRow[]) => void;
  setProjFlow: (rows: ProjFlowRow[]) => void;
  setProjPlan: (rows: ProjPlanRow[]) => void;
  addBranch: (stationId: string) => void;
  removeBranch: (stationId: string, branchId: string) => void;
  updateBranch: (stationId: string, branchId: string, patch: Partial<BranchConfig>) => void;
  loadCase: (json: Record<string, unknown>) => void;
  exportCase: () => Record<string, unknown>;
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  global: {
    FLOW: 450,
    op_mode: 'flow_rate',
    pipeline_category: 'Crude Oil Pipelines',
    pipeline_code: 'PHBPL',
    RateDRA: 0,
    Price_HSD: 90,
    Fuel_density: 0.83,
    Ambient_temp: 35,
    pump_shear_rate: 0,
    MOP_kgcm2: 0,
    dra_lacing_mode: 'auto',
    laced_flow: 450,
    laced_visc: 10,
    laced_suction: 50,
    laced_density: 850,
    baseline_ppm_table: [],
    rpm_step: 50,
    dra_step: 5,
    priority_feasibility: false,
    collect_state_audit: false,
    kv_list: [10],
    rho_list: [850],
    planner_days: 7,
    flow_mode: 'fixed',
    hourly_flow: 450,
  },
  stations: [defaultStation(0), defaultStation(1)],
  terminal: {name: 'Terminal', elevation: 0, min_residual: 50},
  loopline: {
    enabled: false, name: 'Loopline', start_km: 0, end_km: 100,
    L_km: 100, D_in: 16, t_in: 0.25, SMYS_psi: 52000,
    roughness_m: 0.000046, max_dra_pct: 20, elev: 0, peak_profile: [],
  },
  linefill_table: [{product: 'Crude', volume: 5000, visc: 10, density: 850, init_dra: 0}],
  day_plan: [{product: 'Crude', volume: 10000, visc: 10, density: 850, init_dra: 0}],
  proj_flow: [{start: '2024-01-01 00:00', end: '2024-01-07 00:00', flow: 450}],
  proj_plan: [{product: 'Crude', volume: 70000, visc: 10, density: 850}],

  setGlobal: (patch) => set(s => ({global: {...s.global, ...patch}})),

  addStation: () => set(s => ({
    stations: [...s.stations, defaultStation(s.stations.length)],
  })),

  removeStation: (id) => set(s => ({
    stations: s.stations.filter(st => st.id !== id),
  })),

  updateStation: (id, patch) => set(s => ({
    stations: s.stations.map(st => st.id === id ? {...st, ...patch} : st),
  })),

  updatePump: (stationId, type, patch) => set(s => ({
    stations: s.stations.map(st =>
      st.id === stationId
        ? type === 'A'
          ? {...st, pumpA: {...st.pumpA, ...patch}}
          : {...st, pumpB: {...st.pumpB, ...patch}}
        : st
    ),
  })),

  moveStation: (id, dir) => set(s => {
    const idx = s.stations.findIndex(st => st.id === id);
    if (idx < 0) return s;
    const newStations = [...s.stations];
    if (dir === 'up' && idx > 0) [newStations[idx], newStations[idx-1]] = [newStations[idx-1], newStations[idx]];
    if (dir === 'down' && idx < newStations.length - 1) [newStations[idx], newStations[idx+1]] = [newStations[idx+1], newStations[idx]];
    return {stations: newStations};
  }),

  setTerminal: (t) => set(s => ({terminal: {...s.terminal, ...t}})),
  setLoopline: (patch) => set(s => ({loopline: {...s.loopline, ...patch}})),
  setLinefill: (rows) => set({linefill_table: rows}),
  setDayPlan: (rows) => set({day_plan: rows}),
  setProjFlow: (rows) => set({proj_flow: rows}),
  setProjPlan: (rows) => set({proj_plan: rows}),

  addBranch: (stationId) => set(s => ({
    stations: s.stations.map(st => st.id === stationId
      ? {...st, branches: [...st.branches, {
          id: crypto.randomUUID(),
          branch_name: `Branch ${st.branches.length + 1}`,
          flow_m3h: 50,
          terminal_name: 'Branch Terminal',
          terminal_elev: 0,
          terminal_min_pressure: 30,
          stations: [],
        }]}
      : st
    ),
  })),

  removeBranch: (stationId, branchId) => set(s => ({
    stations: s.stations.map(st => st.id === stationId
      ? {...st, branches: st.branches.filter(b => b.id !== branchId)}
      : st
    ),
  })),

  updateBranch: (stationId, branchId, patch) => set(s => ({
    stations: s.stations.map(st => st.id === stationId
      ? {...st, branches: st.branches.map(b => b.id === branchId ? {...b, ...patch} : b)}
      : st
    ),
  })),

  loadCase: (json) => {
    const g = json as Record<string, unknown>;
    // Restore global params
    const global: Partial<GlobalParams> = {
      FLOW: (g.FLOW as number) || 450,
      op_mode: (g.op_mode as GlobalParams['op_mode']) || 'flow_rate',
      RateDRA: (g.RateDRA as number) || 0,
      Price_HSD: (g.Price_HSD as number) || 90,
      Fuel_density: (g.Fuel_density as number) || 0.83,
      Ambient_temp: (g.Ambient_temp as number) || 35,
      MOP_kgcm2: (g.MOP_kgcm2 as number) || 0,
      pump_shear_rate: (g.pump_shear_rate as number) || 0,
      kv_list: (g.KV_list as number[]) || [10],
      rho_list: (g.rho_list as number[]) || [850],
    };
    set(s => ({global: {...s.global, ...global}}));
    // Restore terminal
    if (g.terminal) {
      const t = g.terminal as Record<string, unknown>;
      set({terminal: {name: (t.name as string) || 'Terminal', elevation: (t.elev as number) || 0, min_residual: (t.min_residual as number) || 50}});
    }
    if (g.linefill) set({linefill_table: g.linefill as LinefillRow[]});
    if (g.day_plan) set({day_plan: g.day_plan as DayPlanRow[]});
  },

  exportCase: () => {
    const {global: g, stations, terminal, loopline, linefill_table, day_plan, proj_flow, proj_plan} = get();

    const serializedStations = stations.map(s => {
      const pump_types: Record<string, unknown> = {};

      for (const [key, pmp] of Object.entries({A: s.pumpA, B: s.pumpB})) {
        if (!pmp.enabled) continue;
        const coeffsQH = computePumpCoeffs(pmp.qh_table);
        const coeffsEff = computeEffCoeffs(pmp.eff_table);
        pump_types[key] = {
          available: pmp.available,
          ...coeffsQH,
          ...coeffsEff,
          MinRPM: pmp.min_rpm,
          DOL: pmp.dol_rpm,
          power_type: pmp.power_source,
          rate: pmp.tariff_type === 'Fixed' ? pmp.fixed_rate : 0,
          tariffs: pmp.tariff_type === 'Varying' ? pmp.time_slots.map(ts => ({start: ts.start, end: ts.end, rate: ts.rate})) : [],
          sfc: pmp.sfc,
          sfc_mode: pmp.sfc_mode,
          engine_params: pmp.sfc_mode === 'iso3046' ? {
            make: pmp.engine_make, model: pmp.engine_model,
            rated_power: pmp.rated_power, sfc50: pmp.sfc50, sfc75: pmp.sfc75, sfc100: pmp.sfc100,
          } : {},
        };
      }

      return {
        name: s.name,
        elev: s.elevation,
        is_pump: s.is_pump,
        L: s.length_km,
        max_dra_pct: s.max_dra_pct,
        min_suction_head: s.min_suction_head,
        OD: s.OD_in,
        wall_thk: s.wall_thk_in,
        SMYS: s.SMYS_psi,
        roughness: s.roughness_m,
        max_pumps: s.max_pumps,
        delivery: s.delivery,
        supply: s.supply,
        pump_types,
        peaks: s.peaks_table,
        loopline: loopline.enabled ? {
          name: loopline.name, start_km: loopline.start_km, end_km: loopline.end_km,
          L: loopline.L_km, D: loopline.D_in, t: loopline.t_in, SMYS: loopline.SMYS_psi,
          roughness: loopline.roughness_m, max_dra_pct: loopline.max_dra_pct, elev: loopline.elev,
          peak_profile: loopline.peak_profile,
        } : null,
      };
    });

    const branches = stations.flatMap(s => s.branches.map(b => ({
      name: b.branch_name,
      flow_m3h: b.flow_m3h,
      terminal: {name: b.terminal_name, elev: b.terminal_elev, min_residual: b.terminal_min_pressure},
      stations: b.stations,
      uid: b.id,
    })));

    return {
      stations: serializedStations,
      terminal: {name: terminal.name, elev: terminal.elevation, min_residual: terminal.min_residual},
      FLOW: g.FLOW,
      KV_list: g.kv_list,
      rho_list: g.rho_list,
      RateDRA: g.RateDRA,
      Price_HSD: g.Price_HSD,
      Fuel_density: g.Fuel_density,
      Ambient_temp: g.Ambient_temp,
      MOP_kgcm2: g.MOP_kgcm2,
      op_mode: g.op_mode,
      linefill: linefill_table,
      day_plan,
      proj_flow,
      proj_plan,
      pump_shear_rate: g.pump_shear_rate,
      rpm_step: g.rpm_step,
      dra_step: g.dra_step,
      priority_feasibility: g.priority_feasibility,
      collect_state_audit: g.collect_state_audit,
      planner_days: g.planner_days,
      branches,
    };
  },
}));
