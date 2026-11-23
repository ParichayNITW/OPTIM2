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
  saveCase: () => Record<string, unknown>;
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  global: {
    FLOW: 1000,
    op_mode: 'flow_rate',
    pipeline_category: 'Crude Oil Pipelines',
    pipeline_code: 'PHBPL',
    RateDRA: 0,
    Price_HSD: 70,
    Fuel_density: 0.820,
    Ambient_temp: 25,
    pump_shear_rate: 0,
    MOP_kgcm2: 100,
    dra_lacing_mode: 'auto',
    laced_flow: 1000,
    laced_visc: 10,
    laced_suction: 50,
    laced_density: 850,
    baseline_ppm_table: [],
    rpm_step: 25,
    dra_step: 2,
    priority_feasibility: false,
    collect_state_audit: false,
    kv_list: [10],
    rho_list: [850],
    planner_days: 7,
    flow_mode: 'fixed',
    hourly_flow: 1000,
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

    const mapPump = (pt: Record<string, unknown> | null | undefined, forceEnabled: boolean): PumpConfig => {
      if (!pt || !pt.head_data) return {...defaultPumpConfig(), enabled: false};
      const headData = pt.head_data as Array<Record<string, number>>;
      const effData = (pt.eff_data as Array<Record<string, number>>) || [];
      return {
        ...defaultPumpConfig(),
        enabled: forceEnabled,
        available: (pt.available as number) ?? 1,
        name: (pt.name as string) || '',
        qh_table: headData.map(r => ({ flow: r['Flow (m³/hr)'] ?? r.flow ?? 0, head: r['Head (m)'] ?? r.head ?? 0 })),
        eff_table: effData.map(r => ({ flow: r['Flow (m³/hr)'] ?? r.flow ?? 0, eff: r['Efficiency (%)'] ?? r.eff ?? 0 })),
        power_source: ((pt.power_type as string) === 'Diesel' ? 'Diesel' : 'Grid') as 'Grid' | 'Diesel',
        min_rpm: (pt.MinRPM as number) || 1000,
        dol_rpm: (pt.DOL as number) || 1480,
        fixed_rate: (pt.rate as number) || 7.5,
        sfc: (pt.sfc as number) || 220,
        sfc_mode: (pt.sfc_mode as string) === 'iso3046' ? 'iso3046' : 'manual',
      };
    };

    const opModeRaw = (g.op_mode as string) || 'flow_rate';
    const opMode: GlobalParams['op_mode'] =
      (opModeRaw.toLowerCase().includes('daily') || opModeRaw.toLowerCase().includes('schedule')) ? 'daily_schedule' :
      opModeRaw.toLowerCase().includes('planner') ? 'pumping_planner' : 'flow_rate';

    const global: Partial<GlobalParams> = {
      FLOW: (g.FLOW as number) || 1000,
      op_mode: opMode,
      pipeline_category: (g.pipeline_category as string) || 'Crude Oil Pipelines',
      pipeline_code: (g.pipeline_code as string) || '',
      RateDRA: (g.RateDRA as number) || 0,
      Price_HSD: (g.Price_HSD as number) || 70,
      Fuel_density: (g.Fuel_density as number) || 820,
      Ambient_temp: (g.Ambient_temp as number) || 25,
      MOP_kgcm2: ((g.MOP_kgcm2 ?? g.mop_kgcm2) as number) || 100,
      pump_shear_rate: (g.pump_shear_rate as number) || 0,
      kv_list: ((g.KV_list ?? g.kv_list) as number[]) || [10],
      rho_list: (g.rho_list as number[]) || [850],
      dra_lacing_mode: (g.dra_lacing_mode as GlobalParams['dra_lacing_mode']) || 'auto',
      laced_flow: ((g.max_laced_flow_m3h ?? g.laced_flow) as number) || 1000,
      laced_visc: ((g.max_laced_visc_cst ?? g.laced_visc) as number) || 10,
      laced_suction: ((g.min_laced_suction_m ?? g.laced_suction) as number) || 0,
      laced_density: ((g.laced_density_kgm3 ?? g.laced_density) as number) || 820,
      hourly_flow: (g.hourly_flow as number) || 1000,
      rpm_step: (g.rpm_step as number) || 25,
      dra_step: (g.dra_step as number) || 2,
      priority_feasibility: Boolean(g.priority_feasibility),
      collect_state_audit: Boolean(g.collect_state_audit),
      planner_days: (g.planner_days as number) || 1,
    };
    set(s => ({ global: {...s.global, ...global} }));

    if (g.terminal) {
      const t = g.terminal as Record<string, unknown>;
      set({ terminal: { name: (t.name as string) || 'Terminal', elevation: (t.elev as number) || 0, min_residual: (t.min_residual as number) || 50 } });
    }

    if (Array.isArray(g.stations)) {
      const stations: StationData[] = (g.stations as Array<Record<string, unknown>>).map(stn => {
        const pumpTypes = (stn.pump_types as Record<string, Record<string, unknown>>) || {};
        const ptA = pumpTypes.A || null;
        const ptB = pumpTypes.B || null;
        const D = (stn.D as number) || 0.4;
        const t = (stn.t as number) || 0.006;
        return {
          id: crypto.randomUUID(),
          name: (stn.name as string) || 'Station',
          elevation: (stn.elev as number) || 0,
          is_pump: Boolean(stn.is_pump),
          length_km: (stn.L as number) || 50,
          max_dra_pct: (stn.max_dr as number) ?? 20,
          min_suction_head: (stn.min_residual as number) || 0,
          OD_in: D / 0.0254,
          wall_thk_in: t / 0.0254,
          SMYS_psi: (stn.SMYS as number) || 52000,
          roughness_m: (stn.rough as number) || 0.000046,
          max_pumps: (stn.max_pumps as number) || 3,
          delivery: (stn.delivery as number) || 0,
          supply: (stn.supply as number) || 0,
          pumpA: mapPump(ptA, Boolean(ptA && ptA.head_data)),
          pumpB: mapPump(ptB && (ptB.head_data !== null) ? ptB : null, Boolean(ptB && ptB.head_data && (ptB.available as number) > 0)),
          peaks_table: [],
          branches: [],
        };
      });
      set({ stations });
    }

    if (Array.isArray(g.linefill)) {
      set({ linefill_table: (g.linefill as Array<Record<string, unknown>>).map(r => ({
        product: ((r.Product ?? r.product) as string) || '',
        volume: ((r['Volume (m³)'] ?? r.volume) as number) || 0,
        visc: ((r['Viscosity (cSt)'] ?? r.visc) as number) || 10,
        density: ((r['Density (kg/m³)'] ?? r.density) as number) || 850,
        init_dra: ((r['Initial DRA (ppm)'] ?? r.init_dra) as number) || 0,
      })) });
    }

    if (Array.isArray(g.day_plan)) {
      set({ day_plan: (g.day_plan as Array<Record<string, unknown>>).map(r => ({
        product: ((r.Product ?? r.product) as string) || '',
        volume: ((r['Volume (m³)'] ?? r.volume) as number) || 0,
        visc: ((r['Viscosity (cSt)'] ?? r.visc) as number) || 10,
        density: ((r['Density (kg/m³)'] ?? r.density) as number) || 850,
        init_dra: ((r['Initial DRA (ppm)'] ?? r.init_dra) as number) || 0,
      })) });
    }
  },

  saveCase: () => {
    const {global: g, stations, terminal, linefill_table, day_plan} = get();
    const serializedStations = stations.map(s => {
      const pump_types: Record<string, unknown> = {};
      if (s.pumpA.enabled && s.pumpA.qh_table.length > 0) {
        const qh = computePumpCoeffs(s.pumpA.qh_table);
        const ef = computeEffCoeffs(s.pumpA.eff_table);
        pump_types['A'] = {
          available: s.pumpA.available, name: s.pumpA.name,
          head_data: s.pumpA.qh_table.map(p => ({'Flow (m³/hr)': p.flow, 'Head (m)': p.head})),
          eff_data: s.pumpA.eff_table.map(p => ({'Flow (m³/hr)': p.flow, 'Efficiency (%)': p.eff})),
          power_type: s.pumpA.power_source, MinRPM: s.pumpA.min_rpm, DOL: s.pumpA.dol_rpm,
          rate: s.pumpA.fixed_rate, tariffs: [], sfc_mode: s.pumpA.sfc_mode, sfc: s.pumpA.sfc, engine_params: {},
          ...qh, ...ef,
        };
      }
      if (s.pumpB.enabled && s.pumpB.qh_table.length > 0) {
        const qh = computePumpCoeffs(s.pumpB.qh_table);
        const ef = computeEffCoeffs(s.pumpB.eff_table);
        pump_types['B'] = {
          available: s.pumpB.available, name: s.pumpB.name,
          head_data: s.pumpB.qh_table.map(p => ({'Flow (m³/hr)': p.flow, 'Head (m)': p.head})),
          eff_data: s.pumpB.eff_table.map(p => ({'Flow (m³/hr)': p.flow, 'Efficiency (%)': p.eff})),
          power_type: s.pumpB.power_source, MinRPM: s.pumpB.min_rpm, DOL: s.pumpB.dol_rpm,
          rate: s.pumpB.fixed_rate, tariffs: [], sfc_mode: s.pumpB.sfc_mode, sfc: s.pumpB.sfc, engine_params: {},
          ...qh, ...ef,
        };
      }
      return {
        name: s.name, elev: s.elevation, is_pump: s.is_pump, L: s.length_km,
        D: s.OD_in * 0.0254, t: s.wall_thk_in * 0.0254,
        SMYS: s.SMYS_psi, rough: s.roughness_m,
        max_dr: s.max_dra_pct, min_residual: s.min_suction_head,
        max_pumps: s.max_pumps, delivery: s.delivery, supply: s.supply,
        pump_types, branches: [],
      };
    });
    return {
      stations: serializedStations,
      terminal: {name: terminal.name, elev: terminal.elevation, min_residual: terminal.min_residual},
      FLOW: g.FLOW, RateDRA: g.RateDRA, Price_HSD: g.Price_HSD,
      Fuel_density: g.Fuel_density, Ambient_temp: g.Ambient_temp,
      MOP_kgcm2: g.MOP_kgcm2, op_mode: g.op_mode, hourly_flow: g.hourly_flow,
      pipeline_category: g.pipeline_category, pipeline_code: g.pipeline_code,
      rpm_step: g.rpm_step, dra_step: g.dra_step,
      pump_shear_rate: g.pump_shear_rate,
      priority_feasibility: g.priority_feasibility,
      collect_state_audit: g.collect_state_audit,
      planner_days: g.planner_days,
      linefill: linefill_table.map(r => ({'Product': r.product, 'Volume (m³)': r.volume, 'Viscosity (cSt)': r.visc, 'Density (kg/m³)': r.density, 'Initial DRA (ppm)': r.init_dra})),
      day_plan: day_plan.map(r => ({'Product': r.product, 'Volume (m³)': r.volume, 'Viscosity (cSt)': r.visc, 'Density (kg/m³)': r.density, 'Initial DRA (ppm)': r.init_dra})),
    };
  },

  exportCase: () => {
    const {global: g, stations, terminal, loopline, linefill_table, day_plan, proj_flow, proj_plan} = get();

    // Build loopline config once with correct field names and unit conversions
    const looplineConfigForSolver = loopline.enabled ? {
      name: loopline.name,
      start_km: loopline.start_km,
      end_km: loopline.end_km,
      L: loopline.L_km,
      D: loopline.D_in * 0.0254,        // inches → meters
      t: loopline.t_in * 0.0254,         // inches → meters
      SMYS: loopline.SMYS_psi,
      rough: loopline.roughness_m,       // correct key for solver
      max_dr: loopline.max_dra_pct,      // correct key for solver
      elev: loopline.elev,
      peaks: loopline.peak_profile,
    } : null;

    // Compute cumulative KP for loopline station mapping
    let cumKp = 0;
    const cumKpAtStation: number[] = stations.map(s => {
      const kp = cumKp;
      cumKp += s.length_km;
      return kp;
    });

    const serializedStations = stations.map((s, idx) => {
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

      // Assign loopline only to the station whose cumulative KP is within [start_km, end_km)
      // This prevents the same loopline from being added to every station
      const stationKp = cumKpAtStation[idx];
      const nextKp = idx < stations.length - 1 ? cumKpAtStation[idx + 1] : stationKp + s.length_km;
      const hasLoopline = looplineConfigForSolver !== null &&
        stationKp < loopline.end_km &&
        nextKp > loopline.start_km;

      return {
        name: s.name,
        elev: s.elevation,
        is_pump: s.is_pump,
        L: s.length_km,
        max_dr: s.max_dra_pct,          // correct key: solver reads stn.get('max_dr')
        min_residual: s.min_suction_head, // correct key: solver reads stn.get('min_residual')
        D: s.OD_in * 0.0254,            // inches → meters; solver reads stn.get('D')
        t: s.wall_thk_in * 0.0254,      // inches → meters; solver reads stn.get('t')
        SMYS: s.SMYS_psi,
        rough: s.roughness_m,            // correct key: solver reads stn.get('rough')
        max_pumps: s.max_pumps,
        delivery: s.delivery,
        supply: s.supply,
        pump_types,
        peaks: s.peaks_table,
        loopline: hasLoopline ? looplineConfigForSolver : null,
      };
    });

    // Broadcast KV_list and rho_list to match station count (solver indexes by position)
    const nStations = serializedStations.length;
    const kvRaw = g.kv_list?.length > 0 ? g.kv_list : [10];
    const rhoRaw = g.rho_list?.length > 0 ? g.rho_list : [850];
    const lastKV = kvRaw[kvRaw.length - 1] ?? 10;
    const lastRho = rhoRaw[rhoRaw.length - 1] ?? 850;
    const KV_list = [...kvRaw, ...Array.from({length: Math.max(0, nStations - kvRaw.length)}, () => lastKV)];
    const rho_list = [...rhoRaw, ...Array.from({length: Math.max(0, nStations - rhoRaw.length)}, () => lastRho)];

    const branches = stations.flatMap(s => s.branches.map(b => ({
      name: b.branch_name,
      flow_m3h: b.flow_m3h,
      terminal: {name: b.terminal_name, elev: b.terminal_elev, min_residual: b.terminal_min_pressure},
      stations: b.stations,
      uid: b.id,
      KV: KV_list[0],   // pass fluid viscosity to branch solver
      rho: rho_list[0],  // pass fluid density to branch solver
    })));

    return {
      stations: serializedStations,
      terminal: {name: terminal.name, elev: terminal.elevation, min_residual: terminal.min_residual},
      FLOW: g.FLOW,
      KV_list,
      rho_list,
      RateDRA: g.RateDRA,
      Price_HSD: g.Price_HSD,
      Fuel_density: g.Fuel_density,
      Ambient_temp: g.Ambient_temp,
      mop_kgcm2: g.MOP_kgcm2,          // correct snake_case: backend expects mop_kgcm2
      op_mode: g.op_mode,
      linefill: linefill_table.map(r => ({'Product': r.product, 'Volume (m³)': r.volume, 'Viscosity (cSt)': r.visc, 'Density (kg/m³)': r.density, 'Initial DRA (ppm)': r.init_dra ?? 0})),
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
