/**
 * Parses the flat-key solver result dict into a structured ParsedResult.
 * Solver stores per-station metrics as: {metric}_{name.lower().replace(' ','_')}
 * e.g. sdh_origin, power_cost_station_1, dra_ppm_pump_station
 */

export interface ParsedStation {
  name: string;
  key: string;
  elevation: number;
  length_km: number;
  is_pump: boolean;
  // Solver output — zero when not in result
  sdh: number;              // station discharge head (m)
  num_pumps: number;        // pumps in service
  speed: number;            // pump speed RPM
  efficiency: number;       // pump efficiency %
  power_cost: number;       // power / fuel cost (₹)
  dra_cost: number;         // DRA cost (₹)
  dra_ppm: number;          // DRA dosage ppm (main pipeline)
  drag_reduction: number;   // drag reduction % (main pipeline)
  residual_head_in: number; // arriving HGL at this station (m)
  residual_head_out: number;// HGL at downstream end of this segment (m)
  maop: number;             // MAOP in metres (maop_head)
  maop_kgcm2: number;       // MAOP in kg/cm²
  head_loss: number;        // segment head loss downstream (m)
  velocity: number;         // flow velocity (m/s)
  reynolds: number;         // Reynolds number
  motor_kw: number;         // motor shaft power (kW)
  tdh: number;              // total dynamic head (m)
  pipeline_flow: number;    // actual flow in this segment (m³/h)
  coef_A: number;           // Q-H poly A (Q² coeff)
  coef_B: number;           // Q-H poly B (Q¹ coeff)
  coef_C: number;           // Q-H poly C (constant)
  rho: number;              // fluid density at station (kg/m³)
}

export interface ParsedHGLPoint {
  km: number;
  elev: number;
  label: string;
  head_in: number;   // arriving HGL before pumping
  sdh: number;       // discharge head after pumping (zero at non-pump stations)
  maop: number;      // MAOP in metres
}

export interface ParsedResult {
  flow: number;
  totalCost: number;
  totalPowerCost: number;
  totalDraCost: number;
  totalPumpsRunning: number;
  stations: ParsedStation[];
  hglData: ParsedHGLPoint[];
  terminalKm: number;
  terminalElev: number;
  terminalMinResidual: number;
  terminalHead: number;
}

// ---------------------------------------------------------------------------
// Schedule / Planner result types
// ---------------------------------------------------------------------------

export interface HourlyRow {
  hour: number;
  hour_label: string;
  step: number;
  flow: number;
  power_cost: number;
  dra_cost: number;
  total_cost: number;
  num_pumps: number;
  error: boolean;
  error_msg?: string;
  // per-station metrics — keyed by station key
  stations: Record<string, {
    num_pumps: number;
    speed: number;
    efficiency: number;
    power_cost: number;
    dra_cost: number;
    dra_ppm: number;
    drag_reduction: number;
    sdh: number;
    residual_head_in: number;
    motor_kw: number;
  }>;
}

export interface LinefillSnap {
  Product: string;
  'Volume (m³)': number;
  'Viscosity (cSt)': number;
  'Density (kg/m³)': number;
  'Initial DRA (ppm)': number;
}

export interface FinalSegmentSlice {
  segment_idx: number;
  station_name: string;
  slices: Array<{length_km: number; kv: number; rho: number}>;
  avg_kv: number;
  avg_rho: number;
}

export interface IntervalRow {
  start: string;
  end: string;
  duration_hours: number;
  flow_m3h: number;
  volume_m3: number;
  cost: number;
}

export interface ParsedScheduleResult {
  op_mode: 'daily_schedule' | 'pumping_planner';
  hourly: HourlyRow[];              // 24 rows for daily_schedule
  linefill_snaps: LinefillSnap[][]; // per-hour linefill queue snapshots
  final_vol: LinefillSnap[];        // next-day starting state
  final_segment_slices: FinalSegmentSlice[];
  hourly_flows: number[];
  total_cost: number;
  total_throughput: number;
  backtrack_notes: string[];
  enforced_actions: Array<{hour: string; action: string}>;
  hourly_errors: string[];
  interval_results: IntervalRow[];  // pumping_planner only
  branch_schedules: Record<string, unknown[]>;
  // Convenience aggregates
  avg_flow: number;
  avg_efficiency: number;
  peak_cost_hour: number;
}

export function parseScheduleResult(
  result: Record<string, unknown>,
  stationList: Array<{name: string; elevation: number; length_km: number; is_pump: boolean}>,
): ParsedScheduleResult {
  const op_mode = (result.op_mode as string || 'daily_schedule') as 'daily_schedule' | 'pumping_planner';
  const rawHours = (result.mainline_schedule as Array<Record<string, unknown>>) || [];
  const rawIntervals = (result.interval_results as IntervalRow[]) || [];

  const hourly: HourlyRow[] = rawHours.map(r => {
    const isErr = Boolean(r.error);
    const pc = Number(r.power_cost || 0);
    const dc = Number(r.dra_cost   || 0);
    const stationMetrics: HourlyRow['stations'] = {};
    for (const s of stationList) {
      const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
      stationMetrics[k] = {
        num_pumps:        Number(r[`num_pumps_${k}`]   || 0),
        speed:            Number(r[`speed_${k}`]        || 0),
        efficiency:       Number(r[`efficiency_${k}`]   || 0),
        power_cost:       Number(r[`power_cost_${k}`]   || 0),
        dra_cost:         Number(r[`dra_cost_${k}`]     || 0),
        dra_ppm:          Number(r[`dra_ppm_${k}`]      || 0),
        drag_reduction:   Number(r[`drag_reduction_${k}`] || 0),
        sdh:              Number(r[`sdh_${k}`]           || 0),
        residual_head_in: Number(r[`residual_head_in_${k}`] || 0),
        motor_kw:         Number(r[`motor_kw_${k}`]     || 0),
      };
    }
    return {
      hour:       Number(r.hour  || 0),
      hour_label: String(r.hour_label || '00:00'),
      step:       Number(r.step  || 0),
      flow:       Number(r.FLOW  || 0),
      power_cost: pc,
      dra_cost:   dc,
      total_cost: pc + dc,
      num_pumps:  Number(r.num_pumps || 0),
      error:      isErr,
      error_msg:  isErr ? String(r.message || '') : undefined,
      stations:   stationMetrics,
    };
  });

  const hourly_flows    = (result.hourly_flows    as number[]) || hourly.map(h => h.flow);
  const total_cost      = Number(result.total_cost      || 0);
  const total_throughput = Number(result.total_throughput || 0);
  const backtrack_notes  = (result.backtrack_notes  as string[]) || [];
  const enforced_actions = (result.enforced_actions as Array<{hour: string; action: string}>) || [];
  const hourly_errors    = (result.hourly_errors    as string[]) || [];
  const linefill_snaps   = (result.linefill_snaps   as LinefillSnap[][]) || [];
  const final_vol        = (result.final_vol        as LinefillSnap[]) || [];
  const final_segment_slices = (result.final_segment_slices as FinalSegmentSlice[]) || [];
  const branch_schedules = (result.branch_schedules as Record<string, unknown[]>) || {};

  const goodHours = hourly.filter(h => !h.error && h.flow > 0);
  const avg_flow = goodHours.length > 0
    ? goodHours.reduce((a, h) => a + h.flow, 0) / goodHours.length : 0;

  const effValues = goodHours.flatMap(h =>
    stationList.filter(s => s.is_pump).map(s => {
      const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
      return h.stations[k]?.efficiency || 0;
    }).filter(e => e > 0)
  );
  const avg_efficiency = effValues.length > 0
    ? effValues.reduce((a, v) => a + v, 0) / effValues.length : 0;

  const peak_cost_hour = hourly.reduce(
    (best, h, i) => h.total_cost > (hourly[best]?.total_cost || 0) ? i : best, 0
  );

  return {
    op_mode,
    hourly,
    linefill_snaps,
    final_vol,
    final_segment_slices,
    hourly_flows,
    total_cost,
    total_throughput,
    backtrack_notes,
    enforced_actions,
    hourly_errors,
    interval_results: rawIntervals,
    branch_schedules,
    avg_flow,
    avg_efficiency,
    peak_cost_hour,
  };
}

function g(ml: Record<string, unknown>, key: string): number {
  const v = ml[key];
  if (v === null || v === undefined) return 0;
  const n = Number(v);
  return isNaN(n) ? 0 : n;
}

export function parseResult(
  result: Record<string, unknown>,
  stationList: Array<{name: string; elevation: number; length_km: number; is_pump: boolean}>,
  terminal: {name: string; elevation: number; min_residual: number},
): ParsedResult {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const flow = g(ml, 'FLOW');
  const totalCost = g(ml, 'total_cost');

  const stations: ParsedStation[] = stationList.map(s => {
    const key = s.name.trim().toLowerCase().replace(/\s+/g, '_');
    return {
      name: s.name,
      key,
      elevation: s.elevation,
      length_km: s.length_km,
      is_pump: s.is_pump,
      sdh:                g(ml, `sdh_${key}`),
      num_pumps:          g(ml, `num_pumps_${key}`),
      speed:              g(ml, `speed_${key}`),
      efficiency:         g(ml, `efficiency_${key}`),
      power_cost:         g(ml, `power_cost_${key}`),
      dra_cost:           g(ml, `dra_cost_${key}`),
      dra_ppm:            g(ml, `dra_ppm_${key}`),
      drag_reduction:     g(ml, `drag_reduction_${key}`),
      residual_head_in:   g(ml, `residual_head_in_${key}`),
      residual_head_out:  g(ml, `residual_head_out_${key}`),
      maop:               g(ml, `maop_${key}`),
      maop_kgcm2:         g(ml, `maop_kgcm2_${key}`),
      head_loss:          g(ml, `head_loss_${key}`),
      velocity:           g(ml, `velocity_${key}`),
      reynolds:           g(ml, `reynolds_${key}`),
      motor_kw:           g(ml, `motor_kw_${key}`),
      tdh:                g(ml, `tdh_${key}`),
      pipeline_flow:      g(ml, `pipeline_flow_${key}`) || flow,
      coef_A:             g(ml, `coef_A_${key}`),
      coef_B:             g(ml, `coef_B_${key}`),
      coef_C:             g(ml, `coef_C_${key}`),
      rho:                g(ml, `rho_${key}`) || 850,
    };
  });

  const totalPowerCost = stations.reduce((acc, s) => acc + s.power_cost, 0);
  const totalDraCost   = stations.reduce((acc, s) => acc + s.dra_cost, 0);
  const totalPumpsRunning = stations.reduce((acc, s) => acc + Math.round(s.num_pumps), 0);

  // Terminal head: solver writes rh_kgcm2_{term_name} and rho_{term_name}
  const termKey = terminal.name.trim().toLowerCase().replace(/\s+/g, '_');
  const termRho = g(ml, `rho_${termKey}`) || 850;
  const termRhKgcm2 = g(ml, `rh_kgcm2_${termKey}`);
  // Convert kg/cm² → metres: h = p * 10000 / rho
  const terminalHead = termRhKgcm2 > 0 ? termRhKgcm2 * 10000 / termRho
    : g(ml, 'residual') || g(ml, 'terminal_head') || g(ml, 'delivery_head');

  // Build HGL data points
  let cumKm = 0;
  const hglData: ParsedHGLPoint[] = [];
  stations.forEach(s => {
    // Only add if we have any result head data
    if (s.residual_head_in > 0 || s.sdh > 0) {
      hglData.push({
        km:      cumKm,
        elev:    s.elevation,
        label:   s.name,
        head_in: s.residual_head_in,
        sdh:     s.sdh,
        maop:    s.maop,
      });
    }
    cumKm += s.length_km;
  });

  if (terminalHead > 0 || hglData.length > 0) {
    hglData.push({
      km:      cumKm,
      elev:    terminal.elevation,
      label:   terminal.name,
      head_in: terminalHead,
      sdh:     0,
      maop:    0,
    });
  }

  return {
    flow,
    totalCost,
    totalPowerCost,
    totalDraCost,
    totalPumpsRunning,
    stations,
    hglData,
    terminalKm: cumKm,
    terminalElev: terminal.elevation,
    terminalMinResidual: terminal.min_residual,
    terminalHead,
  };
}
