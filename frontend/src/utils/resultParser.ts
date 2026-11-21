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
