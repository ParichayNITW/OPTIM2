import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult, ParsedStation } from '../../../utils/resultParser';
import type { StationData } from '../../../store/pipelineStore';

interface Props {
  parsed: ParsedResult;
  fullStations?: StationData[];
}

// Affinity law: H at speed ratio k → H_k(Q) = A*Q^2 + k*B*Q + k^2*C
function affinityHead(A: number, B: number, C: number, k: number, q: number): number {
  return A * q * q + k * B * q + k * k * C;
}

// Fit QH table to polynomial using least-squares (3-coefficient quadratic)
function fitQHPoly(qh: Array<{flow: number; head: number}>): [number, number, number] {
  if (qh.length < 2) return [0, 0, qh[0]?.head ?? 0];
  const n = qh.length;
  let sx4 = 0, sx3 = 0, sx2 = 0, sx1 = 0, s1 = 0;
  let syx2 = 0, syx1 = 0, sy = 0;
  for (const p of qh) {
    const q = p.flow, h = p.head;
    sx4 += q**4; sx3 += q**3; sx2 += q**2; sx1 += q; s1 += 1;
    syx2 += h * q**2; syx1 += h * q; sy += h;
  }
  // Solve 3x3 normal equations [A,B,C]
  const mat = [[sx4,sx3,sx2],[sx3,sx2,sx1],[sx2,sx1,s1]];
  const rhs = [syx2, syx1, sy];
  // Gauss elimination
  for (let col = 0; col < 3; col++) {
    let maxRow = col;
    for (let row = col+1; row < 3; row++) if (Math.abs(mat[row][col]) > Math.abs(mat[maxRow][col])) maxRow = row;
    [mat[col], mat[maxRow]] = [mat[maxRow], mat[col]];
    [rhs[col], rhs[maxRow]] = [rhs[maxRow], rhs[col]];
    for (let row = col+1; row < 3; row++) {
      const f = mat[row][col] / (mat[col][col] || 1e-12);
      for (let j = col; j < 3; j++) mat[row][j] -= f * mat[col][j];
      rhs[row] -= f * rhs[col];
    }
  }
  const x = [0,0,0];
  for (let i = 2; i >= 0; i--) {
    x[i] = rhs[i];
    for (let j = i+1; j < 3; j++) x[i] -= mat[i][j] * x[j];
    x[i] /= (mat[i][i] || 1e-12);
  }
  return [x[0], x[1], x[2]];
}

const COLORS_SPEED = ['#cbd5e1','#94a3b8','#64748b','#475569','#1e293b'];
const COLORS_N = ['#1d4ed8','#7c3aed','#dc2626'];

export function PumpSystemTab({ parsed, fullStations }: Props) {
  const pumpStations = parsed.stations.filter(
    s => s.is_pump && (s.coef_A !== 0 || s.coef_C !== 0) && s.tdh > 0
  );
  const [selected, setSelected] = useState(0);
  const [activeTab, setActiveTab] = useState<'hq'|'interaction'|'efficiency'>('hq');

  const hasData = pumpStations.length > 0;
  const idx = Math.min(selected, pumpStations.length - 1);
  const s: ParsedStation | undefined = pumpStations[idx];

  // Find matching StationData for the selected station
  const stData: StationData | undefined = useMemo(() => {
    if (!s || !fullStations) return undefined;
    return fullStations.find(sd => sd.name.trim().toLowerCase() === s.name.trim().toLowerCase());
  }, [s, fullStations]);

  // Compute all curves from actual StationData
  const curves = useMemo(() => {
    if (!s) return null;

    // --- Determine pump types to draw ---
    const pumpTypes: Array<{
      label: string;
      A: number; B: number; C: number;
      minRpm: number; dolRpm: number;
      effTable: Array<{flow: number; eff: number}>;
    }> = [];

    if (stData?.pumpA?.enabled && stData.pumpA.qh_table.length >= 2) {
      const [A, B, C] = fitQHPoly(stData.pumpA.qh_table);
      pumpTypes.push({
        label: 'Type A',
        A, B, C,
        minRpm: stData.pumpA.min_rpm || 900,
        dolRpm: stData.pumpA.dol_rpm || 1480,
        effTable: stData.pumpA.eff_table || [],
      });
    }
    if (stData?.pumpB?.enabled && stData.pumpB.qh_table.length >= 2) {
      const [A, B, C] = fitQHPoly(stData.pumpB.qh_table);
      pumpTypes.push({
        label: 'Type B',
        A, B, C,
        minRpm: stData.pumpB.min_rpm || 900,
        dolRpm: stData.pumpB.dol_rpm || 1480,
        effTable: stData.pumpB.eff_table || [],
      });
    }

    // Fallback: use parsed coefficients when no StationData
    if (pumpTypes.length === 0) {
      pumpTypes.push({
        label: 'Pump',
        A: s.coef_A, B: s.coef_B, C: s.coef_C,
        minRpm: s.speed > 0 ? Math.round(s.speed * 0.6) : 900,
        dolRpm: s.speed > 0 ? s.speed : 1480,
        effTable: [],
      });
    }

    const qOpMax = Math.max(s.pipeline_flow * 1.9, 150);
    const nPoints = 60;

    // --- H-Q speed family traces per pump type ---
    const hqTraces: Plotly.Data[] = [];
    const effTraces: Plotly.Data[] = [];
    const typeColors = ['#1d4ed8', '#dc2626', '#16a34a'];

    for (let ti = 0; ti < pumpTypes.length; ti++) {
      const pt = pumpTypes[ti];
      const baseColor = typeColors[ti % typeColors.length];

      const rpmRange: number[] = [];
      for (let rpm = pt.minRpm; rpm <= pt.dolRpm; rpm += 100) rpmRange.push(rpm);
      if (rpmRange[rpmRange.length - 1] !== pt.dolRpm) rpmRange.push(pt.dolRpm);

      for (let ri = 0; ri < rpmRange.length; ri++) {
        const rpm = rpmRange[ri];
        const k = rpm / pt.dolRpm;
        const alpha = 0.25 + 0.75 * (ri / Math.max(rpmRange.length - 1, 1));
        const colorHex = ri === rpmRange.length - 1 ? baseColor : `${baseColor}${Math.round(alpha * 255).toString(16).padStart(2,'0')}`;
        const qArr: number[] = [], hArr: number[] = [];
        for (let i = 0; i <= nPoints; i++) {
          const q = (i / nPoints) * qOpMax * Math.min(k + 0.1, 1.0);
          const h = affinityHead(pt.A, pt.B, pt.C, k, q);
          if (h < 0) break;
          qArr.push(q); hArr.push(h);
        }
        const isDOL = ri === rpmRange.length - 1;
        hqTraces.push({
          x: qArr, y: hArr,
          type: 'scatter', mode: 'lines',
          name: `${pt.label} ${rpm} rpm`,
          legendgroup: pt.label,
          showlegend: isDOL || ri === 0,
          line: { color: isDOL ? baseColor : colorHex, width: isDOL ? 2.5 : 1.2, dash: isDOL ? 'solid' : 'dot' },
          hovertemplate: `${pt.label} ${rpm}rpm: Q=%{x:.0f} m³/h, H=%{y:.1f} m<extra></extra>`,
        } as Plotly.Data);

        // Efficiency at this speed (affinity: same eff, flow scales with k)
        if (pt.effTable.length > 0) {
          const efX: number[] = [], efY: number[] = [];
          for (const ep of pt.effTable) {
            efX.push(k * ep.flow);
            efY.push(ep.eff);
          }
          effTraces.push({
            x: efX, y: efY,
            type: 'scatter', mode: 'lines',
            name: `${pt.label} ${rpm} rpm`,
            legendgroup: `eff_${pt.label}`,
            showlegend: isDOL,
            line: { color: isDOL ? baseColor : colorHex, width: isDOL ? 2.5 : 1.2, dash: isDOL ? 'solid' : 'dot' },
            hovertemplate: `${pt.label} ${rpm}rpm: Q=%{x:.0f}, Eff=%{y:.1f}%<extra></extra>`,
          } as Plotly.Data);
        }
      }
    }

    // --- System curves (N-pump interaction) ---
    const k_fric = s.pipeline_flow > 0 && s.head_loss > 0
      ? s.head_loss / (s.pipeline_flow * s.pipeline_flow) : 0;
    const H_static = s.head_loss > 0 ? s.tdh - s.head_loss : 0;
    const kf = k_fric > 0 ? k_fric : (s.pipeline_flow > 0 ? s.tdh / (s.pipeline_flow * s.pipeline_flow) : 0);
    const hs = k_fric > 0 ? H_static : 0;

    const sysXnoDRA: number[] = [], sysYnoDRA: number[] = [];
    const sysXDRA: number[] = [], sysYDRA: number[] = [];
    for (let i = 0; i <= 60; i++) {
      const q = (i / 60) * qOpMax;
      sysXnoDRA.push(q);
      sysYnoDRA.push(hs + kf * q * q);
      if (s.drag_reduction > 0) {
        sysXDRA.push(q);
        sysYDRA.push(hs + (1 - s.drag_reduction / 100) * kf * q * q);
      }
    }

    const sysNoDRA: Plotly.Data = {
      x: sysXnoDRA, y: sysYnoDRA,
      type: 'scatter', mode: 'lines', name: 'System (No DRA)',
      line: { color: '#dc2626', width: 2.5 },
      hovertemplate: 'Q: %{x:.0f} m³/h, H_sys: %{y:.1f} m<extra>No DRA</extra>',
    } as Plotly.Data;
    const sysDRA: Plotly.Data | null = s.drag_reduction > 0 ? {
      x: sysXDRA, y: sysYDRA,
      type: 'scatter', mode: 'lines',
      name: `System (DRA ${s.dra_ppm.toFixed(0)} ppm, ${s.drag_reduction.toFixed(1)}% DR)`,
      line: { color: '#16a34a', width: 2, dash: 'dashdot' },
      hovertemplate: 'Q: %{x:.0f} m³/h, H_sys: %{y:.1f} m<extra>With DRA</extra>',
    } as Plotly.Data : null;

    // --- N-pump DOL curves (series: H_N = N * H_1) ---
    const maxN = stData?.max_pumps ?? Math.max(Math.round(s.num_pumps), 1);
    const nPumpTraces: Plotly.Data[] = [];
    // Use first pump type (or only type) for N-pump interaction
    const pt0 = pumpTypes[0];
    const rpmSteps: number[] = [];
    for (let rpm = pt0.minRpm; rpm <= pt0.dolRpm; rpm += 100) rpmSteps.push(rpm);
    if (rpmSteps[rpmSteps.length - 1] !== pt0.dolRpm) rpmSteps.push(pt0.dolRpm);

    for (let N = 1; N <= Math.min(maxN, 3); N++) {
      const nColor = COLORS_N[(N - 1) % COLORS_N.length];
      for (let ri = 0; ri < rpmSteps.length; ri++) {
        const rpm = rpmSteps[ri];
        const k = rpm / pt0.dolRpm;
        const isDOL = ri === rpmSteps.length - 1;
        const qArr: number[] = [], hArr: number[] = [];
        for (let i = 0; i <= nPoints; i++) {
          const q = (i / nPoints) * qOpMax * Math.min(k + 0.1, 1.0);
          const h1 = affinityHead(pt0.A, pt0.B, pt0.C, k, q);
          if (h1 < 0) break;
          qArr.push(q); hArr.push(N * h1);
        }
        nPumpTraces.push({
          x: qArr, y: hArr,
          type: 'scatter', mode: 'lines',
          name: `${N}P ${rpm}rpm`,
          legendgroup: `${N}P`,
          showlegend: isDOL,
          line: { color: nColor, width: isDOL ? 2.5 : 1.2, dash: isDOL ? 'solid' : 'dot' },
          hovertemplate: `${N} pump(s) ${rpm}rpm: Q=%{x:.0f}, H=%{y:.1f} m<extra></extra>`,
        } as Plotly.Data);
      }
    }

    return { hqTraces, effTraces, sysNoDRA, sysDRA, nPumpTraces, opX: [s.pipeline_flow], opY: [s.tdh] };
  }, [s, stData]);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        No pump system data. Pump station with Q-H curve required.
      </div>
    );
  }

  const tabs = [
    { id: 'hq' as const, label: 'H-Q Curves (Speed Family)' },
    { id: 'interaction' as const, label: 'Pump-System Interaction' },
    { id: 'efficiency' as const, label: 'Efficiency Curves' },
  ];

  return (
    <div className="p-3 space-y-3">
      {/* Station selector */}
      {pumpStations.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">Station:</span>
          {pumpStations.map((ps, i) => (
            <button key={ps.key} onClick={() => setSelected(i)}
              className={`px-2 py-1 rounded text-xs font-medium border ${
                i === idx ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
              }`}>
              {ps.name}
            </button>
          ))}
        </div>
      )}

      {/* Sub-tabs */}
      <div className="flex gap-0 border-b border-gray-200">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)}
            className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
              activeTab === t.id ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Info row */}
      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'Station', val: s?.name ?? '—' },
          { label: 'Op. Flow', val: s ? `${s.pipeline_flow.toFixed(1)} m³/h` : '—' },
          { label: 'TDH', val: s ? `${s.tdh.toFixed(1)} m` : '—' },
          { label: 'Efficiency', val: s ? `${s.efficiency.toFixed(1)}%` : '—' },
        ].map(c => (
          <div key={c.label} className="bg-indigo-50 rounded px-2 py-1.5">
            <div className="text-gray-500">{c.label}</div>
            <div className="font-semibold text-gray-800">{c.val}</div>
          </div>
        ))}
      </div>

      {/* H-Q speed family */}
      {activeTab === 'hq' && curves && (
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">
            H-Q Curves — All Pump Types at All Speed Steps (Min RPM to DOL, 100 RPM intervals)
          </h4>
          <Plot
            data={[
              ...curves.hqTraces,
              {
                x: curves.opX, y: curves.opY, type: 'scatter', mode: 'markers',
                name: 'Operating Point',
                marker: { color: '#f59e0b', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
                hovertemplate: `<b>OP</b><br>Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>`,
              } as Plotly.Data,
            ]}
            layout={{
              height: 420,
              margin: { t: 20, r: 20, b: 90, l: 70 },
              xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
              yaxis: { title: 'Head H (m)', rangemode: 'nonnegative' },
              legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Pump-System Interaction (N pumps + system curves) */}
      {activeTab === 'interaction' && curves && (
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">
            Pump-System Interaction — 1/2/3 Pump Configurations at All Speed Steps with System Curves
          </h4>
          <Plot
            data={[
              ...curves.nPumpTraces,
              curves.sysNoDRA,
              ...(curves.sysDRA ? [curves.sysDRA] : []),
              {
                x: curves.opX, y: curves.opY, type: 'scatter', mode: 'markers',
                name: 'Operating Point',
                marker: { color: '#f59e0b', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
                hovertemplate: `<b>OP</b><br>Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>`,
              } as Plotly.Data,
            ]}
            layout={{
              height: 420,
              margin: { t: 20, r: 20, b: 90, l: 70 },
              xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
              yaxis: { title: 'Head H (m)', rangemode: 'nonnegative' },
              legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Efficiency curves */}
      {activeTab === 'efficiency' && curves && (
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">
            Pump Efficiency vs Flow — All Types at All Speed Steps
          </h4>
          {curves.effTraces.length === 0 ? (
            <div className="p-4 text-center text-gray-400 text-sm bg-gray-50 rounded">
              No efficiency table data available for this station.
              {s && <span> Operating efficiency: {s.efficiency.toFixed(1)}% at {s.pipeline_flow.toFixed(0)} m³/h</span>}
            </div>
          ) : (
            <Plot
              data={[
                ...curves.effTraces,
                {
                  x: [s?.pipeline_flow ?? 0], y: [s?.efficiency ?? 0],
                  type: 'scatter', mode: 'markers',
                  name: 'Operating Point',
                  marker: { color: '#f59e0b', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
                  hovertemplate: `<b>OP</b><br>Q: %{x:.1f} m³/h<br>Eff: %{y:.1f}%<extra></extra>`,
                } as Plotly.Data,
              ]}
              layout={{
                height: 380,
                margin: { t: 20, r: 20, b: 90, l: 70 },
                xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
                yaxis: { title: 'Efficiency (%)', range: [0, 100] },
                legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
                plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
              } as unknown as Plotly.Layout}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          )}
        </div>
      )}
    </div>
  );
}
