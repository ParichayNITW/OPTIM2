import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult, ParsedStation } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

// Affinity law: H at speed ratio k → H_k(Q) = A*Q^2 + k*B*Q + k^2*C
function affinityHead(A: number, B: number, C: number, k: number, q: number): number {
  return A * q * q + k * B * q + k * k * C;
}

function buildSpeedFamily(s: ParsedStation, nPoints = 60): Plotly.Data[] {
  const { coef_A, coef_B, coef_C, pipeline_flow } = s;
  if (coef_A === 0 && coef_B === 0 && coef_C === 0) return [];

  const qMax = Math.max(pipeline_flow * 1.8, 100);
  const speedRatios = [0.50, 0.65, 0.80, 0.90, 1.00];
  const colors = ['#94a3b8', '#6366f1', '#0284c7', '#f59e0b', '#1d4ed8'];

  return speedRatios.map((k, ci) => {
    const qArr: number[] = [], hArr: number[] = [];
    for (let i = 0; i <= nPoints; i++) {
      const q = (i / nPoints) * qMax * k;
      const h = affinityHead(coef_A, coef_B, coef_C, k, q);
      if (h < 0) break;
      qArr.push(q); hArr.push(h);
    }
    const nDesign = s.speed > 0 ? s.speed : 1500;
    return {
      x: qArr, y: hArr,
      type: 'scatter', mode: 'lines',
      name: `${Math.round(k * nDesign)} rpm (${Math.round(k * 100)}%)`,
      line: { color: colors[ci], width: k === 1.0 ? 2.5 : 1.5, dash: k === 1.0 ? 'solid' : 'dot' },
      hovertemplate: `Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>${Math.round(k * 100)}% speed</extra>`,
    } as Plotly.Data;
  });
}

export function PumpSystemTab({ parsed }: Props) {
  const pumpStations = parsed.stations.filter(s => s.is_pump && (s.coef_A !== 0 || s.coef_C !== 0) && s.tdh > 0);
  const [selected, setSelected] = useState(0);

  const hasData = pumpStations.length > 0;

  const { speedFamily, sysX, sysY, opX, opY, affinityParX, affinityParY } = useMemo(() => {
    if (!hasData) return { speedFamily: [], sysX: [], sysY: [], opX: [], opY: [], affinityParX: [], affinityParY: [] };
    const idx = Math.min(selected, pumpStations.length - 1);
    const s = pumpStations[idx];

    const family = buildSpeedFamily(s);

    // System resistance: H_sys = k_sys * Q^2 (parabola through op point)
    const k_sys = s.pipeline_flow > 0 ? s.tdh / (s.pipeline_flow * s.pipeline_flow) : 0;
    const qMax = Math.max(s.pipeline_flow * 1.8, 100);
    const sysX: number[] = [], sysY: number[] = [];
    for (let i = 0; i <= 60; i++) {
      const q = (i / 60) * qMax;
      sysX.push(q); sysY.push(k_sys * q * q);
    }

    // Affinity parabola: locus of points (k*Q_op, k^2*H_op) for k in [0,1]
    const affinityParX: number[] = [], affinityParY: number[] = [];
    for (let i = 0; i <= 30; i++) {
      const kk = i / 30;
      affinityParX.push(kk * s.pipeline_flow);
      affinityParY.push(kk * kk * s.tdh);
    }

    return { speedFamily: family, sysX, sysY, opX: [s.pipeline_flow], opY: [s.tdh], affinityParX, affinityParY };
  }, [selected, pumpStations, hasData]);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        No pump system data. Pump station with Q-H curve required.
      </div>
    );
  }

  const idx = Math.min(selected, pumpStations.length - 1);
  const s = pumpStations[idx];

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

      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'Design Speed', val: `${s.speed > 0 ? s.speed.toFixed(0) : 'N/A'} rpm` },
          { label: 'Op. Flow', val: `${s.pipeline_flow.toFixed(1)} m³/h` },
          { label: 'TDH', val: `${s.tdh.toFixed(1)} m` },
          { label: 'Efficiency', val: `${s.efficiency.toFixed(1)}%` },
        ].map(c => (
          <div key={c.label} className="bg-indigo-50 rounded px-2 py-1.5">
            <div className="text-gray-500">{c.label}</div>
            <div className="font-semibold text-gray-800">{c.val}</div>
          </div>
        ))}
      </div>

      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">
          Pump Speed Family (Affinity Laws) with System Resistance
        </h4>
        <Plot
          data={[
            ...speedFamily,
            {
              x: sysX, y: sysY, type: 'scatter', mode: 'lines',
              name: 'System Curve', line: { color: '#dc2626', width: 2 },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H_sys: %{y:.1f} m<extra></extra>',
            } as Plotly.Data,
            {
              x: affinityParX, y: affinityParY, type: 'scatter', mode: 'lines',
              name: 'Affinity Parabola', line: { color: '#d97706', width: 1.5, dash: 'longdash' },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>Affinity</extra>',
            } as Plotly.Data,
            {
              x: opX, y: opY, type: 'scatter', mode: 'markers',
              name: 'Operating Point',
              marker: { color: '#16a34a', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
              hovertemplate: `<b>Operating Point</b><br>Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>`,
            } as Plotly.Data,
          ]}
          layout={{
            height: 380,
            margin: { t: 20, r: 30, b: 80, l: 70 },
            xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
            yaxis: { title: 'Head H (m)', rangemode: 'nonnegative' },
            legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
            plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
          } as unknown as Plotly.Layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
