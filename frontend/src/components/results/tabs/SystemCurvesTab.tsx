import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult, ParsedStation } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

function buildPumpCurve(s: ParsedStation, nPoints = 60): { qArr: number[]; hArr: number[] } {
  const { coef_A, coef_B, coef_C, pipeline_flow, tdh } = s;
  if (coef_A === 0 && coef_B === 0 && coef_C === 0) return { qArr: [], hArr: [] };

  // Q_max: estimate from operating point with 1.8x margin, min 100 m³/h
  const qMax = Math.max(pipeline_flow * 1.8, 100);
  const qArr: number[] = [];
  const hArr: number[] = [];

  for (let i = 0; i <= nPoints; i++) {
    const q = (i / nPoints) * qMax;
    const h = coef_A * q * q + coef_B * q + coef_C;
    if (h < 0) break;
    qArr.push(q);
    hArr.push(h);
  }
  return { qArr, hArr };
}

export function SystemCurvesTab({ parsed }: Props) {
  const pumpStations = parsed.stations.filter(s => s.is_pump && (s.coef_A !== 0 || s.coef_C !== 0) && s.tdh > 0);
  const [selected, setSelected] = useState(0);

  const hasData = pumpStations.length > 0;

  const { pumpCurveX, pumpCurveY, sysXnoDRA, sysYnoDRA, sysXDRA, sysYDRA, opX, opY } = useMemo(() => {
    const empty = { pumpCurveX: [], pumpCurveY: [], sysXnoDRA: [], sysYnoDRA: [], sysXDRA: [], sysYDRA: [], opX: [], opY: [] };
    if (!hasData || pumpStations.length === 0) return empty;
    const idx = Math.min(selected, pumpStations.length - 1);
    const s = pumpStations[idx];
    const { qArr, hArr } = buildPumpCurve(s);

    const qMax = qArr.length > 0 ? qArr[qArr.length - 1] : s.pipeline_flow * 1.8;

    // Static head + friction coefficient
    const k_fric = s.pipeline_flow > 0 && s.head_loss > 0
      ? s.head_loss / (s.pipeline_flow * s.pipeline_flow) : 0;
    const H_static = s.head_loss > 0 ? s.tdh - s.head_loss : 0;
    const kFricFallback = s.pipeline_flow > 0 ? s.tdh / (s.pipeline_flow * s.pipeline_flow) : 0;
    const kf = k_fric > 0 ? k_fric : kFricFallback;
    const hs = k_fric > 0 ? H_static : 0;

    // No DRA system curve
    const sysXnoDRA: number[] = [], sysYnoDRA: number[] = [];
    for (let i = 0; i <= 50; i++) {
      const q = (i / 50) * qMax;
      sysXnoDRA.push(q);
      sysYnoDRA.push(hs + kf * q * q);
    }

    // With DRA system curve (only when drag_reduction > 0)
    const sysXDRA: number[] = [], sysYDRA: number[] = [];
    if (s.drag_reduction > 0) {
      for (let i = 0; i <= 50; i++) {
        const q = (i / 50) * qMax;
        sysXDRA.push(q);
        sysYDRA.push(hs + (1 - s.drag_reduction / 100) * kf * q * q);
      }
    }

    return {
      pumpCurveX: qArr, pumpCurveY: hArr,
      sysXnoDRA, sysYnoDRA, sysXDRA, sysYDRA,
      opX: [s.pipeline_flow], opY: [s.tdh],
    };
  }, [selected, pumpStations, hasData]);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        No pump curve data available. Pump station with Q-H table required.
      </div>
    );
  }

  const idx = Math.min(selected, pumpStations.length - 1);
  const s = pumpStations[idx];

  return (
    <div className="p-3 space-y-3">
      {/* Station selector */}
      {pumpStations.length > 1 && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-600 font-medium">Station:</span>
          {pumpStations.map((ps, i) => (
            <button key={ps.key} onClick={() => setSelected(i)}
              className={`px-3 py-1 rounded text-xs font-medium border transition-colors ${
                i === idx ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400'
              }`}>
              {ps.name}
            </button>
          ))}
        </div>
      )}

      {/* Pump info row */}
      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'Station', val: s.name },
          { label: 'Operating Flow', val: `${s.pipeline_flow.toFixed(1)} m³/h` },
          { label: 'TDH', val: `${s.tdh.toFixed(1)} m` },
          { label: 'Efficiency', val: `${s.efficiency.toFixed(1)}%` },
        ].map(c => (
          <div key={c.label} className="bg-blue-50 rounded px-2 py-1.5">
            <div className="text-gray-500">{c.label}</div>
            <div className="font-semibold text-gray-800">{c.val}</div>
          </div>
        ))}
      </div>

      {/* Pump curve + System curves */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">Pump H-Q Curve and System Resistance Curve</h4>
        <Plot
          data={[
            {
              x: pumpCurveX, y: pumpCurveY,
              type: 'scatter', mode: 'lines', name: 'Pump H-Q Curve',
              line: { color: '#1d4ed8', width: 2.5 },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>Pump Curve</extra>',
            } as Plotly.Data,
            {
              x: sysXnoDRA, y: sysYnoDRA,
              type: 'scatter', mode: 'lines', name: 'System Curve (No DRA)',
              line: { color: '#dc2626', width: 2, dash: 'dash' },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>No DRA</extra>',
            } as Plotly.Data,
            ...(sysXDRA.length > 0 ? [{
              x: sysXDRA, y: sysYDRA,
              type: 'scatter', mode: 'lines',
              name: `System Curve (DRA ${pumpStations[Math.min(selected, pumpStations.length-1)]?.dra_ppm?.toFixed(0) ?? ''}ppm)`,
              line: { color: '#16a34a', width: 2, dash: 'dashdot' },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>With DRA</extra>',
            } as Plotly.Data] : []),
            {
              x: opX, y: opY,
              type: 'scatter', mode: 'markers', name: 'Operating Point',
              marker: { color: '#7c3aed', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
              hovertemplate: `<b>Operating Point</b><br>Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>`,
            } as Plotly.Data,
          ]}
          layout={{
            height: 380,
            margin: { t: 20, r: 30, b: 70, l: 70 },
            xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
            yaxis: { title: 'Head H (m)', rangemode: 'nonnegative' },
            legend: { orientation: 'h', y: -0.25 },
            plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
            annotations: [{
              x: opX[0] ?? 0, y: opY[0] ?? 0,
              text: `  OP (${(opX[0] ?? 0).toFixed(0)}, ${(opY[0] ?? 0).toFixed(0)})`,
              xanchor: 'left', showarrow: false,
              font: { color: '#7c3aed', size: 11, family: 'monospace' },
            }],
          } as unknown as Plotly.Layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
