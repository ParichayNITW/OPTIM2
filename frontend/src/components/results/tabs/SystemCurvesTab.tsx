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

  const { pumpCurveX, pumpCurveY, sysX, sysY, opX, opY } = useMemo(() => {
    if (!hasData || pumpStations.length === 0) return { pumpCurveX: [], pumpCurveY: [], sysX: [], sysY: [], opX: [], opY: [] };
    const idx = Math.min(selected, pumpStations.length - 1);
    const s = pumpStations[idx];
    const { qArr, hArr } = buildPumpCurve(s);

    // System resistance: H_sys = H_static + k * Q^2 where through origin simplified to H_sys = k * Q^2
    // Using operating point (pipeline_flow, tdh) → k = tdh / pipeline_flow^2
    const k = s.pipeline_flow > 0 ? s.tdh / (s.pipeline_flow * s.pipeline_flow) : 0;
    const qMax = qArr.length > 0 ? qArr[qArr.length - 1] : s.pipeline_flow * 1.8;
    const sysX: number[] = [], sysY: number[] = [];
    for (let i = 0; i <= 50; i++) {
      const q = (i / 50) * qMax;
      sysX.push(q);
      sysY.push(k * q * q);
    }

    return {
      pumpCurveX: qArr, pumpCurveY: hArr,
      sysX, sysY,
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

      {/* Pump curve + System curve */}
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
              x: sysX, y: sysY,
              type: 'scatter', mode: 'lines', name: 'System Curve',
              line: { color: '#dc2626', width: 2, dash: 'dash' },
              hovertemplate: 'Q: %{x:.0f} m³/h<br>H: %{y:.1f} m<extra>System Curve</extra>',
            } as Plotly.Data,
            {
              x: opX, y: opY,
              type: 'scatter', mode: 'markers', name: 'Operating Point',
              marker: { color: '#16a34a', size: 14, symbol: 'star', line: { width: 1.5, color: 'white' } },
              hovertemplate: `<b>Operating Point</b><br>Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>`,
            } as Plotly.Data,
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 30, b: 60, l: 70 },
            xaxis: { title: 'Flow Rate Q (m³/h)', rangemode: 'nonnegative' },
            yaxis: { title: 'Head H (m)', rangemode: 'nonnegative' },
            legend: { orientation: 'h', y: -0.2 },
            plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 11 },
            annotations: [{
              x: opX[0] ?? 0, y: opY[0] ?? 0,
              text: `  OP (${(opX[0] ?? 0).toFixed(0)}, ${(opY[0] ?? 0).toFixed(0)})`,
              xanchor: 'left', showarrow: false,
              font: { color: '#16a34a', size: 11, family: 'monospace' },
            }],
          } as unknown as Plotly.Layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Coefficients table */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">Pump H-Q Polynomial: H = A·Q² + B·Q + C</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-600">
                {['Station', 'A (Q² coeff)', 'B (Q coeff)', 'C (Constant / Shutoff Head m)',
                  'Operating Q (m³/h)', 'TDH (m)', 'Efficiency %'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pumpStations.map((ps, i) => (
                <tr key={ps.key} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 font-medium border-b">{ps.name}</td>
                  <td className="px-2 py-1 border-b font-mono">{ps.coef_A.toExponential(4)}</td>
                  <td className="px-2 py-1 border-b font-mono">{ps.coef_B.toExponential(4)}</td>
                  <td className="px-2 py-1 border-b font-mono">{ps.coef_C.toFixed(2)}</td>
                  <td className="px-2 py-1 border-b">{ps.pipeline_flow.toFixed(1)}</td>
                  <td className="px-2 py-1 border-b">{ps.tdh.toFixed(1)}</td>
                  <td className="px-2 py-1 border-b">{ps.efficiency.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
