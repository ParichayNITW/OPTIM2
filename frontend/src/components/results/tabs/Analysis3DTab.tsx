import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult, ParsedStation } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

function build3DSurface(s: ParsedStation) {
  const { coef_A, coef_B, coef_C, pipeline_flow, speed } = s;
  if (coef_A === 0 && coef_B === 0 && coef_C === 0) return null;

  const nDesign = speed > 0 ? speed : 1500;
  const qMax = Math.max(pipeline_flow * 1.8, 100);
  const nQ = 20, nK = 20;

  // Q axis: 0 to qMax
  const qRange = Array.from({ length: nQ }, (_, i) => (i / (nQ - 1)) * qMax);
  // Speed axis (RPM): 50% to 100% of design
  const nRange = Array.from({ length: nK }, (_, j) => (0.50 + (j / (nK - 1)) * 0.50) * nDesign);

  // z[j][i] = head at q=qRange[i], N=nRange[j]
  const z: number[][] = nRange.map(N => {
    const k = N / nDesign;
    return qRange.map(q => {
      // Affinity law: H_k(Q) = A*Q^2 + k*B*Q + k^2*C
      const h = coef_A * q * q + k * coef_B * q + k * k * coef_C;
      return Math.max(0, h);
    });
  });

  return { qRange, nRange, z };
}

export function Analysis3DTab({ parsed }: Props) {
  const pumpStations = parsed.stations.filter(s => s.is_pump && (s.coef_A !== 0 || s.coef_C !== 0) && s.tdh > 0);
  const [selected, setSelected] = useState(0);

  const hasData = pumpStations.length > 0;
  const idx = Math.min(selected, pumpStations.length - 1);

  const surface = useMemo(() => {
    if (!hasData || pumpStations.length === 0) return null;
    return build3DSurface(pumpStations[idx]);
  }, [idx, pumpStations, hasData]);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        No pump data for 3D analysis. Pump station with Q-H curve required.
      </div>
    );
  }

  const s = pumpStations[idx];

  return (
    <div className="p-3 space-y-3">
      {pumpStations.length > 1 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">Station:</span>
          {pumpStations.map((ps, i) => (
            <button key={ps.key} onClick={() => setSelected(i)}
              className={`px-2 py-1 rounded text-xs font-medium border ${
                i === idx ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300'
              }`}>
              {ps.name}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-3 gap-2 text-xs">
        {[
          { label: 'Station', val: s.name },
          { label: 'Design Speed', val: `${s.speed > 0 ? s.speed.toFixed(0) : 'N/A'} rpm` },
          { label: 'TDH at Design', val: `${s.tdh.toFixed(1)} m` },
        ].map(c => (
          <div key={c.label} className="bg-purple-50 rounded px-2 py-1.5">
            <div className="text-gray-500">{c.label}</div>
            <div className="font-semibold text-gray-800">{c.val}</div>
          </div>
        ))}
      </div>

      {surface ? (
        <>
          <h4 className="text-xs font-semibold text-gray-600">3D Pump Performance Surface (Affinity Laws)</h4>
          <Plot
            data={[
              {
                type: 'surface',
                x: surface.qRange,
                y: surface.nRange,
                z: surface.z,
                colorscale: 'Jet',
                colorbar: { title: 'Head (m)', len: 0.8 },
                hovertemplate: 'Q: %{x:.0f} m³/h<br>N: %{y:.0f} rpm<br>H: %{z:.1f} m<extra></extra>',
                contours: {
                  z: { show: true, usecolormap: true, highlightcolor: '#42f462', project: { z: true } },
                },
              } as Plotly.Data,
              {
                type: 'scatter3d', mode: 'markers',
                x: [s.pipeline_flow], y: [s.speed > 0 ? s.speed : 1500], z: [s.tdh],
                marker: { size: 8, color: '#16a34a', symbol: 'diamond' },
                name: 'Operating Point',
                hovertemplate: `<b>Operating Point</b><br>Q: ${s.pipeline_flow.toFixed(1)} m³/h<br>N: ${s.speed > 0 ? s.speed.toFixed(0) : 'N/A'} rpm<br>H: ${s.tdh.toFixed(1)} m<extra></extra>`,
              } as Plotly.Data,
            ]}
            layout={{
              height: 450,
              margin: { t: 10, r: 10, b: 10, l: 10 },
              scene: {
                xaxis: { title: 'Q (m³/h)' },
                yaxis: { title: 'Speed N (rpm)' },
                zaxis: { title: 'Head H (m)' },
                camera: { eye: { x: 1.6, y: -1.6, z: 1.0 } },
              },
              paper_bgcolor: 'transparent',
              font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
          <p className="text-xs text-gray-400">
            Surface computed via pump affinity laws: H(Q,N) = A·Q² + (N/N₀)·B·Q + (N/N₀)²·C
          </p>
        </>
      ) : (
        <div className="text-gray-400 text-sm text-center py-8">
          Insufficient pump curve data to generate 3D surface
        </div>
      )}
    </div>
  );
}
