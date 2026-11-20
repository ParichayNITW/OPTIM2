import React, { useState } from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function Analysis3DTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const surfaces = (ml.surfaces_3d || {}) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;
  const stationNames = stationsUsed.map((s, i) => (s.name || `S${i+1}`) as string);
  const [selected, setSelected] = useState(stationNames[0] || '');

  const surf = (surfaces[selected] || {}) as {
    rpm_grid?: number[][], flow_grid?: number[][], cost_grid?: number[][]
  };

  const hasData = surf.rpm_grid && surf.flow_grid && surf.cost_grid;

  return (
    <div className="p-4 space-y-3">
      {stationNames.length > 0 && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-600 font-medium">Station:</label>
          <select value={selected} onChange={e => setSelected(e.target.value)}
            className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-blue-400">
            {stationNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      )}

      <h4 className="text-sm font-semibold text-gray-700">Cost Surface: Speed vs Flow</h4>

      {hasData ? (
        <Plot
          data={[{
            type: 'surface',
            x: surf.flow_grid,
            y: surf.rpm_grid,
            z: surf.cost_grid,
            colorscale: 'RdYlGn',
            reversescale: true,
            colorbar: {title: 'Cost (₹)', titleside: 'right', len: 0.6},
          } as Plotly.Data]}
          layout={{
            height: 420,
            margin: {t: 20, r: 20, b: 10, l: 20},
            scene: {
              xaxis: {title: 'Flow (m³/h)'},
              yaxis: {title: 'Speed (RPM)'},
              zaxis: {title: 'Cost (₹)'},
            },
            paper_bgcolor: 'transparent',
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          3D surface data not available. Run with candidate log to see this chart.
        </div>
      )}
    </div>
  );
}
