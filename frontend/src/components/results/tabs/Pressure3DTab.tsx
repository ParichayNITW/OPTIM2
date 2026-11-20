import React from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function Pressure3DTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const hglData = (ml.hgl_data || []) as Array<{km: number; head: number; elev: number; pressure_barg?: number}>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  // Build 3D profile from HGL data
  let cumKm = 0;
  const kpArr: number[] = [];
  const elevArr: number[] = [];
  const headArr: number[] = [];
  const pressureArr: number[] = [];

  if (hglData.length > 0) {
    hglData.forEach(pt => {
      kpArr.push(pt.km);
      elevArr.push(pt.elev);
      headArr.push(pt.head);
      pressureArr.push(pt.pressure_barg || Math.max(0, (pt.head - pt.elev) * 0.0981));
    });
  } else {
    stationsUsed.forEach((s, i) => {
      kpArr.push(cumKm);
      elevArr.push((s.elevation || 0) as number);
      const h = (s.sdh || s.discharge_head || 0) as number;
      headArr.push(h);
      const elev = (s.elevation || 0) as number;
      pressureArr.push(Math.max(0, (h - elev) * 0.0981));
      if (i < stationsUsed.length - 1) cumKm += (stationsUsed[i].length_km || 0) as number;
    });
  }

  const hasData = kpArr.length > 1;

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">3D Pressure Profile</h4>

      {hasData ? (
        <Plot
          data={[
            {
              type: 'scatter3d', mode: 'lines+markers',
              x: kpArr, y: elevArr, z: pressureArr,
              line: {color: pressureArr, colorscale: 'Jet', width: 6},
              marker: {size: 3, color: pressureArr, colorscale: 'Jet'},
              name: 'Pressure',
            } as Plotly.Data,
            {
              type: 'scatter3d', mode: 'lines',
              x: kpArr, y: elevArr, z: elevArr.map(() => 0),
              line: {color: '#92400e', width: 2},
              name: 'Terrain',
            } as Plotly.Data,
          ]}
          layout={{
            height: 450,
            margin: {t: 20, r: 20, b: 10, l: 20},
            scene: {
              xaxis: {title: 'KP (km)'},
              yaxis: {title: 'Elevation (m)'},
              zaxis: {title: 'Pressure (barg)'},
              camera: {eye: {x: 1.5, y: -1.5, z: 1}},
            },
            paper_bgcolor: 'transparent',
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          No pressure profile data available yet.
        </div>
      )}
    </div>
  );
}
