import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

export function Pressure3DTab({ parsed }: Props) {
  const hasData = parsed.hglData.length > 1;

  // pressure_barg = (head - elev) * rho * g / 100000, g=9.81, /100000 → Pa to barg
  const toPressure = (head: number, elev: number, rho = 850) =>
    Math.max(0, (head - elev) * rho * 9.81 / 100000);

  // Build 3D arrays from hglData
  const kpArr: number[] = [];
  const elevArr: number[] = [];
  const pressureArr: number[] = [];

  parsed.hglData.forEach(pt => {
    const head = pt.sdh > 0 ? pt.sdh : pt.head_in;
    kpArr.push(pt.km);
    elevArr.push(pt.elev);
    pressureArr.push(toPressure(head, pt.elev));
  });

  // Build maop pressure line
  const maopKpArr: number[] = [];
  const maopElevArr: number[] = [];
  const maopPressArr: number[] = [];
  parsed.hglData.forEach(pt => {
    if (pt.maop > 0) {
      maopKpArr.push(pt.km);
      maopElevArr.push(pt.elev);
      maopPressArr.push(toPressure(pt.maop, pt.elev));
    }
  });

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">3D Pressure Profile along Pipeline</h4>

      {hasData ? (
        <>
          <Plot
            data={[
              {
                type: 'scatter3d', mode: 'lines+markers',
                x: kpArr, y: elevArr, z: pressureArr,
                line: { color: pressureArr, colorscale: 'Jet', width: 6 },
                marker: { size: 3, color: pressureArr, colorscale: 'Jet',
                  colorbar: { title: 'P (barg)', x: 1.0, len: 0.8 } },
                name: 'Operating Pressure',
                hovertemplate: 'KP: %{x:.1f} km<br>Elev: %{y:.1f} m<br>P: %{z:.2f} barg<extra></extra>',
              } as Plotly.Data,
              ...(maopPressArr.length > 1 ? [{
                type: 'scatter3d', mode: 'lines',
                x: maopKpArr, y: maopElevArr, z: maopPressArr,
                line: { color: '#dc2626', width: 2.5, dash: 'dot' } as Plotly.ScatterLine,
                name: 'MAOP',
                hovertemplate: 'KP: %{x:.1f} km<br>Elev: %{y:.1f} m<br>MAOP: %{z:.2f} barg<extra></extra>',
              } as Plotly.Data] : []),
              {
                type: 'scatter3d', mode: 'lines',
                x: kpArr, y: elevArr, z: elevArr.map(() => 0),
                line: { color: '#92400e', width: 1.5 },
                name: 'Terrain Profile',
                hoverinfo: 'skip',
              } as Plotly.Data,
            ]}
            layout={{
              height: 450,
              margin: { t: 20, r: 10, b: 10, l: 10 },
              scene: {
                xaxis: { title: 'KP (km)' },
                yaxis: { title: 'Elevation (m)' },
                zaxis: { title: 'Pressure (barg)' },
                camera: { eye: { x: 1.5, y: -1.5, z: 1 } },
              },
              paper_bgcolor: 'transparent',
              font: { size: 10 },
              legend: { x: 0, y: 1 },
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />

          {/* Table */}
          <h4 className="text-sm font-semibold text-gray-700">Pressure at Each Station</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-gray-100 text-gray-600">
                  {['Station', 'KP (km)', 'Elevation (m)', 'Head (m)', 'Pressure (barg)', 'MAOP (barg)', '% of MAOP'].map(h => (
                    <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {parsed.hglData.map((pt, i) => {
                  const head = pt.sdh > 0 ? pt.sdh : pt.head_in;
                  const pres = toPressure(head, pt.elev);
                  const maopP = pt.maop > 0 ? toPressure(pt.maop, pt.elev) : 0;
                  const pct = maopP > 0 ? (pres / maopP) * 100 : 0;
                  return (
                    <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-2 py-1 font-medium border-b">{pt.label}</td>
                      <td className="px-2 py-1 border-b">{pt.km.toFixed(1)}</td>
                      <td className="px-2 py-1 border-b">{pt.elev.toFixed(1)}</td>
                      <td className="px-2 py-1 border-b">{head.toFixed(1)}</td>
                      <td className="px-2 py-1 border-b">
                        <span className={pct > 90 ? 'text-red-600 font-medium' : pct > 70 ? 'text-amber-600' : ''}>
                          {pres.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-2 py-1 border-b">{maopP > 0 ? maopP.toFixed(2) : '—'}</td>
                      <td className="px-2 py-1 border-b">
                        {pct > 0 ? (
                          <div className="flex items-center gap-1">
                            <div className={`h-2 rounded ${pct > 90 ? 'bg-red-400' : pct > 70 ? 'bg-amber-400' : 'bg-green-400'}`}
                              style={{ width: `${Math.min(pct, 100) * 0.5}px` }} />
                            <span>{pct.toFixed(1)}%</span>
                          </div>
                        ) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          Run optimization to see the 3D pressure profile
        </div>
      )}
    </div>
  );
}
