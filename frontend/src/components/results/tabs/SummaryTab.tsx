import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

const fmtN = (n: number, d = 1) => n.toLocaleString('en-IN', { maximumFractionDigits: d });
const fmtCur = (n: number) => `₹${fmtN(n, 0)}`;

function KpiCard({ label, value, unit, color }: { label: string; value: string; unit?: string; color: string }) {
  return (
    <div className={`rounded-lg border-l-4 bg-white shadow-sm p-3 ${color}`}>
      <div className="text-xs text-gray-500 font-medium">{label}</div>
      <div className="text-lg font-bold text-gray-900 leading-tight">{value}</div>
      {unit && <div className="text-xs text-gray-400">{unit}</div>}
    </div>
  );
}

export function SummaryTab({ parsed }: Props) {
  const hasData = parsed.hglData.length > 1;

  // Build sawtooth HGL line from parsed.hglData
  const { hglX, hglY, elevX, elevY, maopX, maopY } = useMemo(() => {
    const hX: number[] = [], hY: number[] = [];
    const eX: number[] = [], eY: number[] = [];
    const mX: (number | null)[] = [], mY: (number | null)[] = [];

    parsed.hglData.forEach((pt, i) => {
      eX.push(pt.km);
      eY.push(pt.elev);

      if (i === 0) {
        // origin: start at discharge head
        hX.push(pt.km);
        hY.push(pt.sdh > 0 ? pt.sdh : pt.head_in);
      } else {
        // arriving head (after pipe losses from upstream)
        hX.push(pt.km);
        hY.push(pt.head_in);
        // pump jump at this station
        if (pt.sdh > 0 && pt.sdh > pt.head_in) {
          hX.push(pt.km);
          hY.push(pt.sdh);
        }
      }

      // MAOP step (use previous station's maop for the segment ending here)
      if (pt.maop > 0 && i > 0) {
        const prev = parsed.hglData[i - 1];
        mX.push(prev.km, pt.km, null);
        mY.push(pt.maop, pt.maop, null);
      }
    });
    return { hglX: hX, hglY: hY, elevX: eX, elevY: eY, maopX: mX, maopY: mY };
  }, [parsed.hglData]);

  return (
    <div className="p-3 space-y-4">
      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <KpiCard label="Flow Rate" value={fmtN(parsed.flow, 1)} unit="m³/h" color="border-blue-500" />
        <KpiCard label="Total Cost" value={fmtCur(parsed.totalCost)} color="border-indigo-500" />
        <KpiCard label="Power Cost" value={fmtCur(parsed.totalPowerCost)} color="border-amber-500" />
        <KpiCard label="DRA Cost" value={fmtCur(parsed.totalDraCost)} color="border-green-500" />
        <KpiCard label="Pumps Running" value={String(parsed.totalPumpsRunning)} unit="units" color="border-cyan-500" />
      </div>

      {/* HGL Chart */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Hydraulic Grade Line</h4>
        {hasData ? (
          <Plot
            data={[
              {
                x: elevX, y: elevY, fill: 'tozeroy', fillcolor: 'rgba(161,140,120,0.3)',
                type: 'scatter', mode: 'lines', line: { color: '#92400e', width: 1 },
                name: 'Terrain',
              } as Plotly.Data,
              {
                x: maopX as number[], y: maopY as number[],
                type: 'scatter', mode: 'lines', line: { color: '#dc2626', width: 1.5, dash: 'dash' },
                name: 'MAOP',
              } as Plotly.Data,
              {
                x: hglX, y: hglY,
                type: 'scatter', mode: 'lines+markers',
                line: { color: '#1d4ed8', width: 2.5 },
                marker: { size: 5, color: '#1d4ed8' },
                name: 'HGL',
              } as Plotly.Data,
              {
                x: [parsed.terminalKm], y: [parsed.terminalMinResidual > 0
                  ? parsed.terminalElev + parsed.terminalMinResidual * 10000 / 850 : parsed.terminalHead],
                type: 'scatter', mode: 'markers',
                marker: { size: 10, color: '#16a34a', symbol: 'diamond' },
                name: 'Min Residual',
              } as Plotly.Data,
            ]}
            layout={{
              height: 300,
              margin: { t: 20, r: 20, b: 50, l: 60 },
              xaxis: { title: 'KP (km)' },
              yaxis: { title: 'Head / Elevation (m)' },
              legend: { orientation: 'h', y: -0.25 },
              plot_bgcolor: '#f8fafc',
              paper_bgcolor: 'transparent',
              font: { size: 11 },
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        ) : (
          <div className="flex items-center justify-center h-32 bg-gray-50 rounded text-gray-400 text-sm">
            Run optimization to see the hydraulic grade line
          </div>
        )}
      </div>

      {/* Station table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Station Summary</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-600">
                {['Station', 'KP (km)', 'Elev (m)', 'Type', 'SDH (m)', 'Pumps', 'Speed (rpm)',
                  'Eff %', 'TDH (m)', 'Head Loss (m)', 'Power (₹)', 'DRA ppm', 'DR%'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsed.stations.map((s, i) => {
                let cumKm = 0;
                for (let j = 0; j < i; j++) cumKm += parsed.stations[j].length_km;
                return (
                  <tr key={s.key} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-1 font-medium text-gray-800 border-b whitespace-nowrap">{s.name}</td>
                    <td className="px-2 py-1 border-b">{fmtN(cumKm, 1)}</td>
                    <td className="px-2 py-1 border-b">{fmtN(s.elevation, 1)}</td>
                    <td className="px-2 py-1 border-b">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        i === 0 ? 'bg-green-100 text-green-800' :
                        s.is_pump ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-700'
                      }`}>
                        {i === 0 ? 'Origin' : s.is_pump ? 'Pump' : 'Pass-thru'}
                      </span>
                    </td>
                    <td className="px-2 py-1 border-b">{s.sdh > 0 ? fmtN(s.sdh, 1) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.num_pumps > 0 ? s.num_pumps : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.speed > 0 ? fmtN(s.speed, 0) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.efficiency > 0 ? fmtN(s.efficiency, 1) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.tdh > 0 ? fmtN(s.tdh, 1) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.head_loss > 0 ? fmtN(s.head_loss, 1) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.power_cost > 0 ? fmtCur(s.power_cost) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.dra_ppm > 0 ? fmtN(s.dra_ppm, 1) : '—'}</td>
                    <td className="px-2 py-1 border-b">{s.drag_reduction > 0 ? `${fmtN(s.drag_reduction, 1)}%` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
