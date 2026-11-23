import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

export function DRACurvesTab({ parsed }: Props) {
  const draStations = parsed.stations.filter(s => s.is_pump);
  const hasActiveDRA = draStations.some(s => s.dra_ppm > 0 || s.drag_reduction > 0);
  const hasData = draStations.length > 0;

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        No pump stations found
      </div>
    );
  }

  const names = draStations.map(s => s.name);
  const ppms  = draStations.map(s => s.dra_ppm);
  const drs   = draStations.map(s => s.drag_reduction);
  const costs = draStations.map(s => s.dra_cost);

  return (
    <div className="p-3 space-y-4">
      {!hasActiveDRA && (
        <div className="bg-blue-50 border border-blue-200 rounded px-3 py-2 text-xs text-blue-700">
          No DRA injection in current run — showing station capacities (all values at 0 ppm baseline)
        </div>
      )}
      {/* KPI summary */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border-l-4 border-green-500 bg-green-50 p-3">
          <div className="text-xs text-gray-500">Total DRA Cost</div>
          <div className="text-base font-bold text-gray-900">
            {`₹${parsed.totalDraCost.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
          </div>
        </div>
        <div className="rounded-lg border-l-4 border-teal-500 bg-teal-50 p-3">
          <div className="text-xs text-gray-500">Injection Points</div>
          <div className="text-base font-bold text-gray-900">{draStations.length}</div>
        </div>
        <div className="rounded-lg border-l-4 border-cyan-500 bg-cyan-50 p-3">
          <div className="text-xs text-gray-500">Max DRA ppm</div>
          <div className="text-base font-bold text-gray-900">{Math.max(...ppms).toFixed(1)} ppm</div>
        </div>
      </div>

      {/* DRA ppm and DR% bar charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">DRA Dosage (ppm)</h4>
          <Plot
            data={[{
              x: names, y: ppms, type: 'bar', name: 'DRA ppm',
              marker: { color: '#14b8a6' },
              hovertemplate: '%{x}<br>%{y:.1f} ppm<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 220, margin: { t: 10, r: 10, b: 80, l: 60 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: 'ppm' },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Drag Reduction (%)</h4>
          <Plot
            data={[{
              x: names, y: drs, type: 'bar', name: 'DR %',
              marker: { color: '#8b5cf6' },
              hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 220, margin: { t: 10, r: 10, b: 80, l: 60 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: '%', range: [0, 100] },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* DRA ppm vs DR% scatter */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">DRA Dosage vs Drag Reduction Characteristic</h4>
        <Plot
          data={[{
            x: ppms, y: drs,
            type: 'scatter', mode: 'text+markers',
            text: names, textposition: 'top center',
            marker: { size: 10, color: costs, colorscale: 'Viridis', showscale: true,
              colorbar: { title: 'DRA Cost (₹)', len: 0.8 } },
            hovertemplate: '<b>%{text}</b><br>DRA: %{x:.1f} ppm<br>DR: %{y:.1f}%<extra></extra>',
          } as Plotly.Data]}
          layout={{
            height: 280, margin: { t: 20, r: 80, b: 60, l: 60 },
            xaxis: { title: 'DRA Dosage (ppm)' },
            yaxis: { title: 'Drag Reduction (%)' },
            plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
            showlegend: false,
          } as unknown as Plotly.Layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* DRA table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">DRA Injection Summary</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-600">
                {['Station', 'DRA ppm', 'Drag Reduction %', 'DRA Cost (₹)'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {draStations.map((s, i) => (
                <tr key={s.key} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 font-medium border-b">{s.name}</td>
                  <td className="px-2 py-1 border-b">{s.dra_ppm.toFixed(1)}</td>
                  <td className="px-2 py-1 border-b">{s.drag_reduction.toFixed(1)}%</td>
                  <td className="px-2 py-1 border-b text-green-700">
                    {`₹${s.dra_cost.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
