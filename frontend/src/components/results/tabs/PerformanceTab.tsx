import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

export function PerformanceTab({ parsed }: Props) {
  const names = parsed.stations.map(s => s.name);
  const hasData = parsed.stations.some(s => s.velocity > 0 || s.efficiency > 0);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        Run optimization to see performance metrics
      </div>
    );
  }

  const velocities   = parsed.stations.map(s => s.velocity);
  const efficiencies = parsed.stations.map(s => s.efficiency);
  const headLosses   = parsed.stations.map(s => s.head_loss);
  const motorKws     = parsed.stations.map(s => s.motor_kw);
  const reynolds     = parsed.stations.map(s => s.reynolds);

  return (
    <div className="p-3 space-y-4">
      {/* Top row charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Velocity */}
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Flow Velocity (m/s)</h4>
          <Plot
            data={[{
              x: names, y: velocities, type: 'bar', name: 'Velocity',
              marker: {
                color: velocities.map(v =>
                  v < 1.0 ? '#10b981' : v < 2.0 ? '#f59e0b' : '#ef4444'
                ),
              },
              hovertemplate: '%{x}<br>%{y:.2f} m/s<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 200, margin: { t: 10, r: 10, b: 70, l: 50 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: 'm/s' },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
              shapes: [{ type: 'line', x0: -0.5, x1: names.length - 0.5, y0: 1.5, y1: 1.5, line: { color: '#ef4444', dash: 'dot', width: 1 } }],
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Efficiency */}
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Pump Efficiency (%)</h4>
          <Plot
            data={[{
              x: names.filter((_, i) => parsed.stations[i].is_pump && efficiencies[i] > 0),
              y: efficiencies.filter((_, i) => parsed.stations[i].is_pump && efficiencies[i] > 0),
              type: 'bar', name: 'Efficiency',
              marker: {
                color: efficiencies
                  .filter((_, i) => parsed.stations[i].is_pump && efficiencies[i] > 0)
                  .map(e => e >= 70 ? '#10b981' : e >= 60 ? '#f59e0b' : '#ef4444'),
              },
              hovertemplate: '%{x}<br>%{y:.1f}%<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 200, margin: { t: 10, r: 10, b: 70, l: 50 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: '%', range: [0, 100] },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Head Loss */}
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Segment Head Loss (m)</h4>
          <Plot
            data={[{
              x: names, y: headLosses, type: 'bar', name: 'Head Loss',
              marker: { color: '#6366f1' },
              hovertemplate: '%{x}<br>%{y:.1f} m<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 200, margin: { t: 10, r: 10, b: 70, l: 60 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: 'm' },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Motor kW */}
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Motor Shaft Power (kW)</h4>
          <Plot
            data={[{
              x: names.filter((_, i) => motorKws[i] > 0),
              y: motorKws.filter(v => v > 0),
              type: 'bar', name: 'Motor kW',
              marker: { color: '#0284c7' },
              hovertemplate: '%{x}<br>%{y:.1f} kW<extra></extra>',
            } as Plotly.Data]}
            layout={{
              height: 200, margin: { t: 10, r: 10, b: 70, l: 60 },
              xaxis: { tickangle: -30, automargin: true },
              yaxis: { title: 'kW' },
              plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
              showlegend: false,
            } as unknown as Plotly.Layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Performance table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Detailed Performance Metrics</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-600">
                {['Station', 'Velocity (m/s)', 'Reynolds', 'Efficiency %', 'TDH (m)',
                  'Motor kW', 'Num Pumps', 'Speed (rpm)', 'Pipeline Flow (m³/h)'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsed.stations.map((s, i) => (
                <tr key={s.key} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 font-medium border-b">{s.name}</td>
                  <td className="px-2 py-1 border-b">
                    <span className={s.velocity > 2.0 ? 'text-red-600 font-medium' : s.velocity > 1.5 ? 'text-amber-600' : ''}>
                      {s.velocity > 0 ? s.velocity.toFixed(2) : '—'}
                    </span>
                  </td>
                  <td className="px-2 py-1 border-b">{s.reynolds > 0 ? (s.reynolds / 1000).toFixed(1) + 'k' : '—'}</td>
                  <td className="px-2 py-1 border-b">
                    <span className={s.efficiency < 60 && s.efficiency > 0 ? 'text-red-600' : s.efficiency < 70 && s.efficiency > 0 ? 'text-amber-600' : 'text-green-700'}>
                      {s.efficiency > 0 ? s.efficiency.toFixed(1) + '%' : '—'}
                    </span>
                  </td>
                  <td className="px-2 py-1 border-b">{s.tdh > 0 ? s.tdh.toFixed(1) : '—'}</td>
                  <td className="px-2 py-1 border-b">{s.motor_kw > 0 ? s.motor_kw.toFixed(1) : '—'}</td>
                  <td className="px-2 py-1 border-b">{s.num_pumps > 0 ? s.num_pumps : '—'}</td>
                  <td className="px-2 py-1 border-b">{s.speed > 0 ? s.speed.toFixed(0) : '—'}</td>
                  <td className="px-2 py-1 border-b">{s.pipeline_flow > 0 ? s.pipeline_flow.toFixed(1) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
