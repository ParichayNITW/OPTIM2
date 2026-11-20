import React from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function PerformanceTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  const names = stationsUsed.map((s, i) => (s.name || `S${i+1}`) as string);
  const velocities = stationsUsed.map(s => (s.velocity || 0) as number);
  const reynolds = stationsUsed.map(s => (s.reynolds || 0) as number);
  const headLoss = stationsUsed.map(s => (s.head_loss || 0) as number);
  const motorKw = stationsUsed.map(s => (s.motor_kw || s.power_kw || 0) as number);

  if (names.length === 0) return <div className="p-4 text-sm text-gray-400">No performance data yet.</div>;

  return (
    <div className="p-4 space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* Head Loss */}
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Head Loss per Station</h4>
          <Plot
            data={[{x: names, y: headLoss, type: 'bar', marker: {color: '#6366f1'}, name: 'Head Loss'} as Plotly.Data]}
            layout={{
              height: 220, margin: {t: 10, r: 10, b: 60, l: 60},
              xaxis: {tickangle: -30}, yaxis: {title: 'Head Loss (m)'},
              plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent', font: {size: 10},
            } as unknown as Plotly.Layout}
            config={{displayModeBar: false, responsive: true}}
            style={{width: '100%'}}
          />
        </div>

        {/* Power */}
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Motor Power (kW)</h4>
          <Plot
            data={[{x: names, y: motorKw, type: 'bar', marker: {color: '#22c55e'}, name: 'kW'} as Plotly.Data]}
            layout={{
              height: 220, margin: {t: 10, r: 10, b: 60, l: 60},
              xaxis: {tickangle: -30}, yaxis: {title: 'Power (kW)'},
              plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent', font: {size: 10},
            } as unknown as Plotly.Layout}
            config={{displayModeBar: false, responsive: true}}
            style={{width: '100%'}}
          />
        </div>
      </div>

      {/* Velocity & Reynolds */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Velocity & Reynolds Number</h4>
        <Plot
          data={[
            {x: names, y: velocities, type: 'bar', name: 'Velocity (m/s)', marker: {color: '#06b6d4'}, yaxis: 'y'} as Plotly.Data,
            {x: names, y: reynolds, type: 'bar', name: 'Reynolds No.', marker: {color: '#f97316'}, yaxis: 'y2'} as Plotly.Data,
          ]}
          layout={{
            barmode: 'group', height: 240,
            margin: {t: 10, r: 70, b: 60, l: 60},
            xaxis: {tickangle: -30},
            yaxis: {title: 'Velocity (m/s)'},
            yaxis2: {title: 'Reynolds No.', overlaying: 'y', side: 'right'},
            plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
            legend: {orientation: 'h', y: -0.35, font: {size: 10}},
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead className="bg-gray-100">
            <tr>
              {['Station', 'Head Loss (m)', 'Velocity (m/s)', 'Reynolds No.', 'Motor (kW)', 'Efficiency (%)'].map(h => (
                <th key={h} className="px-2 py-1.5 border border-gray-200 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stationsUsed.map((s, i) => (
              <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-2 py-1 border border-gray-200 font-medium">{names[i]}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">{headLoss[i].toFixed(1)}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">{velocities[i].toFixed(2)}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">{reynolds[i].toFixed(0)}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">{motorKw[i].toFixed(0)}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">{((s.efficiency || 0) as number).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
