import React from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function DRACurvesTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const draCurves = (ml.dra_curves || {}) as Record<string, {flows: number[], dr_pct: number[]}>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  const names = Object.keys(draCurves);
  const palette = ['#f97316','#eab308','#22c55e','#06b6d4','#8b5cf6','#ec4899'];

  const plotData: Plotly.Data[] = names.map((name, i) => ({
    x: draCurves[name].flows, y: draCurves[name].dr_pct,
    type: 'scatter', mode: 'lines',
    line: {color: palette[i % palette.length], width: 2},
    name,
  } as Plotly.Data));

  // Add actual operating points as markers
  stationsUsed.forEach((s, i) => {
    const ppm = (s.dra_ppm || 0) as number;
    const dr = (s.drag_reduction || s.dr_pct || 0) as number;
    const flow = (ml.FLOW || 0) as number;
    if (ppm > 0 || dr > 0) {
      plotData.push({
        x: [flow], y: [dr],
        type: 'scatter', mode: 'markers',
        marker: {color: palette[i % palette.length], size: 12, symbol: 'diamond', line: {width: 2, color: 'white'}},
        name: `${(s.name || `S${i+1}`) as string} op.`,
        showlegend: true,
      } as Plotly.Data);
    }
  });

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">DRA Drag Reduction % vs Flow</h4>
      {plotData.length > 0 ? (
        <Plot
          data={plotData}
          layout={{
            height: 340, margin: {t: 10, r: 20, b: 50, l: 70},
            xaxis: {title: 'Flow Rate (m³/h)', gridcolor: '#f3f4f6'},
            yaxis: {title: 'Drag Reduction (%)', range: [0, 100], gridcolor: '#f3f4f6'},
            plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
            legend: {orientation: 'h', y: -0.2, font: {size: 10}},
            font: {size: 11},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          DRA curve data not available. Run with DRA to see this chart.
        </div>
      )}

      {stationsUsed.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead className="bg-gray-100">
              <tr>
                {['Station', 'DRA PPM', 'DR (%)', 'DRA Cost (₹)'].map(h => (
                  <th key={h} className="px-2 py-1.5 border border-gray-200 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stationsUsed.map((s, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 border border-gray-200 font-medium">{(s.name || `S${i+1}`) as string}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">{((s.dra_ppm || 0) as number).toFixed(0)}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">{((s.drag_reduction || s.dr_pct || 0) as number).toFixed(1)}%</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">₹{((s.dra_cost || 0) as number).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
