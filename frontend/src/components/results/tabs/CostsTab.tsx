import React from 'react';
import Plot from 'react-plotly.js';

interface Props {
  result: Record<string, unknown>;
}

export function CostsTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  const names = stationsUsed.map((s, i) => (s.name || `S${i+1}`) as string);
  const powerCosts = stationsUsed.map(s => (s.power_cost || 0) as number);
  const draCosts = stationsUsed.map(s => (s.dra_cost || 0) as number);
  const totalPerStation = powerCosts.map((p, i) => p + draCosts[i]);
  const grandTotal = totalPerStation.reduce((a, b) => a + b, 0);

  if (names.length === 0) {
    return <div className="p-4 text-sm text-gray-400">No cost data available yet.</div>;
  }

  return (
    <div className="p-4 space-y-4">
      {/* Stacked bar chart */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Cost Breakdown by Station</h4>
        <Plot
          data={[
            {
              x: names, y: powerCosts, type: 'bar', name: 'Power & Fuel Cost',
              marker: {color: '#3b82f6'},
            } as Plotly.Data,
            {
              x: names, y: draCosts, type: 'bar', name: 'DRA Cost',
              marker: {color: '#f59e0b'},
            } as Plotly.Data,
          ]}
          layout={{
            barmode: 'stack',
            height: 280,
            margin: {t: 10, r: 20, b: 70, l: 70},
            xaxis: {title: 'Station', tickangle: -30},
            yaxis: {title: 'Cost (₹)'},
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'transparent',
            legend: {orientation: 'h', y: -0.35, font: {size: 10}},
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      </div>

      {/* Pie chart */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Cost Contribution (%)</h4>
        <Plot
          data={[{
            labels: names,
            values: totalPerStation,
            type: 'pie',
            hole: 0.4,
            textinfo: 'label+percent',
          } as Plotly.Data]}
          layout={{
            height: 260,
            margin: {t: 10, r: 20, b: 10, l: 20},
            paper_bgcolor: 'transparent',
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      </div>

      {/* Cost table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Cost Summary Table</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead className="bg-gray-100">
              <tr>
                {['Station', 'Power & Fuel (₹)', 'DRA (₹)', 'Total (₹)', 'Share (%)'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left border border-gray-200 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {names.map((name, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 border border-gray-200 font-medium">{name}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">₹{powerCosts[i].toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">₹{draCosts[i].toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right font-medium">₹{totalPerStation[i].toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">{grandTotal > 0 ? (100 * totalPerStation[i] / grandTotal).toFixed(1) : '0.0'}%</td>
                </tr>
              ))}
              <tr className="bg-blue-50 font-bold">
                <td className="px-2 py-1 border border-gray-200">TOTAL</td>
                <td className="px-2 py-1 border border-gray-200 text-right">₹{powerCosts.reduce((a, b) => a + b, 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">₹{draCosts.reduce((a, b) => a + b, 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">₹{grandTotal.toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                <td className="px-2 py-1 border border-gray-200 text-right">100%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
