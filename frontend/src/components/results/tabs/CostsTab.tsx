import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

const fmtCur = (n: number) => `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export function CostsTab({ parsed }: Props) {
  const names = parsed.stations.map(s => s.name);
  const powerCosts = parsed.stations.map(s => s.power_cost);
  const draCosts = parsed.stations.map(s => s.dra_cost);

  const hasData = powerCosts.some(v => v > 0) || draCosts.some(v => v > 0);

  if (!hasData) {
    return (
      <div className="p-4 flex items-center justify-center h-48 text-gray-400 text-sm">
        Run optimization to see cost breakdown
      </div>
    );
  }

  return (
    <div className="p-3 space-y-4">
      {/* KPI summary */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Total Cost', value: fmtCur(parsed.totalCost), color: 'bg-indigo-50 border-indigo-400' },
          { label: 'Power Cost', value: fmtCur(parsed.totalPowerCost), color: 'bg-amber-50 border-amber-400' },
          { label: 'DRA Cost', value: fmtCur(parsed.totalDraCost), color: 'bg-green-50 border-green-400' },
        ].map(c => (
          <div key={c.label} className={`rounded-lg border-l-4 p-3 ${c.color}`}>
            <div className="text-xs text-gray-500">{c.label}</div>
            <div className="text-base font-bold text-gray-900">{c.value}</div>
          </div>
        ))}
      </div>

      {/* Stacked bar chart */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Cost per Station</h4>
        <Plot
          data={[
            {
              x: names, y: powerCosts, type: 'bar', name: 'Power/Fuel Cost',
              marker: { color: '#f59e0b' },
              hovertemplate: '%{x}<br>Power: ₹%{y:,.0f}<extra></extra>',
            } as Plotly.Data,
            {
              x: names, y: draCosts, type: 'bar', name: 'DRA Cost',
              marker: { color: '#10b981' },
              hovertemplate: '%{x}<br>DRA: ₹%{y:,.0f}<extra></extra>',
            } as Plotly.Data,
          ]}
          layout={{
            barmode: 'stack',
            height: 280,
            margin: { t: 20, r: 20, b: 90, l: 80 },
            xaxis: { tickangle: -30 },
            yaxis: { title: 'Cost (₹)' },
            legend: { orientation: 'h', y: -0.35 },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: 'transparent',
            font: { size: 11 },
          } as unknown as Plotly.Layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Cost table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-700 mb-1">Cost Breakdown by Station</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-gray-100 text-gray-600">
                {['Station', 'Type', 'Power/Fuel Cost (₹)', 'DRA Cost (₹)', 'Total (₹)', '% of Total'].map(h => (
                  <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsed.stations.map((s, i) => {
                const stationTotal = s.power_cost + s.dra_cost;
                const pct = parsed.totalCost > 0 ? (stationTotal / parsed.totalCost) * 100 : 0;
                return (
                  <tr key={s.key} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-1 font-medium border-b">{s.name}</td>
                    <td className="px-2 py-1 border-b">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${i === 0 ? 'bg-green-100 text-green-800' : s.is_pump ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'}`}>
                        {i === 0 ? 'Origin' : s.is_pump ? 'Pump' : 'Pass-thru'}
                      </span>
                    </td>
                    <td className="px-2 py-1 border-b text-amber-700">{s.power_cost > 0 ? fmtCur(s.power_cost) : '—'}</td>
                    <td className="px-2 py-1 border-b text-green-700">{s.dra_cost > 0 ? fmtCur(s.dra_cost) : '—'}</td>
                    <td className="px-2 py-1 border-b font-medium">{stationTotal > 0 ? fmtCur(stationTotal) : '—'}</td>
                    <td className="px-2 py-1 border-b">
                      {pct > 0 ? (
                        <div className="flex items-center gap-1">
                          <div className="h-2 bg-indigo-400 rounded" style={{ width: `${Math.min(pct, 100) * 0.6}px` }} />
                          <span>{pct.toFixed(1)}%</span>
                        </div>
                      ) : '—'}
                    </td>
                  </tr>
                );
              })}
              <tr className="bg-gray-100 font-semibold">
                <td className="px-2 py-1.5 border-t" colSpan={2}>Total</td>
                <td className="px-2 py-1.5 border-t text-amber-700">{fmtCur(parsed.totalPowerCost)}</td>
                <td className="px-2 py-1.5 border-t text-green-700">{fmtCur(parsed.totalDraCost)}</td>
                <td className="px-2 py-1.5 border-t">{fmtCur(parsed.totalCost)}</td>
                <td className="px-2 py-1.5 border-t">100%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
