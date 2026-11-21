import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

const fmtCur = (n: number) => `₹${Math.abs(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export function SavingsTab({ parsed }: Props) {
  const hasData = parsed.flow > 0 && parsed.totalCost > 0;

  // DRA savings estimate: compare with-DRA vs no-DRA
  // Without DRA, the same throughput requires more pump power (or is infeasible).
  // Estimate: without DRA, head loss increases by drag_reduction % → power cost scales inversely.
  // Simplified: savings = DRA cost savings vs extra power without DRA.
  const draStations = parsed.stations.filter(s => s.dra_ppm > 0);
  const avgDR = draStations.length > 0
    ? draStations.reduce((a, s) => a + s.drag_reduction, 0) / draStations.length : 0;

  // Extra power cost without DRA: power cost scales ~linearly with head loss
  // Estimate: without DRA, power cost would be totalPowerCost / (1 - avgDR/100)
  const powerCostWithoutDra = avgDR > 0 && avgDR < 100
    ? parsed.totalPowerCost / (1 - avgDR / 100) : parsed.totalPowerCost;
  const powerSavings = powerCostWithoutDra - parsed.totalPowerCost;
  const netSavings = powerSavings - parsed.totalDraCost;

  const perM3Cost = parsed.flow > 0 ? parsed.totalCost / parsed.flow : 0;

  return (
    <div className="p-3 space-y-4">
      <h4 className="text-sm font-semibold text-gray-700">Cost Savings Analysis</h4>

      {!hasData ? (
        <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
          Run optimization to see savings analysis
        </div>
      ) : (
        <>
          {/* KPI summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {[
              { label: 'Total Optimized Cost', val: fmtCur(parsed.totalCost), color: 'border-indigo-500 bg-indigo-50' },
              { label: 'Power/Fuel Cost', val: fmtCur(parsed.totalPowerCost), color: 'border-amber-400 bg-amber-50' },
              { label: 'DRA Cost', val: fmtCur(parsed.totalDraCost), color: 'border-green-500 bg-green-50' },
              { label: 'Cost per m³', val: `₹${perM3Cost.toFixed(2)}/m³`, color: 'border-blue-500 bg-blue-50' },
            ].map(c => (
              <div key={c.label} className={`rounded-lg border-l-4 p-3 ${c.color}`}>
                <div className="text-xs text-gray-500">{c.label}</div>
                <div className="text-sm font-bold text-gray-900">{c.val}</div>
              </div>
            ))}
          </div>

          {/* DRA savings waterfall chart */}
          {draStations.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 mb-1">DRA Net Savings Waterfall</h4>
              <Plot
                data={[{
                  x: ['Base Power Cost\n(without DRA)', 'Power Savings\nfrom DRA', 'DRA Chemical\nCost', 'Net Cost\nwith DRA'],
                  y: [powerCostWithoutDra, -powerSavings, parsed.totalDraCost, parsed.totalCost],
                  type: 'waterfall',
                  measure: ['absolute', 'relative', 'relative', 'total'],
                  decreasing: { marker: { color: '#10b981' } },
                  increasing: { marker: { color: '#ef4444' } },
                  totals: { marker: { color: '#6366f1' } },
                  connector: { line: { color: '#9ca3af', width: 1.5 } },
                  hovertemplate: '%{x}<br>₹%{y:,.0f}<extra></extra>',
                } as Plotly.Data]}
                layout={{
                  height: 300,
                  margin: { t: 20, r: 20, b: 70, l: 80 },
                  yaxis: { title: 'Cost (₹)' },
                  plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
                  showlegend: false,
                } as unknown as Plotly.Layout}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Savings table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-gray-100 text-gray-600">
                  {['Item', 'Value', 'Notes'].map(h => (
                    <th key={h} className="px-2 py-1.5 text-left font-semibold border-b">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { item: 'Optimized Total Cost', val: fmtCur(parsed.totalCost), note: 'Power + DRA (this run)' },
                  { item: 'Est. Power Cost Without DRA', val: fmtCur(powerCostWithoutDra), note: `+${avgDR.toFixed(1)}% head loss without drag reduction` },
                  { item: 'Power Savings from DRA', val: fmtCur(powerSavings), note: 'Avoided pumping energy' },
                  { item: 'DRA Chemical Cost', val: fmtCur(parsed.totalDraCost), note: 'Actual DRA expenditure' },
                  {
                    item: 'Net DRA Benefit',
                    val: `${netSavings >= 0 ? '+' : '-'}${fmtCur(netSavings)}`,
                    note: netSavings >= 0 ? 'Positive ROI — DRA is beneficial' : 'Review DRA dosage or pricing',
                  },
                  { item: 'DRA Injection Points', val: `${draStations.length}`, note: draStations.map(s => s.name).join(', ') || 'None' },
                  { item: 'Avg Drag Reduction', val: `${avgDR.toFixed(1)}%`, note: 'Across active injection stations' },
                  { item: 'Specific Cost', val: `₹${perM3Cost.toFixed(2)}/m³`, note: 'Total cost per cubic meter throughput' },
                ].map((r, i) => (
                  <tr key={r.item} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-1.5 font-medium border-b text-gray-700">{r.item}</td>
                    <td className={`px-2 py-1.5 border-b font-semibold ${
                      r.item === 'Net DRA Benefit'
                        ? netSavings >= 0 ? 'text-green-700' : 'text-red-600' : 'text-gray-900'
                    }`}>{r.val}</td>
                    <td className="px-2 py-1.5 border-b text-gray-500">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-xs text-gray-400">
            * DRA savings estimate uses simplified linear head-loss model. Actual savings depend on pipeline hydraulics, DRA efficiency curve, and scheduling period.
          </p>
        </>
      )}
    </div>
  );
}
