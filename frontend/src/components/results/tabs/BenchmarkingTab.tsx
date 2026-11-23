import React from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

const fmtCur = (n: number) => `₹${n.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

// Industry reference benchmarks (typical trunk crude oil pipeline)
const BENCHMARKS = [
  { metric: 'Specific Energy (kWh/m³/100km)', typical: 1.8, good: 1.4, unit: 'kWh/m³/100km' },
  { metric: 'Pump Efficiency', typical: 72, good: 78, unit: '%' },
  { metric: 'DRA Cost Fraction', typical: 10, good: 5, unit: '% of total cost' },
];

export function BenchmarkingTab({ parsed }: Props) {
  const hasData = parsed.flow > 0 && parsed.totalCost > 0;

  // Compute benchmarkable metrics from parsed result
  const totalKm = parsed.terminalKm;
  const avgEff = (() => {
    const pumpStns = parsed.stations.filter(s => s.is_pump && s.efficiency > 0);
    return pumpStns.length > 0 ? pumpStns.reduce((a, s) => a + s.efficiency, 0) / pumpStns.length : 0;
  })();
  const draFrac = parsed.totalCost > 0 ? (parsed.totalDraCost / parsed.totalCost) * 100 : 0;
  const totalMotorKw = parsed.stations.reduce((a, s) => a + s.motor_kw, 0);
  const specificEnergy = parsed.flow > 0 && totalKm > 0 && totalMotorKw > 0
    ? (totalMotorKw / parsed.flow) / (totalKm / 100)   // kW/(m³/h) = kWh/m³, then per 100km
    : 0;

  const rows = [
    {
      metric: 'Flow Rate', value: `${parsed.flow.toFixed(1)} m³/h`,
      note: 'Actual optimized throughput',
    },
    {
      metric: 'Total Cost', value: fmtCur(parsed.totalCost),
      note: 'Power + DRA per scheduling period',
    },
    {
      metric: 'Power Cost', value: fmtCur(parsed.totalPowerCost),
      note: `${parsed.totalCost > 0 ? ((parsed.totalPowerCost / parsed.totalCost) * 100).toFixed(1) : 0}% of total`,
    },
    {
      metric: 'DRA Cost', value: fmtCur(parsed.totalDraCost),
      note: `${draFrac.toFixed(1)}% of total`,
    },
    {
      metric: 'Total Pumps Running', value: `${parsed.totalPumpsRunning}`,
      note: 'Across all stations',
    },
    {
      metric: 'Avg Pump Efficiency', value: avgEff > 0 ? `${avgEff.toFixed(1)}%` : 'N/A',
      note: avgEff > 75 ? 'Good' : avgEff > 65 ? 'Acceptable' : avgEff > 0 ? 'Below benchmark' : '—',
    },
    {
      metric: 'Specific Energy', value: specificEnergy > 0 ? `${specificEnergy.toFixed(2)} kWh/m³/100km` : 'N/A',
      note: specificEnergy > 0 && specificEnergy < 1.5 ? 'Efficient' : specificEnergy >= 1.5 && specificEnergy <= 2.5 ? 'Typical' : specificEnergy > 2.5 ? 'High — check efficiency' : '—',
    },
    {
      metric: 'DRA Penetration', value: `${parsed.stations.filter(s => s.dra_ppm > 0).length} / ${parsed.stations.length} stations`,
      note: 'Stations with active DRA injection',
    },
    {
      metric: 'Terminal Delivery Head', value: `${parsed.terminalHead.toFixed(1)} m`,
      note: `Min required: ${parsed.terminalMinResidual.toFixed(1)} m`,
    },
  ];

  return (
    <div className="p-3 space-y-4">
      <h4 className="text-sm font-semibold text-gray-700">Optimization Run Benchmarking</h4>

      {!hasData ? (
        <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
          Run optimization to see benchmarking results
        </div>
      ) : (
        <>
          {/* Metrics table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-gray-100 text-gray-600">
                  {['Metric', 'This Run', 'Notes / Benchmark'].map(h => (
                    <th key={h} className="px-2 py-1.5 text-left font-semibold border-b">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.metric} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-1.5 font-medium border-b text-gray-700">{r.metric}</td>
                    <td className="px-2 py-1.5 border-b font-semibold text-gray-900">{r.value}</td>
                    <td className="px-2 py-1.5 border-b text-gray-500">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Cost breakdown pie chart */}
          {parsed.totalCost > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-600 mb-1">Cost Composition</h4>
              <Plot
                data={[{
                  labels: ['Power/Fuel', 'DRA'],
                  values: [parsed.totalPowerCost, parsed.totalDraCost],
                  type: 'pie', hole: 0.4,
                  marker: { colors: ['#f59e0b', '#10b981'] },
                  textinfo: 'label+percent',
                  hovertemplate: '%{label}: ₹%{value:,.0f}<br>%{percent}<extra></extra>',
                } as Plotly.Data]}
                layout={{
                  height: 260, margin: { t: 10, r: 10, b: 10, l: 10 },
                  paper_bgcolor: 'transparent', font: { size: 11 }, showlegend: true,
                  legend: { orientation: 'h', y: -0.1 },
                } as unknown as Plotly.Layout}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Industry benchmarks guide */}
          <div>
            <h4 className="text-xs font-semibold text-gray-600 mb-1">Industry Reference Benchmarks</h4>
            <div className="text-xs text-gray-500 space-y-1">
              {BENCHMARKS.map(b => (
                <div key={b.metric} className="flex gap-2">
                  <span className="font-medium text-gray-700 w-56">{b.metric}:</span>
                  <span>Typical {b.typical} {b.unit} · Good &lt;{b.good} {b.unit}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
