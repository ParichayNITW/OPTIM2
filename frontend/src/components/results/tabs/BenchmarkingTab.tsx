import React from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function BenchmarkingTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const benchmarks = (ml.benchmarks || {}) as Record<string, number>;

  const totalCost = (ml.total_cost || 0) as number;
  const powerCost = (ml.power_cost || 0) as number;
  const draCost = (ml.dra_cost || 0) as number;
  const flow = (ml.FLOW || 0) as number;

  const specificCost = flow > 0 ? totalCost / flow : 0;

  const benchmarkRows = [
    {metric: 'Total Cost (₹)', actual: totalCost, bm: benchmarks.total_cost},
    {metric: 'Power Cost (₹)', actual: powerCost, bm: benchmarks.power_cost},
    {metric: 'DRA Cost (₹)', actual: draCost, bm: benchmarks.dra_cost},
    {metric: 'Specific Cost (₹/m³h)', actual: specificCost, bm: benchmarks.specific_cost},
  ].filter(r => r.actual > 0 || r.bm != null);

  const hasChart = benchmarkRows.some(r => r.bm != null && r.bm > 0);

  return (
    <div className="p-4 space-y-4">
      <h4 className="text-sm font-semibold text-gray-700">Benchmarking Comparison</h4>

      {hasChart ? (
        <Plot
          data={[
            {
              type: 'bar', name: 'Optimized',
              x: benchmarkRows.map(r => r.metric), y: benchmarkRows.map(r => r.actual),
              marker: {color: '#3b82f6'},
            } as Plotly.Data,
            {
              type: 'bar', name: 'Benchmark',
              x: benchmarkRows.map(r => r.metric), y: benchmarkRows.map(r => r.bm || 0),
              marker: {color: '#d1d5db'},
            } as Plotly.Data,
          ]}
          layout={{
            barmode: 'group', height: 300,
            margin: {t: 10, r: 20, b: 70, l: 70},
            xaxis: {tickangle: -20},
            yaxis: {title: 'Value'},
            plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
            legend: {orientation: 'h', y: -0.35, font: {size: 10}},
            font: {size: 10},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      ) : null}

      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead className="bg-gray-100">
            <tr>
              {['Metric', 'Optimized', 'Benchmark', 'Difference', 'Savings (%)'].map(h => (
                <th key={h} className="px-2 py-1.5 border border-gray-200 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {benchmarkRows.map((row, i) => {
              const diff = row.bm != null ? row.actual - row.bm : null;
              const pct = row.bm != null && row.bm > 0 ? ((row.bm - row.actual) / row.bm) * 100 : null;
              return (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1 border border-gray-200 font-medium">{row.metric}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">{row.actual.toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
                  <td className="px-2 py-1 border border-gray-200 text-right">{row.bm != null ? row.bm.toLocaleString('en-IN', {maximumFractionDigits: 0}) : '—'}</td>
                  <td className={`px-2 py-1 border border-gray-200 text-right ${diff != null && diff < 0 ? 'text-green-600' : diff != null && diff > 0 ? 'text-red-600' : ''}`}>
                    {diff != null ? (diff >= 0 ? '+' : '') + diff.toLocaleString('en-IN', {maximumFractionDigits: 0}) : '—'}
                  </td>
                  <td className={`px-2 py-1 border border-gray-200 text-right font-medium ${pct != null && pct > 0 ? 'text-green-600' : pct != null && pct < 0 ? 'text-red-600' : ''}`}>
                    {pct != null ? (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%' : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="bg-gray-50 border border-gray-200 rounded p-3 text-xs text-gray-500">
        Note: Benchmark values are populated when benchmark data is provided by the solver.
        Positive "Savings" = your optimized cost is lower than the benchmark.
      </div>
    </div>
  );
}
