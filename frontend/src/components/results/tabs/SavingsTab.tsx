import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function SavingsTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const baseTotalCost = (ml.total_cost || 0) as number;
  const basePowerCost = (ml.power_cost || 0) as number;
  const baseDraCost = (ml.dra_cost || 0) as number;

  const [draRateAdj, setDraRateAdj] = useState(0);
  const [fuelPriceAdj, setFuelPriceAdj] = useState(0);
  const [flowAdj, setFlowAdj] = useState(0);
  const [opsHoursAdj, setOpsHoursAdj] = useState(0);

  const adjustedDraCost = useMemo(() => baseDraCost * (1 + draRateAdj / 100), [baseDraCost, draRateAdj]);
  const adjustedPowerCost = useMemo(() => basePowerCost * (1 + fuelPriceAdj / 100) * (1 + flowAdj / 100) * (1 + opsHoursAdj / 100), [basePowerCost, fuelPriceAdj, flowAdj, opsHoursAdj]);
  const adjustedTotal = adjustedDraCost + adjustedPowerCost;
  const saving = baseTotalCost - adjustedTotal;
  const savingPct = baseTotalCost > 0 ? (saving / baseTotalCost) * 100 : 0;

  const months = Array.from({length: 12}, (_, i) => `Month ${i + 1}`);
  const cumulativeSavings = months.map((_, i) => saving * (i + 1));

  return (
    <div className="p-4 space-y-4">
      <h4 className="text-sm font-semibold text-gray-700">Savings Simulator</h4>
      <p className="text-xs text-gray-500">Adjust parameters below to simulate how cost changes from the optimized baseline.</p>

      {/* Sliders */}
      <div className="grid grid-cols-2 gap-4">
        {[
          {label: 'DRA Rate Change (%)', value: draRateAdj, setter: setDraRateAdj, min: -50, max: 100},
          {label: 'Fuel/Tariff Price Change (%)', value: fuelPriceAdj, setter: setFuelPriceAdj, min: -50, max: 100},
          {label: 'Flow Rate Change (%)', value: flowAdj, setter: setFlowAdj, min: -30, max: 50},
          {label: 'Operating Hours Change (%)', value: opsHoursAdj, setter: setOpsHoursAdj, min: -30, max: 50},
        ].map(({label, value, setter, min, max}) => (
          <div key={label} className="space-y-1">
            <div className="flex justify-between text-xs text-gray-600">
              <span>{label}</span>
              <span className={`font-medium ${value > 0 ? 'text-red-600' : value < 0 ? 'text-green-600' : ''}`}>
                {value > 0 ? '+' : ''}{value}%
              </span>
            </div>
            <input type="range" min={min} max={max} step={1} value={value}
              onChange={e => setter(parseInt(e.target.value))}
              className="w-full" />
          </div>
        ))}
      </div>

      {/* KPI summary */}
      <div className="grid grid-cols-3 gap-3">
        {[
          {label: 'Baseline Cost', value: `₹${baseTotalCost.toLocaleString('en-IN', {maximumFractionDigits: 0})}`, color: 'bg-gray-50 border-gray-200 text-gray-700'},
          {label: 'Adjusted Cost', value: `₹${adjustedTotal.toLocaleString('en-IN', {maximumFractionDigits: 0})}`, color: 'bg-blue-50 border-blue-200 text-blue-700'},
          {label: saving >= 0 ? 'Saving' : 'Extra Cost', value: `₹${Math.abs(saving).toLocaleString('en-IN', {maximumFractionDigits: 0})} (${Math.abs(savingPct).toFixed(1)}%)`, color: saving >= 0 ? 'bg-green-50 border-green-200 text-green-700' : 'bg-red-50 border-red-200 text-red-700'},
        ].map(kpi => (
          <div key={kpi.label} className={`border rounded-lg p-3 ${kpi.color}`}>
            <div className="text-xs font-medium opacity-70">{kpi.label}</div>
            <div className="text-base font-bold mt-1">{kpi.value}</div>
          </div>
        ))}
      </div>

      {/* Cumulative savings chart */}
      {baseTotalCost > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-gray-600 mb-2">Cumulative Annual Savings Projection</h5>
          <Plot
            data={[{
              x: months, y: cumulativeSavings,
              type: 'scatter', mode: 'lines+markers',
              fill: saving >= 0 ? 'tozeroy' : undefined,
              fillcolor: 'rgba(34,197,94,0.1)',
              line: {color: saving >= 0 ? '#22c55e' : '#ef4444', width: 2},
              marker: {size: 6},
              name: 'Cumulative Savings',
            } as Plotly.Data]}
            layout={{
              height: 220, margin: {t: 10, r: 20, b: 50, l: 80},
              xaxis: {gridcolor: '#f3f4f6'},
              yaxis: {title: '₹', gridcolor: '#f3f4f6', zeroline: true, zerolinecolor: '#d1d5db'},
              plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
              font: {size: 10},
            } as unknown as Plotly.Layout}
            config={{displayModeBar: false, responsive: true}}
            style={{width: '100%'}}
          />
        </div>
      )}
    </div>
  );
}
