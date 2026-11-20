import React, { useState } from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function SystemCurvesTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const sysCurves = (ml.system_curves || {}) as Record<string, unknown>;
  const pumpCurves = (ml.pump_curves || {}) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  const stationNames = stationsUsed.map((s, i) => (s.name || `S${i+1}`) as string);
  const [selected, setSelected] = useState(stationNames[0] || '');

  const sysCurveData = (sysCurves[selected] || {}) as {flows?: number[], heads?: number[]};
  const pumpCurveData = (pumpCurves[selected] || {}) as {flows?: number[], heads?: number[]};

  const plotData: Plotly.Data[] = [];

  if (sysCurveData.flows && sysCurveData.heads) {
    plotData.push({
      x: sysCurveData.flows, y: sysCurveData.heads,
      type: 'scatter', mode: 'lines',
      line: {color: '#3b82f6', width: 2},
      name: 'System Curve',
    } as Plotly.Data);
  }

  if (pumpCurveData.flows && pumpCurveData.heads) {
    plotData.push({
      x: pumpCurveData.flows, y: pumpCurveData.heads,
      type: 'scatter', mode: 'lines',
      line: {color: '#22c55e', width: 2},
      name: 'Pump Curve (H-Q)',
    } as Plotly.Data);
  }

  const hasData = plotData.length > 0;

  return (
    <div className="p-4 space-y-3">
      {stationNames.length > 0 && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-600 font-medium">Station:</label>
          <select value={selected} onChange={e => setSelected(e.target.value)}
            className="text-xs border border-gray-300 rounded px-2 py-1 focus:outline-none focus:border-blue-400">
            {stationNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      )}

      {hasData ? (
        <Plot
          data={plotData}
          layout={{
            height: 350, margin: {t: 10, r: 20, b: 50, l: 70},
            xaxis: {title: 'Flow Rate (m³/h)', gridcolor: '#f3f4f6'},
            yaxis: {title: 'Head (m)', gridcolor: '#f3f4f6'},
            plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
            legend: {orientation: 'h', y: -0.2, font: {size: 10}},
            font: {size: 11},
          } as unknown as Plotly.Layout}
          config={{displayModeBar: false, responsive: true}}
          style={{width: '100%'}}
        />
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          System curve data not available in this result.
        </div>
      )}
    </div>
  );
}
