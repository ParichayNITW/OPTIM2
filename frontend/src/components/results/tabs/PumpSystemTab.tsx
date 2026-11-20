import React, { useState } from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function PumpSystemTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;
  const pumpSystemData = (ml.pump_system_data || {}) as Record<string, unknown>;

  const stationNames = stationsUsed.map((s, i) => (s.name || `S${i+1}`) as string);
  const [selected, setSelected] = useState(stationNames[0] || '');

  const data = (pumpSystemData[selected] || {}) as {
    system?: {flows: number[], heads: number[]},
    pump_family?: Array<{n: number, flows: number[], heads: number[]}>,
    operating_point?: {flow: number, head: number, rpm: number},
  };

  const plotData: Plotly.Data[] = [];

  if (data.system) {
    plotData.push({
      x: data.system.flows, y: data.system.heads,
      type: 'scatter', mode: 'lines',
      line: {color: '#3b82f6', width: 2},
      name: 'System Curve',
    } as Plotly.Data);
  }

  if (data.pump_family) {
    const palette = ['#22c55e','#86efac','#4ade80','#16a34a','#166534'];
    data.pump_family.forEach((pc, i) => {
      plotData.push({
        x: pc.flows, y: pc.heads, type: 'scatter', mode: 'lines',
        line: {color: palette[i % palette.length], width: 1.5, dash: i === 0 ? 'solid' : 'dot'},
        name: `${pc.n} RPM`,
      } as Plotly.Data);
    });
  }

  if (data.operating_point) {
    plotData.push({
      x: [data.operating_point.flow], y: [data.operating_point.head],
      type: 'scatter', mode: 'markers',
      marker: {color: '#ef4444', size: 14, symbol: 'star'},
      name: `Op. Point (${data.operating_point.rpm.toFixed(0)} RPM)`,
    } as Plotly.Data);
  }

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

      {plotData.length > 0 ? (
        <Plot
          data={plotData}
          layout={{
            height: 360, margin: {t: 10, r: 20, b: 50, l: 70},
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
          Pump-system intersection data not available.
        </div>
      )}

      {data.operating_point && (
        <div className="bg-blue-50 border border-blue-200 rounded p-3 text-xs">
          <span className="font-semibold text-blue-700">Operating Point: </span>
          <span className="text-blue-600">
            Flow = {data.operating_point.flow.toFixed(1)} m³/h,
            Head = {data.operating_point.head.toFixed(1)} m,
            Speed = {data.operating_point.rpm.toFixed(0)} RPM
          </span>
        </div>
      )}
    </div>
  );
}
