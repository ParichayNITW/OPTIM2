import React from 'react';
import Plot from 'react-plotly.js';

interface Props { result: Record<string, unknown>; }

export function SensitivityTab({result}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const sensitivity = (ml.sensitivity || {}) as Record<string, Array<{pct: number, cost: number}>>;

  const paramNames = Object.keys(sensitivity);
  const hasData = paramNames.length > 0;

  const palette = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#06b6d4'];

  const plotData: Plotly.Data[] = paramNames.map((param, i) => {
    const pts = sensitivity[param];
    return {
      x: pts.map(p => p.pct), y: pts.map(p => p.cost),
      type: 'scatter', mode: 'lines+markers',
      line: {color: palette[i % palette.length], width: 2},
      marker: {size: 6},
      name: param,
    } as Plotly.Data;
  });

  // Tornado chart: cost change at ±20%
  const tornadoData: Array<{name: string, low: number, high: number, base: number}> = [];
  paramNames.forEach(param => {
    const pts = sensitivity[param];
    if (pts.length >= 2) {
      const base = pts.find(p => p.pct === 0)?.cost || pts[Math.floor(pts.length / 2)].cost;
      const lo = Math.min(...pts.map(p => p.cost));
      const hi = Math.max(...pts.map(p => p.cost));
      tornadoData.push({name: param, low: lo - base, high: hi - base, base});
    }
  });
  tornadoData.sort((a, b) => (Math.abs(b.high - b.low) - Math.abs(a.high - a.low)));

  return (
    <div className="p-4 space-y-4">
      {hasData ? (
        <>
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Sensitivity: Cost vs Parameter Change</h4>
            <Plot
              data={plotData}
              layout={{
                height: 300, margin: {t: 10, r: 20, b: 50, l: 80},
                xaxis: {title: 'Parameter Change (%)', gridcolor: '#f3f4f6', zeroline: true, zerolinecolor: '#d1d5db'},
                yaxis: {title: 'Total Cost (₹)', gridcolor: '#f3f4f6'},
                plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
                legend: {orientation: 'h', y: -0.2, font: {size: 10}},
                font: {size: 11},
              } as unknown as Plotly.Layout}
              config={{displayModeBar: false, responsive: true}}
              style={{width: '100%'}}
            />
          </div>

          {tornadoData.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2">Tornado Chart (Impact at Extreme Values)</h4>
              <Plot
                data={[
                  {
                    type: 'bar', orientation: 'h',
                    y: tornadoData.map(d => d.name),
                    x: tornadoData.map(d => d.high),
                    name: 'Upside', marker: {color: '#ef4444'},
                  } as Plotly.Data,
                  {
                    type: 'bar', orientation: 'h',
                    y: tornadoData.map(d => d.name),
                    x: tornadoData.map(d => d.low),
                    name: 'Downside', marker: {color: '#22c55e'},
                  } as Plotly.Data,
                ]}
                layout={{
                  barmode: 'overlay', height: 220,
                  margin: {t: 10, r: 20, b: 50, l: 120},
                  xaxis: {title: 'Cost Change (₹)', gridcolor: '#f3f4f6', zeroline: true, zerolinecolor: '#d1d5db'},
                  plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
                  legend: {orientation: 'h', y: -0.3, font: {size: 10}},
                  font: {size: 10},
                } as unknown as Plotly.Layout}
                config={{displayModeBar: false, responsive: true}}
                style={{width: '100%'}}
              />
            </div>
          )}
        </>
      ) : (
        <div className="flex items-center justify-center h-48 text-gray-400 text-sm">
          Sensitivity analysis data not available.
        </div>
      )}
    </div>
  );
}
