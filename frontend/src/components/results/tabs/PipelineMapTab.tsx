import React from 'react';
import Plot from 'react-plotly.js';

interface Props {
  result: Record<string, unknown>;
  stationsInput: Array<{name: string; length_km: number; elevation: number; is_pump: boolean}>;
  terminal: {name: string; elevation: number};
}

export function PipelineMapTab({result, stationsInput, terminal}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;

  let cumKm = 0;
  const nodes = stationsInput.map((s, i) => {
    const kp = cumKm;
    if (i < stationsInput.length - 1) cumKm += s.length_km;
    const used = stationsUsed.find(u => (u.name as string) === s.name) || {};
    return {
      name: s.name, kp, elev: s.elevation, is_pump: s.is_pump,
      sdh: (used.sdh || used.discharge_head || 0) as number,
      dra_ppm: (used.dra_ppm || 0) as number,
      power_cost: (used.power_cost || 0) as number,
    };
  });
  const terminalKp = cumKm;

  // Pipe segments
  const segX: number[] = [];
  const segY: number[] = [];
  nodes.forEach((n, i) => {
    const nextKp = i < nodes.length - 1 ? nodes[i + 1].kp : terminalKp;
    const nextElev = i < nodes.length - 1 ? nodes[i + 1].elev : terminal.elevation;
    segX.push(n.kp, nextKp, null as never);
    segY.push(n.elev, nextElev, null as never);
  });

  const getColor = (n: typeof nodes[0]) => {
    if (n.kp === 0) return '#22c55e';
    if (!n.is_pump) return '#9ca3af';
    if (n.dra_ppm > 0 && n.is_pump) return '#06b6d4';
    if (n.dra_ppm > 0) return '#f59e0b';
    return '#3b82f6';
  };

  const hoverText = nodes.map(n =>
    `${n.name}<br>KP: ${n.kp.toFixed(1)} km<br>SDH: ${n.sdh.toFixed(1)} m` +
    (n.dra_ppm > 0 ? `<br>DRA: ${n.dra_ppm.toFixed(0)} ppm` : '') +
    (n.power_cost > 0 ? `<br>Cost: ₹${n.power_cost.toFixed(0)}` : '')
  );

  const data: Plotly.Data[] = [
    {
      x: segX, y: segY, type: 'scatter', mode: 'lines',
      line: {color: '#6366f1', width: 3},
      name: 'Pipeline', showlegend: false,
    } as Plotly.Data,
    {
      x: nodes.map(n => n.kp), y: nodes.map(n => n.elev),
      type: 'scatter', mode: 'markers+text' as never,
      marker: {color: nodes.map(getColor), size: 14, symbol: 'circle',
               line: {width: 2, color: 'white'}},
      text: nodes.map(n => n.name),
      textposition: 'top center', textfont: {size: 10},
      hovertemplate: '%{customdata}<extra></extra>',
      customdata: hoverText,
      name: 'Stations', showlegend: false,
    } as Plotly.Data,
    {
      x: [terminalKp], y: [terminal.elevation],
      type: 'scatter', mode: 'markers+text' as never,
      marker: {color: '#ef4444', size: 16, symbol: 'diamond', line: {width: 2, color: 'white'}},
      text: [terminal.name], textposition: 'top center', textfont: {size: 10},
      name: terminal.name, showlegend: false,
    } as Plotly.Data,
  ];

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">Post-Optimization Pipeline Map</h4>
      <Plot
        data={data}
        layout={{
          height: 320, margin: {t: 10, r: 20, b: 50, l: 70},
          xaxis: {title: 'KP (km)', gridcolor: '#f3f4f6'},
          yaxis: {title: 'Elevation (m)', gridcolor: '#f3f4f6'},
          plot_bgcolor: '#fafafa', paper_bgcolor: 'transparent',
          font: {size: 11},
        } as unknown as Plotly.Layout}
        config={{displayModeBar: false, responsive: true}}
        style={{width: '100%'}}
      />
      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        {[
          {color: '#22c55e', label: 'Origin'},
          {color: '#3b82f6', label: 'Pump Station'},
          {color: '#06b6d4', label: 'Pump + DRA'},
          {color: '#f59e0b', label: 'DRA Only'},
          {color: '#9ca3af', label: 'Pass-through'},
          {color: '#ef4444', label: 'Terminal'},
        ].map(({color, label}) => (
          <div key={label} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full inline-block" style={{background: color}} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
