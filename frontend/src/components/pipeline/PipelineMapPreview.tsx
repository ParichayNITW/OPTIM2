import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { usePipelineStore } from '../../store/pipelineStore';

function nodeColor(isFirst: boolean, isLast: boolean, isPump: boolean): string {
  if (isFirst) return '#22c55e';   // green — Origin
  if (isLast) return '#ef4444';    // red — Terminal
  if (isPump) return '#3b82f6';    // blue — Pump
  return '#9ca3af';                // gray — Pass-through
}

export function PipelineMapPreview() {
  const {stations, terminal} = usePipelineStore();

  const {data, layout} = useMemo(() => {
    if (stations.length === 0) return {data: [], layout: {}};

    // Compute cumulative KP — each station's length_km is the downstream pipe segment length
    let cumKm = 0;
    const nodes = stations.map((s, i) => {
      const kp = cumKm;
      cumKm += s.length_km;  // ALL stations contribute (last station's L reaches terminal)
      return {
        name: s.name, kp, elev: s.elevation,
        is_pump: s.is_pump, is_first: i === 0, is_last: false,
        branches: s.branches,
      };
    });
    const terminalKp = cumKm;

    // Pipe segments trace (line)
    const segX: number[] = [];
    const segY: number[] = [];
    nodes.forEach((n, i) => {
      const nextKp = i < nodes.length - 1 ? nodes[i + 1].kp : terminalKp;
      const nextElev = i < nodes.length - 1 ? nodes[i + 1].elev : terminal.elevation;
      segX.push(n.kp, nextKp, null as never);
      segY.push(n.elev, nextElev, null as never);
    });

    // Station scatter
    const stX = nodes.map(n => n.kp);
    const stY = nodes.map(n => n.elev);
    const stColors = nodes.map((n, i) => nodeColor(n.is_first, false, n.is_pump));
    const stText = nodes.map(n => n.name);

    // Terminal
    const traces: Plotly.Data[] = [
      {
        x: segX, y: segY, type: 'scatter', mode: 'lines',
        line: {color: '#6366f1', width: 3},
        name: 'Pipeline', showlegend: false,
      } as Plotly.Data,
      {
        x: stX, y: stY, type: 'scatter', mode: 'markers+text' as never,
        marker: {color: stColors, size: 12, symbol: 'circle'},
        text: stText,
        textposition: 'top center',
        textfont: {size: 10},
        name: 'Stations', showlegend: false,
        hovertemplate: '%{text}<br>KP: %{x:.1f} km<br>Elev: %{y:.1f} m<extra></extra>',
      } as Plotly.Data,
      {
        x: [terminalKp], y: [terminal.elevation], type: 'scatter', mode: 'markers+text' as never,
        marker: {color: '#ef4444', size: 14, symbol: 'diamond'},
        text: [terminal.name],
        textposition: 'top center',
        textfont: {size: 10},
        name: terminal.name, showlegend: false,
      } as Plotly.Data,
    ];

    // Branch lines
    nodes.forEach(n => {
      n.branches.forEach(b => {
        traces.push({
          x: [n.kp, n.kp + 2],
          y: [n.elev, n.elev + 20],
          type: 'scatter', mode: 'lines+markers',
          line: {color: '#f97316', width: 2, dash: 'dot'},
          marker: {color: '#f97316', size: 8, symbol: 'diamond'},
          name: b.branch_name,
          showlegend: false,
          hovertemplate: `${b.branch_name}<br>${b.flow_m3h} m³/h<extra></extra>`,
        } as Plotly.Data);
      });
    });

    const lyt = {
      height: 300,
      margin: {t: 20, r: 20, b: 50, l: 60},
      xaxis: {title: 'KP (km)', gridcolor: '#f3f4f6'},
      yaxis: {title: 'Elevation (m)', gridcolor: '#f3f4f6'},
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'transparent',
      font: {size: 11},
    } as unknown as Plotly.Layout;

    return {data: traces, layout: lyt};
  }, [stations, terminal]);

  if (stations.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        Add stations to see pipeline schematic
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2 px-1">Pipeline Network Preview</h3>
      <Plot
        data={data}
        layout={layout as Plotly.Layout}
        config={{displayModeBar: false, responsive: true}}
        style={{width: '100%'}}
      />
      <div className="flex flex-wrap gap-3 px-2 mt-1 text-xs text-gray-500">
        {[
          {color: '#22c55e', label: 'Origin'},
          {color: '#3b82f6', label: 'Pump Station'},
          {color: '#9ca3af', label: 'Pass-through'},
          {color: '#ef4444', label: 'Terminal'},
          {color: '#f97316', label: 'Branch Tap-off'},
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
