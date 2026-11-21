import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { usePipelineStore } from '../../store/pipelineStore';

function nodeColor(isOrigin: boolean, isPump: boolean): string {
  if (isOrigin) return '#16a34a';
  if (isPump) return '#1967d2';
  return '#6b7280';
}

const TERM_COLOR = '#dc2626';

const LEGEND = [
  { color: '#16a34a', label: 'Origin' },
  { color: '#1967d2', label: 'Pump Station' },
  { color: '#6b7280', label: 'Pass-through' },
  { color: '#dc2626', label: 'Terminal' },
  { color: '#f97316', label: 'Branch Tap-off' },
];

interface BranchInput { branch_name: string; flow_m3h: number; }

export function PipelineMapPreview() {
  const { stations, terminal } = usePipelineStore();

  const { data, layout } = useMemo(() => {
    if (stations.length === 0) return { data: [] as Plotly.Data[], layout: {} as Plotly.Layout };

    // Cumulative KP
    let cumKm = 0;
    const nodes = stations.map((s, i) => {
      const kp = cumKm;
      cumKm += s.length_km;
      return {
        name: s.name, kp, is_origin: i === 0, is_pump: s.is_pump,
        branches: (s.branches || []) as BranchInput[],
      };
    });
    const totalKm = cumKm;

    // Map KP → canvas x in [6, 94]
    const toX = (kp: number, idx: number) =>
      totalKm > 0 ? 6 + (kp / totalKm) * 88 : 6 + (idx / Math.max(1, nodes.length)) * 88;

    const nodeXs = nodes.map((n, i) => toX(n.kp, i));
    const termX = 94;
    const allX = [...nodeXs, termX];
    const allColors = [...nodes.map(n => nodeColor(n.is_origin, n.is_pump)), TERM_COLOR];

    // Spine segments at y=0
    const spineX: (number | null)[] = [];
    const spineY: (number | null)[] = [];
    nodes.forEach((_, i) => {
      const nextX = i < nodes.length - 1 ? nodeXs[i + 1] : termX;
      spineX.push(nodeXs[i], nextX, null);
      spineY.push(0, 0, null);
    });

    // Glow rings (behind nodes)
    const glowTrace: Plotly.Data = {
      x: allX, y: Array(allX.length).fill(0),
      type: 'scatter', mode: 'markers',
      marker: { color: allColors, size: 36, opacity: 0.08, line: { width: 0 } },
      showlegend: false, hoverinfo: 'skip',
    } as Plotly.Data;

    // Node circles
    const nodeTrace: Plotly.Data = {
      x: nodeXs, y: Array(nodeXs.length).fill(0),
      type: 'scatter', mode: 'markers',
      marker: {
        color: nodes.map(n => nodeColor(n.is_origin, n.is_pump)),
        size: 20, line: { width: 2.5, color: 'white' },
      },
      showlegend: false,
      hovertemplate: nodes.map(n =>
        `<b>${n.name}</b><br>KP: ${n.kp.toFixed(1)} km<extra></extra>`
      ),
    } as Plotly.Data;

    // Terminal diamond
    const termTrace: Plotly.Data = {
      x: [termX], y: [0],
      type: 'scatter', mode: 'markers',
      marker: { color: TERM_COLOR, size: 22, symbol: 'diamond', line: { width: 2.5, color: 'white' } },
      showlegend: false,
      hovertemplate: `<b>${terminal.name}</b><br>KP: ${totalKm.toFixed(1)} km<extra></extra>`,
    } as Plotly.Data;

    // Spine line (drawn after glow so it appears on top of glow)
    const spineTrace: Plotly.Data = {
      x: spineX, y: spineY,
      type: 'scatter', mode: 'lines',
      line: { color: '#4f46e5', width: 7 },
      showlegend: false, hoverinfo: 'skip',
    } as Plotly.Data;

    // Branch stubs (dashed orange)
    const branchTraces: Plotly.Data[] = [];
    nodes.forEach((n, i) => {
      n.branches.forEach(b => {
        const goDown = i % 2 === 0;
        branchTraces.push({
          x: [nodeXs[i], nodeXs[i] + 2, nodeXs[i] + 3.5],
          y: [0, goDown ? -1.5 : 1.5, goDown ? -2.0 : 2.0],
          type: 'scatter', mode: 'lines+markers',
          line: { color: '#f97316', width: 2, dash: 'dot' },
          marker: { color: '#f97316', size: 8, symbol: 'diamond' },
          showlegend: false,
          hovertemplate: `<b>${b.branch_name}</b><br>${b.flow_m3h} m³/h<extra></extra>`,
        } as Plotly.Data);
      });
    });

    // Annotations — station labels (alternating above/below), KP ticks, flow arrows
    const ann: Partial<Plotly.Annotations>[] = [];

    nodes.forEach((n, i) => {
      const above = i % 2 === 0;
      const col = nodeColor(n.is_origin, n.is_pump);
      ann.push({
        x: nodeXs[i], y: 0,
        ax: nodeXs[i], ay: above ? 2.0 : -2.0,
        xref: 'x' as const, yref: 'y' as const,
        axref: 'x' as const, ayref: 'y' as const,
        text: `<b>${n.name}</b>`,
        showarrow: true,
        arrowhead: 1, arrowsize: 0.8, arrowwidth: 1.5, arrowcolor: col,
        bgcolor: 'rgba(255,255,255,0.93)', bordercolor: col,
        borderwidth: 1.5, borderpad: 3,
        font: { size: 10, color: '#1f2937' },
      } as Partial<Plotly.Annotations>);

      // KP tick
      ann.push({
        x: nodeXs[i], y: -0.5,
        xref: 'x' as const, yref: 'y' as const,
        text: `${n.kp.toFixed(0)} km`,
        showarrow: false,
        font: { size: 8, color: '#9ca3af' },
        yanchor: 'top',
      } as Partial<Plotly.Annotations>);
    });

    // Terminal label
    const termAbove = nodes.length % 2 === 0;
    ann.push({
      x: termX, y: 0,
      ax: termX, ay: termAbove ? 2.0 : -2.0,
      xref: 'x' as const, yref: 'y' as const,
      axref: 'x' as const, ayref: 'y' as const,
      text: `<b>${terminal.name}</b>`,
      showarrow: true,
      arrowhead: 1, arrowsize: 0.8, arrowwidth: 1.5, arrowcolor: TERM_COLOR,
      bgcolor: 'rgba(255,255,255,0.93)', bordercolor: TERM_COLOR,
      borderwidth: 1.5, borderpad: 3,
      font: { size: 10, color: '#1f2937' },
    } as Partial<Plotly.Annotations>);
    ann.push({
      x: termX, y: -0.5,
      xref: 'x' as const, yref: 'y' as const,
      text: `${totalKm.toFixed(0)} km`,
      showarrow: false,
      font: { size: 8, color: '#9ca3af' },
      yanchor: 'top',
    } as Partial<Plotly.Annotations>);

    // Flow arrows at midpoints
    nodes.forEach((n, i) => {
      const nextX = i < nodes.length - 1 ? nodeXs[i + 1] : termX;
      const midX = (nodeXs[i] + nextX) / 2;
      const halfStep = Math.min(1.5, (nextX - nodeXs[i]) / 4);
      ann.push({
        x: midX + halfStep, y: 0.3,
        ax: midX - halfStep, ay: 0.3,
        xref: 'x' as const, yref: 'y' as const,
        axref: 'x' as const, ayref: 'y' as const,
        arrowhead: 2, arrowsize: 1.0, arrowwidth: 2.5, arrowcolor: '#818cf8',
        text: '', showarrow: true,
      } as Partial<Plotly.Annotations>);
    });

    const lyt: Partial<Plotly.Layout> = {
      height: 360,
      margin: { t: 60, r: 20, b: 60, l: 20 },
      xaxis: { visible: false, range: [-2, 102], fixedrange: true },
      yaxis: { visible: false, range: [-3.5, 3.5], fixedrange: true },
      plot_bgcolor: '#f8fafc',
      paper_bgcolor: 'transparent',
      annotations: ann as Plotly.Annotations[],
      showlegend: false,
      font: { family: 'system-ui, sans-serif' },
      hovermode: 'closest',
    };

    return {
      data: [spineTrace, glowTrace, nodeTrace, termTrace, ...branchTraces],
      layout: lyt,
    };
  }, [stations, terminal]);

  if (stations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-400 text-sm gap-2">
        <div className="text-2xl font-bold text-gray-300">|--o--o--|</div>
        Add stations in the Pipeline tab to see the schematic
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-1 px-1">Pipeline Network Schematic</h3>
      <Plot
        data={data}
        layout={layout as Plotly.Layout}
        config={{ displayModeBar: false, responsive: true, staticPlot: false }}
        style={{ width: '100%' }}
      />
      <div className="flex flex-wrap gap-3 px-2 mt-1 text-xs text-gray-500">
        {LEGEND.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
