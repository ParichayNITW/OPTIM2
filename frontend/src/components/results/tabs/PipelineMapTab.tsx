import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { ParsedResult, ParsedStation } from '../../../utils/resultParser';

interface Props { parsed: ParsedResult; }

function resultNodeColor(s: ParsedStation, isOrigin: boolean): string {
  if (isOrigin) return '#16a34a';
  if (s.is_pump && s.dra_ppm > 0) return '#0284c7';
  if (s.is_pump) return '#1967d2';
  if (s.dra_ppm > 0) return '#d97706';
  return '#6b7280';
}

const TERM_COLOR = '#dc2626';

const LEGEND = [
  { color: '#16a34a', label: 'Origin' },
  { color: '#1967d2', label: 'Pump' },
  { color: '#0284c7', label: 'Pump+DRA' },
  { color: '#d97706', label: 'DRA Only' },
  { color: '#6b7280', label: 'Pass-thru' },
  { color: '#dc2626', label: 'Terminal' },
];

export function PipelineMapTab({ parsed }: Props) {
  const hasResult = parsed.hglData.length > 1;

  const { schematicData, schematicLayout, hglData, hglLayout } = useMemo(() => {
    const stations = parsed.stations;
    if (stations.length === 0) {
      return { schematicData: [] as Plotly.Data[], schematicLayout: {} as Plotly.Layout, hglData: [] as Plotly.Data[], hglLayout: {} as Plotly.Layout };
    }

    // --- SCHEMATIC ---
    let cumKm = 0;
    const nodes = stations.map((s, i) => {
      const kp = cumKm;
      cumKm += s.length_km;
      return { s, kp, is_origin: i === 0 };
    });
    const totalKm = cumKm;

    const toX = (kp: number, idx: number) =>
      totalKm > 0 ? 6 + (kp / totalKm) * 88 : 6 + (idx / Math.max(1, nodes.length)) * 88;

    const nodeXs = nodes.map((n, i) => toX(n.kp, i));
    const termX = 94;
    const allX = [...nodeXs, termX];
    const allColors = [...nodes.map(n => resultNodeColor(n.s, n.is_origin)), TERM_COLOR];

    // Spine
    const spineX: (number | null)[] = [], spineY: (number | null)[] = [];
    nodes.forEach((_, i) => {
      const nextX = i < nodes.length - 1 ? nodeXs[i + 1] : termX;
      spineX.push(nodeXs[i], nextX, null);
      spineY.push(0, 0, null);
    });

    // Pipe segment coloring by pressure ratio (sdh / maop), if data available
    const segTraces: Plotly.Data[] = [];
    if (hasResult) {
      nodes.forEach((n, i) => {
        const nextX = i < nodes.length - 1 ? nodeXs[i + 1] : termX;
        const ratio = n.s.maop > 0 ? n.s.sdh / n.s.maop : 0;
        const segColor = ratio > 0.90 ? '#ef4444' : ratio > 0.70 ? '#f59e0b' : '#4f46e5';
        segTraces.push({
          x: [nodeXs[i], nextX], y: [0, 0],
          type: 'scatter', mode: 'lines',
          line: { color: segColor, width: 7 },
          showlegend: false, hoverinfo: 'skip',
        } as Plotly.Data);
      });
    }

    const glowTrace: Plotly.Data = {
      x: allX, y: Array(allX.length).fill(0),
      type: 'scatter', mode: 'markers',
      marker: { color: allColors, size: 36, opacity: 0.08, line: { width: 0 } },
      showlegend: false, hoverinfo: 'skip',
    } as Plotly.Data;

    const nodeTrace: Plotly.Data = {
      x: nodeXs, y: Array(nodeXs.length).fill(0),
      type: 'scatter', mode: 'markers',
      marker: { color: nodes.map(n => resultNodeColor(n.s, n.is_origin)), size: 20, line: { width: 2.5, color: 'white' } },
      showlegend: false,
      hovertemplate: nodes.map(n => {
        const s = n.s;
        const lines = [
          `<b>${s.name}</b>`,
          `KP: ${n.kp.toFixed(1)} km`,
          s.sdh > 0 ? `SDH: ${s.sdh.toFixed(1)} m` : '',
          s.num_pumps > 0 ? `Pumps: ${s.num_pumps} × ${s.speed > 0 ? s.speed.toFixed(0) : '?'} rpm` : '',
          s.dra_ppm > 0 ? `DRA: ${s.dra_ppm.toFixed(1)} ppm / DR: ${s.drag_reduction.toFixed(1)}%` : '',
          s.power_cost > 0 ? `Cost: ₹${Math.round(s.power_cost + s.dra_cost).toLocaleString()}` : '',
        ].filter(Boolean).join('<br>');
        return `${lines}<extra></extra>`;
      }),
    } as Plotly.Data;

    const termLabel = parsed.hglData.length > 0 ? parsed.hglData[parsed.hglData.length - 1].label : 'Terminal';
    const termTrace: Plotly.Data = {
      x: [termX], y: [0],
      type: 'scatter', mode: 'markers',
      marker: { color: TERM_COLOR, size: 22, symbol: 'diamond', line: { width: 2.5, color: 'white' } },
      showlegend: false,
      hovertemplate: `<b>${termLabel}</b><br>KP: ${totalKm.toFixed(1)} km<br>Delivery Head: ${parsed.terminalHead.toFixed(1)} m<extra></extra>`,
    } as Plotly.Data;

    // Spine fallback (no result = uniform indigo)
    const spineTrace: Plotly.Data = {
      x: spineX, y: spineY, type: 'scatter', mode: 'lines',
      line: { color: 'rgba(79,70,229,0.15)', width: 7 },
      showlegend: false, hoverinfo: 'skip',
    } as Plotly.Data;

    // Annotations
    const ann: Partial<Plotly.Annotations>[] = [];
    nodes.forEach((n, i) => {
      const above = i % 2 === 0;
      const col = resultNodeColor(n.s, n.is_origin);
      const s = n.s;
      const labelLines = [
        `<b>${s.name}</b>`,
        s.sdh > 0 ? `↑ ${s.sdh.toFixed(0)} m` : '',
        s.num_pumps > 0 ? `${s.num_pumps}× ${s.speed > 0 ? s.speed.toFixed(0) : '?'} rpm` : '',
        s.dra_ppm > 0 ? `${s.dra_ppm.toFixed(0)} ppm` : '',
      ].filter(Boolean).join('<br>');

      ann.push({
        x: nodeXs[i], y: 0, ax: nodeXs[i], ay: above ? 2.0 : -2.0,
        xref: 'x' as const, yref: 'y' as const, axref: 'x' as const, ayref: 'y' as const,
        text: labelLines, showarrow: true,
        arrowhead: 1, arrowsize: 0.8, arrowwidth: 1.5, arrowcolor: col,
        bgcolor: 'rgba(255,255,255,0.93)', bordercolor: col, borderwidth: 1.5, borderpad: 3,
        font: { size: 9, color: '#1f2937' }, align: 'left',
      } as Partial<Plotly.Annotations>);
      ann.push({
        x: nodeXs[i], y: -0.5, xref: 'x' as const, yref: 'y' as const,
        text: `${n.kp.toFixed(0)} km`, showarrow: false,
        font: { size: 8, color: '#9ca3af' }, yanchor: 'top',
      } as Partial<Plotly.Annotations>);
    });

    const termAbove = nodes.length % 2 === 0;
    ann.push({
      x: termX, y: 0, ax: termX, ay: termAbove ? 2.0 : -2.0,
      xref: 'x' as const, yref: 'y' as const, axref: 'x' as const, ayref: 'y' as const,
      text: `<b>${termLabel}</b><br>${parsed.terminalHead.toFixed(0)} m`,
      showarrow: true, arrowhead: 1, arrowsize: 0.8, arrowwidth: 1.5, arrowcolor: TERM_COLOR,
      bgcolor: 'rgba(255,255,255,0.93)', bordercolor: TERM_COLOR, borderwidth: 1.5, borderpad: 3,
      font: { size: 9, color: '#1f2937' },
    } as Partial<Plotly.Annotations>);
    ann.push({
      x: termX, y: -0.5, xref: 'x' as const, yref: 'y' as const,
      text: `${totalKm.toFixed(0)} km`, showarrow: false,
      font: { size: 8, color: '#9ca3af' }, yanchor: 'top',
    } as Partial<Plotly.Annotations>);

    // Flow arrows at midpoints
    nodes.forEach((n, i) => {
      const nextX = i < nodes.length - 1 ? nodeXs[i + 1] : termX;
      const midX = (nodeXs[i] + nextX) / 2;
      const halfStep = Math.min(1.5, (nextX - nodeXs[i]) / 4);
      ann.push({
        x: midX + halfStep, y: 0.3, ax: midX - halfStep, ay: 0.3,
        xref: 'x' as const, yref: 'y' as const, axref: 'x' as const, ayref: 'y' as const,
        arrowhead: 2, arrowsize: 1.0, arrowwidth: 2.5, arrowcolor: '#818cf8',
        text: '', showarrow: true,
      } as Partial<Plotly.Annotations>);
    });

    const schLyt: Partial<Plotly.Layout> = {
      height: 380,
      margin: { t: 60, r: 20, b: 60, l: 20 },
      xaxis: { visible: false, range: [-2, 102], fixedrange: true },
      yaxis: { visible: false, range: [-3.5, 3.5], fixedrange: true },
      plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent',
      annotations: ann as Plotly.Annotations[], showlegend: false,
      font: { family: 'system-ui, sans-serif' }, hovermode: 'closest',
    };

    // --- HGL chart ---
    const hX: number[] = [], hY: number[] = [];
    const eX: number[] = [], eY: number[] = [];
    const mX: (number | null)[] = [], mY: (number | null)[] = [];

    parsed.hglData.forEach((pt, i) => {
      eX.push(pt.km); eY.push(pt.elev);
      if (i === 0) {
        hX.push(pt.km); hY.push(pt.sdh > 0 ? pt.sdh : pt.head_in);
      } else {
        hX.push(pt.km); hY.push(pt.head_in);
        if (pt.sdh > 0 && pt.sdh > pt.head_in) { hX.push(pt.km); hY.push(pt.sdh); }
      }
      if (pt.maop > 0 && i > 0) {
        const prev = parsed.hglData[i - 1];
        mX.push(prev.km, pt.km, null); mY.push(pt.maop, pt.maop, null);
      }
    });

    const hglPlotData: Plotly.Data[] = [
      { x: eX, y: eY, fill: 'tozeroy', fillcolor: 'rgba(161,140,120,0.3)', type: 'scatter', mode: 'lines', line: { color: '#92400e', width: 1 }, name: 'Terrain' } as Plotly.Data,
      { x: mX as number[], y: mY as number[], type: 'scatter', mode: 'lines', line: { color: '#dc2626', width: 1.5, dash: 'dash' }, name: 'MAOP' } as Plotly.Data,
      { x: hX, y: hY, type: 'scatter', mode: 'lines+markers', line: { color: '#1d4ed8', width: 2.5 }, marker: { size: 5 }, name: 'HGL' } as Plotly.Data,
    ];

    const hglLyt = {
      height: 240, margin: { t: 20, r: 20, b: 50, l: 70 },
      xaxis: { title: { text: 'KP (km)' } }, yaxis: { title: { text: 'Head / Elev (m)' } },
      legend: { orientation: 'h', y: -0.3 },
      plot_bgcolor: '#f8fafc', paper_bgcolor: 'transparent', font: { size: 10 },
    };

    return {
      schematicData: hasResult
        ? [...segTraces, glowTrace, nodeTrace, termTrace, ...ann.length ? [] : [spineTrace]]
        : [spineTrace, glowTrace, nodeTrace, termTrace],
      schematicLayout: schLyt as Plotly.Layout,
      hglData: hglPlotData,
      hglLayout: hglLyt as Plotly.Layout,
    };
  }, [parsed, hasResult]);

  return (
    <div className="p-3 space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">Pipeline Schematic — Optimization Results</h4>

      <Plot
        data={schematicData}
        layout={schematicLayout}
        config={{ displayModeBar: false, responsive: true, staticPlot: false }}
        style={{ width: '100%' }}
      />

      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        {LEGEND.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
            {label}
          </div>
        ))}
        {hasResult && (
          <>
            <div className="flex items-center gap-1">
              <span className="inline-block w-4 h-1.5 rounded bg-green-500" />
              <span>Pipe &lt;70% MAOP</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-4 h-1.5 rounded bg-amber-400" />
              <span>70–90%</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="inline-block w-4 h-1.5 rounded bg-red-500" />
              <span>&gt;90% MAOP</span>
            </div>
          </>
        )}
      </div>

      {hasResult && (
        <>
          <h4 className="text-sm font-semibold text-gray-700">Hydraulic Grade Line Profile</h4>
          <Plot
            data={hglData}
            layout={hglLayout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </>
      )}
    </div>
  );
}
