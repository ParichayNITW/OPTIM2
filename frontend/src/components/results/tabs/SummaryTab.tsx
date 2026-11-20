import React from 'react';
import Plot from 'react-plotly.js';

interface Props {
  result: Record<string, unknown>;
  stationsInput: Array<{name: string; length_km: number; elevation: number}>;
  terminal: {name: string; elevation: number; min_residual: number};
}

export function SummaryTab({result, stationsInput, terminal}: Props) {
  const ml = (result.mainline || result) as Record<string, unknown>;

  // Build summary table rows
  const stationsUsed = (ml.stations_used || []) as Array<Record<string, unknown>>;
  const totalCost = (ml.total_cost || 0) as number;
  const powerCost = (ml.power_cost || 0) as number;
  const draCost = (ml.dra_cost || 0) as number;
  const numPumps = (ml.num_pumps || 0) as number;

  // HGL data
  const hglData = (ml.hgl_data || []) as Array<{km: number; head: number; discharge_head: number; maop: number; elev: number}>;
  const hasHGL = hglData.length > 0;

  // Build HGL from stations_used if hgl_data not present
  const kpList: number[] = [];
  const elevList: number[] = [];
  const rHList: number[] = [];
  const sdHList: number[] = [];
  const maopList: number[] = [];

  let cumKm = 0;
  stationsInput.forEach((s, i) => {
    kpList.push(cumKm);
    elevList.push(s.elevation);
    const st = stationsUsed.find(su => (su.name as string) === s.name) || {};
    rHList.push((st[`rh_in_${s.name.toLowerCase().replace(/\s+/g, '_')}`] || st.rh_in || 0) as number);
    sdHList.push((st[`sdh_${s.name.toLowerCase().replace(/\s+/g, '_')}`] || st.sdh || 0) as number);
    maopList.push((st[`maop_${s.name.toLowerCase().replace(/\s+/g, '_')}`] || st.maop || 0) as number);
    if (i < stationsInput.length - 1) cumKm += s.length_km;
  });
  kpList.push(cumKm);
  elevList.push(terminal.elevation);
  rHList.push(terminal.min_residual + terminal.elevation);
  sdHList.push(0);
  maopList.push(0);

  const hglPlotData: Plotly.Data[] = [
    {
      x: kpList, y: elevList, type: 'scatter', mode: 'lines',
      fill: 'tozeroy', fillcolor: 'rgba(139,90,43,0.15)',
      line: {color: '#92400e', width: 1.5, dash: 'dot'},
      name: 'Terrain',
    } as Plotly.Data,
  ];

  if (rHList.some(v => v > 0)) {
    hglPlotData.push({
      x: kpList, y: rHList, type: 'scatter', mode: 'lines+markers',
      line: {color: '#3b82f6', width: 2},
      marker: {size: 6},
      name: 'Arrival Head (HGL)',
    } as Plotly.Data);
  }

  if (sdHList.some(v => v > 0)) {
    hglPlotData.push({
      x: kpList, y: sdHList, type: 'scatter', mode: 'markers',
      marker: {color: '#22c55e', size: 10, symbol: 'triangle-up'},
      name: 'Discharge Head',
    } as Plotly.Data);
  }

  if (maopList.some(v => v > 0)) {
    hglPlotData.push({
      x: kpList, y: maopList, type: 'scatter', mode: 'lines',
      line: {color: '#ef4444', width: 1.5, dash: 'dash'},
      name: 'MAOP',
    } as Plotly.Data);
  }

  // Station summary table from stations_used
  const stationRows = stationsUsed.map((s, i) => ({
    name: (s.name || `Station ${i+1}`) as string,
    type: (s.type || (s.is_pump ? 'Pump' : 'Pass')) as string,
    sdh: ((s.sdh || s.discharge_head || 0) as number).toFixed(1),
    pumps: (s.num_pumps || 0) as number,
    rpm: ((s.speed || s.rpm || 0) as number).toFixed(0),
    dra_ppm: ((s.dra_ppm || 0) as number).toFixed(0),
    dr_pct: ((s.drag_reduction || s.dr_pct || 0) as number).toFixed(1),
    power_cost: ((s.power_cost || 0) as number).toFixed(0),
    dra_cost: ((s.dra_cost || 0) as number).toFixed(0),
    efficiency: ((s.efficiency || 0) as number).toFixed(1),
  }));

  return (
    <div className="p-4 space-y-4">
      {/* KPI row */}
      <div className="grid grid-cols-4 gap-3">
        {[
          {label: 'Total Cost', value: `₹${totalCost.toLocaleString('en-IN', {maximumFractionDigits: 0})}`, color: 'bg-blue-50 border-blue-200 text-blue-700'},
          {label: 'Power Cost', value: `₹${powerCost.toLocaleString('en-IN', {maximumFractionDigits: 0})}`, color: 'bg-green-50 border-green-200 text-green-700'},
          {label: 'DRA Cost', value: `₹${draCost.toLocaleString('en-IN', {maximumFractionDigits: 0})}`, color: 'bg-orange-50 border-orange-200 text-orange-700'},
          {label: 'Pumps Running', value: String(numPumps), color: 'bg-purple-50 border-purple-200 text-purple-700'},
        ].map(kpi => (
          <div key={kpi.label} className={`border rounded-lg p-3 ${kpi.color}`}>
            <div className="text-xs font-medium opacity-70">{kpi.label}</div>
            <div className="text-xl font-bold mt-1">{kpi.value}</div>
          </div>
        ))}
      </div>

      {/* HGL Chart */}
      {kpList.length > 1 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Hydraulic Grade Line</h4>
          <Plot
            data={hglPlotData}
            layout={{
              height: 280,
              margin: {t: 10, r: 20, b: 50, l: 60},
              xaxis: {title: 'KP (km)', gridcolor: '#f3f4f6'},
              yaxis: {title: 'Head (m)', gridcolor: '#f3f4f6'},
              plot_bgcolor: '#fafafa',
              paper_bgcolor: 'transparent',
              legend: {orientation: 'h', y: -0.2, font: {size: 10}},
              font: {size: 10},
            } as unknown as Plotly.Layout}
            config={{displayModeBar: false, responsive: true}}
            style={{width: '100%'}}
          />
        </div>
      )}

      {/* Station table */}
      {stationRows.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Station Results</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead className="bg-gray-100">
                <tr>
                  {['Station', 'Type', 'SDH (m)', 'Pumps', 'RPM', 'DRA ppm', 'DR%', 'Eff%', 'Power ₹', 'DRA ₹'].map(h => (
                    <th key={h} className="px-2 py-1.5 text-left border border-gray-200 font-medium text-gray-600 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stationRows.map((row, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-2 py-1 border border-gray-200 font-medium">{row.name}</td>
                    <td className="px-2 py-1 border border-gray-200">{row.type}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.sdh}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.pumps}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.rpm}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.dra_ppm}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.dr_pct}%</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">{row.efficiency}%</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">₹{parseInt(row.power_cost).toLocaleString()}</td>
                    <td className="px-2 py-1 border border-gray-200 text-right">₹{parseInt(row.dra_cost).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Raw result JSON for debugging */}
      {stationRows.length === 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer text-gray-500">Raw solver output (debug)</summary>
          <pre className="mt-2 p-2 bg-gray-50 rounded text-xs overflow-auto max-h-64">
            {JSON.stringify(ml, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
