import React, { useState } from 'react';
import { usePipelineStore } from '../../store/pipelineStore';
import apiClient from '../../api/client';

export function DRALacingPanel() {
  const {global: g, setGlobal, stations, terminal, exportCase} = usePipelineStore();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Array<{segment: string; ppm: number; dr_pct: number}>>([]);
  const [error, setError] = useState('');

  const computeBaseline = async () => {
    setLoading(true);
    setError('');
    try {
      const caseDict = exportCase() as Record<string, unknown>;
      const r = await apiClient.post('/api/baseline-dra', {
        stations: caseDict.stations,
        terminal: caseDict.terminal,
        max_flow_m3h: g.laced_flow || g.FLOW,
        max_visc_cst: g.laced_visc,
        kv_list: caseDict.KV_list,
        rho_list: caseDict.rho_list,
        fluid_density: g.laced_density || undefined,
        mop_kgcm2: g.MOP_kgcm2 || undefined,
        min_suction_head: g.laced_suction,
      });
      const data = r.data as {segments?: Array<{name?: string; dra_ppm?: number; dra_perc?: number}>; dra_ppm?: number};
      const segs = (data.segments || []).map(s => ({
        segment: s.name || '',
        ppm: s.dra_ppm || 0,
        dr_pct: s.dra_perc || 0,
      }));
      setResult(segs);
    } catch {
      setError('Baseline computation failed — ensure stations and pump data are configured');
    } finally {
      setLoading(false);
    }
  };

  const field = (key: keyof typeof g, label: string, unit: string) => (
    <div className="flex items-center gap-2 py-0.5">
      <label className="text-xs text-gray-600 w-44 shrink-0">{label} <span className="text-gray-400">({unit})</span></label>
      <input type="number" value={g[key] as number} min={0} step="any"
        onChange={e => setGlobal({[key]: parseFloat(e.target.value) || 0} as never)}
        className="w-28 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
    </div>
  );

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-700">DRA Lacing Baseline</h3>

      <div className="flex gap-3">
        {(['auto', 'manual'] as const).map(m => (
          <label key={m} className="flex items-center gap-1 text-xs cursor-pointer">
            <input type="radio" name="dra_lacing_mode" checked={g.dra_lacing_mode === m}
              onChange={() => setGlobal({dra_lacing_mode: m})} />
            {m === 'auto' ? 'Auto (Computed)' : 'Manual (Enter PPM per segment)'}
          </label>
        ))}
      </div>

      {g.dra_lacing_mode === 'auto' && (
        <div className="pl-2 border-l-2 border-blue-200 space-y-1">
          {field('laced_flow', 'Target Laced Flow', 'm³/h')}
          {field('laced_visc', 'Target Laced Viscosity', 'cSt')}
          {field('laced_suction', 'Min Suction Pressure', 'm')}
          {field('laced_density', 'Fluid Density for Lacing', 'kg/m³')}
          <button onClick={computeBaseline} disabled={loading}
            className="mt-1 px-3 py-1.5 text-xs bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 font-medium">
            {loading ? 'Computing…' : 'Calculate Baseline DRA PPM'}
          </button>
          {error && <p className="text-xs text-amber-600">{error}</p>}
          {result.length > 0 && (
            <div className="mt-2">
              <table className="w-full text-xs border">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-2 py-1 text-left border-b">Segment</th>
                    <th className="px-2 py-1 text-right border-b">Floor PPM</th>
                    <th className="px-2 py-1 text-right border-b">Floor %DR</th>
                  </tr>
                </thead>
                <tbody>
                  {result.map((row, i) => (
                    <tr key={i} className="border-b">
                      <td className="px-2 py-0.5">{row.segment}</td>
                      <td className="px-2 py-0.5 text-right">{row.ppm.toFixed(0)}</td>
                      <td className="px-2 py-0.5 text-right">{row.dr_pct.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {g.dra_lacing_mode === 'manual' && (
        <div className="pl-2 border-l-2 border-orange-200">
          <p className="text-xs text-gray-500 mb-1">Enter minimum DRA ppm per segment below</p>
          {g.baseline_ppm_table.length === 0 && (
            <p className="text-xs text-gray-400 italic">No segments defined yet — add stations first</p>
          )}
          <div className="space-y-1">
            {g.baseline_ppm_table.map((row, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs text-gray-600 w-32">{row.segment}</span>
                <input type="number" value={row.ppm} min={0}
                  onChange={e => {
                    const updated = [...g.baseline_ppm_table];
                    updated[i] = {...row, ppm: parseFloat(e.target.value) || 0};
                    setGlobal({baseline_ppm_table: updated});
                  }}
                  className="w-24 px-2 py-1 text-xs border border-gray-300 rounded" />
                <span className="text-xs text-gray-400">ppm</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
