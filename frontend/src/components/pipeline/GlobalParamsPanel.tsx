import React, { useEffect, useState } from 'react';
import { usePipelineStore } from '../../store/pipelineStore';
import apiClient from '../../api/client';

interface RegistryData {
  categories: Record<string, string[]>;
  registry: Record<string, {name: string; category: string}>;
}

export function GlobalParamsPanel() {
  const {global: g, setGlobal} = usePipelineStore();
  const [registry, setRegistry] = useState<RegistryData | null>(null);

  useEffect(() => {
    apiClient.get('/pipeline-registry').then(r => setRegistry(r.data)).catch(() => {});
  }, []);

  const categories = registry ? Object.keys(registry.categories) : ['Crude Oil Pipelines'];
  const pipelineCodes = registry?.categories[g.pipeline_category] || [];

  const field = (label: string, key: keyof typeof g, min?: number, step?: number, unit?: string) => (
    <div className="flex items-center gap-2 py-1">
      <label className="text-xs text-gray-600 w-44 shrink-0">{label}{unit && <span className="text-gray-400 ml-1">({unit})</span>}</label>
      <input
        type="number"
        value={(g[key] as number)}
        min={min}
        step={step ?? 'any'}
        onChange={e => setGlobal({[key]: parseFloat(e.target.value) || 0} as never)}
        className="w-28 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400"
      />
    </div>
  );

  return (
    <div className="space-y-1">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Fluid & Cost Parameters</h3>

      {field('Flow Rate', 'FLOW', 0, 10, 'm³/h')}

      <div className="flex items-center gap-2 py-1">
        <label className="text-xs text-gray-600 w-44 shrink-0">Pipeline Category</label>
        <select
          value={g.pipeline_category}
          onChange={e => setGlobal({pipeline_category: e.target.value, pipeline_code: registry?.categories[e.target.value]?.[0] || ''})}
          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400"
        >
          {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
        </select>
      </div>

      <div className="flex items-center gap-2 py-1">
        <label className="text-xs text-gray-600 w-44 shrink-0">Pipeline</label>
        <select
          value={g.pipeline_code}
          onChange={e => setGlobal({pipeline_code: e.target.value})}
          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400"
        >
          {pipelineCodes.map(code => (
            <option key={code} value={code}>
              {code}{registry?.registry[code] ? ` — ${registry.registry[code].name}` : ''}
            </option>
          ))}
        </select>
      </div>

      <div className="border-t pt-2 mt-1">
        {field('DRA Cost', 'RateDRA', 0, 1, '₹/kg')}
        {field('Fuel Price (HSD)', 'Price_HSD', 0, 1, '₹/L')}
        {field('Fuel Density', 'Fuel_density', 0, 0.001, 'kg/L')}
        {field('Ambient Temperature', 'Ambient_temp', -20, 1, '°C')}
        {field('MOP', 'MOP_kgcm2', 0, 1, 'kg/cm²')}
      </div>

      <div className="border-t pt-2 mt-1">
        <label className="text-xs text-gray-600 block mb-1">Pump Shear Rate: {g.pump_shear_rate.toFixed(2)}</label>
        <input
          type="range" min={0} max={1} step={0.05}
          value={g.pump_shear_rate}
          onChange={e => setGlobal({pump_shear_rate: parseFloat(e.target.value)})}
          className="w-full h-1.5 accent-blue-500"
        />
      </div>

      <div className="border-t pt-2 mt-1">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500">Viscosities (cSt), comma-sep</label>
            <input
              type="text"
              value={g.kv_list.join(', ')}
              onChange={e => {
                const vals = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                if (vals.length > 0) setGlobal({kv_list: vals});
              }}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Densities (kg/m³), comma-sep</label>
            <input
              type="text"
              value={g.rho_list.join(', ')}
              onChange={e => {
                const vals = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                if (vals.length > 0) setGlobal({rho_list: vals});
              }}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
