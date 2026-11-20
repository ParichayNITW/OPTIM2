import React from 'react';
import { PumpConfig } from '../../store/pipelineStore';

interface Props {
  pump: PumpConfig;
  onChange: (patch: Partial<PumpConfig>) => void;
}

export function DieselConfig({pump, onChange}: Props) {
  const computeISO = () => {
    // Weighted average SFC at typical operating mix: 70% load dominant
    const weighted = 0.3 * pump.sfc50 + 0.5 * pump.sfc75 + 0.2 * pump.sfc100;
    // Convert from gm/kWh to gm/bhp·hr (1 kW = 1.341 bhp)
    const sfc_bhp = weighted / 1.341;
    onChange({sfc: parseFloat(sfc_bhp.toFixed(1))});
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2 items-center">
        <label className="text-xs text-gray-600 w-28 shrink-0">SFC Mode</label>
        <div className="flex gap-3">
          {(['manual', 'iso3046'] as const).map(m => (
            <label key={m} className="flex items-center gap-1 text-xs cursor-pointer">
              <input type="radio" name={`sfc-mode-${pump.name}`} checked={pump.sfc_mode === m}
                onChange={() => onChange({sfc_mode: m})} />
              {m === 'manual' ? 'Manual SFC' : 'ISO 3046 Engine'}
            </label>
          ))}
        </div>
      </div>

      {pump.sfc_mode === 'manual' ? (
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-600 w-28 shrink-0">SFC (gm/bhp·hr)</label>
          <input type="number" value={pump.sfc} min={150} step={1}
            onChange={e => onChange({sfc: parseFloat(e.target.value) || 0})}
            className="w-24 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
        </div>
      ) : (
        <div className="space-y-1.5 pl-2 border-l-2 border-blue-200">
          {[
            {key: 'engine_make', label: 'Engine Make', type: 'text'},
            {key: 'engine_model', label: 'Engine Model', type: 'text'},
            {key: 'rated_power', label: 'Rated Power (kW)', type: 'number'},
          ].map(f => (
            <div key={f.key} className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-28 shrink-0">{f.label}</label>
              <input type={f.type}
                value={String((pump as never)[f.key as keyof PumpConfig] ?? '')}
                onChange={e => onChange({[f.key]: f.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value} as Partial<PumpConfig>)}
                className="w-32 px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
            </div>
          ))}
          <div className="grid grid-cols-3 gap-2 mt-1">
            {[
              {key: 'sfc50', label: 'SFC @ 50% load'},
              {key: 'sfc75', label: 'SFC @ 75% load'},
              {key: 'sfc100', label: 'SFC @ 100% load'},
            ].map(f => (
              <div key={f.key}>
                <label className="text-xs text-gray-500 block">{f.label} (gm/kWh)</label>
                <input type="number" value={(pump as never)[f.key as keyof PumpConfig] as number} min={150} step={1}
                  onChange={e => onChange({[f.key]: parseFloat(e.target.value) || 0} as Partial<PumpConfig>)}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
              </div>
            ))}
          </div>
          <button onClick={computeISO}
            className="mt-1 px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700">
            Compute ISO 3046 SFC
          </button>
          {pump.sfc > 0 && (
            <p className="text-xs text-green-600">Computed SFC: {pump.sfc.toFixed(1)} gm/bhp·hr</p>
          )}
        </div>
      )}
    </div>
  );
}
