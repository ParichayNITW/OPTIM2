import React, { useState } from 'react';
import { PumpConfig, QHPoint, EffPoint, TimeSlot } from '../../store/pipelineStore';
import { EditableTable } from './EditableTable';
import { DieselConfig } from './DieselConfig';

interface Props {
  pump: PumpConfig;
  typeLabel: string;
  onChange: (patch: Partial<PumpConfig>) => void;
}

export function PumpTypeTab({pump, typeLabel, onChange}: Props) {
  const [openSection, setOpenSection] = useState<string>('curves');

  const toggle = (s: string) => setOpenSection(o => o === s ? '' : s);

  if (!pump.enabled) {
    return (
      <div className="p-2">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={false} onChange={() => onChange({enabled: true})} />
          <span className="text-xs text-gray-600">Enable Pump Type {typeLabel}</span>
        </label>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-2">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={pump.enabled} onChange={() => onChange({enabled: false})} />
          <span className="text-xs font-medium text-gray-700">Pump Type {typeLabel} — Active</span>
        </label>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Available:</label>
          <input type="number" value={pump.available} min={1} max={10}
            onChange={e => onChange({available: parseInt(e.target.value) || 1})}
            className="w-12 px-1 py-0.5 text-xs border border-gray-300 rounded" />
        </div>
      </div>

      <div className="flex items-center gap-2">
        <label className="text-xs text-gray-500 w-20 shrink-0">Pump Name</label>
        <input type="text" value={pump.name}
          onChange={e => onChange({name: e.target.value})}
          placeholder={`Pump ${typeLabel}`}
          className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded" />
      </div>

      {/* Q-H Curve */}
      <details open className="border rounded">
        <summary className="px-2 py-1 text-xs font-medium cursor-pointer text-gray-700 bg-gray-50">
          Q-H Curve (Flow vs Head) — min 3 points
        </summary>
        <div className="p-2">
          <EditableTable<QHPoint>
            rows={pump.qh_table}
            columns={[
              {key: 'flow', label: 'Flow (m³/h)', min: 0, step: 10},
              {key: 'head', label: 'Head (m)', min: 0, step: 1},
            ]}
            onChange={rows => onChange({qh_table: rows})}
            minRows={3}
            defaultRow={{flow: 0, head: 0}}
          />
        </div>
      </details>

      {/* Efficiency Curve */}
      <details className="border rounded">
        <summary className="px-2 py-1 text-xs font-medium cursor-pointer text-gray-700 bg-gray-50">
          Efficiency Curve (Flow vs Eff%) — min 5 points
        </summary>
        <div className="p-2">
          <EditableTable<EffPoint>
            rows={pump.eff_table}
            columns={[
              {key: 'flow', label: 'Flow (m³/h)', min: 0, step: 10},
              {key: 'eff', label: 'Efficiency (%)', min: 0, step: 1},
            ]}
            onChange={rows => onChange({eff_table: rows})}
            minRows={5}
            defaultRow={{flow: 0, eff: 0}}
          />
        </div>
      </details>

      {/* Power Source */}
      <details className="border rounded">
        <summary className="px-2 py-1 text-xs font-medium cursor-pointer text-gray-700 bg-gray-50">
          Power Source & Tariff
        </summary>
        <div className="p-2 space-y-2">
          <div className="flex gap-4">
            {(['Grid', 'Diesel'] as const).map(src => (
              <label key={src} className="flex items-center gap-1 text-xs cursor-pointer">
                <input type="radio" name={`power-${pump.name}-${typeLabel}`}
                  checked={pump.power_source === src}
                  onChange={() => onChange({power_source: src})} />
                {src}
              </label>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600 w-28">Min RPM</label>
            <input type="number" value={pump.min_rpm} min={0} step={50}
              onChange={e => onChange({min_rpm: parseFloat(e.target.value) || 0})}
              className="w-24 px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600 w-28">Rated/DOL RPM</label>
            <input type="number" value={pump.dol_rpm} min={0} step={50}
              onChange={e => onChange({dol_rpm: parseFloat(e.target.value) || 0})}
              className="w-24 px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>

          {pump.power_source === 'Grid' && (
            <div className="space-y-1">
              <div className="flex gap-3">
                {(['Fixed', 'Varying'] as const).map(t => (
                  <label key={t} className="flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" name={`tariff-${pump.name}-${typeLabel}`}
                      checked={pump.tariff_type === t}
                      onChange={() => onChange({tariff_type: t})} />
                    {t} Tariff
                  </label>
                ))}
              </div>
              {pump.tariff_type === 'Fixed' ? (
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-600 w-28">Rate (₹/kWh)</label>
                  <input type="number" value={pump.fixed_rate} min={0} step={0.1}
                    onChange={e => onChange({fixed_rate: parseFloat(e.target.value) || 0})}
                    className="w-24 px-2 py-1 text-xs border border-gray-300 rounded" />
                </div>
              ) : (
                <EditableTable<TimeSlot>
                  rows={pump.time_slots}
                  columns={[
                    {key: 'start', label: 'Start', type: 'text'},
                    {key: 'end', label: 'End', type: 'text'},
                    {key: 'rate', label: 'Rate (₹/kWh)', step: 0.1},
                  ]}
                  onChange={rows => onChange({time_slots: rows})}
                  minRows={1}
                  defaultRow={{start: '00:00', end: '06:00', rate: 5.0}}
                />
              )}
            </div>
          )}

          {pump.power_source === 'Diesel' && (
            <DieselConfig pump={pump} onChange={onChange} />
          )}
        </div>
      </details>
    </div>
  );
}
