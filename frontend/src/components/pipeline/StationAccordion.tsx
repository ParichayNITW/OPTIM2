import React, { useState } from 'react';
import { StationData, PeakPoint } from '../../store/pipelineStore';
import { usePipelineStore } from '../../store/pipelineStore';
import { PumpTypeTab } from './PumpTypeTab';
import { BranchBuilder } from './BranchBuilder';
import { EditableTable } from './EditableTable';

interface Props {
  station: StationData;
  index: number;
  total: number;
}

export function StationAccordion({station, index, total}: Props) {
  const {updateStation, updatePump, removeStation, moveStation} = usePipelineStore();
  const [tab, setTab] = useState<'pump' | 'peaks' | 'branch'>('pump');

  const up = (key: string, label: string, min?: number, step?: number) => (
    <div>
      <label className="text-xs text-gray-500 block">{label}</label>
      <input type="number" value={(station as never)[key as keyof StationData] as number} min={min} step={step ?? 'any'}
        onChange={e => updateStation(station.id, {[key]: parseFloat(e.target.value) || 0} as Partial<StationData>)}
        className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
    </div>
  );

  const pumpTypeColor = station.pumpA.enabled && station.pumpB.enabled ? 'bg-purple-50 border-purple-200'
    : station.pumpA.enabled ? 'bg-blue-50 border-blue-200'
    : station.pumpB.enabled ? 'bg-indigo-50 border-indigo-200'
    : '';

  return (
    <details className={`border rounded-lg mb-2 ${station.is_pump ? pumpTypeColor : 'border-gray-200'}`}>
      <summary className="px-3 py-2 cursor-pointer flex items-center justify-between hover:bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2">
          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white ${
            index === 0 ? 'bg-green-500' : station.is_pump ? 'bg-blue-500' : 'bg-gray-400'
          }`}>{index + 1}</span>
          <span className="text-sm font-medium text-gray-800">{station.name || `Station ${index + 1}`}</span>
          {station.is_pump && <span className="text-xs text-blue-600 bg-blue-100 px-1.5 py-0.5 rounded">PUMP</span>}
          <span className="text-xs text-gray-400">{station.length_km} km · {station.OD_in}" OD</span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={e => {e.preventDefault(); moveStation(station.id, 'up');}}
            disabled={index === 0}
            className="text-gray-400 hover:text-gray-600 disabled:opacity-20 px-1">↑</button>
          <button onClick={e => {e.preventDefault(); moveStation(station.id, 'down');}}
            disabled={index === total - 1}
            className="text-gray-400 hover:text-gray-600 disabled:opacity-20 px-1">↓</button>
          <button onClick={e => {e.preventDefault(); removeStation(station.id);}}
            className="text-red-400 hover:text-red-600 px-1 text-xs">✕</button>
        </div>
      </summary>

      <div className="px-3 pb-3 pt-1 space-y-3">
        {/* Basic station properties */}
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="text-xs text-gray-500 block">Name</label>
            <input type="text" value={station.name}
              onChange={e => updateStation(station.id, {name: e.target.value})}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
          {up('elevation', 'Elevation (m)')}
          {up('length_km', 'Length to Next (km)', 0, 1)}
        </div>

        <div className="grid grid-cols-3 gap-2">
          {up('OD_in', 'OD (in)', 0, 0.5)}
          {up('wall_thk_in', 'Wall Thk (in)', 0, 0.025)}
          {up('SMYS_psi', 'SMYS (psi)', 0, 1000)}
        </div>

        <div className="grid grid-cols-3 gap-2">
          {up('roughness_m', 'Roughness (m)', 0, 0.000001)}
          {up('max_dra_pct', 'Max DRA%', 0, 1)}
          {up('min_suction_head', 'Min Suction Head (m)', 0, 1)}
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="flex items-center gap-1.5 text-xs cursor-pointer mt-3">
              <input type="checkbox" checked={station.is_pump}
                onChange={e => updateStation(station.id, {is_pump: e.target.checked})} />
              <span className="text-gray-600">Is Pumping Station</span>
            </label>
          </div>
          {up('max_pumps', 'Max Pumps', 1, 1)}
          {up('delivery', 'Delivery (m³/h)', 0, 10)}
        </div>

        {/* Pump config tabs — only show if pump station */}
        {station.is_pump && (
          <div className="border rounded-lg overflow-hidden">
            <div className="flex border-b bg-gray-50">
              {(['pump', 'peaks', 'branch'] as const).map(t => (
                <button key={t}
                  onClick={() => setTab(t)}
                  className={`flex-1 py-1.5 text-xs font-medium transition-colors ${
                    tab === t ? 'bg-white border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'
                  }`}>
                  {t === 'pump' ? 'Pumps' : t === 'peaks' ? 'Peaks' : 'Branches'}
                </button>
              ))}
            </div>

            {tab === 'pump' && (
              <div>
                <div className="border-b">
                  <div className="flex">
                    <div className="flex-1 border-r">
                      <div className="text-xs font-semibold text-gray-600 px-2 py-1 bg-blue-50 border-b">Type A</div>
                      <PumpTypeTab pump={station.pumpA} typeLabel="A"
                        onChange={patch => updatePump(station.id, 'A', patch)} />
                    </div>
                    <div className="flex-1">
                      <div className="text-xs font-semibold text-gray-600 px-2 py-1 bg-indigo-50 border-b">Type B</div>
                      <PumpTypeTab pump={station.pumpB} typeLabel="B"
                        onChange={patch => updatePump(station.id, 'B', patch)} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === 'peaks' && (
              <div className="p-2">
                <p className="text-xs text-gray-500 mb-1">Intermediate elevation peaks along the downstream pipe segment</p>
                <EditableTable<PeakPoint>
                  rows={station.peaks_table}
                  columns={[
                    {key: 'location_km', label: 'Location (km from station)', min: 0, step: 1},
                    {key: 'elevation_m', label: 'Elevation (m)', step: 1},
                  ]}
                  onChange={rows => updateStation(station.id, {peaks_table: rows})}
                  minRows={0}
                  defaultRow={{location_km: 0, elevation_m: 0}}
                  addLabel="+ Add Peak"
                />
              </div>
            )}

            {tab === 'branch' && (
              <div className="p-2">
                <BranchBuilder stationId={station.id} branches={station.branches} />
              </div>
            )}
          </div>
        )}

        {!station.is_pump && (
          <div className="text-xs text-gray-400 italic">Pass-through station — no pump configuration needed</div>
        )}
      </div>
    </details>
  );
}
