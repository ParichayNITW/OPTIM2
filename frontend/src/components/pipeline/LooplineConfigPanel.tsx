import React from 'react';
import { usePipelineStore, PeakPoint } from '../../store/pipelineStore';
import { EditableTable } from './EditableTable';

export function LooplineConfigPanel() {
  const {loopline, setLoopline} = usePipelineStore();

  const field = (key: keyof typeof loopline, label: string, min?: number, step?: number) => (
    <div>
      <label className="text-xs text-gray-500 block">{label}</label>
      <input type="number" value={loopline[key] as number} min={min} step={step ?? 'any'}
        onChange={e => setLoopline({[key]: parseFloat(e.target.value) || 0} as never)}
        className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
    </div>
  );

  return (
    <div>
      <label className="flex items-center gap-2 cursor-pointer mb-3">
        <input type="checkbox" checked={loopline.enabled}
          onChange={e => setLoopline({enabled: e.target.checked})} />
        <span className="text-sm font-semibold text-gray-700">Enable Parallel Loopline</span>
      </label>

      {loopline.enabled && (
        <div className="space-y-3 pl-2 border-l-2 border-gray-200">
          <div>
            <label className="text-xs text-gray-500 block">Loopline Name</label>
            <input type="text" value={loopline.name}
              onChange={e => setLoopline({name: e.target.value})}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>

          <div className="grid grid-cols-3 gap-2">
            {field('start_km', 'Start KP (km)', 0, 1)}
            {field('end_km', 'End KP (km)', 0, 1)}
            {field('L_km', 'Length (km)', 0, 1)}
          </div>

          <div className="grid grid-cols-3 gap-2">
            {field('D_in', 'Diameter (in)', 0, 0.5)}
            {field('t_in', 'Wall Thickness (in)', 0, 0.025)}
            {field('SMYS_psi', 'SMYS (psi)', 0, 1000)}
          </div>

          <div className="grid grid-cols-3 gap-2">
            {field('roughness_m', 'Roughness (m)', 0, 0.000001)}
            {field('max_dra_pct', 'Max DRA%', 0, 1)}
            {field('elev', 'Elevation (m)')}
          </div>

          <div>
            <p className="text-xs text-gray-500 mb-1">Peak elevation profile along loopline</p>
            <EditableTable<PeakPoint>
              rows={loopline.peak_profile}
              columns={[
                {key: 'location_km', label: 'Location (km)', min: 0, step: 1},
                {key: 'elevation_m', label: 'Elevation (m)', step: 1},
              ]}
              onChange={rows => setLoopline({peak_profile: rows})}
              minRows={0}
              defaultRow={{location_km: 0, elevation_m: 0}}
              addLabel="+ Add Peak"
            />
          </div>
        </div>
      )}
    </div>
  );
}
