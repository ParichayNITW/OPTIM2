import React from 'react';
import { usePipelineStore } from '../../store/pipelineStore';

export function AdvancedSearchPanel() {
  const {global: g, setGlobal} = usePipelineStore();

  return (
    <details className="border rounded-lg">
      <summary className="px-3 py-2 cursor-pointer text-sm font-semibold text-gray-700 bg-gray-50 rounded-lg">
        Advanced Search Depth
      </summary>
      <div className="p-3 space-y-2">
        <div className="grid grid-cols-2 gap-2">
          {[
            {key: 'rpm_step', label: 'RPM Step (rpm)', min: 10, step: 10},
            {key: 'dra_step', label: 'DRA Step (%DR)', min: 1, step: 1},
          ].map(f => (
            <div key={f.key}>
              <label className="text-xs text-gray-600 block">{f.label}</label>
              <input type="number"
                value={(g as never)[f.key as keyof typeof g] as number}
                min={f.min} step={f.step}
                onChange={e => setGlobal({[f.key]: parseFloat(e.target.value) || f.min} as never)}
                className="w-full px-2 py-1 text-xs border border-gray-300 rounded focus:outline-none focus:border-blue-400" />
            </div>
          ))}
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={g.priority_feasibility}
            onChange={e => setGlobal({priority_feasibility: e.target.checked})} />
          <span className="text-xs text-gray-600">Priority: Feasibility over cost</span>
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={g.collect_state_audit}
            onChange={e => setGlobal({collect_state_audit: e.target.checked})} />
          <span className="text-xs text-gray-600">Capture candidate log (shows DP state tree)</span>
        </label>
      </div>
    </details>
  );
}
