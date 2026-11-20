import React from 'react';
import { usePipelineStore } from '../../store/pipelineStore';
import { StationAccordion } from './StationAccordion';

export function StationBuilder() {
  const {stations, addStation, terminal, setTerminal} = usePipelineStore();

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Pipeline Stations</h3>
        <button
          onClick={addStation}
          className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
        >
          + Add Station
        </button>
      </div>

      {stations.length === 0 && (
        <div className="text-center py-8 text-gray-400 text-sm border-2 border-dashed rounded-lg">
          No stations — click &ldquo;Add Station&rdquo; to begin
        </div>
      )}

      <div>
        {stations.map((station, i) => (
          <StationAccordion key={station.id} station={station} index={i} total={stations.length} />
        ))}
      </div>

      {/* Terminal Station */}
      <div className="mt-4 border border-red-200 rounded-lg p-3 bg-red-50">
        <h4 className="text-sm font-semibold text-red-700 mb-2">Terminal Station</h4>
        <div className="grid grid-cols-3 gap-2">
          <div>
            <label className="text-xs text-gray-600 block">Name</label>
            <input type="text" value={terminal.name}
              onChange={e => setTerminal({name: e.target.value})}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
          <div>
            <label className="text-xs text-gray-600 block">Elevation (m)</label>
            <input type="number" value={terminal.elevation}
              onChange={e => setTerminal({elevation: parseFloat(e.target.value) || 0})}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
          <div>
            <label className="text-xs text-gray-600 block">Min Residual Head (m)</label>
            <input type="number" value={terminal.min_residual} min={0}
              onChange={e => setTerminal({min_residual: parseFloat(e.target.value) || 0})}
              className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
        </div>
      </div>
    </div>
  );
}
