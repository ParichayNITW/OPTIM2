import React from 'react';
import { usePipelineStore, BranchConfig, BranchStationInput } from '../../store/pipelineStore';

interface Props {
  stationId: string;
  branches: BranchConfig[];
}

const defaultBranchStation = (): BranchStationInput => ({
  name: 'Branch Station', elev: 0, L: 20, OD: 12, wall: 0.25, SMYS: 52000,
  roughness: 0.000046, min_residual: 30, max_dra: 20, is_pump: false,
  pumpA: {enabled: false, available: 2, name: '', qh_table: [], eff_table: [],
    power_source: 'Grid', min_rpm: 1000, dol_rpm: 1480, tariff_type: 'Fixed',
    fixed_rate: 7.5, time_slots: [], sfc_mode: 'manual', sfc: 220, engine_make: '',
    engine_model: '', rated_power: 1000, sfc50: 220, sfc75: 210, sfc100: 215},
  pumpB: {enabled: false, available: 2, name: '', qh_table: [], eff_table: [],
    power_source: 'Grid', min_rpm: 1000, dol_rpm: 1480, tariff_type: 'Fixed',
    fixed_rate: 7.5, time_slots: [], sfc_mode: 'manual', sfc: 220, engine_make: '',
    engine_model: '', rated_power: 1000, sfc50: 220, sfc75: 210, sfc100: 215},
});

export function BranchBuilder({stationId, branches}: Props) {
  const {addBranch, removeBranch, updateBranch} = usePipelineStore();

  const updateBranchStation = (branchId: string, stIdx: number, patch: Partial<BranchStationInput>) => {
    const branch = branches.find(b => b.id === branchId);
    if (!branch) return;
    const newStations = branch.stations.map((s, i) => i === stIdx ? {...s, ...patch} : s);
    updateBranch(stationId, branchId, {stations: newStations});
  };

  const addBranchStation = (branchId: string) => {
    const branch = branches.find(b => b.id === branchId);
    if (!branch) return;
    updateBranch(stationId, branchId, {stations: [...branch.stations, defaultBranchStation()]});
  };

  const removeBranchStation = (branchId: string, stIdx: number) => {
    const branch = branches.find(b => b.id === branchId);
    if (!branch) return;
    updateBranch(stationId, branchId, {stations: branch.stations.filter((_, i) => i !== stIdx)});
  };

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-orange-700">Branch / Tap-off Lines</span>
        <button onClick={() => addBranch(stationId)}
          className="text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded hover:bg-orange-200 font-medium">
          + Add Branch
        </button>
      </div>

      {branches.map(branch => (
        <details key={branch.id} className="border border-orange-200 rounded mb-2">
          <summary className="px-2 py-1.5 cursor-pointer flex items-center justify-between bg-orange-50">
            <span className="text-xs font-medium text-orange-800">
              {branch.branch_name || 'Branch'} — {branch.flow_m3h} m³/h
            </span>
            <button onClick={e => {e.preventDefault(); removeBranch(stationId, branch.id);}}
              className="text-xs text-red-500 hover:text-red-700 ml-2">Remove</button>
          </summary>
          <div className="p-2 space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-500 block">Branch Name</label>
                <input type="text" value={branch.branch_name}
                  onChange={e => updateBranch(stationId, branch.id, {branch_name: e.target.value})}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
              <div>
                <label className="text-xs text-gray-500 block">Branch Flow (m³/h)</label>
                <input type="number" value={branch.flow_m3h} min={0}
                  onChange={e => updateBranch(stationId, branch.id, {flow_m3h: parseFloat(e.target.value) || 0})}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="text-xs text-gray-500">Terminal Name</label>
                <input type="text" value={branch.terminal_name}
                  onChange={e => updateBranch(stationId, branch.id, {terminal_name: e.target.value})}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
              <div>
                <label className="text-xs text-gray-500">Terminal Elev (m)</label>
                <input type="number" value={branch.terminal_elev}
                  onChange={e => updateBranch(stationId, branch.id, {terminal_elev: parseFloat(e.target.value) || 0})}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
              <div>
                <label className="text-xs text-gray-500">Min Pressure (m)</label>
                <input type="number" value={branch.terminal_min_pressure}
                  onChange={e => updateBranch(stationId, branch.id, {terminal_min_pressure: parseFloat(e.target.value) || 0})}
                  className="w-full px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-600">Branch Stations ({branch.stations.length})</span>
                <button onClick={() => addBranchStation(branch.id)}
                  className="text-xs text-blue-600 hover:text-blue-800">+ Add Station</button>
              </div>
              {branch.stations.map((bs, stIdx) => (
                <div key={stIdx} className="border rounded p-2 mb-1 bg-gray-50">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs font-medium">{bs.name}</span>
                    <button onClick={() => removeBranchStation(branch.id, stIdx)}
                      className="text-xs text-red-400 hover:text-red-600">✕</button>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {[
                      {key: 'name', label: 'Name', type: 'text'},
                      {key: 'elev', label: 'Elev (m)', type: 'number'},
                      {key: 'L', label: 'Length (km)', type: 'number'},
                      {key: 'OD', label: 'OD (in)', type: 'number'},
                      {key: 'wall', label: 'Wall (in)', type: 'number'},
                      {key: 'SMYS', label: 'SMYS (psi)', type: 'number'},
                      {key: 'roughness', label: 'Roughness (m)', type: 'number'},
                      {key: 'min_residual', label: 'Min Head (m)', type: 'number'},
                    ].map(f => (
                      <div key={f.key}>
                        <label className="text-xs text-gray-400">{f.label}</label>
                        <input type={f.type}
                          value={String((bs as never)[f.key as keyof BranchStationInput] ?? '')}
                          onChange={e => updateBranchStation(branch.id, stIdx, {
                            [f.key]: f.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value
                          } as Partial<BranchStationInput>)}
                          className="w-full px-1 py-0.5 text-xs border border-gray-200 rounded" />
                      </div>
                    ))}
                  </div>
                  <label className="flex items-center gap-1 text-xs mt-1 cursor-pointer">
                    <input type="checkbox" checked={bs.is_pump}
                      onChange={e => updateBranchStation(branch.id, stIdx, {is_pump: e.target.checked})} />
                    Pumping Station
                  </label>
                </div>
              ))}
            </div>
          </div>
        </details>
      ))}
    </div>
  );
}
