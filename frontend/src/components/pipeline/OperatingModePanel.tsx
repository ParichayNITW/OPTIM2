import React from 'react';
import { usePipelineStore, LinefillRow, DayPlanRow, ProjFlowRow, ProjPlanRow } from '../../store/pipelineStore';
import { EditableTable } from './EditableTable';

const LINEFILL_COLS = [
  {key: 'product' as const, label: 'Product', type: 'text' as const},
  {key: 'volume' as const, label: 'Volume (m³)', min: 0, step: 100},
  {key: 'visc' as const, label: 'Viscosity (cSt)', min: 0, step: 1},
  {key: 'density' as const, label: 'Density (kg/m³)', min: 0, step: 1},
  {key: 'init_dra' as const, label: 'Initial DRA ppm', min: 0, step: 10},
];
const DEFAULT_LF_ROW: LinefillRow = {product: 'Crude', volume: 5000, visc: 10, density: 850, init_dra: 0};

export function OperatingModePanel() {
  const {global: g, setGlobal, linefill_table, setLinefill, day_plan, setDayPlan, proj_flow, setProjFlow, proj_plan, setProjPlan} = usePipelineStore();

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Operating Mode</h3>
        <div className="flex gap-2">
          {(['flow_rate', 'daily_schedule', 'pumping_planner'] as const).map(mode => (
            <label key={mode}
              className={`flex-1 text-center py-2 text-xs rounded-lg cursor-pointer border transition-colors ${
                g.op_mode === mode
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'border-gray-300 text-gray-600 hover:border-blue-400'
              }`}>
              <input type="radio" name="op_mode" className="sr-only"
                checked={g.op_mode === mode} onChange={() => setGlobal({op_mode: mode})} />
              {mode === 'flow_rate' ? 'Flow Rate' : mode === 'daily_schedule' ? 'Daily Schedule' : 'Pumping Planner'}
            </label>
          ))}
        </div>
      </div>

      {/* Linefill section — always shown */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">
          Linefill at Start (07:00)
        </h4>
        <EditableTable<LinefillRow>
          rows={linefill_table}
          columns={LINEFILL_COLS}
          onChange={setLinefill}
          minRows={1}
          defaultRow={DEFAULT_LF_ROW}
          addLabel="+ Add Product Batch"
        />
      </div>

      {/* Daily schedule mode */}
      {g.op_mode === 'daily_schedule' && (
        <div className="space-y-2">
          <div>
            <h4 className="text-xs font-semibold text-gray-600 mb-1">Daily Pumping Plan</h4>
            <EditableTable<DayPlanRow>
              rows={day_plan}
              columns={LINEFILL_COLS}
              onChange={setDayPlan}
              minRows={1}
              defaultRow={{product: 'Crude', volume: 10000, visc: 10, density: 850, init_dra: 0}}
              addLabel="+ Add Product"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-gray-600 block mb-1">Flow Mode</label>
            <div className="flex gap-3">
              {(['fixed', 'variable'] as const).map(m => (
                <label key={m} className="flex items-center gap-1 text-xs cursor-pointer">
                  <input type="radio" name="flow_mode" checked={g.flow_mode === m}
                    onChange={() => setGlobal({flow_mode: m})} />
                  {m === 'fixed' ? 'Fixed Flow Rate' : 'Variable Flow (optimizer decides)'}
                </label>
              ))}
            </div>
            {g.flow_mode === 'fixed' && (
              <div className="flex items-center gap-2 mt-1">
                <label className="text-xs text-gray-600 w-32">Hourly Flow (m³/h)</label>
                <input type="number" value={g.hourly_flow} min={0}
                  onChange={e => setGlobal({hourly_flow: parseFloat(e.target.value) || 0})}
                  className="w-24 px-2 py-1 text-xs border border-gray-300 rounded" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Pumping planner mode */}
      {g.op_mode === 'pumping_planner' && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-600 w-32">Planning Days</label>
            <input type="number" value={g.planner_days} min={1} max={90}
              onChange={e => setGlobal({planner_days: parseInt(e.target.value) || 7})}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded" />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-gray-600 mb-1">Projected Flow Schedule</h4>
            <EditableTable<ProjFlowRow>
              rows={proj_flow}
              columns={[
                {key: 'start', label: 'Start', type: 'text'},
                {key: 'end', label: 'End', type: 'text'},
                {key: 'flow', label: 'Flow (m³/h)', min: 0},
              ]}
              onChange={setProjFlow}
              minRows={1}
              defaultRow={{start: '2024-01-01 00:00', end: '2024-01-07 00:00', flow: 450}}
            />
          </div>
          <div>
            <h4 className="text-xs font-semibold text-gray-600 mb-1">Projected Pumping Plan</h4>
            <EditableTable<ProjPlanRow>
              rows={proj_plan}
              columns={[
                {key: 'product', label: 'Product', type: 'text'},
                {key: 'volume', label: 'Volume (m³)', min: 0},
                {key: 'visc', label: 'Viscosity (cSt)', min: 0},
                {key: 'density', label: 'Density (kg/m³)', min: 0},
              ]}
              onChange={setProjPlan}
              minRows={1}
              defaultRow={{product: 'Crude', volume: 70000, visc: 10, density: 850}}
            />
          </div>
        </div>
      )}
    </div>
  );
}
