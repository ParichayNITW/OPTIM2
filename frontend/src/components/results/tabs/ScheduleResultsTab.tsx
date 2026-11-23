import React, { useState } from 'react';
import { ParsedScheduleResult } from '../../../utils/resultParser';
import type { StationData } from '../../../store/pipelineStore';

interface Props {
  schedule: ParsedScheduleResult;
  stations: StationData[];
  onHourSelect?: (hour: number) => void;
}

export function ScheduleResultsTab({ schedule, stations, onHourSelect }: Props) {
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const pumpStations = stations.filter(s => s.is_pump);

  const selectRow = (step: number) => {
    setSelectedStep(step);
    if (onHourSelect) onHourSelect(schedule.hourly[step]?.hour || 0);
  };

  const fmt = (v: number, d = 0) => v === 0 ? '—' : v.toLocaleString('en-IN', {maximumFractionDigits: d});
  const fmtCost = (v: number) => v === 0 ? '—' : `₹${v.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;

  const totalPow = schedule.hourly.reduce((a, h) => a + h.power_cost, 0);
  const totalDra = schedule.hourly.reduce((a, h) => a + h.dra_cost, 0);

  return (
    <div className="space-y-3">
      {/* Summary row */}
      <div className="grid grid-cols-4 gap-3">
        {[
          {label: 'Total Cost', val: `₹${(schedule.total_cost).toLocaleString('en-IN', {maximumFractionDigits:0})}`},
          {label: 'Total Throughput', val: `${fmt(schedule.total_throughput, 0)} m³`},
          {label: 'Avg Flow', val: `${fmt(schedule.avg_flow, 1)} m³/h`},
          {label: 'Avg Efficiency', val: `${fmt(schedule.avg_efficiency, 1)}%`},
        ].map(({label, val}) => (
          <div key={label} className="bg-blue-50 rounded-lg p-3 border border-blue-100">
            <div className="text-xs text-blue-600 font-medium">{label}</div>
            <div className="text-base font-bold text-blue-900 mt-0.5">{val}</div>
          </div>
        ))}
      </div>

      {/* Cost breakdown */}
      <div className="flex gap-4 text-xs text-gray-600 bg-gray-50 rounded px-3 py-2">
        <span>Power: <strong>{fmtCost(totalPow)}</strong></span>
        <span>DRA: <strong>{fmtCost(totalDra)}</strong></span>
        {schedule.hourly_errors.length > 0 && (
          <span className="text-orange-600">⚠ {schedule.hourly_errors.length} infeasible hour(s)</span>
        )}
        {schedule.backtrack_notes.length > 0 && (
          <span className="text-blue-600">↩ {schedule.backtrack_notes.length} retry(s)</span>
        )}
      </div>

      {/* Hourly table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-xs">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-2 py-2 text-left font-semibold text-gray-700 sticky left-0 bg-gray-50">Time</th>
              <th className="px-2 py-2 text-right font-semibold text-gray-700">Flow m³/h</th>
              <th className="px-2 py-2 text-right font-semibold text-gray-700">Power ₹</th>
              <th className="px-2 py-2 text-right font-semibold text-gray-700">DRA ₹</th>
              {pumpStations.map(s => (
                <React.Fragment key={s.id}>
                  <th className="px-2 py-2 text-center font-semibold text-gray-700 border-l border-gray-200" colSpan={4}>
                    {s.name}
                  </th>
                </React.Fragment>
              ))}
            </tr>
            <tr className="border-t border-gray-100">
              <th className="px-2 py-1 sticky left-0 bg-gray-50"></th>
              <th className="px-2 py-1"></th>
              <th className="px-2 py-1"></th>
              <th className="px-2 py-1"></th>
              {pumpStations.map(s => (
                <React.Fragment key={s.id}>
                  <th className="px-1 py-1 text-center text-gray-500 font-normal border-l border-gray-200">Pumps</th>
                  <th className="px-1 py-1 text-center text-gray-500 font-normal">Eff%</th>
                  <th className="px-1 py-1 text-center text-gray-500 font-normal">DRA ppm</th>
                  <th className="px-1 py-1 text-center text-gray-500 font-normal">SDH m</th>
                </React.Fragment>
              ))}
            </tr>
          </thead>
          <tbody>
            {schedule.hourly.map((row, idx) => {
              const isSelected = selectedStep === idx;
              const isErr = row.error;
              return (
                <tr
                  key={idx}
                  onClick={() => selectRow(idx)}
                  className={`border-t border-gray-100 cursor-pointer transition-colors ${
                    isSelected ? 'bg-blue-100' : isErr ? 'bg-red-50' : idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'
                  } hover:bg-blue-50`}
                >
                  <td className={`px-2 py-1.5 font-mono font-semibold sticky left-0 ${isSelected ? 'bg-blue-100' : isErr ? 'bg-red-50' : idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                    {row.hour_label}
                  </td>
                  <td className="px-2 py-1.5 text-right">{fmt(row.flow, 0)}</td>
                  <td className="px-2 py-1.5 text-right text-orange-700">{fmtCost(row.power_cost)}</td>
                  <td className="px-2 py-1.5 text-right text-teal-700">{fmtCost(row.dra_cost)}</td>
                  {pumpStations.map(s => {
                    const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
                    const sm = row.stations[k];
                    return (
                      <React.Fragment key={s.id}>
                        <td className="px-1 py-1.5 text-center border-l border-gray-100">{sm?.num_pumps || '—'}</td>
                        <td className="px-1 py-1.5 text-center">{sm?.efficiency ? `${sm.efficiency.toFixed(1)}` : '—'}</td>
                        <td className="px-1 py-1.5 text-center">{sm?.dra_ppm ? `${sm.dra_ppm.toFixed(0)}` : '—'}</td>
                        <td className="px-1 py-1.5 text-center">{sm?.sdh ? `${sm.sdh.toFixed(0)}` : '—'}</td>
                      </React.Fragment>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {selectedStep !== null && (
        <p className="text-xs text-gray-500">↑ Click a row to show its HGL in the HGL tab</p>
      )}
    </div>
  );
}
