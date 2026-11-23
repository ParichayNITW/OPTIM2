import React from 'react';
import { ParsedScheduleResult, IntervalRow } from '../../../utils/resultParser';

interface Props {
  schedule: ParsedScheduleResult;
}

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const header = cols.join(',');
  const body = rows.map(r => cols.map(c => String(r[c] ?? '')).join(',')).join('\n');
  const blob = new Blob([header + '\n' + body], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export function PumpingPlannerTab({ schedule }: Props) {
  const rows: IntervalRow[] = schedule.interval_results;
  const totalVol  = rows.reduce((a, r) => a + r.volume_m3, 0);
  const totalCost = rows.reduce((a, r) => a + r.cost, 0);
  const totalHrs  = rows.reduce((a, r) => a + r.duration_hours, 0);

  const fmtCost = (v: number) => v === 0 ? '—' : `₹${v.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
  const fmtVol  = (v: number) => v === 0 ? '—' : v.toLocaleString('en-IN', {maximumFractionDigits: 0});

  return (
    <div className="space-y-3">
      {/* Summary tiles */}
      <div className="grid grid-cols-3 gap-3">
        {[
          {label: 'Planning Horizon', val: `${rows.length} intervals / ${totalHrs} h`},
          {label: 'Total Volume', val: `${fmtVol(totalVol)} m³`},
          {label: 'Total Cost', val: fmtCost(totalCost)},
        ].map(({label, val}) => (
          <div key={label} className="bg-indigo-50 rounded-lg p-3 border border-indigo-100">
            <div className="text-xs text-indigo-600 font-medium">{label}</div>
            <div className="text-base font-bold text-indigo-900 mt-0.5">{val}</div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text-gray-700">Multi-Day Interval Schedule</h4>
        <button
          onClick={() => downloadCsv('pumping_planner.csv', rows as unknown as Record<string, unknown>[])}
          className="px-2 py-1 text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded hover:bg-indigo-100"
        >
          ⬇ CSV
        </button>
      </div>

      {rows.length === 0 ? (
        <p className="text-xs text-gray-400">No pumping planner data available.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-gray-200">
          <table className="w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left font-semibold text-gray-700">#</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-700">Start</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-700">End</th>
                <th className="px-3 py-2 text-right font-semibold text-gray-700">Dur (h)</th>
                <th className="px-3 py-2 text-right font-semibold text-gray-700">Flow (m³/h)</th>
                <th className="px-3 py-2 text-right font-semibold text-gray-700">Volume (m³)</th>
                <th className="px-3 py-2 text-right font-semibold text-gray-700">Cost (₹)</th>
                <th className="px-3 py-2 text-right font-semibold text-gray-700">₹/m³</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className={`border-t border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                  <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                  <td className="px-3 py-1.5 font-mono">{row.start}</td>
                  <td className="px-3 py-1.5 font-mono">{row.end}</td>
                  <td className="px-3 py-1.5 text-right">{row.duration_hours}</td>
                  <td className="px-3 py-1.5 text-right">{row.flow_m3h.toFixed(0)}</td>
                  <td className="px-3 py-1.5 text-right">{fmtVol(row.volume_m3)}</td>
                  <td className="px-3 py-1.5 text-right text-orange-700">{fmtCost(row.cost)}</td>
                  <td className="px-3 py-1.5 text-right text-gray-600">
                    {row.volume_m3 > 0 ? (row.cost / row.volume_m3).toFixed(2) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot className="bg-gray-100 font-semibold">
              <tr className="border-t-2 border-gray-300">
                <td className="px-3 py-2" colSpan={3}>Total / Average</td>
                <td className="px-3 py-2 text-right">{totalHrs}</td>
                <td className="px-3 py-2 text-right">
                  {totalHrs > 0 ? (totalVol / totalHrs).toFixed(0) : '—'}
                </td>
                <td className="px-3 py-2 text-right">{fmtVol(totalVol)}</td>
                <td className="px-3 py-2 text-right text-orange-700">{fmtCost(totalCost)}</td>
                <td className="px-3 py-2 text-right">
                  {totalVol > 0 ? (totalCost / totalVol).toFixed(2) : '—'}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      )}
    </div>
  );
}
