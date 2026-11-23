import React from 'react';
import { ParsedScheduleResult } from '../../../utils/resultParser';

interface Props {
  schedule: ParsedScheduleResult;
}

export function ScheduleAuditTab({ schedule }: Props) {
  const hasIssues = schedule.hourly_errors.length > 0
    || schedule.backtrack_notes.length > 0
    || schedule.enforced_actions.length > 0;

  return (
    <div className="space-y-4">
      {!hasIssues && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm text-green-700">
          ✓ All 24 hours solved successfully with no backtracks or enforced actions.
        </div>
      )}

      {/* Hourly errors */}
      {schedule.hourly_errors.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-red-700 mb-1">
            Infeasible Hours ({schedule.hourly_errors.length})
          </h4>
          <div className="space-y-1">
            {schedule.hourly_errors.map((err, i) => (
              <div key={i} className="bg-red-50 border border-red-100 rounded px-3 py-1.5 text-xs text-red-800">
                {err}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Backtrack notes */}
      {schedule.backtrack_notes.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-orange-700 mb-1">
            Backtrack Events ({schedule.backtrack_notes.length})
          </h4>
          <div className="space-y-1">
            {schedule.backtrack_notes.map((note, i) => (
              <div key={i} className="bg-orange-50 border border-orange-100 rounded px-3 py-1.5 text-xs text-orange-800">
                {note}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Enforced actions */}
      {schedule.enforced_actions.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-blue-700 mb-1">
            Enforced Actions ({schedule.enforced_actions.length})
          </h4>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">Hour</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">Action</th>
                </tr>
              </thead>
              <tbody>
                {schedule.enforced_actions.map((ea, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}>
                    <td className="px-3 py-1.5 font-mono text-blue-700">{ea.hour}</td>
                    <td className="px-3 py-1.5 text-gray-700">{ea.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-hour status summary */}
      <div>
        <h4 className="text-xs font-semibold text-gray-700 mb-2">Hour-by-Hour Status</h4>
        <div className="flex flex-wrap gap-1.5">
          {schedule.hourly.map(h => (
            <div
              key={h.step}
              title={h.error ? h.error_msg : `${h.hour_label}: ₹${(h.total_cost).toFixed(0)}`}
              className={`w-9 h-6 rounded flex items-center justify-center text-xs font-mono cursor-default ${
                h.error ? 'bg-red-200 text-red-800' : 'bg-green-100 text-green-800'
              }`}
            >
              {h.hour_label.slice(0, 2)}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400 mt-1">Green = feasible, Red = infeasible. Hover for cost/error.</p>
      </div>
    </div>
  );
}
