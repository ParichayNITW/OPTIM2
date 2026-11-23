import React, { useState } from 'react';

interface Props {
  jobId?: string;
  opMode?: string;
}

interface ExportBtn {
  label: string;
  endpoint: string;
  filename: string;
  color: string;
  icon: string;
  modes: string[];  // which op_modes show this button
}

const EXPORTS: ExportBtn[] = [
  {
    label: 'Schedule CSV',
    endpoint: 'schedule-csv',
    filename: 'schedule.csv',
    color: 'green',
    icon: '📊',
    modes: ['daily_schedule', 'pumping_planner'],
  },
  {
    label: 'Linefill Snapshots CSV',
    endpoint: 'linefill-csv',
    filename: 'linefill_snapshots.csv',
    color: 'green',
    icon: '🔄',
    modes: ['daily_schedule'],
  },
  {
    label: 'Next-Day Linefill CSV',
    endpoint: 'nextday-linefill-csv',
    filename: 'nextday_linefill.csv',
    color: 'green',
    icon: '📋',
    modes: ['daily_schedule', 'pumping_planner'],
  },
  {
    label: 'Segment Batches CSV',
    endpoint: 'segments-csv',
    filename: 'segment_batches.csv',
    color: 'green',
    icon: '🗂',
    modes: ['daily_schedule', 'pumping_planner'],
  },
  {
    label: 'Full Results CSV',
    endpoint: 'result-csv',
    filename: 'optimization_result.csv',
    color: 'blue',
    icon: '📄',
    modes: ['flow_rate', 'daily_schedule', 'pumping_planner'],
  },
  {
    label: 'Excel Report',
    endpoint: 'excel',
    filename: 'pipeline_report.xlsx',
    color: 'emerald',
    icon: '📗',
    modes: ['flow_rate', 'daily_schedule', 'pumping_planner'],
  },
  {
    label: 'PDF Report',
    endpoint: 'pdf',
    filename: 'pipeline_report.pdf',
    color: 'red',
    icon: '📕',
    modes: ['flow_rate', 'daily_schedule', 'pumping_planner'],
  },
];

const COLOR_CLASSES: Record<string, string> = {
  green:   'bg-green-50 text-green-700 border-green-200 hover:bg-green-100',
  blue:    'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100',
  emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100',
  red:     'bg-red-50 text-red-700 border-red-200 hover:bg-red-100',
};

export function ReportsTab({ jobId, opMode = 'flow_rate' }: Props) {
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!jobId) {
    return (
      <div className="p-6 text-center text-sm text-gray-400">
        Run an optimization first to enable report downloads.
      </div>
    );
  }

  const handleDownload = async (btn: ExportBtn) => {
    setLoading(btn.endpoint);
    setError(null);
    try {
      const res = await fetch(`/api/results/${jobId}/export/${btn.endpoint}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text.slice(0, 200)}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = btn.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(null);
    }
  };

  const visible = EXPORTS.filter(b => b.modes.includes(opMode));

  return (
    <div className="p-4 space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-1">Download Reports</h3>
        <p className="text-xs text-gray-400 mb-3">Job ID: <span className="font-mono">{jobId}</span></p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {visible.map(btn => (
            <button
              key={btn.endpoint}
              onClick={() => handleDownload(btn)}
              disabled={loading === btn.endpoint}
              className={`flex items-center gap-2 px-3 py-2.5 text-xs font-medium border rounded-lg transition-colors disabled:opacity-50 disabled:cursor-wait ${COLOR_CLASSES[btn.color]}`}
            >
              <span>{btn.icon}</span>
              <span>{loading === btn.endpoint ? 'Generating…' : btn.label}</span>
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-xs text-red-700">
          <span className="font-semibold">Export error: </span>{error}
        </div>
      )}

      <div className="border-t pt-3">
        <p className="text-xs text-gray-400">
          CSV files open in Excel or any spreadsheet app. Excel report contains multiple worksheets.
          PDF report includes charts and a full 24-hour schedule table.
        </p>
      </div>
    </div>
  );
}
