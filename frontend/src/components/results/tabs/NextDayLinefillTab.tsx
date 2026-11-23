import React from 'react';
import { ParsedScheduleResult, LinefillSnap, FinalSegmentSlice } from '../../../utils/resultParser';

interface Props {
  schedule: ParsedScheduleResult;
}

function downloadCsv(filename: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const header = cols.join(',');
  const body = rows.map(r => cols.map(c => {
    const v = r[c];
    return typeof v === 'string' && v.includes(',') ? `"${v}"` : String(v ?? '');
  }).join(',')).join('\n');
  const blob = new Blob([header + '\n' + body], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export function NextDayLinefillTab({ schedule }: Props) {
  const finalVol: LinefillSnap[] = schedule.final_vol;
  const slices: FinalSegmentSlice[] = schedule.final_segment_slices;

  const totalVol = finalVol.reduce((a, r) => a + Number(r['Volume (m³)'] || 0), 0);

  return (
    <div className="space-y-4">
      {/* Final linefill queue */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <h4 className="text-xs font-semibold text-gray-700">
            Next-Day Starting Linefill — Total: {totalVol.toLocaleString('en-IN', {maximumFractionDigits: 0})} m³
          </h4>
          <button
            onClick={() => downloadCsv('next_day_linefill.csv', finalVol as unknown as Record<string, unknown>[])}
            className="px-2 py-1 text-xs bg-blue-50 text-blue-700 border border-blue-200 rounded hover:bg-blue-100 transition-colors"
          >
            ⬇ CSV
          </button>
        </div>
        {finalVol.length === 0 ? (
          <p className="text-xs text-gray-400">No linefill data available.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">#</th>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">Product</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Volume (m³)</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Viscosity (cSt)</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Density (kg/m³)</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Initial DRA (ppm)</th>
                </tr>
              </thead>
              <tbody>
                {finalVol.map((row, i) => (
                  <tr key={i} className={`border-t border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                    <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                    <td className="px-3 py-1.5 font-medium text-gray-800">{row.Product}</td>
                    <td className="px-3 py-1.5 text-right">{Number(row['Volume (m³)']).toLocaleString('en-IN', {maximumFractionDigits: 1})}</td>
                    <td className="px-3 py-1.5 text-right">{Number(row['Viscosity (cSt)']).toFixed(2)}</td>
                    <td className="px-3 py-1.5 text-right">{Number(row['Density (kg/m³)']).toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-right">{Number(row['Initial DRA (ppm)']).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Segment-wise batch composition */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <h4 className="text-xs font-semibold text-gray-700">Segment-wise Batch Composition</h4>
          <button
            onClick={() => {
              const rows: Record<string, unknown>[] = slices.flatMap(seg =>
                seg.slices.map((sl, bi) => ({
                  'Segment': seg.station_name,
                  'Batch #': bi + 1,
                  'Length (km)': sl.length_km.toFixed(3),
                  'Viscosity (cSt)': sl.kv.toFixed(2),
                  'Density (kg/m³)': sl.rho.toFixed(1),
                }))
              );
              downloadCsv('segment_batches.csv', rows);
            }}
            className="px-2 py-1 text-xs bg-blue-50 text-blue-700 border border-blue-200 rounded hover:bg-blue-100 transition-colors"
          >
            ⬇ CSV
          </button>
        </div>
        {slices.length === 0 ? (
          <p className="text-xs text-gray-400">No segment data available.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-gray-700">Segment</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Avg KV (cSt)</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Avg Density</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">Slices</th>
                  <th className="px-3 py-2 text-right font-semibold text-gray-700">KV Range</th>
                </tr>
              </thead>
              <tbody>
                {slices.map((seg, i) => {
                  const kvVals = seg.slices.map(s => s.kv);
                  const minKv = kvVals.length > 0 ? Math.min(...kvVals) : seg.avg_kv;
                  const maxKv = kvVals.length > 0 ? Math.max(...kvVals) : seg.avg_kv;
                  return (
                    <tr key={i} className={`border-t border-gray-100 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                      <td className="px-3 py-1.5 font-medium text-gray-800">{seg.station_name}</td>
                      <td className="px-3 py-1.5 text-right">{seg.avg_kv.toFixed(2)}</td>
                      <td className="px-3 py-1.5 text-right">{seg.avg_rho.toFixed(1)}</td>
                      <td className="px-3 py-1.5 text-right">{seg.slices.length}</td>
                      <td className="px-3 py-1.5 text-right">
                        {minKv === maxKv ? minKv.toFixed(2) : `${minKv.toFixed(2)}–${maxKv.toFixed(2)}`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
