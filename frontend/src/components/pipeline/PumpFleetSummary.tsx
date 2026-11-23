import React from 'react';
import { usePipelineStore, PumpConfig } from '../../store/pipelineStore';

function pumpStats(pump: PumpConfig) {
  const shutoffHead = pump.qh_table.length > 0 ? Math.max(...pump.qh_table.map(r => r.head)) : 0;
  const bep = pump.eff_table.length > 0
    ? pump.eff_table.reduce((best, r) => r.eff > best.eff ? r : best, pump.eff_table[0])
    : null;
  return { shutoffHead, bepFlow: bep?.flow ?? 0, bepEff: bep?.eff ?? 0 };
}

export function PumpFleetSummary() {
  const { stations } = usePipelineStore();
  const pumpStations = stations.filter(s => s.is_pump);

  if (pumpStations.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-400 italic">
        No pump stations configured — enable "Is Pumping Station" in the Pipeline tab.
      </div>
    );
  }

  const rows = pumpStations.flatMap(stn =>
    (['A', 'B'] as const)
      .map(t => ({ stn, t, pump: t === 'A' ? stn.pumpA : stn.pumpB }))
      .filter(({ pump }) => pump.enabled)
  );

  if (rows.length === 0) {
    return (
      <div className="p-3 space-y-2">
        <h3 className="text-sm font-semibold text-gray-700">Pump Overview</h3>
        <p className="text-xs text-gray-400 italic">
          {pumpStations.length} pump station(s) found but no Type A or Type B pumps enabled.
          Configure pump types in the Pipeline tab.
        </p>
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Pump Overview</h3>
        <span className="text-xs text-gray-400">{rows.length} pump unit(s) across {pumpStations.length} station(s)</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-gray-100 text-gray-600">
              {['Station', 'Type', 'Avail.', 'Name', 'Min / DOL RPM', 'Shutoff Head (m)', 'BEP Flow (m³/h)', 'BEP Eff (%)', 'Power', 'Tariff'].map(h => (
                <th key={h} className="px-2 py-1.5 text-left font-semibold border-b whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ stn, t, pump }, i) => {
              const { shutoffHead, bepFlow, bepEff } = pumpStats(pump);
              return (
                <tr key={`${stn.id}-${t}`} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-2 py-1.5 border-b font-medium text-gray-800">{stn.name}</td>
                  <td className="px-2 py-1.5 border-b">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                      t === 'A' ? 'bg-blue-100 text-blue-700' : 'bg-indigo-100 text-indigo-700'
                    }`}>Type {t}</span>
                  </td>
                  <td className="px-2 py-1.5 border-b text-center">{pump.available}</td>
                  <td className="px-2 py-1.5 border-b text-gray-600">{pump.name || '—'}</td>
                  <td className="px-2 py-1.5 border-b">{pump.min_rpm} / {pump.dol_rpm}</td>
                  <td className="px-2 py-1.5 border-b font-medium">{shutoffHead > 0 ? shutoffHead.toFixed(0) : '—'}</td>
                  <td className="px-2 py-1.5 border-b">{bepFlow > 0 ? bepFlow.toFixed(0) : '—'}</td>
                  <td className="px-2 py-1.5 border-b">
                    {bepEff > 0 ? (
                      <span className={`font-medium ${bepEff >= 75 ? 'text-green-600' : bepEff >= 65 ? 'text-amber-600' : 'text-red-500'}`}>
                        {bepEff.toFixed(1)}
                      </span>
                    ) : '—'}
                  </td>
                  <td className="px-2 py-1.5 border-b">
                    <span className={`px-1.5 py-0.5 rounded text-xs ${
                      pump.power_source === 'Diesel' ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700'
                    }`}>{pump.power_source}</span>
                  </td>
                  <td className="px-2 py-1.5 border-b text-gray-600">
                    {pump.power_source === 'Grid'
                      ? pump.tariff_type === 'Fixed'
                        ? `₹${pump.fixed_rate}/kWh`
                        : `${pump.time_slots.length} slot(s)`
                      : `SFC ${pump.sfc} g/kWh`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Quick stats strip */}
      <div className="grid grid-cols-3 gap-2 mt-2">
        {[
          { label: 'Total Units', value: rows.length },
          { label: 'Grid Powered', value: rows.filter(r => r.pump.power_source === 'Grid').length },
          { label: 'Diesel Powered', value: rows.filter(r => r.pump.power_source === 'Diesel').length },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white border rounded p-2 text-center">
            <div className="text-lg font-bold text-gray-800">{value}</div>
            <div className="text-xs text-gray-500">{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
