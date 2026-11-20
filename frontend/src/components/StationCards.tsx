import type { MainlineResult } from '../types'
import { extractStationPoints } from './HGLChart'

interface StationCardsProps {
  result: MainlineResult
}

const fmtRs = (v: number) =>
  new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v)

function PressureBar({ value, max }: { value: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  const color = pct > 90 ? 'bg-red-500' : pct > 75 ? 'bg-yellow-500' : 'bg-green-500'
  return (
    <div className="w-full h-1.5 bg-gray-200 rounded-full mt-1">
      <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function StationCards({ result }: StationCardsProps) {
  const points = extractStationPoints(result)

  if (points.length === 0) {
    return <div className="text-gray-400 text-sm">No station data</div>
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Station Operating Points</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {points.map(p => (
          <div key={p.name} className="bg-white rounded-lg border border-gray-200 p-3 shadow-sm">
            <div className="text-xs font-semibold text-gray-800 truncate mb-2">{p.name}</div>

            {p.num_pumps > 0 ? (
              <>
                <div className="flex justify-between text-xs text-gray-600">
                  <span>Pumps</span>
                  <span className="font-medium text-gray-900">{p.num_pumps}</span>
                </div>
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Speed</span>
                  <span className="font-medium text-gray-900">{p.speed.toFixed(0)} RPM</span>
                </div>
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Efficiency</span>
                  <span className="font-medium text-gray-900">{p.efficiency.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Motor kW</span>
                  <span className="font-medium text-gray-900">{p.motor_kw.toFixed(0)}</span>
                </div>
                <div className="mt-2 pt-2 border-t border-gray-100">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Power</span>
                    <span className="text-blue-700 font-medium">{fmtRs(p.power_cost)}</span>
                  </div>
                  {p.dra_ppm > 0 && (
                    <div className="flex justify-between text-xs mt-0.5">
                      <span className="text-gray-500">DRA {p.dra_ppm.toFixed(0)} ppm</span>
                      <span className="text-amber-700 font-medium">{fmtRs(p.dra_cost)}</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-xs text-gray-400 italic">Delivery / terminal</div>
            )}

            {p.maop > 0 && (
              <div className="mt-2">
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Discharge</span>
                  <span>{p.discharge_head.toFixed(0)} / {p.maop.toFixed(0)} m</span>
                </div>
                <PressureBar value={p.discharge_head} max={p.maop} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
