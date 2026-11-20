import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { MainlineResult } from '../types'

interface DRAProfileChartProps {
  result: MainlineResult
}

export function DRAProfileChart({ result }: DRAProfileChartProps) {
  const segments = result.dra_segments ?? []
  const frontKm = result.dra_front_km ?? 0

  if (segments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-32 text-gray-400 text-sm gap-1">
        <span>No DRA injection in this run</span>
      </div>
    )
  }

  let cumKm = 0
  const chartData = segments.map(seg => {
    cumKm += seg.length_km
    return { km: +cumKm.toFixed(2), 'DRA (ppm)': +seg.dra_ppm.toFixed(1) }
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-700">DRA Concentration Profile</h3>
        {frontKm > 0 && (
          <span className="text-xs text-gray-500">Treated length: {frontKm.toFixed(1)} km</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="km" label={{ value: 'km from origin', position: 'insideBottom', offset: -10 }} tick={{ fontSize: 11 }} />
          <YAxis label={{ value: 'ppm', angle: -90, position: 'insideLeft' }} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v: number) => [`${v} ppm`, 'DRA']} labelFormatter={l => `${l} km`} />
          <Area type="stepAfter" dataKey="DRA (ppm)" stroke="#f59e0b" fill="#fef3c7" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
