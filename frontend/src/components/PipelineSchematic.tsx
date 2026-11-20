import type { MainlineResult } from '../types'
import { extractStationPoints } from './HGLChart'

interface PipelineSchematicProps {
  result: MainlineResult
}

export function PipelineSchematic({ result }: PipelineSchematicProps) {
  const points = extractStationPoints(result)
  if (points.length === 0) return null

  const totalKm = points[points.length - 1]?.km ?? 1
  const W = 800
  const H = 100
  const pad = 40

  const xOf = (km: number) => pad + ((km / totalKm) * (W - 2 * pad))

  const pct = (p: { discharge_head: number; maop: number }) =>
    p.maop > 0 ? Math.min(1, p.discharge_head / p.maop) : 0

  const color = (p: { discharge_head: number; maop: number }) => {
    const r = pct(p)
    if (r > 0.9) return '#ef4444'
    if (r > 0.75) return '#f59e0b'
    if (r > 0.3) return '#22c55e'
    return '#94a3b8'
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Pipeline Schematic</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 100 }}>
        {/* Mainline pipe */}
        <line x1={pad} y1={H / 2} x2={W - pad} y2={H / 2} stroke="#94a3b8" strokeWidth={4} strokeLinecap="round" />

        {points.map((p, i) => {
          const x = xOf(p.km)
          const isFirst = i === 0
          const isLast = i === points.length - 1
          const fill = color(p)
          return (
            <g key={p.name}>
              {/* Station marker */}
              <rect
                x={x - 6} y={H / 2 - 12}
                width={12} height={24}
                rx={2}
                fill={isFirst || isLast ? '#1d4ed8' : fill}
                stroke="#fff"
                strokeWidth={1.5}
              />
              {/* Name label */}
              <text
                x={x} y={H / 2 - 18}
                textAnchor="middle"
                fontSize={8}
                fill="#374151"
              >
                {p.name.length > 8 ? p.name.slice(0, 8) + '…' : p.name}
              </text>
              {/* km label */}
              <text
                x={x} y={H / 2 + 28}
                textAnchor="middle"
                fontSize={7}
                fill="#6b7280"
              >
                {p.km.toFixed(0)} km
              </text>
            </g>
          )
        })}
      </svg>
      <div className="flex gap-4 mt-1 text-xs text-gray-500">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500 inline-block" /> &lt;75% MAOP</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-yellow-500 inline-block" /> 75–90% MAOP</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block" /> &gt;90% MAOP</span>
      </div>
    </div>
  )
}
