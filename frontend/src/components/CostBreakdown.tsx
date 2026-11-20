import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { MainlineResult } from '../types'
import { extractStationPoints } from './HGLChart'

interface CostBreakdownProps {
  result: MainlineResult
}

const fmtRs = (v: number) =>
  new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v)

export function CostBreakdown({ result }: CostBreakdownProps) {
  const points = extractStationPoints(result)
  const totalCost = typeof result.total_cost === 'number' ? result.total_cost : 0

  const chartData = points
    .filter(p => p.power_cost > 0 || p.dra_cost > 0)
    .map(p => ({
      name: p.name.length > 10 ? p.name.slice(0, 10) + '…' : p.name,
      'Power Cost (₹)': +p.power_cost.toFixed(0),
      'DRA Cost (₹)': +p.dra_cost.toFixed(0),
    }))

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-700">Cost Breakdown by Station</h3>
        <span className="text-sm font-bold text-gray-900">Total: {fmtRs(totalCost)}</span>
      </div>

      {chartData.length === 0 ? (
        <div className="flex items-center justify-center h-32 text-gray-400 text-sm">No cost data</div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
            <Tooltip formatter={(v: number) => [fmtRs(v)]} />
            <Legend verticalAlign="top" />
            <Bar dataKey="Power Cost (₹)" fill="#3b82f6" stackId="a" />
            <Bar dataKey="DRA Cost (₹)" fill="#f59e0b" stackId="a" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
