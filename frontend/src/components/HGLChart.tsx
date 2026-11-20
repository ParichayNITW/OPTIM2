import {
  CartesianGrid, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { MainlineResult, StationPoint } from '../types'

function normalize(name: string) {
  return name.trim().toLowerCase().replace(/\s+/g, '_')
}

function getNum(result: MainlineResult, key: string): number {
  const v = result[key]
  return typeof v === 'number' ? v : 0
}

export function extractStationPoints(result: MainlineResult): StationPoint[] {
  const stations = result.stations_used ?? []
  const points: StationPoint[] = []
  let cumKm = 0

  for (const st of stations) {
    const rawName = String(st.name ?? '')
    const n = normalize(rawName)
    const segLen = typeof st.length_km === 'number' ? st.length_km :
                   typeof st.segment_length === 'number' ? st.segment_length : 0
    cumKm += segLen

    points.push({
      name: rawName,
      km: cumKm,
      arrival_head: getNum(result, `residual_head_in_${n}`),
      discharge_head: getNum(result, `sdh_${n}`),
      maop: getNum(result, `maop_${n}`),
      elev: typeof st.elev === 'number' ? st.elev : 0,
      num_pumps: getNum(result, `num_pumps_${n}`),
      speed: getNum(result, `speed_${n}`),
      efficiency: getNum(result, `efficiency_${n}`),
      motor_kw: getNum(result, `motor_kw_${n}`),
      power_cost: getNum(result, `power_cost_${n}`),
      dra_cost: getNum(result, `dra_cost_${n}`),
      dra_ppm: getNum(result, `dra_ppm_${n}`),
      drag_reduction: getNum(result, `drag_reduction_${n}`),
    })
  }
  return points
}

interface HGLChartProps {
  result: MainlineResult
}

const CustomDot = (props: { cx?: number; cy?: number; payload?: StationPoint }) => {
  if (!props.cx || !props.cy) return null
  return <circle cx={props.cx} cy={props.cy} r={5} fill="#2563eb" stroke="#fff" strokeWidth={2} />
}

export function HGLChart({ result }: HGLChartProps) {
  const points = extractStationPoints(result)

  if (points.length === 0) {
    return <div className="flex items-center justify-center h-48 text-gray-400 text-sm">No hydraulic data available</div>
  }

  const chartData = points.map(p => ({
    km: +p.km.toFixed(2),
    name: p.name,
    'Arrival Head (m)': +p.arrival_head.toFixed(1),
    'Discharge Head (m)': +p.discharge_head.toFixed(1),
    'MAOP (m)': +p.maop.toFixed(1),
    'Elevation (m)': +p.elev.toFixed(1),
  }))

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Hydraulic Grade Line (HGL)</h3>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="km" label={{ value: 'Distance from Origin (km)', position: 'insideBottom', offset: -10 }} tick={{ fontSize: 11 }} />
          <YAxis label={{ value: 'Head (m)', angle: -90, position: 'insideLeft', offset: 10 }} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(val: number, name: string) => [`${val.toFixed(1)} m`, name]}
            labelFormatter={(label) => `${label} km`}
          />
          <Legend verticalAlign="top" />
          <Line type="monotone" dataKey="MAOP (m)" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="6 3" dot={false} />
          <Line type="monotone" dataKey="Discharge Head (m)" stroke="#16a34a" strokeWidth={2} dot={<CustomDot />} />
          <Line type="monotone" dataKey="Arrival Head (m)" stroke="#2563eb" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="Elevation (m)" stroke="#9ca3af" strokeWidth={1} strokeDasharray="4 2" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
