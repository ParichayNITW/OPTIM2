import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { submitOptimization } from '../api/optimize'
import { listPipelines } from '../api/pipelines'
import type { OptimizeRequest } from '../types'

const ECON_FIELDS: { key: keyof OptimizeRequest; label: string; unit: string; step: number }[] = [
  { key: 'FLOW',         label: 'Flow Rate',       unit: 'm³/h',  step: 10 },
  { key: 'Price_HSD',    label: 'HSD Price',       unit: '₹/L',   step: 0.1 },
  { key: 'RateDRA',      label: 'DRA Rate',        unit: 'mg/L·km', step: 0.001 },
  { key: 'Fuel_density', label: 'Fuel Density',    unit: 'kg/L',  step: 0.001 },
  { key: 'Ambient_temp', label: 'Ambient Temp',    unit: '°C',    step: 1 },
  { key: 'hours',        label: 'Operating Hours', unit: 'h',     step: 1 },
]

export function OptimizePage() {
  const navigate = useNavigate()
  const [jsonText, setJsonText] = useState('')
  const [overrides, setOverrides] = useState<Partial<OptimizeRequest>>({})
  const [selectedPipelineId, setSelectedPipelineId] = useState<number | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const { data: pipelines = [] } = useQuery({
    queryKey: ['pipelines'],
    queryFn: listPipelines,
  })

  const handlePipelineSelect = (id: number) => {
    setSelectedPipelineId(id)
    const p = pipelines.find(p => p.id === id)
    if (p) setJsonText(JSON.stringify(p.config, null, 2))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    let parsed: OptimizeRequest
    try {
      parsed = JSON.parse(jsonText)
    } catch {
      setError('Invalid JSON — check the case payload above')
      return
    }

    // Apply economic overrides
    const payload: OptimizeRequest = { ...parsed, ...overrides }

    if (!payload.stations || !Array.isArray(payload.stations)) {
      setError('Payload must have a "stations" array')
      return
    }

    setSubmitting(true)
    try {
      const { job_id } = await submitOptimization(payload)
      navigate(`/results/${job_id}`)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Failed to submit job')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-bold text-gray-900">New Optimization</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Paste a case JSON (exported from Streamlit) or load a saved pipeline, then optionally override economic parameters.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Pipeline selector */}
        {pipelines.length > 0 && (
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <label className="block text-xs font-medium text-gray-700 mb-1">Load saved pipeline</label>
            <select
              value={selectedPipelineId ?? ''}
              onChange={e => handlePipelineSelect(Number(e.target.value))}
              className="border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            >
              <option value="">— select a pipeline —</option>
              {pipelines.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>
        )}

        {/* JSON input */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-gray-700">Case JSON payload</label>
            <span className="text-xs text-gray-400">Export from Streamlit → Save Case → paste here</span>
          </div>
          <textarea
            value={jsonText}
            onChange={e => setJsonText(e.target.value)}
            required
            rows={12}
            placeholder={'{\n  "stations": [...],\n  "terminal": {...},\n  "FLOW": 1200,\n  ...\n}'}
            className="w-full font-mono text-xs border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500 resize-y"
          />
        </div>

        {/* Economic overrides */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-xs font-semibold text-gray-700 mb-3">Override economic parameters (optional)</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {ECON_FIELDS.map(f => (
              <div key={f.key}>
                <label className="block text-xs text-gray-600 mb-1">{f.label} <span className="text-gray-400">({f.unit})</span></label>
                <input
                  type="number"
                  step={f.step}
                  min={0}
                  value={(overrides[f.key] as number | undefined) ?? ''}
                  onChange={e => {
                    const v = e.target.value === '' ? undefined : Number(e.target.value)
                    setOverrides(prev => ({ ...prev, [f.key]: v }))
                  }}
                  placeholder="from JSON"
                  className="w-full border border-gray-300 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                />
              </div>
            ))}
          </div>
        </div>

        {error && <div className="text-sm text-red-600 bg-red-50 px-4 py-3 rounded-lg">{error}</div>}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={submitting || !jsonText.trim()}
            className="bg-brand-600 hover:bg-brand-700 disabled:opacity-60 text-white font-medium text-sm px-6 py-2.5 rounded-md transition-colors"
          >
            {submitting ? 'Submitting…' : '▶ Run Optimization'}
          </button>
        </div>
      </form>
    </div>
  )
}
