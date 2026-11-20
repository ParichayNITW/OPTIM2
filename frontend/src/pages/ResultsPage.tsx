import { useParams } from 'react-router-dom'
import { CostBreakdown } from '../components/CostBreakdown'
import { DRAProfileChart } from '../components/DRAProfileChart'
import { HGLChart } from '../components/HGLChart'
import { JobStatusBadge } from '../components/JobStatusBadge'
import { PipelineSchematic } from '../components/PipelineSchematic'
import { StationCards } from '../components/StationCards'
import { useJobPolling } from '../hooks/useJobPolling'
import type { MainlineResult } from '../types'

function fmt(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v)
}

function elapsed(a: string, b: string | null | undefined) {
  if (!b) return null
  const ms = new Date(b).getTime() - new Date(a).getTime()
  return ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60000).toFixed(1)} min`
}

export function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const { data: job, isLoading } = useJobPolling(jobId)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">Loading…</div>
    )
  }

  if (!job) {
    return <div className="text-red-600">Job not found.</div>
  }

  const mainline = job.result?.mainline as MainlineResult | undefined
  const branches = job.result?.branches ?? []
  const totalCost = mainline?.total_cost

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Optimization Results</h1>
          <div className="text-xs text-gray-400 mt-0.5 font-mono">{job.job_id}</div>
        </div>
        <div className="text-right">
          <JobStatusBadge status={job.status} />
          {job.created_at && (
            <div className="text-xs text-gray-400 mt-1">
              Started: {new Date(job.created_at).toLocaleString()}
              {elapsed(job.created_at, job.finished_at) && (
                <> · Took {elapsed(job.created_at, job.finished_at)}</>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Running state */}
      {(job.status === 'pending' || job.status === 'running') && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
          <div className="text-2xl mb-2">⚙️</div>
          <div className="text-sm font-medium text-blue-800">DP solver running…</div>
          <div className="text-xs text-blue-600 mt-1">Enumerating pump type combinations → global optimum</div>
          <div className="mt-3 h-1.5 bg-blue-200 rounded-full overflow-hidden">
            <div className="h-full bg-blue-500 rounded-full animate-[progress_2s_ease-in-out_infinite]" style={{ width: '60%' }} />
          </div>
        </div>
      )}

      {/* Error */}
      {job.status === 'failed' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="text-sm font-medium text-red-800">Optimization failed</div>
          <div className="text-xs text-red-600 mt-1 font-mono">{job.error}</div>
        </div>
      )}

      {/* Success results */}
      {job.status === 'success' && mainline && (
        <>
          {/* Summary banner */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-brand-50 border border-brand-200 rounded-lg p-4">
              <div className="text-xs text-brand-700 font-medium">Total Operating Cost</div>
              <div className="text-2xl font-bold text-brand-900 mt-1">
                {totalCost != null ? fmt(totalCost) : '—'}
              </div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-xs text-green-700 font-medium">Solver Passes</div>
              <div className="text-sm font-semibold text-green-900 mt-1">
                {(mainline.executed_passes ?? []).join(' → ') || 'coarse → refine'}
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <div className="text-xs text-amber-700 font-medium">DRA Coverage</div>
              <div className="text-2xl font-bold text-amber-900 mt-1">
                {mainline.dra_front_km != null ? `${(mainline.dra_front_km as number).toFixed(1)} km` : '—'}
              </div>
            </div>
          </div>

          {/* Pipeline schematic */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <PipelineSchematic result={mainline} />
          </div>

          {/* HGL chart */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <HGLChart result={mainline} />
          </div>

          {/* Station cards */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <StationCards result={mainline} />
          </div>

          {/* Cost + DRA side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
              <CostBreakdown result={mainline} />
            </div>
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
              <DRAProfileChart result={mainline} />
            </div>
          </div>

          {/* Branch results */}
          {branches.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Branch Lines</h3>
              <div className="space-y-2">
                {branches.map((br, i) => (
                  <div key={i} className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-3">
                    <div className="text-sm font-medium text-gray-800">{br.branch_name || `Branch ${i + 1}`}</div>
                    <div className="text-sm text-gray-600">
                      {br.total_cost != null ? fmt(Number(br.total_cost)) : '—'}
                    </div>
                    {br.error && <span className="text-xs text-red-600">Error</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Raw JSON toggle */}
          <details className="bg-gray-50 rounded-lg border border-gray-200">
            <summary className="px-4 py-3 text-xs font-medium text-gray-600 cursor-pointer">Raw result JSON</summary>
            <pre className="px-4 pb-4 text-xs font-mono text-gray-700 overflow-auto max-h-96 whitespace-pre-wrap">
              {JSON.stringify(job.result, null, 2)}
            </pre>
          </details>
        </>
      )}
    </div>
  )
}
