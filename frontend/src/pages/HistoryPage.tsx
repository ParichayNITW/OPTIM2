import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { listJobs } from '../api/optimize'
import { JobStatusBadge } from '../components/JobStatusBadge'
import type { JobStatus } from '../types'

const PAGE_SIZE = 25

export function HistoryPage() {
  const [offset, setOffset] = useState(0)
  const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all')

  const { data, isLoading } = useQuery({
    queryKey: ['jobs', offset],
    queryFn: () => listJobs(PAGE_SIZE, offset),
    refetchInterval: 15000,
  })

  const jobs = (data?.jobs ?? []).filter(j => statusFilter === 'all' || j.status === statusFilter)
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / PAGE_SIZE)
  const page = Math.floor(offset / PAGE_SIZE) + 1

  return (
    <div className="space-y-5 max-w-4xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">Job History</h1>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Filter:</span>
          {(['all', 'success', 'running', 'pending', 'failed'] as const).map(s => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
                statusFilter === s
                  ? 'bg-brand-600 text-white border-brand-600'
                  : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
              }`}
            >
              {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-sm text-gray-400">Loading…</div>
        ) : jobs.length === 0 ? (
          <div className="p-8 text-center text-sm text-gray-400">No jobs found.</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600">Job ID</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600">Status</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600">Started</th>
                <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600">Duration</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {jobs.map(j => {
                const dur = j.finished_at
                  ? (() => {
                      const ms = new Date(j.finished_at).getTime() - new Date(j.created_at).getTime()
                      return ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60000).toFixed(1)}m`
                    })()
                  : null
                return (
                  <tr key={j.job_id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-mono text-xs text-gray-700">{j.job_id.slice(0, 16)}…</td>
                    <td className="px-4 py-3"><JobStatusBadge status={j.status} /></td>
                    <td className="px-4 py-3 text-xs text-gray-500">{new Date(j.created_at).toLocaleString()}</td>
                    <td className="px-4 py-3 text-xs text-gray-500">{dur ?? '—'}</td>
                    <td className="px-4 py-3 text-right">
                      <Link to={`/results/${j.job_id}`} className="text-xs text-brand-600 hover:underline">View</Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500 text-xs">Page {page} of {totalPages} ({total} total)</span>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset(o => Math.max(0, o - PAGE_SIZE))}
              disabled={offset === 0}
              className="text-xs px-3 py-1.5 border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >← Prev</button>
            <button
              onClick={() => setOffset(o => o + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total}
              className="text-xs px-3 py-1.5 border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >Next →</button>
          </div>
        </div>
      )}
    </div>
  )
}
