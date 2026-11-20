import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listJobs } from '../api/optimize'
import { JobStatusBadge } from '../components/JobStatusBadge'
import { useAuth } from '../hooks/useAuth'

function relativeTime(isoStr: string) {
  const diff = Date.now() - new Date(isoStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export function DashboardPage() {
  const { user } = useAuth()
  const { data, isLoading } = useQuery({
    queryKey: ['jobs-dashboard'],
    queryFn: () => listJobs(10, 0),
    refetchInterval: 10000,
  })

  const jobs = data?.jobs ?? []
  const counts = { success: 0, running: 0, failed: 0, pending: 0 }
  for (const j of jobs) counts[j.status] = (counts[j.status] ?? 0) + 1

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-0.5">Welcome back, {user?.name || user?.email}</p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-4 gap-4">
        {([
          { label: 'Total Jobs', value: data?.total ?? 0, color: 'text-gray-900' },
          { label: 'Completed', value: counts.success, color: 'text-green-700' },
          { label: 'Running', value: counts.running, color: 'text-blue-700' },
          { label: 'Failed', value: counts.failed, color: 'text-red-700' },
        ] as const).map(s => (
          <div key={s.label} className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Quick actions */}
      <div className="flex gap-3">
        <Link to="/optimize"
          className="inline-flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium px-4 py-2 rounded-md transition-colors">
          ▶ New Optimization
        </Link>
        <Link to="/history"
          className="inline-flex items-center gap-2 bg-white hover:bg-gray-50 text-gray-700 text-sm font-medium px-4 py-2 rounded-md border border-gray-300 transition-colors">
          ☰ View All Jobs
        </Link>
      </div>

      {/* Recent jobs */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="px-5 py-4 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-800">Recent Jobs</h2>
        </div>
        {isLoading ? (
          <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
        ) : jobs.length === 0 ? (
          <div className="px-5 py-8 text-center text-sm text-gray-400">
            No jobs yet. <Link to="/optimize" className="text-brand-600 hover:underline">Run your first optimization.</Link>
          </div>
        ) : (
          <ul className="divide-y divide-gray-100">
            {jobs.map(j => (
              <li key={j.job_id} className="px-5 py-3 flex items-center justify-between hover:bg-gray-50">
                <div>
                  <Link to={`/results/${j.job_id}`} className="text-sm font-medium text-brand-700 hover:underline">
                    {j.job_id.slice(0, 8)}…
                  </Link>
                  <div className="text-xs text-gray-400 mt-0.5">{relativeTime(j.created_at)}</div>
                </div>
                <JobStatusBadge status={j.status} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
