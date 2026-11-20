import type { JobStatus } from '../types'

const styles: Record<JobStatus, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  running: 'bg-blue-100 text-blue-800 animate-pulse',
  success: 'bg-green-100 text-green-800',
  failed:  'bg-red-100 text-red-800',
}

const labels: Record<JobStatus, string> = {
  pending: 'Pending',
  running: 'Running…',
  success: 'Complete',
  failed:  'Failed',
}

export function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${styles[status]}`}>
      {labels[status]}
    </span>
  )
}
