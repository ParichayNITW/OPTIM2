import { useQuery } from '@tanstack/react-query'
import { getResult } from '../api/optimize'
import type { JobResult } from '../types'

const TERMINAL_STATES = new Set(['success', 'failed'])

export function useJobPolling(jobId: string | undefined) {
  return useQuery<JobResult>({
    queryKey: ['job', jobId],
    queryFn: () => getResult(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && TERMINAL_STATES.has(status) ? false : 2000
    },
    staleTime: 0,
  })
}
