import client from './client'
import type { JobListResponse, JobResult, OptimizeRequest, OptimizeResponse } from '../types'

export const submitOptimization = (payload: OptimizeRequest) =>
  client.post<OptimizeResponse>('/optimize', payload).then(r => r.data)

export const getResult = (jobId: string) =>
  client.get<JobResult>(`/results/${jobId}`).then(r => r.data)

export const listJobs = (limit = 20, offset = 0) =>
  client.get<JobListResponse>('/jobs', { params: { limit, offset } }).then(r => r.data)
