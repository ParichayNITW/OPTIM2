import client from './client'
import type { Pipeline, PipelineCreate } from '../types'

export const listPipelines = () =>
  client.get<Pipeline[]>('/pipelines').then(r => r.data)

export const createPipeline = (body: PipelineCreate) =>
  client.post<Pipeline>('/pipelines', body).then(r => r.data)

export const getPipeline = (id: number) =>
  client.get<Pipeline>(`/pipelines/${id}`).then(r => r.data)

export const updatePipeline = (id: number, body: PipelineCreate) =>
  client.put<Pipeline>(`/pipelines/${id}`, body).then(r => r.data)

export const deletePipeline = (id: number) =>
  client.delete(`/pipelines/${id}`)
