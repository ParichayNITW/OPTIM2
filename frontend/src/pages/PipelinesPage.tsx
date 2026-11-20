import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { createPipeline, deletePipeline, listPipelines } from '../api/pipelines'
import type { Pipeline } from '../types'

function PipelineCard({ pipeline, onDelete }: { pipeline: Pipeline; onDelete: () => void }) {
  const [confirmDelete, setConfirmDelete] = useState(false)

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-semibold text-gray-800">{pipeline.name}</div>
          {pipeline.description && (
            <div className="text-xs text-gray-500 mt-0.5">{pipeline.description}</div>
          )}
          <div className="text-xs text-gray-400 mt-1">
            {new Date(pipeline.updated_at).toLocaleDateString()}
          </div>
        </div>
        <div className="flex gap-2">
          {!confirmDelete ? (
            <button
              onClick={() => setConfirmDelete(true)}
              className="text-xs text-red-500 hover:text-red-700"
            >Delete</button>
          ) : (
            <div className="flex gap-1 items-center">
              <span className="text-xs text-gray-500">Sure?</span>
              <button onClick={onDelete} className="text-xs text-red-600 font-medium">Yes</button>
              <button onClick={() => setConfirmDelete(false)} className="text-xs text-gray-500">No</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function PipelinesPage() {
  const qc = useQueryClient()
  const { data: pipelines = [], isLoading } = useQuery({ queryKey: ['pipelines'], queryFn: listPipelines })
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [jsonText, setJsonText] = useState('')
  const [jsonError, setJsonError] = useState('')
  const [showForm, setShowForm] = useState(false)

  const createMutation = useMutation({
    mutationFn: createPipeline,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['pipelines'] }); setShowForm(false); setName(''); setDescription(''); setJsonText('') },
  })

  const deleteMutation = useMutation({
    mutationFn: deletePipeline,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['pipelines'] }),
  })

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault()
    setJsonError('')
    let config: Record<string, unknown>
    try { config = JSON.parse(jsonText) } catch { setJsonError('Invalid JSON'); return }
    createMutation.mutate({ name, description, config })
  }

  return (
    <div className="space-y-5 max-w-3xl">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">Saved Pipelines</h1>
        <button
          onClick={() => setShowForm(!showForm)}
          className="text-sm font-medium bg-brand-600 hover:bg-brand-700 text-white px-4 py-1.5 rounded-md transition-colors"
        >
          {showForm ? 'Cancel' : '+ Save New'}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-3">
          <h2 className="text-sm font-semibold text-gray-800">Save Pipeline Configuration</h2>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
            <input type="text" required value={name} onChange={e => setName(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Description (optional)</label>
            <input type="text" value={description} onChange={e => setDescription(e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Case JSON</label>
            <textarea value={jsonText} onChange={e => setJsonText(e.target.value)} required rows={8}
              placeholder={'{\n  "stations": [...],\n  ...\n}'}
              className="w-full font-mono text-xs border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-brand-500 resize-y" />
            {jsonError && <div className="text-xs text-red-600 mt-1">{jsonError}</div>}
          </div>
          <button type="submit" disabled={createMutation.isPending}
            className="bg-brand-600 hover:bg-brand-700 disabled:opacity-60 text-white text-sm font-medium px-5 py-2 rounded-md transition-colors">
            {createMutation.isPending ? 'Saving…' : 'Save Pipeline'}
          </button>
        </form>
      )}

      {isLoading ? (
        <div className="text-sm text-gray-400">Loading…</div>
      ) : pipelines.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center text-sm text-gray-400">
          No saved pipelines yet. Save a case JSON to quickly re-run it.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {pipelines.map(p => (
            <PipelineCard key={p.id} pipeline={p} onDelete={() => deleteMutation.mutate(p.id)} />
          ))}
        </div>
      )}
    </div>
  )
}
