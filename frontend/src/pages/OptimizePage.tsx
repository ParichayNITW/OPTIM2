import React, { useState, useRef } from 'react';
import { GlobalParamsPanel } from '../components/pipeline/GlobalParamsPanel';
import { StationBuilder } from '../components/pipeline/StationBuilder';
import { LooplineConfigPanel } from '../components/pipeline/LooplineConfigPanel';
import { OperatingModePanel } from '../components/pipeline/OperatingModePanel';
import { DRALacingPanel } from '../components/pipeline/DRALacingPanel';
import { AdvancedSearchPanel } from '../components/pipeline/AdvancedSearchPanel';
import { PipelineMapPreview } from '../components/pipeline/PipelineMapPreview';
import { ResultsPanel } from '../components/results/ResultsPanel';
import { usePipelineStore } from '../store/pipelineStore';
import { submitOptimization, getResult } from '../api/optimize';

type InputTab = 'pipeline' | 'pumps' | 'loopline' | 'operating';

export function OptimizePage() {
  const [inputTab, setInputTab] = useState<InputTab>('pipeline');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [jobStatus, setJobStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const {stations, terminal, exportCase, loadCase} = usePipelineStore();

  const handleRun = async () => {
    setError('');
    setResult(null);
    setJobStatus('Submitting…');
    setRunning(true);

    try {
      const payload = exportCase();
      const {job_id} = await submitOptimization(payload as never);
      setJobStatus('Running…');

      pollRef.current = setInterval(async () => {
        try {
          const r = await getResult(job_id);
          if (r.status === 'success' || r.status === 'failed') {
            clearInterval(pollRef.current!);
            setRunning(false);
            if (r.status === 'success' && r.result) {
              setResult(r.result as unknown as Record<string, unknown>);
              setJobStatus('');
            } else {
              setError(r.error || 'Optimization failed.');
              setJobStatus('');
            }
          }
        } catch {
          // keep polling
        }
      }, 2000);
    } catch (err: unknown) {
      setRunning(false);
      setJobStatus('');
      const msg = (err as {response?: {data?: {detail?: string}}})?.response?.data?.detail;
      setError(msg ?? 'Failed to submit job');
    }
  };

  const handleSaveCase = () => {
    const json = JSON.stringify(exportCase(), null, 2);
    const blob = new Blob([json], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_case_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoadCase = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const json = JSON.parse(ev.target?.result as string);
        loadCase(json);
        setResult(null);
        setError('');
      } catch {
        setError('Invalid case file — could not parse JSON.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const INPUT_TABS: {id: InputTab; label: string}[] = [
    {id: 'pipeline',  label: 'Pipeline'},
    {id: 'pumps',     label: 'Pumps'},
    {id: 'loopline',  label: 'Loopline'},
    {id: 'operating', label: 'Operating Mode'},
  ];

  const stationsForResults = stations.map(s => ({
    name: s.name, length_km: s.length_km, elevation: s.elevation, is_pump: s.is_pump,
  }));

  return (
    <div className="flex h-[calc(100vh-56px)] overflow-hidden">
      {/* ── LEFT PANEL: inputs ── */}
      <div className="w-2/5 min-w-[380px] flex flex-col border-r bg-white overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-3 py-2 border-b bg-gray-50 shrink-0">
          <span className="text-sm font-bold text-gray-800 mr-auto">Pipeline Builder</span>
          <label className="cursor-pointer text-xs text-gray-600 hover:text-gray-900 border border-gray-300 rounded px-2 py-1 hover:bg-gray-100">
            Load Case
            <input type="file" accept=".json" onChange={handleLoadCase} className="hidden" />
          </label>
          <button onClick={handleSaveCase}
            className="text-xs text-gray-600 hover:text-gray-900 border border-gray-300 rounded px-2 py-1 hover:bg-gray-100">
            Save Case
          </button>
          <button onClick={handleRun} disabled={running || stations.length === 0}
            className="text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded px-3 py-1 transition-colors">
            {running ? (jobStatus || 'Running…') : '▶ Run'}
          </button>
        </div>

        {/* Input tabs */}
        <div className="flex border-b bg-gray-50 shrink-0">
          {INPUT_TABS.map(t => (
            <button key={t.id} onClick={() => setInputTab(t.id)}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
                inputTab === t.id
                  ? 'border-blue-600 text-blue-700 bg-white'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Error bar */}
        {error && (
          <div className="shrink-0 px-3 py-2 bg-red-50 border-b border-red-200 text-xs text-red-600">
            {error}
            <button onClick={() => setError('')} className="ml-2 font-bold">×</button>
          </div>
        )}

        {/* Input content */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {inputTab === 'pipeline' && (
            <>
              <GlobalParamsPanel />
              <StationBuilder />
              <DRALacingPanel />
              <AdvancedSearchPanel />
            </>
          )}
          {inputTab === 'pumps' && <StationBuilder />}
          {inputTab === 'loopline' && <LooplineConfigPanel />}
          {inputTab === 'operating' && <OperatingModePanel />}
        </div>
      </div>

      {/* ── RIGHT PANEL: map or results ── */}
      <div className="flex-1 overflow-hidden flex flex-col bg-gray-50">
        {result ? (
          <ResultsPanel
            result={result}
            stationsInput={stationsForResults}
            terminal={terminal}
          />
        ) : (
          <div className="flex-1 overflow-y-auto p-4">
            <PipelineMapPreview />
            {running && (
              <div className="mt-4 flex items-center gap-2 text-sm text-blue-700">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                {jobStatus}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
