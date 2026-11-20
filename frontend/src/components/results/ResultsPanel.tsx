import React, { useState } from 'react';
import { SummaryTab } from './tabs/SummaryTab';
import { CostsTab } from './tabs/CostsTab';
import { PerformanceTab } from './tabs/PerformanceTab';
import { SystemCurvesTab } from './tabs/SystemCurvesTab';
import { PumpSystemTab } from './tabs/PumpSystemTab';
import { DRACurvesTab } from './tabs/DRACurvesTab';
import { Analysis3DTab } from './tabs/Analysis3DTab';
import { Pressure3DTab } from './tabs/Pressure3DTab';
import { PipelineMapTab } from './tabs/PipelineMapTab';
import { SensitivityTab } from './tabs/SensitivityTab';
import { BenchmarkingTab } from './tabs/BenchmarkingTab';
import { SavingsTab } from './tabs/SavingsTab';

interface StationInput {
  name: string; length_km: number; elevation: number; is_pump: boolean;
}
interface TerminalInput {
  name: string; elevation: number; min_residual: number;
}

interface Props {
  result: Record<string, unknown>;
  stationsInput: StationInput[];
  terminal: TerminalInput;
}

const TABS = [
  {id: 'summary',      label: 'Summary'},
  {id: 'costs',        label: 'Costs'},
  {id: 'performance',  label: 'Performance'},
  {id: 'system',       label: 'System Curves'},
  {id: 'pump_system',  label: 'Pump-System'},
  {id: 'dra',          label: 'DRA Curves'},
  {id: 'analysis3d',   label: '3D Analysis'},
  {id: 'pressure3d',   label: '3D Pressure'},
  {id: 'map',          label: 'Pipeline Map'},
  {id: 'sensitivity',  label: 'Sensitivity'},
  {id: 'benchmarking', label: 'Benchmarking'},
  {id: 'savings',      label: 'Savings'},
];

export function ResultsPanel({result, stationsInput, terminal}: Props) {
  const [active, setActive] = useState('summary');

  const renderTab = () => {
    switch (active) {
      case 'summary':     return <SummaryTab result={result} stationsInput={stationsInput} terminal={terminal} />;
      case 'costs':       return <CostsTab result={result} />;
      case 'performance': return <PerformanceTab result={result} />;
      case 'system':      return <SystemCurvesTab result={result} />;
      case 'pump_system': return <PumpSystemTab result={result} />;
      case 'dra':         return <DRACurvesTab result={result} />;
      case 'analysis3d':  return <Analysis3DTab result={result} />;
      case 'pressure3d':  return <Pressure3DTab result={result} />;
      case 'map':         return <PipelineMapTab result={result} stationsInput={stationsInput} terminal={terminal} />;
      case 'sensitivity': return <SensitivityTab result={result} />;
      case 'benchmarking':return <BenchmarkingTab result={result} />;
      case 'savings':     return <SavingsTab result={result} />;
      default:            return null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex overflow-x-auto border-b bg-white shrink-0">
        {TABS.map(tab => (
          <button key={tab.id} onClick={() => setActive(tab.id)}
            className={`px-3 py-2 text-xs whitespace-nowrap font-medium border-b-2 transition-colors ${
              active === tab.id
                ? 'border-blue-600 text-blue-700 bg-blue-50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {renderTab()}
      </div>
    </div>
  );
}
