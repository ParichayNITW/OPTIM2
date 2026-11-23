import React, { useState, useMemo } from 'react';
import { parseResult, parseScheduleResult, ParsedResult, ParsedScheduleResult } from '../../utils/resultParser';
import { SummaryTab } from './tabs/SummaryTab';
import { CostsTab } from './tabs/CostsTab';
import { PerformanceTab } from './tabs/PerformanceTab';
import { SystemCurvesTab } from './tabs/SystemCurvesTab';
import { PumpSystemTab } from './tabs/PumpSystemTab';
import { DRACurvesTab } from './tabs/DRACurvesTab';
import { Analysis3DTab } from './tabs/Analysis3DTab';
import { Pressure3DTab } from './tabs/Pressure3DTab';
import { PipelineMapTab } from './tabs/PipelineMapTab';
// import { SensitivityTab } from './tabs/SensitivityTab';
import { BenchmarkingTab } from './tabs/BenchmarkingTab';
import { SavingsTab } from './tabs/SavingsTab';
import { ReportsTab } from './tabs/ReportsTab';
import { ScheduleResultsTab } from './tabs/ScheduleResultsTab';
import { HourlyChartsTab } from './tabs/HourlyChartsTab';
import { DRAHeatmapTab } from './tabs/DRAHeatmapTab';
import { NextDayLinefillTab } from './tabs/NextDayLinefillTab';
import { PumpingPlannerTab } from './tabs/PumpingPlannerTab';
import { ScheduleAuditTab } from './tabs/ScheduleAuditTab';
import type { StationData } from '../../store/pipelineStore';

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
  fullStations?: StationData[];
  jobId?: string;
}

const FLOW_RATE_TABS = [
  { id: 'summary',      label: 'Summary' },
  { id: 'costs',        label: 'Costs' },
  { id: 'performance',  label: 'Performance' },
  { id: 'system',       label: 'System Curves' },
  { id: 'pump_system',  label: 'Pump-System' },
  { id: 'dra',          label: 'DRA Curves' },
  { id: 'analysis3d',   label: '3D Analysis' },
  { id: 'pressure3d',   label: '3D Pressure' },
  { id: 'map',          label: 'Pipeline Map' },
  // { id: 'sensitivity',  label: 'Sensitivity' },
  { id: 'benchmarking', label: 'Benchmarking' },
  { id: 'savings',      label: 'Savings' },
  { id: 'reports',      label: 'Reports' },
];

const SCHEDULE_TABS = [
  { id: 'schedule',     label: 'Schedule Results' },
  { id: 'hourly_charts',label: 'Hourly Charts' },
  { id: 'dra_heatmap',  label: 'DRA Heatmap' },
  { id: 'next_day_lf',  label: 'Next-Day Linefill' },
  { id: 'system',       label: 'System Curves' },
  { id: 'pump_system',  label: 'Pump-System' },
  { id: 'dra',          label: 'DRA Curves' },
  { id: 'summary',      label: 'Summary' },
  { id: 'costs',        label: 'Costs' },
  { id: 'performance',  label: 'Performance' },
  { id: 'benchmarking', label: 'Benchmarking' },
  { id: 'reports',      label: 'Reports' },
];

const PLANNER_TABS = [
  { id: 'planner',      label: 'Pumping Planner' },
  { id: 'next_day_lf',  label: 'Next-Day Linefill' },
  { id: 'system',       label: 'System Curves' },
  { id: 'pump_system',  label: 'Pump-System' },
  { id: 'dra',          label: 'DRA Curves' },
  { id: 'summary',      label: 'Summary' },
  { id: 'costs',        label: 'Costs' },
  { id: 'reports',      label: 'Reports' },
];

export function ResultsPanel({ result, stationsInput, terminal, fullStations, jobId }: Props) {
  const opMode = (result.op_mode as string) || 'flow_rate';
  const isSchedule = opMode === 'daily_schedule';
  const isPlanner  = opMode === 'pumping_planner';
  const isScheduleMode = isSchedule || isPlanner;

  const TABS = isSchedule ? SCHEDULE_TABS : isPlanner ? PLANNER_TABS : FLOW_RATE_TABS;
  const [active, setActive] = useState(TABS[0].id);

  const parsed: ParsedResult = useMemo(
    () => parseResult(result, stationsInput, terminal),
    [result, stationsInput, terminal],
  );

  const schedule: ParsedScheduleResult | null = useMemo(() => {
    if (!isScheduleMode) return null;
    return parseScheduleResult(result, stationsInput, terminal);
  }, [result, stationsInput, isScheduleMode, terminal]);

  const parsedHour0: ParsedResult | null = useMemo(() => {
    if (!isScheduleMode) return null;
    const hourlyArr = (result.mainline_schedule as Record<string, unknown>[] | undefined) || [];
    const firstGood = hourlyArr.find(h => !h.error);
    if (!firstGood) return null;
    return parseResult({...firstGood, op_mode: 'flow_rate'}, stationsInput, terminal);
  }, [isScheduleMode, result, stationsInput, terminal]);

  const stationsForTabs: StationData[] = fullStations || stationsInput.map(s => ({
    id: s.name,
    name: s.name,
    elevation: s.elevation,
    is_pump: s.is_pump,
    length_km: s.length_km,
    max_dra_pct: 20,
    min_suction_head: 0,
    OD_in: 20,
    wall_thk_in: 0.375,
    SMYS_psi: 52000,
    roughness_m: 0.000046,
    max_pumps: 3,
    delivery: 0,
    supply: 0,
    pumpA: {} as never,
    pumpB: {} as never,
    peaks_table: [],
    branches: [],
  }));

  const renderTab = () => {
    if (isScheduleMode && schedule) {
      switch (active) {
        case 'schedule':      return <ScheduleResultsTab schedule={schedule} stations={stationsForTabs} terminal={terminal} />;
        case 'hourly_charts': return <HourlyChartsTab schedule={schedule} stations={stationsForTabs} />;
        case 'dra_heatmap':   return <DRAHeatmapTab schedule={schedule} stations={stationsForTabs} />;
        case 'next_day_lf':   return <NextDayLinefillTab schedule={schedule} />;
        case 'planner':       return <PumpingPlannerTab schedule={schedule} />;
        case 'audit':         return <ScheduleAuditTab schedule={schedule} />;
        case 'system':
          return parsedHour0 ? (
            <><p className="text-xs text-blue-700 bg-blue-50 px-3 py-1.5 border-b border-blue-100">Curves/data as of start time (Hour 0)</p><SystemCurvesTab parsed={parsedHour0} /></>
          ) : null;
        case 'pump_system':
          return parsedHour0 ? (
            <><p className="text-xs text-blue-700 bg-blue-50 px-3 py-1.5 border-b border-blue-100">Curves/data as of start time (Hour 0)</p><PumpSystemTab parsed={parsedHour0} fullStations={stationsForTabs} /></>
          ) : null;
        case 'dra':
          return parsedHour0 ? (
            <><p className="text-xs text-blue-700 bg-blue-50 px-3 py-1.5 border-b border-blue-100">Curves/data as of start time (Hour 0)</p><DRACurvesTab parsed={parsedHour0} /></>
          ) : null;
        // Fall through to common tabs:
      }
    }
    switch (active) {
      case 'summary':      return <SummaryTab parsed={parsed} />;
      case 'costs':        return <CostsTab parsed={parsed} />;
      case 'performance':  return <PerformanceTab parsed={parsed} />;
      case 'system':       return <SystemCurvesTab parsed={parsed} />;
      case 'pump_system':  return <PumpSystemTab parsed={parsed} fullStations={stationsForTabs} />;
      case 'dra':          return <DRACurvesTab parsed={parsed} />;
      case 'analysis3d':   return <Analysis3DTab parsed={parsed} />;
      case 'pressure3d':   return <Pressure3DTab parsed={parsed} />;
      case 'map':          return <PipelineMapTab parsed={parsed} />;
      // case 'sensitivity':  return <SensitivityTab result={result} />;
      case 'benchmarking': return <BenchmarkingTab parsed={parsed} />;
      case 'savings':      return <SavingsTab parsed={parsed} />;
      case 'reports':      return <ReportsTab jobId={jobId} opMode={opMode} />;
      default:             return null;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {isScheduleMode && schedule && (
        <div className="bg-blue-600 text-white px-3 py-1.5 text-xs flex items-center gap-3 shrink-0">
          <span className="font-semibold">
            {isSchedule ? '24-hr Daily Schedule' : 'Multi-Day Planner'}
          </span>
          <span>Total: ₹{schedule.total_cost.toLocaleString('en-IN', {maximumFractionDigits: 0})}</span>
          <span>{schedule.total_throughput.toLocaleString('en-IN', {maximumFractionDigits: 0})} m³</span>
          {schedule.hourly_errors.length > 0 && (
            <span className="bg-orange-400 px-1.5 rounded text-orange-900 font-medium">
              ⚠ {schedule.hourly_errors.length} infeasible
            </span>
          )}
        </div>
      )}
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
      <div className="flex-1 overflow-y-auto">
        {renderTab()}
      </div>
    </div>
  );
}
