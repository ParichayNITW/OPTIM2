import React from 'react';
import { ParsedScheduleResult } from '../../../utils/resultParser';
import type { StationData } from '../../../store/pipelineStore';

interface Props {
  schedule: ParsedScheduleResult;
  stations: StationData[];
}

function clamp01(t: number) { return Math.min(1, Math.max(0, t)); }

function colorScale(val: number, max: number, r0: number, g0: number, b0: number, r1: number, g1: number, b1: number) {
  const t = max > 0 ? clamp01(val / max) : 0;
  const r = Math.round(r0 + (r1 - r0) * t);
  const g = Math.round(g0 + (g1 - g0) * t);
  const b = Math.round(b0 + (b1 - b0) * t);
  return `rgb(${r},${g},${b})`;
}

interface HeatmapProps {
  title: string;
  pumpStations: StationData[];
  hours: string[];
  matrix: number[][];
  colorFn: (val: number, max: number) => string;
  unit: string;
  maxOverride?: number;
}

function Heatmap({ title, pumpStations, hours, matrix, colorFn, unit, maxOverride }: HeatmapProps) {
  if (pumpStations.length === 0) return null;

  const cellW = Math.max(14, Math.min(28, 580 / Math.max(hours.length, 1)));
  const cellH = 20;
  const leftPad = 90;
  const topPad = 24;
  const totalW = hours.length * (cellW + 1) + leftPad + 10;
  const totalH = pumpStations.length * (cellH + 2) + topPad + 20;
  const allVals = matrix.flat();
  const maxVal = maxOverride ?? Math.max(...allVals, 1);

  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-600 mb-1">{title}</h4>
      <div className="overflow-x-auto">
        <svg width={totalW} height={totalH}>
          {/* Hour headers */}
          {hours.map((lbl, hi) => (
            hi % 4 === 0 && (
              <text key={hi}
                x={leftPad + hi * (cellW + 1) + cellW / 2}
                y={topPad - 4}
                textAnchor="middle" fontSize={8} fill="#6b7280">{lbl}</text>
            )
          ))}
          {/* Cells */}
          {pumpStations.map((s, si) =>
            matrix[si].map((val, hi) => {
              const fill = colorFn(val, maxVal);
              const textFill = val / maxVal > 0.6 ? '#fff' : '#1e3a5f';
              return (
                <g key={`${si}-${hi}`}>
                  <rect
                    x={leftPad + hi * (cellW + 1)}
                    y={topPad + si * (cellH + 2)}
                    width={cellW} height={cellH}
                    fill={fill} rx={2}
                  />
                  {cellW >= 20 && val > 0 && (
                    <text
                      x={leftPad + hi * (cellW + 1) + cellW / 2}
                      y={topPad + si * (cellH + 2) + 13}
                      textAnchor="middle" fontSize={7} fill={textFill}>
                      {val.toFixed(0)}
                    </text>
                  )}
                </g>
              );
            })
          )}
          {/* Station labels */}
          {pumpStations.map((s, si) => (
            <text key={si}
              x={leftPad - 4}
              y={topPad + si * (cellH + 2) + 13}
              textAnchor="end" fontSize={8} fill="#374151">{s.name}</text>
          ))}
          {/* Unit label */}
          <text x={leftPad - 4} y={topPad - 10} textAnchor="end" fontSize={7} fill="#9ca3af">{unit}</text>
        </svg>
      </div>
    </div>
  );
}

export function DRAHeatmapTab({ schedule, stations }: Props) {
  const pumpStations = stations.filter(s => s.is_pump);
  const hours = schedule.hourly.map(h => h.hour_label);

  const ppmMatrix: number[][] = pumpStations.map(s => {
    const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
    return schedule.hourly.map(h => h.stations[k]?.dra_ppm || 0);
  });
  const drMatrix: number[][] = pumpStations.map(s => {
    const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
    return schedule.hourly.map(h => h.stations[k]?.drag_reduction || 0);
  });
  const costMatrix: number[][] = pumpStations.map(s => {
    const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
    return schedule.hourly.map(h => h.stations[k]?.dra_cost || 0);
  });

  return (
    <div className="space-y-5">
      <Heatmap
        title="DRA Dosage (ppm)"
        pumpStations={pumpStations}
        hours={hours}
        matrix={ppmMatrix}
        unit="ppm"
        colorFn={(v, max) => v === 0 ? '#f9fafb' : colorScale(v, max, 255, 251, 235, 239, 68, 68)}
      />
      <Heatmap
        title="Drag Reduction (%)"
        pumpStations={pumpStations}
        hours={hours}
        matrix={drMatrix}
        unit="%DR"
        colorFn={(v, max) => v === 0 ? '#f9fafb' : colorScale(v, max, 240, 253, 244, 22, 163, 74)}
      />
      <Heatmap
        title="DRA Cost (₹/hr)"
        pumpStations={pumpStations}
        hours={hours}
        matrix={costMatrix}
        unit="₹"
        colorFn={(v, max) => v === 0 ? '#f9fafb' : colorScale(v, max, 239, 246, 255, 29, 78, 216)}
      />
    </div>
  );
}
