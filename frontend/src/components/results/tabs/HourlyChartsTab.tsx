import React from 'react';
import { ParsedScheduleResult } from '../../../utils/resultParser';
import type { StationData } from '../../../store/pipelineStore';

interface Props {
  schedule: ParsedScheduleResult;
  stations: StationData[];
}

export function HourlyChartsTab({ schedule, stations }: Props) {
  const pumpStations = stations.filter(s => s.is_pump);
  const hours = schedule.hourly.map(h => h.hour_label);

  // Colour palette
  const COLORS = ['#3b82f6','#f59e0b','#10b981','#ef4444','#8b5cf6','#ec4899'];

  const barWidth = Math.max(12, Math.min(32, 600 / Math.max(hours.length, 1)));
  const chartH = 160;
  const leftPad = 50, rightPad = 10, topPad = 10, bottomPad = 24;
  const chartW = hours.length * (barWidth + 2) + leftPad + rightPad;

  // ----- cost bar chart -----
  const costs = schedule.hourly.map(h => h.power_cost + h.dra_cost);
  const maxCost = Math.max(...costs, 1);

  function costBars() {
    return hours.map((lbl, i) => {
      const pc = schedule.hourly[i].power_cost;
      const dc = schedule.hourly[i].dra_cost;
      const total = pc + dc;
      const hPow = ((pc / maxCost) * (chartH - topPad - bottomPad));
      const hDra = ((dc / maxCost) * (chartH - topPad - bottomPad));
      const y0 = chartH - bottomPad;
      const x = leftPad + i * (barWidth + 2);
      return (
        <g key={i}>
          {/* Power bar */}
          <rect x={x} y={y0 - hPow - hDra} width={barWidth} height={hPow}
            fill="#f59e0b" opacity={0.85}/>
          {/* DRA bar stacked */}
          <rect x={x} y={y0 - hDra} width={barWidth} height={hDra}
            fill="#10b981" opacity={0.85}/>
          <text x={x + barWidth/2} y={chartH - 2} textAnchor="middle" fontSize={8} fill="#6b7280">
            {lbl.slice(0,2)}
          </text>
        </g>
      );
    });
  }

  // ----- efficiency heatmap -----
  const effMatrix: number[][] = pumpStations.map(s => {
    const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
    return schedule.hourly.map(h => h.stations[k]?.efficiency || 0);
  });

  const heatCellW = Math.max(14, Math.min(28, 600 / Math.max(hours.length, 1)));
  const heatCellH = 20;

  function effHeatmap() {
    return effMatrix.map((row, si) =>
      row.map((val, hi) => {
        const alpha = val > 0 ? 0.2 + (val / 100) * 0.8 : 0.05;
        return (
          <g key={`${si}-${hi}`}>
            <rect
              x={leftPad + hi * (heatCellW + 1)}
              y={topPad + si * (heatCellH + 2)}
              width={heatCellW} height={heatCellH}
              fill={`rgba(59, 130, 246, ${alpha})`}
              rx={2}
            />
            {val > 0 && heatCellW >= 20 && (
              <text
                x={leftPad + hi * (heatCellW + 1) + heatCellW/2}
                y={topPad + si * (heatCellH + 2) + 13}
                textAnchor="middle" fontSize={7} fill="#1e3a5f">
                {val.toFixed(0)}
              </text>
            )}
          </g>
        );
      })
    );
  }

  // ----- cumulative cost -----
  let cumCost = 0;
  const cumCosts = schedule.hourly.map(h => { cumCost += h.total_cost; return cumCost; });
  const maxCum = Math.max(cumCost, 1);
  const lineW = Math.max(400, chartW);

  function cumLine() {
    const pts = cumCosts.map((v, i) => {
      const x = leftPad + (i / Math.max(hours.length - 1, 1)) * (lineW - leftPad - rightPad);
      const y = chartH - bottomPad - (v / maxCum) * (chartH - topPad - bottomPad);
      return `${x},${y}`;
    });
    return <polyline points={pts.join(' ')} fill="none" stroke="#3b82f6" strokeWidth={2}/>;
  }

  const heatW = hours.length * (heatCellW + 1) + leftPad + rightPad;
  const heatH = pumpStations.length * (heatCellH + 2) + topPad + bottomPad + 10;

  return (
    <div className="space-y-5">

      {/* 1. Cost Breakdown Bar Chart */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">Hourly Cost (₹) — Power vs DRA</h4>
        <div className="flex gap-3 mb-1">
          {[{color:'#f59e0b',label:'Power'},{color:'#10b981',label:'DRA'}].map(({color,label}) => (
            <span key={label} className="flex items-center gap-1 text-xs text-gray-600">
              <span style={{background:color}} className="inline-block w-3 h-3 rounded-sm"/>
              {label}
            </span>
          ))}
        </div>
        <div className="overflow-x-auto">
          <svg width={chartW} height={chartH}>
            {/* Y axis */}
            {[0,0.25,0.5,0.75,1].map(frac => {
              const y = chartH - bottomPad - frac * (chartH - topPad - bottomPad);
              return (
                <g key={frac}>
                  <line x1={leftPad} y1={y} x2={chartW - rightPad} y2={y}
                    stroke="#e5e7eb" strokeWidth={1}/>
                  <text x={leftPad - 4} y={y + 3} textAnchor="end" fontSize={8} fill="#9ca3af">
                    {(frac * maxCost / 1000).toFixed(0)}k
                  </text>
                </g>
              );
            })}
            {costBars()}
          </svg>
        </div>
      </div>

      {/* 2. Efficiency Heatmap */}
      {pumpStations.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">Pump Efficiency % (by station × hour)</h4>
          <div className="overflow-x-auto">
            <svg width={heatW} height={heatH}>
              {effHeatmap()}
              {/* Station labels */}
              {pumpStations.map((s, si) => (
                <text key={si} x={leftPad - 4} y={topPad + si * (heatCellH + 2) + 13}
                  textAnchor="end" fontSize={8} fill="#374151">{s.name}</text>
              ))}
              {/* Hour labels */}
              {hours.map((lbl, hi) => (
                hi % 4 === 0 && (
                  <text key={hi}
                    x={leftPad + hi * (heatCellW + 1) + heatCellW/2}
                    y={topPad + pumpStations.length * (heatCellH + 2) + 10}
                    textAnchor="middle" fontSize={8} fill="#6b7280">{lbl.slice(0,2)}</text>
                )
              ))}
            </svg>
          </div>
        </div>
      )}

      {/* 3. Cumulative Cost Line */}
      <div>
        <h4 className="text-xs font-semibold text-gray-600 mb-1">Cumulative Cost (₹)</h4>
        <div className="overflow-x-auto">
          <svg width={lineW} height={chartH}>
            {[0,0.25,0.5,0.75,1].map(frac => {
              const y = chartH - bottomPad - frac * (chartH - topPad - bottomPad);
              return (
                <g key={frac}>
                  <line x1={leftPad} y1={y} x2={lineW - rightPad} y2={y}
                    stroke="#e5e7eb" strokeWidth={1}/>
                  <text x={leftPad - 4} y={y + 3} textAnchor="end" fontSize={8} fill="#9ca3af">
                    {(frac * maxCum / 1000).toFixed(0)}k
                  </text>
                </g>
              );
            })}
            {cumLine()}
            {/* hour ticks */}
            {hours.map((lbl, i) => (
              i % 4 === 0 && (
                <text key={i}
                  x={leftPad + (i / Math.max(hours.length - 1, 1)) * (lineW - leftPad - rightPad)}
                  y={chartH - 4} textAnchor="middle" fontSize={8} fill="#6b7280">{lbl}</text>
              )
            ))}
          </svg>
        </div>
      </div>

      {/* 4. Per-station DRA ppm lines */}
      {pumpStations.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-600 mb-1">DRA Dosage (ppm) by Station</h4>
          <div className="flex gap-3 mb-1 flex-wrap">
            {pumpStations.map((s, si) => (
              <span key={s.id} className="flex items-center gap-1 text-xs text-gray-600">
                <span style={{background: COLORS[si % COLORS.length]}} className="inline-block w-3 h-1.5 rounded-sm"/>
                {s.name}
              </span>
            ))}
          </div>
          <div className="overflow-x-auto">
            <svg width={lineW} height={chartH}>
              {(() => {
                const allPpm = pumpStations.flatMap(s => {
                  const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
                  return schedule.hourly.map(h => h.stations[k]?.dra_ppm || 0);
                });
                const maxPpm = Math.max(...allPpm, 1);
                return pumpStations.map((s, si) => {
                  const k = s.name.trim().toLowerCase().replace(/\s+/g, '_');
                  const vals = schedule.hourly.map(h => h.stations[k]?.dra_ppm || 0);
                  const pts = vals.map((v, i) => {
                    const x = leftPad + (i / Math.max(hours.length - 1, 1)) * (lineW - leftPad - rightPad);
                    const y = chartH - bottomPad - (v / maxPpm) * (chartH - topPad - bottomPad);
                    return `${x},${y}`;
                  });
                  return (
                    <polyline key={s.id} points={pts.join(' ')} fill="none"
                      stroke={COLORS[si % COLORS.length]} strokeWidth={1.5} opacity={0.85}/>
                  );
                });
              })()}
              {hours.map((lbl, i) => (
                i % 4 === 0 && (
                  <text key={i}
                    x={leftPad + (i / Math.max(hours.length - 1, 1)) * (lineW - leftPad - rightPad)}
                    y={chartH - 4} textAnchor="middle" fontSize={8} fill="#6b7280">{lbl}</text>
                )
              ))}
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}
