'use client';

import React, { useState } from 'react';
import { CorrelationMatrixResult } from '@/lib/quant-engine';

interface CorrelationHeatmapProps {
  result: CorrelationMatrixResult;
  returnsByTicker: { [ticker: string]: number[] };
}

export function CorrelationHeatmap({
  result,
  returnsByTicker,
}: CorrelationHeatmapProps) {
  const [selectedPair, setSelectedPair] = useState<{ tA: string; tB: string } | null>(null);

  const { tickers, matrix, avgCorrelation, strongestPair, weakestPair } = result;

  if (!tickers || tickers.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-slate-200 dark:border-slate-800 rounded-lg text-slate-500">
        No correlation matrix available. Input multiple tickers.
      </div>
    );
  }

  // Color interpolator for correlation:
  // -1.0 -> Rose (#f43f5e)
  //  0.0 -> Slate (#64748b / transparent neutral)
  // +1.0 -> Emerald (#10b981) or Sky (#0ea5e9)
  const getCellBg = (corr: number) => {
    if (corr >= 0) {
      // 0 to 1 -> sky/emerald
      const alpha = Math.min(0.9, Math.max(0.08, corr * 0.85));
      return `rgba(14, 165, 233, ${alpha})`;
    } else {
      // -1 to 0 -> rose
      const alpha = Math.min(0.9, Math.max(0.08, Math.abs(corr) * 0.85));
      return `rgba(244, 63, 94, ${alpha})`;
    }
  };

  const getTextColor = (corr: number) => {
    if (Math.abs(corr) > 0.45) return '#ffffff';
    return undefined; // inherit
  };

  // Compute scatter data if pair selected
  const activePair = selectedPair || (tickers.length >= 2 ? { tA: tickers[0], tB: tickers[1] } : null);
  const pairReturnsA = activePair ? returnsByTicker[activePair.tA] || [] : [];
  const pairReturnsB = activePair ? returnsByTicker[activePair.tB] || [] : [];

  const scatterLen = Math.min(pairReturnsA.length, pairReturnsB.length);
  const scatterPoints: { x: number; y: number }[] = [];
  for (let i = 0; i < scatterLen; i++) {
    scatterPoints.push({ x: pairReturnsA[i], y: pairReturnsB[i] });
  }

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              Cross-Asset Return Correlation Matrix
            </h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · {tickers.length} Assets Synchronized
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Click any cell to inspect the pairwise co-movement scatter and regression slope
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div>
            <span className="text-slate-500 dark:text-slate-400">Avg Correlation:</span>{' '}
            <span className="font-mono font-bold text-slate-900 dark:text-slate-100 tabular-nums">
              {avgCorrelation.toFixed(2)}
            </span>
          </div>
          <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
          <div>
            <span className="text-slate-500 dark:text-slate-400">Max Co-movement:</span>{' '}
            <span className="font-mono font-bold text-sky-600 dark:text-sky-400 tabular-nums">
              {strongestPair.tickerA}/{strongestPair.tickerB} ({strongestPair.corr.toFixed(2)})
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap Matrix Table */}
        <div className="lg:col-span-7 overflow-x-auto">
          <div className="inline-block min-w-full align-middle">
            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-800 border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden">
              <thead className="bg-slate-50 dark:bg-slate-900/80">
                <tr>
                  <th className="py-2.5 px-3 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Asset
                  </th>
                  {tickers.map((t) => (
                    <th
                      key={t}
                      className="py-2.5 px-3 text-center text-xs font-mono font-bold text-slate-700 dark:text-slate-300"
                    >
                      {t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono text-xs">
                {tickers.map((rowTicker, i) => (
                  <tr key={rowTicker} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30">
                    <td className="py-2.5 px-3 whitespace-nowrap font-bold text-slate-900 dark:text-slate-100 bg-slate-50/50 dark:bg-slate-900/40">
                      {rowTicker}
                    </td>
                    {tickers.map((colTicker, j) => {
                      const corr = matrix[i]?.[j] ?? (i === j ? 1 : 0);
                      const isSelf = i === j;
                      const isSelected =
                        activePair &&
                        ((activePair.tA === rowTicker && activePair.tB === colTicker) ||
                          (activePair.tA === colTicker && activePair.tB === rowTicker));

                      return (
                        <td
                          key={colTicker}
                          onClick={() => !isSelf && setSelectedPair({ tA: rowTicker, tB: colTicker })}
                          style={{
                            backgroundColor: getCellBg(corr),
                            color: getTextColor(corr),
                          }}
                          className={`py-2.5 px-3 text-center font-bold tabular-nums cursor-pointer transition-all ${
                            isSelected
                              ? 'ring-2 ring-sky-400 font-extrabold z-10 scale-[1.03]'
                              : 'hover:opacity-90'
                          } ${isSelf ? 'cursor-default font-semibold opacity-60' : ''}`}
                        >
                          {corr.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mt-3">
            <div className="flex items-center gap-2">
              <span className="text-[11px]">Negative Correlation</span>
              <span className="w-12 h-2.5 rounded-xs bg-rose-500 inline-block" />
              <span className="w-12 h-2.5 rounded-xs bg-slate-400 inline-block" />
              <span className="w-12 h-2.5 rounded-xs bg-sky-500 inline-block" />
              <span className="text-[11px]">Positive Correlation</span>
            </div>
            <span>Least Correlated: <span className="font-mono text-rose-500">{weakestPair.tickerA}/{weakestPair.tickerB} ({weakestPair.corr.toFixed(2)})</span></span>
          </div>
        </div>

        {/* Pairwise Scatter View */}
        <div className="lg:col-span-5 bg-slate-50 dark:bg-slate-900/60 p-4 rounded-lg border border-slate-200 dark:border-slate-800">
          {activePair ? (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  {activePair.tA} vs {activePair.tB} Return Scatter
                </span>
                <span className="text-xs font-mono text-sky-600 dark:text-sky-400 font-bold tabular-nums">
                  r = {matrix[tickers.indexOf(activePair.tA)]?.[tickers.indexOf(activePair.tB)]?.toFixed(3)}
                </span>
              </div>

              {/* Mini Scatter Plot */}
              <div className="w-full aspect-[4/3] bg-white dark:bg-[#0c1222] rounded-md border border-slate-200 dark:border-slate-800 relative p-2">
                <svg viewBox="0 0 300 220" className="w-full h-full overflow-visible">
                  {/* Axes */}
                  <line x1="30" y1="110" x2="280" y2="110" stroke="#64748b" strokeWidth="0.8" strokeDasharray="2 2" />
                  <line x1="155" y1="20" x2="155" y2="200" stroke="#64748b" strokeWidth="0.8" strokeDasharray="2 2" />

                  {/* Scatter points */}
                  {scatterPoints.slice(0, 180).map((pt, idx) => {
                    const cx = 155 + pt.x * 2000;
                    const cy = 110 - pt.y * 2000;
                    return (
                      <circle
                        key={idx}
                        cx={Math.max(35, Math.min(275, cx))}
                        cy={Math.max(25, Math.min(195, cy))}
                        r="2.5"
                        fill="#38bdf8"
                        opacity="0.65"
                      />
                    );
                  })}

                  {/* Axis labels */}
                  <text x="280" y="105" textAnchor="end" className="text-[9px] font-mono fill-slate-500">
                    {activePair.tA} Return
                  </text>
                  <text x="160" y="25" textAnchor="start" className="text-[9px] font-mono fill-slate-500">
                    {activePair.tB} Return
                  </text>
                </svg>
              </div>

              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
                Slope reflects sensitivity between {activePair.tA} and {activePair.tB}. High clustering along 45° indicates systemic co-movement.
              </p>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-slate-400">
              Select an asset pair to view return scatter
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
