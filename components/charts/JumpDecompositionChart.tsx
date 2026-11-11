'use client';

import React from 'react';
import { DailyRealizedVol } from '@/lib/quant-engine';
import { Zap, AlertTriangle } from 'lucide-react';

interface JumpDecompositionChartProps {
  dailyRV: DailyRealizedVol[];
  ticker: string;
}

export function JumpDecompositionChart({
  dailyRV,
  ticker,
}: JumpDecompositionChartProps) {
  if (!dailyRV || dailyRV.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center border border-slate-200 dark:border-slate-800 rounded-lg text-slate-500 text-xs">
        No realized volatility days computed
      </div>
    );
  }

  const n = dailyRV.length;

  let maxAnnVol = 0;
  for (const d of dailyRV) {
    if (d.annualizedRealizedVol > maxAnnVol) maxAnnVol = d.annualizedRealizedVol;
  }
  maxAnnVol = Math.max(0.2, maxAnnVol * 1.2);

  const width = 800;
  const height = 240;
  const padLeft = 55;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getY = (val: number) => padTop + chartH - (val / maxAnnVol) * chartH;

  const slotW = chartW / n;
  const barW = Math.min(48, Math.max(16, slotW * 0.55));

  // Compute jump statistics
  let totalRv = 0;
  let totalJump = 0;
  let jumpDayCount = 0;

  for (const d of dailyRV) {
    totalRv += d.realizedVariance;
    totalJump += d.jumpComponent;
    if (d.hasSignificantJump) jumpDayCount++;
  }

  const avgJumpRatio = totalRv > 0 ? (totalJump / totalRv) * 100 : 0;

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              Realized Volatility &amp; Jump Decomposition (Barndorff-Nielsen &amp; Shephard)
            </h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · {ticker} (Recent {n} Sessions)
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Disentangles continuous diffusive market variance (BV_t) from discrete jump shocks (J_t)
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div>
            <span className="text-slate-500 dark:text-slate-400">Jump Share:</span>{' '}
            <span className="font-mono font-bold text-amber-500 tabular-nums">
              {avgJumpRatio.toFixed(1)}% of Total RV
            </span>
          </div>
          <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
          <div>
            <span className="text-slate-500 dark:text-slate-400">Significant Jumps (p &lt; 0.05):</span>{' '}
            <span className="font-mono font-bold text-slate-900 dark:text-slate-100 tabular-nums">
              {jumpDayCount} of {n} days
            </span>
          </div>
        </div>
      </div>

      <div className="relative w-full aspect-[8/2.4] min-h-[190px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible">
          {/* Grid lines */}
          {[0, 0.33, 0.66, 1].map((pct) => {
            const val = pct * maxAnnVol;
            const yPos = getY(val);
            return (
              <g key={pct}>
                <line
                  x1={padLeft}
                  y1={yPos}
                  x2={width - padRight}
                  y2={yPos}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800/80"
                  strokeDasharray="2 3"
                />
                <text
                  x={padLeft - 6}
                  y={yPos + 4}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
                >
                  {(val * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}

          {/* Stacked Bars per Day */}
          {dailyRV.map((d, i) => {
            const centerX = padLeft + (i + 0.5) * slotW;
            const x0 = centerX - barW / 2;

            // Continuous diffusion portion (Bipower Variation)
            const contVol = d.annualizedContinuousVol;
            const totalVol = d.annualizedRealizedVol;

            const yTotal = getY(totalVol);
            const yCont = getY(contVol);
            const baseZero = padTop + chartH;

            const hCont = Math.max(0, baseZero - yCont);
            const hJump = Math.max(0, yCont - yTotal);

            return (
              <g key={d.date} className="group cursor-pointer">
                {/* Continuous Diffusion bar bottom */}
                <rect
                  x={x0}
                  y={yCont}
                  width={barW}
                  height={hCont}
                  fill="#0ea5e9"
                  className="opacity-80 group-hover:opacity-100 transition-opacity"
                  rx="1"
                />

                {/* Jump Component bar top */}
                {hJump > 0 && (
                  <rect
                    x={x0}
                    y={yTotal}
                    width={barW}
                    height={hJump}
                    fill={d.hasSignificantJump ? '#f43f5e' : '#f59e0b'}
                    className="opacity-90 group-hover:opacity-100 transition-opacity"
                    rx="1"
                  />
                )}

                {/* Jump Alert Marker on Top if significant */}
                {d.hasSignificantJump && (
                  <g transform={`translate(${centerX - 5}, ${yTotal - 14})`}>
                    <polygon points="5,0 10,9 0,9" fill="#f43f5e" />
                  </g>
                )}

                {/* Total Vol readout on top of bar */}
                <text
                  x={centerX}
                  y={yTotal - 4}
                  textAnchor="middle"
                  className="text-[9px] font-mono fill-slate-600 dark:fill-slate-400 font-bold tabular-nums opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  {(totalVol * 100).toFixed(1)}%
                </text>

                {/* Date on X axis */}
                <text
                  x={centerX}
                  y={height - 8}
                  textAnchor="middle"
                  className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
                >
                  {d.date.slice(5)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-500 dark:text-slate-400 mt-2 pt-2 border-t border-slate-100 dark:border-slate-800/60 gap-2">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-sky-500 inline-block" />
            Continuous Diffusion (Bipower Variation BV_t)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-amber-500 inline-block" />
            Normal Jump Noise
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-rose-500 inline-block" />
            Significant Jump Anomaly (Z &gt; 1.96)
          </span>
        </div>
        <div className="flex items-center gap-1 font-mono text-[11px]">
          <span>Bars/Day: ~390 mins</span>
        </div>
      </div>
    </div>
  );
}
