'use client';

import React, { useState } from 'react';
import { DiurnalPoint } from '@/lib/quant-engine';

interface IntradayDiurnalChartProps {
  points: DiurnalPoint[];
  ticker: string;
}

export function IntradayDiurnalChart({
  points,
  ticker,
}: IntradayDiurnalChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!points || points.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-slate-200 dark:border-slate-800 rounded-lg text-slate-500 text-xs">
        No intraday 1-minute observations available
      </div>
    );
  }

  const n = points.length;

  let maxVol = 0;
  for (const p of points) {
    if (p.upperBand > maxVol) maxVol = p.upperBand;
  }
  maxVol = Math.max(0.15, maxVol * 1.15);

  const width = 800;
  const height = 280;
  const padLeft = 55;
  const padRight = 25;
  const padTop = 20;
  const padBottom = 35;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (m: number) => padLeft + (m / Math.max(1, n - 1)) * chartW;
  const getY = (val: number) => padTop + chartH - (val / maxVol) * chartH;

  // Build SVG path for confidence band and mean line
  const upperPts: string[] = [];
  const lowerPts: string[] = [];
  const meanPts: string[] = [];

  points.forEach((p) => {
    const x = getX(p.minuteOfDay);
    upperPts.push(`${x},${getY(p.upperBand)}`);
    lowerPts.push(`${x},${getY(p.lowerBand)}`);
    meanPts.push(`${x},${getY(p.avgAnnualizedVol)}`);
  });

  const bandArea = `M ${upperPts.join(' L ')} L ${lowerPts.reverse().join(' L ')} Z`;

  const activeIndex = hoverIndex !== null ? hoverIndex : Math.floor(n / 2);
  const activePt = points[activeIndex];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const clampedX = Math.max(padLeft, Math.min(width - padRight, normX));
    const ratio = (clampedX - padLeft) / chartW;
    const idx = Math.round(ratio * (n - 1));
    setHoverIndex(Math.max(0, Math.min(n - 1, idx)));
  };

  // Specific time marker lines
  const keyTimes = [
    { minute: 0, label: '09:30 Open' },
    { minute: 120, label: '11:30 EU Close' },
    { minute: 210, label: '13:00 Lunch Lull' },
    { minute: 360, label: '15:30 Power Hr' },
    { minute: 389, label: '16:00 MOC' },
  ];

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              Intraday Diurnal U-Shape Volatility Smile
            </h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · {ticker} (390 Trading Minutes)
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Empirical minute-by-minute volatility profile showing opening price discovery, midday lull, and market-on-close auction
          </p>
        </div>

        {activePt && (
          <div className="text-right">
            <span className="text-xs text-slate-500 dark:text-slate-400 mr-1.5">
              Time {activePt.timeLabel}:
            </span>
            <span className="font-mono text-sm font-bold text-sky-500 tabular-nums">
              {(activePt.avgAnnualizedVol * 100).toFixed(1)}% Vol
            </span>
          </div>
        )}
      </div>

      <div className="relative w-full aspect-[8/2.8] min-h-[200px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id="diurnalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.04" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.33, 0.66, 1].map((pct) => {
            const val = pct * maxVol;
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

          {/* Key session markers */}
          {keyTimes.map(({ minute, label }) => {
            const x = getX(minute);
            return (
              <g key={minute}>
                <line
                  x1={x}
                  y1={padTop}
                  x2={x}
                  y2={padTop + chartH}
                  stroke="currentColor"
                  className="text-slate-300 dark:text-slate-700/80"
                  strokeDasharray="3 3"
                  strokeWidth="0.8"
                />
                <text
                  x={x}
                  y={padTop + 12}
                  textAnchor="middle"
                  className="text-[9px] font-mono fill-slate-400 dark:fill-slate-500"
                >
                  {label}
                </text>
              </g>
            );
          })}

          {/* Confidence Band Fill */}
          <path d={bandArea} fill="url(#diurnalGradient)" />

          {/* Mean Curve Line */}
          <path
            d={`M ${meanPts.join(' L ')}`}
            fill="none"
            stroke="#0ea5e9"
            strokeWidth="2.2"
            strokeLinecap="round"
          />

          {/* Hover crosshair */}
          {hoverIndex !== null && (
            <g>
              <line
                x1={getX(activePt.minuteOfDay)}
                y1={padTop}
                x2={getX(activePt.minuteOfDay)}
                y2={padTop + chartH}
                stroke="#38bdf8"
                strokeWidth="1.2"
                strokeDasharray="2 2"
              />
              <circle
                cx={getX(activePt.minuteOfDay)}
                cy={getY(activePt.avgAnnualizedVol)}
                r="4"
                fill="#38bdf8"
                stroke="#0f172a"
                strokeWidth="2"
              />
            </g>
          )}

          {/* X axis labels */}
          <text x={getX(0)} y={height - 8} textAnchor="start" className="text-[10px] font-mono fill-slate-500">
            09:30 AM
          </text>
          <text x={getX(195)} y={height - 8} textAnchor="middle" className="text-[10px] font-mono fill-slate-500">
            12:45 PM (Midday Lull)
          </text>
          <text x={getX(389)} y={height - 8} textAnchor="end" className="text-[10px] font-mono fill-slate-500">
            16:00 PM (Close)
          </text>
        </svg>
      </div>

      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mt-2 pt-2 border-t border-slate-100 dark:border-slate-800/60">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-sky-500 inline-block" />
            Mean 1-Min Annualized Volatility
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-sky-500/20 inline-block" />
            ±1σ Diurnal Dispersion Band
          </span>
        </div>
        <div>
          <span>U-Shape Peak Ratio: </span>
          <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
            {((points[0]?.avgAnnualizedVol || 0.25) / (points[210]?.avgAnnualizedVol || 0.12)).toFixed(1)}x Open vs Midday
          </span>
        </div>
      </div>
    </div>
  );
}
