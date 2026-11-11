'use client';

import React, { useState } from 'react';
import { RollingVolSeries } from '@/lib/quant-engine';

interface RollingVolatilityChartProps {
  series: RollingVolSeries[];
  title?: string;
}

const COLORS = ['#06b6d4', '#DEC084', '#a855f7', '#10b981', '#f43f5e'];

export function RollingVolatilityChart({
  series,
  title = 'Rolling Annualized Volatility (σ × √252)',
}: RollingVolatilityChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!series || series.length === 0 || !series[0].values.length) {
    return (
      <div className="h-60 flex items-center justify-center border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-500">
        No rolling volatility computed
      </div>
    );
  }

  const n = series[0].values.length;
  const dates = series[0].values.map((v) => v.date);

  // Find global min and max vol
  let maxVol = 0;
  for (const s of series) {
    for (const v of s.values) {
      if (v.vol !== null && v.vol > maxVol) maxVol = v.vol;
    }
  }
  maxVol = Math.max(0.2, maxVol * 1.15);

  const width = 800;
  const height = 270;
  const padLeft = 55;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (i: number) => padLeft + (i / Math.max(1, n - 1)) * chartW;
  const getY = (val: number) => padTop + chartH - (val / maxVol) * chartH;

  const activeIndex = hoverIndex !== null ? hoverIndex : n - 1;
  const activeDate = dates[activeIndex];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const clampedX = Math.max(padLeft, Math.min(width - padRight, normX));
    const ratio = (clampedX - padLeft) / chartW;
    const index = Math.round(ratio * (n - 1));
    setHoverIndex(Math.max(0, Math.min(n - 1, index)));
  };

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            {title}
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Standard deviation annualized across user-selected rolling observation windows
          </p>
        </div>

        {/* Legend with active values */}
        <div className="flex flex-wrap items-center gap-4 text-xs">
          {series.map((s, idx) => {
            const color = COLORS[idx % COLORS.length];
            const activeVal = s.values[activeIndex]?.vol;
            return (
              <div key={s.window} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ backgroundColor: color }}
                />
                <span className="text-zinc-500 dark:text-zinc-400 font-medium">
                  {s.window}d:
                </span>
                <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {activeVal !== null && activeVal !== undefined
                    ? `${(activeVal * 100).toFixed(1)}%`
                    : '—'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="relative w-full aspect-[8/2.7] min-h-[200px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
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
                  className="text-zinc-200 dark:text-zinc-800/80"
                  strokeDasharray="2 3"
                />
                <text
                  x={padLeft - 6}
                  y={yPos + 4}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
                >
                  {(val * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}

          {/* Series lines */}
          {series.map((s, idx) => {
            const color = COLORS[idx % COLORS.length];
            const validPoints: string[] = [];

            s.values.forEach((v, i) => {
              if (v.vol !== null) {
                validPoints.push(`${getX(i)},${getY(v.vol)}`);
              }
            });

            if (validPoints.length === 0) return null;

            return (
              <path
                key={s.window}
                d={`M ${validPoints.join(' L ')}`}
                fill="none"
                stroke={color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            );
          })}

          {/* Active hover vertical crosshair */}
          {hoverIndex !== null && (
            <line
              x1={getX(activeIndex)}
              y1={padTop}
              x2={getX(activeIndex)}
              y2={padTop + chartH}
              stroke="#38bdf8"
              strokeWidth="1.2"
              strokeDasharray="2 2"
            />
          )}

          {/* Date ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const i = Math.round(pct * (n - 1));
            return (
              <text
                key={idx}
                x={getX(i)}
                y={height - 6}
                textAnchor="middle"
                className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {dates[i]}
              </text>
            );
          })}
        </svg>
      </div>

      <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400 mt-2">
        <span>Cursor Date: <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">{activeDate}</span></span>
        <span>Annualized Factor: <span className="font-mono">√252 ≈ 15.874</span></span>
      </div>
    </div>
  );
}
