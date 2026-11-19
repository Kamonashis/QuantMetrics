'use client';

import React, { useState } from 'react';
import { RollingVolSeries } from '@/lib/quant-engine';
import { useChartSize } from '@/lib/use-chart-size';
import { SERIES_COLORS as COLORS } from '@/lib/chart-palette';

export interface ReferenceLine {
  /** Horizontal level in the same units as the series (annualized decimal). */
  value: number;
  /** Right-edge + legend label, e.g. '1.25×IV30 35.8%'. */
  label: string;
}

interface RollingVolatilityChartProps {
  series: RollingVolSeries[];
  title?: string;
  /** Optional horizontal reference levels (e.g. IV30, breakeven). Absent by default — call sites opt in. */
  referenceLines?: ReferenceLine[];
}

export function RollingVolatilityChart({
  series,
  title = 'Rolling Annualized Volatility (σ × √252)',
  referenceLines = [],
}: RollingVolatilityChartProps) {
  const { ref, size } = useChartSize<HTMLDivElement>();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!series || series.length === 0 || !series[0].values.length) {
    return (
      <div className="h-60 flex items-center justify-center border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-500">
        No rolling volatility computed
      </div>
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;

  const n = series[0].values.length;
  const dates = series[0].values.map((v) => v.date);

  // Find global min and max vol (reference lines expand the scale so they are never clipped)
  let maxVol = 0;
  for (const s of series) {
    for (const v of s.values) {
      if (v.vol !== null && v.vol > maxVol) maxVol = v.vol;
    }
  }
  for (const rl of referenceLines) {
    if (rl.value > maxVol) maxVol = rl.value;
  }
  maxVol = Math.max(0.2, maxVol * 1.15);

  const padLeft = 46;
  const padRight = 14;
  const padTop = 16;
  const padBottom = 28;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  // Density-adaptive x labels: 3 on xs, 5 otherwise
  const xTickPcts = width < 640 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];

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
          {referenceLines.map((rl) => (
            <div key={rl.label} className="flex items-center gap-1.5">
              <span className="inline-block h-0 w-4 border-t-2 border-dashed border-gold-500" />
              <span className="text-zinc-500 dark:text-zinc-400 font-medium">{rl.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div ref={ref} className="chart-frame">
        {ready && (
        <svg
          width={width}
          height={height}
          className="block h-full w-full cursor-crosshair"
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
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            );
          })}

          {/* Reference levels (e.g. IV30, breakeven) */}
          {referenceLines.map((rl, i) => {
            const y = getY(rl.value);
            return (
              <g key={i}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="#DEC084"
                  strokeWidth="1.2"
                  strokeDasharray="6 4"
                  opacity="0.9"
                />
                <text
                  x={width - padRight - 2}
                  y={y - 3}
                  textAnchor="end"
                  className="text-[8px] sm:text-[9px] font-mono fill-gold-700 dark:fill-gold-300 tabular-nums"
                >
                  {rl.label}
                </text>
              </g>
            );
          })}

          {/* Active hover vertical crosshair */}
          {hoverIndex !== null && (
            <line
              x1={getX(activeIndex)}
              y1={padTop}
              x2={getX(activeIndex)}
              y2={padTop + chartH}
              stroke="#8b8b93"
              strokeWidth="0.8"
              strokeDasharray="2 2"
            />
          )}

          {/* Date ticks */}
          {xTickPcts.map((pct, idx) => {
            const i = Math.round(pct * (n - 1));
            return (
              <text
                key={idx}
                x={getX(i)}
                y={height - 8}
                textAnchor={idx === 0 ? 'start' : idx === xTickPcts.length - 1 ? 'end' : 'middle'}
                className="text-[9px] sm:text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {dates[i]}
              </text>
            );
          })}
        </svg>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400 mt-2">
        <span>Cursor Date: <span className="font-mono text-zinc-800 dark:text-zinc-200 font-semibold">{activeDate}</span></span>
        <span>Annualized Factor: <span className="font-mono">√252 ≈ 15.874</span></span>
      </div>
    </div>
  );
}
