'use client';

import React, { useState } from 'react';

interface ReturnsChartProps {
  dates: string[];
  returns: number[];
  returnType: 'simple' | 'log';
  title?: string;
}

export function ReturnsChart({
  dates,
  returns,
  returnType,
  title = 'Daily Returns Trajectory',
}: ReturnsChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!returns || returns.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-500">
        No return series calculated
      </div>
    );
  }

  const alignedDates = dates.slice(1);
  const n = returns.length;

  let maxAbs = 0;
  for (const r of returns) {
    const abs = Math.abs(r);
    if (abs > maxAbs) maxAbs = abs;
  }
  maxAbs = Math.max(0.02, maxAbs * 1.1);

  const width = 800;
  const height = 260;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 15;
  const padBottom = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const zeroY = padTop + chartH / 2;

  const getX = (i: number) => padLeft + (i / Math.max(1, n - 1)) * chartW;
  const getY = (val: number) => padTop + chartH / 2 - (val / maxAbs) * (chartH / 2);

  const activeIndex = hoverIndex !== null ? hoverIndex : n - 1;
  const activeRet = returns[activeIndex];
  const activeDate = alignedDates[activeIndex];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const clampedX = Math.max(padLeft, Math.min(width - padRight, normX));
    const ratio = (clampedX - padLeft) / chartW;
    const index = Math.round(ratio * (n - 1));
    setHoverIndex(Math.max(0, Math.min(n - 1, index)));
  };

  const barW = Math.max(1, chartW / n - 0.2);

  return (
    <div className="panel p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
            {title}
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            {returnType === 'log' ? 'Continuously compounded log returns r_t = ln(P_t/P_{t-1})' : 'Simple percentage returns R_t = (P_t - P_{t-1})/P_{t-1}'}
          </p>
        </div>

        {activeDate && (
          <div className="text-right">
            <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1.5">{activeDate}:</span>
            <span
              className={`font-mono text-sm font-bold tabular-nums ${
                activeRet >= 0 ? 'text-gold-500' : 'text-rose-500'
              }`}
            >
              {(activeRet * 100).toFixed(2)}%
            </span>
          </div>
        )}
      </div>

      <div className="relative w-full aspect-[8/2.6] min-h-[190px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {/* Zero center axis line */}
          <line
            x1={padLeft}
            y1={zeroY}
            x2={width - padRight}
            y2={zeroY}
            stroke="currentColor"
            className="text-zinc-300 dark:text-zinc-700"
            strokeWidth="1.2"
          />

          {/* Plus / minus reference lines */}
          {[-maxAbs * 0.5, maxAbs * 0.5].map((val) => (
            <g key={val}>
              <line
                x1={padLeft}
                y1={getY(val)}
                x2={width - padRight}
                y2={getY(val)}
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800/80"
                strokeDasharray="2 3"
                strokeWidth="1"
              />
              <text
                x={padLeft - 8}
                y={getY(val) + 3}
                textAnchor="end"
                className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {(val * 100).toFixed(1)}%
              </text>
            </g>
          ))}

          <text
            x={padLeft - 8}
            y={zeroY + 3}
            textAnchor="end"
            className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
          >
            0.0%
          </text>

          {/* Return bars */}
          {returns.map((r, i) => {
            const x = getX(i);
            const y = getY(r);
            const isPos = r >= 0;
            const top = isPos ? y : zeroY;
            const h = Math.max(1, Math.abs(y - zeroY));

            return (
              <rect
                key={i}
                x={x - barW / 2}
                y={top}
                width={barW}
                height={h}
                fill={isPos ? '#10b981' : '#f43f5e'}
                opacity={hoverIndex === i ? 1 : 0.75}
              />
            );
          })}

          {/* Hover highlight line */}
          {hoverIndex !== null && (
            <line
              x1={getX(activeIndex)}
              y1={padTop}
              x2={getX(activeIndex)}
              y2={padTop + chartH}
              stroke="#38bdf8"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          )}

          {/* Date labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
            const i = Math.round(pct * (n - 1));
            return (
              <text
                key={idx}
                x={getX(i)}
                y={height - 5}
                textAnchor="middle"
                className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {alignedDates[i]}
              </text>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
