'use client';

import React, { useState, useRef } from 'react';
import { PriceBar } from '@/lib/quant-engine';

interface TimeSeriesChartProps {
  bars: PriceBar[];
  title?: string;
  subtitle?: string;
  showVolume?: boolean;
  valueFormatter?: (val: number) => string;
}

export function TimeSeriesChart({
  bars,
  title = 'Historical Price',
  subtitle,
  showVolume = true,
  valueFormatter = (v) => `$${v.toFixed(2)}`,
}: TimeSeriesChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (!bars || bars.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-slate-200 dark:border-slate-800 rounded-lg text-slate-500">
        No price data available
      </div>
    );
  }

  // Find min/max for scaling
  let minPrice = Infinity;
  let maxPrice = -Infinity;
  let maxVol = 0;

  for (const b of bars) {
    if (b.close < minPrice) minPrice = b.close;
    if (b.close > maxPrice) maxPrice = b.close;
    if (b.volume > maxVol) maxVol = b.volume;
  }

  // Pad price range by 3%
  const pricePadding = (maxPrice - minPrice) * 0.04 || 1;
  const yMin = Math.max(0, minPrice - pricePadding);
  const yMax = maxPrice + pricePadding;

  const width = 800;
  const height = 340;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 20;
  const padBottom = showVolume ? 70 : 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const volH = showVolume ? 45 : 0;

  const getX = (i: number) => padLeft + (i / Math.max(1, bars.length - 1)) * chartW;
  const getY = (val: number) => padTop + chartH - ((val - yMin) / (yMax - yMin)) * chartH;
  const getVolY = (vol: number) => height - 15 - (vol / (maxVol || 1)) * volH;

  // Build SVG path for price line and area
  const points = bars.map((b, i) => `${getX(i)},${getY(b.close)}`).join(' ');
  const areaPath = `M ${getX(0)},${padTop + chartH} L ${points} L ${getX(
    bars.length - 1
  )},${padTop + chartH} Z`;

  // Determine trend color
  const isUp = bars[bars.length - 1].close >= bars[0].close;
  const strokeColor = isUp ? '#10b981' : '#f43f5e';
  const fillColor = isUp ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)';

  // Hovered item
  const activeBar = hoverIndex !== null && bars[hoverIndex] ? bars[hoverIndex] : bars[bars.length - 1];
  const activeIndex = hoverIndex !== null ? hoverIndex : bars.length - 1;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const clampedX = Math.max(padLeft, Math.min(width - padRight, normX));
    const ratio = (clampedX - padLeft) / chartW;
    const index = Math.round(ratio * (bars.length - 1));
    setHoverIndex(Math.max(0, Math.min(bars.length - 1, index)));
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
  };

  // Y-axis grid ticks (4 ticks)
  const yTicks = [0, 0.33, 0.66, 1].map((pct) => {
    const val = yMin + pct * (yMax - yMin);
    const yPos = getY(val);
    return { val, yPos };
  });

  // X-axis date labels (5 dates)
  const xLabelsCount = 5;
  const xLabels = Array.from({ length: xLabelsCount }, (_, idx) => {
    const i = Math.round((idx / (xLabelsCount - 1)) * (bars.length - 1));
    return { date: bars[i].date, xPos: getX(i) };
  });

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              {title}
            </h3>
            {subtitle && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                · {subtitle}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Historical trajectory across {bars.length} observations
          </p>
        </div>

        {activeBar && (
          <div className="flex items-baseline gap-3 text-right">
            <div>
              <span className="text-xs text-slate-500 dark:text-slate-400 mr-1.5">
                {activeBar.date}:
              </span>
              <span className="font-mono text-base font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                {valueFormatter(activeBar.close)}
              </span>
            </div>
            {showVolume && (
              <div className="text-xs text-slate-500 dark:text-slate-400">
                Vol: <span className="font-mono tabular-nums">{(activeBar.volume / 1e6).toFixed(2)}M</span>
              </div>
            )}
          </div>
        )}
      </div>

      <div ref={containerRef} className="relative w-full aspect-[8/3.4] min-h-[220px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={isUp ? '#10b981' : '#f43f5e'} stopOpacity="0.25" />
              <stop offset="100%" stopColor={isUp ? '#10b981' : '#f43f5e'} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {yTicks.map(({ val, yPos }) => (
            <g key={val}>
              <line
                x1={padLeft}
                y1={yPos}
                x2={width - padRight}
                y2={yPos}
                stroke="currentColor"
                className="text-slate-200 dark:text-slate-800"
                strokeDasharray="3 3"
                strokeWidth="1"
              />
              <text
                x={padLeft - 8}
                y={yPos + 4}
                textAnchor="end"
                className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
              >
                {val.toFixed(1)}
              </text>
            </g>
          ))}

          {/* Volume bars (subtle bottom strip) */}
          {showVolume &&
            bars.map((b, i) => {
              const x = getX(i);
              const y = getVolY(b.volume);
              const barW = Math.max(1, chartW / bars.length - 0.5);
              const isBarUp = i === 0 || b.close >= bars[i - 1].close;
              return (
                <rect
                  key={b.date + i}
                  x={x - barW / 2}
                  y={y}
                  width={barW}
                  height={height - 15 - y}
                  fill={isBarUp ? '#10b981' : '#f43f5e'}
                  opacity={hoverIndex === i ? 0.8 : 0.3}
                />
              );
            })}

          {/* Price Area & Line */}
          <path d={areaPath} fill="url(#priceGradient)" />
          <path
            d={`M ${points}`}
            fill="none"
            stroke={strokeColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* X Axis dates */}
          {xLabels.map(({ date, xPos }) => (
            <text
              key={date}
              x={xPos}
              y={height - 2}
              textAnchor="middle"
              className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
            >
              {date}
            </text>
          ))}

          {/* Active crosshair vertical line and point */}
          {activeBar && (
            <g>
              <line
                x1={getX(activeIndex)}
                y1={padTop}
                x2={getX(activeIndex)}
                y2={padTop + chartH}
                stroke="#38bdf8"
                strokeWidth="1.2"
                strokeDasharray="2 2"
              />
              <circle
                cx={getX(activeIndex)}
                cy={getY(activeBar.close)}
                r="4"
                fill="#38bdf8"
                stroke="#0f172a"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
