'use client';

import React, { useState, useRef, useMemo } from 'react';
import { PriceBar } from '@/lib/quant-engine';

export interface StructuralBreakMarker {
  date: string;
  index?: number;
  reason: string;
  severity?: 'High' | 'Moderate' | 'Low' | string;
}

export interface CusumSqPoint {
  date: string;
  s: number;
  upperBand: number;
  lowerBand: number;
}

interface TimeSeriesChartProps {
  bars: PriceBar[];
  title?: string;
  subtitle?: string;
  showVolume?: boolean;
  valueFormatter?: (val: number) => string;
  breakPoints?: StructuralBreakMarker[];
  cusumSqPoints?: CusumSqPoint[];
}

export function TimeSeriesChart({
  bars,
  title = 'Historical Price',
  subtitle,
  showVolume = true,
  valueFormatter = (v) => `$${v.toFixed(2)}`,
  breakPoints = [],
  cusumSqPoints = [],
}: TimeSeriesChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [showBreaks, setShowBreaks] = useState<boolean>(true);
  const [showCusumCurve, setShowCusumCurve] = useState<boolean>(false);

  const validBars = useMemo(() => bars || [], [bars]);

  // Find min/max for scaling
  let minPrice = Infinity;
  let maxPrice = -Infinity;
  let maxVol = 0;

  for (const b of validBars) {
    if (b.close < minPrice) minPrice = b.close;
    if (b.close > maxPrice) maxPrice = b.close;
    if (b.volume > maxVol) maxVol = b.volume;
  }

  // Pad price range by 4%
  const pricePadding = (maxPrice - minPrice) * 0.04 || 1;
  const yMin = Math.max(0, minPrice - pricePadding);
  const yMax = maxPrice + pricePadding;

  const width = 800;
  const height = 340;
  const padLeft = 60;
  const padRight = 20;
  const padTop = 24;
  const padBottom = showVolume ? 70 : 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;
  const volH = showVolume ? 45 : 0;

  const getX = (i: number) => padLeft + (i / Math.max(1, validBars.length - 1)) * chartW;
  const getY = (val: number) => padTop + chartH - ((val - yMin) / (yMax - yMin || 1)) * chartH;
  const getVolY = (vol: number) => height - 15 - (vol / (maxVol || 1)) * volH;

  // Build SVG path for price line and area
  const points = validBars.map((b, i) => `${getX(i)},${getY(b.close)}`).join(' ');
  const areaPath = validBars.length > 0 ? `M ${getX(0)},${padTop + chartH} L ${points} L ${getX(
    validBars.length - 1
  )},${padTop + chartH} Z` : '';

  // Determine trend color
  const isUp = validBars.length > 0 && validBars[validBars.length - 1].close >= validBars[0].close;
  const strokeColor = isUp ? '#10b981' : '#f43f5e';

  // Hovered item
  const activeBar = hoverIndex !== null && validBars[hoverIndex] ? validBars[hoverIndex] : validBars[validBars.length - 1];
  const activeIndex = hoverIndex !== null ? hoverIndex : validBars.length - 1;

  // Map break points to actual bar coordinates
  const resolvedBreakPoints = useMemo(() => {
    if (!breakPoints || breakPoints.length === 0 || validBars.length === 0) return [];
    return breakPoints
      .map((bp) => {
        let idx = validBars.findIndex((b) => b.date === bp.date);
        if (idx === -1 && bp.index !== undefined && bp.index >= 0 && bp.index < validBars.length) {
          idx = bp.index;
        }
        if (idx === -1) {
          // Find closest date within bars
          let closestDist = Infinity;
          let closestIdx = -1;
          const targetTime = new Date(bp.date).getTime();
          for (let i = 0; i < validBars.length; i++) {
            const t = new Date(validBars[i].date).getTime();
            const dist = Math.abs(t - targetTime);
            if (dist < closestDist) {
              closestDist = dist;
              closestIdx = i;
            }
          }
          idx = closestIdx;
        }
        if (idx === -1) return null;
        const xPos = padLeft + (idx / Math.max(1, validBars.length - 1)) * chartW;
        const yPos = padTop + chartH - ((validBars[idx].close - yMin) / (yMax - yMin || 1)) * chartH;
        return {
          ...bp,
          index: idx,
          bar: validBars[idx],
          x: xPos,
          y: yPos,
        };
      })
      .filter((bp): bp is NonNullable<typeof bp> => bp !== null);
  }, [breakPoints, validBars, chartW, padLeft, yMin, yMax, chartH, padTop]);

  // Check if active hover bar is near a break point (within 3 bars)
  const nearbyBreakPoint = useMemo(() => {
    if (!showBreaks || resolvedBreakPoints.length === 0 || hoverIndex === null) return null;
    return resolvedBreakPoints.find((bp) => Math.abs(bp.index - hoverIndex) <= 2) || null;
  }, [showBreaks, resolvedBreakPoints, hoverIndex]);

  if (!bars || bars.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-500">
        No price data available
      </div>
    );
  }

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
    <div className="panel p-5 space-y-4">
      {/* Top Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              {title}
            </h3>
            {subtitle && (
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                · {subtitle}
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Historical trajectory across {bars.length} observations
            {resolvedBreakPoints.length > 0 && (
              <span className="text-gold-600 dark:text-gold-400 font-medium">
                {' '}· {resolvedBreakPoints.length} CUSUM regime shift(s) mapped
              </span>
            )}
          </p>
        </div>

        {/* Feature Toggles & Active Price */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {resolvedBreakPoints.length > 0 && (
            <button
              type="button"
              onClick={() => setShowBreaks((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium transition-colors cursor-pointer ${
                showBreaks
                  ? 'bg-gold-500/15 text-gold-700 dark:text-gold-300 border border-gold-500/40 shadow-xs'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-transparent hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
              title="Toggle volatility regime shift break lines on chart"
            >
              <span className={`w-2 h-2 rounded-full ${showBreaks ? 'bg-gold-500 animate-pulse' : 'bg-zinc-400'}`} />
              <span>Break Points ({resolvedBreakPoints.length})</span>
            </button>
          )}

          {cusumSqPoints.length > 0 && (
            <button
              type="button"
              onClick={() => setShowCusumCurve((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono font-medium transition-colors cursor-pointer ${
                showCusumCurve
                  ? 'bg-gold-500/15 text-gold-700 dark:text-gold-300 border border-gold-500/40 shadow-xs'
                  : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border border-transparent hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
              title="Toggle CUSUM of squares diagnostic curve"
            >
              <span>📊 CUSUM S_k Band</span>
            </button>
          )}

          {activeBar && (
            <div className="flex items-baseline gap-2.5 text-right pl-1">
              <div>
                <span className="text-xs text-zinc-500 dark:text-zinc-400 mr-1.5">
                  {activeBar.date}:
                </span>
                <span className="font-mono text-sm sm:text-base font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {valueFormatter(activeBar.close)}
                </span>
              </div>
              {showVolume && (
                <div className="hidden sm:block text-xs text-zinc-500 dark:text-zinc-400">
                  Vol: <span className="font-mono tabular-nums">{(activeBar.volume / 1e6).toFixed(2)}M</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Break Point Alert Callout on Hover */}
      {nearbyBreakPoint && (
        <div className="bg-gold-500/10 border border-gold-500/40 rounded-lg px-3.5 py-2 text-xs flex items-center justify-between text-gold-800 dark:text-gold-200 animate-in fade-in duration-150">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-gold-500 animate-ping" />
            <span className="font-semibold">Regime Break Point:</span>
            <span className="font-mono font-bold">{nearbyBreakPoint.bar.date}</span>
            <span className="text-zinc-400">|</span>
            <span>{nearbyBreakPoint.reason}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono font-medium">Price: {valueFormatter(nearbyBreakPoint.bar.close)}</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-gold-500/20 text-gold-700 dark:text-gold-300 font-mono">
              {nearbyBreakPoint.severity || 'Significant'}
            </span>
          </div>
        </div>
      )}

      {/* Primary Price SVG Chart */}
      <div ref={containerRef} className="relative w-full aspect-[8/3.4] min-h-[240px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible select-none"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={isUp ? '#10b981' : '#f43f5e'} stopOpacity="0.22" />
              <stop offset="100%" stopColor={isUp ? '#10b981' : '#f43f5e'} stopOpacity="0.0" />
            </linearGradient>

            <filter id="breakGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" />
            </filter>
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
                className="text-zinc-200 dark:text-zinc-800/80"
                strokeDasharray="3 3"
                strokeWidth="1"
              />
              <text
                x={padLeft - 8}
                y={yPos + 4}
                textAnchor="end"
                className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {val.toFixed(1)}
              </text>
            </g>
          ))}

          {/* Volatility Break Regime Shifts (Vertical Indicators & Shading) */}
          {showBreaks &&
            resolvedBreakPoints.map((bp, bpIdx) => {
              const isHovered = nearbyBreakPoint?.bar.date === bp.bar.date;
              return (
                <g key={`break-${bp.bar.date}-${bpIdx}`} className="transition-opacity">
                  {/* Subtle vertical glow stripe */}
                  <rect
                    x={bp.x - 14}
                    y={padTop}
                    width={28}
                    height={chartH}
                    fill="#DEC084"
                    fillOpacity={isHovered ? 0.12 : 0.05}
                    className="pointer-events-none"
                  />

                  {/* Vertical dashed break line */}
                  <line
                    x1={bp.x}
                    y1={padTop}
                    x2={bp.x}
                    y2={padTop + chartH}
                    stroke="#DEC084"
                    strokeWidth={isHovered ? 2.2 : 1.6}
                    strokeDasharray="5 3"
                  />

                  {/* Top indicator tag badge */}
                  <g
                    transform={`translate(${bp.x}, ${padTop - 12})`}
                    className="cursor-pointer"
                  >
                    <rect
                      x="-38"
                      y="0"
                      width="76"
                      height="16"
                      rx="8"
                      fill="#78350f"
                      stroke="#DEC084"
                      strokeWidth={isHovered ? 1.5 : 1}
                      filter={isHovered ? 'url(#breakGlow)' : undefined}
                    />
                    <text
                      x="0"
                      y="11.5"
                      textAnchor="middle"
                      className="text-[9px] font-mono font-bold fill-gold-300 pointer-events-none"
                    >
                      ⚡ {bp.bar.date.length > 5 ? bp.bar.date.slice(5) : bp.bar.date}
                    </text>
                  </g>

                  {/* Break point marker on price trajectory */}
                  <circle
                    cx={bp.x}
                    cy={bp.y}
                    r={isHovered ? 10 : 7}
                    fill="#DEC084"
                    fillOpacity={isHovered ? 0.45 : 0.25}
                  />
                  <circle
                    cx={bp.x}
                    cy={bp.y}
                    r={isHovered ? 5.5 : 4}
                    fill="#DEC084"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                </g>
              );
            })}

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
              className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
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
                r="4.5"
                fill="#38bdf8"
                stroke="#0E0E11"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>
      </div>

      {/* Optional Interactive CUSUM S_k Trajectory Sub-Chart */}
      {showCusumCurve && cusumSqPoints.length > 0 && (
        <div className="pt-3 border-t border-zinc-100 dark:border-zinc-800/60">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-gold-500" />
              <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                Recursive CUSUM of Squares Trajectory (S_k)
              </span>
              <span className="text-[10px] text-zinc-400">
                Brown-Durbin-Evans / Inclán-Tiao 95% Confidence Bounds
              </span>
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-gold-500 flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 bg-gold-500" /> S_k
              </span>
              <span className="text-gold-500 flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 bg-gold-500 border-b border-dashed" /> ±c_α (95%)
              </span>
              <span className="text-zinc-400 flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 bg-zinc-400 border-b border-dashed" /> E[S_k] = k/n
              </span>
            </div>
          </div>

          <div className="relative w-full h-[110px]">
            <svg viewBox={`0 0 ${width} 110`} className="w-full h-full overflow-visible">
              {/* Background bounding box */}
              <rect
                x={padLeft}
                y={10}
                width={chartW}
                height={80}
                fill="none"
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800"
                strokeWidth="1"
              />

              {/* Diagonal Expected line (k / n) */}
              <line
                x1={padLeft}
                y1={90}
                x2={padLeft + chartW}
                y2={10}
                stroke="#94a3b8"
                strokeDasharray="3 3"
                strokeWidth="1"
              />

              {/* Upper 95% Band */}
              {(() => {
                const upperPoints = cusumSqPoints
                  .map((p, idx) => {
                    const x = padLeft + (idx / Math.max(1, cusumSqPoints.length - 1)) * chartW;
                    const y = 90 - p.upperBand * 80;
                    return `${x},${y}`;
                  })
                  .join(' ');
                return (
                  <polyline
                    points={upperPoints}
                    fill="none"
                    stroke="#DEC084"
                    strokeDasharray="4 3"
                    strokeWidth="1.2"
                  />
                );
              })()}

              {/* Lower 95% Band */}
              {(() => {
                const lowerPoints = cusumSqPoints
                  .map((p, idx) => {
                    const x = padLeft + (idx / Math.max(1, cusumSqPoints.length - 1)) * chartW;
                    const y = 90 - p.lowerBand * 80;
                    return `${x},${y}`;
                  })
                  .join(' ');
                return (
                  <polyline
                    points={lowerPoints}
                    fill="none"
                    stroke="#DEC084"
                    strokeDasharray="4 3"
                    strokeWidth="1.2"
                  />
                );
              })()}

              {/* Empirical S_k Curve */}
              {(() => {
                const skPoints = cusumSqPoints
                  .map((p, idx) => {
                    const x = padLeft + (idx / Math.max(1, cusumSqPoints.length - 1)) * chartW;
                    const y = 90 - p.s * 80;
                    return `${x},${y}`;
                  })
                  .join(' ');
                return (
                  <polyline
                    points={skPoints}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                );
              })()}

              {/* Vertical Break Point markers matching the upper chart */}
              {resolvedBreakPoints.map((bp, bpIdx) => (
                <g key={`cusum-break-${bpIdx}`}>
                  <line
                    x1={bp.x}
                    y1={10}
                    x2={bp.x}
                    y2={90}
                    stroke="#DEC084"
                    strokeDasharray="3 3"
                    strokeWidth="1.5"
                  />
                  <circle
                    cx={bp.x}
                    cy={10 + 4}
                    r="3"
                    fill="#DEC084"
                  />
                </g>
              ))}

              {/* Y Axis bounds labels */}
              <text x={padLeft - 6} y={14} textAnchor="end" className="text-[9px] font-mono fill-zinc-400">1.0</text>
              <text x={padLeft - 6} y={54} textAnchor="end" className="text-[9px] font-mono fill-zinc-400">0.5</text>
              <text x={padLeft - 6} y={92} textAnchor="end" className="text-[9px] font-mono fill-zinc-400">0.0</text>
            </svg>
          </div>
        </div>
      )}
    </div>
  );
}
