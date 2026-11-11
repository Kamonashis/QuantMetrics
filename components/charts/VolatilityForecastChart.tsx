'use client';

import React, { useState } from 'react';
import {
  GarchModelSummary,
  GarchForecastPoint,
  EwmaSummary,
} from '@/lib/quant-engine';

interface VolatilityForecastChartProps {
  dates: string[];
  historicalVol: (number | null)[];
  ewmaVol: number[];
  garchVol: number[];
  forecastPoints: GarchForecastPoint[];
  garchSummary: GarchModelSummary;
  ewmaSummary: EwmaSummary;
}

export function VolatilityForecastChart({
  dates,
  historicalVol,
  ewmaVol,
  garchVol,
  forecastPoints,
  garchSummary,
  ewmaSummary,
}: VolatilityForecastChartProps) {
  const [showEWMA, setShowEWMA] = useState(true);
  const [showGARCH, setShowGARCH] = useState(true);
  const [showHist, setShowHist] = useState(true);
  const [hoverIndex, setHoverIndex] = useState<{ type: 'hist' | 'fc'; idx: number } | null>(null);

  const nHist = dates.length;
  const nFc = forecastPoints.length;
  const totalPoints = nHist + nFc;

  if (totalPoints === 0) return null;

  // Max volatility across all series
  let maxV = garchSummary.longRunAnnualizedVol * 1.2;
  historicalVol.forEach((v) => { if (v !== null && v > maxV) maxV = v; });
  ewmaVol.forEach((v) => { if (v > maxV) maxV = v; });
  garchVol.forEach((v) => { if (v > maxV) maxV = v; });
  forecastPoints.forEach((p) => { if (p.upper95 > maxV) maxV = p.upper95; });
  maxV = Math.max(0.2, maxV * 1.15);

  const width = 850;
  const height = 320;
  const padLeft = 55;
  const padRight = 30;
  const padTop = 20;
  const padBottom = 35;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (globalIdx: number) =>
    padLeft + (globalIdx / Math.max(1, totalPoints - 1)) * chartW;
  const getY = (val: number) =>
    padTop + chartH - (val / maxV) * chartH;

  const splitX = getX(nHist - 1);

  // Build forecast area band paths
  const upper95Points: string[] = [];
  const lower95Points: string[] = [];
  const upper68Points: string[] = [];
  const lower68Points: string[] = [];
  const fcLinePoints: string[] = [];

  // Start forecast path from last GARCH point
  const lastGarchVol = garchVol[nHist - 1] || forecastPoints[0]?.annualizedVol || 0.2;
  fcLinePoints.push(`${splitX},${getY(lastGarchVol)}`);
  upper95Points.push(`${splitX},${getY(lastGarchVol)}`);
  lower95Points.push(`${splitX},${getY(lastGarchVol)}`);
  upper68Points.push(`${splitX},${getY(lastGarchVol)}`);
  lower68Points.push(`${splitX},${getY(lastGarchVol)}`);

  forecastPoints.forEach((p, i) => {
    const x = getX(nHist + i);
    fcLinePoints.push(`${x},${getY(p.annualizedVol)}`);
    upper95Points.push(`${x},${getY(p.upper95)}`);
    lower95Points.push(`${x},${getY(p.lower95)}`);
    upper68Points.push(`${x},${getY(p.upper68)}`);
    lower68Points.push(`${x},${getY(p.lower68)}`);
  });

  const area95Path = `M ${upper95Points.join(' L ')} L ${lower95Points.reverse().join(' L ')} Z`;
  const area68Path = `M ${upper68Points.join(' L ')} L ${lower68Points.reverse().join(' L ')} Z`;

  // Historical paths
  const histPoints: string[] = [];
  historicalVol.forEach((v, i) => {
    if (v !== null) histPoints.push(`${getX(i)},${getY(v)}`);
  });

  const ewmaPoints = ewmaVol.map((v, i) => `${getX(i)},${getY(v)}`).join(' L ');
  const garchPoints = garchVol.map((v, i) => `${getX(i)},${getY(v)}`).join(' L ');

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const clampedX = Math.max(padLeft, Math.min(width - padRight, normX));
    const ratio = (clampedX - padLeft) / chartW;
    const gIdx = Math.round(ratio * (totalPoints - 1));

    if (gIdx < nHist) {
      setHoverIndex({ type: 'hist', idx: gIdx });
    } else {
      setHoverIndex({ type: 'fc', idx: Math.min(nFc - 1, gIdx - nHist) });
    }
  };

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/80 rounded-xl p-5 shadow-xs">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm tracking-tight">
              Volatility Modeling: GARCH(1,1), EWMA & Forecast Projection
            </h3>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              · Term Structure Projection
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Model comparison alongside {nFc}-day GARCH variance forecast with 68% and 95% confidence cones
          </p>
        </div>

        {/* Series Visibility Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={() => setShowGARCH(!showGARCH)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border ${
              showGARCH
                ? 'bg-sky-500/10 border-sky-500/30 text-sky-600 dark:text-sky-400'
                : 'border-slate-200 dark:border-slate-800 text-slate-400'
            }`}
          >
            GARCH(1,1)
          </button>
          <button
            onClick={() => setShowEWMA(!showEWMA)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border ${
              showEWMA
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400'
                : 'border-slate-200 dark:border-slate-800 text-slate-400'
            }`}
          >
            EWMA (λ={ewmaSummary.lambda})
          </button>
          <button
            onClick={() => setShowHist(!showHist)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border ${
              showHist
                ? 'bg-slate-500/10 border-slate-500/30 text-slate-600 dark:text-slate-400'
                : 'border-slate-200 dark:border-slate-800 text-slate-400'
            }`}
          >
            Historical (30d)
          </button>
        </div>
      </div>

      <div className="relative w-full aspect-[8.5/3.2] min-h-[220px]">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-full cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          <defs>
            <linearGradient id="fcGradient95" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.04" />
            </linearGradient>
            <linearGradient id="fcGradient68" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.28" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.10" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
            const val = pct * maxV;
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

          {/* Long run unconditional volatility reference line */}
          <line
            x1={padLeft}
            y1={getY(garchSummary.longRunAnnualizedVol)}
            x2={width - padRight}
            y2={getY(garchSummary.longRunAnnualizedVol)}
            stroke="#10b981"
            strokeWidth="1.2"
            strokeDasharray="4 4"
          />
          <text
            x={width - padRight + 4}
            y={getY(garchSummary.longRunAnnualizedVol) + 3}
            className="text-[9px] font-mono fill-emerald-500 tabular-nums"
          >
            V_L: {(garchSummary.longRunAnnualizedVol * 100).toFixed(1)}%
          </text>

          {/* Vertical boundary separating historical and forecast */}
          <line
            x1={splitX}
            y1={padTop}
            x2={splitX}
            y2={padTop + chartH}
            stroke="#94a3b8"
            strokeWidth="1.5"
            strokeDasharray="3 3"
          />
          <text
            x={splitX - 6}
            y={padTop + 12}
            textAnchor="end"
            className="text-[10px] font-medium fill-slate-500 dark:fill-slate-400"
          >
            Historical
          </text>
          <text
            x={splitX + 6}
            y={padTop + 12}
            textAnchor="start"
            className="text-[10px] font-medium fill-sky-500"
          >
            Forecast Projection →
          </text>

          {/* Confidence Cones */}
          <path d={area95Path} fill="url(#fcGradient95)" />
          <path d={area68Path} fill="url(#fcGradient68)" />

          {/* Historical line: 30d Historical */}
          {showHist && histPoints.length > 0 && (
            <path
              d={`M ${histPoints.join(' L ')}`}
              fill="none"
              stroke="#64748b"
              strokeWidth="1.5"
              strokeDasharray="2 2"
            />
          )}

          {/* Historical line: EWMA */}
          {showEWMA && (
            <path
              d={`M ${ewmaPoints}`}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}

          {/* Historical line: GARCH */}
          {showGARCH && (
            <path
              d={`M ${garchPoints}`}
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          )}

          {/* Forecast trajectory line */}
          <path
            d={`M ${fcLinePoints.join(' L ')}`}
            fill="none"
            stroke="#0ea5e9"
            strokeWidth="2.2"
            strokeDasharray="4 2"
          />

          {/* Hover indicator */}
          {hoverIndex && (
            <g>
              <line
                x1={
                  hoverIndex.type === 'hist'
                    ? getX(hoverIndex.idx)
                    : getX(nHist + hoverIndex.idx)
                }
                y1={padTop}
                x2={
                  hoverIndex.type === 'hist'
                    ? getX(hoverIndex.idx)
                    : getX(nHist + hoverIndex.idx)
                }
                y2={padTop + chartH}
                stroke="#38bdf8"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
            </g>
          )}

          {/* X axis labels */}
          <text
            x={getX(0)}
            y={height - 8}
            className="text-[9px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
          >
            {dates[0]}
          </text>
          <text
            x={splitX}
            y={height - 8}
            textAnchor="middle"
            className="text-[9px] font-mono fill-slate-700 dark:fill-slate-300 font-semibold tabular-nums"
          >
            Today ({dates[nHist - 1]})
          </text>
          <text
            x={getX(totalPoints - 1)}
            y={height - 8}
            textAnchor="end"
            className="text-[9px] font-mono fill-sky-500 tabular-nums"
          >
            +{nFc}d ({forecastPoints[nFc - 1]?.date})
          </text>
        </svg>
      </div>

      {/* Model Diagnostic Footnotes & Parameter Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/80 text-xs">
        <div className="bg-slate-50 dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800/60">
          <span className="text-slate-500 dark:text-slate-400 block">GARCH Parameters</span>
          <span className="font-mono font-bold text-slate-900 dark:text-slate-100 tabular-nums text-xs">
            α: {garchSummary.alpha.toFixed(3)} · β: {garchSummary.beta.toFixed(3)}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            ω: {garchSummary.omega.toExponential(2)}
          </span>
        </div>

        <div className="bg-slate-50 dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800/60">
          <span className="text-slate-500 dark:text-slate-400 block">Shock Persistence (α+β)</span>
          <span className="font-mono font-bold text-slate-900 dark:text-slate-100 tabular-nums text-xs">
            {garchSummary.persistence.toFixed(4)}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Half-life: {garchSummary.halfLife ? `${garchSummary.halfLife.toFixed(1)} days` : 'Infinite'}
          </span>
        </div>

        <div className="bg-slate-50 dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800/60">
          <span className="text-slate-500 dark:text-slate-400 block">Unconditional Long-Run Vol</span>
          <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400 tabular-nums text-xs">
            {(garchSummary.longRunAnnualizedVol * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Log-Likelihood: {garchSummary.logLikelihood.toFixed(1)}
          </span>
        </div>

        <div className="bg-slate-50 dark:bg-slate-900/60 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800/60">
          <span className="text-slate-500 dark:text-slate-400 block">EWMA Latest vs Mean</span>
          <span className="font-mono font-bold text-amber-600 dark:text-amber-400 tabular-nums text-xs">
            {(ewmaSummary.latestAnnualizedVol * 100).toFixed(1)}% / {(ewmaSummary.meanAnnualizedVol * 100).toFixed(1)}%
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400 block mt-0.5">
            Range: {(ewmaSummary.minAnnualizedVol * 100).toFixed(1)}% - {(ewmaSummary.maxAnnualizedVol * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}
