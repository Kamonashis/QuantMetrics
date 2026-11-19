'use client';

import React, { useState } from 'react';
import {
  GarchModelSummary,
  GarchForecastPoint,
} from '@/lib/quant-engine';
import { useChartSize } from '@/lib/use-chart-size';

interface VolatilityForecastChartProps {
  dates: string[];
  historicalVol: (number | null)[];
  /** The selected model's conditional vol series (EWMA / GJR-GARCH / HMM regime-mixture) */
  modelVol: number[];
  modelLabel: string;
  forecastPoints: GarchForecastPoint[];
  modelSummary: GarchModelSummary;
}

/**
 * Forward projection chart: 30d historical vol, the selected model's conditional vol line,
 * and the forward variance cone (68% / 95%). One model line only — no dual-model comparison.
 */
export function VolatilityForecastChart({
  dates,
  historicalVol,
  modelVol,
  modelLabel,
  forecastPoints,
  modelSummary,
}: VolatilityForecastChartProps) {
  const [showModel, setShowModel] = useState(true);
  const [showHist, setShowHist] = useState(true);
  const [hoverIndex, setHoverIndex] = useState<{ type: 'hist' | 'fc'; idx: number } | null>(null);
  const { ref, size } = useChartSize<HTMLDivElement>();

  const nHist = dates.length;
  const nFc = forecastPoints.length;
  const totalPoints = nHist + nFc;

  if (totalPoints === 0) return null;

  // Max volatility across all series
  let maxV = modelSummary.longRunAnnualizedVol * 1.2;
  historicalVol.forEach((v) => { if (v !== null && v > maxV) maxV = v; });
  modelVol.forEach((v) => { if (v > maxV) maxV = v; });
  forecastPoints.forEach((p) => { if (p.upper95 > maxV) maxV = p.upper95; });
  maxV = Math.max(0.2, maxV * 1.15);

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const padLeft = 46;
  const padRight = 16;
  const padTop = 16;
  const padBottom = 30;

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

  // Start forecast path from the last model-vol point
  const lastModelVol = modelVol[nHist - 1] || forecastPoints[0]?.annualizedVol || 0.2;
  fcLinePoints.push(`${splitX},${getY(lastModelVol)}`);
  upper95Points.push(`${splitX},${getY(lastModelVol)}`);
  lower95Points.push(`${splitX},${getY(lastModelVol)}`);
  upper68Points.push(`${splitX},${getY(lastModelVol)}`);
  lower68Points.push(`${splitX},${getY(lastModelVol)}`);

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

  const modelPoints = modelVol.map((v, i) => `${getX(i)},${getY(v)}`).join(' L ');

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

  const flatProjection = modelSummary.persistence >= 1;

  return (
    <div className="panel p-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Volatility Forecast Projection
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · forward variance projection
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            {nFc}-day {modelLabel} projection with 68% and 95% confidence cones
          </p>
        </div>

        {/* Series Visibility Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={() => setShowModel(!showModel)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border ${
              showModel
                ? 'bg-gold-500/10 border-gold-500/30 text-gold-600 dark:text-gold-400'
                : 'border-zinc-200 dark:border-zinc-800 text-zinc-400'
            }`}
          >
            {modelLabel}
          </button>
          <button
            onClick={() => setShowHist(!showHist)}
            className={`px-2.5 py-1 rounded-md font-medium transition-colors border ${
              showHist
                ? 'bg-zinc-500/10 border-zinc-500/30 text-zinc-600 dark:text-zinc-400'
                : 'border-zinc-200 dark:border-zinc-800 text-zinc-400'
            }`}
          >
            Historical (30d)
          </button>
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
          <defs>
            <linearGradient id="fcGradient95" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#DEC084" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#DEC084" stopOpacity="0.04" />
            </linearGradient>
            <linearGradient id="fcGradient68" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#DEC084" stopOpacity="0.26" />
              <stop offset="100%" stopColor="#DEC084" stopOpacity="0.10" />
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

          {/* Long-run unconditional vol reference — only when the model mean-reverts (not EWMA) */}
          {!flatProjection && (
            <>
              <line
                x1={padLeft}
                y1={getY(modelSummary.longRunAnnualizedVol)}
                x2={width - padRight}
                y2={getY(modelSummary.longRunAnnualizedVol)}
                stroke="#3fb27f"
                strokeWidth="0.8"
                strokeDasharray="4 4"
              />
              <text
                x={width - padRight - 2}
                y={getY(modelSummary.longRunAnnualizedVol) + 3}
                textAnchor="end"
                className="text-[9px] font-mono fill-gold-500 tabular-nums"
              >
                V_L: {(modelSummary.longRunAnnualizedVol * 100).toFixed(1)}%
              </text>
            </>
          )}

          {/* Vertical boundary separating historical and forecast */}
          <line
            x1={splitX}
            y1={padTop}
            x2={splitX}
            y2={padTop + chartH}
            stroke="#94a3b8"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <text
            x={splitX - 6}
            y={padTop + 12}
            textAnchor="end"
            className="text-[10px] font-medium fill-zinc-500 dark:fill-zinc-400"
          >
            Historical
          </text>
          <text
            x={splitX > width - 110 ? width - 4 : splitX + 6}
            y={padTop + 12}
            textAnchor={splitX > width - 110 ? 'end' : 'start'}
            className="text-[10px] font-medium fill-gold-500"
          >
            Forecast →
          </text>

          {/* Confidence cones */}
          <path d={area95Path} fill="url(#fcGradient95)" />
          <path d={area68Path} fill="url(#fcGradient68)" />

          {/* Historical 30d line */}
          {showHist && histPoints.length > 0 && (
            <path
              d={`M ${histPoints.join(' L ')}`}
              fill="none"
              stroke="#64748b"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          )}

          {/* Model conditional vol line */}
          {showModel && (
            <path
              d={`M ${modelPoints}`}
              fill="none"
              stroke="#DEC084"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Forecast trajectory line */}
          <path
            d={`M ${fcLinePoints.join(' L ')}`}
            fill="none"
            stroke="#DEC084"
            strokeWidth="1.3"
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
                stroke="#DEC084"
                strokeWidth="0.65"
                strokeDasharray="2 2"
                opacity="0.7"
              />
            </g>
          )}

          {/* X axis labels */}
          <text
            x={getX(0)}
            y={height - 8}
            className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
          >
            {dates[0]}
          </text>
          <text
            x={splitX}
            y={height - 8}
            textAnchor="middle"
            className="text-[9px] font-mono fill-zinc-700 dark:fill-zinc-300 font-semibold tabular-nums"
          >
            Today ({dates[nHist - 1]})
          </text>
          <text
            x={getX(totalPoints - 1)}
            y={height - 8}
            textAnchor="end"
            className="text-[9px] font-mono fill-gold-500 tabular-nums"
          >
            +{nFc}d ({forecastPoints[nFc - 1]?.date})
          </text>
        </svg>
        )}
      </div>

      {/* Model Diagnostic Footnotes & Parameter Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/80 text-xs">
        <div className="bg-zinc-50 dark:bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800/60">
          <span className="text-zinc-500 dark:text-zinc-400 block">Model Weights (α, β)</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {modelSummary.alpha.toFixed(3)} · {modelSummary.beta.toFixed(3)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            ω: {modelSummary.omega.toExponential(2)}
          </span>
        </div>

        <div className="bg-zinc-50 dark:bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800/60">
          <span className="text-zinc-500 dark:text-zinc-400 block">Shock Persistence (α+β)</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {flatProjection ? '1.0000' : modelSummary.persistence.toFixed(4)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            {flatProjection ? 'No mean reversion' : `Half-life: ${modelSummary.halfLife ? `${modelSummary.halfLife.toFixed(1)} days` : 'Infinite'}`}
          </span>
        </div>

        <div className="bg-zinc-50 dark:bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800/60">
          <span className="text-zinc-500 dark:text-zinc-400 block">Projection Level (start)</span>
          <span className="font-mono font-bold text-gold-600 dark:text-gold-400 tabular-nums text-xs">
            {(lastModelVol * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            {flatProjection ? 'flat-forward from latest EWMA vol' : `reverts toward V_L ${(modelSummary.longRunAnnualizedVol * 100).toFixed(1)}%`}
          </span>
        </div>

        <div className="bg-zinc-50 dark:bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800/60">
          <span className="text-zinc-500 dark:text-zinc-400 block">Projection Horizon</span>
          <span className="font-mono font-bold text-zinc-900 dark:text-zinc-100 tabular-nums text-xs">
            {nFc} trading days
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            68% / 95% cones widen as √h
          </span>
        </div>
      </div>
    </div>
  );
}
