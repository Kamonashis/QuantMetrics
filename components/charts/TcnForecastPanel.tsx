'use client';

import React, { useState } from 'react';
import type { DecompositionResult } from '@/lib/quant-engine';
import {
  DEFAULT_TCN_CONFIG,
  MIN_BARS_FOR_TCN,
  computeParameterCount,
  computeReceptiveField,
  generateLookbackSweepGrid,
  sanitizeTcnConfig,
  type ComponentForecast,
  type DecompositionComponentKey,
  type SensitivityStep,
  type TcnAttentionConfig,
  type TcnDecompositionForecast,
  type TcnSensitivityResult,
  type TcnTrainingProgress,
} from '@/lib/tcn-attention';
import { TcnAttentionEquation } from '@/components/math/MathEquations';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import {
  Brain,
  Play,
  RotateCcw,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Sparkles,
  AlertCircle,
  AlertTriangle,
  ShieldCheck,
  SlidersHorizontal,
} from 'lucide-react';

interface TcnForecastPanelProps {
  ticker: string;
  decomposition: DecompositionResult;
  config: TcnAttentionConfig;
  onConfigChange: (next: TcnAttentionConfig) => void;
  forecast: TcnDecompositionForecast | null;
  sensitivityEnabled: boolean;
  onSensitivityEnabledChange: (enabled: boolean) => void;
  sensitivityStep: SensitivityStep;
  onSensitivityStepChange: (step: SensitivityStep) => void;
  sensitivityResult: TcnSensitivityResult | null;
  isSensitivityStale: boolean;
  isTraining: boolean;
  progress: TcnTrainingProgress | null;
  isStale: boolean;
  onTrain: () => void;
}

const COMPONENT_META: Record<
  DecompositionComponentKey,
  { short: string; label: string; color: string }
> = {
  trend: { short: 'T', label: 'Trend (T)', color: '#3fb27f' },
  cyclical: { short: 'C', label: 'Cyclical (C)', color: '#d4a853' },
  seasonal: { short: 'S', label: 'Seasonal (S)', color: '#a1a1aa' },
  residual: { short: 'I', label: 'Residual (I)', color: '#e0655f' },
};

const COMPONENT_KEYS: DecompositionComponentKey[] = ['trend', 'cyclical', 'seasonal', 'residual'];

const DILATION_PRESETS: { label: string; value: string; dilations: number[] }[] = [
  { label: '1, 2 (Short)', value: '1,2', dilations: [1, 2] },
  { label: '1, 2, 4 (Balanced)', value: '1,2,4', dilations: [1, 2, 4] },
  { label: '1, 2, 4, 8 (Wide)', value: '1,2,4,8', dilations: [1, 2, 4, 8] },
  { label: '1, 2, 4, 8, 16 (Deep)', value: '1,2,4,8,16', dilations: [1, 2, 4, 8, 16] },
];

function formatComponentValue(
  key: DecompositionComponentKey,
  val: number,
  model: 'additive' | 'multiplicative'
): string {
  if (key === 'trend') return `$${val.toFixed(2)}`;
  if (model === 'multiplicative') return `${val.toFixed(4)}×`;
  return `${val >= 0 ? '+' : ''}${val.toFixed(2)}`;
}

function formatSignedDollar(val: number): string {
  const sign = val >= 0 ? '+' : '-';
  return `${sign}$${Math.abs(val).toFixed(2)}`;
}

function formatSignedPct(val: number): string {
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(2)}%`;
}

function isDefaultConfig(cfg: TcnAttentionConfig): boolean {
  return (
    cfg.lookback === DEFAULT_TCN_CONFIG.lookback &&
    cfg.kernelSize === DEFAULT_TCN_CONFIG.kernelSize &&
    cfg.numHeads === DEFAULT_TCN_CONFIG.numHeads &&
    cfg.hiddenChannels === DEFAULT_TCN_CONFIG.hiddenChannels &&
    cfg.epochs === DEFAULT_TCN_CONFIG.epochs &&
    cfg.learningRate === DEFAULT_TCN_CONFIG.learningRate &&
    cfg.dilations.join(',') === DEFAULT_TCN_CONFIG.dilations.join(',')
  );
}

/** Pure SVG horizontal 99% prediction interval gauge with last close & predicted point */
function PredictionIntervalBar({
  lastPrice,
  predictedPrice,
  lower99,
  upper99,
}: {
  lastPrice: number;
  predictedPrice: number;
  lower99: number;
  upper99: number;
}) {
  const minVal = Math.min(lower99, lastPrice, predictedPrice);
  const maxVal = Math.max(upper99, lastPrice, predictedPrice);
  const pad = Math.max((maxVal - minVal) * 0.12, lastPrice * 0.005, 0.25);
  const domMin = minVal - pad;
  const domMax = maxVal + pad;
  const span = Math.max(domMax - domMin, 1e-6);

  const w = 520;
  const h = 54;
  const xL = 16;
  const xR = w - 16;
  const trackW = xR - xL;
  const toX = (v: number) => xL + ((v - domMin) / span) * trackW;

  const xLow = toX(lower99);
  const xHigh = toX(upper99);
  const xLast = toX(lastPrice);
  const xPred = toX(predictedPrice);
  const midY = 24;
  const isUp = predictedPrice >= lastPrice;

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="block w-full h-[54px] select-none"
        role="img"
        aria-label={`99% prediction interval from $${lower99.toFixed(2)} to $${upper99.toFixed(2)}, predicted $${predictedPrice.toFixed(2)}`}
      >
        {/* Base rail */}
        <line
          x1={xL}
          y1={midY}
          x2={xR}
          y2={midY}
          stroke="currentColor"
          className="text-zinc-200 dark:text-zinc-800"
          strokeWidth={2}
          strokeLinecap="round"
        />

        {/* 99% confidence range band */}
        <rect
          x={Math.min(xLow, xHigh)}
          y={midY - 5}
          width={Math.max(2, Math.abs(xHigh - xLow))}
          height={10}
          rx={3}
          fill="#DEC084"
          fillOpacity={0.22}
          stroke="#DEC084"
          strokeOpacity={0.6}
          strokeWidth={1}
        />

        {/* Interval end caps */}
        <line x1={xLow} y1={midY - 8} x2={xLow} y2={midY + 8} stroke="#DEC084" strokeWidth={1.5} />
        <line x1={xHigh} y1={midY - 8} x2={xHigh} y2={midY + 8} stroke="#DEC084" strokeWidth={1.5} />

        {/* Last close reference tick */}
        <line
          x1={xLast}
          y1={midY - 11}
          x2={xLast}
          y2={midY + 11}
          stroke="currentColor"
          className="text-zinc-500 dark:text-zinc-400"
          strokeWidth={1.5}
          strokeDasharray="2 2"
        />

        {/* Connector from last close to predicted */}
        <line
          x1={xLast}
          y1={midY}
          x2={xPred}
          y2={midY}
          stroke={isUp ? '#3fb27f' : '#e0655f'}
          strokeWidth={2.5}
        />

        {/* Predicted point marker */}
        <circle
          cx={xPred}
          cy={midY}
          r={5}
          fill={isUp ? '#3fb27f' : '#e0655f'}
          stroke="#08080a"
          strokeWidth={1.5}
        />

        {/* Labels */}
        <text
          x={xLow}
          y={46}
          textAnchor="start"
          className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
        >
          L99 ${lower99.toFixed(2)}
        </text>
        <text
          x={xHigh}
          y={46}
          textAnchor="end"
          className="text-[10px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
        >
          U99 ${upper99.toFixed(2)}
        </text>
        <text
          x={xLast}
          y={10}
          textAnchor="middle"
          className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
        >
          Last ${lastPrice.toFixed(2)}
        </text>
      </svg>
    </div>
  );
}

/** Pure SVG Attention Weight Bar Chart across lookback lags */
function AttentionBarChart({
  comp,
  selectedHead,
}: {
  comp: ComponentForecast;
  selectedHead: 'mean' | number;
}) {
  const [hoverLag, setHoverLag] = useState<number | null>(null);
  const weights =
    selectedHead === 'mean'
      ? comp.attentionWeights
      : comp.attentionByHead[selectedHead] ?? comp.attentionWeights;

  const L = weights.length;
  if (L === 0) return null;

  const maxW = Math.max(...weights, 1e-6);
  // Identify top-3 lag indices for champagne gold highlight
  const sortedIndices = weights
    .map((w, idx) => ({ w, idx }))
    .sort((a, b) => b.w - a.w)
    .slice(0, 3);
  const topSet = new Set(sortedIndices.map((x) => x.idx));

  const w = 420;
  const h = 152;
  const padL = 34;
  const padR = 12;
  const padT = 14;
  const padB = 26;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;
  const slotW = chartW / L;
  const barW = Math.max(2, slotW * 0.72);

  const activeIdx = hoverLag !== null ? hoverLag : sortedIndices[0]?.idx ?? L - 1;
  const activeLagSteps = L - activeIdx; // lag = L - idx => t-(L-1) .. t-0 (wait: idx L-1 is t-0)
  const lagLabel = activeLagSteps - 1 === 0 ? 't' : `t-${activeLagSteps - 1}`;

  return (
    <div>
      <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 dark:text-zinc-400 mb-1.5 tabular-nums">
        <span>
          Focus lag:{' '}
          <strong className="text-gold-600 dark:text-gold-400">{lagLabel}</strong>
        </span>
        <span>
          Weight:{' '}
          <strong className="text-zinc-900 dark:text-zinc-100">
            {((weights[activeIdx] ?? 0) * 100).toFixed(1)}%
          </strong>
        </span>
      </div>

      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="block w-full h-[152px] select-none"
        onMouseLeave={() => setHoverLag(null)}
      >
        {/* Horizontal gridlines */}
        {[0, 0.5, 1].map((frac, idx) => {
          const y = padT + chartH * (1 - frac);
          return (
            <g key={idx}>
              <line
                x1={padL}
                y1={y}
                x2={w - padR}
                y2={y}
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800/80"
                strokeDasharray="2 3"
                strokeWidth={0.75}
              />
              <text
                x={padL - 5}
                y={y + 3}
                textAnchor="end"
                className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {(frac * maxW * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {/* Bars */}
        {weights.map((val, idx) => {
          const barH = Math.max(1, (val / maxW) * chartH);
          const x = padL + idx * slotW + (slotW - barW) / 2;
          const y = padT + chartH - barH;
          const isTop = topSet.has(idx);
          const isHovered = hoverLag === idx;
          return (
            <g
              key={idx}
              onMouseEnter={() => setHoverLag(idx)}
              className="cursor-pointer"
            >
              <rect
                x={padL + idx * slotW}
                y={padT}
                width={slotW}
                height={chartH}
                fill="transparent"
              />
              <rect
                x={x}
                y={y}
                width={barW}
                height={barH}
                rx={1.5}
                fill={isTop || isHovered ? '#DEC084' : '#71717a'}
                fillOpacity={isHovered ? 1 : isTop ? 0.9 : 0.45}
              />
            </g>
          );
        })}

        {/* X-axis ticks */}
        {[0, Math.floor((L - 1) / 2), L - 1].map((idx) => {
          const lagOffset = L - 1 - idx;
          const x = padL + idx * slotW + slotW / 2;
          return (
            <text
              key={idx}
              x={x}
              y={h - 8}
              textAnchor={idx === 0 ? 'start' : idx === L - 1 ? 'end' : 'middle'}
              className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
            >
              {lagOffset === 0 ? 't (latest)' : `t-${lagOffset}`}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

/** Pure SVG Dual-Line Train vs Holdout Loss Convergence Curve */
function LossCurveChart({ comp }: { comp: ComponentForecast }) {
  const train = comp.trainLossCurve;
  const holdout = comp.holdoutLossCurve;
  const n = train.length;
  if (n === 0) return null;

  const allVals = [...train, ...holdout].filter((v) => Number.isFinite(v));
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const span = Math.max(maxL - minL, 1e-6);

  const w = 420;
  const h = 152;
  const padL = 38;
  const padR = 12;
  const padT = 14;
  const padB = 26;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const toX = (i: number) => padL + (i / Math.max(1, n - 1)) * chartW;
  const toY = (v: number) => padT + chartH - ((v - minL) / span) * chartH;

  const trainPath = train
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`)
    .join(' ');
  const holdoutPath = holdout
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`)
    .join(' ');

  const bestIdx = Math.max(0, Math.min(n - 1, comp.bestEpoch - 1));
  const bestX = toX(bestIdx);
  const lastTrain = train[n - 1] ?? 0;
  const lastHold = holdout[bestIdx] ?? holdout[holdout.length - 1] ?? 0;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 mb-1.5 tabular-nums">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-[#DEC084] inline-block" />
            Train {lastTrain.toFixed(3)}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-[#3fb27f] inline-block" />
            Holdout {lastHold.toFixed(3)}
          </span>
        </div>
        <span>
          Best ep <strong className="text-zinc-900 dark:text-zinc-100">{comp.bestEpoch}</strong>/{n}
        </span>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="block w-full h-[152px] select-none">
        {/* Horizontal gridlines */}
        {[0, 0.5, 1].map((frac, idx) => {
          const val = minL + frac * span;
          const y = toY(val);
          return (
            <g key={idx}>
              <line
                x1={padL}
                y1={y}
                x2={w - padR}
                y2={y}
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800/80"
                strokeDasharray="2 3"
                strokeWidth={0.75}
              />
              <text
                x={padL - 5}
                y={y + 3}
                textAnchor="end"
                className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Best epoch marker */}
        <line
          x1={bestX}
          y1={padT}
          x2={bestX}
          y2={padT + chartH}
          stroke="#3fb27f"
          strokeOpacity={0.55}
          strokeDasharray="3 2"
          strokeWidth={1}
        />

        {/* Holdout curve */}
        <path
          d={holdoutPath}
          fill="none"
          stroke="#3fb27f"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Train curve */}
        <path
          d={trainPath}
          fill="none"
          stroke="#DEC084"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* X-axis ticks */}
        {[0, Math.floor((n - 1) / 2), n - 1].map((idx) => (
          <text
            key={idx}
            x={toX(idx)}
            y={h - 8}
            textAnchor={idx === 0 ? 'start' : idx === n - 1 ? 'end' : 'middle'}
            className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
          >
            ep {idx + 1}
          </text>
        ))}
      </svg>
    </div>
  );
}

/** Pure SVG Lookback-vs-Price Stability Chart with 99% PI whiskers and interactive inspection */
function LookbackSensitivityChart({
  sensitivity,
}: {
  sensitivity: TcnSensitivityResult;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);

  const { runs, summary, baseForecast } = sensitivity;
  const k = runs.length;
  if (k === 0) return null;

  const baseIdx = Math.max(0, runs.findIndex((r) => r.isBase));
  const safePinned = pinnedIdx !== null && pinnedIdx < k ? pinnedIdx : null;
  const activeIdx = hoverIdx !== null && hoverIdx < k ? hoverIdx : safePinned ?? baseIdx;
  const activeRun = runs[activeIdx];
  const lastPrice = baseForecast.lastPrice;

  let minY = Math.min(lastPrice, summary.minPrice);
  let maxY = Math.max(lastPrice, summary.maxPrice);
  for (const r of runs) {
    if (r.forecast.lower99 < minY) minY = r.forecast.lower99;
    if (r.forecast.upper99 > maxY) maxY = r.forecast.upper99;
  }
  const padY = Math.max((maxY - minY) * 0.14, lastPrice * 0.004, 0.25);
  const domMin = minY - padY;
  const domMax = maxY + padY;
  const spanY = Math.max(domMax - domMin, 1e-6);

  const w = 680;
  const h = 196;
  const padL = 58;
  const padR = 24;
  const padT = 18;
  const padB = 34;
  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const toX = (i: number) =>
    k === 1 ? padL + chartW / 2 : padL + ((i + 0.5) / k) * chartW;
  const toY = (val: number) => padT + chartH - ((val - domMin) / spanY) * chartH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => domMin + f * spanY);
  const linePath = runs
    .map((r, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(r.forecast.predictedPrice).toFixed(1)}`)
    .join(' ');

  return (
    <div className="space-y-2">
      {/* Interactive Run Readout Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <span>
            Window{' '}
            <strong className="text-gold-600 dark:text-gold-400">
              L = {activeRun.lookback}d{activeRun.isBase ? ' (base)' : ''}
            </strong>
          </span>
          <span>·</span>
          <span>
            Pred{' '}
            <strong className="text-zinc-900 dark:text-zinc-100">
              ${activeRun.forecast.predictedPrice.toFixed(2)}
            </strong>{' '}
            <span
              className={
                activeRun.forecast.deltaDollar >= 0 ? 'text-[#3fb27f]' : 'text-[#e0655f]'
              }
            >
              ({formatSignedDollar(activeRun.forecast.deltaDollar)} /{' '}
              {formatSignedPct(activeRun.forecast.deltaPct)})
            </span>
          </span>
          <span>·</span>
          <span>
            99% PI [${activeRun.forecast.lower99.toFixed(2)}, ${activeRun.forecast.upper99.toFixed(2)}]
          </span>
          <span>·</span>
          <span>
            OOS RMSE{' '}
            <strong className="text-zinc-900 dark:text-zinc-100">
              ${activeRun.forecast.holdoutPriceRmse.toFixed(2)}
            </strong>
          </span>
        </div>

        {safePinned !== null && safePinned !== baseIdx && (
          <button
            type="button"
            onClick={() => setPinnedIdx(null)}
            className="text-[11px] text-gold-600 dark:text-gold-400 hover:underline cursor-pointer whitespace-nowrap shrink-0"
          >
            Reset to base ({runs[baseIdx].lookback}d)
          </button>
        )}
      </div>

      {/* Pure SVG Plot */}
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="block w-full h-[196px] select-none"
        role="img"
        aria-label="Lookback window sensitivity chart showing next-day predicted prices and 99% prediction intervals across tested lookback lengths"
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Horizontal Gridlines & Price Axis */}
        {yTicks.map((val, idx) => {
          const y = toY(val);
          return (
            <g key={idx}>
              <line
                x1={padL}
                y1={y}
                x2={w - padR}
                y2={y}
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800/80"
                strokeDasharray="2 3"
                strokeWidth={0.75}
              />
              <text
                x={padL - 7}
                y={y + 3}
                textAnchor="end"
                className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
              >
                ${val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Last Close Horizontal Reference */}
        <line
          x1={padL}
          y1={toY(lastPrice)}
          x2={w - padR}
          y2={toY(lastPrice)}
          stroke="currentColor"
          className="text-zinc-500 dark:text-zinc-400"
          strokeDasharray="4 3"
          strokeWidth={1}
          opacity={0.75}
        />
        <text
          x={w - padR}
          y={toY(lastPrice) - 4}
          textAnchor="end"
          className="text-[8.5px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
        >
          Last ${lastPrice.toFixed(2)}
        </text>

        {/* Sweep Mean Horizontal Reference */}
        <line
          x1={padL}
          y1={toY(summary.meanPrice)}
          x2={w - padR}
          y2={toY(summary.meanPrice)}
          stroke="#DEC084"
          strokeDasharray="2 2"
          strokeWidth={1}
          opacity={0.8}
        />

        {/* Predicted Trajectory Connector Across Lookbacks */}
        {k > 1 && (
          <path
            d={linePath}
            fill="none"
            stroke="#DEC084"
            strokeOpacity={0.7}
            strokeWidth={1.5}
          />
        )}

        {/* Per-Window 99% PI Whiskers + Point Markers */}
        {runs.map((run, idx) => {
          const cx = toX(idx);
          const cy = toY(run.forecast.predictedPrice);
          const yLow = toY(run.forecast.lower99);
          const yHigh = toY(run.forecast.upper99);
          const isActive = activeIdx === idx;
          const isBestRmse = run.lookback === summary.bestRmseLookback;
          const isUp = run.forecast.deltaDollar >= 0;
          const slotW = chartW / Math.max(1, k);

          return (
            <g
              key={run.lookback}
              onMouseEnter={() => setHoverIdx(idx)}
              onClick={() => setPinnedIdx(idx)}
              className="cursor-pointer"
            >
              {/* Hit area + active subtle column highlight */}
              <rect
                x={cx - slotW / 2}
                y={padT}
                width={slotW}
                height={chartH}
                fill={isActive ? '#DEC084' : 'transparent'}
                fillOpacity={isActive ? 0.07 : 0}
              />

              {/* 99% PI vertical bar */}
              <line
                x1={cx}
                y1={yLow}
                x2={cx}
                y2={yHigh}
                stroke={run.isBase ? '#DEC084' : '#71717a'}
                strokeOpacity={isActive || run.isBase ? 0.9 : 0.55}
                strokeWidth={run.isBase ? 2 : 1.5}
              />
              {/* Caps */}
              <line
                x1={cx - 6}
                y1={yLow}
                x2={cx + 6}
                y2={yLow}
                stroke={run.isBase ? '#DEC084' : '#71717a'}
                strokeWidth={1.3}
              />
              <line
                x1={cx - 6}
                y1={yHigh}
                x2={cx + 6}
                y2={yHigh}
                stroke={run.isBase ? '#DEC084' : '#71717a'}
                strokeWidth={1.3}
              />

              {/* Best OOS RMSE outer ring */}
              {isBestRmse && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={8}
                  fill="none"
                  stroke="#3fb27f"
                  strokeDasharray="2 2"
                  strokeWidth={1.2}
                />
              )}

              {/* Predicted price dot */}
              <circle
                cx={cx}
                cy={cy}
                r={run.isBase ? 5.5 : 4.5}
                fill={run.isBase ? '#DEC084' : isUp ? '#3fb27f' : '#e0655f'}
                stroke="#08080a"
                strokeWidth={1.5}
              />

              {/* X-axis Lookback label */}
              <text
                x={cx}
                y={h - 14}
                textAnchor="middle"
                className={`text-[10px] font-mono tabular-nums ${
                  run.isBase
                    ? 'fill-gold-600 dark:fill-gold-400 font-bold'
                    : 'fill-zinc-600 dark:fill-zinc-400'
                }`}
              >
                L={run.lookback}d
              </text>
              <text
                x={cx}
                y={h - 3}
                textAnchor="middle"
                className="text-[8.5px] font-mono fill-zinc-400 dark:fill-zinc-500 tabular-nums"
              >
                {run.isBase ? 'base' : isBestRmse ? 'min RMSE' : `$${run.forecast.predictedPrice.toFixed(1)}`}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend strip (unboxed middot metadata) */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#DEC084] inline-block" />
            Base lookback
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#3fb27f] inline-block" />
            Up / <span className="w-2 h-2 rounded-full bg-[#e0655f] inline-block" /> Down
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full border border-dashed border-[#3fb27f] inline-block" />
            Lowest OOS RMSE
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dotted border-[#DEC084] inline-block" />
            Sweep mean ${summary.meanPrice.toFixed(2)}
          </span>
        </div>
        <span>Click any lookback column to pin readout</span>
      </div>
    </div>
  );
}

export function TcnForecastPanel({
  ticker,
  decomposition,
  config,
  onConfigChange,
  forecast,
  sensitivityEnabled,
  onSensitivityEnabledChange,
  sensitivityStep,
  onSensitivityStepChange,
  sensitivityResult,
  isSensitivityStale,
  isTraining,
  progress,
  isStale,
  onTrain,
}: TcnForecastPanelProps) {
  const [diagComponent, setDiagComponent] = useState<DecompositionComponentKey>('trend');
  const [diagHead, setDiagHead] = useState<'mean' | number>('mean');

  const nObs = decomposition.observed.length;
  const effectiveConfig = sanitizeTcnConfig(config, Math.max(nObs, MIN_BARS_FOR_TCN));
  const receptiveField = computeReceptiveField(effectiveConfig.dilations, effectiveConfig.kernelSize);
  const paramCount = computeParameterCount(effectiveConfig);
  const insufficientData = nObs < MIN_BARS_FOR_TCN;
  const configChangedFromDefault = !isDefaultConfig(config);
  const plannedSweepGrid = generateLookbackSweepGrid(
    effectiveConfig.lookback,
    sensitivityStep,
    Math.max(nObs, MIN_BARS_FOR_TCN)
  );

  const activeDiagComp = forecast ? forecast.components[diagComponent] : null;
  const safeDiagHead: 'mean' | number =
    diagHead === 'mean' || (activeDiagComp && diagHead < activeDiagComp.attentionByHead.length)
      ? diagHead
      : 'mean';

  const handleHeadsChange = (heads: number) => {
    let nextHidden = config.hiddenChannels;
    if (nextHidden % heads !== 0) {
      nextHidden = Math.max(heads * 2, Math.round(nextHidden / heads) * heads);
      if (nextHidden > 16) nextHidden = 16;
      if (nextHidden % heads !== 0) nextHidden = heads * 2;
    }
    onConfigChange({
      ...config,
      numHeads: heads,
      hiddenChannels: nextHidden,
    });
  };

  const handleHiddenChange = (hidden: number) => {
    let nextHeads = config.numHeads;
    if (hidden % nextHeads !== 0) {
      nextHeads = hidden % 4 === 0 ? 4 : hidden % 2 === 0 ? 2 : 1;
    }
    onConfigChange({
      ...config,
      hiddenChannels: hidden,
      numHeads: nextHeads,
    });
  };

  const dilationKey = config.dilations.join(',');

  // Max absolute dollar contribution for diverging waterfall bar scaling
  const maxAbsDollar = forecast
    ? Math.max(
        ...COMPONENT_KEYS.map((k) => Math.abs(forecast.components[k].contributionDollar)),
        0.01
      )
    : 1;

  return (
    <div className="mt-5 pt-5 border-t border-zinc-200/80 dark:border-zinc-800/80 space-y-4">
      {/* Section Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <Brain className="w-4 h-4 text-gold-500 shrink-0" />
            <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              TCN + Multi-Head Self-Attention Next-Day Component Forecast
            </h4>
            <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
              · RF {receptiveField}d · {paramCount.toLocaleString()} params/comp · {effectiveConfig.numHeads}H
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-zinc-500 dark:text-zinc-400">
            <TcnAttentionEquation display="inline" className="text-zinc-700 dark:text-zinc-300" />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {configChangedFromDefault && (
            <button
              type="button"
              disabled={isTraining}
              onClick={() => onConfigChange({ ...DEFAULT_TCN_CONFIG })}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-transparent text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:border-gold-500/60 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors cursor-pointer disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Defaults</span>
            </button>
          )}
          <ShimmerButton
            type="button"
            disabled={isTraining || insufficientData}
            onClick={onTrain}
            className="h-9 px-3.5 text-xs whitespace-nowrap shrink-0"
          >
            {isTraining ? (
              <>
                <RotateCcw className="w-3.5 h-3.5 animate-spin text-gold-400" />
                <span>
                  {progress && progress.totalWindows && progress.windowIndex !== undefined
                    ? `Win ${progress.windowIndex + 1}/${progress.totalWindows} (${progress.windowLookback}d) · ${COMPONENT_META[progress.component].short} ${progress.epoch}/${progress.totalEpochs}`
                    : progress
                    ? `Training ${COMPONENT_META[progress.component].short} (${progress.epoch}/${progress.totalEpochs})`
                    : 'Training…'}
                </span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current text-gold-500" />
                <span>
                  {sensitivityEnabled
                    ? sensitivityResult && !isSensitivityStale
                      ? `Retrain ${plannedSweepGrid.length}-Window Sweep`
                      : `Run ${plannedSweepGrid.length}-Window Sweep`
                    : forecast
                    ? 'Retrain & Predict Next-Day'
                    : 'Train & Predict Next-Day'}
                </span>
              </>
            )}
          </ShimmerButton>
        </div>
      </div>

      {/* Hyperparameter Lab Strip */}
      <div className="panel-inset p-3.5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="kicker">Hyperparameter Lab</span>
          <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
            Chronological 80/20 walk-forward · Causal masked attention · Adam + Huber loss
            {!isTraining && (isStale || isSensitivityStale)
              ? sensitivityEnabled
                ? ' · Sweep queued — click Run Sweep'
                : ' · Params modified — click Retrain'
              : ''}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5 text-xs">
          {/* Lookback */}
          <div>
            <label
              htmlFor="tcn-lookback"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Lookback L
            </label>
            <select
              id="tcn-lookback"
              value={config.lookback}
              disabled={isTraining}
              onChange={(e) =>
                onConfigChange({ ...config, lookback: parseInt(e.target.value, 10) })
              }
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              {[16, 24, 32, 48].map((l) => (
                <option key={l} value={l}>
                  {l}d
                </option>
              ))}
            </select>
          </div>

          {/* Dilations */}
          <div>
            <label
              htmlFor="tcn-dilations"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Dilations
            </label>
            <select
              id="tcn-dilations"
              value={dilationKey}
              disabled={isTraining}
              onChange={(e) => {
                const preset = DILATION_PRESETS.find((p) => p.value === e.target.value);
                if (preset) {
                  onConfigChange({ ...config, dilations: [...preset.dilations] });
                }
              }}
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              {DILATION_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>
                  [{p.value}]
                </option>
              ))}
            </select>
          </div>

          {/* Kernel Size */}
          <div>
            <label
              htmlFor="tcn-kernel"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Kernel k
            </label>
            <select
              id="tcn-kernel"
              value={config.kernelSize}
              disabled={isTraining}
              onChange={(e) =>
                onConfigChange({ ...config, kernelSize: parseInt(e.target.value, 10) })
              }
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              <option value={2}>k = 2</option>
              <option value={3}>k = 3</option>
            </select>
          </div>

          {/* Attention Heads */}
          <div>
            <label
              htmlFor="tcn-heads"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Attn Heads
            </label>
            <select
              id="tcn-heads"
              value={config.numHeads}
              disabled={isTraining}
              onChange={(e) => handleHeadsChange(parseInt(e.target.value, 10))}
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              <option value={1}>1 Head</option>
              <option value={2}>2 Heads</option>
              <option value={4}>4 Heads</option>
            </select>
          </div>

          {/* Hidden Channels */}
          <div>
            <label
              htmlFor="tcn-hidden"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Channels d
            </label>
            <select
              id="tcn-hidden"
              value={config.hiddenChannels}
              disabled={isTraining}
              onChange={(e) => handleHiddenChange(parseInt(e.target.value, 10))}
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              {[8, 12, 16].map((d) => (
                <option key={d} value={d}>
                  d = {d}
                </option>
              ))}
            </select>
          </div>

          {/* Epochs */}
          <div>
            <label
              htmlFor="tcn-epochs"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Epochs
            </label>
            <select
              id="tcn-epochs"
              value={config.epochs}
              disabled={isTraining}
              onChange={(e) =>
                onConfigChange({ ...config, epochs: parseInt(e.target.value, 10) })
              }
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              {[20, 40, 60, 100].map((ep) => (
                <option key={ep} value={ep}>
                  {ep} ep
                </option>
              ))}
            </select>
          </div>

          {/* Learning Rate */}
          <div>
            <label
              htmlFor="tcn-lr"
              className="block text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 mb-1"
            >
              Learn Rate
            </label>
            <select
              id="tcn-lr"
              value={config.learningRate}
              disabled={isTraining}
              onChange={(e) =>
                onConfigChange({ ...config, learningRate: parseFloat(e.target.value) })
              }
              className="h-8 w-full px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
            >
              <option value={0.005}>0.005</option>
              <option value={0.01}>0.010</option>
              <option value={0.02}>0.020</option>
            </select>
          </div>
        </div>

        {/* Sensitivity Analysis Toggle & Step Row */}
        <div className="pt-3 border-t border-zinc-200/70 dark:border-zinc-800/70 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 text-xs">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              role="switch"
              aria-checked={sensitivityEnabled}
              disabled={isTraining}
              onClick={() => onSensitivityEnabledChange(!sensitivityEnabled)}
              className="inline-flex items-center gap-2 cursor-pointer disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 rounded-md py-0.5 whitespace-nowrap shrink-0"
            >
              <span
                className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-150 ${
                  sensitivityEnabled
                    ? 'bg-gold-500/25 border-gold-500'
                    : 'bg-zinc-200 dark:bg-zinc-800 border-zinc-300 dark:border-zinc-700'
                }`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 rounded-full transition-transform duration-150 ${
                    sensitivityEnabled
                      ? 'bg-gold-500 translate-x-[18px]'
                      : 'bg-zinc-500 dark:bg-zinc-400 translate-x-[2px]'
                  }`}
                />
              </span>
              <SlidersHorizontal className="w-3.5 h-3.5 text-gold-500 shrink-0" />
              <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                Sensitivity Analysis
              </span>
            </button>

            {sensitivityEnabled && (
              <div className="flex items-center gap-1.5">
                <label
                  htmlFor="tcn-sensitivity-step"
                  className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 whitespace-nowrap"
                >
                  Step:
                </label>
                <select
                  id="tcn-sensitivity-step"
                  value={sensitivityStep}
                  disabled={isTraining}
                  onChange={(e) =>
                    onSensitivityStepChange(parseInt(e.target.value, 10) as SensitivityStep)
                  }
                  className="h-7 px-2 rounded-md border border-zinc-200 dark:border-zinc-700/80 bg-white dark:bg-zinc-900 font-mono text-xs text-zinc-900 dark:text-zinc-100 tabular-nums focus:outline-hidden focus:ring-1 focus:ring-gold-500"
                >
                  <option value={2}>±2d</option>
                  <option value={4}>±4d</option>
                  <option value={6}>±6d</option>
                </select>
              </div>
            )}
          </div>

          <div className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
            {sensitivityEnabled ? (
              <span>
                Sweep:{' '}
                {plannedSweepGrid.map((wVal, idx) => (
                  <React.Fragment key={wVal}>
                    {idx > 0 && ' · '}
                    <span
                      className={
                        wVal === effectiveConfig.lookback
                          ? 'text-gold-600 dark:text-gold-400 font-semibold'
                          : ''
                      }
                    >
                      {wVal}d{wVal === effectiveConfig.lookback ? ' (base)' : ''}
                    </span>
                  </React.Fragment>
                ))}
              </span>
            ) : (
              <span>
                Enable to test {plannedSweepGrid.length} lookback variations around L={effectiveConfig.lookback}d on Retrain
              </span>
            )}
          </div>
        </div>

        {/* Live Training Progress Bar (compositor-only transform: scaleX) */}
        {isTraining && (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
              <span>
                {progress && progress.totalWindows && progress.windowIndex !== undefined
                  ? `Window ${progress.windowIndex + 1}/${progress.totalWindows} (L = ${progress.windowLookback}d) · Fitting ${COMPONENT_META[progress.component].label} · Epoch ${progress.epoch}/${progress.totalEpochs}`
                  : progress
                  ? `Fitting ${COMPONENT_META[progress.component].label} · Epoch ${progress.epoch}/${progress.totalEpochs} (${progress.componentIndex + 1}/${progress.totalComponents})`
                  : 'Initializing TCN-Attention networks…'}
              </span>
              <span>{Math.round((progress?.overallFraction ?? 0.05) * 100)}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
              <div
                className="h-full w-full bg-gold-500 origin-left transition-transform duration-150"
                style={{
                  transform: `scaleX(${Math.max(0.03, Math.min(1, progress?.overallFraction ?? 0.05))})`,
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Short History Empty Guard */}
      {insufficientData && (
        <div className="panel-inset p-4 flex items-start gap-2.5 text-xs text-zinc-600 dark:text-zinc-400">
          <AlertCircle className="w-4 h-4 text-gold-500 shrink-0 mt-0.5" />
          <div>
            <strong className="text-zinc-900 dark:text-zinc-100">
              Insufficient history for TCN-Attention training:
            </strong>{' '}
            {nObs} observations available, minimum {MIN_BARS_FOR_TCN} daily bars required for
            chronological train/holdout windows. Expand the date range above to run component
            forecasting.
          </div>
        </div>
      )}

      {/* Forecast Output Suite */}
      {!insufficientData && forecast && (
        <div className="space-y-4">
          {/* Reconstructed Next-Day Price Hero + 95% Prediction Interval */}
          <div className="panel-inset p-4 space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
              {/* Left KPIs */}
              <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    Last Close ({forecast.lastDate})
                  </div>
                  <div className="mt-1 font-mono text-xl font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                    ${forecast.lastPrice.toFixed(2)}
                  </div>
                  <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
                    {ticker} · {forecast.model}
                  </div>
                </div>

                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-gold-600 dark:text-gold-400">
                    Predicted ({forecast.nextDateLabel})
                  </div>
                  <div className="mt-1 font-mono text-2xl font-bold text-gold-600 dark:text-gold-400 tabular-nums">
                    ${forecast.predictedPrice.toFixed(2)}
                  </div>
                  <div
                    className={`mt-0.5 inline-flex items-center gap-1 font-mono text-xs font-semibold tabular-nums ${
                      forecast.deltaDollar > 0.005
                        ? 'text-[#3fb27f]'
                        : forecast.deltaDollar < -0.005
                        ? 'text-[#e0655f]'
                        : 'text-zinc-500 dark:text-zinc-400'
                    }`}
                  >
                    {forecast.deltaDollar > 0.005 ? (
                      <TrendingUp className="w-3.5 h-3.5" />
                    ) : forecast.deltaDollar < -0.005 ? (
                      <TrendingDown className="w-3.5 h-3.5" />
                    ) : (
                      <Minus className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {formatSignedDollar(forecast.deltaDollar)} ({formatSignedPct(forecast.deltaPct)})
                    </span>
                  </div>
                </div>

                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    99% Prediction Interval
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                    ${forecast.lower99.toFixed(2)} – ${forecast.upper99.toFixed(2)}
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
                    Width ±${((forecast.upper99 - forecast.lower99) / 2).toFixed(2)} (Student-t)
                  </div>
                </div>
              </div>

              {/* Right 99% Interval Visual Gauge */}
              <div className="lg:col-span-5">
                <PredictionIntervalBar
                  lastPrice={forecast.lastPrice}
                  predictedPrice={forecast.predictedPrice}
                  lower99={forecast.lower99}
                  upper99={forecast.upper99}
                />
              </div>
            </div>

            {/* Live Reconstructed Equation */}
            <div className="pt-3 border-t border-zinc-200/70 dark:border-zinc-800/70 flex flex-col sm:flex-row sm:items-center justify-between gap-2 font-mono text-xs tabular-nums">
              <div className="text-zinc-600 dark:text-zinc-300 overflow-x-auto">
                <span className="text-zinc-400 mr-1.5">Reconstruction:</span>
                {forecast.model === 'multiplicative' ? (
                  <>
                    <span className="text-[#3fb27f]">${forecast.components.trend.predictedValue.toFixed(2)}</span>
                    <span className="text-zinc-400 mx-1">×</span>
                    <span className="text-gold-600 dark:text-gold-400">
                      {forecast.components.cyclical.predictedValue.toFixed(4)}
                    </span>
                    <span className="text-zinc-400 mx-1">×</span>
                    <span className="text-zinc-700 dark:text-zinc-300">
                      {forecast.components.seasonal.predictedValue.toFixed(4)}
                    </span>
                    <span className="text-zinc-400 mx-1">×</span>
                    <span className="text-[#e0655f]">
                      {forecast.components.residual.predictedValue.toFixed(4)}
                    </span>
                    <span className="text-zinc-400 mx-1.5">=</span>
                    <strong className="text-zinc-900 dark:text-zinc-100">
                      ${forecast.predictedPrice.toFixed(2)}
                    </strong>
                  </>
                ) : (
                  <>
                    <span className="text-[#3fb27f]">${forecast.components.trend.predictedValue.toFixed(2)}</span>
                    <span className="text-zinc-400 mx-1">+</span>
                    <span className="text-gold-600 dark:text-gold-400">
                      ({forecast.components.cyclical.predictedValue >= 0 ? '+' : ''}
                      {forecast.components.cyclical.predictedValue.toFixed(2)})
                    </span>
                    <span className="text-zinc-400 mx-1">+</span>
                    <span className="text-zinc-700 dark:text-zinc-300">
                      ({forecast.components.seasonal.predictedValue >= 0 ? '+' : ''}
                      {forecast.components.seasonal.predictedValue.toFixed(2)})
                    </span>
                    <span className="text-zinc-400 mx-1">+</span>
                    <span className="text-[#e0655f]">
                      ({forecast.components.residual.predictedValue >= 0 ? '+' : ''}
                      {forecast.components.residual.predictedValue.toFixed(2)})
                    </span>
                    <span className="text-zinc-400 mx-1.5">=</span>
                    <strong className="text-zinc-900 dark:text-zinc-100">
                      ${forecast.predictedPrice.toFixed(2)}
                    </strong>
                  </>
                )}
              </div>

              {/* Out-of-Sample Holdout Summary Strip */}
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-zinc-500 dark:text-zinc-400 shrink-0">
                <span>
                  OOS RMSE{' '}
                  <strong className="text-zinc-900 dark:text-zinc-100">
                    ${forecast.holdoutPriceRmse.toFixed(2)}
                  </strong>
                </span>
                <span>·</span>
                <span>
                  Naive RW{' '}
                  <strong className="text-zinc-700 dark:text-zinc-300">
                    ${forecast.naiveBaselineRmse.toFixed(2)}
                  </strong>
                </span>
                <span>·</span>
                <span>
                  MAE{' '}
                  <strong className="text-zinc-900 dark:text-zinc-100">
                    ${forecast.holdoutPriceMae.toFixed(2)}
                  </strong>
                </span>
                <span>·</span>
                <span>
                  Dir Acc{' '}
                  <strong
                    className={
                      forecast.directionalAccuracy >= 50
                        ? 'text-[#3fb27f]'
                        : 'text-[#e0655f]'
                    }
                  >
                    {forecast.directionalAccuracy.toFixed(1)}%
                  </strong>
                </span>
                <span>·</span>
                <span>
                  {forecast.holdoutCount}d holdout · {forecast.trainingDurationMs}ms
                </span>
              </div>
            </div>
          </div>

          {/* Component Contribution Breakdown Table + Diverging Waterfall */}
          <div className="panel-inset p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5 text-gold-500" />
                <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  Component Forecast &amp; Signed Attribution Breakdown
                </span>
              </div>
              <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                Click any component row to inspect its attention &amp; loss diagnostics
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left font-mono tabular-nums">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-800 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                    <th className="py-2 pr-3 font-sans font-medium">Component</th>
                    <th className="py-2 px-2 text-right">Current (t)</th>
                    <th className="py-2 px-2 text-right">Predicted (t+1)</th>
                    <th className="py-2 px-2 text-right">99% Band</th>
                    <th className="py-2 px-2 text-right">Attrib ($)</th>
                    <th className="py-2 px-2 text-right">Attrib (%)</th>
                    <th className="py-2 pl-2 text-right">OOS RMSE</th>
                    <th className="py-2 pl-3 w-36 hidden md:table-cell">Move Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-200/60 dark:divide-zinc-800/60">
                  {COMPONENT_KEYS.map((key) => {
                    const comp = forecast.components[key];
                    const meta = COMPONENT_META[key];
                    const isSelected = diagComponent === key;
                    const barPct = Math.min(100, (Math.abs(comp.contributionDollar) / maxAbsDollar) * 100);
                    const isPos = comp.contributionDollar >= 0;

                    return (
                      <tr
                        key={key}
                        onClick={() => setDiagComponent(key)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-gold-500/10'
                            : 'hover:bg-zinc-100/70 dark:hover:bg-zinc-800/40'
                        }`}
                      >
                        <td className="py-2 pr-3 font-sans">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: meta.color }}
                            />
                            <span className="font-medium text-zinc-900 dark:text-zinc-100">
                              {meta.label}
                            </span>
                          </div>
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-300">
                          {formatComponentValue(key, comp.currentValue, forecast.model)}
                        </td>
                        <td className="py-2 px-2 text-right font-semibold text-zinc-900 dark:text-zinc-100">
                          {formatComponentValue(key, comp.predictedValue, forecast.model)}
                        </td>
                        <td className="py-2 px-2 text-right text-zinc-500 dark:text-zinc-400">
                          [{formatComponentValue(key, comp.lower99, forecast.model)},{' '}
                          {formatComponentValue(key, comp.upper99, forecast.model)}]
                        </td>
                        <td
                          className={`py-2 px-2 text-right font-semibold ${
                            isPos ? 'text-[#3fb27f]' : 'text-[#e0655f]'
                          }`}
                        >
                          {formatSignedDollar(comp.contributionDollar)}
                        </td>
                        <td
                          className={`py-2 px-2 text-right ${
                            isPos ? 'text-[#3fb27f]' : 'text-[#e0655f]'
                          }`}
                        >
                          {formatSignedPct(comp.contributionPct)}
                        </td>
                        <td className="py-2 pl-2 text-right text-zinc-500 dark:text-zinc-400">
                          {key === 'trend'
                            ? `$${comp.holdoutRmse.toFixed(2)}`
                            : forecast.model === 'multiplicative'
                            ? comp.holdoutRmse.toFixed(4)
                            : comp.holdoutRmse.toFixed(2)}
                        </td>
                        <td className="py-2 pl-3 hidden md:table-cell">
                          <div className="flex items-center h-3 w-full">
                            <div className="w-1/2 flex justify-end pr-0.5">
                              {!isPos && (
                                <div
                                  className="h-2 rounded-l-xs bg-[#e0655f]"
                                  style={{ width: `${barPct}%` }}
                                />
                              )}
                            </div>
                            <div className="w-px h-3 bg-zinc-300 dark:bg-zinc-700 shrink-0" />
                            <div className="w-1/2 flex justify-start pl-0.5">
                              {isPos && (
                                <div
                                  className="h-2 rounded-r-xs bg-[#3fb27f]"
                                  style={{ width: `${barPct}%` }}
                                />
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
                <tfoot>
                  <tr className="border-t border-zinc-300 dark:border-zinc-700 font-semibold">
                    <td className="py-2 pr-3 font-sans text-zinc-900 dark:text-zinc-100">
                      Reconstructed Total (Ŷ<sub>t+1</sub>)
                    </td>
                    <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-300">
                      ${forecast.lastPrice.toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-right text-gold-600 dark:text-gold-400">
                      ${forecast.predictedPrice.toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-right text-zinc-500 dark:text-zinc-400">
                      [${forecast.lower99.toFixed(2)}, ${forecast.upper99.toFixed(2)}]
                    </td>
                    <td
                      className={`py-2 px-2 text-right ${
                        forecast.deltaDollar >= 0 ? 'text-[#3fb27f]' : 'text-[#e0655f]'
                      }`}
                    >
                      {formatSignedDollar(forecast.deltaDollar)}
                    </td>
                    <td
                      className={`py-2 px-2 text-right ${
                        forecast.deltaPct >= 0 ? 'text-[#3fb27f]' : 'text-[#e0655f]'
                      }`}
                    >
                      {formatSignedPct(forecast.deltaPct)}
                    </td>
                    <td className="py-2 pl-2 text-right text-zinc-700 dark:text-zinc-300">
                      ${forecast.holdoutPriceRmse.toFixed(2)}
                    </td>
                    <td className="py-2 pl-3 hidden md:table-cell text-[10px] text-zinc-400">
                      Exact LMDI sum
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>

          {/* Lookback Sensitivity & Model Stability Suite */}
          {sensitivityEnabled && (
            <div className="panel-inset p-4 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-gold-500 shrink-0" />
                    <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      Lookback Window Sensitivity &amp; Model Stability
                    </span>
                    {sensitivityResult && (
                      <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
                        · {sensitivityResult.runs.length} windows · step ±{sensitivityResult.step}d ·{' '}
                        {sensitivityResult.summary.totalDurationMs}ms
                        {isSensitivityStale ? ' · out of date' : ''}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                    Trains identical TCN-Attention architectures across neighboring lookback windows to isolate context-length sensitivity
                  </p>
                </div>

                {sensitivityResult && (
                  <div
                    className={`inline-flex items-center gap-1.5 font-mono text-xs font-semibold shrink-0 ${
                      sensitivityResult.summary.stabilityLabel === 'Stable'
                        ? 'text-[#3fb27f]'
                        : sensitivityResult.summary.stabilityLabel === 'Moderate'
                        ? 'text-gold-600 dark:text-gold-400'
                        : 'text-[#e0655f]'
                    }`}
                  >
                    {sensitivityResult.summary.stabilityLabel === 'Stable' ? (
                      <ShieldCheck className="w-4 h-4 shrink-0" />
                    ) : sensitivityResult.summary.stabilityLabel === 'Moderate' ? (
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                    ) : (
                      <AlertCircle className="w-4 h-4 shrink-0" />
                    )}
                    <span>{sensitivityResult.summary.stabilityLabel} Stability</span>
                  </div>
                )}
              </div>

              {!sensitivityResult ? (
                <div className="py-5 px-4 rounded-lg border border-dashed border-zinc-300 dark:border-zinc-700/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="text-zinc-600 dark:text-zinc-400">
                    <strong className="text-zinc-900 dark:text-zinc-100">
                      Sensitivity sweep ready ({plannedSweepGrid.map((v) => `${v}d`).join(', ')}):
                    </strong>{' '}
                    Click <strong className="text-zinc-900 dark:text-zinc-100">Run {plannedSweepGrid.length}-Window Sweep</strong> to evaluate forecast dispersion, directional agreement, and per-component stability.
                  </div>
                  <button
                    type="button"
                    disabled={isTraining}
                    onClick={onTrain}
                    className="inline-flex items-center justify-center gap-1.5 h-8 px-3 rounded-md border border-gold-500/60 bg-gold-500/10 text-gold-700 dark:text-gold-300 font-medium text-xs hover:bg-gold-500/20 transition-colors cursor-pointer whitespace-nowrap shrink-0 disabled:opacity-50"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    <span>Run {plannedSweepGrid.length}-Window Sweep</span>
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Stability KPI Strip */}
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 py-3 border-y border-zinc-200/70 dark:border-zinc-800/70 font-mono tabular-nums">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        Price Range
                      </div>
                      <div className="mt-1 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                        ${sensitivityResult.summary.minPrice.toFixed(2)} – $
                        {sensitivityResult.summary.maxPrice.toFixed(2)}
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                        Spread ${sensitivityResult.summary.priceRangeDollar.toFixed(2)} (
                        {sensitivityResult.summary.priceRangePct.toFixed(2)}%)
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        Cross-Window σ
                      </div>
                      <div className="mt-1 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                        ±${sensitivityResult.summary.stdDevPrice.toFixed(2)}
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                        CV {sensitivityResult.summary.cvPct.toFixed(2)}% · Mean $
                        {sensitivityResult.summary.meanPrice.toFixed(2)}
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        σ / 99% PI Ratio
                      </div>
                      <div
                        className={`mt-1 text-sm font-semibold ${
                          sensitivityResult.summary.dispersionToPiRatio <= 0.25
                            ? 'text-[#3fb27f]'
                            : sensitivityResult.summary.dispersionToPiRatio <= 0.55
                            ? 'text-gold-600 dark:text-gold-400'
                            : 'text-[#e0655f]'
                        }`}
                      >
                        {(sensitivityResult.summary.dispersionToPiRatio * 100).toFixed(1)}%
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                        of base 99% half-width
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        Directional Consensus
                      </div>
                      <div
                        className={`mt-1 text-sm font-semibold inline-flex items-center gap-1 ${
                          sensitivityResult.summary.dominantDirection === 'up'
                            ? 'text-[#3fb27f]'
                            : sensitivityResult.summary.dominantDirection === 'down'
                            ? 'text-[#e0655f]'
                            : 'text-zinc-500 dark:text-zinc-400'
                        }`}
                      >
                        {sensitivityResult.summary.dominantDirection === 'up' ? (
                          <TrendingUp className="w-3.5 h-3.5" />
                        ) : sensitivityResult.summary.dominantDirection === 'down' ? (
                          <TrendingDown className="w-3.5 h-3.5" />
                        ) : (
                          <Minus className="w-3.5 h-3.5" />
                        )}
                        <span>
                          {Math.max(
                            sensitivityResult.summary.upCount,
                            sensitivityResult.summary.downCount
                          )}
                          /{sensitivityResult.runs.length}{' '}
                          {sensitivityResult.summary.dominantDirection === 'up'
                            ? 'Up'
                            : sensitivityResult.summary.dominantDirection === 'down'
                            ? 'Down'
                            : 'Split'}
                        </span>
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                        {sensitivityResult.summary.directionalAgreementPct.toFixed(0)}% agreement
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                        Best OOS RMSE
                      </div>
                      <div className="mt-1 text-sm font-semibold text-gold-600 dark:text-gold-400">
                        L = {sensitivityResult.summary.bestRmseLookback}d
                      </div>
                      <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                        RMSE ${sensitivityResult.summary.bestRmseValue.toFixed(2)}
                      </div>
                    </div>
                  </div>

                  {/* Pure SVG Lookback-vs-Price Stability Chart */}
                  <LookbackSensitivityChart sensitivity={sensitivityResult} />

                  {/* Per-Component Stability Breakdown Table */}
                  <div className="pt-3 border-t border-zinc-200/70 dark:border-zinc-800/70 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                        Per-Component Lookback Stability Breakdown
                      </span>
                      <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                        Variance Share attributes cross-window price drift across T, C, S, I
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left font-mono tabular-nums">
                        <thead>
                          <tr className="border-b border-zinc-200 dark:border-zinc-800 text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                            <th className="py-2 pr-3 font-sans font-medium">Component</th>
                            <th className="py-2 px-2 text-right">Base (L={sensitivityResult.baseLookback}d)</th>
                            <th className="py-2 px-2 text-right">Sweep Mean</th>
                            <th className="py-2 px-2 text-right">Min – Max</th>
                            <th className="py-2 px-2 text-right">Std Dev (σ)</th>
                            <th className="py-2 px-2 text-right">Attrib Spread ($)</th>
                            <th className="py-2 pl-3 w-40">Variance Share</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-200/60 dark:divide-zinc-800/60">
                          {COMPONENT_KEYS.map((key) => {
                            const cs = sensitivityResult.componentStability[key];
                            const meta = COMPONENT_META[key];
                            const isSelected = diagComponent === key;
                            return (
                              <tr
                                key={key}
                                onClick={() => setDiagComponent(key)}
                                className={`cursor-pointer transition-colors ${
                                  isSelected
                                    ? 'bg-gold-500/10'
                                    : 'hover:bg-zinc-100/70 dark:hover:bg-zinc-800/40'
                                }`}
                              >
                                <td className="py-2 pr-3 font-sans font-medium text-zinc-900 dark:text-zinc-100">
                                  {meta.label}
                                </td>
                                <td className="py-2 px-2 text-right text-zinc-700 dark:text-zinc-200">
                                  {formatComponentValue(key, cs.baseValue, sensitivityResult.model)}
                                </td>
                                <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-300">
                                  {formatComponentValue(key, cs.meanValue, sensitivityResult.model)}
                                </td>
                                <td className="py-2 px-2 text-right text-zinc-500 dark:text-zinc-400">
                                  {formatComponentValue(key, cs.minValue, sensitivityResult.model)} –{' '}
                                  {formatComponentValue(key, cs.maxValue, sensitivityResult.model)}
                                </td>
                                <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-300">
                                  {key === 'trend'
                                    ? `±$${cs.stdDevValue.toFixed(2)}`
                                    : sensitivityResult.model === 'multiplicative'
                                    ? `±${cs.stdDevValue.toFixed(4)}`
                                    : `±${cs.stdDevValue.toFixed(2)}`}
                                </td>
                                <td className="py-2 px-2 text-right text-zinc-900 dark:text-zinc-100">
                                  ${cs.contribDollarSpread.toFixed(2)}
                                </td>
                                <td className="py-2 pl-3">
                                  <div className="flex items-center gap-2">
                                    <div className="flex-1 h-1.5 rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
                                      <div
                                        className="h-full rounded-full"
                                        style={{
                                          width: `${Math.max(2, Math.min(100, cs.varianceSharePct))}%`,
                                          backgroundColor: meta.color,
                                        }}
                                      />
                                    </div>
                                    <span className="w-10 text-right text-[11px] text-zinc-700 dark:text-zinc-300">
                                      {cs.varianceSharePct.toFixed(0)}%
                                    </span>
                                  </div>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                        <tfoot>
                          <tr className="border-t border-zinc-300 dark:border-zinc-700 font-semibold">
                            <td className="py-2 pr-3 font-sans text-zinc-900 dark:text-zinc-100">
                              Reconstructed Price (Ŷ<sub>t+1</sub>)
                            </td>
                            <td className="py-2 px-2 text-right text-gold-600 dark:text-gold-400">
                              ${sensitivityResult.baseForecast.predictedPrice.toFixed(2)}
                            </td>
                            <td className="py-2 px-2 text-right text-zinc-900 dark:text-zinc-100">
                              ${sensitivityResult.summary.meanPrice.toFixed(2)}
                            </td>
                            <td className="py-2 px-2 text-right text-zinc-600 dark:text-zinc-300">
                              ${sensitivityResult.summary.minPrice.toFixed(2)} – $
                              {sensitivityResult.summary.maxPrice.toFixed(2)}
                            </td>
                            <td className="py-2 px-2 text-right text-zinc-900 dark:text-zinc-100">
                              ±${sensitivityResult.summary.stdDevPrice.toFixed(2)}
                            </td>
                            <td className="py-2 px-2 text-right text-zinc-900 dark:text-zinc-100">
                              ${sensitivityResult.summary.priceRangeDollar.toFixed(2)}
                            </td>
                            <td className="py-2 pl-3 text-right text-[11px] text-zinc-500 dark:text-zinc-400">
                              100% total
                            </td>
                          </tr>
                        </tfoot>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 2-Column Diagnostics: Self-Attention Lag Weights + Loss Convergence Curves */}
          {activeDiagComp && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Attention Weights */}
              <div className="panel-inset p-4">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <div>
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      <Activity className="w-3.5 h-3.5 text-gold-500" />
                      <span>Causal Self-Attention Weights</span>
                    </div>
                    <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                      Top-3 attended historical lags highlighted in gold (sums to 100%)
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    {/* Component switcher */}
                    <div className="inline-flex rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-0.5 text-[10px] font-mono">
                      {COMPONENT_KEYS.map((k) => (
                        <button
                          key={k}
                          type="button"
                          onClick={() => setDiagComponent(k)}
                          className={`px-2 py-0.5 rounded-xs cursor-pointer transition-colors ${
                            diagComponent === k
                              ? 'bg-gold-500/20 text-gold-700 dark:text-gold-300 font-bold'
                              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                          }`}
                        >
                          {COMPONENT_META[k].short}
                        </button>
                      ))}
                    </div>

                    {/* Head selector */}
                    <select
                      aria-label="Attention head filter"
                      value={safeDiagHead === 'mean' ? 'mean' : String(safeDiagHead)}
                      onChange={(e) =>
                        setDiagHead(e.target.value === 'mean' ? 'mean' : parseInt(e.target.value, 10))
                      }
                      className="h-6 px-1.5 rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-[10px] font-mono text-zinc-700 dark:text-zinc-300"
                    >
                      <option value="mean">Mean</option>
                      {activeDiagComp.attentionByHead.map((_, hIdx) => (
                        <option key={hIdx} value={hIdx}>
                          Head {hIdx + 1}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <AttentionBarChart comp={activeDiagComp} selectedHead={safeDiagHead} />
              </div>

              {/* Training & Holdout Loss Convergence */}
              <div className="panel-inset p-4">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <div>
                    <div className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      Training vs. Holdout Loss — {COMPONENT_META[diagComponent].label}
                    </div>
                    <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                      {activeDiagComp.trainWindows} train · {activeDiagComp.holdoutWindows} holdout
                      windows · Best holdout checkpoint restored
                    </p>
                  </div>
                </div>

                <LossCurveChart comp={activeDiagComp} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
