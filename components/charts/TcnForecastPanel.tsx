'use client';

import React, { useEffect, useMemo, useState } from 'react';
import type { Data, Layout, Shape } from 'plotly.js';
import type { DecompositionResult } from '@/lib/quant-engine';
import {
  DEFAULT_TCN_CONFIG,
  MIN_BARS_FOR_TCN,
  computeHistoricalBacktest,
  computeParameterCount,
  computeReceptiveField,
  generateLookbackSweepGrid,
  sanitizeTcnConfig,
  type BacktestCostBps,
  type ComponentForecast,
  type DecompositionComponentKey,
  type HistoricalBacktestResult,
  type SensitivityStep,
  type TcnAttentionConfig,
  type TcnDecompositionForecast,
  type TcnSensitivityResult,
  type TcnTrainingProgress,
} from '@/lib/tcn-attention';
import { TcnAttentionEquation } from '@/components/math/MathEquations';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { getHmmRegimeToken } from '@/lib/hmm-regime-overlay';
import {
  HmmRegimeOverlayToggle,
  HmmRegimeScorecard,
} from '@/components/charts/HmmRegimeOverlayControls';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import {
  buildHmmRegimeShapes,
  getPlotlyThemeTokens,
  QUANT_COLORS,
} from '@/lib/plotly-theme';
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
  Radio,
  History,
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

const BACKTEST_COST_OPTIONS: BacktestCostBps[] = [0, 5, 10, 25];

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

/** Interactive Plotly Attention Weight Bar Chart across lookback lags */
function AttentionBarChart({
  comp,
  selectedHead,
}: {
  comp: ComponentForecast;
  selectedHead: 'mean' | number;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const weights = useMemo(
    () =>
      selectedHead === 'mean'
        ? comp.attentionWeights
        : comp.attentionByHead[selectedHead] ?? comp.attentionWeights,
    [comp.attentionWeights, comp.attentionByHead, selectedHead]
  );

  const { traces, layout, topLagLabel, topWeightPct } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    topLagLabel: string;
    topWeightPct: number;
  }>(() => {
    const L = weights.length;
    if (L === 0) return { traces: [], layout: {}, topLagLabel: 't', topWeightPct: 0 };

    const sortedIndices = weights
      .map((w, idx) => ({ w, idx }))
      .sort((a, b) => b.w - a.w)
      .slice(0, 3);
    const topSet = new Set(sortedIndices.map((x) => x.idx));

    const lagOffsets = weights.map((_, idx) => -(L - 1 - idx));
    const labels = lagOffsets.map((off) => (off === 0 ? 't' : `t${off}`));
    const pcts = weights.map((w) => w * 100);
    const colors = weights.map((_, idx) =>
      topSet.has(idx) ? QUANT_COLORS.goldLight : 'rgba(113, 113, 122, 0.55)'
    );

    const bestIdx = sortedIndices[0]?.idx ?? L - 1;
    const bestOff = L - 1 - bestIdx;

    return {
      topLagLabel: bestOff === 0 ? 't' : `t-${bestOff}`,
      topWeightPct: (weights[bestIdx] ?? 0) * 100,
      traces: [
        {
          type: 'bar',
          name: 'Attention Weight',
          x: lagOffsets,
          y: pcts,
          customdata: labels,
          marker: { color: colors },
          hovertemplate: 'Lag %{customdata} · %{y:.1f}%<extra>Attention</extra>',
        },
      ],
      layout: {
        margin: { l: 40, r: 10, t: 8, b: 28 },
        bargap: 0.22,
        xaxis: {
          title: { text: 'Lag offset', font: { size: 9.5, color: tokens.textMuted } },
          gridcolor: tokens.grid,
        },
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          nticks: 4,
          gridcolor: tokens.grid,
        },
      },
    };
  }, [weights, tokens]);

  if (weights.length === 0) return null;

  return (
    <div>
      <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 dark:text-zinc-400 mb-1.5 tabular-nums">
        <span>
          Peak lag:{' '}
          <strong className="text-gold-600 dark:text-gold-400">{topLagLabel}</strong>
        </span>
        <span>
          Weight:{' '}
          <strong className="text-zinc-900 dark:text-zinc-100">{topWeightPct.toFixed(1)}%</strong>
        </span>
      </div>

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        height={170}
        exportFilename="quantmetrics-tcn-attention"
        uirevisionKey={`att-${weights.length}-${String(selectedHead)}`}
      />
    </div>
  );
}

/** Interactive Plotly Dual-Line Train vs Holdout Loss Convergence Curve */
function LossCurveChart({ comp }: { comp: ComponentForecast }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const train = comp.trainLossCurve;
  const holdout = comp.holdoutLossCurve;
  const n = train.length;

  const { traces, layout, lastTrain, lastHold } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    lastTrain: number;
    lastHold: number;
  }>(() => {
    if (n === 0) return { traces: [], layout: {}, lastTrain: 0, lastHold: 0 };
    const epochs = Array.from({ length: n }, (_, i) => i + 1);
    const bestEpoch = Math.max(1, Math.min(n, comp.bestEpoch));

    return {
      lastTrain: train[n - 1] ?? 0,
      lastHold: holdout[bestEpoch - 1] ?? holdout[holdout.length - 1] ?? 0,
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Holdout Loss',
          x: epochs,
          y: holdout,
          line: { color: QUANT_COLORS.positive, width: 1.8 },
          hovertemplate: '%{y:.4f}<extra>Holdout</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Train Loss',
          x: epochs,
          y: train,
          line: { color: QUANT_COLORS.goldLight, width: 1.8 },
          hovertemplate: '%{y:.4f}<extra>Train</extra>',
        },
      ],
      layout: {
        margin: { l: 42, r: 10, t: 8, b: 28 },
        shapes: [
          {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: bestEpoch,
            x1: bestEpoch,
            y0: 0,
            y1: 1,
            line: { color: QUANT_COLORS.positive, width: 1.1, dash: 'dot' },
          },
        ],
        xaxis: {
          title: { text: 'Epoch', font: { size: 9.5, color: tokens.textMuted } },
          gridcolor: tokens.grid,
        },
        yaxis: {
          nticks: 4,
          gridcolor: tokens.grid,
        },
      },
    };
  }, [train, holdout, n, comp.bestEpoch, tokens]);

  if (n === 0) return null;

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

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        height={170}
        exportFilename="quantmetrics-tcn-loss"
        uirevisionKey={`loss-${n}-${comp.bestEpoch}`}
      />
    </div>
  );
}

/** Interactive Plotly Lookback-vs-Price Stability Chart with 99% PI Error Whiskers */
function LookbackSensitivityChart({
  sensitivity,
}: {
  sensitivity: TcnSensitivityResult;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);

  const { runs, summary, baseForecast } = sensitivity;
  const k = runs.length;
  const baseIdx = Math.max(0, runs.findIndex((r) => r.isBase));
  const safePinned = pinnedIdx !== null && pinnedIdx < k ? pinnedIdx : null;
  const activeIdx = safePinned ?? baseIdx;
  const activeRun = runs[activeIdx];
  const lastPrice = baseForecast.lastPrice;

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (k === 0) return { traces: [], layout: {} };
    const xs = runs.map((r) => r.lookback);
    const ys = runs.map((r) => r.forecast.predictedPrice);
    const errPlus = runs.map((r) => Math.max(0, r.forecast.upper99 - r.forecast.predictedPrice));
    const errMinus = runs.map((r) => Math.max(0, r.forecast.predictedPrice - r.forecast.lower99));
    const colors = runs.map((r) =>
      r.isBase
        ? QUANT_COLORS.goldLight
        : r.forecast.deltaDollar >= 0
        ? QUANT_COLORS.positive
        : QUANT_COLORS.negative
    );
    const sizes = runs.map((r) =>
      r.lookback === summary.bestRmseLookback ? 11 : r.isBase ? 10 : 8
    );
    const custom = runs.map((r) => [
      r.forecast.lower99.toFixed(2),
      r.forecast.upper99.toFixed(2),
      r.forecast.holdoutPriceRmse.toFixed(2),
      r.isBase ? ' (base)' : r.lookback === summary.bestRmseLookback ? ' (min RMSE)' : '',
    ]);

    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Predicted Price (99% PI)',
          x: xs,
          y: ys,
          customdata: custom,
          line: { color: QUANT_COLORS.gold, width: 1.7 },
          marker: {
            color: colors,
            size: sizes,
            line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1.4 },
          },
          error_y: {
            type: 'data',
            symmetric: false,
            array: errPlus,
            arrayminus: errMinus,
            color: 'rgba(222, 192, 132, 0.72)',
            thickness: 1.5,
            width: 5,
          },
          hovertemplate:
            'L=%{x}d%{customdata[3]}<br>Pred: $%{y:.2f}<br>99% PI: [$%{customdata[0]}, $%{customdata[1]}]<br>OOS RMSE: $%{customdata[2]}<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 56, r: 16, t: 12, b: 36 },
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: lastPrice,
            y1: lastPrice,
            line: { color: tokens.textMuted, width: 1.1, dash: 'dash' },
          },
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: summary.meanPrice,
            y1: summary.meanPrice,
            line: { color: QUANT_COLORS.goldLight, width: 1.1, dash: 'dot' },
          },
        ],
        annotations: [
          {
            xref: 'paper',
            yref: 'y',
            x: 1,
            y: lastPrice,
            text: `Last $${lastPrice.toFixed(2)}`,
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { size: 9.5, color: tokens.textMuted },
          },
        ],
        xaxis: {
          title: { text: 'Lookback Window L (days)', font: { size: 10, color: tokens.textMuted } },
          ticksuffix: 'd',
          gridcolor: tokens.grid,
        },
        yaxis: {
          tickprefix: '$',
          gridcolor: tokens.grid,
        },
      },
    };
  }, [k, runs, summary.bestRmseLookback, summary.meanPrice, lastPrice, isDark, tokens]);

  if (k === 0 || !activeRun) return null;

  return (
    <div className="space-y-2">
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

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        height={220}
        exportFilename="quantmetrics-tcn-lookback-sweep"
        uirevisionKey={`sweep-${k}-${summary.meanPrice.toFixed(2)}`}
        onClick={(ev) => {
          const pts = (ev as { points?: Array<{ pointIndex?: number }> })?.points;
          const idx = pts?.[0]?.pointIndex;
          if (typeof idx === 'number' && idx >= 0 && idx < k) {
            setPinnedIdx(idx);
          }
        }}
        showWheelHint
      />

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
            <span className="w-3 border-t border-dotted border-[#DEC084] inline-block" />
            Sweep mean ${summary.meanPrice.toFixed(2)}
          </span>
        </div>
        <span>Click any lookback marker to pin readout</span>
      </div>
    </div>
  );
}

/** Interactive Plotly 30-Day Signal-Weighted Cumulative P&L vs Buy & Hold Chart with Conviction Bars & Causal HMM Overlay */
function HistoricalBacktestChart({
  backtest,
}: {
  backtest: HistoricalBacktestResult;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState(true);
  const { days, costBps, regimeBreakdown } = backtest;
  const n = days.length;

  const hasHmmRegimes = days.some(
    (d) => d.hmmState === 0 || d.hmmState === 1 || d.hmmState === 2
  );

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (n < 2) return { traces: [], layout: {} };
    const dates = days.map((d) => d.date);
    const stratCum = days.map((d) => d.strategyCumPct);
    const benchCum = days.map((d) => d.benchmarkCumPct);
    const weightsPct = days.map((d) => d.weight * 100);
    const weightColors = days.map((d) =>
      d.weight >= 0 ? 'rgba(63, 178, 127, 0.72)' : 'rgba(224, 101, 95, 0.72)'
    );

    const dayMeta = days.map((d) => {
      const tok = d.hmmState !== undefined ? getHmmRegimeToken(d.hmmState) : null;
      const regimeTxt = tok
        ? `${tok.label}${d.hmmProb !== undefined ? ` (${Math.round(d.hmmProb * 100)}%)` : ''}`
        : '—';
      return [
        formatSignedPct(d.predictedReturn * 100),
        formatSignedPct(d.actualReturn * 100),
        formatSignedPct(d.netReturn * 100),
        regimeTxt,
      ];
    });

    const dataTraces: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Buy & Hold Benchmark',
        x: dates,
        y: benchCum,
        yaxis: 'y',
        line: { color: tokens.textMuted, width: 1.5, dash: 'dash' },
        hovertemplate: '%{y:+.2f}%<extra>Buy &amp; Hold</extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: `TCN Net (${costBps} bps)`,
        x: dates,
        y: stratCum,
        customdata: dayMeta,
        yaxis: 'y',
        line: { color: QUANT_COLORS.goldLight, width: 2.1 },
        fill: 'tozeroy',
        fillcolor: 'rgba(222, 192, 132, 0.14)',
        hovertemplate:
          '<b>%{y:+.2f}%</b> · Day Net %{customdata[2]}<br>Pred %{customdata[0]} · Actual %{customdata[1]} · %{customdata[3]}<extra>TCN Net</extra>',
      },
      {
        type: 'bar',
        name: 'Conviction w_t',
        x: dates,
        y: weightsPct,
        yaxis: 'y2',
        marker: { color: weightColors },
        hovertemplate: '%{y:+.0f}%<extra>Weight w_t</extra>',
      },
    ];

    // Color-coded Daily OOS Prediction Points by Origin HMM Regime
    if (hasHmmRegimes && showRegimeOverlay) {
      const rx: string[] = [];
      const ry: number[] = [];
      const rc: string[] = [];
      const rl: string[] = [];
      for (const d of days) {
        if (d.hmmState === undefined) continue;
        const tok = getHmmRegimeToken(d.hmmState);
        rx.push(d.date);
        ry.push(d.strategyCumPct);
        rc.push(tok.color);
        rl.push(`${tok.label}${d.hmmProb !== undefined ? ` (${Math.round(d.hmmProb * 100)}%)` : ''}`);
      }
      if (rx.length > 0) {
        dataTraces.push({
          type: 'scatter',
          mode: 'markers',
          name: 'Origin HMM State',
          x: rx,
          y: ry,
          customdata: rl,
          yaxis: 'y',
          marker: {
            color: rc,
            size: 6.5,
            line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 },
          },
          showlegend: false,
          hoverinfo: 'skip',
        });
      }
    }

    const shapes: Partial<Shape>[] = [
      ...buildHmmRegimeShapes(
        dates,
        days.map((d) => d.hmmState),
        hasHmmRegimes && showRegimeOverlay,
        0.13
      ),
      {
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: 0,
        y1: 0,
        line: { color: tokens.zeroLine, width: 1 },
      },
    ];

    return {
      traces: dataTraces,
      layout: {
        shapes,
        bargap: 0.25,
        yaxis: {
          domain: [0.28, 1],
          ticksuffix: '%',
          gridcolor: tokens.grid,
        },
        yaxis2: {
          domain: [0, 0.2],
          ticksuffix: '%',
          nticks: 3,
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
          gridcolor: tokens.grid,
          fixedrange: true,
        },
      },
    };
  }, [n, days, costBps, hasHmmRegimes, showRegimeOverlay, isDark, tokens]);

  if (n < 2) return null;

  return (
    <div className="space-y-2.5">
      {hasHmmRegimes && (
        <HmmRegimeOverlayToggle
          enabled={showRegimeOverlay}
          onToggle={() => setShowRegimeOverlay((v) => !v)}
          caption="origin t−1 regime"
        />
      )}

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider
        hovermode="x unified"
        showLegend
        height={310}
        exportFilename={`quantmetrics-tcn-backtest-${costBps}bps`}
        uirevisionKey={`bt-${n}-${costBps}`}
        showWheelHint
      />

      {/* Per-Regime 30-Day Backtest Accuracy Scorecard */}
      {hasHmmRegimes && regimeBreakdown.length > 0 && (
        <HmmRegimeScorecard
          title={`30-Day Backtest Performance by Causal HMM Regime (${costBps} bps)`}
          subtitle="Stratified by dominant HMM regime at t−1 prediction origin"
          columns={[
            { key: 'hit', label: 'Dir Hit Rate' },
            { key: 'rmse', label: 'Price RMSE' },
            { key: 'meanNet', label: 'Mean Net / Day' },
            { key: 'cumNet', label: 'Cum Net P&L' },
          ]}
          rows={regimeBreakdown.map((r) => ({
            state: r.state,
            count: r.count,
            cells: {
              hit:
                r.directionalHitRatePct !== null
                  ? `${r.directionalHitRatePct.toFixed(0)}%`
                  : '—',
              rmse: r.rmseDollar !== null ? `$${r.rmseDollar.toFixed(2)}` : '—',
              meanNet:
                r.meanNetReturnPct !== null ? (
                  <span
                    className={
                      r.meanNetReturnPct >= 0
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-rose-600 dark:text-rose-400'
                    }
                  >
                    {r.meanNetReturnPct >= 0 ? '+' : ''}
                    {r.meanNetReturnPct.toFixed(2)}%
                  </span>
                ) : (
                  '—'
                ),
              cumNet:
                r.cumNetReturnPct !== null ? (
                  <span
                    className={
                      r.cumNetReturnPct >= 0
                        ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                        : 'text-rose-600 dark:text-rose-400 font-semibold'
                    }
                  >
                    {r.cumNetReturnPct >= 0 ? '+' : ''}
                    {r.cumNetReturnPct.toFixed(2)}%
                  </span>
                ) : (
                  '—'
                ),
            },
          }))}
        />
      )}
    </div>
  );
}

const persistedPanelUiState: {
  diagComponent: DecompositionComponentKey;
  diagHead: 'mean' | number;
  backtestVisible: boolean;
  backtestCostBps: BacktestCostBps;
} = {
  diagComponent: 'trend',
  diagHead: 'mean',
  backtestVisible: true,
  backtestCostBps: 0,
};

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
  const [diagComponent, setDiagComponent] = useState<DecompositionComponentKey>(
    () => persistedPanelUiState.diagComponent
  );
  const [diagHead, setDiagHead] = useState<'mean' | number>(
    () => persistedPanelUiState.diagHead
  );
  const [backtestVisible, setBacktestVisible] = useState<boolean>(
    () => persistedPanelUiState.backtestVisible
  );
  const [backtestCostBps, setBacktestCostBps] = useState<BacktestCostBps>(
    () => persistedPanelUiState.backtestCostBps
  );

  useEffect(() => {
    persistedPanelUiState.diagComponent = diagComponent;
    persistedPanelUiState.diagHead = diagHead;
    persistedPanelUiState.backtestVisible = backtestVisible;
    persistedPanelUiState.backtestCostBps = backtestCostBps;
  }, [diagComponent, diagHead, backtestVisible, backtestCostBps]);

  const nObs = decomposition.observed.length;
  const backtest = forecast
    ? computeHistoricalBacktest(forecast.backtestSignals, backtestCostBps)
    : null;
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
              · RF {receptiveField}d · {paramCount.toLocaleString()} params/comp · {effectiveConfig.numHeads}H · 6ch (+HMM)
              {forecast?.hmmRegime
                ? ` · HMM ${forecast.hmmRegime.dominantLabel} (${Math.round(forecast.hmmRegime.probs[0] * 100)}/${Math.round(forecast.hmmRegime.probs[1] * 100)}/${Math.round(forecast.hmmRegime.probs[2] * 100)}%)`
                : ''}
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

      {/* Manual-training empty state */}
      {!insufficientData && !forecast && !isTraining && (
        <div className="panel-inset p-4 flex items-start gap-2.5 text-xs text-zinc-600 dark:text-zinc-400">
          <AlertCircle className="w-4 h-4 text-gold-500 shrink-0 mt-0.5" />
          <div>
            <strong className="text-zinc-900 dark:text-zinc-100">No cached model for this setup yet.</strong>{' '}
            A fit with the default parameters starts automatically once live bars are available (synthetic history
            is never auto-trained). You can also press <span className="font-mono text-gold-700 dark:text-gold-300">Train &amp; Predict Next-Day</span>{' '}
            to fit on the current {nObs} observations. Live quote ticks never start a fit.
            {isStale && ' Cached parameters differ from the current parameters — retrain to refresh.'}
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
                  SNR{' '}
                  <strong
                    className={
                      forecast.signalToNoise.regime === 'Strong Signal'
                        ? 'text-[#3fb27f]'
                        : forecast.signalToNoise.regime === 'Moderate Signal'
                        ? 'text-gold-600 dark:text-gold-400'
                        : 'text-[#e0655f]'
                    }
                  >
                    {forecast.signalToNoise.ratio >= 100
                      ? '>99×'
                      : `${forecast.signalToNoise.ratio.toFixed(2)}×`}
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

          {/* Signal-to-Noise Ratio (SNR) Diagnostic Indicator */}
          <div className="panel-inset p-4 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                  <Radio className="w-3.5 h-3.5 text-gold-500 shrink-0" />
                  <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    Signal-to-Noise Ratio (Cyclical vs. Residuals)
                  </span>
                  <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
                    · Var(Ĉ<sub>t</sub>) / Var(I<sub>t</sub>) ·{' '}
                    {forecast.model === 'multiplicative' ? 'log-factor space' : 'additive $² space'} ·{' '}
                    {forecast.signalToNoise.sampleCount}d holdout
                  </span>
                </div>
                <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                  Measures variance captured by the TCN-predicted cyclical wave relative to irregular residual noise on out-of-sample dates
                </p>
              </div>

              <div
                className={`inline-flex items-center gap-1.5 font-mono text-xs font-semibold shrink-0 ${
                  forecast.signalToNoise.regime === 'Strong Signal'
                    ? 'text-[#3fb27f]'
                    : forecast.signalToNoise.regime === 'Moderate Signal'
                    ? 'text-gold-600 dark:text-gold-400'
                    : 'text-[#e0655f]'
                }`}
              >
                {forecast.signalToNoise.regime === 'Strong Signal' ? (
                  <Radio className="w-3.5 h-3.5 shrink-0" />
                ) : forecast.signalToNoise.regime === 'Moderate Signal' ? (
                  <Activity className="w-3.5 h-3.5 shrink-0" />
                ) : (
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                )}
                <span>{forecast.signalToNoise.regime}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center pt-2.5 border-t border-zinc-200/70 dark:border-zinc-800/70 font-mono tabular-nums">
              {/* Ratio & Decibels */}
              <div className="sm:col-span-3">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Cyclical SNR
                </div>
                <div
                  className={`mt-1 text-xl font-bold ${
                    forecast.signalToNoise.regime === 'Strong Signal'
                      ? 'text-[#3fb27f]'
                      : forecast.signalToNoise.regime === 'Moderate Signal'
                      ? 'text-gold-600 dark:text-gold-400'
                      : 'text-[#e0655f]'
                  }`}
                >
                  {forecast.signalToNoise.ratio >= 100
                    ? '>99×'
                    : `${forecast.signalToNoise.ratio.toFixed(2)}×`}
                </div>
                <div className="mt-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                  {forecast.signalToNoise.db >= 0 ? '+' : ''}
                  {forecast.signalToNoise.db.toFixed(2)} dB (0 dB = 1.0×)
                </div>
              </div>

              {/* Signal Variance */}
              <div className="sm:col-span-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Var(Ĉ<sub>t</sub>) Signal
                </div>
                <div className="mt-1 text-sm font-semibold text-gold-600 dark:text-gold-400">
                  {forecast.model === 'multiplicative'
                    ? `${(forecast.signalToNoise.signalVariance * 10000).toFixed(2)} bps²`
                    : `$² ${forecast.signalToNoise.signalVariance.toFixed(3)}`}
                </div>
                <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                  σ ={' '}
                  {forecast.model === 'multiplicative'
                    ? `${(forecast.signalToNoise.signalStdDev * 100).toFixed(2)}%`
                    : `$${forecast.signalToNoise.signalStdDev.toFixed(2)}`}
                </div>
              </div>

              {/* Residual Noise Variance */}
              <div className="sm:col-span-2">
                <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Var(I<sub>t</sub>) Residual
                </div>
                <div className="mt-1 text-sm font-semibold text-[#e0655f]">
                  {forecast.model === 'multiplicative'
                    ? `${(forecast.signalToNoise.noiseVariance * 10000).toFixed(2)} bps²`
                    : `$² ${forecast.signalToNoise.noiseVariance.toFixed(3)}`}
                </div>
                <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                  σ ={' '}
                  {forecast.model === 'multiplicative'
                    ? `${(forecast.signalToNoise.noiseStdDev * 100).toFixed(2)}%`
                    : `$${forecast.signalToNoise.noiseStdDev.toFixed(2)}`}
                </div>
              </div>

              {/* Signal vs Noise Power Split Meter */}
              <div className="sm:col-span-5 space-y-1.5">
                <div className="flex items-center justify-between text-[10px] text-zinc-500 dark:text-zinc-400">
                  <span>
                    Cyclical Signal{' '}
                    <strong className="text-gold-600 dark:text-gold-400">
                      {forecast.signalToNoise.signalSharePct.toFixed(1)}%
                    </strong>
                  </span>
                  <span>
                    Residual Noise{' '}
                    <strong className="text-[#e0655f]">
                      {(100 - forecast.signalToNoise.signalSharePct).toFixed(1)}%
                    </strong>
                  </span>
                </div>
                <div className="relative h-2.5 w-full rounded-full bg-[#e0655f]/35 overflow-hidden">
                  <div
                    className="h-full bg-[#d4a853] rounded-l-full"
                    style={{
                      width: `${Math.max(2, Math.min(98, forecast.signalToNoise.signalSharePct))}%`,
                    }}
                  />
                  {/* 1.0x (0 dB) equilibrium midline */}
                  <div
                    className="absolute top-0 bottom-0 left-1/2 w-px bg-zinc-900 dark:bg-zinc-100 opacity-70"
                    title="1.0× (0 dB) equilibrium"
                  />
                </div>
                <div className="flex items-center justify-between text-[9px] text-zinc-400 dark:text-zinc-500">
                  <span>Noise-Dominated (&lt;0.75×)</span>
                  <span>1.0× (0 dB)</span>
                  <span>Strong (≥2.0×)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Historical Backtest Overlay (Trailing <= 30 Trading Days, Strictly OOS) */}
          {backtest && (
            <div className="panel-inset p-4 space-y-3.5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                    <button
                      type="button"
                      role="switch"
                      aria-checked={backtestVisible}
                      onClick={() => setBacktestVisible((v) => !v)}
                      className="inline-flex items-center gap-2 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 rounded-md py-0.5 whitespace-nowrap shrink-0"
                    >
                      <span
                        className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors duration-150 ${
                          backtestVisible
                            ? 'bg-gold-500/25 border-gold-500'
                            : 'bg-zinc-200 dark:bg-zinc-800 border-zinc-300 dark:border-zinc-700'
                        }`}
                      >
                        <span
                          className={`inline-block h-3.5 w-3.5 rounded-full transition-transform duration-150 ${
                            backtestVisible
                              ? 'bg-gold-500 translate-x-[18px]'
                              : 'bg-zinc-500 dark:bg-zinc-400 translate-x-[2px]'
                          }`}
                        />
                      </span>
                      <History className="w-3.5 h-3.5 text-gold-500 shrink-0" />
                      <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                        Historical Backtest ({backtest.summary.tradingDays}D)
                      </span>
                    </button>

                    <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
                      · Last {backtest.summary.tradingDays} trading days · strictly OOS · Signal-weighted tanh(r̂/σ)
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
                    Compounds daily exposure w<sub>t</sub> ∈ [-1, +1] sized by out-of-sample predicted return magnitude normalized on training-period volatility
                  </p>
                </div>

                {backtestVisible ? (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400 whitespace-nowrap">
                      Friction:
                    </span>
                    <div
                      role="group"
                      aria-label="Transaction cost friction in basis points"
                      className="inline-flex rounded-md border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-0.5 text-[10px] font-mono tabular-nums"
                    >
                      {BACKTEST_COST_OPTIONS.map((bps) => (
                        <button
                          key={bps}
                          type="button"
                          onClick={() => setBacktestCostBps(bps)}
                          className={`px-2 py-0.5 rounded-xs cursor-pointer transition-colors whitespace-nowrap shrink-0 ${
                            backtestCostBps === bps
                              ? 'bg-gold-500/20 text-gold-700 dark:text-gold-300 font-bold'
                              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200'
                          }`}
                        >
                          {bps} bps
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="font-mono text-xs tabular-nums text-zinc-600 dark:text-zinc-300 whitespace-nowrap shrink-0">
                    Net{' '}
                    <strong
                      className={
                        backtest.summary.strategyCumPct >= 0 ? 'text-[#3fb27f]' : 'text-[#e0655f]'
                      }
                    >
                      {formatSignedPct(backtest.summary.strategyCumPct)}
                    </strong>{' '}
                    <span className="text-zinc-400">vs B&amp;H</span>{' '}
                    <span>{formatSignedPct(backtest.summary.benchmarkCumPct)}</span>{' '}
                    <span className="text-zinc-400">· {backtestCostBps} bps</span>
                  </div>
                )}
              </div>

              {backtestVisible &&
                (backtest.days.length < 2 ? (
                  <div className="py-4 px-3 rounded-lg border border-dashed border-zinc-300 dark:border-zinc-700/80 text-xs text-zinc-500 dark:text-zinc-400">
                    At least 2 strictly out-of-sample holdout trading sessions are required to evaluate cumulative P&amp;L. Expand the date range or reduce Lookback L to accrue a full 30-day holdout window.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* 6-Metric Backtest Performance Strip */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 py-3 border-y border-zinc-200/70 dark:border-zinc-800/70 font-mono tabular-nums">
                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Strategy P&amp;L
                        </div>
                        <div
                          className={`mt-1 text-sm font-bold inline-flex items-center gap-1 ${
                            backtest.summary.strategyCumPct > 0
                              ? 'text-[#3fb27f]'
                              : backtest.summary.strategyCumPct < 0
                              ? 'text-[#e0655f]'
                              : 'text-zinc-700 dark:text-zinc-200'
                          }`}
                        >
                          {backtest.summary.strategyCumPct > 0 ? (
                            <TrendingUp className="w-3.5 h-3.5 shrink-0" />
                          ) : backtest.summary.strategyCumPct < 0 ? (
                            <TrendingDown className="w-3.5 h-3.5 shrink-0" />
                          ) : (
                            <Minus className="w-3.5 h-3.5 shrink-0" />
                          )}
                          <span>{formatSignedPct(backtest.summary.strategyCumPct)}</span>
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          Gross {formatSignedPct(backtest.summary.grossCumPct)}
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Excess vs B&amp;H
                        </div>
                        <div
                          className={`mt-1 text-sm font-semibold ${
                            backtest.summary.excessReturnPct >= 0
                              ? 'text-[#3fb27f]'
                              : 'text-[#e0655f]'
                          }`}
                        >
                          {formatSignedPct(backtest.summary.excessReturnPct)}
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          B&amp;H {formatSignedPct(backtest.summary.benchmarkCumPct)}
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Ann. Sharpe
                        </div>
                        <div
                          className={`mt-1 text-sm font-semibold ${
                            backtest.summary.annualizedSharpe >= 1
                              ? 'text-[#3fb27f]'
                              : backtest.summary.annualizedSharpe >= 0
                              ? 'text-gold-600 dark:text-gold-400'
                              : 'text-[#e0655f]'
                          }`}
                        >
                          {backtest.summary.annualizedSharpe.toFixed(2)}
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          PF{' '}
                          {backtest.summary.profitFactor === null
                            ? '∞'
                            : backtest.summary.profitFactor.toFixed(2)}
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Max Drawdown
                        </div>
                        <div className="mt-1 text-sm font-semibold text-[#e0655f]">
                          {backtest.summary.maxDrawdownPct.toFixed(2)}%
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          Peak-to-trough
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Win Rate
                        </div>
                        <div
                          className={`mt-1 text-sm font-semibold ${
                            backtest.summary.winRatePct >= 50
                              ? 'text-[#3fb27f]'
                              : 'text-zinc-900 dark:text-zinc-100'
                          }`}
                        >
                          {backtest.summary.winRatePct.toFixed(1)}%
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          {backtest.summary.winDays}/{backtest.summary.tradingDays} green days
                        </div>
                      </div>

                      <div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                          Avg Turnover
                        </div>
                        <div className="mt-1 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                          {(backtest.summary.avgDailyTurnover * 100).toFixed(0)}%/d
                        </div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
                          Drag -{Math.abs(backtest.summary.totalCostDragPct).toFixed(2)}% · |w̄|{' '}
                          {(backtest.summary.avgAbsWeight * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>

                    {/* Interactive Cumulative P&L vs Buy & Hold SVG Chart */}
                    <HistoricalBacktestChart backtest={backtest} />
                  </div>
                ))}
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
