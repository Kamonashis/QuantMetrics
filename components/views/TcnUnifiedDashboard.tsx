'use client';

import React, { useMemo, useState } from 'react';
import { motion } from 'motion/react';
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  Database,
  Layers,
  MinusCircle,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import type { Opportunity } from '@/lib/opportunity-engine';
import {
  useTcnDiagnosticsBundle,
  type TrainableTcnHead,
} from '@/hooks/useTcnDiagnosticsBundle';
import {
  aggregateCausalAttentionTimeline,
  buildCrossModuleRegimeBreakdown,
  buildHorizonComparisonMatrix,
  buildUnifiedOosScorecard,
  summarizeCacheAndLatency,
  type OosScorecardRow,
} from '@/lib/tcn-diagnostics';
import { computeHistoricalBacktest } from '@/lib/tcn-attention';
import type { Data, Layout } from 'plotly.js';
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

type DashboardPanel = 'scorecard' | 'matrix' | 'attention' | 'cache';
type AttentionFilter = 'all' | 'volatility' | 'risk_var' | 'regression';
type EdgeNavView =
  | 'home'
  | 'analysis'
  | 'modeling'
  | 'volatility'
  | 'fno'
  | 'correlation'
  | 'regression';

interface TcnUnifiedDashboardProps {
  ticker: string;
  bars: PriceBar[];
  allStockBars: Record<string, PriceBar[]>;
  iv30: number | null;
  rv20: number | null;
  topOpportunity: Opportunity | null;
  onNavigate: (view: EdgeNavView, params?: Record<string, string>) => void;
}

const PANEL_TABS: { id: DashboardPanel; label: string }[] = [
  { id: 'scorecard', label: 'OOS Scorecard' },
  { id: 'matrix', label: '1d · 5d · 22d Matrix' },
  { id: 'attention', label: 'Attention Timeline' },
  { id: 'cache', label: 'Cache & Latency' },
];

const ATTN_FILTERS: { id: AttentionFilter; label: string }[] = [
  { id: 'all', label: 'All Heads' },
  { id: 'volatility', label: 'Volatility' },
  { id: 'risk_var', label: 'Risk VaR' },
  { id: 'regression', label: 'Regression' },
];



function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtSignedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;
}

function VerdictLabel({
  text,
  tone,
}: {
  text: string;
  tone: OosScorecardRow['verdictTone'];
}) {
  if (tone === 'positive') {
    return (
      <span className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold text-emerald-700 dark:text-emerald-400">
        <CheckCircle2 className="h-3 w-3 shrink-0" />
        <span>{text}</span>
      </span>
    );
  }
  if (tone === 'negative') {
    return (
      <span className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold text-rose-600 dark:text-rose-400">
        <AlertTriangle className="h-3 w-3 shrink-0" />
        <span>{text}</span>
      </span>
    );
  }
  if (tone === 'gold') {
    return (
      <span className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold text-gold-700 dark:text-gold-300">
        <Activity className="h-3 w-3 shrink-0" />
        <span>{text}</span>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] font-medium text-zinc-500 dark:text-zinc-400">
      <MinusCircle className="h-3 w-3 shrink-0" />
      <span>{text}</span>
    </span>
  );
}

export function TcnUnifiedDashboard({
  ticker,
  bars,
  allStockBars,
  iv30,
  rv20,
  topOpportunity,
  onNavigate,
}: TcnUnifiedDashboardProps) {
  const [activePanel, setActivePanel] = useState<DashboardPanel>('scorecard');
  const [attentionFilter, setAttentionFilter] = useState<AttentionFilter>('all');
  const [hoveredLag, setHoveredLag] = useState<number | null>(null);
  const [showAttentionRegimeOverlay, setShowAttentionRegimeOverlay] = useState(true);

  const primaryOrientation: 'short-premium' | 'long-gamma' | 'neutral' =
    topOpportunity?.family === 'premium-harvest'
      ? 'short-premium'
      : topOpportunity?.family === 'long-gamma'
        ? 'long-gamma'
        : 'neutral';

  const bundle = useTcnDiagnosticsBundle({
    ticker,
    bars,
    allStockBars,
    iv30,
    rv20,
    primaryOrientation,
  });

  const scorecardRows = useMemo(
    () =>
      buildUnifiedOosScorecard({
        volatility: bundle.volatility,
        riskVar: bundle.riskVar,
        regression: bundle.regression,
        confluence: bundle.confluence,
        statuses: bundle.statuses,
      }),
    [bundle.volatility, bundle.riskVar, bundle.regression, bundle.confluence, bundle.statuses]
  );

  const matrixRows = useMemo(
    () =>
      buildHorizonComparisonMatrix({
        volatility: bundle.volatility,
        riskVar: bundle.riskVar,
        regression: bundle.regression,
        iv30,
      }),
    [bundle.volatility, bundle.riskVar, bundle.regression, iv30]
  );

  const attentionData = useMemo(
    () =>
      aggregateCausalAttentionTimeline({
        volatility: bundle.volatility,
        riskVar: bundle.riskVar,
        regression: bundle.regression,
      }),
    [bundle.volatility, bundle.riskVar, bundle.regression]
  );

  const telemetry = useMemo(
    () =>
      summarizeCacheAndLatency({
        volatility: bundle.volatility,
        riskVar: bundle.riskVar,
        regression: bundle.regression,
        decompForecast: bundle.decompForecast,
        statuses: bundle.statuses,
        multiDomainCounts: bundle.multiDomainCounts,
        decompStats: bundle.decompStats,
      }),
    [
      bundle.volatility,
      bundle.riskVar,
      bundle.regression,
      bundle.decompForecast,
      bundle.statuses,
      bundle.multiDomainCounts,
      bundle.decompStats,
    ]
  );

  const decompBacktest = useMemo(() => {
    const sigs = bundle.decompForecast?.backtestSignals;
    if (!sigs || sigs.length < 2) return null;
    return computeHistoricalBacktest(sigs, 0);
  }, [bundle.decompForecast]);

  const crossModuleRegimeRows = useMemo(
    () =>
      buildCrossModuleRegimeBreakdown({
        volatility: bundle.volatility,
        riskVar: bundle.riskVar,
        regression: bundle.regression,
        decompBacktest,
      }),
    [bundle.volatility, bundle.riskVar, bundle.regression, decompBacktest]
  );

  const hasCrossModuleRegimes = crossModuleRegimeRows.some(
    (r) => r.volCount > 0 || r.riskCount > 0 || r.regCount > 0 || r.decompCount > 0
  );

  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const hasAttentionRegimes = useMemo(
    () => attentionData.lagCurves.some((pt) => pt.hmmState !== null),
    [attentionData.lagCurves]
  );

  const filteredShockDates = useMemo(() => {
    if (attentionFilter === 'all') return attentionData.jointShockDates;
    return attentionData.jointShockDates.filter((d) =>
      d.modules.includes(attentionFilter)
    );
  }, [attentionData.jointShockDates, attentionFilter]);

  const readyModulesCount =
    (bundle.volatility ? 1 : 0) +
    (bundle.riskVar ? 1 : 0) +
    (bundle.regression ? 1 : 0) +
    (bundle.confluence.hasVolArtifact ? 1 : 0);

  const { attnTraces, attnLayout } = useMemo<{
    attnTraces: Data[];
    attnLayout: Partial<Layout>;
  }>(() => {
    const pts = attentionData.lagCurves;
    if (pts.length === 0) return { attnTraces: [], attnLayout: {} };

    const labels = pts.map((pt) => `t−${pt.lag}`);
    const consensusPct = pts.map((pt) => pt.consensusWeight * 100);
    const barColors = pts.map((pt) => {
      const tok = pt.hmmState !== null ? getHmmRegimeToken(pt.hmmState) : null;
      const base = showAttentionRegimeOverlay && tok ? tok.color : QUANT_COLORS.gold;
      const alpha = hoveredLag === pt.lag ? 0.55 : 0.24;
      if (base.startsWith('#') && base.length === 7) {
        const r = parseInt(base.slice(1, 3), 16);
        const g = parseInt(base.slice(3, 5), 16);
        const b = parseInt(base.slice(5, 7), 16);
        return `rgba(${r},${g},${b},${alpha})`;
      }
      return base;
    });

    const traces: Data[] = [
      {
        type: 'bar',
        name: 'Consensus',
        x: labels,
        y: consensusPct,
        marker: { color: barColors },
        hovertemplate: 'Consensus: %{y:.1f}%<extra></extra>',
      },
    ];

    if (showAttentionRegimeOverlay && hasAttentionRegimes) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'HMM Origin Regime',
        x: labels,
        y: consensusPct,
        marker: {
          size: pts.map((pt) => (hoveredLag === pt.lag ? 8 : 5)),
          color: pts.map((pt) =>
            pt.hmmState !== null ? getHmmRegimeToken(pt.hmmState).color : QUANT_COLORS.gold
          ),
          line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 },
        },
        text: pts.map((pt) =>
          pt.hmmState !== null
            ? `${getHmmRegimeToken(pt.hmmState).label}${pt.hmmProb !== null ? ` (${Math.round(pt.hmmProb * 100)}%)` : ''}`
            : 'N/A'
        ),
        hovertemplate: 'Regime: %{text}<extra></extra>',
        showlegend: false,
      });
    }

    if (attentionFilter === 'all' || attentionFilter === 'volatility') {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Volatility',
        x: labels,
        y: pts.map((pt) => (pt.volWeight !== null ? pt.volWeight * 100 : null)),
        line: { color: QUANT_COLORS.gold, width: 2 },
        connectgaps: true,
        hovertemplate: 'Volatility: %{y:.1f}%<extra></extra>',
      });
    }

    if (attentionFilter === 'all' || attentionFilter === 'risk_var') {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Risk VaR',
        x: labels,
        y: pts.map((pt) => (pt.riskWeight !== null ? pt.riskWeight * 100 : null)),
        line: { color: '#E11D48', width: 1.8, dash: 'dot' },
        connectgaps: true,
        hovertemplate: 'Risk VaR: %{y:.1f}%<extra></extra>',
      });
    }

    if (attentionFilter === 'all' || attentionFilter === 'regression') {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Regression',
        x: labels,
        y: pts.map((pt) => (pt.regWeight !== null ? pt.regWeight * 100 : null)),
        line: { color: '#10B981', width: 1.8 },
        connectgaps: true,
        hovertemplate: 'Regression: %{y:.1f}%<extra></extra>',
      });
    }

    const shapes = buildHmmRegimeShapes(
      labels,
      pts.map((pt) => pt.hmmState),
      showAttentionRegimeOverlay && hasAttentionRegimes
    );

    return {
      attnTraces: traces,
      attnLayout: {
        margin: { l: 44, r: 14, t: 10, b: 32 },
        barmode: 'overlay',
        shapes,
        xaxis: {
          type: 'category',
          tickfont: { size: 9.5, color: tokens.textMuted },
        },
        yaxis: {
          ticksuffix: '%',
          rangemode: 'tozero',
          tickfont: { size: 9.5, color: tokens.textMuted },
        },
      },
    };
  }, [
    attentionData.lagCurves,
    attentionFilter,
    hoveredLag,
    showAttentionRegimeOverlay,
    hasAttentionRegimes,
    isDark,
    tokens,
  ]);

  const activeLagPoint =
    hoveredLag !== null
      ? attentionData.lagCurves.find((p) => p.lag === hoveredLag) ?? null
      : null;

  const alignmentTone: OosScorecardRow['verdictTone'] =
    bundle.confluence.alignment === 'ALIGNED'
      ? 'positive'
      : bundle.confluence.alignment === 'DIVERGENT'
        ? 'negative'
        : bundle.confluence.alignment === 'MIXED'
          ? 'gold'
          : 'neutral';

  const baselTone: OosScorecardRow['verdictTone'] =
    bundle.riskVar?.backtestRow.baselZone === 'Green'
      ? 'positive'
      : bundle.riskVar?.backtestRow.baselZone === 'Red'
        ? 'negative'
        : bundle.riskVar
          ? 'gold'
          : 'neutral';

  return (
    <section
      className="panel p-5"
      aria-label="Unified TCN-Attention Diagnostics Dashboard"
    >
      {/* Header Row */}
      <div className="flex flex-col gap-3 border-b border-[var(--hairline)] pb-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            <span className="kicker inline-flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5" /> TCN-Attention Unified Diagnostics
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-[10px] tabular-nums">
              {ticker.toUpperCase()}
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-[10px] tabular-nums">
              {readyModulesCount}/4 modules ready
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-[10px] tabular-nums">
              {telemetry.cacheHitsCount} cache hits
            </span>
            {(bundle.volatility?.hmmRegime ?? bundle.riskVar?.hmmRegime ?? bundle.regression?.hmmRegime) && (() => {
              const hmm =
                bundle.volatility?.hmmRegime ??
                bundle.riskVar?.hmmRegime ??
                bundle.regression?.hmmRegime!;
              return (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono text-[10px] tabular-nums text-gold-700 dark:text-gold-300">
                    HMM Feature: {hmm.dominantLabel} ({Math.round(hmm.probs[0] * 100)}/{Math.round(hmm.probs[1] * 100)}/{Math.round(hmm.probs[2] * 100)}%)
                  </span>
                </>
              );
            })()}
          </div>
          <h2 className="mt-1 font-display text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Comparative Multi-Horizon Model Review
          </h2>
          <p className="mt-0.5 max-w-3xl text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">
            Aggregates strictly out-of-sample Volatility (QLIKE), Risk VaR/ES (Pinball), Multi-Asset Regression (Huber), and Signal Confluence across 1d, 5d, and 22d horizons conditioned on causal 3-state HMM volatility regime probabilities.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {bundle.isTrainingAny && bundle.activeStepLabel && (
            <span className="font-mono text-[10px] tabular-nums text-gold-700 dark:text-gold-300">
              Fitting {Math.min(bundle.completedSteps + 1, Math.max(1, bundle.totalSteps))}/{Math.max(1, bundle.totalSteps)}: {bundle.activeStepLabel}
            </span>
          )}
          <button
            type="button"
            onClick={bundle.retrainAll}
            disabled={bundle.isTrainingAny || bars.length < 70}
            className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[10px] font-semibold text-zinc-800 transition-colors hover:border-gold-500/50 hover:text-gold-700 disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap shrink-0 dark:text-zinc-100 dark:hover:text-gold-300"
          >
            <RefreshCw className={`h-3 w-3 ${bundle.isTrainingAny ? 'animate-spin' : ''}`} />
            {bundle.isTrainingAny ? 'Training…' : 'Retrain All'}
          </button>
          <button
            type="button"
            onClick={bundle.clearAllCaches}
            disabled={bundle.isTrainingAny}
            className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-transparent px-3 font-mono text-[10px] font-medium text-zinc-600 transition-colors hover:border-rose-500/40 hover:text-rose-600 disabled:cursor-not-allowed disabled:opacity-50 whitespace-nowrap shrink-0 dark:text-zinc-400 dark:hover:text-rose-400"
          >
            <Trash2 className="h-3 w-3" />
            Clear Cache
          </button>
        </div>
      </div>

      {/* Executive Summary Strip — single-elevation hairline-divided columns */}
      <div className="grid grid-cols-2 divide-y divide-[var(--hairline)] border-b border-[var(--hairline)] sm:grid-cols-3 lg:grid-cols-5 lg:divide-x lg:divide-y-0">
        <div className="py-3 pr-3">
          <div className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            Signal Confluence
          </div>
          <div className="mt-1">
            <VerdictLabel text={bundle.confluence.alignment} tone={alignmentTone} />
          </div>
          <div className="mt-1 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            {bundle.confluence.volRegimeRead === 'UNAVAILABLE'
              ? 'Needs live IV30 chain'
              : bundle.confluence.volRegimeRead}
          </div>
        </div>

        <div className="py-3 lg:px-4">
          <div className="font-mono text-[9.5px] text-gold-700 dark:text-gold-300">
            IV30 − TCN RV22 (Direct)
          </div>
          <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
            {fmtSignedPct(bundle.confluence.forwardVrp22, 1)}
          </div>
          <div className="mt-0.5 font-mono text-[9.5px] tabular-nums text-zinc-500 dark:text-zinc-400">
            RV {fmtPct(bundle.confluence.tcnRv1d, 0)} / {fmtPct(bundle.confluence.tcnRv5d, 0)} / {fmtPct(bundle.confluence.tcnRv22d, 0)}
          </div>
        </div>

        <div className="py-3 pr-3 lg:px-4">
          <div className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            1d Tail Risk (VaR99 / ES99)
          </div>
          <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
            {bundle.riskVar
              ? `${fmtPct(bundle.riskVar.horizons[1].var99, 2)} / ${fmtPct(bundle.riskVar.horizons[1].es99, 2)}`
              : bundle.statuses.risk_var === 'training'
                ? 'Training…'
                : '—'}
          </div>
          <div className="mt-0.5">
            {bundle.riskVar ? (
              <VerdictLabel
                text={`Basel ${bundle.riskVar.backtestRow.baselZone}`}
                tone={baselTone}
              />
            ) : (
              <span className="font-mono text-[9.5px] text-zinc-500">
                Pinball quantile OOS
              </span>
            )}
          </div>
        </div>

        <div className="py-3 lg:px-4">
          <div className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            5d Multi-Asset Forecast
          </div>
          <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
            {bundle.regression
              ? fmtSignedPct(bundle.regression.horizons[5].predictedReturn, 2)
              : bundle.statuses.regression === 'training'
                ? 'Training…'
                : '—'}
          </div>
          <div className="mt-0.5 font-mono text-[9.5px] tabular-nums text-zinc-500 dark:text-zinc-400">
            {bundle.regression
              ? `Lift ${bundle.regression.horizons[5].nonlinearLiftR2 >= 0 ? '+' : ''}${bundle.regression.horizons[5].nonlinearLiftR2.toFixed(3)} · ${bundle.regression.independentTickers.join(', ')}`
              : 'Auto-paired benchmarks'}
          </div>
        </div>

        <div className="col-span-2 py-3 sm:col-span-1 lg:pl-4">
          <div className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            Decomposition t+1 Tilt
          </div>
          <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
            {bundle.confluence.priceTiltPct !== null
              ? `${bundle.confluence.priceTiltPct >= 0 ? '+' : ''}${bundle.confluence.priceTiltPct.toFixed(2)}%`
              : '—'}
          </div>
          <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
            {bundle.confluence.hasDecompArtifact ? (
              <span>
                {bundle.confluence.cyclicalSnrRegime ?? 'Cached'}
                {bundle.confluence.backtestSharpe30d !== null
                  ? ` · 30d Sh ${bundle.confluence.backtestSharpe30d.toFixed(2)}`
                  : ''}
              </span>
            ) : (
              <button
                type="button"
                onClick={() => onNavigate('analysis')}
                className="inline-flex cursor-pointer items-center gap-1 font-semibold text-gold-700 hover:underline dark:text-gold-300"
              >
                Run Decomposition <ArrowUpRight className="h-2.5 w-2.5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Segmented Comparative View Tabs */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div
          role="tablist"
          aria-label="TCN-Attention comparative diagnostic panels"
          className="flex flex-wrap items-center gap-1 rounded-lg bg-[var(--bg-inset)] p-1"
        >
          {PANEL_TABS.map((tab) => {
            const active = activePanel === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                type="button"
                aria-selected={active}
                onClick={() => setActivePanel(tab.id)}
                className={`relative cursor-pointer rounded-md px-3 py-1.5 font-mono text-[10px] font-semibold transition-colors whitespace-nowrap shrink-0 ${
                  active
                    ? 'text-zinc-950'
                    : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="tcn-unified-tab-pill"
                    transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
                    className="absolute inset-0 rounded-md bg-gold-500"
                  />
                )}
                <span className="relative z-10">{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
          Basis: RV √252 vs IV √365 direct · Causal 3-state HMM regime inputs · Zero holdout leakage
        </div>
      </div>

      {/* Panel Content */}
      <motion.div
        key={activePanel}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        className="mt-4"
      >
        {/* ── VIEW 1: CROSS-MODULE OOS SCORECARD ─────────────────────────────── */}
        {activePanel === 'scorecard' && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[780px] border-collapse text-left font-mono text-[11px]">
              <thead>
                <tr className="border-b border-[var(--hairline)] text-[9.5px] text-zinc-500 dark:text-zinc-400">
                  <th className="py-2 pr-3 font-semibold">Module &amp; Loss Head</th>
                  <th className="py-2 pr-3 font-semibold">Primary OOS Metric</th>
                  <th className="py-2 pr-3 font-semibold">Secondary Validation</th>
                  <th className="py-2 pr-3 font-semibold">Diagnostic Verdict</th>
                  <th className="py-2 pr-3 text-right font-semibold">OOS n</th>
                  <th className="py-2 text-right font-semibold">Drill Down</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--hairline)]">
                {scorecardRows.map((row) => {
                  const isTrainingRow = row.status === 'training';
                  return (
                    <tr
                      key={row.module}
                      className="h-11 transition-colors hover:bg-[var(--bg-inset)]/60"
                    >
                      <td className="py-2 pr-3">
                        <div className="font-semibold text-zinc-900 dark:text-zinc-100">
                          {row.label}
                        </div>
                        <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                          {row.headSpec}
                        </div>
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        {isTrainingRow ? (
                          <div className="h-4 w-24 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                        ) : (
                          <>
                            <div className="font-semibold text-zinc-900 dark:text-zinc-100">
                              {row.primaryMetricValue}
                            </div>
                            <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                              {row.primaryMetricLabel}
                            </div>
                          </>
                        )}
                      </td>
                      <td className="py-2 pr-3 tabular-nums">
                        {isTrainingRow ? (
                          <div className="h-4 w-28 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
                        ) : (
                          <>
                            <div className="text-zinc-800 dark:text-zinc-200">
                              {row.secondaryMetricValue}
                            </div>
                            <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                              {row.secondaryMetricLabel}
                            </div>
                          </>
                        )}
                      </td>
                      <td className="py-2 pr-3">
                        <VerdictLabel text={row.verdict} tone={row.verdictTone} />
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                        {row.oosSampleCount !== null ? row.oosSampleCount : '—'}
                      </td>
                      <td className="py-2 text-right">
                        <button
                          type="button"
                          onClick={() => onNavigate(row.targetView, row.targetParams)}
                          className="inline-flex cursor-pointer items-center gap-1 font-mono text-[10px] font-semibold text-gold-700 hover:underline whitespace-nowrap dark:text-gold-300"
                        >
                          {row.targetLabel} <ArrowUpRight className="h-3 w-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {bundle.confluence.rationale.length > 0 && (
              <div className="mt-3 border-t border-[var(--hairline)] pt-2.5 text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-400">
                <span className="font-mono text-[10px] font-semibold text-zinc-800 dark:text-zinc-200">
                  Cross-Module Read:{' '}
                </span>
                {bundle.confluence.rationale.join(' ')}
                {topOpportunity
                  ? ` Evaluated against top-ranked Edge setup (${topOpportunity.title}).`
                  : ''}
              </div>
            )}

            {hasCrossModuleRegimes && (
              <HmmRegimeScorecard
                className="mt-4"
                title="Cross-Module OOS Accuracy by Causal HMM Volatility Regime"
                subtitle="1d OOS holdout metrics stratified by dominant HMM state at prediction origin (t−1)"
                columns={[
                  { key: 'vol', label: 'Vol 1d QLIKE' },
                  { key: 'risk', label: 'Risk 95/99% Hits' },
                  { key: 'reg', label: 'Reg 1d Hit / RMSE' },
                  { key: 'decomp', label: 'Decomp 30d Hit / P&L' },
                ]}
                rows={crossModuleRegimeRows.map((r) => {
                  const primaryN = Math.max(r.volCount, r.riskCount, r.regCount, r.decompCount);
                  return {
                    state: r.state,
                    count: primaryN,
                    cells: {
                      vol:
                        r.volCount > 0 && r.volMeanQlike !== null
                          ? `${r.volMeanQlike.toFixed(3)} (n=${r.volCount})`
                          : '—',
                      risk:
                        r.riskCount > 0 &&
                        r.riskExc95RatePct !== null &&
                        r.riskExc99RatePct !== null
                          ? `${r.riskExc95RatePct.toFixed(1)}% / ${r.riskExc99RatePct.toFixed(1)}%`
                          : '—',
                      reg:
                        r.regCount > 0 && r.regHitRatePct !== null && r.regOosRmse !== null
                          ? `${r.regHitRatePct.toFixed(0)}% · ${(r.regOosRmse * 100).toFixed(2)}%`
                          : '—',
                      decomp:
                        r.decompCount > 0 &&
                        r.decompHitRatePct !== null &&
                        r.decompCumNetPct !== null ? (
                          <span
                            className={
                              r.decompCumNetPct >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-600 dark:text-rose-400'
                            }
                          >
                            {r.decompHitRatePct.toFixed(0)}% · {r.decompCumNetPct >= 0 ? '+' : ''}
                            {r.decompCumNetPct.toFixed(2)}%
                          </span>
                        ) : (
                          '—'
                        ),
                    },
                  };
                })}
              />
            )}
          </div>
        )}

        {/* ── VIEW 2: SIDE-BY-SIDE 1D · 5D · 22D MATRIX ──────────────────────── */}
        {activePanel === 'matrix' && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] border-collapse text-left font-mono text-[11px]">
              <thead>
                <tr className="border-b border-[var(--hairline)] text-[9.5px] text-zinc-500 dark:text-zinc-400">
                  <th className="py-2 pr-3 font-semibold">Horizon</th>
                  <th className="py-2 pr-3 text-right font-semibold">TCN RV (99% Band)</th>
                  <th className="py-2 pr-3 text-right font-semibold">VaR 95% / 99%</th>
                  <th className="py-2 pr-3 text-right font-semibold">ES 99% (ES/VaR)</th>
                  <th className="py-2 pr-3 text-right font-semibold">Multi-Asset Reg (99% PI)</th>
                  <th className="py-2 pr-3 text-right font-semibold">Reg Hit Rate</th>
                  <th className="py-2 text-right font-semibold">Curve / VRP Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--hairline)]">
                {matrixRows.map((row) => (
                  <tr
                    key={row.horizon}
                    className="h-12 transition-colors hover:bg-[var(--bg-inset)]/60"
                  >
                    <td className="py-2 pr-3 font-semibold text-zinc-900 dark:text-zinc-100">
                      {row.label}
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      <div className="font-semibold text-gold-700 dark:text-gold-300">
                        {fmtPct(row.volAnnualized, 1)}
                      </div>
                      <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                        {row.volLower99 !== null && row.volUpper99 !== null
                          ? `${fmtPct(row.volLower99, 0)} – ${fmtPct(row.volUpper99, 0)}`
                          : '—'}
                      </div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      <div className="text-zinc-900 dark:text-zinc-100">
                        {fmtPct(row.var95, 2)} / {fmtPct(row.var99, 2)}
                      </div>
                      <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                        loss magnitude
                      </div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      <div className="font-semibold text-rose-600 dark:text-rose-400">
                        {fmtPct(row.es99, 2)}
                      </div>
                      <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                        {row.tailRatio !== null ? `${row.tailRatio.toFixed(2)}× VaR99` : '—'}
                      </div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      <div
                        className={`font-semibold ${
                          row.regReturn === null
                            ? 'text-zinc-500'
                            : row.regReturn >= 0
                              ? 'text-emerald-700 dark:text-emerald-400'
                              : 'text-rose-600 dark:text-rose-400'
                        }`}
                      >
                        {fmtSignedPct(row.regReturn, 2)}
                      </div>
                      <div className="text-[9.5px] text-zinc-500 dark:text-zinc-400">
                        {row.regLower99 !== null && row.regUpper99 !== null
                          ? `${fmtSignedPct(row.regLower99, 1)} to ${fmtSignedPct(row.regUpper99, 1)}`
                          : '—'}
                      </div>
                    </td>
                    <td className="py-2 pr-3 text-right tabular-nums text-zinc-800 dark:text-zinc-200">
                      {row.regHitRatePct !== null ? `${row.regHitRatePct.toFixed(0)}%` : '—'}
                    </td>
                    <td className="py-2 text-right tabular-nums text-[10px] text-zinc-600 dark:text-zinc-300">
                      {row.vrpNote}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {bundle.regression && bundle.regression.factorImportance.length > 0 && (
              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-2.5 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
                <span className="font-semibold text-zinc-700 dark:text-zinc-200">
                  Regression Sequence Factor Importance:
                </span>
                {bundle.regression.factorImportance.map((f) => (
                  <span
                    key={f.ticker}
                    className={`tabular-nums ${f.isHmmRegime ? 'text-gold-700 dark:text-gold-300' : ''}`}
                  >
                    {f.ticker}
                    {f.isDependentLag ? ' (self-lag)' : f.isHmmRegime ? ' (3-state)' : ''}{' '}
                    <strong className={f.isHmmRegime ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-900 dark:text-zinc-100'}>
                      {f.importancePct.toFixed(0)}%
                    </strong>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── VIEW 3: UNIFIED CAUSAL ATTENTION TIMELINE ──────────────────────── */}
        {activePanel === 'attention' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex flex-wrap items-center gap-1 rounded-lg bg-[var(--bg-inset)] p-1">
                  {ATTN_FILTERS.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setAttentionFilter(f.id)}
                      className={`cursor-pointer rounded-md px-2.5 py-1 font-mono text-[10px] font-semibold transition-colors whitespace-nowrap shrink-0 ${
                        attentionFilter === f.id
                          ? 'bg-gold-500 text-zinc-950'
                          : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
                {hasAttentionRegimes && (
                  <HmmRegimeOverlayToggle
                    enabled={showAttentionRegimeOverlay}
                    onToggle={() => setShowAttentionRegimeOverlay((v) => !v)}
                  />
                )}
              </div>

              <div className="flex flex-wrap items-center gap-4 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-3 bg-[#DEC084]" /> Volatility
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-3 bg-[#E11D48]" /> Risk VaR
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-3 bg-[#10B981]" /> Regression
                </span>
              </div>
            </div>

            {attentionData.lagCurves.length === 0 ? (
              <div className="flex h-36 items-center justify-center border border-dashed border-[var(--hairline)] font-mono text-[11px] text-zinc-500">
                {bundle.isTrainingAny
                  ? 'Training causal self-attention heads…'
                  : 'Requires ≥ 70 daily bars to extract causal self-attention lag profiles.'}
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
                {/* Overlaid SVG Attention Profile */}
                <div className="xl:col-span-7">
                  <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
                    <span>
                      Normalized Causal Lag Attention (t−{attentionData.maxLookback} → t−1)
                    </span>
                    <span className="inline-flex flex-wrap items-center gap-1.5 tabular-nums text-zinc-800 dark:text-zinc-200">
                      {activeLagPoint ? (
                        <>
                          <span>
                            Lag t−{activeLagPoint.lag}: Vol {fmtPct(activeLagPoint.volWeight, 1)} · Risk {fmtPct(activeLagPoint.riskWeight, 1)} · Reg {fmtPct(activeLagPoint.regWeight, 1)}
                          </span>
                          {activeLagPoint.hmmState !== null && (() => {
                            const tok = getHmmRegimeToken(activeLagPoint.hmmState);
                            return (
                              <>
                                <span className="text-zinc-400">·</span>
                                <span className="inline-flex items-center gap-1">
                                  <span
                                    className="w-2 h-2 rounded-full shrink-0"
                                    style={{ backgroundColor: tok.color }}
                                  />
                                  <strong className={tok.textClass}>
                                    {tok.label}
                                    {activeLagPoint.hmmProb !== null
                                      ? ` (${Math.round(activeLagPoint.hmmProb * 100)}%)`
                                      : ''}
                                  </strong>
                                </span>
                              </>
                            );
                          })()}
                        </>
                      ) : (
                        'Hover lag columns to inspect weights & HMM regime'
                      )}
                    </span>
                  </div>
                  <div className="w-full border border-[var(--hairline)] bg-[var(--bg-inset)]/40 p-1.5">
                    <PlotlyChart
                      data={attnTraces}
                      layout={attnLayout}
                      plotType="categorical"
                      hovermode="x unified"
                      showLegend={false}
                      height={208}
                      showWheelHint
                      exportFilename={`${ticker}-tcn-attention-profile`}
                      uirevisionKey={`tcn-attn-${ticker}-${attentionFilter}-${showAttentionRegimeOverlay}`}
                      onHover={(ev) => {
                        const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                        if (pt && typeof pt.pointIndex === 'number') {
                          const item = attentionData.lagCurves[pt.pointIndex];
                          if (item) setHoveredLag(item.lag);
                        }
                      }}
                      onUnhover={() => setHoveredLag(null)}
                    />
                  </div>
                </div>

                {/* Joint Shock-Date Consensus Ledger */}
                <div className="xl:col-span-5">
                  <div className="mb-1.5 flex items-center justify-between font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
                    <span className="inline-flex items-center gap-1">
                      <Layers className="h-3 w-3" /> Joint Shock-Date Ledger
                    </span>
                    <span className="tabular-nums">
                      {filteredShockDates.length} dates ranked
                    </span>
                  </div>
                  <div className="overflow-x-auto border border-[var(--hairline)]">
                    <table className="w-full border-collapse text-left font-mono text-[10px]">
                      <thead>
                        <tr className="border-b border-[var(--hairline)] bg-[var(--bg-inset)]/60 text-[9px] text-zinc-500">
                          <th className="px-2.5 py-1.5 font-semibold">Date (Lag)</th>
                          <th className="px-2 py-1.5 font-semibold">Consensus</th>
                          <th className="px-2 py-1.5 text-right font-semibold">Σ Weight</th>
                          <th className="px-2.5 py-1.5 font-semibold">Shock Metric</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[var(--hairline)]">
                        {filteredShockDates.length === 0 ? (
                          <tr>
                            <td colSpan={4} className="px-2.5 py-4 text-center text-zinc-500">
                              No shock dates for selected filter.
                            </td>
                          </tr>
                        ) : (
                          filteredShockDates.map((d) => (
                            <tr
                              key={d.date}
                              onMouseEnter={() => setHoveredLag(d.lag + 1)}
                              onMouseLeave={() => setHoveredLag(null)}
                              className="transition-colors hover:bg-[var(--bg-inset)]/70"
                            >
                              <td className="px-2.5 py-1.5 tabular-nums font-semibold text-zinc-900 dark:text-zinc-100 whitespace-nowrap">
                                {d.date}{' '}
                                <span className="font-normal text-zinc-500">
                                  (t−{d.lag + 1})
                                </span>
                              </td>
                              <td className="px-2 py-1.5 tabular-nums whitespace-nowrap">
                                <span
                                  className={
                                    d.moduleCount >= 2
                                      ? 'font-semibold text-gold-700 dark:text-gold-300'
                                      : 'text-zinc-500 dark:text-zinc-400'
                                  }
                                >
                                  {d.moduleCount}/{Math.max(1, attentionData.activeModules.length)} heads
                                </span>
                              </td>
                              <td className="px-2 py-1.5 text-right tabular-nums text-zinc-800 dark:text-zinc-200">
                                {(d.combinedWeight * 100).toFixed(1)}%
                              </td>
                              <td className="px-2.5 py-1.5 text-[9.5px] text-zinc-500 dark:text-zinc-400 truncate max-w-[180px]">
                                {d.metricSummary}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── VIEW 4: CACHE TELEMETRY & LATENCY CONTROLS ─────────────────────── */}
        {activePanel === 'cache' && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
              <div className="flex flex-wrap items-center gap-2">
                <Database className="h-3.5 w-3.5 text-gold-600 dark:text-gold-400" />
                <span>
                  LRU + sessionStorage pool: <strong className="tabular-nums text-zinc-900 dark:text-zinc-100">{telemetry.multiDomainTotal}</strong> multi-domain entries
                </span>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">
                  Vol {telemetry.multiDomainCounts.volatility} / Risk {telemetry.multiDomainCounts.risk_var} / Reg {telemetry.multiDomainCounts.regression} / Corr {telemetry.multiDomainCounts.correlation}
                </span>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">
                  Decomp {telemetry.decompCachedCount} ({telemetry.decompWeightTensors} tensors)
                </span>
              </div>
              <div className="tabular-nums">
                Active params: <strong className="text-zinc-900 dark:text-zinc-100">{telemetry.totalParameters.toLocaleString()}</strong> · Total fit: <strong className="text-zinc-900 dark:text-zinc-100">{telemetry.totalLatencyMs}ms</strong>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[740px] border-collapse text-left font-mono text-[11px]">
                <thead>
                  <tr className="border-b border-[var(--hairline)] text-[9.5px] text-zinc-500 dark:text-zinc-400">
                    <th className="py-2 pr-3 font-semibold">Head / Model</th>
                    <th className="py-2 pr-3 font-semibold">Cache State</th>
                    <th className="py-2 pr-3 text-right font-semibold">Fit Latency</th>
                    <th className="py-2 pr-3 text-right font-semibold">Parameters</th>
                    <th className="py-2 pr-3 text-right font-semibold">Receptive Field</th>
                    <th className="py-2 pr-3 text-right font-semibold">As-Of Date</th>
                    <th className="py-2 text-right font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--hairline)]">
                  {telemetry.rows.map((r) => (
                    <tr
                      key={r.module}
                      className="h-10 transition-colors hover:bg-[var(--bg-inset)]/60"
                    >
                      <td className="py-2 pr-3 font-semibold text-zinc-900 dark:text-zinc-100">
                        {r.label}
                      </td>
                      <td className="py-2 pr-3">
                        <VerdictLabel
                          text={r.cacheState}
                          tone={
                            r.cacheState === 'Cache Hit'
                              ? 'positive'
                              : r.cacheState === 'Fresh Fit'
                                ? 'gold'
                                : r.cacheState === 'Insufficient'
                                  ? 'negative'
                                  : 'neutral'
                          }
                        />
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-zinc-800 dark:text-zinc-200">
                        {r.latencyMs !== null ? `${r.latencyMs}ms` : '—'}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-zinc-700 dark:text-zinc-300">
                        {r.parameterCount !== null ? r.parameterCount.toLocaleString() : '—'}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-zinc-700 dark:text-zinc-300">
                        {r.receptiveField !== null ? `${r.receptiveField} td` : '—'}
                      </td>
                      <td className="py-2 pr-3 text-right tabular-nums text-zinc-500 dark:text-zinc-400">
                        {r.asOfDate ?? '—'}
                      </td>
                      <td className="py-2 text-right">
                        {r.canRetrain ? (
                          <button
                            type="button"
                            onClick={() => bundle.retrainSingle(r.module as TrainableTcnHead)}
                            disabled={bundle.isTrainingAny || bars.length < 70}
                            className="inline-flex cursor-pointer items-center gap-1 font-mono text-[10px] font-semibold text-gold-700 hover:underline disabled:cursor-not-allowed disabled:opacity-45 dark:text-gold-300"
                          >
                            <RefreshCw className="h-2.5 w-2.5" /> Retrain
                          </button>
                        ) : (
                          <button
                            type="button"
                            onClick={() => onNavigate('analysis')}
                            className="inline-flex cursor-pointer items-center gap-1 font-mono text-[10px] font-semibold text-zinc-600 hover:text-gold-700 hover:underline dark:text-zinc-400 dark:hover:text-gold-300"
                          >
                            Open Decomp <ArrowUpRight className="h-2.5 w-2.5" />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </motion.div>
    </section>
  );
}
