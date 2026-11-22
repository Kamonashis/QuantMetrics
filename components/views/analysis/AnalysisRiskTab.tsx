'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  VaRBacktestModelResult,
  DrawdownResult,
  DownsideBenchmarkResult,
} from '@/lib/analysis-engine';
import {
  getCachedTcnRiskVarForecast,
  runTcnRiskVarForecast,
  type TcnRiskVarForecast,
} from '@/lib/tcn-multi-domain';
import {
  KupiecPofEquation,
  ChristoffersenEquation,
  DrawdownEquation,
  PsrEquation,
} from '@/components/math/MathEquations';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { RegimeChip } from '@/components/ui/RegimeChip';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { VaRExplorer } from '@/components/ui/VarExplorer';
import type { Data, Layout } from 'plotly.js';
import { regimeNames } from '@/components/charts/RegimeTimeline';
import type { HmmFit } from '@/lib/hmm';
import {
  getHmmRegimeToken,
  type HmmStateIdx,
} from '@/lib/hmm-regime-overlay';
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
  ShieldAlert,
  TrendingDown,
  Scale,
  Info,
  RotateCcw,
  Sparkles,
} from 'lucide-react';

interface VarOosMonitorPoint {
  date: string;
  loss: number;
  var95: number;
  var99: number;
  hmmState: HmmStateIdx;
  hmmProb: number;
  hit95: boolean;
  hit99: boolean;
}

function TcnVarOosMonitorChart({ tcnRisk }: { tcnRisk: TcnRiskVarForecast }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const points = useMemo<VarOosMonitorPoint[]>(() => {
    const out: VarOosMonitorPoint[] = [];
    const n = tcnRisk.oosVar95Series.length;
    for (let i = Math.max(0, tcnRisk.holdoutStartIdx); i < n; i++) {
      const v95 = tcnRisk.oosVar95Series[i];
      const v99 = tcnRisk.oosVar99Series[i];
      const loss = tcnRisk.oosDailyLossSeries?.[i];
      if (!Number.isFinite(v95) || !Number.isFinite(v99) || !Number.isFinite(loss)) continue;
      // Strictly causal prediction-origin regime at t - 1 (= i - 1)
      const originIdx = Math.max(0, i - 1);
      out.push({
        date: tcnRisk.dates?.[i] ?? `OOS-${out.length + 1}`,
        loss,
        var95: v95,
        var99: v99,
        hmmState: tcnRisk.hmmStates?.[originIdx] ?? 0,
        hmmProb: tcnRisk.hmmProbs?.[originIdx] ?? 1 / 3,
        hit95: loss > v95,
        hit99: loss > v99,
      });
    }
    return out;
  }, [tcnRisk]);

  const { traces, layout } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
  }>(() => {
    if (points.length < 3) return { traces: [], layout: {} };
    const xs = points.map((p) => p.date);
    const lossesPct = points.map((p) => p.loss * 100);
    const var95Pct = points.map((p) => p.var95 * 100);
    const var99Pct = points.map((p) => p.var99 * 100);

    const pointColors = points.map((p) =>
      showRegimeOverlay
        ? getHmmRegimeToken(p.hmmState).color
        : p.hit99
          ? QUANT_COLORS.negative
          : p.hit95
            ? QUANT_COLORS.gold
            : '#71717a'
    );

    const barColors = pointColors.map((hex) => {
      if (hex.startsWith('#') && hex.length === 7) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r},${g},${b},0.32)`;
      }
      return hex;
    });

    const data: Data[] = [
      {
        type: 'bar',
        name: 'Realized Loss (−r_t)',
        x: xs,
        y: lossesPct,
        marker: { color: barColors },
        hovertemplate: 'Loss: %{y:+.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'markers',
        name: 'OOS Daily Loss',
        x: xs,
        y: lossesPct,
        marker: {
          size: points.map((p) => (p.hit99 ? 8 : p.hit95 ? 6.5 : 4.5)),
          color: pointColors,
          line: {
            color: points.map((p) =>
              p.hit99 ? QUANT_COLORS.negative : isDark ? '#08080A' : '#FFFFFF'
            ),
            width: points.map((p) => (p.hit99 ? 1.8 : 0.8)),
          },
        },
        text: points.map((p) => {
          const tok = getHmmRegimeToken(p.hmmState);
          const tag = p.hit99 ? 'VaR 99% Breach' : p.hit95 ? 'VaR 95% Breach' : 'Within Envelope';
          return `${tok.label} (${Math.round(p.hmmProb * 100)}%) · ${tag}`;
        }),
        hovertemplate: '%{text}<extra></extra>',
        showlegend: false,
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'TCN VaR 95%',
        x: xs,
        y: var95Pct,
        line: { color: QUANT_COLORS.gold, width: 1.6, dash: 'dash' },
        hovertemplate: 'VaR 95%: %{y:.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'TCN VaR 99%',
        x: xs,
        y: var99Pct,
        line: { color: QUANT_COLORS.negative, width: 1.8 },
        hovertemplate: 'VaR 99%: %{y:.2f}%<extra></extra>',
      },
    ];

    const shapes: Layout['shapes'] = [
      ...buildHmmRegimeShapes(
        xs,
        points.map((p) => p.hmmState),
        showRegimeOverlay
      ),
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: 0,
        y1: 0,
        line: { color: tokens.zeroLine, width: 1 },
      },
    ];

    return {
      traces: data,
      layout: {
        margin: { l: 48, r: 16, t: 10, b: 36 },
        shapes,
        yaxis: {
          ticksuffix: '%',
          tickfont: { size: 10, color: tokens.textMuted },
        },
      },
    };
  }, [points, showRegimeOverlay, isDark, tokens]);

  if (points.length < 3) return null;

  const n = points.length;
  const activeIdx = hoverIdx !== null && hoverIdx < n ? hoverIdx : n - 1;
  const activePt = points[activeIdx];
  const activeTok = getHmmRegimeToken(activePt.hmmState);

  return (
    <div className="mt-4 pt-4 border-t border-[var(--hairline)] space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            Strictly OOS 1d Realized Loss vs. TCN VaR 95% / 99% Thresholds
          </h4>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
            Positive values indicate daily downside loss (−r_t); points above curves denote VaR exceedances.
          </p>
        </div>
        <HmmRegimeOverlayToggle
          enabled={showRegimeOverlay}
          onToggle={() => setShowRegimeOverlay((v) => !v)}
          caption="origin t−1 regime"
        />
      </div>

      {/* Live Hover Readout Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <strong className="text-zinc-900 dark:text-zinc-100">{activePt.date}</strong>
          <span>·</span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: activeTok.color }}
            />
            <strong className={activeTok.textClass}>
              {activeTok.label} ({Math.round(activePt.hmmProb * 100)}%)
            </strong>
          </span>
          <span>·</span>
          <span>
            Realized Loss{' '}
            <strong className={activePt.loss > 0 ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}>
              {activePt.loss >= 0 ? '+' : ''}
              {(activePt.loss * 100).toFixed(2)}%
            </strong>
          </span>
          <span>·</span>
          <span>VaR95 {(activePt.var95 * 100).toFixed(2)}%</span>
          <span>·</span>
          <span>VaR99 {(activePt.var99 * 100).toFixed(2)}%</span>
        </div>

        <span
          className={`font-semibold ${
            activePt.hit99
              ? 'text-rose-600 dark:text-rose-400'
              : activePt.hit95
                ? 'text-gold-700 dark:text-gold-300'
                : 'text-emerald-700 dark:text-emerald-400'
          }`}
        >
          {activePt.hit99 ? 'VaR 99% Breach' : activePt.hit95 ? 'VaR 95% Breach' : 'Within VaR Envelope'}
        </span>
      </div>

      <div className="w-full rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)]/40 p-1.5">
        <PlotlyChart
          data={traces}
          layout={layout}
          plotType="2d"
          rangeSlider
          hovermode="x unified"
          showLegend={false}
          height={240}
          showWheelHint
          exportFilename={`${tcnRisk.ticker}-tcn-var-oos`}
          uirevisionKey={`tcn-var-oos-${tcnRisk.ticker}-${showRegimeOverlay}`}
          onHover={(ev) => {
            const pt = (ev as { points?: Array<{ x?: unknown }> })?.points?.[0];
            if (pt?.x !== undefined) {
              const idx = points.findIndex((p) => p.date === String(pt.x));
              if (idx >= 0) setHoverIdx(idx);
            }
          }}
          onUnhover={() => setHoverIdx(null)}
        />
      </div>

      {/* Legend Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dashed border-[#DEC084] inline-block" />
            TCN VaR 95%
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#E0655F] inline-block" />
            TCN VaR 99%
          </span>
          <span>Larger ringed dots mark VaR exceedances</span>
        </div>
        <span>Hover sessions to inspect tail breaches &amp; origin regime</span>
      </div>

      {/* Per-Regime VaR Exceedance Scorecard */}
      {tcnRisk.regimePerformance && tcnRisk.regimePerformance.length > 0 && (
        <HmmRegimeScorecard
          title="TCN 1d OOS VaR Coverage by Causal HMM Regime"
          subtitle="Stratified by dominant HMM state at t−1 prediction origin"
          columns={[
            { key: 'exc95', label: '95% Breaches' },
            { key: 'exc99', label: '99% Breaches' },
            { key: 'var99', label: 'Mean VaR 99%' },
            { key: 'loss', label: 'Mean Daily Loss' },
          ]}
          rows={tcnRisk.regimePerformance.map((r) => ({
            state: r.state,
            count: r.count,
            cells: {
              exc95:
                r.exceedance95RatePct !== null
                  ? `${r.exceedance95Count} (${r.exceedance95RatePct.toFixed(1)}%)`
                  : '—',
              exc99:
                r.exceedance99RatePct !== null ? (
                  <span
                    className={
                      r.exceedance99RatePct > 2.5
                        ? 'text-rose-600 dark:text-rose-400 font-semibold'
                        : 'text-zinc-800 dark:text-zinc-200'
                    }
                  >
                    {r.exceedance99Count} ({r.exceedance99RatePct.toFixed(1)}%)
                  </span>
                ) : (
                  '—'
                ),
              var99: r.meanVar99 !== null ? `-${(r.meanVar99 * 100).toFixed(2)}%` : '—',
              loss:
                r.meanRealizedLoss !== null
                  ? `${r.meanRealizedLoss >= 0 ? '-' : '+'}${(Math.abs(r.meanRealizedLoss) * 100).toFixed(2)}%`
                  : '—',
            },
          }))}
        />
      )}
    </div>
  );
}

interface AnalysisRiskTabProps {
  varBacktestModels: VaRBacktestModelResult[];
  drawdownResult: DrawdownResult;
  downsideResult: DownsideBenchmarkResult;
  ticker: string;
  hmmFit?: HmmFit | null;
  returns?: number[];
  dates?: string[];
}

export function AnalysisRiskTab({
  varBacktestModels,
  drawdownResult,
  downsideResult,
  ticker,
  hmmFit,
  returns = [],
  dates = [],
}: AnalysisRiskTabProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [ddHoverIdx, setDdHoverIdx] = useState<number | null>(null);
  const [betaHoverIdx, setBetaHoverIdx] = useState<number | null>(null);

  const retLen = returns.length;
  const firstDate = dates[0] ?? '';
  const lastDate = dates[dates.length - 1] ?? '';
  const lastRet = returns[retLen - 1] ?? 0;

  const [tcnRisk, setTcnRisk] = useState<TcnRiskVarForecast | null>(() =>
    retLen >= 70 ? getCachedTcnRiskVarForecast(ticker, returns, dates) : null
  );
  const [tcnTraining, setTcnTraining] = useState(false);
  const [retrainNonce, setRetrainNonce] = useState(0);
  const retRef = useRef(returns);
  const datesRef = useRef(dates);
  useEffect(() => {
    retRef.current = returns;
    datesRef.current = dates;
  }, [returns, dates]);

  useEffect(() => {
    if (retLen < 70) return;
    let cancelled = false;
    const force = retrainNonce > 0;
    const timer = setTimeout(() => {
      if (cancelled) return;
      const curRets = retRef.current;
      const curDates = datesRef.current;
      if (!force) {
        const cached = getCachedTcnRiskVarForecast(ticker, curRets, curDates);
        if (cached) {
          setTcnRisk(cached);
          setTcnTraining(false);
          return;
        }
      }
      setTcnTraining(true);
      void runTcnRiskVarForecast(ticker, curRets, curDates, {
        forceRetrain: force,
        yieldControl: () => new Promise<void>((r) => setTimeout(r, 0)),
        isCancelled: () => cancelled,
      }).then((res) => {
        if (!cancelled) {
          if (res) setTcnRisk(res);
          setTcnTraining(false);
        }
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ticker, retLen, firstDate, lastDate, lastRet, retrainNonce]);

  const activeTcnRisk = tcnRisk && tcnRisk.ticker === ticker.toUpperCase() ? tcnRisk : null;
  const allVarModels = useMemo(
    () => (activeTcnRisk ? [...varBacktestModels, activeTcnRisk.backtestRow] : varBacktestModels),
    [varBacktestModels, activeTcnRisk]
  );

  const ddData = drawdownResult.underwaterCurve;
  const nDd = ddData.length;
  const activeDdIdx = ddHoverIdx !== null && ddHoverIdx < nDd ? ddHoverIdx : nDd - 1;
  const activeDdPoint = ddData[activeDdIdx] || { date: '', drawdownPct: 0 };

  const { ddTraces, ddLayout } = useMemo<{
    ddTraces: Data[];
    ddLayout: Partial<Layout>;
  }>(() => {
    if (ddData.length === 0) return { ddTraces: [], ddLayout: {} };
    const xs = ddData.map((d) => d.date);
    const ys = ddData.map((d) => -Math.abs(d.drawdownPct));
    return {
      ddTraces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Drawdown',
          x: xs,
          y: ys,
          fill: 'tozeroy',
          fillcolor: 'rgba(244,63,94,0.22)',
          line: { color: '#f43f5e', width: 1.6 },
          hovertemplate: 'Drawdown: %{y:.2f}%<extra></extra>',
        },
      ],
      ddLayout: {
        margin: { l: 52, r: 16, t: 10, b: 36 },
        yaxis: {
          title: { text: 'DD %', font: { size: 10, color: tokens.textMuted } },
          ticksuffix: '%',
          range: [Math.min(-5, Math.min(...ys) * 1.1), 0.5],
          zeroline: true,
          zerolinecolor: tokens.zeroLine,
        },
      },
    };
  }, [ddData, tokens]);

  const betaData = downsideResult.rollingBetaSeries;
  const nBeta = betaData.length;
  const activeBetaIdx = betaHoverIdx !== null && betaHoverIdx < nBeta ? betaHoverIdx : nBeta - 1;
  const activeBetaPoint = betaData[activeBetaIdx] || { date: '', beta: 1 };

  const { betaTraces, betaLayout } = useMemo<{
    betaTraces: Data[];
    betaLayout: Partial<Layout>;
  }>(() => {
    if (betaData.length === 0) return { betaTraces: [], betaLayout: {} };
    return {
      betaTraces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: '60d Rolling β',
          x: betaData.map((b) => b.date),
          y: betaData.map((b) => b.beta),
          line: { color: '#10b981', width: 1.8 },
          hovertemplate: 'β vs SPY: %{y:.2f}<extra></extra>',
        },
      ],
      betaLayout: {
        margin: { l: 44, r: 14, t: 10, b: 36 },
        shapes: [
          {
            type: 'line',
            xref: 'paper',
            x0: 0,
            x1: 1,
            yref: 'y',
            y0: 1,
            y1: 1,
            line: { color: tokens.zeroLine, width: 1, dash: 'dash' },
          },
        ],
        yaxis: {
          title: { text: 'Beta (β)', font: { size: 10, color: tokens.textMuted } },
        },
      },
    };
  }, [betaData, tokens]);

  return (
    <div className="space-y-6">
      {/* Regime-Conditioned VaR Attribution — HMM posterior tags each violation day */}
      {hmmFit && (
        <div className="panel p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-1">
            <div>
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-gold-500" />
                <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                  Regime-Conditioned VaR Attribution (HMM)
                </h3>
                <RegimeChip fit={hmmFit} />
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Percentage of each model&apos;s 99% exceedances that landed in each HMM state. Stress-heavy clustering ⇒ regime mixture, not a parameter error — the exceedance clustering Christoffersen already flags.
              </p>
            </div>
          </div>

          <RegimeTransitionMatrix fit={hmmFit} labels={regimeNames(hmmFit.k)} className="mt-3 border-t border-zinc-100 dark:border-zinc-800/60 pt-3 lg:max-w-md" showStats={false} />

          <div className="mt-3 overflow-x-auto">
            <table className="table-sticky-first w-full text-xs text-left">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 font-mono">
                  <th className="py-2 px-2">Model</th>
                  <th className="py-2 px-2">99% Hits</th>
                  {regimeNames(hmmFit.k).map((name) => (
                    <th key={name} className="py-2 px-2 text-right">{name}</th>
                  ))}
                  <th className="py-2 px-2 text-right">Stress share</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60 font-mono">
                {allVarModels.map((m) => {
                  const hits = m.exceedanceIdx99.length;
                  const byState = new Array(hmmFit.k).fill(0);
                  for (const idx of m.exceedanceIdx99) {
                    const v = hmmFit.viterbi[idx];
                    if (v !== undefined && v >= 0) byState[v]++;
                  }
                  const stressShare = hits > 0 ? (byState[hmmFit.k - 1] / hits) * 100 : null;
                  return (
                    <tr key={m.key} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                      <td className="py-2 px-2 font-semibold text-zinc-800 dark:text-zinc-200">{m.modelName}</td>
                      <td className="py-2 px-2">{hits}</td>
                      {byState.map((c, si) => (
                        <td key={si} className="py-2 px-2 text-right tabular-nums">
                          {hits > 0 ? c : '—'}
                          {hits > 0 && <span className="text-zinc-400 pl-1">({((c / hits) * 100).toFixed(0)}%)</span>}
                        </td>
                      ))}
                      <td className="py-2 px-2 text-right">
                        <span className={stressShare !== null && stressShare > 50 ? 'text-rose-500 font-semibold' : stressShare !== null && stressShare > 33 ? 'text-gold-600 dark:text-gold-400' : 'text-zinc-500'}>
                          {stressShare !== null ? `${stressShare.toFixed(0)}%` : '—'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Multi-Horizon TCN-Attention Tail Risk Forecast (1d / 5d / 22d) */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                TCN-Attention Multi-Horizon Tail Risk Forecast (1d · 5d · 22d)
              </h3>
              {activeTcnRisk && (
                <span className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 tabular-nums">
                  {activeTcnRisk.cacheHit ? 'cached' : `${activeTcnRisk.trainingDurationMs}ms`} · Pinball Quantile Head
                  {activeTcnRisk.hmmRegime
                    ? ` · HMM ${activeTcnRisk.hmmRegime.dominantLabel} (${Math.round(activeTcnRisk.hmmRegime.probs[0] * 100)}/${Math.round(activeTcnRisk.hmmRegime.probs[1] * 100)}/${Math.round(activeTcnRisk.hmmRegime.probs[2] * 100)}%)`
                    : ''}
                </span>
              )}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Conditional downside VaR (95% &amp; 99%) and Expected Shortfall across 1-, 5-, and 22-session horizons conditioned on causal 3-state HMM volatility regimes with strictly out-of-sample backtest coverage.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setRetrainNonce((n) => n + 1)}
            disabled={tcnTraining}
            className="inline-flex cursor-pointer items-center gap-1.5 self-start sm:self-auto rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[10px] font-semibold text-zinc-700 hover:border-gold-500/50 dark:text-zinc-300 disabled:opacity-50"
          >
            <RotateCcw className={`h-3.5 w-3.5 ${tcnTraining ? 'animate-spin' : ''}`} />
            {tcnTraining ? 'Training…' : 'Retrain Quantile Head'}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
          {([1, 5, 22] as const).map((h) => {
            const hz = activeTcnRisk?.horizons[h];
            return (
              <div key={h} className="panel-inset p-3.5 border border-[var(--hairline)]">
                <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-zinc-400">
                  <span className="font-semibold text-gold-700 dark:text-gold-300">{h}-Session Horizon</span>
                  <span>OOS n={hz?.holdoutCount ?? 0}</span>
                </div>
                <div className="mt-2.5 grid grid-cols-2 gap-2 font-mono text-xs tabular-nums">
                  <div>
                    <div className="text-[10px] text-zinc-400">VaR 95%</div>
                    <div className="text-base font-bold text-zinc-900 dark:text-zinc-100">
                      {hz ? `-${(hz.var95 * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-zinc-400">ES 95%</div>
                    <div className="text-base font-bold text-zinc-700 dark:text-zinc-300">
                      {hz ? `-${(hz.es95 * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-zinc-400">VaR 99%</div>
                    <div className="text-base font-bold text-rose-600 dark:text-rose-400">
                      {hz ? `-${(hz.var99 * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-zinc-400">ES 99%</div>
                    <div className="text-base font-bold text-rose-700 dark:text-rose-300">
                      {hz ? `-${(hz.es99 * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </div>
                  </div>
                </div>
                <div className="mt-2.5 pt-2 border-t border-[var(--hairline)] flex items-center justify-between font-mono text-[10px] text-zinc-500 dark:text-zinc-400 tabular-nums">
                  <span>ES/VaR 99: {hz ? `${hz.tailRatio.toFixed(2)}×` : '—'}</span>
                  <span>OOS hits: {hz ? `${hz.oosExceedanceRate95.toFixed(1)}% / ${hz.oosExceedanceRate99.toFixed(1)}%` : '—'}</span>
                </div>
              </div>
            );
          })}
        </div>

        {activeTcnRisk && activeTcnRisk.topShockDates.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
            <span className="uppercase tracking-wider text-zinc-400">Top attended tail-risk shocks:</span>
            {activeTcnRisk.topShockDates.map((s) => (
              <span key={s.date} className="tabular-nums">
                <strong className="text-zinc-800 dark:text-zinc-200">{s.date}</strong> (w={(s.weight * 100).toFixed(1)}%, r={s.metricValue >= 0 ? '+' : ''}{s.metricValue.toFixed(1)}%)
              </span>
            ))}
          </div>
        )}

        {activeTcnRisk && <TcnVarOosMonitorChart tcnRisk={activeTcnRisk} />}
      </div>

      {/* 6-Model VaR & ES Backtesting Comparison Table */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-rose-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                6-Model Value-at-Risk (VaR) &amp; Expected Shortfall (ES) Backtesting Suite
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Kupiec (1995) unconditional coverage &amp; Christoffersen (1998) exceedance clustering tests with Basel traffic lights (incl. TCN-Attention Quantile OOS)
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <KupiecPofEquation className="hidden lg:inline" />
            <ChristoffersenEquation className="hidden lg:inline" />
          </div>
        </div>

        <VaRExplorer models={allVarModels} />

        <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span>
            Under Basel Committee standards, 99% VaR with fewer than 5 exceptions over 250 days falls into the Green zone
            (no regulatory capital multiplier penalty). TCN-Attention is evaluated strictly out-of-sample on its trailing holdout split.
          </span>
        </div>
      </div>

      {/* Underwater Drawdown Curve & Analytics */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-rose-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Underwater Drawdown Curve &amp; Peak-to-Trough Analytics
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Continuous percentage decline from running peak price
            </p>
          </div>

          <div className="font-mono text-xs text-zinc-400">
            {activeDdPoint.date}: <strong className="text-rose-500">{activeDdPoint.drawdownPct.toFixed(2)}%</strong>
          </div>
        </div>

        {/* Drawdown Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Current Drawdown</div>
            <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
              {drawdownResult.currentDrawdownPct.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Maximum Drawdown (MDD)</div>
            <div className="mt-1 text-lg font-bold font-mono text-rose-600 dark:text-rose-400">
              {drawdownResult.maxDrawdownPct.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Longest Drawdown</div>
            <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
              {drawdownResult.maxDrawdownDurationDays} days
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Calmar Ratio</div>
            <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
              {drawdownResult.calmarRatio.toFixed(2)}
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Ulcer Index</div>
            <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
              {drawdownResult.ulcerIndex.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Underwater Chart */}
        <div className="mt-4">
          <ChartFullscreen title="Underwater Drawdown Curve">
            <PlotlyChart
              data={ddTraces}
              layout={ddLayout}
              plotType="2d"
              rangeSlider
              hovermode="x unified"
              showLegend={false}
              height={250}
              showWheelHint
              exportFilename={`${ticker}-underwater-drawdown`}
              uirevisionKey={`dd-${ticker}-${nDd}`}
              onHover={(ev) => {
                const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                if (pt && typeof pt.pointIndex === 'number') setDdHoverIdx(pt.pointIndex);
              }}
              onUnhover={() => setDdHoverIdx(null)}
            />
          </ChartFullscreen>
        </div>
      </div>

      {/* Downside Risk & Asymmetric Beta to SPY Benchmark */}
      <div className="grid-cards">
        {/* Downside Risk Metrics */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Downside Risk &amp; Benchmark Asymmetry (vs. SPY)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Separating downside co-crash exposure from upside market participation
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              Beta: <strong className="text-gold-500">{activeBetaPoint.beta.toFixed(2)}</strong>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Semi-Deviation (Ann.)</div>
              <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {(downsideResult.semiDeviationAnn * 100).toFixed(2)}%
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Sortino Ratio</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.sortinoRatio.toFixed(2)}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Downside Beta (β⁻)</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.downsideBeta.toFixed(2)}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">SPY Down-days</div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Upside Beta (β⁺)</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.upsideBeta.toFixed(2)}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">SPY Up-days</div>
            </div>
          </div>

          {/* Rolling Beta Chart */}
          <div className="mt-4">
            <ChartFullscreen title="60-Day Rolling Beta to SPY">
              <PlotlyChart
                data={betaTraces}
                layout={betaLayout}
                plotType="2d"
                rangeSlider
                hovermode="x unified"
                showLegend={false}
                height={210}
                showWheelHint
                exportFilename={`${ticker}-rolling-beta-spy`}
                uirevisionKey={`beta-${ticker}-${nBeta}`}
                onHover={(ev) => {
                  const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                  if (pt && typeof pt.pointIndex === 'number') setBetaHoverIdx(pt.pointIndex);
                }}
                onUnhover={() => setBetaHoverIdx(null)}
              />
            </ChartFullscreen>
          </div>

        </div>

        {/* Probabilistic & Deflated Sharpe Ratio (Bailey & López de Prado) */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Probabilistic Sharpe Ratio (PSR) &amp; Deflated Sharpe (DSR)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Bailey &amp; López de Prado (2012): correcting point Sharpe for non-normality and selection bias
              </p>
            </div>
            <PsrEquation className="hidden md:inline" />
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Point Sharpe Ratio (Ann.)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {downsideResult.sharpeRatioAnn.toFixed(2)}
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                HAC SE: ±{downsideResult.sharpeStdErrorHac.toFixed(3)}
              </div>
            </div>

            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Probabilistic Sharpe (PSR)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-gold-600 dark:text-gold-400">
                {(downsideResult.probabilisticSharpeRatio * 100).toFixed(1)}%
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                Confidence that SR &gt; 0
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-zinc-500 dark:text-zinc-400">Deflated Sharpe Ratio (DSR)</div>
                <div className="text-lg font-bold font-mono text-gold-600 dark:text-gold-400 mt-1">
                  {(downsideResult.deflatedSharpeRatio * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-right text-[11px] text-zinc-400 max-w-xs">
                Penalizes for multiple testing / backtest overfitting across hyperparameter search trials.
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>
              Standard i.i.d. Sharpe standard errors assume zero skewness and kurtosis = 3. In the presence of negative
              skew and fat tails, PSR proves that the observed risk-adjusted return is statistically genuine rather than
              sampling luck.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
