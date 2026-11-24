'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { RotateCcw, Sparkles } from 'lucide-react';
import {
  getCachedTcnRegressionForecast,
  runTcnMultiAssetRegression,
  type TcnMultiAssetRegressionResult,
} from '@/lib/tcn-multi-domain';
import type { FactorSample } from '@/lib/factor-regression';
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
import { Callout } from './lab-ui';

function TcnRegressionOosChart({ tcnReg }: { tcnReg: TcnMultiAssetRegressionResult }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const hist = tcnReg.oosHistory;
  const series = useMemo(() => {
    if (!hist || hist.length === 0) return [];
    let cumAct = 0;
    let cumPred = 0;
    let cumOls = 0;
    return hist.map((pt) => {
      cumAct += pt.actualReturn * 100;
      cumPred += pt.predictedReturn * 100;
      cumOls += pt.olsPredictedReturn * 100;
      const hit =
        (pt.predictedReturn >= 0 && pt.actualReturn >= 0) ||
        (pt.predictedReturn < 0 && pt.actualReturn < 0);
      return {
        ...pt,
        cumAct,
        cumPred,
        cumOls,
        hit,
      };
    });
  }, [hist]);

  const { traces, layout } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
  }>(() => {
    if (series.length < 3) return { traces: [], layout: {} };

    const xs = series.map((s) => s.date);
    const regimeColors = series.map((s) => getHmmRegimeToken(s.hmmState).color);
    const stride = Math.max(1, Math.ceil(series.length / 44));

    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Lag-Matched OLS',
        x: xs,
        y: series.map((s) => s.cumOls),
        line: { color: '#64748b', width: 1.2, dash: 'dot' },
        hovertemplate: 'OLS Cum: %{y:+.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Realized Cum Return',
        x: xs,
        y: series.map((s) => s.cumAct),
        line: { color: '#94a3b8', width: 1.6, dash: 'dash' },
        hovertemplate: 'Realized Cum: %{y:+.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'TCN Cum Pred',
        x: xs,
        y: series.map((s) => s.cumPred),
        line: { color: QUANT_COLORS.gold, width: 2 },
        hovertemplate: 'TCN Cum Pred: %{y:+.2f}%<extra></extra>',
      },
    ];

    if (showRegimeOverlay) {
      const mX: string[] = [];
      const mY: number[] = [];
      const mColor: string[] = [];
      const mText: string[] = [];
      for (let i = 0; i < series.length; i++) {
        if (i % stride !== 0 && i !== series.length - 1) continue;
        const s = series[i];
        const tok = getHmmRegimeToken(s.hmmState);
        mX.push(s.date);
        mY.push(s.cumPred);
        mColor.push(regimeColors[i]);
        mText.push(`${tok.label} (${Math.round(s.hmmProb * 100)}%)`);
      }
      data.push({
        type: 'scatter',
        mode: 'markers',
        name: 'HMM Origin Regime',
        x: mX,
        y: mY,
        marker: {
          size: 6,
          color: mColor,
          line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 },
        },
        text: mText,
        hovertemplate: 'Regime: %{text}<extra></extra>',
        showlegend: false,
      });
    }

    const shapes: Layout['shapes'] = [
      ...buildHmmRegimeShapes(
        xs,
        series.map((s) => s.hmmState),
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
        margin: { l: 50, r: 16, t: 10, b: 36 },
        shapes,
        yaxis: {
          ticksuffix: '%',
          tickfont: { size: 10, color: tokens.textMuted },
        },
      },
    };
  }, [series, showRegimeOverlay, isDark, tokens]);

  if (series.length < 3) return null;

  const n = series.length;
  const activeIdx = hoverIdx !== null && hoverIdx < n ? hoverIdx : n - 1;
  const activePt = series[activeIdx];
  const activeTok = getHmmRegimeToken(activePt.hmmState);

  return (
    <div className="mt-4 pt-4 border-t border-[var(--hairline)] space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            Strictly OOS 1d Cumulative Forecast vs. Realized Return ({tcnReg.dependentTicker})
          </h4>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
            Cumulative 1d predicted path (TCN-Attention vs. lag-matched OLS) color-coded by causal HMM regime at origin t−1.
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
            TCN r̂{' '}
            <strong className={activePt.predictedReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
              {activePt.predictedReturn >= 0 ? '+' : ''}
              {(activePt.predictedReturn * 100).toFixed(2)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            Actual r{' '}
            <strong className={activePt.actualReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
              {activePt.actualReturn >= 0 ? '+' : ''}
              {(activePt.actualReturn * 100).toFixed(2)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            OLS {activePt.olsPredictedReturn >= 0 ? '+' : ''}
            {(activePt.olsPredictedReturn * 100).toFixed(2)}%
          </span>
        </div>

        <span
          className={`font-semibold ${
            activePt.hit
              ? 'text-emerald-700 dark:text-emerald-400'
              : 'text-rose-600 dark:text-rose-400'
          }`}
        >
          {activePt.hit ? 'Direction Hit' : 'Direction Miss'}
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
          exportFilename={`${tcnReg.dependentTicker}-tcn-regression-oos`}
          uirevisionKey={`tcn-reg-oos-${tcnReg.dependentTicker}-${showRegimeOverlay}`}
          onHover={(ev) => {
            const pt = (ev as { points?: Array<{ x?: unknown }> })?.points?.[0];
            if (pt?.x !== undefined) {
              const idx = series.findIndex((s) => s.date === String(pt.x));
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
            <span className="w-3 h-0.5 bg-[#DEC084] inline-block" />
            TCN Cum Pred
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dashed border-[#94a3b8] inline-block" />
            Realized Cum Return
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dotted border-[#64748b] inline-block" />
            Lag-Matched OLS
          </span>
        </div>
        <span>Hover sessions to compare TCN vs. OLS across HMM regimes</span>
      </div>

      {/* Per-Regime Regression Accuracy Scorecard */}
      {tcnReg.regimePerformance && tcnReg.regimePerformance.length > 0 && (
        <HmmRegimeScorecard
          title={`TCN 1d OOS Regression Accuracy by Causal HMM Regime (${tcnReg.dependentTicker})`}
          subtitle="Stratified by dominant HMM state at t−1 prediction origin"
          columns={[
            { key: 'hit', label: 'Dir Hit Rate' },
            { key: 'tcnRmse', label: 'TCN OOS RMSE' },
            { key: 'olsRmse', label: 'OLS OOS RMSE' },
            { key: 'means', label: 'Mean Pred / Act' },
          ]}
          rows={tcnReg.regimePerformance.map((r) => ({
            state: r.state,
            count: r.count,
            cells: {
              hit:
                r.hitRatePct !== null ? (
                  <span
                    className={
                      r.hitRatePct >= 52
                        ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                        : 'text-zinc-800 dark:text-zinc-200'
                    }
                  >
                    {r.hitRatePct.toFixed(0)}%
                  </span>
                ) : (
                  '—'
                ),
              tcnRmse: r.oosRmse !== null ? `${(r.oosRmse * 100).toFixed(2)}%` : '—',
              olsRmse: r.olsOosRmse !== null ? `${(r.olsOosRmse * 100).toFixed(2)}%` : '—',
              means:
                r.meanPredReturn !== null && r.meanActualReturn !== null
                  ? `${r.meanPredReturn >= 0 ? '+' : ''}${(r.meanPredReturn * 100).toFixed(2)}% / ${r.meanActualReturn >= 0 ? '+' : ''}${(r.meanActualReturn * 100).toFixed(2)}%`
                  : '—',
            },
          }))}
        />
      )}
    </div>
  );
}

/**
 * Experimental: a nonlinear sequence model trained on lagged factor returns and
 * scored on a trailing chronological holdout. Kept apart from the descriptive
 * regression so in-sample fit is never presented as a forecast.
 */
export function PredictTab({ sample }: { sample: FactorSample }) {
  const dependentTicker = sample.asset;
  // TCN inputs: the asset plus up to six factor columns, all on the date-aligned sample.
  const { returnsByTicker, independentTickers } = useMemo(() => {
    const map: Record<string, number[]> = { [sample.asset]: sample.y };
    const keys: string[] = [];
    for (const c of sample.columns.slice(0, 6)) {
      const key = c.short;
      if (map[key]) continue;
      map[key] = c.values;
      keys.push(key);
    }
    return { returnsByTicker: map, independentTickers: keys };
  }, [sample]);
  const dates = sample.dates;

  const indepKey = independentTickers.join(',');
  const yLen = sample.y.length;
  const lastDate = dates[dates.length - 1] ?? '';
  const firstDate = dates[0] ?? '';

  const [tcnReg, setTcnReg] = useState<TcnMultiAssetRegressionResult | null>(() =>
    yLen >= 70
      ? getCachedTcnRegressionForecast(dependentTicker, independentTickers, returnsByTicker, dates)
      : null
  );
  const [tcnTraining, setTcnTraining] = useState(false);
  const [trainRequest, setTrainRequest] = useState(0);
  const returnsRef = useRef(returnsByTicker);
  const datesRef = useRef(dates);
  const indepRef = useRef(independentTickers);
  const trainingRef = useRef(false);
  useEffect(() => {
    returnsRef.current = returnsByTicker;
    datesRef.current = dates;
    indepRef.current = independentTickers;
  }, [returnsByTicker, dates, independentTickers]);

  // Cache hydration follows data changes — it never starts a fit.
  useEffect(() => {
    if (yLen < 70) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled || trainingRef.current) return;
      const cached = getCachedTcnRegressionForecast(
        dependentTicker,
        indepRef.current,
        returnsRef.current,
        datesRef.current
      );
      setTcnReg((prev) => {
        if (cached) return cached;
        return prev &&
          prev.dependentTicker === dependentTicker.toUpperCase() &&
          prev.independentTickers.join(',') === indepRef.current.map((t) => t.toUpperCase()).join(',')
          ? prev
          : null;
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [dependentTicker, indepKey, yLen, firstDate, lastDate]);

  // Training is manual-only.
  useEffect(() => {
    if (trainRequest === 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled) return;
      trainingRef.current = true;
      setTcnTraining(true);
      void runTcnMultiAssetRegression(dependentTicker, indepRef.current, returnsRef.current, datesRef.current, {
        forceRetrain: true,
        yieldControl: () => new Promise<void>((r) => setTimeout(r, 0)),
        isCancelled: () => cancelled,
      }).then((res) => {
        trainingRef.current = false;
        if (!cancelled) {
          if (res) setTcnReg(res);
          setTcnTraining(false);
        }
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      trainingRef.current = false;
    };
  }, [trainRequest, dependentTicker, indepKey]);

  const activeTcnReg =
    tcnReg &&
    tcnReg.dependentTicker === dependentTicker.toUpperCase() &&
    tcnReg.independentTickers.join(',') === independentTickers.map((t) => t.toUpperCase()).join(',')
      ? tcnReg
      : null;

  return (
    <div className="space-y-6">
      <Callout tone="warn" title="Experimental research. Not a trading signal.">
        The Explain and Stability tabs describe how {dependentTicker} moved with each factor on the same days. Same-day factor
        returns are not known in advance, so those fits cannot predict anything. This tab trains a nonlinear sequence model
        on <em>lagged</em> returns of {dependentTicker} and {independentTickers.join(', ')}, then scores it on a trailing holdout
        it never saw, against a lag-matched OLS baseline. Treat it as a forecast only if it beats that baseline out of sample.
      </Callout>
      {/* Multi-Asset Nonlinear Sequence Panel (TCN-Attention 1d / 5d / 22d) */}
        <div className="panel p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
            <div>
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-gold-500" />
                <span className="kicker">TCN-Attention Multi-Asset Sequence Forecasting</span>
                {activeTcnReg && (
                  <span className="font-mono text-[10px] text-zinc-400 tabular-nums">
                    · {activeTcnReg.cacheHit ? 'cached' : `${activeTcnReg.trainingDurationMs}ms`} · RF {activeTcnReg.receptiveField}d · {activeTcnReg.parameterCount} params
                    {activeTcnReg.hmmRegime
                      ? ` · HMM ${activeTcnReg.hmmRegime.dominantLabel} (${Math.round(activeTcnReg.hmmRegime.probs[0] * 100)}/${Math.round(activeTcnReg.hmmRegime.probs[1] * 100)}/${Math.round(activeTcnReg.hmmRegime.probs[2] * 100)}%)`
                      : ''}
                  </span>
                )}
              </div>
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm mt-0.5">
                Out-of-sample forecast test: {dependentTicker} ← lagged [{independentTickers.join(', ')}] (1d · 5d · 22d)
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setTrainRequest((prev) => prev + 1)}
              disabled={tcnTraining || yLen < 70}
              className="inline-flex cursor-pointer items-center gap-1.5 self-start sm:self-auto rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[10px] font-semibold text-zinc-700 hover:border-gold-500/50 dark:text-zinc-300 disabled:opacity-50"
            >
              <RotateCcw className={`h-3.5 w-3.5 ${tcnTraining ? 'animate-spin' : ''}`} />
              {tcnTraining ? 'Training…' : activeTcnReg ? 'Retrain Sequence Head' : 'Train Sequence Head'}
            </button>
          </div>

          {!activeTcnReg && !tcnTraining && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              {yLen < 70
                ? 'Needs at least 70 aligned returns to train.'
                : 'Not trained for this setup yet. Training runs in your browser and takes a few seconds; press Train Sequence Head to start.'}
            </p>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {([1, 5, 22] as const).map((h) => {
              const hz = activeTcnReg?.horizons[h];
              const lift = hz ? hz.nonlinearLiftR2 : null;
              // OOS R² ≤ 0 means the model did no better than a flat forecast on unseen data.
              const beatsFlat = hz ? hz.oosR2 > 0 : false;
              return (
                <div key={h} className="panel-inset p-3.5 border border-[var(--hairline)]">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-zinc-400">
                    <span className="font-semibold text-gold-700 dark:text-gold-300">{h}D Cumulative Return</span>
                    <span>OOS n={hz?.holdoutCount ?? 0}</span>
                  </div>
                  <div className="mt-1.5 flex items-baseline justify-between font-mono tabular-nums">
                    <span className={`text-xl font-bold ${!hz || !beatsFlat ? 'text-zinc-400 dark:text-zinc-500' : hz.predictedReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                      {hz ? `${hz.predictedReturn >= 0 ? '+' : ''}${(hz.predictedReturn * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </span>
                    <span className="text-[10px] text-zinc-500 dark:text-zinc-400">
                      {hz ? `99% [${(hz.lower99 * 100).toFixed(1)}%, ${(hz.upper99 * 100).toFixed(1)}%]` : '—'}
                    </span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-[var(--hairline)] grid grid-cols-3 gap-2 font-mono text-[10px] tabular-nums">
                    <div>
                      <span className="text-zinc-400 block">OOS R²</span>
                      <span className={`font-semibold ${hz && !beatsFlat ? 'text-rose-500' : 'text-zinc-800 dark:text-zinc-100'}`}>{hz ? `${(hz.oosR2 * 100).toFixed(1)}%` : '—'}</span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block">Lift vs OLS</span>
                      <span className={`font-semibold ${lift !== null && lift >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500'}`}>
                        {lift !== null ? `${lift >= 0 ? '+' : ''}${(lift * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block">Dir Hit</span>
                      <span className="font-semibold text-gold-700 dark:text-gold-300">{hz ? `${hz.directionalAccuracyPct.toFixed(0)}%` : '—'}</span>
                    </div>
                  </div>
                  {hz && (
                    <p className={`mt-2 text-[11px] ${beatsFlat ? 'text-[var(--positive)]' : 'text-zinc-500 dark:text-zinc-400'}`}>
                      {beatsFlat
                        ? 'Beat a flat forecast on unseen data.'
                        : 'Did not beat a flat (zero-change) forecast on unseen data. Not usable as a forecast.'}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {activeTcnReg && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="panel-inset p-3.5 border border-[var(--hairline)]">
                <div className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 mb-2">
                  Cross-Asset Permutation Factor Importance (Holdout RMSE Impact)
                </div>
                <div className="space-y-2">
                  {activeTcnReg.factorImportance.map((f) => (
                    <div key={f.ticker} className="flex items-center gap-3 font-mono text-[11px]">
                      <span
                        className={`w-28 shrink-0 font-semibold truncate ${
                          f.isHmmRegime
                            ? 'text-gold-700 dark:text-gold-300'
                            : 'text-zinc-800 dark:text-zinc-200'
                        }`}
                      >
                        {f.ticker}{f.isDependentLag ? ' (lag)' : f.isHmmRegime ? ' (3s)' : ''}
                      </span>
                      <div className="h-2.5 flex-1 overflow-hidden rounded-sm bg-zinc-200/70 dark:bg-zinc-800">
                        <div
                          className={`h-full rounded-sm ${f.isHmmRegime ? 'bg-emerald-500/80' : 'bg-gold-500/80'}`}
                          style={{ width: `${Math.max(3, f.importancePct).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="w-24 shrink-0 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                        {f.importancePct.toFixed(1)}% <span className="text-zinc-400">(+{f.rmseIncreaseBps.toFixed(1)}bp)</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel-inset p-3.5 border border-[var(--hairline)]">
                <div className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 mb-2">
                  Causal Self-Attention Transmission Dates
                </div>
                <div className="space-y-1.5 font-mono text-[11px]">
                  {activeTcnReg.topTransmissionDates.map((d) => (
                    <div key={d.date} className="flex items-center justify-between border-b border-[var(--hairline)] last:border-0 pb-1 last:pb-0 tabular-nums">
                      <span className="text-zinc-700 dark:text-zinc-200 font-semibold">
                        {d.date} <span className="font-normal text-zinc-400">(lag t-{d.lag})</span>
                      </span>
                      <span className="text-gold-700 dark:text-gold-300 font-semibold">
                        attn {(d.weight * 100).toFixed(1)}% · {d.metricLabel} {d.metricValue >= 0 ? '+' : ''}{d.metricValue.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 font-mono text-[9.5px] text-zinc-400">
                  Evaluated strictly out-of-sample on the trailing chronological holdout split against a lag-matched OLS linear baseline.
                </p>
              </div>
            </div>
          )}

          {activeTcnReg && <TcnRegressionOosChart tcnReg={activeTcnReg} />}
        </div>
    </div>
  );
}
