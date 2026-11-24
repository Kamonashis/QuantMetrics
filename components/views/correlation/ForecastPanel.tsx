'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { RotateCcw, Sparkles } from 'lucide-react';
import {
  getCachedTcnCorrelationForecast,
  runTcnCorrelationForecast,
  type TcnCorrelationForecast,
} from '@/lib/tcn-multi-domain';
import type { CorrelationSample } from '@/lib/correlation-engine';
import { HmmRegimeScorecard } from '@/components/charts/HmmRegimeOverlayControls';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { buildHmmRegimeShapes, getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout } from '@/components/ui/lab-ui';
import type { Pair } from './shared';

const FORECAST_DAYS = 30;

/**
 * Experimental: a sequence model that forecasts the pair's correlation 1, 5 and
 * 22 days ahead, scored on a trailing holdout. Training is manual.
 */
export function ForecastPanel({ sample, pair }: { sample: CorrelationSample; pair: Pair }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [a, b] = pair;
  const ia = sample.tickers.indexOf(a);
  const ib = sample.tickers.indexOf(b);
  const returnsA = ia >= 0 ? sample.returns[ia] : null;
  const returnsB = ib >= 0 ? sample.returns[ib] : null;
  const dates = sample.dates;
  const enough = returnsA !== null && returnsB !== null && ia !== ib && dates.length >= 70;

  const [result, setResult] = useState<TcnCorrelationForecast | null>(() =>
    enough ? getCachedTcnCorrelationForecast(a, b, returnsA, returnsB, dates, FORECAST_DAYS) : null
  );
  const [training, setTraining] = useState(false);
  const [trainRequest, setTrainRequest] = useState(0);
  const dataRef = useRef({ returnsA, returnsB, dates });
  const trainingRef = useRef(false);
  useEffect(() => {
    dataRef.current = { returnsA, returnsB, dates };
  }, [returnsA, returnsB, dates]);

  const firstDate = dates[0] ?? '';
  const lastDate = dates[dates.length - 1] ?? '';

  // Cache hydration follows the pair and data; it never starts a fit.
  useEffect(() => {
    if (!enough) return;
    const timer = setTimeout(() => {
      if (trainingRef.current) return;
      const d = dataRef.current;
      if (!d.returnsA || !d.returnsB) return;
      setResult(getCachedTcnCorrelationForecast(a, b, d.returnsA, d.returnsB, d.dates, FORECAST_DAYS));
    }, 0);
    return () => clearTimeout(timer);
  }, [a, b, enough, firstDate, lastDate]);

  useEffect(() => {
    if (trainRequest === 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      const d = dataRef.current;
      if (cancelled || !d.returnsA || !d.returnsB) return;
      trainingRef.current = true;
      setTraining(true);
      void runTcnCorrelationForecast(a, b, d.returnsA, d.returnsB, d.dates, FORECAST_DAYS, {
        forceRetrain: true,
        yieldControl: () => new Promise<void>((r) => setTimeout(r, 0)),
        isCancelled: () => cancelled,
      }).then((res) => {
        trainingRef.current = false;
        if (!cancelled) {
          if (res) setResult(res);
          setTraining(false);
        }
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      trainingRef.current = false;
    };
  }, [trainRequest, a, b]);

  const active = result && result.tickerA === a.toUpperCase() && result.tickerB === b.toUpperCase() ? result : null;

  const chart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!active) return { traces: [], layout: {} };
    const dyn = active.dynamicCorrelations;
    const xs = dyn.map((p) => p.date);
    const split = Math.max(0, Math.min(dyn.length, active.holdoutStartIdx));
    const shapes: Layout['shapes'] = [...buildHmmRegimeShapes(xs, active.hmmStates, active.hmmStates.length === dyn.length)];
    if (split > 0 && split < dyn.length) {
      shapes.push({ type: 'line', xref: 'x', yref: 'paper', x0: xs[split], x1: xs[split], y0: 0, y1: 1, line: { color: tokens.textMuted, width: 1, dash: 'dot' } });
    }
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Fitted (training period)',
          x: xs.slice(0, split + 1),
          y: dyn.slice(0, split + 1).map((p) => p.correlation),
          line: { color: QUANT_COLORS.slate, width: 1.3 },
          hovertemplate: '%{y:.2f}<extra>training</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Unseen period',
          x: xs.slice(split),
          y: dyn.slice(split).map((p) => p.correlation),
          line: { color: QUANT_COLORS.gold, width: 2 },
          hovertemplate: '%{y:.2f}<extra>unseen</extra>',
        },
      ],
      layout: { margin: { l: 44, r: 12, t: 24, b: 30 }, shapes, yaxis: { range: [-1, 1] } },
    };
  }, [active, tokens]);

  return (
    <section className="panel p-5">
      <div className="mb-4 flex flex-col gap-3 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="kicker mb-1 flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5" /> Experimental forecast
          </p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            Where might {a} / {b} correlation go next?
          </h3>
          <p className="mt-0.5 max-w-3xl text-xs text-zinc-500 dark:text-zinc-400">
            A neural sequence model (TCN with attention) trained in your browser on this pair and scored on the most recent stretch it never saw.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setTrainRequest((v) => v + 1)}
          disabled={training || !enough}
          className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 self-start rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[11px] font-semibold text-zinc-700 hover:border-gold-500/50 disabled:opacity-50 dark:text-zinc-300"
        >
          <RotateCcw className={`h-3.5 w-3.5 ${training ? 'animate-spin' : ''}`} />
          {training ? 'Training…' : active ? 'Retrain' : 'Train model'}
        </button>
      </div>

      {!active && !training && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {enough ? 'Not trained for this pair yet. Training takes a few seconds.' : 'Needs two different assets and at least 70 daily returns.'}
        </p>
      )}

      {active && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            {([1, 5, 22] as const).map((h) => {
              const hz = active.horizons[h];
              return (
                <div key={h} className="panel-inset p-3.5">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-zinc-400">
                    <span className="font-semibold text-gold-700 dark:text-gold-300">{h}-day ahead</span>
                    <span>tested on {hz.holdoutCount} days</span>
                  </div>
                  <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                    <span className="text-xl font-bold text-zinc-900 dark:text-zinc-100">{hz.predictedCorr.toFixed(2)}</span>
                    <span className="text-[11px] text-zinc-500">
                      {hz.deltaVsCurrent >= 0 ? '+' : ''}
                      {hz.deltaVsCurrent.toFixed(2)} vs now
                    </span>
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                    99% range {hz.lower99.toFixed(2)} to {hz.upper99.toFixed(2)} · typical miss {hz.oosMae.toFixed(2)}
                  </p>
                </div>
              );
            })}
          </div>

          <Callout tone="warn" title="Read with care">
            The &quot;typical miss&quot; is the model&apos;s average error on unseen days. The model does not report how a naive forecast
            (&quot;correlation stays where it is&quot;) would have scored, so these figures cannot show that it adds anything. Use the
            ranges, not the point forecasts.
          </Callout>

          <PlotlyChart
            data={chart.traces}
            layout={chart.layout}
            plotType="2d"
            hovermode="x unified"
            showLegend
            height={260}
            exportFilename={`${a}-${b}-correlation-forecast`}
            uirevisionKey={`tcn-corr-${a}-${b}`}
          />
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
            Shading marks the model&apos;s volatility regime at each date (calm, watchful, crisis). The dotted line is where the unseen period
            starts. Trained in {active.cacheHit ? 'an earlier run (cached)' : `${(active.trainingDurationMs / 1000).toFixed(1)}s`} ·{' '}
            {active.parameterCount.toLocaleString()} parameters.
          </p>

          {active.regimePerformance.length > 0 && (
            <HmmRegimeScorecard
              title="1-day forecast error on unseen days, by volatility regime"
              subtitle="Regime is the model's estimate the day before each forecast"
              columns={[
                { key: 'mae', label: 'Typical miss' },
                { key: 'pred', label: 'Mean forecast' },
                { key: 'act', label: 'Mean outcome' },
              ]}
              rows={active.regimePerformance.map((r) => ({
                state: r.state,
                count: r.count,
                cells: {
                  mae: r.oosMae !== null ? r.oosMae.toFixed(3) : '—',
                  pred: r.meanPredCorr !== null ? r.meanPredCorr.toFixed(2) : '—',
                  act: r.meanActualCorr !== null ? r.meanActualCorr.toFixed(2) : '—',
                },
              }))}
            />
          )}
        </div>
      )}
    </section>
  );
}
