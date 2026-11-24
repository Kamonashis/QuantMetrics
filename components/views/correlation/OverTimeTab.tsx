'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  compareRecentToPrior,
  ewmaPair,
  fitDcc,
  matrixDrift,
  percentileRank,
  quantile,
  rollingBasket,
  rollingPair,
  type CorrelationSample,
} from '@/lib/correlation-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout, SectionHeader, SegmentedControl, StatTile, fmtP, pText } from '@/components/ui/lab-ui';
import { ForecastPanel } from './ForecastPanel';
import { DataTable, PairPicker, PercentileChip, fmtCorr, num, ordinal, td, tdClass, thClass, type Pair } from './shared';

type WindowSize = 63 | 126 | 252;
const WINDOW_LABEL: Record<WindowSize, string> = { 63: 'three months', 126: 'six months', 252: 'year' };
const WINDOW_ADJ: Record<WindowSize, string> = { 63: 'three-month', 126: 'six-month', 252: 'one-year' };
type BasketMetric = 'avgCorr' | 'effectiveBets' | 'topFactorShare';

function hline(y: number, color: string, dash: 'dot' | 'dash' | 'solid' = 'dot'): NonNullable<Layout['shapes']>[number] {
  return { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: y, y1: y, line: { color, width: 1, dash } };
}

export function OverTimeTab({ sample, pair, onPairChange }: { sample: CorrelationSample; pair: Pair; onPairChange: (p: Pair) => void }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [windowSize, setWindowSize] = useState<WindowSize>(126);
  const [metric, setMetric] = useState<BasketMetric>('avgCorr');
  const [showDcc, setShowDcc] = useState(true);

  const n = sample.dates.length;
  const usable = ([252, 126, 63] as const).filter((w) => n >= 2 * w + 5);
  const win: WindowSize | null = usable.includes(windowSize) ? windowSize : usable[usable.length - 1] ?? null;

  const basket = useMemo(() => (win ? rollingBasket(sample, win) : null), [sample, win]);
  const drift = useMemo(() => (win ? matrixDrift(sample, win) : null), [sample, win]);
  const comparison = useMemo(() => (win ? compareRecentToPrior(sample, win) : null), [sample, win]);

  const ia = sample.tickers.indexOf(pair[0]);
  const ib = sample.tickers.indexOf(pair[1]);
  const validPair = ia >= 0 && ib >= 0 && ia !== ib;
  const pairRoll = useMemo(
    () => (validPair && win ? rollingPair(sample.returns[ia], sample.returns[ib], sample.dates, win) : null),
    [sample, ia, ib, validPair, win]
  );
  const ewma = useMemo(() => (validPair ? ewmaPair(sample.returns[ia], sample.returns[ib]) : null), [sample, ia, ib, validPair]);
  const dcc = useMemo(() => (validPair ? fitDcc(sample.returns[ia], sample.returns[ib]) : null), [sample, ia, ib, validPair]);

  const basketStats = useMemo(() => {
    if (!basket) return null;
    const mk = (arr: number[]) => {
      const last = arr[arr.length - 1];
      const sorted = [...arr].sort((a, b) => a - b);
      return { last, pct: percentileRank(arr, last), p10: quantile(sorted, 0.1), p50: quantile(sorted, 0.5), p90: quantile(sorted, 0.9) };
    };
    return { avgCorr: mk(basket.avgCorr), effectiveBets: mk(basket.effectiveBets), topFactorShare: mk(basket.topFactorShare) };
  }, [basket]);

  const basketChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!basket || !basketStats) return { traces: [], layout: {} };
    const st = basketStats[metric];
    const scale = metric === 'topFactorShare' ? 100 : 1;
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          x: basket.dates,
          y: basket[metric].map((v) => v * scale),
          line: { color: QUANT_COLORS.gold, width: 2 },
          hovertemplate: metric === 'topFactorShare' ? '%{y:.0f}%<extra></extra>' : '%{y:.2f}<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 46, r: 14, t: 8, b: 30 },
        shapes: [hline(st.p10 * scale, tokens.textMuted), hline(st.p50 * scale, tokens.textMuted, 'dash'), hline(st.p90 * scale, tokens.textMuted)],
        yaxis: { ticksuffix: metric === 'topFactorShare' ? '%' : '' },
      },
    };
  }, [basket, basketStats, metric, tokens]);

  const driftStats = useMemo(() => {
    if (!drift) return null;
    const last = drift.drift[drift.drift.length - 1];
    const sorted = [...drift.drift].sort((a, b) => a - b);
    return { last, pct: percentileRank(drift.drift, last), p50: quantile(sorted, 0.5), p90: quantile(sorted, 0.9) };
  }, [drift]);

  const driftChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!drift || !driftStats) return { traces: [], layout: {} };
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          x: drift.dates,
          y: drift.drift,
          line: { color: QUANT_COLORS.slate, width: 1.6 },
          fill: 'tozeroy',
          fillcolor: 'rgba(140, 140, 154, 0.12)',
          hovertemplate: 'typical pair moved %{y:.2f}<extra></extra>',
        },
      ],
      layout: { margin: { l: 46, r: 14, t: 8, b: 30 }, shapes: [hline(driftStats.p90, QUANT_COLORS.amber, 'dash')], yaxis: { rangemode: 'tozero' } },
    };
  }, [drift, driftStats]);

  const pairChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!pairRoll) return { traces: [], layout: {} };
    const traces: Data[] = [
      { type: 'scatter', mode: 'lines', x: pairRoll.dates, y: pairRoll.high, line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      {
        type: 'scatter',
        mode: 'lines',
        x: pairRoll.dates,
        y: pairRoll.low,
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(201, 164, 92, 0.16)',
        name: '95% range',
        hoverinfo: 'skip',
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: pairRoll.dates,
        y: pairRoll.corr,
        name: `Rolling ${win}-day`,
        line: { color: QUANT_COLORS.gold, width: 2 },
        hovertemplate: '%{y:.2f}<extra>rolling</extra>',
      },
    ];
    if (ewma) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: sample.dates,
        y: ewma,
        name: 'Fast (exponential, ~1 month)',
        line: { color: QUANT_COLORS.slate, width: 1, dash: 'dot' },
        hovertemplate: '%{y:.2f}<extra>fast</extra>',
      });
    }
    if (dcc && showDcc) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: sample.dates,
        y: dcc.path,
        name: 'DCC model',
        line: { color: tokens.text, width: 1.2 },
        hovertemplate: '%{y:.2f}<extra>DCC</extra>',
      });
    }
    return { traces, layout: { margin: { l: 46, r: 14, t: 26, b: 30 }, shapes: [hline(0, tokens.zeroLine, 'solid')] } };
  }, [pairRoll, ewma, dcc, showDcc, sample.dates, win, tokens]);

  if (!win || !basket || !basketStats) {
    return <Callout tone="warn">At least {2 * 63 + 5} daily returns are needed to study how correlation changes. Widen the date range.</Callout>;
  }

  const avg = basketStats.avgCorr;
  const fullAvgNote =
    avg.pct <= 0.1
      ? 'unusually low for this basket: diversification is working better than usual'
      : avg.pct >= 0.9
        ? 'unusually high for this basket: diversification is weaker than usual'
        : 'within its normal range';
  const metricLabel: Record<BasketMetric, string> = {
    avgCorr: 'Average correlation',
    effectiveBets: 'Independent bets',
    topFactorShare: 'Top-factor share',
  };

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <SectionHeader
          kicker="Basket"
          title="Is diversification holding?"
          description={`Each point uses the previous ${WINDOW_LABEL[win]} of daily returns. Dotted lines mark the 10th and 90th percentiles of the series; the dashed line is its median.`}
          right={
            <SegmentedControl
              ariaLabel="Rolling window"
              value={win}
              onChange={(v) => setWindowSize(v)}
              options={[
                { value: 63 as WindowSize, label: '3M' },
                { value: 126 as WindowSize, label: '6M' },
                { value: 252 as WindowSize, label: '1Y' },
              ].filter((o) => usable.includes(o.value))}
            />
          }
        />
        <p className="mb-4 text-sm text-zinc-700 dark:text-zinc-300">
          Over the last {WINDOW_LABEL[win]}, average correlation is <strong className="font-mono">{avg.last.toFixed(2)}</strong>, at the{' '}
          {ordinal(avg.pct)} percentile of its history. That is {fullAvgNote}.
        </p>
        <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          {(['avgCorr', 'effectiveBets', 'topFactorShare'] as const).map((m) => {
            const st = basketStats[m];
            const active = m === metric;
            return (
              <button
                key={m}
                type="button"
                onClick={() => setMetric(m)}
                aria-pressed={active}
                className={`cursor-pointer rounded-xl border p-3.5 text-left transition-colors ${
                  active ? 'border-gold-500/60 bg-gold-500/10' : 'border-[var(--hairline)] bg-[var(--bg-inset)] hover:border-gold-500/40'
                }`}
              >
                <span className="block text-[11px] text-zinc-500 dark:text-zinc-400">{metricLabel[m]}</span>
                <span className="mt-0.5 block font-mono text-xl font-bold tabular-nums text-zinc-900 dark:text-zinc-100">
                  {m === 'topFactorShare' ? `${(st.last * 100).toFixed(0)}%` : st.last.toFixed(m === 'avgCorr' ? 2 : 1)}
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                  <PercentileChip value={st.pct} />
                  median {m === 'topFactorShare' ? `${(st.p50 * 100).toFixed(0)}%` : st.p50.toFixed(m === 'avgCorr' ? 2 : 1)}
                </span>
              </button>
            );
          })}
        </div>
        <PlotlyChart
          data={basketChart.traces}
          layout={basketChart.layout}
          plotType="2d"
          hovermode="x unified"
          showLegend={false}
          height={260}
          exportFilename={`rolling-${metric}`}
          uirevisionKey={`roll-${metric}-${win}-${sample.tickers.join('-')}`}
        />
        <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
          Independent bets = the number of uncorrelated, equally risky assets that would diversify as well as this basket. Top-factor share = how
          much of the basket&apos;s variance one common driver explains.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="Structure"
            title="How fast is the pattern changing?"
            description={`How much a typical pair's correlation moved between one ${WINDOW_ADJ[win]} window and the next. The dashed line is the 90th percentile.`}
          />
          {drift && driftStats ? (
            <>
              <p className="mb-3 flex flex-wrap items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                Latest: a typical pair moved <strong className="font-mono">{driftStats.last.toFixed(2)}</strong> <PercentileChip value={driftStats.pct} />
                <span>median {driftStats.p50.toFixed(2)}</span>
              </p>
              <PlotlyChart
                data={driftChart.traces}
                layout={driftChart.layout}
                plotType="2d"
                hovermode="x unified"
                showLegend={false}
                height={220}
                exportFilename="correlation-drift"
                uirevisionKey={`drift-${win}-${sample.tickers.join('-')}`}
              />
            </>
          ) : (
            <Callout>Not enough history for two full windows.</Callout>
          )}
        </section>

        <section className="panel p-5">
          <SectionHeader
            kicker="Latest change"
            title="Is the recent shift real?"
            description={
              comparison
                ? `Compares ${comparison.recentStart} onward with the ${WINDOW_LABEL[win]} before it (${comparison.priorStart} to ${comparison.priorEnd}).`
                : undefined
            }
          />
          {comparison ? (
            <>
              <Callout
                tone={comparison.test && comparison.test.p < 0.01 ? 'warn' : 'info'}
                title={
                  !comparison.test
                    ? 'The joint test could not be computed.'
                    : comparison.test.p < 0.01
                      ? 'Yes. The correlation matrix changed by more than chance explains.'
                      : comparison.test.p < 0.1
                        ? 'Possibly. The evidence of a change is weak.'
                        : 'No clear evidence. The difference is within normal sampling noise.'
                }
              >
                Average correlation went from {comparison.priorAvg.toFixed(2)} to {comparison.recentAvg.toFixed(2)}.
                {comparison.test && (
                  <>
                    {' '}
                    Jennrich test: <span className="font-mono">χ²({comparison.test.df}) = {comparison.test.chi2.toFixed(1)}, {pText(comparison.test.p)}</span>.
                    It assumes normal returns, so it tends to flag changes too readily.
                  </>
                )}
              </Callout>
              <div className="mt-3">
                <DataTable>
                  <thead>
                    <tr>
                      <th className={thClass}>Biggest movers</th>
                      <th className={thClass}>Before</th>
                      <th className={thClass}>Now</th>
                      <th className={thClass}>Change</th>
                      <th className={thClass}>p</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.pairs.slice(0, 6).map((p) => (
                      <tr key={`${p.a}-${p.b}`}>
                        <td className={tdClass}>
                          <button type="button" onClick={() => onPairChange([p.a, p.b])} className="cursor-pointer font-semibold text-zinc-900 underline-offset-2 hover:underline dark:text-zinc-100">
                            {p.a} / {p.b}
                          </button>
                        </td>
                        <td className={tdClass}>{num(p.prior)}</td>
                        <td className={tdClass}>{num(p.recent)}</td>
                        <td className={td(null, 'font-semibold')}>{fmtCorr(p.change)}</td>
                        <td className={td(p.p < 0.01 && 'text-amber-600 dark:text-amber-400')}>{fmtP(p.p)}</td>
                      </tr>
                    ))}
                  </tbody>
                </DataTable>
                <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                  With {comparison.pairs.length} pairs tested, a few low p-values are expected by chance.
                </p>
              </div>
            </>
          ) : (
            <Callout>Not enough history for two full windows.</Callout>
          )}
        </section>
      </div>

      <section className="panel p-5">
        <SectionHeader
          kicker="Pair"
          title={`${pair[0]} / ${pair[1]} through time`}
          description="The gold line and band are the rolling estimate and its 95% range. If the band is wide, short-term wiggles are mostly noise."
          right={<PairPicker tickers={sample.tickers} pair={pair} onChange={onPairChange} />}
        />
        {pairRoll ? (
          <>
            <PlotlyChart
              data={pairChart.traces}
              layout={pairChart.layout}
              plotType="2d"
              hovermode="x unified"
              showLegend
              height={300}
              exportFilename={`${pair[0]}-${pair[1]}-rolling-correlation`}
              uirevisionKey={`pair-roll-${pair.join('-')}-${win}`}
            />
            {dcc ? (
              <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
                <div className="grid grid-cols-2 gap-3">
                  <StatTile label="Model estimate, next day" value={dcc.next.toFixed(2)} tone="gold" sub={`Long-run level ${dcc.longRun.toFixed(2)}`} />
                  <StatTile
                    label="Half-life of a shift"
                    value={dcc.halfLifeDays === null ? '—' : dcc.nearIntegrated ? '> 6 mo' : `${Math.round(dcc.halfLifeDays)} d`}
                    sub={`a = ${dcc.a.toFixed(3)}, b = ${dcc.b.toFixed(3)}`}
                  />
                </div>
                <Callout
                  tone="info"
                  title={
                    dcc.lrP < 0.01
                      ? 'Correlation between this pair does change over time.'
                      : 'A constant correlation fits this pair about as well as a changing one.'
                  }
                >
                  The DCC model first removes each asset&apos;s own volatility swings (GARCH), then lets correlation react to recent co-movement.
                  Against a constant-correlation model: <span className="font-mono">LR = {dcc.lrStat.toFixed(1)}, {pText(dcc.lrP)}</span>.
                  {dcc.nearIntegrated && ' Shifts barely decay, so the long-run level is a weak anchor.'}{' '}
                  <button type="button" onClick={() => setShowDcc((v) => !v)} className="cursor-pointer font-semibold text-gold-700 underline-offset-2 hover:underline dark:text-gold-300">
                    {showDcc ? 'Hide' : 'Show'} on chart
                  </button>
                </Callout>
              </div>
            ) : (
              <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">The DCC model needs at least 250 daily returns.</p>
            )}
          </>
        ) : (
          <Callout>Pick two different assets.</Callout>
        )}
      </section>

      {validPair && <ForecastPanel key={pair.join('|')} sample={sample} pair={pair} />}
    </div>
  );
}
