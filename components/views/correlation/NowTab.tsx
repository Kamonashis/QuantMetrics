'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { Check, Copy, Download, RefreshCw } from 'lucide-react';
import {
  blockBootstrapInterval,
  correlationCsv,
  correlationLatex,
  hierarchicalOrder,
  horizonCorrelations,
  intradayCorrelationBySampling,
  ledoitWolf,
  matrixDrift,
  partialCorrelationMatrix,
  pearson,
  percentileRank,
  rollingBasket,
  spearman,
  spearmanMatrix,
  type BasketSummary,
  type CorrelationSample,
  type IntradayCorrelation,
} from '@/lib/correlation-engine';
import type { IntradayBar } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout, SectionHeader, SegmentedControl, StatTile } from '@/components/ui/lab-ui';
import { DataTable, MatrixHeatmap, PairPicker, fmtCorr, num, ordinal, tdClass, thClass, type Pair } from './shared';

type Measure = 'pearson' | 'rank' | 'partial' | 'shrunk';

const MEASURE_NOTE: Record<Measure, string> = {
  pearson: 'Standard correlation of daily returns.',
  rank: 'Spearman rank correlation: ignores the size of moves, so a few extreme days cannot dominate.',
  partial: 'Correlation of each pair after removing what every other asset in the basket explains. Shows direct links.',
  shrunk: 'Ledoit–Wolf estimate: pulls noisy pair values toward the basket average. Better for building portfolios.',
};

const RECENT_WINDOW = 126;

interface IntradayState {
  key: string;
  status: 'loading' | 'done' | 'error';
  result?: IntradayCorrelation;
  synthetic?: string[];
}

async function fetchIntraday(ticker: string): Promise<{ bars: IntradayBar[]; source: string }> {
  const res = await fetch(`/api/market-data?ticker=${encodeURIComponent(ticker)}&interval=1m&range=7d`, { cache: 'no-store' });
  const json = await res.json();
  const payload = json?.data?.[ticker];
  return { bars: payload?.intradayBars ?? [], source: typeof payload?.source === 'string' ? payload.source : 'synthetic_gbm' };
}

function intradayReadout(result: IntradayCorrelation): string {
  const rows = result.rows.filter((r): r is typeof r & { corr: number } => r.corr !== null);
  if (rows.length < 2) return 'Too few matching bars to compare sampling intervals.';
  const first = rows[0];
  const last = rows[rows.length - 1];
  const band = (1.96 * (1 - last.corr * last.corr)) / Math.sqrt(last.observations);
  const diff = last.corr - first.corr;
  if (Math.abs(diff) <= band) {
    return `Correlation is similar at ${first.minutes}-minute and ${last.minutes}-minute sampling (the gap is within the ±${band.toFixed(2)} uncertainty), so asynchronous trading is not distorting the short-interval figure much.`;
  }
  return diff > 0
    ? `Correlation rises from ${first.corr.toFixed(2)} at ${first.minutes}-minute sampling to ${last.corr.toFixed(2)} at ${last.minutes}-minute sampling. Short intervals understate it because the two assets do not trade at the same instant (the Epps effect).`
    : `Correlation is lower at ${last.minutes}-minute sampling (${last.corr.toFixed(2)}) than at ${first.minutes}-minute sampling (${first.corr.toFixed(2)}). That is unusual; with only ${last.observations} returns at the longer interval it may not persist.`;
}

export function NowTab({
  sample,
  corr,
  summary,
  pair,
  onPairChange,
}: {
  sample: CorrelationSample;
  corr: number[][];
  summary: BasketSummary;
  pair: Pair;
  onPairChange: (p: Pair) => void;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [measure, setMeasure] = useState<Measure>('pearson');
  const [clustered, setClustered] = useState(false);
  const [copied, setCopied] = useState(false);
  const [intraday, setIntraday] = useState<IntradayState | null>(null);

  const n = sample.tickers.length;
  const T = sample.dates.length;

  // Headline context: where the last six months sit against the sample's own history.
  const recent = useMemo(() => {
    const roll = rollingBasket(sample, RECENT_WINDOW);
    const drift = matrixDrift(sample, 63);
    if (!roll) return null;
    const last = roll.avgCorr[roll.avgCorr.length - 1];
    const driftLast = drift ? drift.drift[drift.drift.length - 1] : null;
    return {
      avg: last,
      pct: percentileRank(roll.avgCorr, last),
      bets: roll.effectiveBets[roll.effectiveBets.length - 1],
      driftPct: drift && driftLast !== null ? percentileRank(drift.drift, driftLast) : null,
    };
  }, [sample]);

  const shrunk = useMemo(() => ledoitWolf(sample.returns), [sample]);
  const matrix = useMemo<number[][] | null>(() => {
    if (measure === 'rank') return spearmanMatrix(sample.returns);
    if (measure === 'partial') return partialCorrelationMatrix(corr);
    if (measure === 'shrunk') return shrunk.corr;
    return corr;
  }, [measure, sample, corr, shrunk]);

  const order = useMemo(() => (clustered ? hierarchicalOrder(corr) : sample.tickers.map((_, i) => i)), [clustered, corr, sample.tickers]);
  const orderedTickers = order.map((i) => sample.tickers[i]);
  const orderedMatrix = matrix ? order.map((i) => order.map((j) => matrix[i][j])) : null;

  const ia = sample.tickers.indexOf(pair[0]);
  const ib = sample.tickers.indexOf(pair[1]);
  const pairStats = useMemo(() => {
    if (ia < 0 || ib < 0 || ia === ib) return null;
    const x = sample.returns[ia];
    const y = sample.returns[ib];
    const r = pearson(x, y);
    const ci = blockBootstrapInterval(x, y);
    let sxx = 0;
    let sxy = 0;
    const mx = x.reduce((a, b) => a + b, 0) / x.length;
    const my = y.reduce((a, b) => a + b, 0) / y.length;
    for (let t = 0; t < x.length; t++) {
      sxx += (x[t] - mx) ** 2;
      sxy += (x[t] - mx) * (y[t] - my);
    }
    const beta = sxx > 0 ? sxy / sxx : 0;
    return { r, ci, rank: spearman(x, y), beta, alpha: my - beta * mx };
  }, [sample, ia, ib]);

  const horizons = useMemo(() => horizonCorrelations(sample), [sample]);

  const scatter = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!pairStats) return { traces: [], layout: {} };
    const x = sample.returns[ia].map((v) => v * 100);
    const y = sample.returns[ib].map((v) => v * 100);
    const lo = Math.min(...x);
    const hi = Math.max(...x);
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'markers',
          x,
          y,
          text: sample.dates as unknown as string[],
          marker: { color: QUANT_COLORS.gold, size: 4, opacity: 0.45 },
          hovertemplate: '%{text}<br>' + pair[0] + ' %{x:+.2f}% · ' + pair[1] + ' %{y:+.2f}%<extra></extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: [lo, hi],
          y: [pairStats.alpha * 100 + pairStats.beta * lo, pairStats.alpha * 100 + pairStats.beta * hi],
          line: { color: tokens.text, width: 1.5 },
          hoverinfo: 'skip',
        },
      ],
      layout: {
        margin: { l: 48, r: 10, t: 8, b: 40 },
        xaxis: { title: { text: `${pair[0]} daily return`, font: { size: 10, color: tokens.textMuted } }, ticksuffix: '%', zeroline: true, zerolinecolor: tokens.zeroLine },
        yaxis: { title: { text: `${pair[1]} daily return`, font: { size: 10, color: tokens.textMuted } }, ticksuffix: '%', zeroline: true, zerolinecolor: tokens.zeroLine },
      },
    };
  }, [pairStats, sample, ia, ib, pair, tokens]);

  const pairKey = `${pair[0]}|${pair[1]}`;
  const activeIntraday = intraday && intraday.key === pairKey ? intraday : null;
  const loadIntraday = () => {
    setIntraday({ key: pairKey, status: 'loading' });
    void Promise.all([fetchIntraday(pair[0]), fetchIntraday(pair[1])])
      .then(([a, b]) => {
        const synthetic = [a, b].map((d, i) => (d.source === 'live_market' ? null : pair[i])).filter((t): t is string => t !== null);
        setIntraday({
          key: pairKey,
          status: 'done',
          synthetic,
          result: synthetic.length === 0 ? intradayCorrelationBySampling(a.bars, b.bars) : undefined,
        });
      })
      .catch(() => setIntraday({ key: pairKey, status: 'error' }));
  };

  const download = () => {
    if (!orderedMatrix) return;
    const blob = new Blob([correlationCsv(orderedTickers, orderedMatrix)], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `correlation-${measure}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const copyLatex = () => {
    if (!orderedMatrix) return;
    void navigator.clipboard.writeText(correlationLatex(orderedTickers, orderedMatrix, `${measure} correlation, ${sample.audit.start} to ${sample.audit.end}`)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const direction = recent ? (recent.avg < summary.avgCorr - 0.03 ? 'lower than' : recent.avg > summary.avgCorr + 0.03 ? 'higher than' : 'about the same as') : null;
  const buttonClass =
    'inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[11px] font-semibold text-zinc-700 hover:border-gold-500/50 disabled:opacity-50 dark:text-zinc-300';

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <p className="kicker mb-2">The short answer</p>
        <p className="font-display text-lg leading-snug text-zinc-900 dark:text-zinc-100 sm:text-xl">
          These {n} assets diversify about as well as{' '}
          <strong className="tabular-nums text-gold-600 dark:text-gold-300">{summary.effectiveBets.toFixed(1)}</strong> independent ones.
          Their average correlation is <strong className="tabular-nums">{summary.avgCorr.toFixed(2)}</strong> over the full sample
          {recent && direction && (
            <>
              {' '}
              and <strong className="tabular-nums">{recent.avg.toFixed(2)}</strong> over the last six months, {direction} the long-run level.
            </>
          )}
          {!recent && '.'}
        </p>
        {recent && (
          <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
            The six-month reading sits at the {ordinal(recent.pct)} percentile of its own history since {sample.audit.start}.
            {recent.driftPct !== null && recent.driftPct >= 0.9 && ' The pattern of which assets move together is also shifting unusually fast; see Over time.'}
          </p>
        )}
        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-5">
          <StatTile label="Average correlation" value={num(summary.avgCorr)} tone="gold" sub={`${(n * (n - 1)) / 2} pairs · ${T.toLocaleString()} days`} />
          <StatTile label="Independent bets" value={summary.effectiveBets.toFixed(1)} sub={`out of ${n} assets`} />
          <StatTile label="Top common factor" value={`${(summary.topFactorShare * 100).toFixed(0)}%`} sub="of total variance" />
          <StatTile label="Tightest pair" value={num(summary.strongest.corr)} sub={`${summary.strongest.a} / ${summary.strongest.b}`} />
          <StatTile label="Loosest pair" value={num(summary.weakest.corr)} sub={`${summary.weakest.a} / ${summary.weakest.b}`} />
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="Matrix"
            title="Who moves with whom"
            description={MEASURE_NOTE[measure]}
            right={
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={download} className={buttonClass} disabled={!orderedMatrix}>
                  <Download className="h-3.5 w-3.5" /> CSV
                </button>
                <button type="button" onClick={copyLatex} className={buttonClass} disabled={!orderedMatrix}>
                  {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} {copied ? 'Copied' : 'LaTeX'}
                </button>
              </div>
            }
          />
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <SegmentedControl
              ariaLabel="Correlation measure"
              value={measure}
              onChange={setMeasure}
              options={[
                { value: 'pearson', label: 'Standard' },
                { value: 'rank', label: 'Rank' },
                { value: 'partial', label: 'Direct links' },
                { value: 'shrunk', label: 'Shrunk' },
              ]}
            />
            <SegmentedControl
              ariaLabel="Row order"
              value={clustered ? 'clustered' : 'entered'}
              onChange={(v) => setClustered(v === 'clustered')}
              options={[
                { value: 'entered', label: 'As entered' },
                { value: 'clustered', label: 'Grouped' },
              ]}
            />
          </div>
          {orderedMatrix ? (
            <MatrixHeatmap tickers={orderedTickers} matrix={orderedMatrix} selected={pair} onSelect={onPairChange} exportName={`correlation-${measure}`} />
          ) : (
            <Callout tone="warn">Direct links cannot be computed: two or more assets are almost perfectly correlated.</Callout>
          )}
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            Click a cell to inspect that pair. The colour scale is fitted to the values shown.
            {measure === 'shrunk' && ` Shrinkage intensity ${(shrunk.delta * 100).toFixed(0)}%.`}
          </p>
        </section>

        <section className="panel p-5">
          <SectionHeader kicker="Pair" title={`${pair[0]} and ${pair[1]}`} right={<PairPicker tickers={sample.tickers} pair={pair} onChange={onPairChange} />} />
          {pairStats ? (
            <>
              <p className="mb-3 text-xs leading-relaxed text-zinc-600 dark:text-zinc-300">
                Correlation <strong className="font-mono">{pairStats.r.toFixed(2)}</strong>, with a 95% range of{' '}
                <span className="font-mono">
                  {pairStats.ci.low.toFixed(2)} to {pairStats.ci.high.toFixed(2)}
                </span>
                . On a day {pair[0]} moved 1%, {pair[1]} moved {Math.abs(pairStats.beta).toFixed(2)}% in the {pairStats.beta >= 0 ? 'same' : 'opposite'} direction on
                average. {pair[0]} explains {(pairStats.r * pairStats.r * 100).toFixed(0)}% of {pair[1]}&apos;s daily variance.
              </p>
              <div className="mb-3 grid grid-cols-3 gap-3">
                <StatTile label="Correlation" value={num(pairStats.r)} tone="gold" />
                <StatTile label="Rank correlation" value={num(pairStats.rank)} />
                <StatTile label="Shared variance" value={`${(pairStats.r * pairStats.r * 100).toFixed(0)}%`} />
              </div>
              <PlotlyChart
                data={scatter.traces}
                layout={scatter.layout}
                plotType="2d"
                hovermode="closest"
                showLegend={false}
                enableWheelPan={false}
                height={260}
                exportFilename={`${pair[0]}-${pair[1]}-scatter`}
                uirevisionKey={`scatter-${pairKey}`}
              />
              <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                The range comes from a block bootstrap ({pairStats.ci.blockLength}-day blocks), which allows for volatility clustering. It is wider than
                the textbook formula suggests.
              </p>
            </>
          ) : (
            <Callout>Pick two different assets.</Callout>
          )}
        </section>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="Holding period"
            title="Does correlation depend on the horizon?"
            description="The same sample re-measured with weekly and monthly returns (non-overlapping). Fewer observations mean wider uncertainty."
          />
          <DataTable>
            <thead>
              <tr>
                <th className={thClass}>Return horizon</th>
                <th className={thClass}>Observations</th>
                <th className={thClass}>Basket average</th>
                <th className={thClass}>
                  {pair[0]} / {pair[1]}
                </th>
              </tr>
            </thead>
            <tbody>
              {horizons.map((h) => (
                <tr key={h.days}>
                  <td className={tdClass}>{h.days === 1 ? 'Daily' : h.days === 5 ? 'Weekly (5 days)' : 'Monthly (21 days)'}</td>
                  <td className={tdClass}>{h.observations.toLocaleString()}</td>
                  <td className={tdClass}>{num(h.avgCorr)}</td>
                  <td className={tdClass}>{ia >= 0 && ib >= 0 && ia !== ib ? num(h.matrix[ia][ib]) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
          {horizons.length > 1 && (
            <p className="mt-3 text-xs text-zinc-600 dark:text-zinc-300">
              {Math.abs(horizons[horizons.length - 1].avgCorr - horizons[0].avgCorr) < 0.05
                ? 'Average correlation is about the same at every horizon, so daily figures are a fair guide for longer holding periods.'
                : horizons[horizons.length - 1].avgCorr > horizons[0].avgCorr
                  ? `Average correlation is higher over longer horizons (${horizons[horizons.length - 1].avgCorr.toFixed(2)} vs ${horizons[0].avgCorr.toFixed(2)} daily). Daily figures understate how much these assets move together over a month.`
                  : `Average correlation is lower over longer horizons (${horizons[horizons.length - 1].avgCorr.toFixed(2)} vs ${horizons[0].avgCorr.toFixed(2)} daily). Daily figures overstate how much these assets move together over a month.`}
            </p>
          )}
        </section>

        <section className="panel p-5">
          <SectionHeader
            kicker="Intraday check"
            title={`${pair[0]} / ${pair[1]} within the day`}
            description="Correlation from 1-minute bars over the last few sessions, re-sampled at coarser intervals. A few days of data: treat it as a spot check, not an estimate."
            right={
              <button type="button" onClick={loadIntraday} disabled={activeIntraday?.status === 'loading'} className={buttonClass}>
                <RefreshCw className={`h-3.5 w-3.5 ${activeIntraday?.status === 'loading' ? 'animate-spin' : ''}`} />
                {activeIntraday?.status === 'done' ? 'Reload' : 'Load intraday'}
              </button>
            }
          />
          {!activeIntraday && <p className="text-xs text-zinc-500 dark:text-zinc-400">Not loaded. This fetches 1-minute bars for both assets.</p>}
          {activeIntraday?.status === 'loading' && <p className="text-xs text-zinc-500 dark:text-zinc-400">Loading 1-minute bars…</p>}
          {activeIntraday?.status === 'error' && <Callout tone="warn">Could not load intraday data.</Callout>}
          {activeIntraday?.status === 'done' && activeIntraday.synthetic && activeIntraday.synthetic.length > 0 && (
            <Callout tone="warn">
              Live 1-minute data is unavailable for {activeIntraday.synthetic.join(' and ')}. Nothing is computed from simulated bars.
            </Callout>
          )}
          {activeIntraday?.status === 'done' && activeIntraday.result && (
            <>
              <DataTable>
                <thead>
                  <tr>
                    <th className={thClass}>Sampling interval</th>
                    <th className={thClass}>Returns used</th>
                    <th className={thClass}>Correlation</th>
                    <th className={thClass}>± (approx.)</th>
                  </tr>
                </thead>
                <tbody>
                  {activeIntraday.result.rows.map((r) => (
                    <tr key={r.minutes}>
                      <td className={tdClass}>{r.minutes} min</td>
                      <td className={tdClass}>{r.observations.toLocaleString()}</td>
                      <td className={tdClass}>{r.corr === null ? 'too few' : fmtCorr(r.corr)}</td>
                      <td className={tdClass}>{r.corr === null ? '—' : ((1.96 * (1 - r.corr * r.corr)) / Math.sqrt(r.observations)).toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </DataTable>
              <p className="mt-3 text-xs text-zinc-600 dark:text-zinc-300">{intradayReadout(activeIntraday.result)}</p>
              <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                {activeIntraday.result.days} sessions, {activeIntraday.result.overlappingBars.toLocaleString()} matching minutes. Returns never span
                the overnight gap. Daily correlation for comparison: {pairStats ? pairStats.r.toFixed(2) : '—'}.
              </p>
            </>
          )}
        </section>
      </div>

    </div>
  );
}
