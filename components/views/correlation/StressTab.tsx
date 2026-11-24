'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  basketExceedance,
  basketTailDependence,
  exceedanceCorrelations,
  worstBasketDays,
  type CorrelationSample,
} from '@/lib/correlation-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout, SectionHeader, SegmentedControl, StatTile } from '@/components/ui/lab-ui';
import { DataTable, PairPicker, fmtCorr, td, tdClass, thClass, type Pair } from './shared';

export function StressTab({
  sample,
  pair,
  onPairChange,
  weights,
}: {
  sample: CorrelationSample;
  pair: Pair;
  onPairChange: (p: Pair) => void;
  weights: number[];
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [tailQ, setTailQ] = useState<0.05 | 0.1>(0.05);

  const basket = useMemo(() => basketExceedance(sample, 0.2), [sample]);
  const tails = useMemo(() => basketTailDependence(sample, tailQ), [sample, tailQ]);
  const worst = useMemo(() => worstBasketDays(sample, weights, 10), [sample, weights]);

  const ia = sample.tickers.indexOf(pair[0]);
  const ib = sample.tickers.indexOf(pair[1]);
  const validPair = ia >= 0 && ib >= 0 && ia !== ib;
  const pairEx = useMemo(() => (validPair ? exceedanceCorrelations(sample.returns[ia], sample.returns[ib]) : null), [sample, ia, ib, validPair]);

  const chart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!pairEx) return { traces: [], layout: {} };
    const pts = pairEx.points;
    const x = [...pts.map((p) => `Worst ${Math.round(p.q * 100)}%`), ...[...pts].reverse().map((p) => `Best ${Math.round(p.q * 100)}%`)];
    const actual = [...pts.map((p) => p.lower), ...[...pts].reverse().map((p) => p.upper)];
    const normal = [...pts.map((p) => p.normal), ...[...pts].reverse().map((p) => p.normal)];
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'If returns were normal',
          x,
          y: normal,
          line: { color: tokens.textMuted, width: 1.5, dash: 'dash' },
          marker: { size: 5 },
          hovertemplate: '%{y:.2f}<extra>normal</extra>',
        },
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Observed',
          x,
          y: actual,
          connectgaps: false,
          line: { color: QUANT_COLORS.gold, width: 2 },
          marker: { size: 7 },
          hovertemplate: '%{y:.2f}<extra>observed</extra>',
        },
      ],
      layout: {
        margin: { l: 46, r: 14, t: 26, b: 44 },
        xaxis: { type: 'category', tickangle: 0 },
        yaxis: { title: { text: 'Correlation within the tail', font: { size: 10, color: tokens.textMuted } } },
      },
    };
  }, [pairEx, tokens]);

  const asym = basket.lowerAvg !== null && basket.upperAvg !== null ? basket.lowerAvg - basket.upperAvg : null;
  const excess = basket.lowerAvg !== null && basket.normalAvg !== null ? basket.lowerAvg - basket.normalAvg : null;
  const flagged = tails.filter((t) => t.lower - t.normal > 2 * t.se).length;

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <p className="kicker mb-2">The short answer</p>
        {basket.lowerAvg !== null && basket.upperAvg !== null && basket.normalAvg !== null && asym !== null && excess !== null ? (
          <>
            <p className="font-display text-lg leading-snug text-zinc-900 dark:text-zinc-100 sm:text-xl">
              On days when two assets both had a bottom-20% return, their correlation averaged{' '}
              <strong className="tabular-nums text-gold-600 dark:text-gold-300">{basket.lowerAvg.toFixed(2)}</strong>. If returns were normally
              distributed it would be <strong className="tabular-nums">{basket.normalAvg.toFixed(2)}</strong>.{' '}
              {excess > 0.1
                ? asym > 0.1
                  ? 'These assets cling together in sell-offs, and more so than in rallies.'
                  : 'These assets cling together in large moves in both directions, not only in sell-offs.'
                : 'Sell-offs look no more correlated than a normal distribution implies.'}
            </p>
            <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
              {excess > 0.1 && asym <= 0.1
                ? 'A similar rise in joint rallies points to volatility clustering (big days come together) more than to a sell-off-specific effect. Either way, diversification is weakest on the days with the largest moves.'
                : 'Conditioning on a tail lowers correlation even when nothing has changed, so the fair comparison is the normal benchmark, not the full-sample correlation.'}{' '}
              Averaged over {basket.pairsUsed} pairs.
            </p>
            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <StatTile label="Joint sell-offs (worst 20%)" value={basket.lowerAvg.toFixed(2)} tone="gold" sub="Observed correlation" />
              <StatTile label="Joint rallies (best 20%)" value={basket.upperAvg.toFixed(2)} sub="Observed correlation" />
              <StatTile label="Normal benchmark" value={basket.normalAvg.toFixed(2)} sub="Same overall correlations, no fat tails" />
            </div>
          </>
        ) : (
          <Callout tone="warn">Too few joint tail days to measure tail correlation. Widen the date range.</Callout>
        )}
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Pair"
          title={`${pair[0]} / ${pair[1]}: correlation inside the tails`}
          description="Left: days when both were among their worst. Right: days when both were among their best. The dashed line is what a normal distribution with the same overall correlation would show."
          right={<PairPicker tickers={sample.tickers} pair={pair} onChange={onPairChange} />}
        />
        {pairEx ? (
          <>
            <PlotlyChart
              data={chart.traces}
              layout={chart.layout}
              plotType="categorical"
              hovermode="x unified"
              showLegend
              enableWheelPan={false}
              height={280}
              exportFilename={`${pair[0]}-${pair[1]}-exceedance`}
              uirevisionKey={`exceed-${pair.join('-')}`}
            />
            <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
              Full-sample correlation {pairEx.rho.toFixed(2)}. Points with fewer than 20 joint days are left blank.
            </p>
          </>
        ) : (
          <Callout>Pick two different assets.</Callout>
        )}
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Joint crashes"
          title="When one has a very bad day, how often does the other?"
          description={`A "very bad day" is one of an asset's worst ${Math.round(tailQ * 100)}% of days. Compared with a normal model matched to each pair's rank correlation.`}
          right={
            <SegmentedControl
              ariaLabel="Tail size"
              value={tailQ}
              onChange={(v) => setTailQ(v)}
              options={[
                { value: 0.05 as const, label: 'Worst 5%' },
                { value: 0.1 as const, label: 'Worst 10%' },
              ]}
            />
          }
        />
        <DataTable>
          <thead>
            <tr>
              <th className={thClass}>Pair</th>
              <th className={thClass} title="Days on which both had a very bad day">Joint bad days</th>
              <th className={thClass} title="Expected count under the normal model">Expected if normal</th>
              <th className={thClass} title="Chance the second has a very bad day given the first does">Chance both</th>
              <th className={thClass}>If normal</th>
              <th className={thClass}>Excess</th>
            </tr>
          </thead>
          <tbody>
            {tails.map((t) => {
              const ex = t.lower - t.normal;
              const sig = ex > 2 * t.se;
              return (
                <tr key={`${t.a}-${t.b}`}>
                  <td className={tdClass}>
                    <button type="button" onClick={() => onPairChange([t.a, t.b])} className="cursor-pointer font-semibold text-zinc-900 underline-offset-2 hover:underline dark:text-zinc-100">
                      {t.a} / {t.b}
                    </button>
                  </td>
                  <td className={tdClass}>{t.jointLower}</td>
                  <td className={tdClass}>{t.expectedJointNormal.toFixed(1)}</td>
                  <td className={tdClass}>
                    {(t.lower * 100).toFixed(0)}% <span className="text-zinc-400">±{(t.se * 100).toFixed(0)}</span>
                  </td>
                  <td className={tdClass}>{(t.normal * 100).toFixed(0)}%</td>
                  <td className={td(sig && 'text-amber-600 dark:text-amber-400', 'font-semibold')}>
                    {ex >= 0 ? '+' : ''}
                    {(ex * 100).toFixed(0)} pts
                  </td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
        <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
          {flagged} of {tails.length} pairs crash together more often than the normal model allows by over two standard errors (highlighted). ±
          is one standard error.
        </p>
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Worst days"
          title="The basket's ten worst days"
          description="Using the weights from the Portfolio tab. Shows whether diversification helped on the days it mattered."
        />
        <DataTable>
          <thead>
            <tr>
              <th className={thClass}>Date</th>
              <th className={thClass}>Basket</th>
              {sample.tickers.map((t) => (
                <th key={t} className={thClass}>
                  {t}
                </th>
              ))}
              <th className={thClass}>Fell</th>
            </tr>
          </thead>
          <tbody>
            {worst.map((d) => (
              <tr key={d.date}>
                <td className={tdClass}>{d.date}</td>
                <td className={td('text-[var(--negative)]', 'font-semibold')}>{(d.basket * 100).toFixed(1)}%</td>
                {d.assets.map((v, i) => (
                  <td key={sample.tickers[i]} className={td(v >= 0 && 'text-[var(--positive)]')}>
                    {fmtCorr(v * 100, 1)}%
                  </td>
                ))}
                <td className={tdClass}>
                  {d.fell}/{sample.tickers.length}
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
        <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
          On {worst.filter((d) => d.fell === sample.tickers.length).length} of these {worst.length} days every asset fell.
        </p>
      </section>
    </div>
  );
}
