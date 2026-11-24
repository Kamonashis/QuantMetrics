'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  averageCorrelation,
  leadLag,
  minimumSpanningTree,
  principalFactors,
  residualCorrelation,
  type CorrelationSample,
  type PrincipalFactor,
} from '@/lib/correlation-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout, SectionHeader, fmtP } from '@/components/ui/lab-ui';
import { DataTable, MatrixHeatmap, PairPicker, fmtCorr, td, tdClass, thClass, type Pair } from './shared';

const FACTOR_COLORS = [QUANT_COLORS.gold, QUANT_COLORS.slate, '#D4D4D8'];
const NONE = '__common__';

function describeFactor(f: PrincipalFactor, tickers: readonly string[]): string {
  if (f.sameSign) return 'Every asset moves the same way: a shared, market-wide driver.';
  const strong = tickers.filter((_, i) => Math.abs(f.loadings[i]) > 0.3);
  const pos = strong.filter((t) => f.loadings[tickers.indexOf(t)] > 0);
  const neg = strong.filter((t) => f.loadings[tickers.indexOf(t)] < 0);
  const rest = tickers.filter((t) => !strong.includes(t));
  if (pos.length > 0 && neg.length > 0) return `Separates ${pos.join(', ')} from ${neg.join(', ')}.`;
  if (strong.length === 1) return `Mostly specific to ${strong[0]}.`;
  if (strong.length > 1) return `A shared driver of ${strong.join(', ')}${rest.length > 0 ? `; ${rest.join(', ')} barely take${rest.length === 1 ? 's' : ''} part` : ''}.`;
  return 'No clear grouping.';
}

export function StructureTab({
  sample,
  corr,
  pair,
  onPairChange,
}: {
  sample: CorrelationSample;
  corr: number[][];
  pair: Pair;
  onPairChange: (p: Pair) => void;
}) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [marketChoice, setMarketChoice] = useState<string>('SPY');
  const market = marketChoice !== NONE && sample.tickers.includes(marketChoice) && sample.tickers.length > 2 ? marketChoice : NONE;

  const pca = useMemo(() => principalFactors(corr, sample.dates.length, 3), [corr, sample.dates.length]);
  const residual = useMemo(() => residualCorrelation(sample, market === NONE ? null : sample.tickers.indexOf(market)), [sample, market]);
  const tree = useMemo(() => minimumSpanningTree(residual.matrix, residual.tickers), [residual]);

  const ia = sample.tickers.indexOf(pair[0]);
  const ib = sample.tickers.indexOf(pair[1]);
  const validPair = ia >= 0 && ib >= 0 && ia !== ib;
  const ll = useMemo(() => (validPair ? leadLag(sample.returns[ia], sample.returns[ib], 5) : null), [sample, ia, ib, validPair]);

  const factorChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(
    () => ({
      traces: pca.factors.map((f, k) => ({
        type: 'bar',
        name: `Factor ${f.index} (${(f.share * 100).toFixed(0)}%)`,
        x: [...sample.tickers],
        y: f.loadings,
        marker: { color: FACTOR_COLORS[k] },
        hovertemplate: `%{x}: %{y:+.2f}<extra>Factor ${f.index}</extra>`,
      })),
      layout: {
        barmode: 'group',
        margin: { l: 44, r: 12, t: 26, b: 34 },
        yaxis: { range: [-1, 1], zeroline: true, zerolinecolor: tokens.zeroLine, title: { text: 'Correlation with factor', font: { size: 10, color: tokens.textMuted } } },
      },
    }),
    [pca, sample.tickers, tokens]
  );

  const network = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    const names = residual.tickers;
    const n = names.length;
    const pos = names.map((_, i) => {
      const a = (2 * Math.PI * i) / n - Math.PI / 2;
      return { x: Math.cos(a), y: Math.sin(a) };
    });
    const traces: Data[] = tree.edges.map((e) => {
      const i = names.indexOf(e.a);
      const j = names.indexOf(e.b);
      return {
        type: 'scatter',
        mode: 'lines',
        x: [pos[i].x, pos[j].x],
        y: [pos[i].y, pos[j].y],
        line: { color: e.corr >= 0 ? QUANT_COLORS.gold : QUANT_COLORS.slate, width: 1 + 7 * Math.min(1, Math.abs(e.corr)) },
        hoverinfo: 'text',
        text: `${e.a} – ${e.b}: ${fmtCorr(e.corr)}`,
        showlegend: false,
      } as Data;
    });
    traces.push({
      type: 'scatter',
      mode: 'markers',
      x: pos.map((p) => p.x),
      y: pos.map((p) => p.y),
      marker: { size: tree.degree.map((d) => 12 + 4 * d), color: isDark ? '#101014' : '#FFFFFF', line: { color: QUANT_COLORS.gold, width: 2 } },
      hoverinfo: 'text',
      text: names.map((t, i) => `${t}: ${tree.degree[i]} link${tree.degree[i] === 1 ? '' : 's'}`),
      showlegend: false,
    } as Data);
    traces.push({
      type: 'scatter',
      mode: 'text',
      x: pos.map((p) => p.x * 1.26),
      y: pos.map((p) => p.y * 1.2),
      text: [...names],
      textfont: { size: 11, color: tokens.text },
      hoverinfo: 'skip',
      showlegend: false,
    } as Data);
    return {
      traces,
      layout: {
        margin: { l: 10, r: 10, t: 10, b: 10 },
        xaxis: { visible: false, range: [-1.7, 1.7], fixedrange: true },
        yaxis: { visible: false, range: [-1.45, 1.45], fixedrange: true },
      },
    };
  }, [residual, tree, tokens, isDark]);

  const lagChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!ll) return { traces: [], layout: {} };
    const side = ll.lags.filter((l) => l.lag !== 0);
    return {
      traces: [
        {
          type: 'bar',
          x: side.map((l) => l.lag),
          y: side.map((l) => l.corr),
          marker: { color: side.map((l) => (Math.abs(l.corr) > 1.96 * l.se ? QUANT_COLORS.gold : QUANT_COLORS.slate)) },
          error_y: { type: 'data', array: side.map((l) => 1.96 * l.se), color: tokens.textMuted, thickness: 1, width: 3 },
          hovertemplate: 'lag %{x}: %{y:+.3f}<extra></extra>',
        },
      ],
      layout: {
        margin: { l: 46, r: 12, t: 8, b: 44 },
        xaxis: { title: { text: `← ${pair[0]} earlier · days · ${pair[1]} earlier →`, font: { size: 10, color: tokens.textMuted } }, dtick: 1 },
        yaxis: { zeroline: true, zerolinecolor: tokens.zeroLine },
      },
    };
  }, [ll, pair, tokens]);

  const rawOthers = useMemo(() => {
    if (market === NONE) return averageCorrelation(corr);
    const keep = sample.tickers.map((_, i) => i).filter((i) => sample.tickers[i] !== market);
    return averageCorrelation(keep.map((i) => keep.map((j) => corr[i][j])));
  }, [corr, sample.tickers, market]);
  const residAvg = averageCorrelation(residual.matrix);
  const links = [...tree.edges].sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
  const signal = pca.factors.filter((f) => f.aboveNoise).length;

  const leaderText = !ll
    ? null
    : ll.leader === 'A'
      ? `${pair[0]}'s past moves help predict ${pair[1]}, and not the other way round.`
      : ll.leader === 'B'
        ? `${pair[1]}'s past moves help predict ${pair[0]}, and not the other way round.`
        : ll.leader === 'both'
          ? 'Each helps predict the other. That is feedback or a shared driver, not one leading the other.'
          : 'Neither helps predict the other once each asset’s own past is accounted for.';

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <SectionHeader
          kicker="Common factors"
          title="What drives the basket together?"
          description="The correlation matrix broken into independent drivers, largest first. Bars show how strongly each asset moves with each driver."
        />
        <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <PlotlyChart
            data={factorChart.traces}
            layout={factorChart.layout}
            plotType="categorical"
            hovermode="closest"
            showLegend
            enableWheelPan={false}
            height={280}
            exportFilename="correlation-factors"
            uirevisionKey={`factors-${sample.tickers.join('-')}`}
          />
          <ul className="space-y-3 text-xs">
            {pca.factors.map((f, k) => (
              <li key={f.index} className="flex gap-2.5">
                <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ backgroundColor: FACTOR_COLORS[k] }} />
                <div>
                  <p className="font-semibold text-zinc-900 dark:text-zinc-100">
                    Factor {f.index}: {(f.share * 100).toFixed(0)}% of variance{' '}
                    <span className={`font-mono text-[10px] font-normal ${f.aboveNoise ? 'text-[var(--positive)]' : 'text-zinc-400'}`}>
                      {f.aboveNoise ? 'real signal' : 'within noise'}
                    </span>
                  </p>
                  <p className="mt-0.5 text-zinc-600 dark:text-zinc-400">{describeFactor(f, sample.tickers)}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">
          {signal} of the first {pca.factors.length} factors stand out from what random data of this size would produce (eigenvalue above{' '}
          {pca.noiseEdge.toFixed(2)}, the Marchenko–Pastur edge). Treat the others as noise.
        </p>
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Beneath the market"
          title="What is left after removing the common driver?"
          description="Most stocks are correlated simply because they all follow the market. Removing it shows which assets are linked for other reasons."
          right={
            <label className="flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
              Remove
              <select
                value={market}
                onChange={(e) => setMarketChoice(e.target.value)}
                className="cursor-pointer rounded-lg border border-zinc-200 bg-zinc-50 px-2 py-1 font-mono text-xs font-bold text-zinc-900 focus:outline-none focus:ring-1 focus:ring-gold-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              >
                <option value={NONE}>First common factor</option>
                {sample.tickers.length > 2 &&
                  sample.tickers.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
              </select>
            </label>
          }
        />
        {residual.tickers.length < 3 ? (
          <Callout>Add a third asset to see what links remain once the common driver is removed.</Callout>
        ) : (
        <>
        <p className="mb-4 text-sm text-zinc-700 dark:text-zinc-300">
          After removing {residual.removed}, average correlation among the {market === NONE ? '' : 'other '}assets falls from{' '}
          <strong className="font-mono">{rawOthers.toFixed(2)}</strong> to <strong className="font-mono">{residAvg.toFixed(2)}</strong>.
          {links[0] && (
            <>
              {' '}
              The strongest remaining link is {links[0].a} – {links[0].b} at {fmtCorr(links[0].corr)}.
            </>
          )}
        </p>
        <div className="grid gap-6 lg:grid-cols-2">
          <div>
            <p className="mb-1 text-[11px] font-semibold text-zinc-500 dark:text-zinc-400">Leftover correlation</p>
            <MatrixHeatmap tickers={residual.tickers} matrix={residual.matrix} exportName="residual-correlation" />
          </div>
          <div>
            <p className="mb-1 text-[11px] font-semibold text-zinc-500 dark:text-zinc-400">Strongest chain of links</p>
            <PlotlyChart
              data={network.traces}
              layout={network.layout}
              plotType="2d"
              hovermode="closest"
              showLegend={false}
              enableWheelPan={false}
              height={Math.max(220, Math.min(520, (residual.tickers.length - 1) * 46 + 70))}
              exportFilename="residual-network"
              uirevisionKey={`net-${residual.tickers.join('-')}-${market}`}
            />
            <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
              Each asset is joined to its closest relative so that all are connected with the fewest links (a minimum spanning tree). Thicker
              lines are stronger links.
            </p>
          </div>
        </div>
        </>
        )}
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Timing"
          title={`Does ${pair[0]} or ${pair[1]} move first?`}
          description="Bars show the correlation between one asset's return and the other's return a few days earlier, with 95% ranges that allow for volatility clustering."
          right={<PairPicker tickers={sample.tickers} pair={pair} onChange={onPairChange} />}
        />
        {ll ? (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <PlotlyChart
              data={lagChart.traces}
              layout={lagChart.layout}
              plotType="categorical"
              hovermode="closest"
              showLegend={false}
              enableWheelPan={false}
              height={240}
              exportFilename={`${pair[0]}-${pair[1]}-lead-lag`}
              uirevisionKey={`lag-${pair.join('-')}`}
            />
            <div className="space-y-3">
              <Callout tone={ll.leader === 'A' || ll.leader === 'B' ? 'warn' : 'info'} title={leaderText ?? undefined}>
                Tested on daily returns with five lags, controlling for each asset&apos;s own past. Same-day correlation is{' '}
                {fmtCorr(ll.lags.find((l) => l.lag === 0)?.corr)}.
              </Callout>
              <DataTable>
                <thead>
                  <tr>
                    <th className={thClass}>Direction</th>
                    <th className={thClass}>Test</th>
                    <th className={thClass}>p</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: `${pair[0]} → ${pair[1]}`, g: ll.aLeadsB },
                    { label: `${pair[1]} → ${pair[0]}`, g: ll.bLeadsA },
                  ].map((r) => (
                    <tr key={r.label}>
                      <td className={tdClass}>{r.label}</td>
                      <td className={tdClass}>{r.g ? `χ²(${r.g.df}) = ${r.g.wald.toFixed(1)}` : '—'}</td>
                      <td className={td(r.g && r.g.p < 0.025 ? 'text-amber-600 dark:text-amber-400' : null)}>{r.g ? fmtP(r.g.p) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </DataTable>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
                Daily lead–lag between liquid stocks is usually tiny and unstable. Treat a significant result as a prompt to look closer, not as
                a trading rule.
              </p>
            </div>
          </div>
        ) : (
          <Callout>Pick two different assets with enough history.</Callout>
        )}
      </section>
    </div>
  );
}
