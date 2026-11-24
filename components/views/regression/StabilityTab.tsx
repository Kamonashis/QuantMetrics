'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  columnCorrelations,
  compareWithMarketOnly,
  influentialDates,
  outlierSensitiveCoefficients,
  pct,
  rollingFactorFit,
  type FactorFit,
  type FactorSample,
} from '@/lib/factor-regression';
import { standardNormalQuantile } from '@/lib/quant-engine';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { Callout, SectionHeader, SegmentedControl, fmtNum, pText } from '@/components/ui/lab-ui';

type CheckStatus = 'pass' | 'note' | 'flag';

const STATUS_STYLE: Record<CheckStatus, { dot: string; label: string }> = {
  pass: { dot: 'bg-[var(--positive)]', label: 'OK' },
  note: { dot: 'bg-amber-500', label: 'Note' },
  flag: { dot: 'bg-[var(--negative)]', label: 'Check' },
};

function CheckRow({ status, title, detail, stat }: { status: CheckStatus; title: string; detail: string; stat: string }) {
  const s = STATUS_STYLE[status];
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)] gap-3 py-2.5">
      <span className={`mt-1.5 h-2 w-2 rounded-full ${s.dot}`} aria-label={s.label} />
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3">
          <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{title}</span>
          <span className="font-mono text-[10px] tabular-nums text-zinc-500">{stat}</span>
        </div>
        <p className="mt-0.5 text-xs leading-relaxed text-zinc-600 dark:text-zinc-400">{detail}</p>
      </div>
    </li>
  );
}

export function StabilityTab({ sample, fit }: { sample: FactorSample; fit: FactorFit }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [windowSize, setWindowSize] = useState<63 | 126 | 252>(126);
  const [coefKey, setCoefKey] = useState<string>('MKT');
  const asset = sample.asset;

  const maxWindow = sample.y.length;
  const effectiveWindow = windowSize <= maxWindow ? windowSize : ([126, 63] as const).find((w) => w <= maxWindow) ?? 63;
  const rolling = useMemo(() => rollingFactorFit(sample, effectiveWindow), [sample, effectiveWindow]);
  const comparison = useMemo(() => compareWithMarketOnly(sample, fit), [sample, fit]);
  const correlations = useMemo(() => columnCorrelations(sample.columns), [sample.columns]);
  const influential = useMemo(() => influentialDates(sample, fit, 6), [sample, fit]);
  const sensitive = useMemo(() => outlierSensitiveCoefficients(fit), [fit]);

  const activeKey = rolling?.series.some((s) => s.key === coefKey) ? coefKey : 'MKT';
  const activeSeries = rolling?.series.find((s) => s.key === activeKey) ?? null;
  const fullSample = fit.coefficients.find((c) => c.key === activeKey);

  const rollingChart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!rolling || !activeSeries) return { traces: [], layout: {} };
    const x = rolling.dates;
    const scale = activeKey === 'ALPHA' ? 252 * 100 : 1;
    const sc = (arr: (number | null)[]) => arr.map((v) => (v === null ? null : v * scale));
    const data: Data[] = [
      { type: 'scatter', mode: 'lines', x, y: sc(activeSeries.high), line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
      {
        type: 'scatter',
        mode: 'lines',
        x,
        y: sc(activeSeries.low),
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(201, 164, 92, 0.18)',
        name: '95% range',
        hoverinfo: 'skip',
      },
      {
        type: 'scatter',
        mode: 'lines',
        x,
        y: sc(activeSeries.estimate),
        line: { color: QUANT_COLORS.gold, width: 2 },
        name: `${effectiveWindow}-day estimate`,
        hovertemplate: activeKey === 'ALPHA' ? '%{y:+.1f}%/yr<extra></extra>' : '%{y:.2f}<extra></extra>',
      },
    ];
    const shapes: Layout['shapes'] = [];
    if (fullSample) {
      shapes.push({
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        y0: fullSample.estimate * scale,
        y1: fullSample.estimate * scale,
        line: { color: tokens.textMuted, width: 1, dash: 'dash' },
      });
    }
    shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: { color: tokens.zeroLine, width: 1 } });
    return {
      traces: data,
      layout: {
        margin: { l: 52, r: 16, t: 10, b: 36 },
        shapes,
        yaxis: { ticksuffix: activeKey === 'ALPHA' ? '%' : '' },
      },
    };
  }, [rolling, activeSeries, activeKey, fullSample, tokens, effectiveWindow]);

  const r2Chart = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!rolling) return { traces: [], layout: {} };
    return {
      traces: [
        {
          type: 'scatter',
          mode: 'lines',
          x: rolling.dates,
          y: rolling.r2.map((v) => (v === null ? null : v * 100)),
          line: { color: QUANT_COLORS.slate, width: 1.5 },
          fill: 'tozeroy',
          fillcolor: 'rgba(140, 140, 154, 0.12)',
          hovertemplate: 'R² %{y:.0f}%<extra></extra>',
        },
      ],
      layout: { margin: { l: 52, r: 16, t: 6, b: 30 }, yaxis: { ticksuffix: '%', range: [0, 100] } },
    };
  }, [rolling]);

  const residualCharts = useMemo(() => {
    const z = fit.residuals.map((e) => e / (fit.sigma || 1));
    const sorted = [...z].sort((a, b) => a - b);
    const n = sorted.length;
    const theo = sorted.map((_, i) => standardNormalQuantile((i + 1 - 0.375) / (n + 0.25)));
    const bound = Math.max(4, Math.abs(sorted[0] ?? 0), Math.abs(sorted[n - 1] ?? 0));
    const qq: { traces: Data[]; layout: Partial<Layout> } = {
      traces: [
        { type: 'scatter', mode: 'lines', x: [-bound, bound], y: [-bound, bound], line: { color: QUANT_COLORS.goldLight, width: 1, dash: 'dash' }, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'markers', x: theo, y: sorted, marker: { color: QUANT_COLORS.slate, size: 4, opacity: 0.7 }, hovertemplate: 'normal %{x:.2f} · actual %{y:.2f}<extra></extra>' },
      ],
      layout: {
        margin: { l: 44, r: 12, t: 6, b: 40 },
        xaxis: { title: { text: 'Normal quantile', font: { size: 10, color: tokens.textMuted } } },
        yaxis: { title: { text: 'Residual (σ units)', font: { size: 10, color: tokens.textMuted } } },
      },
    };
    const ts: { traces: Data[]; layout: Partial<Layout> } = {
      traces: [
        {
          type: 'bar',
          x: sample.dates,
          y: fit.residuals.map((e) => e * 100),
          marker: { color: fit.residuals.map((e) => (e >= 0 ? QUANT_COLORS.positive : QUANT_COLORS.negative)) },
          hovertemplate: '%{x}: %{y:+.2f}%<extra>Residual</extra>',
        },
      ],
      layout: { margin: { l: 44, r: 12, t: 6, b: 30 }, yaxis: { ticksuffix: '%' }, bargap: 0 },
    };
    return { qq, ts };
  }, [fit, sample.dates, tokens]);

  const checks = useMemo(() => {
    const out: { status: CheckStatus; title: string; detail: string; stat: string }[] = [];
    const lb = fit.ljungBox;
    out.push({
      status: lb.p < 0.05 ? 'note' : 'pass',
      title: 'Leftover moves repeat day to day',
      detail:
        lb.p < 0.05
          ? 'Residuals are autocorrelated. Newey–West ranges already allow for this, but a lagged factor may be missing.'
          : 'No meaningful autocorrelation is left in the residuals.',
      stat: `Ljung–Box Q(${lb.lags}) = ${lb.q.toFixed(1)}, ${pText(lb.p)} · DW ${fit.durbinWatson.toFixed(2)}`,
    });
    const jb = fit.jarqueBera;
    out.push({
      status: jb.p < 0.01 ? 'note' : 'pass',
      title: 'Extreme days happen more than a bell curve predicts',
      detail:
        jb.p < 0.01
          ? `Residuals have fat tails (excess kurtosis ${jb.excessKurtosis.toFixed(1)}). This is normal for daily stock returns. Estimates are still valid, but single-day tail risk is larger than σ suggests.`
          : 'Residual tails look close to normal.',
      stat: `Jarque–Bera = ${jb.stat.toFixed(0)}, ${pText(jb.p)} · skew ${jb.skew.toFixed(2)}`,
    });
    out.push({
      status: fit.maxVif !== null && fit.maxVif > 5 ? 'flag' : fit.maxVif !== null && fit.maxVif > 2.5 ? 'note' : 'pass',
      title: 'Factors overlap with each other',
      detail:
        fit.maxVif === null
          ? 'Only one factor, so there is nothing to overlap.'
          : fit.maxVif > 5
            ? 'At least one factor is largely explained by the others. Its individual exposure is unreliable even if the overall fit is fine.'
            : fit.maxVif > 2.5
              ? 'Some overlap is present. Individual exposures are less precise, as the wider ranges show.'
              : 'Factors are distinct enough to estimate separately.',
      stat: fit.maxVif !== null ? `max VIF ${fit.maxVif.toFixed(2)}` : '—',
    });
    out.push({
      status: sensitive.length > 0 ? 'flag' : 'pass',
      title: 'Results hinge on a few extreme days',
      detail:
        sensitive.length > 0
          ? `The outlier-resistant (Huber) estimate differs by more than one standard error for: ${sensitive.map((c) => c.label).join(', ')}. Check the influential dates below.`
          : 'Down-weighting outlier days leaves every estimate within one standard error.',
      stat: `${sensitive.length} sensitive term${sensitive.length === 1 ? '' : 's'}`,
    });
    if (rolling && activeSeries) {
      const vals = activeSeries.estimate.filter((v): v is number => v !== null);
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      out.push({
        status: 'pass',
        title: `How much the ${activeSeries.label} exposure drifted`,
        detail: `Across rolling ${effectiveWindow}-day windows the estimate ranged from ${activeKey === 'ALPHA' ? pct(lo * 252, 1) : fmtNum(lo)} to ${activeKey === 'ALPHA' ? pct(hi * 252, 1) : fmtNum(hi)}. A full-sample number hides this variation.`,
        stat: `${vals.length} windows`,
      });
    }
    return out;
  }, [fit, sensitive, rolling, activeSeries, activeKey, effectiveWindow]);

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <SectionHeader
          kicker="Through time"
          title={`Did ${asset}'s exposures stay put?`}
          description="The same model re-fitted on a rolling window. The shaded band is the 95% range for each window, and the dashed line is the full-sample estimate."
          right={
            <SegmentedControl
              ariaLabel="Rolling window"
              value={effectiveWindow}
              onChange={(v) => setWindowSize(v as 63 | 126 | 252)}
              options={[
                { value: 63, label: '3M' },
                { value: 126, label: '6M' },
                { value: 252, label: '1Y' },
              ]}
            />
          }
        />
        {rolling ? (
          <>
            <div className="mb-3 flex flex-wrap gap-1.5">
              {rolling.series.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => setCoefKey(s.key)}
                  aria-pressed={s.key === activeKey}
                  className={`cursor-pointer rounded-md border px-2 py-0.5 font-mono text-[11px] transition-colors ${
                    s.key === activeKey
                      ? 'border-gold-500/50 bg-gold-500/15 text-gold-800 dark:text-gold-200'
                      : 'border-[var(--hairline)] text-zinc-500 hover:border-gold-500/40'
                  }`}
                >
                  {s.key === 'ALPHA' ? 'Alpha (%/yr)' : s.label}
                </button>
              ))}
            </div>
            <PlotlyChart
              data={rollingChart.traces}
              layout={rollingChart.layout}
              plotType="2d"
              hovermode="x unified"
              showLegend={false}
              height={280}
              exportFilename={`${asset}-rolling-${activeKey}`}
              uirevisionKey={`roll-${asset}-${activeKey}-${effectiveWindow}`}
            />
            <p className="mb-1 mt-4 text-[11px] font-semibold text-zinc-500 dark:text-zinc-400">Rolling share explained (R²)</p>
            <PlotlyChart
              data={r2Chart.traces}
              layout={r2Chart.layout}
              plotType="2d"
              hovermode="x unified"
              showLegend={false}
              height={140}
              exportFilename={`${asset}-rolling-r2`}
              uirevisionKey={`roll-r2-${asset}-${effectiveWindow}`}
            />
            {rolling.step > 1 && (
              <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                Re-fitted every {rolling.step} sessions to keep the chart responsive.
              </p>
            )}
          </>
        ) : (
          <Callout tone="warn">Not enough observations for a rolling window with this many factors.</Callout>
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="Model check"
            title="Do the extra factors earn their place?"
            description="Compares the selected model with market-only on exactly the same days."
          />
          {comparison ? (
            <>
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-[var(--hairline)] text-zinc-500">
                    <th className="py-2 text-left font-semibold">Model</th>
                    <th className="py-2 text-right font-semibold">Factors</th>
                    <th className="py-2 text-right font-semibold">R²</th>
                    <th className="py-2 text-right font-semibold">Adj. R²</th>
                    <th className="py-2 text-right font-semibold" title="Bayesian information criterion: lower is better">BIC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--hairline)] tabular-nums">
                  {[comparison.restricted, comparison.full].map((m) => (
                    <tr key={m.label}>
                      <td className="py-2 font-semibold text-zinc-900 dark:text-zinc-100">{m.label}</td>
                      <td className="py-2 text-right">{m.k}</td>
                      <td className="py-2 text-right">{(m.r2 * 100).toFixed(1)}%</td>
                      <td className="py-2 text-right">{(m.adjR2 * 100).toFixed(1)}%</td>
                      <td className="py-2 text-right">{m.bic.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4">
                <Callout tone={comparison.verdict === 'adds' ? 'info' : 'warn'} title={
                  comparison.verdict === 'adds'
                    ? 'Yes. The extra factors explain real variation.'
                    : comparison.verdict === 'marginal'
                      ? 'Marginally. They pass a significance test, but BIC is not convinced.'
                      : 'No. Market-only explains about as much.'
                }>
                  Joint Newey–West test that the {comparison.addedKeys.length} added exposures are all zero:{' '}
                  {comparison.wald ? (
                    <span className="font-mono">
                      χ²({comparison.wald.df}) = {comparison.wald.stat.toFixed(1)}, {pText(comparison.wald.p)}
                    </span>
                  ) : (
                    'not computable'
                  )}
                  . Adjusted R² changes by {((comparison.full.adjR2 - comparison.restricted.adjR2) * 100).toFixed(1)} pts.
                </Callout>
              </div>
            </>
          ) : (
            <Callout>Add a style factor or custom regressor to compare against market-only.</Callout>
          )}
        </section>

        <section className="panel p-5">
          <SectionHeader
            kicker="Overlap"
            title="How similar are the factors?"
            description="Correlation between factor returns. Values near ±1 mean the model cannot tell those factors apart."
          />
          {sample.columns.length > 1 ? (
            <div className="overflow-x-auto">
              <table className="text-[11px] font-mono">
                <thead>
                  <tr>
                    <th />
                    {sample.columns.map((c) => (
                      <th key={c.key} className="px-1.5 pb-1.5 text-right font-semibold text-zinc-500">{c.short}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sample.columns.map((row, i) => (
                    <tr key={row.key}>
                      <td className="pr-2 py-1 font-semibold text-zinc-600 dark:text-zinc-300">{row.short}</td>
                      {sample.columns.map((col, j) => {
                        const v = correlations[i][j];
                        const a = Math.min(1, Math.abs(v));
                        return (
                          <td
                            key={col.key}
                            className="px-1.5 py-1 text-right tabular-nums"
                            style={{
                              backgroundColor: i === j ? 'transparent' : v >= 0 ? `rgba(201,164,92,${a * 0.55})` : `rgba(90,169,230,${a * 0.55})`,
                            }}
                          >
                            {i === j ? '—' : v.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Callout>One factor only: no overlap to measure.</Callout>
          )}
        </section>
      </div>

      <section className="panel p-5">
        <SectionHeader kicker="Diagnostics" title="Can these results be trusted?" description="Plain-language checks on what the model left unexplained." />
        <ul className="divide-y divide-[var(--hairline)]">
          {checks.map((c) => (
            <CheckRow key={c.title} {...c} />
          ))}
        </ul>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="Influence"
            title="Days that moved the estimates most"
            description="Ranked by Cook's distance: how much the fitted model would change if that day were removed."
          />
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--hairline)] text-zinc-500">
                <th className="py-2 text-left font-semibold">Date</th>
                <th className="py-2 text-right font-semibold">{asset}</th>
                <th className="py-2 text-right font-semibold">Model</th>
                <th className="py-2 text-right font-semibold">Miss</th>
                <th className="py-2 text-right font-semibold">Cook&apos;s D</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--hairline)] tabular-nums">
              {influential.map((d) => (
                <tr key={d.date}>
                  <td className="py-2 text-zinc-900 dark:text-zinc-100">{d.date}</td>
                  <td className="py-2 text-right">{pct(d.actual)}</td>
                  <td className="py-2 text-right text-zinc-500">{pct(d.fitted)}</td>
                  <td className={`py-2 text-right ${d.residual >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>{pct(d.residual)}</td>
                  <td className={`py-2 text-right ${d.cooks > 4 / fit.n ? 'font-semibold text-amber-600 dark:text-amber-400' : ''}`}>{d.cooks.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            Highlighted values exceed the 4/n rule of thumb ({(4 / fit.n).toFixed(4)}). Earnings dates and market shocks usually top this list.
          </p>
        </section>

        <section className="panel p-5">
          <SectionHeader kicker="Residuals" title="What the model missed" description="Daily misses over time, and their distribution against a normal curve." />
          <PlotlyChart
            data={residualCharts.ts.traces}
            layout={residualCharts.ts.layout}
            plotType="2d"
            hovermode="closest"
            showLegend={false}
            height={150}
            exportFilename={`${asset}-residuals`}
            uirevisionKey={`resid-${asset}-${fit.n}`}
          />
          <PlotlyChart
            data={residualCharts.qq.traces}
            layout={residualCharts.qq.layout}
            plotType="2d"
            hovermode="closest"
            showLegend={false}
            height={200}
            exportFilename={`${asset}-residual-qq`}
            uirevisionKey={`qq-${asset}-${fit.n}`}
          />
          <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
            Points bending away from the dashed line at the ends mean fat tails.
          </p>
        </section>
      </div>
    </div>
  );
}
