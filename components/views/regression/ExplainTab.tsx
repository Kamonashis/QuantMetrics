'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { ChevronDown } from 'lucide-react';
import {
  TRADING_DAYS,
  describeAlpha,
  describeExposure,
  pct,
  type Attribution,
  type FactorFit,
  type FactorSample,
} from '@/lib/factor-regression';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { EvidenceChip, IntervalBar, SectionHeader, StatTile, fmtNum, fmtP } from './lab-ui';

const COMPONENT_COLORS = [QUANT_COLORS.gold, QUANT_COLORS.sky, QUANT_COLORS.violet, QUANT_COLORS.amber, QUANT_COLORS.cyan, QUANT_COLORS.positive, '#F472B6', '#94A3B8'];

function colorFor(i: number): string {
  return COMPONENT_COLORS[i % COMPONENT_COLORS.length];
}

export function ExplainTab({
  sample,
  fit,
  attribution,
  modelLabel,
  benchmark,
}: {
  sample: FactorSample;
  fit: FactorFit;
  attribution: Attribution;
  modelLabel: string;
  benchmark: string;
}) {
  const asset = sample.asset;
  const multi = sample.columns.length > 1;
  const slopes = fit.coefficients.slice(1);
  const alpha = fit.coefficients[0];
  const market = slopes[0];

  const topDriver = useMemo(
    () => [...fit.varianceShares].sort((a, b) => b.share - a.share)[0],
    [fit.varianceShares]
  );
  const idioVol = fit.sigma * Math.sqrt(TRADING_DAYS);

  const domainLo = Math.min(0, ...slopes.map((c) => c.ciLow));
  const domainHi = Math.max(0, ...slopes.map((c) => c.ciHigh));
  const domainPad = (domainHi - domainLo) * 0.08 || 0.1;
  const domain: [number, number] = [domainLo - domainPad, domainHi + domainPad];

  return (
    <div className="space-y-6">
      {/* The answer */}
      <section className="panel p-5">
        <p className="kicker mb-2">The short answer</p>
        <p className="font-display text-lg leading-snug text-zinc-900 dark:text-zinc-100 sm:text-xl">
          The <span className="text-gold-600 dark:text-gold-300">{modelLabel}</span> model explains{' '}
          <strong className="tabular-nums">{(fit.r2 * 100).toFixed(0)}%</strong> of {asset}&apos;s day-to-day return
          variation. The rest, <strong className="tabular-nums">{(fit.residualShare * 100).toFixed(0)}%</strong>, is
          specific to {asset}.
          {topDriver && topDriver.share > 0 && (
            <>
              {' '}
              The biggest driver is <strong>{topDriver.label}</strong>, at{' '}
              <span className="tabular-nums">{(topDriver.share * 100).toFixed(0)}%</span> of the variance.
            </>
          )}
        </p>
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          Fitted on {fit.n.toLocaleString()} daily returns from {sample.audit.start} to {sample.audit.end}. This describes how{' '}
          {asset} moved with each factor in that period. It is not a forecast.
        </p>

        <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile
            label={`Market beta vs ${benchmark}`}
            value={fmtNum(market.estimate)}
            tone="gold"
            sub={`95% range ${fmtNum(market.ciLow)} to ${fmtNum(market.ciHigh)}`}
          />
          <StatTile label="Share explained (R²)" value={`${(fit.r2 * 100).toFixed(1)}%`} sub={`Adjusted ${(fit.adjR2 * 100).toFixed(1)}%`} />
          <StatTile
            label="Alpha, annualized"
            value={pct(alpha.estimate * TRADING_DAYS, 1)}
            tone={alpha.evidence === 'none' || alpha.evidence === 'weak' ? 'default' : alpha.estimate >= 0 ? 'positive' : 'negative'}
            sub={<EvidenceChip evidence={alpha.evidence} p={alpha.p} />}
          />
          <StatTile label="Stock-specific volatility" value={`${(idioVol * 100).toFixed(1)}%`} sub="Annualized residual σ" />
        </div>
      </section>

      {/* Exposures */}
      <section className="panel p-5">
        <SectionHeader
          kicker="Exposures"
          title={`How ${asset} responded to each factor`}
          description="Each dot is the estimated sensitivity and each bar is its 95% range. A grey bar crosses zero, so the data cannot rule out no effect. Ranges use Newey–West standard errors, which allow for volatility clustering and autocorrelation."
        />
        <div className="space-y-4">
          {slopes.map((row, i) => {
            const col = sample.columns[i];
            return (
              <div key={row.key} className="grid gap-2 md:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] md:items-center md:gap-6">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-sm text-zinc-900 dark:text-zinc-100">{col.label}</span>
                    {col.kind === 'style' && (
                      <span className="font-mono text-[10px] text-zinc-400">ETF proxy · {col.short}</span>
                    )}
                    <EvidenceChip evidence={row.evidence} p={row.p} />
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-zinc-600 dark:text-zinc-300">
                    {describeExposure(asset, col, row, multi)}
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  <IntervalBar estimate={row.estimate} low={row.ciLow} high={row.ciHigh} domain={domain} />
                  <span className="w-14 shrink-0 text-right font-mono text-sm font-bold tabular-nums text-zinc-900 dark:text-zinc-100">
                    {row.estimate >= 0 ? '+' : ''}
                    {row.estimate.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
          <div className="border-t border-[var(--hairline)] pt-3 text-xs text-zinc-600 dark:text-zinc-300">
            <span className="mr-2 font-semibold text-zinc-900 dark:text-zinc-100">Alpha</span>
            <EvidenceChip evidence={alpha.evidence} p={alpha.p} />
            <p className="mt-1">{describeAlpha(asset, alpha)}</p>
          </div>
        </div>
        {slopes.length > 3 && (
          <p className="mt-4 text-[11px] text-zinc-500 dark:text-zinc-400">
            With {slopes.length} factors tested together, expect roughly one in twenty to look &quot;moderate&quot; by chance. Weigh single
            moderate results lightly.
          </p>
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <RiskShare fit={fit} asset={asset} />
        <ReturnShare attribution={attribution} asset={asset} />
      </div>

      <AttributionChart attribution={attribution} asset={asset} />

      <StatisticalDetail fit={fit} />
    </div>
  );
}

function RiskShare({ fit, asset }: { fit: FactorFit; asset: string }) {
  const items = [
    ...fit.varianceShares.map((s, i) => ({ ...s, color: colorFor(i) })),
    { key: 'RESID', label: `Specific to ${asset}`, share: fit.residualShare, color: '#52525B' },
  ];
  const positive = items.filter((s) => s.share > 0);
  const total = positive.reduce((a, s) => a + s.share, 0) || 1;
  return (
    <section className="panel p-5">
      <SectionHeader
        kicker="Risk"
        title="What drove the day-to-day swings"
        description={`Share of ${asset}'s return variance attributed to each factor (β × covariance). Shares add up to 100%.`}
      />
      <div className="flex h-4 w-full overflow-hidden rounded-full bg-[var(--bg-inset)]" role="img" aria-label="Variance share bar">
        {positive.map((s) => (
          <div key={s.key} style={{ width: `${(s.share / total) * 100}%`, backgroundColor: s.color }} title={`${s.label}: ${(s.share * 100).toFixed(1)}%`} />
        ))}
      </div>
      <ul className="mt-4 space-y-1.5 text-xs">
        {items.map((s) => (
          <li key={s.key} className="flex items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2 text-zinc-700 dark:text-zinc-300">
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ backgroundColor: s.color }} />
              <span className="truncate">{s.label}</span>
            </span>
            <span className="font-mono tabular-nums text-zinc-900 dark:text-zinc-100">{(s.share * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      {fit.varianceShares.some((s) => s.share < 0) && (
        <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">
          A negative share means that factor offset some of the swings driven by the others.
        </p>
      )}
    </section>
  );
}

function ReturnShare({ attribution, asset }: { attribution: Attribution; asset: string }) {
  const rows = attribution.components.filter((c) => c.key !== 'RESID');
  const resid = attribution.components.find((c) => c.key === 'RESID');
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.annualized)), Math.abs(attribution.actual.annualized), 1e-9);
  return (
    <section className="panel p-5">
      <SectionHeader
        kicker="Return"
        title="Where the average return came from"
        description={`${asset}'s average daily return, annualized and split by source. Daily returns are summed, not compounded.`}
      />
      <ul className="space-y-2 text-xs">
        {rows.map((r, i) => (
          <li key={r.key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_4.5rem] items-center gap-3">
            <span className="flex min-w-0 items-center gap-2 text-zinc-700 dark:text-zinc-300">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-sm"
                style={{ backgroundColor: r.key === 'ALPHA' ? '#E4E4E7' : colorFor(i) }}
              />
              <span className="truncate">{r.label}</span>
            </span>
            <div className="relative h-2 rounded-full bg-[var(--bg-inset)]">
              <div className="absolute inset-y-0 left-1/2 w-px bg-zinc-400/50" />
              <div
                className={`absolute inset-y-0 rounded-full ${r.annualized >= 0 ? 'bg-[var(--positive)]/70' : 'bg-[var(--negative)]/70'}`}
                style={
                  r.annualized >= 0
                    ? { left: '50%', width: `${(Math.abs(r.annualized) / maxAbs) * 50}%` }
                    : { right: '50%', width: `${(Math.abs(r.annualized) / maxAbs) * 50}%` }
                }
              />
            </div>
            <span className="text-right font-mono tabular-nums text-zinc-900 dark:text-zinc-100">{pct(r.annualized, 1)}</span>
          </li>
        ))}
        <li className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_4.5rem] items-center gap-3 border-t border-[var(--hairline)] pt-2 font-semibold">
          <span className="text-zinc-900 dark:text-zinc-100">Actual average</span>
          <span />
          <span className="text-right font-mono tabular-nums text-zinc-900 dark:text-zinc-100">{pct(attribution.actual.annualized, 1)}</span>
        </li>
      </ul>
      {resid && (
        <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">
          Daily noise averages to zero in an OLS fit with an intercept, so the rows add up exactly to the actual average.
        </p>
      )}
    </section>
  );
}

function AttributionChart({ attribution, asset }: { attribution: Attribution; asset: string }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    const x = attribution.dates;
    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: `${asset} actual`,
        x,
        y: attribution.actual.cumulative.map((v) => v * 100),
        line: { color: isDark ? '#F4F4F5' : '#18181B', width: 2 },
        hovertemplate: '%{y:+.1f}%<extra>Actual</extra>',
      },
    ];
    attribution.components
      .filter((c) => c.key !== 'RESID')
      .forEach((c, i) => {
        data.push({
          type: 'scatter',
          mode: 'lines',
          name: c.label,
          x,
          y: c.cumulative.map((v) => v * 100),
          line: { color: c.key === 'ALPHA' ? QUANT_COLORS.slate : colorFor(i), width: 1.4, dash: c.key === 'ALPHA' ? 'dot' : 'solid' },
          hovertemplate: `%{y:+.1f}%<extra>${c.label}</extra>`,
        });
      });
    return {
      traces: data,
      layout: {
        margin: { l: 52, r: 16, t: 10, b: 36 },
        yaxis: { ticksuffix: '%', zeroline: true, zerolinecolor: tokens.zeroLine },
        legend: { orientation: 'h', y: -0.18, font: { size: 10, color: tokens.textMuted } },
      },
    };
  }, [attribution, asset, isDark, tokens]);

  return (
    <section className="panel p-5">
      <SectionHeader
        kicker="Over time"
        title="Cumulative return, split by source"
        description="Each line adds up that source's daily contribution (β × factor return). The gap between the actual line and the factor lines is stock-specific noise."
      />
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        showLegend
        height={300}
        exportFilename={`${asset}-factor-attribution`}
        uirevisionKey={`attr-${asset}-${attribution.components.length}`}
      />
    </section>
  );
}

function StatisticalDetail({ fit }: { fit: FactorFit }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="panel p-5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-3 text-left"
      >
        <div>
          <p className="kicker mb-1">Statistical detail</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">Full coefficient table</h3>
        </div>
        <ChevronDown className={`h-4 w-4 text-zinc-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--hairline)] text-zinc-500">
                <th className="py-2 pr-3 text-left font-semibold">Term</th>
                <th className="px-3 py-2 text-right font-semibold">Estimate</th>
                <th className="px-3 py-2 text-right font-semibold" title="Newey–West HAC standard error">SE (NW)</th>
                <th className="px-3 py-2 text-right font-semibold" title="Classical OLS standard error, for comparison">SE (OLS)</th>
                <th className="px-3 py-2 text-right font-semibold">t</th>
                <th className="px-3 py-2 text-right font-semibold">p</th>
                <th className="px-3 py-2 text-right font-semibold">95% range</th>
                <th className="px-3 py-2 text-right font-semibold" title="Variance inflation factor: overlap with the other regressors">VIF</th>
                <th className="py-2 pl-3 text-right font-semibold" title="Huber M-estimate: down-weights outlier days">Huber</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--hairline)]">
              {fit.coefficients.map((c) => (
                <tr key={c.key}>
                  <td className="py-2 pr-3 font-semibold text-zinc-900 dark:text-zinc-100">{c.label}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-zinc-900 dark:text-zinc-100">{c.estimate.toFixed(c.key === 'ALPHA' ? 5 : 3)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-zinc-600 dark:text-zinc-400">{c.se.toFixed(c.key === 'ALPHA' ? 5 : 3)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-zinc-400">{c.seClassical.toFixed(c.key === 'ALPHA' ? 5 : 3)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{c.t.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtP(c.p)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-zinc-600 dark:text-zinc-400">
                    [{c.ciLow.toFixed(c.key === 'ALPHA' ? 5 : 2)}, {c.ciHigh.toFixed(c.key === 'ALPHA' ? 5 : 2)}]
                  </td>
                  <td className={`px-3 py-2 text-right tabular-nums ${c.vif !== null && c.vif > 5 ? 'text-[var(--negative)] font-semibold' : ''}`}>
                    {c.vif !== null ? c.vif.toFixed(2) : '—'}
                  </td>
                  <td className="py-2 pl-3 text-right tabular-nums text-zinc-600 dark:text-zinc-400">
                    {c.robustEstimate !== null ? c.robustEstimate.toFixed(c.key === 'ALPHA' ? 5 : 3) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 grid gap-x-6 gap-y-1 text-[11px] text-zinc-500 dark:text-zinc-400 sm:grid-cols-2">
            <span>
              n = {fit.n.toLocaleString()} · df = {fit.df} · residual σ = {(fit.sigma * 100).toFixed(3)}%/day
            </span>
            <span>
              Newey–West, Bartlett kernel, {fit.hacLags} lags · t({fit.df}) critical values
            </span>
            {fit.waldAll && (
              <span>
                Joint test, all slopes = 0 (HAC Wald): χ²({fit.waldAll.df}) = {fit.waldAll.stat.toFixed(1)}, p = {fmtP(fit.waldAll.p)}
              </span>
            )}
            {fit.fStat !== null && fit.fP !== null && (
              <span>
                Classical F({fit.p - 1}, {fit.df}) = {fit.fStat.toFixed(1)}, p = {fmtP(fit.fP)}
              </span>
            )}
            <span>
              AIC {fit.aic.toFixed(0)} · BIC {fit.bic.toFixed(0)}
            </span>
          </div>
        </div>
      )}
      {!open && (
        <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
          Estimates, both standard errors, t, p, 95% range, overlap (VIF), and a Huber outlier-resistant estimate for every term.
        </p>
      )}
    </section>
  );
}
