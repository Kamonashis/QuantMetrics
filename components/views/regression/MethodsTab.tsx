'use client';

import React from 'react';
import { MathDisplay } from '@/components/ui/MathDisplay';
import type { FactorFit, FactorSample } from '@/lib/factor-regression';
import { SectionHeader } from './lab-ui';

export function MethodsTab({
  sample,
  fit,
  sourcesByTicker,
}: {
  sample: FactorSample;
  fit: FactorFit | null;
  sourcesByTicker: Record<string, string>;
}) {
  const a = sample.audit;
  const terms = sample.columns.map((c, i) => `\\beta_{${i + 1}}\\,\\mathrm{${c.short.replace(/[^A-Za-z0-9]/g, '')}}_t`).join(' + ');
  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <SectionHeader kicker="Specification" title="Exactly what was fitted" />
        <div className="panel-inset overflow-x-auto px-4 py-3">
          <MathDisplay display="block" formula={`r_{${sample.asset},t} = \\alpha + ${terms} + \\varepsilon_t`} />
        </div>
        <dl className="mt-4 grid gap-x-8 gap-y-2 text-xs sm:grid-cols-2">
          {[
            ['Dependent', `${sample.asset} daily ${a.returnType} return`],
            ['Returns', 'Raw, not in excess of the risk-free rate. Alpha is therefore not Jensen’s alpha; at today’s T-bill yields the gap is roughly the risk-free rate × (1 − β).'],
            ['Sample', `${a.start} → ${a.end} · ${a.returnObs.toLocaleString()} returns on ${a.commonPriceDates.toLocaleString()} shared trading dates`],
            ['Alignment', `Returns are computed between consecutive dates on which every series traded. ${a.droppedAssetSessions} ${sample.asset} session(s) were dropped; ${a.multiSessionReturns} return(s) span a calendar gap.`],
            ['Estimator', 'Ordinary least squares via Gram–Schmidt QR. Exactly collinear inputs are refused, not regularized.'],
            ['Uncertainty', fit ? `Newey–West HAC, Bartlett kernel, ${fit.hacLags} lags (⌊4·(n/100)^(2/9)⌋), small-sample n/(n−k) scaling, t(${fit.df}) critical values` : '—'],
            ['Robustness', 'Huber M-estimator (c = 1.345, MAD scale) fitted alongside. Terms that shift by more than one SE are flagged.'],
            ['Attribution', 'Daily contribution = β × factor return. Variance share = β·cov(factor, r)/var(r), which sums to R².'],
          ].map(([k, v]) => (
            <div key={k} className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-3">
              <dt className="font-semibold text-zinc-500 dark:text-zinc-400">{k}</dt>
              <dd className="text-zinc-700 dark:text-zinc-300">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="panel p-5">
        <SectionHeader kicker="Factors" title="Factor definitions" description="Style factors are long/short ETF spreads: tradable proxies, not the academic Fama–French or AQR portfolios." />
        <ul className="divide-y divide-[var(--hairline)] text-xs">
          {sample.columns.map((c) => (
            <li key={c.key} className="grid gap-1 py-2 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-4">
              <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                {c.label} <span className="font-mono text-[10px] text-zinc-400">{c.short}</span>
              </span>
              <span className="text-zinc-600 dark:text-zinc-400">{c.description}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel p-5">
        <SectionHeader kicker="Data audit" title="Every series in this fit" />
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--hairline)] text-zinc-500">
                <th className="py-2 pr-3 text-left font-semibold">Ticker</th>
                <th className="px-3 py-2 text-left font-semibold">Role</th>
                <th className="px-3 py-2 text-right font-semibold">Bars in window</th>
                <th className="px-3 py-2 text-left font-semibold">First → last</th>
                <th className="px-3 py-2 text-left font-semibold">Price basis</th>
                <th className="py-2 pl-3 text-left font-semibold">Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--hairline)] tabular-nums">
              {a.series.map((s) => {
                const src = sourcesByTicker[s.ticker] ?? 'synthetic_gbm';
                return (
                  <tr key={s.ticker}>
                    <td className="py-2 pr-3 font-semibold text-zinc-900 dark:text-zinc-100">{s.ticker}</td>
                    <td className="px-3 py-2 text-zinc-600 dark:text-zinc-400">{s.roles.join(', ')}</td>
                    <td className="px-3 py-2 text-right">{s.barsInWindow.toLocaleString()}</td>
                    <td className="px-3 py-2 text-zinc-600 dark:text-zinc-400">{s.firstDate ?? '—'} → {s.lastDate ?? '—'}</td>
                    <td className={`px-3 py-2 ${s.basis === 'close' ? 'text-amber-600 dark:text-amber-400' : ''}`}>
                      {s.basis === 'adjusted' ? 'Adjusted (splits + dividends)' : 'Close only'}
                    </td>
                    <td className={`py-2 pl-3 ${src === 'live_market' ? 'text-[var(--positive)]' : 'text-gold-700 dark:text-gold-300'}`}>
                      {src === 'live_market' ? 'Live (Yahoo)' : 'Synthetic'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel p-5">
        <SectionHeader kicker="Limits" title="What this lab does not claim" />
        <ul className="list-disc space-y-1.5 pl-5 text-xs text-zinc-600 dark:text-zinc-400">
          <li>Exposures are associations over the sample. They do not explain why the stock moved, and they can change.</li>
          <li>Same-day factor returns are unknown in advance, so in-sample fits are not forecasts. Forecasting lives in the Predict tab and is scored out of sample.</li>
          <li>ETF proxies carry fees, tracking error and index-construction choices. MTUM, QUAL and USMV begin in 2011–2013, which shortens samples that include them.</li>
          <li>Fundamental, analyst and options data are current snapshots without point-in-time history, so they are excluded from time-series fits. Use the Screen tab for cross-sectional work.</li>
          <li>Yahoo quotes can lag or be revised. Synthetic fallback data is reproducible but not real, and the lab says so when it is in use.</li>
        </ul>
      </section>
    </div>
  );
}
