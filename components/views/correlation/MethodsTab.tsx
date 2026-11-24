'use client';

import React from 'react';
import type { CorrelationSample } from '@/lib/correlation-engine';
import { SectionHeader } from '@/components/ui/lab-ui';
import { DataTable, td, tdClass, thClass } from './shared';

const DEFINITIONS: [string, string][] = [
  ['Correlation', 'Pearson correlation of daily returns on dates when every asset in the basket traded. The 95% range for a pair is a moving-block bootstrap, which allows for volatility clustering.'],
  ['Independent bets', 'N ÷ (1 + (N − 1) × average correlation): the number of uncorrelated, equally risky assets that would diversify as well as the basket at equal risk. For a weighted portfolio it is the diversification ratio squared.'],
  ['Top common factor', 'Largest eigenvalue of the correlation matrix divided by the number of assets. Factors are "real signal" only above the Marchenko–Pastur edge (1 + √(N/T))².'],
  ['Shrunk estimate', 'Ledoit–Wolf (2004) shrinkage of the covariance toward a constant-correlation target, with the intensity estimated from the data.'],
  ['Pattern change', 'Mean absolute change in pair correlations between adjacent, non-overlapping windows. The joint test is Jennrich (1970); it assumes normal returns and over-rejects with fat tails.'],
  ['DCC model', 'Engle (2002) DCC(1,1): variance-targeted GARCH(1,1) for each asset, then the correlation parameters by likelihood on a fine grid. Tested against constant correlation with a likelihood-ratio test.'],
  ['Tail correlation', 'Exceedance correlation (Longin–Solnik, Ang–Chen): correlation on days when both assets are beyond a quantile. The benchmark is the same statistic for a bivariate normal with the pair’s full-sample correlation, computed numerically.'],
  ['Joint crashes', 'Rank-based probability that one asset is in its worst q of days given the other is. The benchmark is a normal copula matched to the pair’s Spearman correlation.'],
  ['Risk share', 'Euler decomposition: weight × (Σw) ÷ portfolio volatility. Covariances are daily and annualised once (× 252) when a volatility is reported.'],
  ['Loss figures', 'One-day. Historical VaR is the empirical quantile; the normal-model figure is z × volatility. Worst-days loss (expected shortfall) is the mean loss beyond the quantile; each asset’s part is its weighted mean return on those same days.'],
  ['Standard mixes', 'All long-only. Minimum variance by an exact active-set solve; equal risk by cyclical coordinate descent; hierarchical risk parity by single-linkage clustering and recursive bisection; max diversification as minimum variance on the correlation matrix.'],
  ['Reality check', 'Walk-forward: weights from the trailing 252 days (shrunk covariance), held 21 days, then reset. No future data is used.'],
  ['Timing test', 'Granger-style regression of each asset on five lags of itself and of the other, with White-robust errors. A leader is named only when exactly one direction is significant at 2.5%.'],
];

export function MethodsTab({ sample, sourcesByTicker }: { sample: CorrelationSample; sourcesByTicker: Record<string, string> }) {
  const a = sample.audit;
  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <SectionHeader
          kicker="Data audit"
          title="Every series in this basket"
          description={`${a.returnObs.toLocaleString()} ${a.returnType} returns on ${a.commonPriceDates.toLocaleString()} shared trading dates, ${a.start} to ${a.end}. ${a.droppedSessions} session(s) of the longest series were dropped to align calendars.`}
        />
        <DataTable>
          <thead>
            <tr>
              <th className={thClass}>Ticker</th>
              <th className={thClass}>Bars in window</th>
              <th className={thClass}>First</th>
              <th className={thClass}>Last</th>
              <th className={thClass}>Price basis</th>
              <th className={thClass}>Source</th>
            </tr>
          </thead>
          <tbody>
            {a.series.map((s) => {
              const live = (sourcesByTicker[s.ticker] ?? 'synthetic_gbm') === 'live_market';
              return (
                <tr key={s.ticker}>
                  <td className={td('text-zinc-900 dark:text-zinc-100', 'font-semibold')}>{s.ticker}</td>
                  <td className={tdClass}>{s.barsInWindow.toLocaleString()}</td>
                  <td className={tdClass}>{s.firstDate ?? '—'}</td>
                  <td className={tdClass}>{s.lastDate ?? '—'}</td>
                  <td className={td(s.basis === 'close' && 'text-amber-600 dark:text-amber-400')}>
                    {s.basis === 'adjusted' ? 'Adjusted (splits + dividends)' : 'Close only'}
                  </td>
                  <td className={td(live ? 'text-[var(--positive)]' : 'text-gold-700 dark:text-gold-300')}>{live ? 'Live (Yahoo)' : 'Simulated'}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </section>

      <section className="panel p-5">
        <SectionHeader kicker="Definitions" title="How each number is computed" />
        <dl className="divide-y divide-[var(--hairline)] text-xs">
          {DEFINITIONS.map(([k, v]) => (
            <div key={k} className="grid gap-1 py-2.5 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-4">
              <dt className="font-semibold text-zinc-900 dark:text-zinc-100">{k}</dt>
              <dd className="leading-relaxed text-zinc-600 dark:text-zinc-400">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="panel p-5">
        <SectionHeader kicker="Limits" title="What this page does not claim" />
        <ul className="list-disc space-y-1.5 pl-5 text-xs text-zinc-600 dark:text-zinc-400">
          <li>Correlation measures co-movement over the sample. It does not say why assets move together, and it changes.</li>
          <li>Every figure is in-sample except the walk-forward reality check and the experimental forecast.</li>
          <li>With many pairs and several tests on one page, some &quot;significant&quot; results are expected by chance.</li>
          <li>Loss figures are one-day, long-only, and ignore liquidity, costs and intraday moves.</li>
          <li>If a series is missing or too short, the page refuses to compute. It never substitutes simulated prices or default values.</li>
          <li>Yahoo data can lag or be revised. When the app falls back to simulated prices, the basket bar names the affected tickers.</li>
        </ul>
      </section>
    </div>
  );
}
