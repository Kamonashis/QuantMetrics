'use client';

import { useMemo, useState } from 'react';
import { RndChart } from '@/components/charts/RndChart';
import { RndCdfChart } from '@/components/charts/RndCdfChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { MathDisplay } from '@/components/ui/MathDisplay';
import { creditBridge } from '@/lib/credit-bridge';
import { buildRndResearch, historicalHorizonReturns, sampleMoments } from '@/lib/rnd-integrity';
import { completedResearchBars, usMarketDate } from '@/lib/options-research';
import type { PriceBar } from '@/lib/quant-engine';
import type { ChainPayload } from './types';

const fmt = (x: number | null | undefined, f: (v: number) => string) => x == null || !Number.isFinite(x) ? '—' : f(x);
const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
const num = (x: number) => x.toFixed(2);
const note = 'mt-2 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400';
const table = 'w-full min-w-[560px] text-right text-xs tabular-nums [&_td]:py-2 [&_th]:py-2 [&_tr]:border-b [&_tr]:border-[var(--hairline)] [&_th:first-child]:text-left [&_td:first-child]:text-left';

export function T5DensityTab({ chain, bars, barsSource, research, ticker }: {
  chain: ChainPayload | null;
  bars: PriceBar[];
  barsSource?: string | null;
  research?: ReturnType<typeof buildRndResearch>;
  ticker: string;
}) {
  const [debtInput, setDebtInput] = useState<{ ticker: string; value: string } | null>(null);
  const local = useMemo(() => research === undefined ? buildRndResearch(chain) : null, [chain, research]);
  const data = research === undefined ? local : research;
  const quoteDate = usMarketDate(chain?.quote?.asOf ?? chain?.asOf ?? '');
  const horizon = data && quoteDate ? Math.round((Date.parse(data.slice.expirationDate) - Date.parse(quoteDate)) / 86_400_000) : null;
  const realized = useMemo(() => barsSource === 'live_market' && quoteDate && !chain?.substitution && horizon
    ? historicalHorizonReturns(completedResearchBars(bars, quoteDate), horizon) : [], [barsSource, quoteDate, chain?.substitution, horizon, bars]);
  const realizedMoments = useMemo(() => sampleMoments(realized), [realized]);
  const equity = chain?.quote?.marketCap;
  const assumed = equity && equity > 0 ? String(Number((equity * 0.3 / 1e9).toPrecision(4))) : '';
  const debtText = debtInput?.ticker === ticker ? debtInput.value : assumed;
  const credit = chain?.substitution || chain?.quote?.securityType === 'ETF' ? null : creditBridge(equity, chain?.iv30,
    debtInput?.ticker === ticker ? Number(debtText) * 1e9 : null, chain?.rate);
  const ccy = chain?.quote?.currency ?? '';
  const money = (v: number) => `${v.toFixed(3)} ${ccy}`.trim();
  if (!data) return <section className="panel p-5"><p className="kicker">Risk-Neutral Density</p><p className={note}>No usable options snapshot. Density and payoff estimates are withheld.</p></section>;
  const { spot, slice, market, fitted, hybrid, quality } = data;
  const primary = hybrid ?? market ?? fitted;
  const label = hybrid ? 'Market interior + SVI wings' : market ? 'Market · observed range only (diagnostic)' : 'SVI fitted density';
  const cdf = hybrid ?? fitted;
  return (
    <div className="space-y-4">
      <section className="panel p-5">
        <p className="kicker">Research data quality · {quality.replace('_', ' ')}</p>
        <p className={note}>Expiry {slice.expirationDate} · {slice.dte} calendar DTE · chain spot {spot.toFixed(2)} {ccy}. A pass is not an execution check.</p>
        <ul className="mt-3 grid gap-2 sm:grid-cols-2">
          {data.checks.map((check) => <li key={check.label} className="panel-inset p-3 text-xs">
            <strong>{check.pass ? 'Pass' : 'Review'} · {check.label}</strong><p className={note}>{check.detail}</p>
          </li>)}
        </ul>
        <p className={note}>Delayed mids and European models are not executable quotes. Check a broker’s current underlying, bid/ask, contract terms, and fees before considering any order. A model gap is not arbitrage.</p>
      </section>

      <section className="panel p-5">
        <p className="kicker">Market-implied distribution vs. historical returns</p>
        <div className="mt-2"><MathDisplay preset="breeden-litzenberger" display="inline" className="text-[11px]" /></div>
        {market && <p className={note}>{market.nUsed} strikes · {market.nDirect} direct · {market.nParity} parity · {market.dropped} filtered. Removed curvature can occur anywhere on the grid; it is not unobserved tail mass.</p>}
        {primary ? <ChartFullscreen title="Risk-neutral density vs. historical returns"><RndChart rnd={primary}
          fittedRnd={primary === fitted ? null : fitted} spot={spot} realizedReturns={realizedMoments ? realized : []} primaryLabel={label} /></ChartFullscreen>
          : <p className={note}>No usable density.</p>}
        {cdf && <RndCdfChart rnd={{ ...cdf, spot }} label={hybrid ? 'Market + SVI wings CDF' : 'SVI fitted CDF'} />}
        <p className={note}>{hybrid ? `SVI supplies ${pct(hybrid.leftTailMass)} below and ${pct(hybrid.rightTailMass)} above the quoted range. ` : ''}
          Historical comparison: {realized.length} overlapping {horizon ?? '—'}-calendar-day log-return windows (last close on or before each horizon); requires 10 observed windows. Proxy and synthetic bars are not mixed into this comparison.</p>
      </section>

      <section className="panel p-5">
        <p className="kicker">Implied vs. historical log-return moments</p>
        <p className={note}>Same return axis and expiry horizon. Flat ATM-IV reference: σ√T = {fmt(data.atmDispersion, pct)} using T on a 365-day basis. Trailing annualized RV20 and historical horizon dispersion are different estimators.</p>
        <div className="overflow-x-auto"><table className={table}>
          <thead><tr><th>Moment</th><th>Market + SVI wings</th><th>SVI</th><th>Historical</th></tr></thead>
          <tbody>{([['mean', 'Mean log return', pct], ['std', 'Dispersion', pct], ['skew', 'Skew', num], ['kurt', 'Excess kurtosis', num]] as const).map(([key, name, f]) =>
            <tr key={key}><td>{name}</td><td>{fmt(data.marketMoments?.[key], f)}</td><td>{fmt(data.fittedMoments?.[key], f)}</td><td>{fmt(realizedMoments?.[key], f)}</td></tr>)}</tbody>
        </table></div>
      </section>

      <section className="panel p-5">
        <p className="kicker">Price with the RND · model estimates</p>
        <div className="overflow-x-auto"><table className={table}>
          <thead><tr><th>Payoff</th><th>Market + SVI wings</th><th>SVI</th><th>Flat ATM BSM</th><th>SVI − BSM</th><th>Quoted strikes</th><th>Payoff from modeled wings</th></tr></thead>
          <tbody>{data.rows.map((row) => <tr key={row.label}><td>{row.label}</td><td>{fmt(row.market, money)}</td><td>{fmt(row.fitted, money)}</td><td>{fmt(row.bs, money)}</td>
            <td>{fmt(row.fitted != null && row.bs != null ? row.fitted - row.bs : null, money)}</td><td>{market ? row.inRange ? 'Inside range' : 'Outside range' : 'Unavailable'}</td><td>{fmt(row.tailShare, pct)}</td></tr>)}</tbody>
        </table></div>
        <p className={note}>Digitals pay one currency unit; price = discount factor × the corresponding probability below. The call spread is per share. Missing estimates are withheld, not treated as zero.</p>
      </section>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="panel p-5">
          <p className="kicker">MFIV vs. expiry ATM IV</p>
          <div className="overflow-x-auto"><table className={`${table} !min-w-[320px]`}>
            <thead><tr><th>Expiry</th><th>MFIV</th><th>ATM IV</th><th>Δ vol pts</th></tr></thead>
            <tbody>{chain?.slices?.map((s) => {
              const diff = s.mfivAnnualized != null && s.atmIv != null ? (s.mfivAnnualized - s.atmIv) * 100 : null;
              return <tr key={s.expirationDate}><td>{s.expirationDate}</td><td>{fmt(s.mfivAnnualized, pct)}</td><td>{fmt(s.atmIv, pct)}</td>
                <td className={diff !== null && Math.abs(diff) > 2 ? 'text-amber-700 dark:text-amber-400' : ''}>{fmt(diff, num)}</td></tr>;
            })}</tbody>
          </table></div>
          <p className={note}>A gap over 2 vol points warrants review. These estimators need not agree: smile shape and strike coverage both matter.</p>
        </section>
        <section className="panel p-5">
          <p className="kicker">Expiry probabilities · risk-neutral</p>
          <div className="overflow-x-auto"><table className={`${table} !min-w-[350px]`}>
            <thead><tr><th>Event</th><th>Market + SVI wings</th><th>SVI</th><th>Flat ATM BSM</th></tr></thead>
            <tbody>{data.probabilities.map((row) => <tr key={row.label}><td>{row.label}</td><td>{fmt(row.market, pct)}</td><td>{fmt(row.fitted, pct)}</td><td>{fmt(row.bs, pct)}</td></tr>)}</tbody>
          </table></div>
          <p className={note}>These are terminal model probabilities, not real-world odds. Separately, the flat-vol, continuously monitored GBM estimate for touching −10% before expiry is {fmt(data.touchDown10, pct)}. Touch is path-dependent, not inferred from the terminal density.</p>
        </section>
      </div>

      <section className="panel p-5">
        <p className="kicker">Credit bridge · Merton-style proxy</p>
        <label className="mt-3 flex items-center gap-2 text-xs">Debt ({ccy || 'currency'} billions)
          <input className="panel-inset w-32 p-2" type="number" min="0" step="any" value={debtText} placeholder="Unavailable"
            onChange={(e) => setDebtInput({ ticker, value: e.target.value })} />
          {debtInput?.ticker === ticker && <button type="button" onClick={() => setDebtInput(null)} className="underline">Reset assumption</button>}
        </label>
        {credit ? <p className="mt-3 text-sm tabular-nums">Assets {(credit.assetValue / 1e9).toFixed(2)}B · Asset vol {pct(credit.assetVol)} · Distance to default {num(credit.dd)}σ · 1y model PD {pct(credit.pd)}</p>
          : <p className={note}>Unavailable without positive equity value, IV30, and debt inputs for an applicable company.</p>}
        <p className={note}>Debt defaults to 30% of market cap; IV30 proxies equity vol and σ_A ≈ σ_E·E/V. Not a calibrated default forecast or a reason to buy or sell volatility.</p>
      </section>
    </div>
  );
}
