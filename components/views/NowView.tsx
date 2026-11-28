'use client';

import React, { useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  BadgeCheck,
  Crosshair,
  Gauge,
  Magnet,
  RefreshCw,
  ShieldAlert,
  TrendingUp,
} from 'lucide-react';
import type { TapeItem } from '@/components/ui/ticker-tape';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import { canonicalVrp, selectFrontSlice } from '@/lib/ticker-snapshot';
import { useWatchlist } from '@/hooks/use-watchlist';

type View = 'home' | 'analysis' | 'modeling' | 'options' | 'correlation' | 'regression';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const signed = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

export function NowView({
  ticker,
  companyName,
  tapeItems,
  onNavigate,
}: {
  ticker: string;
  companyName: string;
  tapeItems: TapeItem[];
  onNavigate: (view: View, params?: Record<string, string>) => void;
}) {
  const { chain, indices, panel, chainLoading, refresh, rvLadder, rv20, vix, curve, session, barsSource, intradaySource } = useTickerSnapshot();
  const watch = useWatchlist();
  const [sortKey, setSortKey] = useState<'symbol' | 'iv30' | 'trapScore'>('symbol');
  const watchRows = [...watch.rows].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return typeof av === 'string' ? av.localeCompare(String(bv)) : (av as number) - (bv as number);
  });

  const iv30 = chain?.iv30 ?? null;
  const vrp = canonicalVrp(iv30, rv20);
  const trap = chain?.trap ?? null;
  const front = selectFrontSlice(chain?.slices ?? []);
  const spot = chain?.quote?.price ?? null;
  const changePct = chain?.quote?.changePct ?? null;

  const headline = !chainAvailable(chain)
    ? 'Chain unavailable — reading price history and accrued data only.'
    : vrp.variancePoints === null
      ? 'Live chain present; realized-history gate still accruing.'
      : vrp.variancePoints >= 0
        ? `Options price ${pct(Math.abs(iv30! - rv20!))} more annualized variance than the recent tape realizes${curve === 'INVERTED' ? ', with an inverted front end' : ''}${trap?.state === 'armed' || trap?.state === 'triggered' ? ` · dealer trap ${trap.state}` : ''}.`
        : `Options price ${pct(Math.abs(iv30! - rv20!))} less annualized variance than the recent tape realizes — premium is negative.`;

  const jump = (view: View, params?: Record<string, string>) => (
    <button
      type="button"
      onClick={() => onNavigate(view, params)}
      className="inline-flex cursor-pointer items-center gap-1 font-mono text-[10px] font-semibold text-gold-700 hover:underline dark:text-gold-300"
    >
      Open detail <ArrowUpRight className="h-3 w-3" />
    </button>
  );

  return (
    <div className="space-y-4">
      {/* Hero — the single biggest thing right now */}
      <section className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="kicker">Now · {companyName}</span>
              <ProvenanceBadge daily={barsSource} intraday={intradaySource} />
              {chain?.substitution && <span className="font-mono text-[9px] text-gold-600 dark:text-gold-300">proxy: {chain.substitution}</span>}
            </div>
            <h1 className="font-display text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              {ticker} <span className="font-mono text-base font-semibold tabular-nums text-zinc-500">{spot !== null ? spot.toFixed(2) : '—'}</span>
              {changePct !== null && (
                <span className={`ml-2 font-mono text-sm tabular-nums ${changePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
                  {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                </span>
              )}
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">{headline}</p>
          </div>
          <button
            type="button"
            onClick={refresh}
            disabled={chainLoading}
            aria-label="Refresh snapshot"
            className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[10px] font-semibold text-zinc-600 disabled:opacity-60 dark:text-zinc-300"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${chainLoading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
          {[
            { label: 'IV30', value: pct(iv30), sub: 'European BSM · √365' },
            { label: 'RV20', value: pct(rv20), sub: 'sample σ · √252' },
            { label: 'IV−RV', value: vrp.premiumPoints === null ? '—' : `${signed(vrp.premiumPoints)} pts`, sub: 'variance risk premium' },
            { label: 'Curve', value: curve ?? '—', sub: 'front vs 45d+' },
            { label: 'VIX', value: vix.level === null ? '—' : vix.level.toFixed(1), sub: vix.percentile === null ? 'stress proxy' : `${vix.percentile.toFixed(0)}th pctile · 1y` },
            { label: 'Trap', value: trap?.state?.toUpperCase() ?? '—', sub: trap ? `score ${trap.score.toFixed(0)}` : 'dealer positioning' },
          ].map((t) => (
            <div key={t.label} className="panel-inset p-3">
              <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-400">{t.label}</div>
              <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">{t.value}</div>
              <div className="truncate font-mono text-[9px] text-zinc-400">{t.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Δ since the prior accrued session */}
      <section className="panel p-5">
        <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><Activity className="h-3.5 w-3.5" /> Since Prior Session</p>
          <span className="font-mono text-[9px] text-zinc-400">{session ? session.date : 'no accrued history yet'}</span>
        </div>
        {session ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { label: 'Spot', value: signed(session.dSpot, 2) },
              { label: 'IV30', value: signed(session.dIv30 === null ? null : session.dIv30 * 100, 2) + ' pts' },
              { label: 'RV20', value: signed(session.dRv20 === null ? null : session.dRv20 * 100, 2) + ' pts' },
              { label: 'Trap score', value: signed(session.dTrapScore, 0) },
            ].map((d) => (
              <div key={d.label} className="panel-inset p-3">
                <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{d.label}</div>
                <div className={`mt-1 font-mono text-base font-semibold tabular-nums ${d.value.startsWith('-') ? 'text-[var(--negative)]' : 'text-[var(--positive)]'}`}>{d.value}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="font-mono text-[10px] text-zinc-400">Need at least two accrued panel sessions. The chain snapshot accrues on every fetch; nothing is interpolated.</p>
        )}
      </section>

      {/* IV vs RV + RV ladder */}
      <section className="panel p-5">
        <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><TrendingUp className="h-3.5 w-3.5" /> Implied vs Realized</p>
          {jump('options', { vol: 't3' })}
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div>
            <p className="mb-2 font-mono text-[10px] text-zinc-400">One VRP definition: IV² − RV² (variance points, ×10⁴)</p>
            <div className={`font-mono text-3xl font-bold tabular-nums ${vrp.variancePoints !== null && vrp.variancePoints >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
              {vrp.variancePoints === null ? '—' : signed(vrp.variancePoints, 0)}
            </div>
            <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
              RV is √252 and IV is √365 — annualizations of the same total variance, compared directly. Other tabs label their own estimators (GARCH path, MZ slope, spread z-score) separately.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="table-sticky-first w-full border-collapse font-mono text-[10px]">
              <thead><tr className="text-left text-[9px] uppercase tracking-wider text-zinc-400"><th className="py-1">Window</th><th className="py-1 text-right">RV (√252)</th><th className="py-1 text-right">n</th></tr></thead>
              <tbody>
                {rvLadder.map((r) => (
                  <tr key={r.window} className="border-t border-[var(--hairline)]">
                    <td className="py-1 text-zinc-500">{r.window}d</td>
                    <td className="py-1 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{pct(r.vol)}</td>
                    <td className="py-1 text-right tabular-nums text-zinc-400">{r.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Trade + risks */}
      <section className="panel p-5">
        <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><Magnet className="h-3.5 w-3.5" /> Trade &amp; Risk Context</p>
          <div className="flex gap-3">
            {jump('options', { vol: 't7' })}
            {jump('options', { fno: 'gex' })}
          </div>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Dealer positioning</div>
            <div className="mt-1 font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-50">{trap?.state?.toUpperCase() ?? '—'}</div>
            <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-400">GEX convention is assumed, not observed. Walls and flip live in T7.</p>
          </div>
          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Front skew (15–45d)</div>
            <div className="mt-1 font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-50">
              {front ? pct(front.rr25 ?? null, 1) : '—'} <span className="text-[10px] text-zinc-400">RR25</span>
            </div>
            <p className="mt-1 font-mono text-[9px] text-zinc-400">{front ? `${front.dte}d slice · BF ${pct(front.bf25 ?? null)}` : 'no usable slice'}</p>
          </div>
          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Decision checklist</div>
            <div className="mt-1 font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-50">T8</div>
            <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-400">Pre-registered rules degrade to insufficient-data rather than defaulting to met. Earnings is checked there.</p>
          </div>
        </div>
      </section>

      {/* One combined watchlist scan — replaces two separate manual toolbar scans */}
      <section className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><Gauge className="h-3.5 w-3.5" /> Watchlist · One Scan</p>
          <div className="flex items-center gap-3">
            {(['symbol', 'iv30', 'trapScore'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setSortKey(k)}
                className={`cursor-pointer font-mono text-[9px] font-semibold uppercase tracking-wider ${sortKey === k ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200'}`}
              >
                {k === 'trapScore' ? 'trap' : k}
              </button>
            ))}
            <button
              type="button"
              onClick={() => void watch.scan(true)}
              disabled={watch.loading}
              className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 font-mono text-[10px] font-semibold disabled:opacity-60"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${watch.loading ? 'animate-spin' : ''}`} /> Scan watchlist
            </button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="table-sticky-first w-full border-collapse font-mono text-[10px]">
            <thead>
              <tr className="text-left text-[9px] uppercase tracking-wider text-zinc-400">
                <th className="py-1">Symbol</th><th className="py-1 text-right">IV30</th><th className="py-1 text-right">RR25</th><th className="py-1 text-right">Trap</th><th className="py-1 text-right">Score</th><th className="py-1 text-right">Source</th>
              </tr>
            </thead>
            <tbody>
              {watchRows.length === 0 && (
                <tr><td colSpan={6} className="py-3 text-center text-zinc-400">Scan to load IV and dealer state for the watchlist in one pass.</td></tr>
              )}
              {watchRows.map((r) => (
                <tr key={r.symbol} className="border-t border-[var(--hairline)]">
                  <td className="py-1 font-semibold text-zinc-800 dark:text-zinc-100">{r.symbol}</td>
                  <td className="py-1 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{pct(r.iv30)}</td>
                  <td className="py-1 text-right tabular-nums text-zinc-500">{pct(r.rr25)}</td>
                  <td className="py-1 text-right text-zinc-700 dark:text-zinc-200">{r.trapState?.toUpperCase() ?? '—'}</td>
                  <td className="py-1 text-right tabular-nums text-zinc-500">{r.trapScore ?? '—'}</td>
                  <td className="py-1 text-right text-zinc-400">{r.available ? 'live' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">One cached pass over the six-name universe; IV is European BSM on delayed quotes, trap state is an assumed GEX convention. T6 and T7 detail panels remain the drill-down.</p>
      </section>

      {/* Cross-section + model trust */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <p className="kicker flex items-center gap-2"><Gauge className="h-3.5 w-3.5" /> Cross-Section</p>
            <div className="flex gap-3">{jump('options', { vol: 't6' })}{jump('correlation')}</div>
          </div>
          <div className="max-h-72 overflow-y-auto font-mono text-[10px]">
            {tapeItems.length === 0 && <p className="text-zinc-400">No cached prices yet.</p>}
            {tapeItems.map((t) => (
              <div key={t.symbol} className="grid grid-cols-3 gap-2 border-t border-[var(--hairline)] py-1.5 first:border-0">
                <span className="font-semibold text-zinc-800 dark:text-zinc-100">{t.symbol}</span>
                <span className="text-right tabular-nums text-zinc-500">{t.price.toFixed(2)}</span>
                <span className={`text-right tabular-nums ${t.changePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>{signed(t.changePct, 2)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <p className="kicker flex items-center gap-2"><ShieldAlert className="h-3.5 w-3.5" /> Model Trust</p>
            <div className="flex gap-3">{jump('modeling')}{jump('options', { vol: 't4' })}</div>
          </div>
          <p className="text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
            Every cone is an in-sample fit. A walk-forward trust card now scores GJR-GARCH and HMM alongside EWMA/GARCH/historical on a held-out tail (Volatility → T4 and Modeling). T4&apos;s in-sample horse race still covers only GARCH(1,1), EWMA and HAR-RV. Read each cone with its <span className="font-mono text-gold-600 dark:text-gold-300">QLIKE score</span> in view.
          </p>
          <div className="mt-3 flex items-center gap-2 font-mono text-[9px] text-zinc-400">
            <BadgeCheck className="h-3.5 w-3.5" /> chain: {chain?.available ? 'live' : 'unavailable'} · indices: {indices?.available ? 'live' : 'fallback'} · panel rows: {panel.length}
          </div>
        </div>
      </section>
    </div>
  );
}

function chainAvailable(chain: ReturnType<typeof useTickerSnapshot>['chain']) {
  return chain?.available === true;
}
