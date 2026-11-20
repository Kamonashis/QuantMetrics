'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { ArrowUpRight, CircleDot, RefreshCw, ShieldAlert, Target } from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import { calculateReturns, decomposeTimeSeries, computeAdfTest } from '@/lib/quant-engine';
import type { TapeItem } from '@/components/ui/ticker-tape';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { EmptyState } from '@/components/ui/empty-state';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import { canonicalRvLadder, canonicalVrp, selectFrontSlice } from '@/lib/ticker-snapshot';
import { evaluateVolSignals, BREAKEVEN_MULT, type SignalRuleResult } from '@/lib/vol-signals';
import { runTrapBacktest } from '@/lib/trap-backtest';
import { historicalAbsMoves, moveVerdict, percentileRankLocal, moveDistributionSummary } from '@/lib/fno/expected-move';
import { computeRollingMomentsWithBands, computeStructuralBreaks, computeHurstAndVarianceRatio } from '@/lib/analysis-engine';
import { fitGaussianHmm } from '@/lib/hmm';
import { percentileRank } from '@/lib/vol-analytics';
import {
  evaluateOpportunities,
  rankOpportunities,
  type Opportunity,
  type OpportunityInputs,
} from '@/lib/opportunity-engine';
import { summarizeHitRates, type HitRate, type SignalJournalRecord } from '@/lib/signal-outcomes';

type View = 'home' | 'analysis' | 'modeling' | 'volatility' | 'fno' | 'correlation' | 'regression';

const pct = (v: number | null | undefined, d = 1) => v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

const STATE_STYLE: Record<string, string> = {
  ACTIONABLE: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  WATCH: 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300',
  BLOCKED: 'border-rose-500/40 bg-rose-500/10 text-rose-500 dark:text-rose-400',
  INSUFFICIENT_DATA: 'border-[var(--hairline)] bg-[var(--bg-inset)] text-zinc-400',
};

const FAMILY_LABEL: Record<string, string> = {
  'premium-harvest': 'Premium harvest',
  'long-gamma': 'Long gamma',
  'directional-trap': 'Directional trap',
  'event-straddle': 'Event straddle',
  'skew-tail': 'Skew / tail',
  'cross-sectional': 'Cross-sectional',
};

/** Build the pre-registered rule inputs from the active ticker's daily bars. */
function signalInputs(bars: PriceBar[]) {
  const dates = bars.map((b) => b.date);
  const prices = bars.map((b) => b.close);
  const returns = calculateReturns(prices, 'log');

  const feats: number[][] = [];
  for (let i = 1; i < bars.length; i++) {
    const b = bars[i];
    const prev = bars[i - 1];
    const o = b.open, h = b.high, l = b.low, c = b.close;
    if (!(o > 0 && h > 0 && l > 0 && c > 0 && prev.close > 0)) continue;
    const r = Math.log(c / prev.close);
    const gk = 0.5 * Math.log(h / l) ** 2 - (2 * Math.LN2 - 1) * Math.log(c / o) ** 2;
    feats.push([r, Math.abs(r), Math.log(Math.max(gk, 1e-8))]);
  }
  const hmm = bars.length >= 60 ? fitGaussianHmm(feats, { k: 3, rankFeature: 2 }) : null;
  let crisisProb: number | null = null;
  let dominant: number | null = null;
  if (hmm && hmm.posteriors.length) {
    const K = hmm.posteriors[0].length;
    const last = hmm.posteriors[hmm.posteriors.length - 1];
    let best = 0;
    for (let k = 1; k < K; k++) if ((last[k] ?? 0) > (last[best] ?? 0)) best = k;
    dominant = best;
    crisisProb = last[K - 1] ?? 0;
  }

  const skew = dates.length >= 70 ? computeRollingMomentsWithBands(dates, returns, 60).rollingSkew.at(-1) : null;
  const breakDate = dates.length >= 20 ? computeStructuralBreaks(dates, returns, prices).breakDates.at(-1) : null;
  const breakIdx = breakDate ? dates.lastIndexOf(breakDate.date) : -1;
  const cusumAgo = breakIdx >= 0 ? dates.length - 1 - breakIdx : null;
  const resid = dates.length >= 60
    ? decomposeTimeSeries(dates, prices, 30, 'additive').residual.filter((v): v is number => v !== null && isFinite(v))
    : [];
  const adfP = resid.length >= 30 ? computeAdfTest(resid).pValue : null;
  const vr = returns.length >= 40 ? computeHurstAndVarianceRatio(returns) : null;
  const vrMinZ = vr ? vr.varianceRatios.reduce((m: number, v: { zStat: number }) => Math.min(m, v.zStat), Infinity) : null;

  return { dates, returns, crisisProb, dominant, skew: skew ?? null, cusumAgo, adfP, vrMinZ: isFinite(vrMinZ as number) ? vrMinZ : null };
}

/** A separate module-level helper so no clock or randomness appears in render. */
function toJournalRecords(
  ops: Opportunity[],
  date: string,
  symbol: string,
  ctx: {
    iv30: number | null; spot: number | null; trap: OpportunityInputs['trap'];
    expectedMove: OpportunityInputs['expectedMove']; crossSectional: OpportunityInputs['crossSectional'];
  }
): SignalJournalRecord[] {
  const ranked = [...ctx.crossSectional].sort((a, b) =>
    ((b.iv30 ?? 0) - (b.rv20 ?? 0)) - ((a.iv30 ?? 0) - (a.rv20 ?? 0))
  );
  return ops.map((op) => ({
    date, symbol, setup: op.id, state: op.state,
    iv30: ctx.iv30,
    breakeven: ctx.iv30 !== null ? ctx.iv30 * BREAKEVEN_MULT : null,
    spot: ctx.spot,
    direction: ctx.trap?.direction ?? 'none',
    wall: ctx.trap?.direction === 'up' ? ctx.trap?.callWall ?? null : ctx.trap?.putWall ?? null,
    flip: ctx.trap?.flip ?? null,
    impliedPct: ctx.expectedMove?.impliedPct ?? null,
    horizon: op.id === 'event-straddle' ? (ctx.expectedMove?.dte ?? 21) : op.id === 'long-gamma' ? 5 : op.id === 'directional-trap' ? 10 : 20,
    edgeEstimate: op.edge.estimate,
    pairLong: ranked.at(-1)?.symbol ?? null,
    pairShort: ranked[0]?.symbol ?? null,
    recordedAt: '',
  }));
}

export function EdgeView({
  ticker,
  companyName,
  bars,
  tapeItems,
  onNavigate,
}: {
  ticker: string;
  companyName: string;
  bars: PriceBar[];
  tapeItems: TapeItem[];
  onNavigate: (view: View, params?: Record<string, string>) => void;
}) {
  const s = useTickerSnapshot();
  const { scanWatchlist } = s;
  const [riskBudget, setRiskBudget] = useState(0.01);
  const [filter, setFilter] = useState<'all' | string>('all');
  const [journal, setJournal] = useState<SignalJournalRecord[]>([]);
  const [journalDate, setJournalDate] = useState('');

  const dates = useMemo(() => bars.map((b) => b.date), [bars]);
  const closes = useMemo(() => bars.map((b) => b.close), [bars]);
  const sig = useMemo(() => signalInputs(bars), [bars]);
  const rvLadder = useMemo(() => canonicalRvLadder(closes), [closes]);
  const rv5 = rvLadder.find((r) => r.window === 5)?.vol ?? null;
  const rv20 = s.rv20 ?? rvLadder.find((r) => r.window === 20)?.vol ?? null;

  const frontSkew = useMemo(() => selectFrontSlice(s.chain?.slices ?? []), [s.chain]);
  const iv30 = s.chain?.iv30 ?? null;
  const vrp = canonicalVrp(iv30, rv20);

  const rules = useMemo<SignalRuleResult[]>(() => evaluateVolSignals({
    iv30,
    rv5,
    rv10: rvLadder.find((r) => r.window === 10)?.vol ?? null,
    rv20,
    rv30: rvLadder.find((r) => r.window === 30)?.vol ?? null,
    skew60: sig.skew?.skew ?? null,
    skew60LowerCi: sig.skew?.lowerCi ?? null,
    skew60UpperCi: sig.skew?.upperCi ?? null,
    vrMinZ: sig.vrMinZ,
    adfResidualP: sig.adfP,
    cusumBreakSessionsAgo: sig.cusumAgo,
    hmmCrisisProb: sig.crisisProb,
    hmmDominantState: sig.dominant,
    rr25Front: frontSkew?.rr25 ?? null,
  }), [iv30, rv5, rv20, rvLadder, sig, frontSkew]);

  const backtest = useMemo(
    () => runTrapBacktest(s.panel as never, closes.map((c, i) => ({ date: dates[i], close: c })), 20),
    [s.panel, closes, dates]
  );

  const carry = useMemo(() => {
    const ivBars = ticker === 'QQQ' ? s.indices?.series?.VXN?.bars ?? [] : s.indices?.series?.VIX?.bars ?? [];
    if (ivBars.length < 300) return null;
    const map = new Map(ivBars.map((b) => [b.date, b.close / 100]));
    let mean = 0, wins = 0, n = 0, worst = Infinity, cumulative = 0;
    const curve: number[] = [];
    for (let i = 21; i < dates.length - 22; i++) {
      const vixT = map.get(dates[i]);
      if (vixT === undefined) continue;
      let fwd = 0;
      for (let j = i + 1; j <= i + 21 && j <= sig.returns.length; j++) fwd += sig.returns[j - 1] ** 2;
      const pl = vixT * vixT * (21 / 252) - fwd;
      mean += pl;
      if (pl > 0) wins++;
      if (pl < worst) worst = pl;
      cumulative += pl;
      curve.push(cumulative);
      n++;
    }
    if (n < 30) return null;
    mean /= n;
    let peak = -Infinity, maxDD = 0;
    for (const c of curve) { if (c > peak) peak = c; maxDD = Math.max(maxDD, peak - c); }
    return { meanMonthlyBp2: mean * 1e4, hitRate: wins / n, worstMonthBp2: worst * 1e4, maxDrawdownBp2: maxDD * 1e4, n };
  }, [ticker, s.indices, dates, sig.returns]);

  const expectedMove = useMemo(() => {
    const m = s.fnoPanel?.expectedMoves?.[0];
    if (!m) return null;
    const implied = m.straddleMove ?? m.sigmaMove ?? null;
    const samples = historicalAbsMoves(bars, m.dte ?? 21);
    const summary = moveDistributionSummary(samples);
    if (implied === null || !summary) return null;
    return {
      dte: m.dte ?? null,
      impliedPct: implied,
      verdict: moveVerdict(implied, samples),
      percentile: percentileRankLocal(samples, implied / 100),
      n: summary.n,
    };
  }, [s.fnoPanel, bars]);

  const crossSectional = useMemo(() => s.watchRows.map((w) => {
    const b = s.allStockBars[w.symbol] ?? [];
    const r20 = canonicalRvLadder(b.map((x) => x.close)).find((r) => r.window === 20)?.vol ?? null;
    return { symbol: w.symbol, iv30: w.iv30, rv20: r20, ivRank: null as number | null };
  }), [s.watchRows, s.allStockBars]);

  const vvixBars = s.indices?.series?.VVIX?.bars ?? [];
  const vvix = vvixBars.length ? vvixBars[vvixBars.length - 1].close : null;
  const vvixPercentile = vvix !== null && vvixBars.length > 30
    ? percentileRank(vvixBars.slice(-252).map((b) => b.close), vvix)
    : null;
  const skewBars = s.indices?.series?.SKEW?.bars ?? [];
  const skewNow = skewBars.length ? skewBars[skewBars.length - 1].close : null;
  const skewPercentile = skewNow !== null && skewBars.length > 30
    ? percentileRank(skewBars.slice(-252).map((b) => b.close), skewNow)
    : null;

  const inputs = useMemo<OpportunityInputs>(() => ({
    symbol: ticker,
    asOf: s.chain?.asOf ?? null,
    rules,
    iv30,
    spot: s.chain?.quote?.price ?? closes.at(-1) ?? null,
    rv5,
    vrpVariancePoints: vrp.variancePoints,
    chainAvailable: s.chain?.available === true,
    fnoAvailable: s.fnoPanel?.available === true,
    barsSource: s.barsSource ?? null,
    accruedSessions: s.chain?.snapshot?.count ?? 0,
    ivRankReady: (s.chain?.snapshot?.count ?? 0) >= 60,
    earningsSessions: s.earnings?.sessions ?? null,
    earningsIsEstimate: s.earnings?.isEstimate ?? null,
    term: s.chain?.termStructure ?? [],
    curve: s.curve,
    frontSkew: frontSkew ? { dte: frontSkew.dte, rr25: frontSkew.rr25 ?? null, bf25: frontSkew.bf25 ?? null } : null,
    vvix,
    vvixPercentile,
    skewPercentile,
    trap: s.chain?.trap ? {
      state: s.chain.trap.state,
      direction: s.chain.trap.direction,
      score: s.chain.trap.score,
      callWall: s.chain.trap.walls.call?.strike ?? null,
      putWall: s.chain.trap.walls.put?.strike ?? null,
      flip: s.chain.trap.gammaFlip.flip ?? null,
      backtestVerdict: backtest.verdict,
      backtestN: backtest.nDedupedEvents,
      convention: s.chain.trap.convention,
    } : null,
    expectedMove,
    carry,
    crossSectional,
    riskBudgetFraction: riskBudget,
  }), [ticker, s.chain, s.fnoPanel, s.barsSource, s.earnings, s.curve, rules, iv30, closes, rv5, vrp.variancePoints, frontSkew, vvix, vvixPercentile, skewPercentile, backtest, expectedMove, carry, crossSectional, riskBudget]);

  const opportunities = useMemo(() => rankOpportunities(evaluateOpportunities(inputs)), [inputs]);
  const visible = filter === 'all' ? opportunities : opportunities.filter((o) => o.family === filter);

  // Scan + journal are side effects: deferred so the compiler never sees setState in render.
  useEffect(() => {
    const t = setTimeout(() => { void scanWatchlist(); }, 0);
    return () => clearTimeout(t);
  }, [scanWatchlist]);

  useEffect(() => {
    const t = setTimeout(() => {
      setJournalDate(new Date().toISOString().slice(0, 10));
      fetch(`/api/signal-journal?symbol=${encodeURIComponent(ticker)}`)
        .then((r) => r.json() as Promise<{ records?: SignalJournalRecord[] }>)
        .then((d) => setJournal(Array.isArray(d?.records) ? d.records : []))
        .catch(() => setJournal([]));
    }, 0);
    return () => clearTimeout(t);
  }, [ticker, journal.length]);

  const journalRecords = useMemo(
    () => journalDate
      ? toJournalRecords(opportunities, journalDate, ticker, {
          iv30, spot: inputs.spot, trap: inputs.trap, expectedMove, crossSectional,
        })
      : [],
    [journalDate, opportunities, ticker, iv30, inputs.spot, inputs.trap, expectedMove, crossSectional]
  );

  useEffect(() => {
    const t = setTimeout(() => {
      fetch(`/api/signal-journal?symbol=${encodeURIComponent(ticker)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ records: journalRecords }),
      }).then(() => {
        fetch(`/api/signal-journal?symbol=${encodeURIComponent(ticker)}`)
          .then((r) => r.json() as Promise<{ records?: SignalJournalRecord[] }>)
          .then((d) => setJournal(Array.isArray(d?.records) ? d.records : []))
          .catch(() => undefined);
      }).catch(() => undefined);
    }, 0);
    return () => clearTimeout(t);
  }, [ticker, journalRecords]);

  const hitRates = useMemo<HitRate[]>(() => summarizeHitRates(journal, s.allStockBars), [journal, s.allStockBars]);
  const counts = useMemo(() => ({
    actionable: opportunities.filter((o) => o.state === 'ACTIONABLE').length,
    watch: opportunities.filter((o) => o.state === 'WATCH').length,
    blocked: opportunities.filter((o) => o.state === 'BLOCKED').length,
  }), [opportunities]);

  const jump = (o: Opportunity) => (
    <button
      type="button"
      onClick={() => onNavigate(o.evidence.view as View, o.evidence.params)}
      className="inline-flex cursor-pointer items-center gap-1 font-mono text-[10px] font-semibold text-gold-700 hover:underline dark:text-gold-300"
    >
      {o.evidence.label} <ArrowUpRight className="h-3 w-3" />
    </button>
  );

  return (
    <div className="space-y-4">
      <section className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="kicker">Edge · {companyName}</span>
              <ProvenanceBadge daily={s.barsSource} intraday={s.intradaySource} />
              {s.earnings && (
                <span className="font-mono text-[9px] text-zinc-400">
                  earnings {s.earnings.date} · {s.earnings.sessions}d{s.earnings.isEstimate ? ' (estimate)' : ''}
                </span>
              )}
            </div>
            <h1 className="font-display text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              {ticker} <span className="font-mono text-base font-semibold tabular-nums text-zinc-500">{pct(s.chain?.quote?.price ?? null)}</span>
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
              Six pre-registered setups ranked from the terminal&apos;s existing signals. Every card shows its trigger, invalidation, evidence and data gates. These are studies of one operator&apos;s rules — not orders or advice.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 font-mono text-[10px] text-zinc-500">
              <label htmlFor="risk-budget">Risk budget</label>
              <input
                id="risk-budget"
                type="range" min="0.25" max="5" step="0.25"
                value={riskBudget * 100}
                onChange={(e) => setRiskBudget(Number(e.target.value) / 100)}
                className="h-1 w-24 accent-gold-500"
              />
              <span className="tabular-nums text-zinc-800 dark:text-zinc-100">{(riskBudget * 100).toFixed(2)}%</span>
            </div>
            <button
              type="button"
              onClick={s.refresh}
              disabled={s.chainLoading}
              className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[10px] font-semibold disabled:opacity-60"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${s.chainLoading ? 'animate-spin' : ''}`} /> Refresh data
            </button>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-[var(--hairline)] pt-3 font-mono text-[10px]">
          <span className="text-emerald-600 dark:text-emerald-400">{counts.actionable} actionable</span>
          <span className="text-gold-700 dark:text-gold-300">{counts.watch} watch</span>
          <span className="text-rose-500 dark:text-rose-400">{counts.blocked} blocked</span>
          <span className="text-zinc-400">IV30 {pct(iv30)} · RV20 {pct(rv20)} · VRP {vrp.variancePoints === null ? '—' : `${vrp.variancePoints.toFixed(0)} bp²`}</span>
        </div>
      </section>

      <section className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><Target className="h-3.5 w-3.5" /> Opportunity Rank</p>
          <div className="flex flex-wrap gap-1">
            {['all', ...Object.keys(FAMILY_LABEL)].map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={`cursor-pointer rounded-md px-2 py-1 font-mono text-[9px] font-semibold ${filter === f ? 'bg-gold-500 text-zinc-950' : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'}`}
              >
                {f === 'all' ? 'All' : FAMILY_LABEL[f]}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          {visible.map((o) => (
            <article key={o.id} className="panel-inset flex flex-col gap-2 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">{o.title}</p>
                  <p className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{o.horizon} · {o.confidence} confidence · {o.staleness} quotes</p>
                </div>
                <span className={`shrink-0 rounded-md border px-2 py-1 font-mono text-[9px] font-semibold ${STATE_STYLE[o.state]}`}>{o.state.replace('_', ' ')}</span>
              </div>
              <p className="text-[11px] leading-relaxed text-zinc-600 dark:text-zinc-300">{o.thesis}</p>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px]">
                <div><dt className="text-zinc-400">Trigger · {o.trigger.metric}</dt><dd className="text-zinc-800 dark:text-zinc-100">{o.trigger.measured ?? '—'}</dd></div>
                <div><dt className="text-zinc-400">Edge</dt><dd className="text-zinc-800 dark:text-zinc-100">{o.edge.estimate ?? '—'} · n={o.edge.n ?? '—'}</dd></div>
                <div className="col-span-2"><dt className="text-zinc-400">Invalidation</dt><dd className="text-zinc-700 dark:text-zinc-200">{o.invalidation}</dd></div>
                <div className="col-span-2"><dt className="text-zinc-400">Scenario size</dt><dd className="text-zinc-700 dark:text-zinc-200">{o.size.scenario}</dd></div>
              </dl>
              {o.gates.length > 0 && (
                <div className="rounded border border-rose-500/30 bg-rose-500/5 px-2 py-1 font-mono text-[9px] text-rose-500 dark:text-rose-400">
                  gates: {o.gates.join(' · ')}
                </div>
              )}
              <div className="mt-auto flex items-center justify-between gap-2 border-t border-[var(--hairline)] pt-2">
                <span className="font-mono text-[9px] text-zinc-400">{o.edge.provenance}</span>
                {jump(o)}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <p className="kicker">Radar · One Scan</p>
            <button type="button" onClick={() => void s.scanWatchlist(true)} className="cursor-pointer font-mono text-[10px] font-semibold text-gold-700 hover:underline dark:text-gold-300">Re-scan</button>
          </div>
          {s.watchRows.length === 0 ? (
            <EmptyState title="Watchlist not scanned" detail="One pass evaluates IV and dealer state for six names" />
          ) : (
            <table className="table-sticky-first w-full border-collapse font-mono text-[10px]">
              <thead><tr className="text-left text-[9px] uppercase tracking-wider text-zinc-400"><th className="py-1">Name</th><th className="py-1 text-right">IV30</th><th className="py-1 text-right">IV−RV</th><th className="py-1 text-right">Trap</th></tr></thead>
              <tbody>
                {s.watchRows.map((w) => {
                  const r20 = crossSectional.find((c) => c.symbol === w.symbol)?.rv20 ?? null;
                  const gap = w.iv30 !== null && r20 !== null ? (w.iv30 - r20) * 100 : null;
                  return (
                    <tr key={w.symbol} className="border-t border-[var(--hairline)]">
                      <td className="py-1 font-semibold">{w.symbol}</td>
                      <td className="py-1 text-right tabular-nums">{pct(w.iv30)}</td>
                      <td className="py-1 text-right tabular-nums">{gap === null ? '—' : `${gap.toFixed(1)} pts`}</td>
                      <td className="py-1 text-right text-zinc-500">{w.trapState?.toUpperCase() ?? '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="mt-2 font-mono text-[9px] text-zinc-400">Single-day IV/RV snapshot — no per-name IV rank until each symbol accrues ~60 sessions.</p>
        </div>

        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <p className="kicker flex items-center gap-2"><ShieldAlert className="h-3.5 w-3.5" /> Signal Journal</p>
            <span className="font-mono text-[9px] text-zinc-400">{journal.length} snapshots · {hitRates.reduce((n, h) => n + h.n, 0)} resolved</span>
          </div>
          {hitRates.length === 0 ? (
            <EmptyState title="No resolved setups yet" detail="Each board load records a daily snapshot; outcomes resolve from forward bars" />
          ) : (
            <table className="table-sticky-first w-full border-collapse font-mono text-[10px]">
              <thead><tr className="text-left text-[9px] uppercase tracking-wider text-zinc-400"><th className="py-1">Setup</th><th className="py-1 text-right">Win</th><th className="py-1 text-right">Loss</th><th className="py-1 text-right">Open</th><th className="py-1 text-right">Hit rate</th></tr></thead>
              <tbody>
                {hitRates.map((h) => (
                  <tr key={h.setup} className="border-t border-[var(--hairline)]">
                    <td className="py-1">{FAMILY_LABEL[h.setup] ?? h.setup}</td>
                    <td className="py-1 text-right tabular-nums text-emerald-600 dark:text-emerald-400">{h.wins}</td>
                    <td className="py-1 text-right tabular-nums text-rose-500">{h.losses}</td>
                    <td className="py-1 text-right tabular-nums text-zinc-400">{h.unresolved}</td>
                    <td className="py-1 text-right tabular-nums">{h.hitRate === null ? 'insufficient history' : `${(h.hitRate * 100).toFixed(0)}% (n=${h.n})`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
            One resolved outcome per recorded signal, fixed before observing the result. Rates below {10} resolved signals are withheld. Records accrue in the local signal journal, never rewritten.
          </p>
        </div>
      </section>

      <section className="panel p-5">
        <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
          <p className="kicker flex items-center gap-2"><CircleDot className="h-3.5 w-3.5" /> Tape &amp; Event Context</p>
          <button type="button" onClick={() => onNavigate('analysis')} className="cursor-pointer font-mono text-[10px] font-semibold text-gold-700 hover:underline dark:text-gold-300">Analysis →</button>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {tapeItems.map((t) => (
            <div key={t.symbol} className="panel-inset p-2">
              <div className="font-mono text-[10px] font-semibold">{t.symbol}</div>
              <div className="font-mono text-[11px] tabular-nums text-zinc-700 dark:text-zinc-200">{t.price.toFixed(2)}</div>
              <div className={`font-mono text-[9px] tabular-nums ${t.changePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>{t.changePct >= 0 ? '+' : ''}{t.changePct.toFixed(2)}%</div>
            </div>
          ))}
        </div>
        <p className="mt-2 font-mono text-[9px] text-zinc-400">
          Earnings date is shown for the active ticker only; watchlist catalysts require a per-name calendar fetch and are deliberately not fabricated here.
        </p>
      </section>
    </div>
  );
}
