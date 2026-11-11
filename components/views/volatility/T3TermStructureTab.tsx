'use client';

import React, { useMemo, useState } from 'react';
import { TrendingUp, LineChart, ShieldAlert, Sigma, AlertTriangle } from 'lucide-react';
import {
  calculateReturns,
  fitGarch11,
  forecastGarch,
} from '@/lib/quant-engine';
import type { PriceBar } from '@/lib/quant-engine';
import { percentileRank, quantileSorted } from '@/lib/vol-analytics';
import { VolTermStructureChart } from '@/components/charts/VolTermStructureChart';
import { IvRvChart } from '@/components/charts/IvRvChart';
import type { ChainPayload, IndicesPayload } from './types';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

export function T3TermStructureTab({
  chain,
  indices,
  bars,
  ticker,
}: {
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  bars: PriceBar[];
  ticker: string;
}) {
  const [axisMode, setAxisMode] = useState<'vol' | 'totalVariance'>('vol');

  const slices = useMemo(() => chain?.slices ?? [], [chain]);
  const tsPoints = useMemo(() => chain?.termStructure ?? [], [chain]);

  // --- RV20 series for the active ticker -----------------------------------
  const { rvSeries, returns, dates } = useMemo(() => {
    if (!bars || bars.length < 40) {
      return { rvSeries: [] as { date: string; vol: number }[], returns: [] as number[], dates: [] as string[] };
    }
    const closes = bars.map((b) => b.close);
    const rets = calculateReturns(closes, 'log');
    const ds = bars.slice(1).map((b) => b.date);
    const out: { date: string; vol: number }[] = [];
    for (let i = 19; i < rets.length; i++) {
      let sum = 0;
      for (let j = i - 19; j <= i; j++) sum += rets[j] * rets[j];
      const ann = Math.sqrt((sum / 20) * 252);
      out.push({ date: ds[i], vol: ann });
    }
    return { rvSeries: out, returns: rets, dates: ds };
  }, [bars]);

  const rv20 = rvSeries.length > 0 ? rvSeries[rvSeries.length - 1].vol : null;

  // --- IV proxy series (indices) --------------------------------------------
  const vix = indices?.series?.VIX?.bars ?? [];
  const vxn = indices?.series?.VXN?.bars ?? [];
  const ivProxySeries = ticker === 'QQQ' ? vxn : vix;
  const ivProxyLabel = ticker === 'QQQ' ? '^VXN (QQQ vol)' : '^VIX (market proxy)';

  const ivRvPoints = useMemo(() => {
    const vixMap = new Map(ivProxySeries.map((b) => [b.date, b.close / 100]));
    return rvSeries
      .filter((r) => vixMap.has(r.date))
      .slice(-504)
      .map((r) => ({ date: r.date, iv: vixMap.get(r.date) ?? null, rv: r.vol }));
  }, [rvSeries, ivProxySeries]);

  // --- Term-structure band from the proxy's 1y history -----------------------
  const band = useMemo(() => {
    const hist = ivProxySeries.slice(-252).map((b) => b.close).sort((a, b) => a - b);
    if (hist.length < 30) return null;
    return {
      low: quantileSorted(hist, 0.1),
      high: quantileSorted(hist, 0.9),
      label: `^VIX 1y 10–90 pctile band (${quantileSorted(hist, 0.1).toFixed(1)}–${quantileSorted(hist, 0.9).toFixed(1)})`,
    };
  }, [ivProxySeries]);

  // --- VRP by tenor ----------------------------------------------------------
  const garchFit = useMemo(() => (returns.length > 50 ? fitGarch11(returns) : null), [returns]);
  const garchPath = useMemo(() => {
    if (!garchFit) return null;
    const latestVar = garchFit.conditionalVariance[garchFit.conditionalVariance.length - 1] ?? 0.0002;
    return forecastGarch(garchFit.summary, latestVar, 120);
  }, [garchFit]);

  const vrpByTenor = useMemo(() => {
    if (!rv20) return [];
    return slices.map((s) => {
      const iv2 = (s.atmIv ?? 0) ** 2;
      const vsRv = iv2 - rv20 * rv20;
      let vsGarch: number | null = null;
      if (garchPath && garchPath.length >= s.dte) {
        const pathVars = garchPath.slice(0, s.dte).map((p) => p.annualizedVol * p.annualizedVol);
        const pathMean = pathVars.reduce((a, b) => a + b, 0) / pathVars.length;
        vsGarch = iv2 - pathMean;
      }
      return { dte: s.dte, iv2, vsRv, vsGarch };
    });
  }, [slices, rv20, garchPath]);

  // --- Short-vol carry diagnostic (honest backtest) ---------------------------
  const carry = useMemo(() => {
    if (ivProxySeries.length < 300) return null;
    const vixMap = new Map(ivProxySeries.map((b) => [b.date, b.close / 100]));
    let mean = 0;
    let wins = 0;
    let n = 0;
    let worst = Infinity;
    let cumulative = 0;
    const curve: number[] = [];
    for (let i = 21; i < dates.length - 22; i++) {
      const d = dates[i];
      const vixT = vixMap.get(d);
      if (vixT === undefined) continue;
      // RV over the following 21 sessions
      let fwd = 0;
      for (let j = i + 1; j <= i + 21 && j < returns.length + 1; j++) {
        if (j - 1 < returns.length) fwd += returns[j - 1] * returns[j - 1];
      }
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
    let peak = -Infinity;
    let maxDD = 0;
    for (const c of curve) {
      if (c > peak) peak = c;
      const dd = peak - c;
      if (dd > maxDD) maxDD = dd;
    }
    return { meanMonthly: mean, hitRate: (wins / n) * 100, worstMonth: worst, maxDrawdown: maxDD, n };
  }, [ivProxySeries, returns, dates]);

  const iv30 = chain?.iv30 ?? null;
  const rvAnnPct = rv20 !== null ? rv20 * 100 : null;
  const invert = (chain?.termStructure ?? []).length >= 2 &&
    ((chain?.termStructure?.[0]?.iv ?? 0) > ([...(chain?.termStructure ?? [])].reverse().find((t) => (t.dte ?? 0) > 45)?.iv ?? 0));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        {/* Term structure chart */}
        <div className="panel p-5 xl:col-span-3">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2">
                <TrendingUp className="h-3.5 w-3.5" /> Term Structure
              </p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                IV bytenor: {iv30 !== null ? `IV30 ${pct(iv30)}` : 'IV curve'}{invert ? ' · curve inverted' : ''}
              </h3>
            </div>
            <div className="inline-flex items-center gap-1 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-1">
              {(['vol', 'totalVariance'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setAxisMode(m)}
                  className={`cursor-pointer rounded-md px-2 py-1 font-mono text-[10px] font-semibold transition-colors ${
                    axisMode === m ? 'bg-gold-500 text-zinc-950' : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                  }`}
                >
                  {m === 'vol' ? 'σ axis' : 'w = σ²T axis'}
                </button>
              ))}
            </div>
          </div>
          <VolTermStructureChart
            points={tsPoints}
            axisMode={axisMode}
            band={band}
            iv30Marker={30}
          />
          {invert && (
            <div className="mt-2 flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-[11px] font-medium text-rose-500 dark:text-rose-400">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              Curve inverted: front-end IV richer than long-dated — the strongest single stress signal available today.
            </div>
          )}
        </div>

        {/* VRP by tenor */}
        <div className="panel p-5 xl:col-span-2">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1 flex items-center gap-2">
              <Sigma className="h-3.5 w-3.5" /> VRP by Tenor
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Variance premium, per horizon
            </h3>
          </div>
          <div className="space-y-1.5 font-mono text-[10px]">
            <div className="grid grid-cols-4 gap-2 text-[8.5px] uppercase tracking-wider text-zinc-400">
              <span>Tenor</span><span className="text-right">IV²</span><span className="text-right">vs RV20</span><span className="text-right">vs GARCH path</span>
            </div>
            {vrpByTenor.length === 0 && <div className="text-zinc-400">RV history unavailable</div>}
            {vrpByTenor.map((v) => (
              <div key={v.dte} className="grid grid-cols-4 gap-2 border-t border-[var(--hairline)] py-1">
                <span className="text-zinc-500 dark:text-zinc-400">{v.dte}d</span>
                <span className="text-right tabular-nums text-zinc-700 dark:text-zinc-200">{(v.iv2 * 1e4).toFixed(0)}</span>
                <span className={`text-right tabular-nums ${v.vsRv >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                  {v.vsRv >= 0 ? '+' : ''}{(v.vsRv * 1e4).toFixed(0)}
                </span>
                <span className={`text-right tabular-nums ${v.vsGarch !== null && v.vsGarch >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                  {v.vsGarch === null ? '—' : `${v.vsGarch >= 0 ? '+' : ''}${(v.vsGarch * 1e4).toFixed(0)}`}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
            variance points (×10⁴), annualized · humped and positive is the normal state · collapsing to zero at the front end = market underpricing immediate risk
          </p>
        </div>
      </div>

      {/* IV vs RV dual panel */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <LineChart className="h-3.5 w-3.5" /> Implied vs Realized
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              The 2-year tape of the premium
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            IV: {ivProxyLabel} · RV: ticker 20d
          </span>
        </div>
        <IvRvChart points={ivRvPoints} ivLabel={ivProxyLabel} />
      </div>

      {/* Short-vol carry with its tail risk shown */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <ShieldAlert className="h-3.5 w-3.5" /> Short-Vol Carry, Honestly Priced
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Simulated short-variance P&amp;L — 21-session windows, {carry?.n ?? 0} samples
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            ^VIX as IV proxy · ticker RV · not executable
          </span>
        </div>
        {carry ? (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="panel-inset p-3">
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Mean / Window</div>
                <div className="mt-1 font-mono text-lg font-bold text-emerald-600 dark:text-emerald-400">
                  +{(carry.meanMonthly * 1e4).toFixed(0)} bp²
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Hit Rate</div>
                <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">
                  {carry.hitRate.toFixed(0)}%
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Worst Window</div>
                <div className="mt-1 font-mono text-lg font-bold text-rose-500 dark:text-rose-400">
                  {(carry.worstMonth * 1e4).toFixed(0)} bp²
                </div>
              </div>
              <div className="panel-inset p-3">
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Max Drawdown</div>
                <div className="mt-1 font-mono text-lg font-bold text-rose-500 dark:text-rose-400">
                  −{(carry.maxDrawdown * 1e4).toFixed(0)} bp²
                </div>
              </div>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
              The premium pays small and often ({carry.hitRate.toFixed(0)}% of windows), and loses large and rarely (worst window{' '}
              {(carry.worstMonth * 1e4).toFixed(0)} bp² vs mean +{(carry.meanMonthly * 1e4).toFixed(0)} bp²). It is not free money — it is
              compensation for underwriting gap risk, and the expected loss clusters exactly when everyone else is short too.
            </p>
          </>
        ) : (
          <div className="text-xs text-zinc-400">Not enough history to score carry honestly (need ≥ 300 aligned sessions).</div>
        )}
        {rvAnnPct !== null && (
          <div className="mt-3 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            ticker RV(20d, ann): {rvAnnPct.toFixed(1)}% · ^VIX percentile 1y: {(() => {
              const h = ivProxySeries.slice(-252).map((b) => b.close);
              return h.length > 30 ? `${percentileRank(h, ivProxySeries[ivProxySeries.length - 1].close)?.toFixed(0)}th` : '—';
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
