'use client';

import React from 'react';
import {
  Gauge,
  TrendingUp,
  Activity,
  ShieldAlert,
  Landmark,
  Sigma,
  Info,
} from 'lucide-react';
import type { ChainPayload, IndicesPayload } from './types';
import { percentileRank } from '@/lib/vol-analytics';

interface T1Props {
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  rv20: number | null; // annualized realized vol, decimal
}

const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(digits)}%`;

export function T1StateTab({ chain, indices, rv20 }: T1Props) {
  const slices = chain?.slices ?? [];
  const headline = slices.find((s) => s.dte >= 21) ?? slices[0] ?? null;
  const front21 = slices.find((s) => s.dte >= 15 && s.dte <= 45) ?? headline;

  const iv30 = chain?.iv30 ?? headline?.atmIv ?? null;

  const vixNow = indices?.series?.VIX?.bars?.[indices.series.VIX.bars.length - 1]?.close ?? null;
  const vix9d = indices?.series?.VIX9D?.bars?.[indices.series.VIX9D.bars.length - 1]?.close ?? null;
  const vix3m = indices?.series?.VIX3M?.bars?.[indices.series.VIX3M.bars.length - 1]?.close ?? null;
  const vxnNow = indices?.series?.VXN?.bars?.[indices.series.VXN.bars.length - 1]?.close ?? null;
  const vvixNow = indices?.series?.VVIX?.bars?.[indices.series.VVIX.bars.length - 1]?.close ?? null;
  const skewNow = indices?.series?.SKEW?.bars?.[indices.series.SKEW.bars.length - 1]?.close ?? null;
  const vixBars = indices?.series?.VIX?.bars ?? [];
  const skewBars = indices?.series?.SKEW?.bars ?? [];
  const vix1y = vixBars.slice(-252).map((b) => b.close);
  const skew1y = skewBars.slice(-252).map((b) => b.close);
  const vixPercentile = vixNow !== null && vix1y.length > 30 ? percentileRank(vix1y, vixNow) : null;
  const skewPercentile = skewNow !== null && skew1y.length > 30 ? percentileRank(skew1y, skewNow) : null;

  // Slope: 30d chain IV vs 90d slice, plus VIX9D/VIX3M short-end
  const term = chain?.termStructure ?? [];
  const frontIv = term.find((t) => t.dte >= 7)?.iv ?? null;
  const backIv = [...term].reverse().find((t) => t.dte >= 60)?.iv ?? null;
  const slopeRatio =
    frontIv !== null && frontIv !== undefined && backIv !== null && backIv !== undefined && backIv > 0
      ? frontIv / backIv
      : null;
  const vixRatio = vixNow !== null && vix3m !== null && vix3m > 0 ? vixNow / vix3m : null;
  const verdict: 'INVERTED' | 'FLAT' | 'CONTANGO' =
    vixRatio === null ? 'FLAT' : vixRatio > 1.04 ? 'INVERTED' : vixRatio < 0.96 ? 'CONTANGO' : 'FLAT';

  const rr25 = front21?.rr25 ?? null;
  const bf25 = front21?.bf25 ?? null;
  const skew9010 = front21?.skew9010 ?? null;

  // The Gap
  const vrp = iv30 !== null && rv20 !== null ? iv30 * iv30 - rv20 * rv20 : null;
  const ivPremiumPts = iv30 !== null && rv20 !== null ? (iv30 - rv20) * 100 : null;

  // Priced distribution
  const band = chain?.pricedBand ?? null;
  const probDrawdown15 = band?.probDrawdown15 ?? null;
  const mcap = chain?.quote?.marketCap ?? null;

  // Surface quality
  const totalUsed = slices.reduce((s, x) => s + x.clean.used, 0);
  const totalContracts = slices.reduce((s, x) => s + x.clean.total, 0);
  const medianSpreads = slices.map((s) => s.clean.medianRelSpread).filter((v): v is number => v !== null);
  const medianSpread = medianSpreads.length > 0 ? medianSpreads.sort((a, b) => a - b)[Math.floor(medianSpreads.length / 2)] : null;

  // IV rank accrual
  const snap = chain?.snapshot;
  const accrualCount = snap?.iv30HistoryCount ?? 0;
  const accrualTarget = 60;
  const ivRankReady = accrualCount >= accrualTarget;

  const proxyNote = chain?.substitution ?? null;
  const requestedTicker = chain?.requested ?? chain?.symbol ?? '';

  return (
    <div className="space-y-4">
      {/* Substitution / proxy badge row */}
      {proxyNote && (
        <div className="flex items-center gap-2 rounded-lg border border-gold-500/40 bg-gold-500/10 px-3 py-2 text-[11px] font-medium text-gold-800 dark:text-gold-200">
          <Info className="h-3.5 w-3.5 shrink-0" />
          <span>{requestedTicker}: {proxyNote}. IV numbers refer to the proxy.</span>
        </div>
      )}

      {/* Axis 1 — LEVEL */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="panel-inset p-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            <Gauge className="h-3.5 w-3.5" /> IV-30 Level
          </div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {pct(iv30)}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
            {chain?.iv30Source === 'term-structure interpolation (w = σ²T)' ? 'w-interpolated from term structure' : 'ATM IV, nearest 30d slice'} · European BSM
          </div>
        </div>

        <div className="panel-inset p-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            <TrendingUp className="h-3.5 w-3.5" /> IV Rank · 52w
          </div>
          {ivRankReady ? (
            <>
              <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
                {pct(percentileRank(chain?.ivHistory30 ?? [], iv30 ?? 0), 0)}
              </div>
              <div className="mt-1 text-[10px] font-mono text-zinc-400">percentile of accrued history ({accrualCount} sessions)</div>
            </>
          ) : (
            <>
              <div className="mt-1.5 font-mono text-sm font-semibold text-gold-700 dark:text-gold-300">
                Accruing {accrualCount}/{accrualTarget}d
              </div>
              <div className="mt-1 text-[10px] font-mono text-zinc-400">
                IV rank unlocks after {accrualTarget} accrued sessions (since {snap?.firstDate ?? 'today'}). History cannot be backfilled.
              </div>
            </>
          )}
        </div>

        <div className="panel-inset p-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            <Landmark className="h-3.5 w-3.5" /> VIX Context
          </div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {vixNow?.toFixed(2) ?? '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
            {vixPercentile !== null ? `${vixPercentile.toFixed(0)}th pctile of 1y · VXN ${vxnNow?.toFixed(1) ?? '—'} · VVIX ${vvixNow?.toFixed(0) ?? '—'}` : 'index vol context'}
          </div>
        </div>

        <div className="panel-inset p-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            <ShieldAlert className="h-3.5 w-3.5" /> ^SKEW Tail Pricing
          </div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {skewNow?.toFixed(0) ?? '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
            {skewPercentile !== null ? `${skewPercentile.toFixed(0)}th pctile of 1y — crash insurance pricing` : 'CBOE tail-risk index'}
          </div>
        </div>
      </div>

      {/* Axis 2 — SLOPE */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="panel-inset p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">30d / 90d IV Ratio</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {slopeRatio !== null ? slopeRatio.toFixed(2) : '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">term structure of {requestedTicker || 'chain'} IV</div>
        </div>
        <div className="panel-inset p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">^VIX / ^VIX3M</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {vixRatio !== null ? vixRatio.toFixed(2) : '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">short vs long end (9d: {vix9d?.toFixed(1) ?? '—'})</div>
        </div>
        <div className="panel-inset col-span-2 p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">Curve Verdict</div>
          <div className="mt-1.5 flex items-center gap-3">
            <span
              className={`rounded-md px-3 py-1 font-mono text-lg font-bold tracking-wider ${
                verdict === 'INVERTED'
                  ? 'bg-rose-500/15 text-rose-500 dark:text-rose-400'
                  : verdict === 'CONTANGO'
                    ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                    : 'bg-zinc-200/60 dark:bg-white/[0.05] text-zinc-500 dark:text-zinc-400'
              }`}
            >
              {verdict}
            </span>
            <span className="text-[10px] leading-snug text-zinc-400 dark:text-zinc-500">
              {verdict === 'INVERTED'
                ? 'Short-dated IV richer than long-dated — classic backwardation. Strongest single stress signal on the board.'
                : verdict === 'CONTANGO'
                  ? 'Long-dated IV richer than short — market calm now, expects turbulence later. Normal positive-carry regime.'
                  : 'Curve flat — no strong term-structure signal either way.'}
            </span>
          </div>
        </div>
      </div>

      {/* Axis 3 — SHAPE */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="panel-inset p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">25Δ Risk Reversal</div>
          <div className={`mt-1.5 font-mono text-2xl font-semibold tabular-nums ${rr25 !== null && rr25 < 0 ? 'text-gold-700 dark:text-gold-300' : 'text-emerald-600 dark:text-emerald-400'}`}>
            {rr25 !== null ? `${(rr25 * 100).toFixed(2)} pts` : '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">σ(25Δ call) − σ(25Δ put) · negative = puts bid</div>
        </div>
        <div className="panel-inset p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">25Δ Butterfly</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {bf25 !== null ? `${bf25 >= 0 ? '+' : ''}${(bf25 * 100).toFixed(2)} pts` : '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">wing demand / smile curvature pricing</div>
        </div>
        <div className="panel-inset p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">Skew 90/10</div>
          <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums text-gold-700 dark:text-gold-300">
            {skew9010 !== null ? `+${(skew9010 * 100).toFixed(2)} pts` : '—'}
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">σ(10Δ put) − σ(10Δ call) · crash tilt</div>
        </div>
        <div className="panel-inset p-4">
          <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
            <Sigma className="h-3.5 w-3.5" /> Surface Quality
          </div>
          <div className="mt-1.5 font-mono text-lg font-semibold tabular-nums text-zinc-900 dark:text-zinc-50 leading-tight">
            {totalUsed}<span className="text-zinc-400"> / {totalContracts}</span>
          </div>
          <div className="mt-1 text-[10px] font-mono text-zinc-400">
            contracts used{medianSpread !== null ? ` · median spread ${(medianSpread * 100).toFixed(1)} vol` : ''} · BSM approx
          </div>
        </div>
      </div>

      {/* Axis 4 — THE GAP (VRP) */}
      <div className="panel p-5">
        <div className="kicker mb-2 flex items-center gap-2">
          <Activity className="h-3.5 w-3.5" /> The Gap — Implied vs Realized
        </div>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="font-mono text-sm text-zinc-600 dark:text-zinc-300">
            Market is charging <strong className="text-gold-700 dark:text-gold-300">{pct(iv30)}</strong> for the next 30 days against{' '}
            <strong className="text-zinc-900 dark:text-zinc-50">{pct(rv20)}</strong> realized —{' '}
            {ivPremiumPts !== null ? (
              <>
                a{' '}
                <strong className={ivPremiumPts >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}>
                  {Math.abs(ivPremiumPts).toFixed(1)}pt {ivPremiumPts >= 0 ? 'premium' : 'discount'}
                </strong>
              </>
            ) : (
              'an unresolved premium'
            )}.
            {vrp !== null && (
              <span className="block mt-1 text-[11px] text-zinc-400">
                VRP = {(vrp * 1e4).toFixed(0)} variance points/yr² · the compensation for writing variance — positive, small, and negatively skewed.
              </span>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-3 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-4 py-3 font-mono text-[11px]">
            <div>
              <div className="text-[9px] uppercase tracking-wider text-zinc-400">IV30² − RV20²</div>
              <div className={`mt-0.5 text-base font-bold ${vrp !== null && vrp >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                {vrp !== null ? `${vrp >= 0 ? '+' : ''}${(vrp * 1e4).toFixed(0)} var pts` : '—'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Priced distribution */}
      {band && (
        <div className="panel p-5">
          <div className="kicker mb-2 flex items-center gap-2">
            <Gauge className="h-3.5 w-3.5" /> What the Market Prices
          </div>
          <p className="text-[12px] leading-relaxed text-zinc-600 dark:text-zinc-300">
            Over the coming <strong>{headline?.dte ?? 30} days</strong> the market prices a{' '}
            <strong className="text-zinc-900 dark:text-zinc-50">±{band.oneSigmaPct.toFixed(1)}%</strong> band (1σ, 68%),
            and a{' '}
            <strong className="text-rose-500 dark:text-rose-400">{probDrawdown15?.toFixed(1)}%</strong> risk-neutral
            probability of a &gt;15% drawdown{mcap ? ` (market cap ${(mcap / 1e9).toFixed(0)}B)` : ''}.{' '}
            Touch probability of −10% ≈ <strong>{band.probTouch10.toFixed(1)}%</strong>.
          </p>
          <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
            Risk-neutral N(−d2) · European BSM approximation · not investment advice
          </p>
        </div>
      )}
    </div>
  );
}
