'use client';

import React, { useMemo, useState } from 'react';
import { Radiation, Ruler, Landmark, Info } from 'lucide-react';
import type { ChainPayload } from './types';
import { RndChart } from '@/components/charts/RndChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { normCdf } from '@/lib/options';
import { calculateReturns } from '@/lib/quant-engine';
import type { PriceBar } from '@/lib/quant-engine';

/** Implied (RND) vs realized moments of horizon returns. */
function computeMoments(
  rnd: ChainPayload['rnd'] | null,
  spot: number,
  realizedHorizon: number[]
) {
  if (!rnd || rnd.strikes.length < 10) return null;
  const ks = rnd.strikes;
  const density = rnd.density;
  const dK = ks[1] - ks[0];
  let area = 0;
  let meanAcc = 0;
  for (let i = 0; i < ks.length - 1; i++) {
    const w = ((density[i] + density[i + 1]) / 2) * dK;
    area += w;
    meanAcc += w * (ks[i] + dK / 2);
  }
  const mean = area > 0 ? meanAcc / area : 0;
  let varAcc = 0;
  let skewAcc = 0;
  let kurtAcc = 0;
  for (let i = 0; i < ks.length - 1; i++) {
    const w = ((density[i] + density[i + 1]) / 2) * dK;
    const mid = ks[i] + dK / 2;
    const diff = mid - mean;
    varAcc += w * diff * diff;
    skewAcc += w * Math.pow(diff, 3);
    kurtAcc += w * Math.pow(diff, 4);
  }
  const variance = area > 0 ? varAcc / area : 0;
  const std = Math.sqrt(variance);
  const skew = variance > 0 ? skewAcc / area / Math.pow(variance, 1.5) : 0;
  const kurt = variance > 0 ? kurtAcc / area / (variance * variance) - 3 : 0;
  // Express dispersion on the same return axis as realized moments
  const expRet = spot > 0 && mean > 0 ? Math.log(mean / spot) : 0;
  const stdLog = spot > 0 ? std / spot : 0;

  let rMean = 0;
  for (const r of realizedHorizon) rMean += r;
  rMean /= Math.max(1, realizedHorizon.length);
  let rVar = 0;
  let rSkew = 0;
  let rKurt = 0;
  for (const r of realizedHorizon) {
    const d = r - rMean;
    rVar += d * d;
    rSkew += d * d * d;
    rKurt += d * d * d * d;
  }
  const rN = Math.max(1, realizedHorizon.length);
  rVar /= rN;
  const rStd = Math.sqrt(rVar);
  rSkew = rVar > 0 ? rSkew / rN / Math.pow(rVar, 1.5) : 0;
  rKurt = rVar > 0 ? rKurt / rN / (rVar * rVar) - 3 : 0;

  return {
    implied: { expRet, std: stdLog, skew, kurt },
    realized: { mean: rMean, std: rStd, skew: rSkew, kurt: rKurt },
  };
}

export function T5DensityTab({
  chain,
  bars,
  ticker,
}: {
  chain: ChainPayload | null;
  bars: PriceBar[];
  ticker: string;
}) {
  const [debtB, setDebtB] = useState<number | null>(null);

  const slices = useMemo(() => chain?.slices ?? [], [chain]);
  const spot = chain?.quote?.price ?? 0;
  const atmIv = chain?.iv30 ?? slices.find((s) => s.dte >= 21)?.atmIv ?? 0.2;
  const headline = slices.find((s) => s.dte >= 21) ?? slices[0] ?? null;

  // Realized horizon distribution (overlapping 21d forward log returns)
  const realizedHorizon = useMemo(() => {
    const closes = (bars ?? []).map((b) => b.close);
    const rets = calculateReturns(closes, 'log');
    const h = 21;
    const out: number[] = [];
    for (let i = 0; i + h < rets.length; i++) {
      let s = 0;
      for (let j = i; j < i + h; j++) s += rets[j];
      out.push(s);
    }
    return out;
  }, [bars]);

  // RND moments vs realized moments
  const moments = useMemo(
    () => computeMoments(chain?.rnd ?? null, spot, realizedHorizon),
    [chain?.rnd, spot, realizedHorizon]
  );

  // MFIV vs ATM consistency table
  const mfivTable = useMemo(
    () =>
      slices.map((s) => ({
        dte: s.dte,
        mfiv: s.mfivAnnualized,
        atm: s.atmIv,
        diffPts: s.mfivAnnualized !== null && s.atmIv !== null ? (s.mfivAnnualized - s.atmIv) * 100 : null,
      })),
    [slices]
  );

  // Risk-neutral probabilities (from ATM IV, d2-based)
  const probs = useMemo(() => {
    const T = headline?.t ?? 30 / 365;
    const r = chain?.rate ?? 0.04;
    const q = chain?.dividendYield ?? 0;
    const n = normCdf;
    const out = (k: number) => {
      const sigT = atmIv * Math.sqrt(T);
      const d2 = (Math.log(spot / (spot * k)) + (r - q - 0.5 * atmIv * atmIv) * T) / sigT;
      return { above: n(d2), touch: Math.min(1, 2 * n(-Math.abs(d2))) };
    };
    return { T, deep: out(0.8), ten: out(0.9), up: out(1.1) };
  }, [headline, atmIv, chain, spot]);

  // Merton distance to default (naive, assumptions labelled)
  const mcap = chain?.quote?.marketCap ?? null;
  const defaultDebt = mcap !== null ? (mcap * 0.3) / 1e9 : 10;
  const effDebtB = debtB ?? defaultDebt;
  const equityB = mcap !== null ? mcap / 1e9 : 10;
  const assetValueB = equityB + effDebtB;
  const volAsset = atmIv * (equityB / Math.max(assetValueB, 1e-9)); // equity elasticity σ_A = σ_E·E/V approx
  const mertonDD = assetValueB > 0 ? (Math.log(assetValueB / Math.max(effDebtB, 1e-9)) + (0.04 - 0.5 * volAsset * volAsset) * 1) / (volAsset * 1) : 0;
  const mertonPD = normCdf(-mertonDD);

  return (
    <div className="space-y-4">
      {/* RND vs realized histogram */}
      <div className="panel p-5">
        <div className="mb-3 border-b border-[var(--hairline)] pb-3">
          <p className="kicker mb-1 flex items-center gap-2"><Radiation className="h-3.5 w-3.5" /> Market-Implied Distribution vs Realized</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            Breeden–Litzenberger density of the fitted smile vs realized 21d returns
          </h3>
        </div>
        {chain?.rnd ? (
          <ChartFullscreen title="Breeden–Litzenberger Risk-Neutral Density vs. Realized">
            <RndChart rnd={chain.rnd} spot={spot} realizedReturns={realizedHorizon} />
          </ChartFullscreen>
        ) : (
          <div className="text-xs text-zinc-400">No fitted smile available for the RND build.</div>
        )}
        <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
          The divergence between the two curves IS the risk premium made visible — the market&apos;s left tail sits materially to the left of the realized one, and that asymmetry is what crash insurance costs.
        </p>
      </div>

      {/* RND moments vs realized */}
      {moments && (
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Implied vs Realized Moments</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">The price of fear, in four numbers</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <th className="py-1.5">Moment</th>
                  <th className="py-1.5 text-right">Implied (RND)</th>
                  <th className="py-1.5 text-right">Realized (21d)</th>
                  <th className="py-1.5 text-right">Gap</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {[
                  { label: 'E[log-return to expiry]', imp: moments.implied.expRet, rl: moments.realized.mean, fmt: (v: number) => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%' },
                  { label: 'σ (dispersion)', imp: moments.implied.std, rl: moments.realized.std, fmt: (v: number) => (v * 100).toFixed(2) + '%' },
                  { label: 'Skew', imp: moments.implied.skew, rl: moments.realized.skew, fmt: (v: number) => v.toFixed(2) },
                  { label: 'Excess kurtosis', imp: moments.implied.kurt, rl: moments.realized.kurt, fmt: (v: number) => v.toFixed(2) },
                ].map(({ label, imp, rl, fmt }) => (
                  <tr key={label} className="border-t border-[var(--hairline)]">
                    <td className="py-1.5 text-zinc-600 dark:text-zinc-300">{label}</td>
                    <td className="py-1.5 text-right tabular-nums text-gold-800 dark:text-gold-200">{fmt(imp)}</td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{fmt(rl)}</td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-500">{(imp - rl >= 0 ? '+' : '') + (imp - rl).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* MFIV vs interpolated ATM */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1 flex items-center gap-2"><Ruler className="h-3.5 w-3.5" /> MFIV vs ATM Surface Check</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Model-free vs interpolation — consistency audit</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <th className="py-1.5">Tenor</th>
                  <th className="py-1.5 text-right">MFIV</th>
                  <th className="py-1.5 text-right">ATM IV</th>
                  <th className="py-1.5 text-right">Δ vol pts</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {mfivTable.map((r) => (
                  <tr key={r.dte} className="border-t border-[var(--hairline)]">
                    <td className="py-1.5 text-zinc-500">{r.dte}d</td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{r.mfiv !== null ? `${(r.mfiv * 100).toFixed(2)}%` : '—'}</td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{r.atm !== null ? `${(r.atm * 100).toFixed(2)}%` : '—'}</td>
                    <td className={`py-1.5 text-right tabular-nums ${r.diffPts !== null && Math.abs(r.diffPts) > 2 ? 'text-rose-500 dark:text-rose-400 font-bold' : 'text-zinc-500'}`}>
                      {r.diffPts !== null ? `${r.diffPts >= 0 ? '+' : ''}${r.diffPts.toFixed(2)}` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 font-mono text-[9px] text-zinc-400">|Δ| &gt; 2 vol pts flags an inconsistent slice (stale wings or thin coverage)</p>
        </div>

        {/* Digital probabilities */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Digital Probabilities</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">P(S<sub>T</sub> beyond K) = N(d2) · touch ≈ 2·N(−|d|)</h3>
          </div>
          <div className="space-y-2 font-mono text-[11px]">
            {[
              { label: 'P(S_T > +10%)', v: probs.up.above, invert: false },
              { label: 'P(S_T > spot)', v: probs.ten.above, invert: false },
              { label: 'P(S_T < −10%)', v: 1 - probs.ten.above, invert: true },
              { label: 'P(S_T < −20%)', v: 1 - probs.deep.above, invert: true },
            ].map(({ label, v }) => (
              <div key={label} className="flex items-center justify-between gap-4 border-b border-[var(--hairline)] pb-2">
                <span className="text-zinc-500 dark:text-zinc-400">{label}</span>
                <span className="tabular-nums text-zinc-900 dark:text-zinc-50">{(v * 100).toFixed(1)}%</span>
              </div>
            ))}
            <div className="flex items-center justify-between gap-4 pb-1">
              <span className="text-zinc-500 dark:text-zinc-400">Touch −10% (≈)</span>
              <span className="tabular-nums text-gold-800 dark:text-gold-200">{(probs.ten.touch * 100).toFixed(1)}%</span>
            </div>
          </div>
          <p className="mt-2 font-mono text-[9px] text-zinc-400">ATM-IV d2 over the {headline?.dte ?? 30}d horizon · European approximation · tail probabilities are the currency of hedging</p>
        </div>
      </div>

      {/* Merton credit bridge */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2"><Landmark className="h-3.5 w-3.5" /> Credit Bridge — Merton (1974)</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Equity-implied distance to default{mcap ? ` · ${ticker} market cap $${(mcap / 1e9).toFixed(1)}B` : ''}</h3>
          </div>
          <label className="flex items-center gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
            Debt $B
            <input
              type="number"
              min={0}
              step={0.5}
              value={effDebtB.toFixed(1)}
              onChange={(e) => setDebtB(Math.max(0, parseFloat(e.target.value) || 0))}
              className="w-20 rounded-md border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 font-mono text-[11px] tabular-nums text-zinc-900 dark:text-zinc-50 focus:outline-none focus:ring-1 focus:ring-gold-400"
            />
          </label>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="panel-inset p-3">
            <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Asset Value V</div>
            <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">${assetValueB.toFixed(1)}B</div>
          </div>
          <div className="panel-inset p-3">
            <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">σ(Asset)</div>
            <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{(volAsset * 100).toFixed(1)}%</div>
            <div className="font-mono text-[8.5px] text-zinc-400">≈ σ_E × E/V (elasticity)</div>
          </div>
          <div className="panel-inset p-3">
            <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Distance to Default</div>
            <div className="mt-1 font-mono text-lg font-bold text-gold-700 dark:text-gold-300">{mertonDD.toFixed(2)}σ</div>
          </div>
          <div className="panel-inset p-3">
            <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Implied PD (1y)</div>
            <div className={`mt-1 font-mono text-lg font-bold ${mertonPD > 0.02 ? 'text-rose-500 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
              {(mertonPD * 100).toFixed(2)}%
            </div>
          </div>
        </div>
        <div className="mt-3 flex items-start gap-2 font-mono text-[9px] leading-relaxed text-zinc-400">
          <Info className="mt-0.5 h-3 w-3 shrink-0" />
          Naive Merton: equity option on assets, no coupons, debt constant, σ_A via leverage elasticity, horizon 1y. Directionally useful for ranking fragility across names — not a ratings replacement.
        </div>
      </div>
    </div>
  );
}
