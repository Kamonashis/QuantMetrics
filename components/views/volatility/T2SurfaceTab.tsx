'use client';

import React, { useState } from 'react';
import { Grid3x3, ChartSpline, MoveHorizontal, ArrowUpRight } from 'lucide-react';
import type { ChainPayload, IndicesPayload } from './types';
import { VolSurfaceHeatmap } from '@/components/charts/VolSurfaceHeatmap';
import { VolatilitySmileChart } from '@/components/charts/VolatilitySmileChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { expectedMove } from '@/lib/options';

export function T2SurfaceTab({ chain, indices }: { chain: ChainPayload | null; indices: IndicesPayload | null }) {
  const slices = chain?.slices ?? [];
  const fitSlices = slices.filter((s) => s.points.length >= 5);
  const defaultExpiry = fitSlices.find((s) => s.dte >= 21 && s.dte <= 45) ?? fitSlices[0] ?? null;
  const [selectedExpiry, setSelectedExpiry] = useState<string | null>(defaultExpiry?.expirationDate ?? null);
  const active = fitSlices.find((s) => s.expirationDate === selectedExpiry) ?? defaultExpiry;

  if (slices.length === 0) {
    return <div className="panel p-6 text-sm text-zinc-500">No chain data available — the surface is dark until quotes arrive.</div>;
  }

  const atmIvForLadder = (s: typeof slices[number]) => s.atmIv ?? 0.2;
  const ladderMax = Math.max(...slices.map((s) => atmIvForLadder(s) * Math.sqrt(s.t))) * 100;

  // Skew term structure
  const rrPoints = slices.filter((s) => s.rr25 !== null).map((s) => ({ dte: s.dte, rr: (s.rr25 as number) * 100 }));
  const rrAbsMax = Math.max(1, ...rrPoints.map((p) => Math.abs(p.rr)));

  // ^SKEW reference for "fear gauge"
  const skewSeries = indices?.series?.SKEW?.bars ?? [];
  const skewNow = skewSeries.length > 0 ? skewSeries[skewSeries.length - 1].close : null;

  return (
    <div className="space-y-4">
      {/* Surface heatmap */}
      <div className="panel p-5">
        <div className="mb-4 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <Grid3x3 className="h-3.5 w-3.5" /> Volatility Surface
            </p>
            <h3 className="font-display text-base font-semibold text-zinc-900 dark:text-zinc-50">
              IV by moneyness × expiry
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            k = ln(K / F) · European BSM · quotes → SVI
          </span>
        </div>
        <ChartFullscreen title="IV by Moneyness × Expiry" subtitle="European BSM · SVI fit">
          <VolSurfaceHeatmap slices={slices} />
        </ChartFullscreen>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        {/* Smile slice with SVI fit */}
        <div className="panel p-5 xl:col-span-3">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2">
                <ChartSpline className="h-3.5 w-3.5" /> Smile + Fit
              </p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                {active?.dte}d expiry · {active?.expirationDate}
              </h3>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {fitSlices.map((s) => (
                <button
                  key={s.expirationDate}
                  onClick={() => setSelectedExpiry(s.expirationDate)}
                  className={`cursor-pointer rounded-md px-2 py-1 font-mono text-[10px] transition-colors ${
                    (s.expirationDate === selectedExpiry) || (!selectedExpiry && s === defaultExpiry)
                      ? 'bg-gold-500 text-zinc-950 font-bold'
                      : 'bg-zinc-100 text-zinc-500 hover:text-zinc-800 dark:bg-white/[0.05] dark:text-zinc-400'
                  }`}
                >
                  {s.dte}d
                </button>
              ))}
            </div>
          </div>
          <ChartFullscreen title="Volatility Smile + SVI Fit" subtitle={active ? `${active.dte}d expiry · ${active.expirationDate}` : undefined}>
            <VolatilitySmileChart slice={active ?? null} />
          </ChartFullscreen>
        </div>

        {/* Skew term structure + expected move ladder */}
        <div className="panel p-5 xl:col-span-2">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1 flex items-center gap-2">
              <ArrowUpRight className="h-3.5 w-3.5" /> 25Δ RR by Expiry
            </p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Skew term structure
            </h3>
          </div>
          <div className="space-y-1.5">
            {rrPoints.length === 0 && <div className="text-xs text-zinc-400">No skew fits available</div>}
            {rrPoints.map((p) => (
              <div key={p.dte} className="flex items-center gap-2 font-mono text-[10px]">
                <span className="w-8 text-right text-zinc-400">{p.dte}d</span>
                <div className="relative h-4 flex-1 rounded bg-zinc-100 dark:bg-white/[0.04]">
                  <div
                    className="absolute top-0 h-full rounded bg-gold-500/80"
                    style={{ width: `${(Math.abs(p.rr) / rrAbsMax) * 50}%`, left: p.rr < 0 ? `${50 - (Math.abs(p.rr) / rrAbsMax) * 50}%` : '50%' }}
                  />
                  <div className="absolute left-1/2 top-0 h-full w-px bg-zinc-300 dark:bg-zinc-700" />
                </div>
                <span className="w-12 tabular-nums text-zinc-700 dark:text-zinc-200">
                  {p.rr >= 0 ? '+' : ''}{p.rr.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
            negative = puts bid · front-end RR usually most negative (near-dated crash fear priced in)
          </p>

          <div className="mt-4 border-t border-[var(--hairline)] pt-3">
            <div className="mb-2 flex items-center gap-2">
              <MoveHorizontal className="h-3.5 w-3.5 text-gold-600 dark:text-gold-400" />
              <h4 className="font-display text-xs font-semibold text-zinc-900 dark:text-zinc-50">Expected-Move Ladder</h4>
            </div>
            <div className="space-y-1.5">
              {slices.map((s) => {
                const em = expectedMove(chain?.quote?.price ?? 100, atmIvForLadder(s), s.t, null, null);
                const pctWidth = (em.oneSigmaPct / Math.max(1, ladderMax)) * 100;
                return (
                  <div key={s.expirationDate} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-8 text-right text-zinc-400">{s.dte}d</span>
                    <div className="relative h-4 flex-1 rounded bg-zinc-100 dark:bg-white/[0.04]">
                      <div className="absolute left-0 top-0 h-full rounded bg-gradient-to-r from-gold-600/80 to-gold-400/60" style={{ width: `${pctWidth}%` }} />
                    </div>
                    <span className="w-12 tabular-nums text-zinc-700 dark:text-zinc-200">±{em.oneSigmaPct.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
            <p className="mt-2 font-mono text-[9px] text-zinc-400">1σ implied move · ±σ√T · 68% band</p>
          </div>

          {skewNow !== null && (
            <div className="mt-4 rounded-lg border border-gold-500/25 bg-gold-500/5 px-3 py-2 font-mono text-[10px] text-gold-800 dark:text-gold-200">
              ^SKEW now <strong>{skewNow.toFixed(0)}</strong> — CBOE tail-risk index: premium paid for out-of-cycle crash insurance.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
