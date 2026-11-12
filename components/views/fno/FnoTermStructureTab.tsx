'use client';

import React from 'react';
import { TrendingUp, Info } from 'lucide-react';
import type { PanelPayload } from './types';
import { VolTermStructureChart } from '@/components/charts/VolTermStructureChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';

const fmtPct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

export function FnoTermStructureTab({ panel, symbol }: { panel: PanelPayload | null; symbol: string }) {
  const ts = panel?.termStructure ?? null;
  const points = ts?.points ?? [];
  const carry = ts?.forwardCarry ?? [];
  const verdict = ts?.verdict;
  const iv30 = panel?.iv30 ?? null;

  const verdictColor =
    verdict?.verdict === 'CONTANGO' ? 'text-emerald-600 dark:text-emerald-400'
    : verdict?.verdict === 'BACKWARDATION' ? 'text-rose-500 dark:text-rose-400'
    : 'text-zinc-500';

  return (
    <div className="space-y-4">
      {/* Verdict header */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'IV-30', value: fmtPct(iv30), tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'Front / Back (60d+)', value: verdict?.ratio !== null && verdict?.ratio !== undefined ? verdict.ratio.toFixed(2) : '—', tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'Front IV', value: fmtPct(verdict?.frontIv), tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'Curve verdict', value: verdict?.verdict ?? '—', tone: verdictColor },
        ].map((s) => (
          <div key={s.label} className="panel-inset p-4">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{s.label}</div>
            <div className={`mt-1.5 font-mono text-xl font-bold tabular-nums ${s.tone}`}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Curve chart */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2"><TrendingUp className="h-3.5 w-3.5" /> Term Structure</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              ATM IV by expiry — is the market bracing for now or later?
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            interpolated on total variance (w = σ²T) · 25Δ RR beneath each point
          </span>
        </div>
        <ChartFullscreen title="ATM IV Term Structure" subtitle="Interpolated on total variance">
          <VolTermStructureChart
            points={points.map((p) => ({ dte: p.dte, iv: p.iv }))}
            axisMode="vol"
            band={null}
            iv30Marker={30}
          />
        </ChartFullscreen>
      </div>

      {/* Carry + skew RR */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Forward Carry</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Implied financing between expiries (via put-call parity)</h3>
          </div>
          {carry.length === 0 ? (
            <div className="text-xs text-zinc-400">Not enough expiries with two-sided quotes to compute carry.</div>
          ) : (
            <div className="space-y-1.5 font-mono text-[10px]">
              <div className="grid grid-cols-4 text-[8.5px] uppercase tracking-wider text-zinc-400">
                <span>Leg</span><span>From</span><span>To</span><span className="text-right">Carry ann.</span>
              </div>
              {carry.map((c, i) => (
                <div key={i} className="grid grid-cols-4 border-t border-[var(--hairline)] py-1.5">
                  <span className="text-zinc-400">{i + 1}</span>
                  <span className="tabular-nums text-zinc-700 dark:text-zinc-200">{c.fromDte}d</span>
                  <span className="tabular-nums text-zinc-700 dark:text-zinc-200">{c.toDte}d</span>
                  <span className={`text-right tabular-nums ${c.carry >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                    {c.carry >= 0 ? '+' : ''}{(c.carry * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          )}
          <p className="mt-2 font-mono text-[9px] text-zinc-400">F2/F1 annualized over the gap · positive = market prices drift or dividend tailwind</p>
        </div>

        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">25Δ Risk Reversal</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Skew per expiry — which direction is it afraid of?</h3>
          </div>
          {points.length === 0 ? (
            <div className="text-xs text-zinc-400">No fitted smiles available.</div>
          ) : (
            <div className="space-y-1.5">
              {points.filter((p) => p.rr25 !== null).map((p) => (
                <div key={p.expiry} className="flex items-center gap-2 font-mono text-[10px]">
                  <span className="w-16 text-zinc-500">{p.dte}d</span>
                  <div className="relative h-3.5 w-40 rounded bg-zinc-100 dark:bg-white/[0.04]">
                    <div
                      className="absolute top-0 h-full rounded bg-gold-500/80"
                      style={{
                        width: `${Math.min(50, Math.abs((p.rr25 as number) * 100))}%`,
                        left: (p.rr25 as number) < 0 ? `${50 - Math.min(50, Math.abs((p.rr25 as number) * 100))}%` : '50%',
                      }}
                    />
                    <div className="absolute left-1/2 top-0 h-full w-px bg-zinc-300 dark:bg-zinc-700" />
                  </div>
                  <span className="tabular-nums text-zinc-700 dark:text-zinc-200">{(p.rr25 as number) >= 0 ? '+' : ''}{((p.rr25 as number) * 100).toFixed(2)} pts</span>
                </div>
              ))}
              {points.every((p) => p.rr25 === null) && (
                <p className="text-[10px] text-zinc-400">Skew fits unavailable — smiles too thin on this surface today.</p>
              )}
            </div>
          )}
          <p className="mt-2 font-mono text-[9px] text-zinc-400">RR = σ(25Δ call) − σ(25Δ put) · negative = puts bid (typical for equities)</p>
        </div>
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>Verdicts &gt; 4% front/back ratio flip the curve read: CONTANGO = market calm now, bracing later (normal); BACKWARDATION = hedging demand today over later (stress or pre-event). Always ask whether the front expiry is an event expiry before reading into it.</span>
      </div>
    </div>
  );
}
