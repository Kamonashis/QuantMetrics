'use client';

import React from 'react';
import { Scale, Info } from 'lucide-react';
import type { PanelPayload } from './types';

const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '—' : v.toFixed(d);

export function FnoPcrTab({ panel, symbol }: { panel: PanelPayload | null; symbol: string }) {
  const pcr = panel?.pcr ?? null;
  const expiries = panel?.expiryLadder ?? [];
  const front = expiries[0];

  const divergence =
    pcr?.aggregate !== null && pcr?.aggregate !== undefined && pcr?.volume !== null && pcr?.volume !== undefined
      ? Math.abs(pcr.aggregate - pcr.volume) > 0.4
      : false;

  const tile = (label: string, value: string, sub: string, tone: 'neutral' | 'warn' = 'neutral') => (
    <div className="panel-inset p-4">
      <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{label}</div>
      <div className={`mt-1.5 font-mono text-2xl font-bold tabular-nums ${tone === 'warn' ? 'text-amber-600 dark:text-amber-400' : 'text-zinc-900 dark:text-zinc-50'}`}>{value}</div>
      <div className="mt-1 text-[9px] text-zinc-400 leading-snug">{sub}</div>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {tile('Aggregate OI PCR', fmt(pcr?.aggregate), 'put OI ÷ call OI across all expiries — positioning')}
        {tile('Volume PCR', fmt(pcr?.volume), 'put volume ÷ call volume — today’s flow')}
        {tile('ATM ±5-strike PCR', fmt(pcr?.atm), 'nearest 5 strikes each side of spot — where signal lives')}
      </div>

      {divergence && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            OI-PCR and Volume-PCR diverge by more than 0.4 — positioning and flow disagree. Read the volume leg as intent, the OI leg as inventory; do not blend them into a single sentiment number.
          </span>
        </div>
      )}

      <div className="panel p-5">
        <div className="mb-3 border-b border-[var(--hairline)] pb-3">
          <p className="kicker mb-1 flex items-center gap-2"><Scale className="h-3.5 w-3.5" /> Reading the Tape</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">How to read PCR without over-reading it</h3>
        </div>
        <div className="grid grid-cols-1 gap-2 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400 sm:grid-cols-3">
          <p><strong className="text-zinc-800 dark:text-zinc-100">PCR &gt; 1.2</strong> — put-heavy book. Hedging pressure dominates; short-squeeze fuel if spot rallies through walls.</p>
          <p><strong className="text-zinc-800 dark:text-zinc-100">PCR 0.7–1.2</strong> — balanced two-sided flow; no directional information by itself.</p>
          <p><strong className="text-zinc-800 dark:text-zinc-100">PCR &lt; 0.7</strong> — call-heavy. Chasing upside; often marks late-cycle greed rather than opportunity.</p>
        </div>
        <p className="mt-2 border-t border-[var(--hairline)] pt-2 font-mono text-[9px] text-zinc-400">
          Front expiry {front?.expiry ?? '—'} ({front?.dte ?? '—'}d) OI {(front?.totalOi ?? 0).toLocaleString()} · PCR is a positioning gauge, never a timing signal · as-of last close
        </p>
      </div>
    </div>
  );
}
