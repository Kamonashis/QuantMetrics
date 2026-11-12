'use client';

import React from 'react';
import { Layers, Flame } from 'lucide-react';
import type { PanelPayload } from './types';
import { OiLadder } from '@/components/charts/fno/OiLadder';

const fmtOi = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toLocaleString();

export function FnoOiTab({ panel, symbol }: { panel: PanelPayload | null; symbol: string }) {
  const spot = panel?.quote?.price ?? 0;
  const ladder = panel?.oiLadder ?? [];
  const concentration = panel?.concentration7d ?? null;
  const expiries = panel?.expiryLadder ?? [];
  const totalOi = expiries.reduce((s, l) => s + l.totalOi, 0);

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      {/* Ladder */}
      <div className="panel p-5 xl:col-span-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <div>
            <p className="kicker mb-1 flex items-center gap-2"><Layers className="h-3.5 w-3.5" /> OI Ladder</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              OI by strike — where the positioning lives (±15%)
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            total OI {fmtOi(totalOi)} · as-of last close
          </span>
        </div>
        <OiLadder rows={ladder} spot={spot} />
      </div>

      {/* OI by expiry */}
      <div className="panel p-5 xl:col-span-2">
        <div className="mb-3 border-b border-[var(--hairline)] pb-3">
          <p className="kicker mb-1 flex items-center gap-2"><Flame className="h-3.5 w-3.5" /> Where OI Expires</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">The short-dated wall decays weekly</h3>
        </div>
        <div className="space-y-2">
          {expiries.slice(0, 5).map((l) => {
            const share = totalOi > 0 ? (l.totalOi / totalOi) * 100 : 0;
            return (
              <div key={l.expiry} className="font-mono text-[10px]">
                <div className="flex items-baseline justify-between">
                  <span className="text-zinc-500 dark:text-zinc-400">{l.expiry} <span className="text-zinc-400">· {l.dte}d</span></span>
                  <span className="tabular-nums text-zinc-700 dark:text-zinc-200">
                    {fmtOi(l.totalOi)} <span className="text-zinc-400">({share.toFixed(0)}%)</span>
                  </span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-zinc-100 dark:bg-white/[0.05]">
                  <div className="h-full rounded-full bg-gradient-to-r from-gold-700 to-gold-400" style={{ width: `${Math.min(100, share)}%` }} />
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 rounded-lg border border-gold-500/25 bg-gold-500/5 px-3 py-2">
          <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">Front ≤ 7 DTE concentration</div>
          <div className="mt-1 font-mono text-2xl font-bold text-gold-800 dark:text-gold-200">
            {concentration !== null ? `${(concentration * 100).toFixed(0)}%` : '—'}
          </div>
          <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-500 dark:text-zinc-400">
            Share of universe OI expiring within a week. &gt; 30% means the positioning is structurally short-dated — walls decay every Friday and ΔOI swings are noisier for signal.
          </p>
        </div>
      </div>
    </div>
  );
}
