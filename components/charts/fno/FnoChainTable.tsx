'use client';

import React from 'react';
import type { ChainRowOut } from '@/components/views/fno/types';

/**
 * Dual-sided option chain table: calls left, strike center, puts right.
 * ATM strike row is pinned and tinted; wide/stale rows are muted.
 */
export function FnoChainTable({
  rows,
  spot,
  atmStrike,
}: {
  rows: ChainRowOut[];
  spot: number;
  atmStrike: number;
}) {
  if (rows.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        No contracts in this window
      </div>
    );
  }

  const maxOi = Math.max(1, ...rows.flatMap((r) => [r.call?.oi ?? 0, r.put?.oi ?? 0]));
  const fmtOi = (v: number | undefined) => v === undefined ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v);
  const fmt = (v: number | null | undefined, d = 2) => v === null || v === undefined || !isFinite(v) ? '—' : v.toFixed(d);
  const fmtGex = (v: number | null | undefined) => {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const sign = v > 0 ? '+' : v < 0 ? '-' : '';
    const abs = Math.abs(v);
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
    return `${sign}$${abs.toFixed(0)}`;
  };
  const gexTone = (v: number | null | undefined) => {
    if (v === null || v === undefined || !isFinite(v) || Math.abs(v) < 1e-6) return 'text-zinc-400';
    return v > 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]';
  };

  return (
    <div className="overflow-x-auto">
      <table className="table-sticky-first w-full min-w-[980px] border-collapse font-mono text-[10.5px]">
        <thead>
          <tr className="text-[8.5px] uppercase tracking-[0.14em] text-zinc-400">
            <th className="px-1.5 py-1 text-right">OI</th>
            <th className="px-1.5 py-1 text-right">Vol</th>
            <th className="px-1.5 py-1 text-right">Δ</th>
            <th className="px-1.5 py-1 text-right">IV%</th>
            <th className="px-1.5 py-1 text-right">Bid</th>
            <th className="px-1.5 py-1 text-right">Ask</th>
            <th className="px-2 py-1 text-center text-gold-600 dark:text-gold-300">Strike</th>
            <th
              className="px-1.5 py-1 text-right"
              title="BSM Gamma (Γ) at strike under solved mid-quote IV"
            >
              Γ
            </th>
            <th
              className="px-1.5 py-1 text-right"
              title="Strike-level Net $GEX per 1% spot move — assumed dealer squeeze convention (dealers long calls +, short puts −)"
            >
              $GEX
            </th>
            <th className="px-1.5 py-1 text-left">Bid</th>
            <th className="px-1.5 py-1 text-left">Ask</th>
            <th className="px-1.5 py-1 text-left">IV%</th>
            <th className="px-1.5 py-1 text-left">Δ</th>
            <th className="px-1.5 py-1 text-left">Vol</th>
            <th className="px-1.5 py-1 text-left">OI</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isAtm = r.strike === atmStrike;
            const dim = (side: 'call' | 'put') => {
              const s = r[side];
              return !s || s.spreadPct === null || (s.spreadPct !== null && s.spreadPct > 25);
            };
            const strikeGamma = r.gamma ?? r.call?.gamma ?? r.put?.gamma ?? null;
            const callGex = r.gex?.callGex ?? r.call?.gex ?? null;
            const putGex = r.gex?.putGex ?? r.put?.gex ?? null;
            const netGex =
              r.gex?.netGex ??
              (callGex !== null || putGex !== null ? (callGex ?? 0) + (putGex ?? 0) : null);
            return (
              <tr
                key={r.strike}
                className={
                  isAtm
                    ? 'bg-gold-500/10 dark:bg-gold-500/[0.07] font-semibold'
                    : r.strike % 10 === 0
                      ? 'bg-zinc-50/50 dark:bg-white/[0.02]'
                      : ''
                }
              >
                {/* CALL side (right-aligned) */}
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                  <div className="flex items-center justify-end gap-1.5">
                    <span className="inline-block h-2 rounded-sm bg-gold-400/70" style={{ width: `${Math.min(100, ((r.call?.oi ?? 0) / maxOi) * 70)}px` }} />
                    {fmtOi(r.call?.oi)}
                  </div>
                </td>
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-500">{fmtOi(r.call?.volume)}</td>
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-600 dark:text-zinc-300">{r.call?.delta != null ? r.call.delta.toFixed(2) : '—'}</td>
                <td className={`px-1.5 py-[3px] text-right tabular-nums ${dim('call') ? 'text-zinc-400' : 'text-gold-800 dark:text-gold-200'}`}>{fmt(r.call?.iv)}</td>
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-600 dark:text-zinc-300">{fmt(r.call?.bid)}</td>
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-600 dark:text-zinc-300">{fmt(r.call?.ask)}</td>
                {/* STRIKE + Γ + $GEX */}
                <td className={`px-2 py-[3px] text-center tabular-nums ${isAtm ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-900 dark:text-zinc-50'}`}>
                  {isAtm && <span className="mr-1 text-[8px]">●</span>}{r.strike}
                </td>
                <td className="px-1.5 py-[3px] text-right tabular-nums text-zinc-500 dark:text-zinc-400">
                  {fmt(strikeGamma, 4)}
                </td>
                <td
                  className={`px-1.5 py-[3px] text-right tabular-nums font-medium ${gexTone(netGex)}`}
                  title={`Assumed squeeze convention (dealers long calls +, short puts −) · Call GEX ${fmtGex(callGex)} · Put GEX ${fmtGex(putGex)}`}
                >
                  {fmtGex(netGex)}
                </td>
                {/* PUT side (left-aligned) */}
                <td className="px-1.5 py-[3px] tabular-nums text-zinc-600 dark:text-zinc-300">{fmt(r.put?.bid)}</td>
                <td className="px-1.5 py-[3px] tabular-nums text-zinc-600 dark:text-zinc-300">{fmt(r.put?.ask)}</td>
                <td className={`px-1.5 py-[3px] tabular-nums ${dim('put') ? 'text-zinc-400' : 'text-gold-800 dark:text-gold-200'}`}>{fmt(r.put?.iv)}</td>
                <td className="px-1.5 py-[3px] tabular-nums text-zinc-600 dark:text-zinc-300">{r.put?.delta != null ? r.put.delta.toFixed(2) : '—'}</td>
                <td className="px-1.5 py-[3px] tabular-nums text-zinc-500">{fmtOi(r.put?.volume)}</td>
                <td className="px-1.5 py-[3px] tabular-nums text-zinc-600 dark:text-zinc-300">
                  <div className="flex items-center gap-1.5">
                    {fmtOi(r.put?.oi)}
                    <span className="inline-block h-2 rounded-sm bg-rose-400/70" style={{ width: `${Math.min(100, ((r.put?.oi ?? 0) / maxOi) * 70)}px` }} />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 font-mono text-[9px] text-zinc-400">
        ● = ATM strike ({atmStrike}, spot {spot > 0 ? `$${spot.toFixed(2)}` : '—'}) · greyed = spread &gt; 25% of mid (unreliable) · $GEX = net dealer gamma exposure per 1% spot move under assumed squeeze convention (calls +, puts −)
      </p>
    </div>
  );
}
