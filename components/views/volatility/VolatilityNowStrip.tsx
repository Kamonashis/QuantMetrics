'use client';

import React from 'react';
import { Zap } from 'lucide-react';
import { percentileRank, termStructureVerdict } from '@/lib/vol-analytics';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import type { ChainPayload, IndicesPayload } from './types';

type JumpTab = 't1' | 't3' | 't7' | 't8';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

/** A one-screen digest of the vol suite. Every tile deep-links to its owning tab. */
export function VolatilityNowStrip({
  chain,
  indices,
  rv20,
  barsSource,
  onJump,
}: {
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  rv20: number | null;
  barsSource?: string | null;
  onJump: (tab: JumpTab) => void;
}) {
  const ts = chain?.termStructure ?? [];
  const shortIv = ts.find((p) => p.dte >= 7 && p.iv !== null)?.iv ?? ts[0]?.iv ?? null;
  const longIv = [...ts].reverse().find((p) => p.dte > 45 && p.iv !== null)?.iv ?? ts[ts.length - 1]?.iv ?? null;
  const curve = shortIv !== null && longIv !== null ? termStructureVerdict(shortIv, longIv) : null;

  const vixBars = indices?.series?.VIX?.bars ?? [];
  const vixNow = vixBars.length > 0 ? vixBars[vixBars.length - 1].close : null;
  const vixPct = vixNow !== null && vixBars.length > 30
    ? percentileRank(vixBars.slice(-252).map((b) => b.close), vixNow)
    : null;

  const trap = chain?.trap ?? null;
  const iv30 = chain?.iv30 ?? null;
  const gap = iv30 !== null && rv20 !== null ? iv30 - rv20 : null;

  const tiles: { label: string; value: string; sub: string; tab: JumpTab; tone?: string }[] = [
    { label: 'IV 30d', value: pct(iv30), sub: 'European BSM · √365', tab: 't1' },
    { label: 'RV 20d', value: pct(rv20), sub: 'trailing returns · √252', tab: 't8' },
    {
      label: 'IV − RV',
      value: gap === null ? '—' : `${gap >= 0 ? '+' : ''}${pct(gap)}`,
      sub: 'variance risk premium',
      tab: 't3',
      tone: gap === null ? undefined : gap >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]',
    },
    { label: 'Curve', value: curve ?? '—', sub: 'front vs 45d+ slope', tab: 't3' },
    {
      label: 'VIX',
      value: vixNow === null ? '—' : vixNow.toFixed(1),
      sub: vixPct === null ? 'market stress' : `${vixPct.toFixed(0)}th pctile · 1y`,
      tab: 't1',
    },
    {
      label: 'Trap',
      value: trap?.state ? trap.state.toUpperCase() : '—',
      sub: trap?.score !== undefined ? `score ${trap.score.toFixed(0)}` : 'dealer positioning',
      tab: 't7',
    },
  ];

  return (
    <section className="panel p-4 sm:p-5" aria-label="Volatility at a glance">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
        <div className="flex items-center gap-2">
          <Zap className="h-3.5 w-3.5 text-gold-500" aria-hidden="true" />
          <p className="kicker">Now · At A Glance</p>
          <span className="text-[10px] text-zinc-400">click a tile to open its detail</span>
        </div>
        <ProvenanceBadge daily={barsSource} />
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
        {tiles.map((tile) => (
          <button
            key={tile.label}
            type="button"
            onClick={() => onJump(tile.tab)}
            aria-label={`${tile.label} ${tile.value} — open detail`}
            className="group flex min-w-0 cursor-pointer flex-col items-start gap-1 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2.5 text-left transition-colors hover:border-gold-500/50 hover:bg-gold-500/5 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60"
          >
            <span className="font-mono text-[9px] font-semibold uppercase tracking-[0.14em] text-zinc-400">{tile.label}</span>
            <span className={`truncate font-mono text-base font-semibold tabular-nums text-zinc-900 dark:text-zinc-50 ${tile.tone ?? ''}`}>
              {tile.value}
            </span>
            <span className="truncate font-mono text-[9px] text-zinc-400">{tile.sub}</span>
          </button>
        ))}
      </div>
      <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
        IV and RV are annualizations of the same variance (IV √365, RV √252) and are compared directly. Quotes are ~15-min delayed European BSM approximations — not executable.
      </p>
    </section>
  );
}
