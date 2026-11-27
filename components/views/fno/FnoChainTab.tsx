'use client';

import React, { useEffect, useMemo, useState } from 'react';
import type { PanelPayload, ChainPayload } from './types';
import type { TrapPayload } from '@/components/views/volatility/types';
import { FnoChainTable } from '@/components/charts/fno/FnoChainTable';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { Info } from 'lucide-react';

const fmtGexCompact = (v: number | null | undefined) => {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  const sign = v > 0 ? '+' : v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
};

export function FnoChainTab({
  panel,
  symbol,
  trap = null,
}: {
  panel: PanelPayload | null;
  symbol: string;
  trap?: TrapPayload | null;
}) {
  const [chain, setChain] = useState<ChainPayload | null>(null);
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const expiries = (panel?.expiryLadder ?? []).slice(0, 6).map((l) => l.expiry);
  // Prefer the nearest expiry with a solvable ATM IV (0-DTE quotes are often stale by close)
  const liquidExpiry = (panel?.expiryLadder ?? []).find((l) => l.atmIv !== null)?.expiry ?? null;
  const [expiry, setExpiry] = useState<string | undefined>(undefined);
  const effectiveExpiry = expiry ?? liquidExpiry ?? expiries[0] ?? null;

  useEffect(() => {
    if (!effectiveExpiry) return;
    let cancelled = false;
    fetch(`/api/fno?mode=chain&symbol=${encodeURIComponent(symbol)}&expiry=${encodeURIComponent(effectiveExpiry)}`)
      .then((r) => r.json())
      .then((d: ChainPayload) => {
        if (!cancelled) { setChain(d); setLoadedFor(effectiveExpiry); }
      })
      .catch(() => {
        if (!cancelled) { setChain({ available: false, reason: 'chain fetch failed' }); setLoadedFor(effectiveExpiry); }
      });
    return () => {
      cancelled = true;
    };
  }, [effectiveExpiry, symbol, panel]);

  const loading = effectiveExpiry !== null && effectiveExpiry !== loadedFor;

  const spot = (panel?.quote?.price ?? chain?.quote?.price) ?? 0;
  const atmStrike = chain?.atmStrike ?? panel?.expiryLadder?.find((l) => l.expiry === effectiveExpiry)?.atmStrike ?? 0;
  const activeTrap = panel?.trap ?? trap ?? null;

  const expiryNetGex = useMemo(() => {
    if (!chain?.available || !chain.rows?.length) return null;
    let sum = 0;
    let any = false;
    for (const r of chain.rows) {
      const g = r.gex?.netGex ?? ((r.call?.gex != null || r.put?.gex != null) ? (r.call?.gex ?? 0) + (r.put?.gex ?? 0) : null);
      if (g !== null && isFinite(g)) {
        sum += g;
        any = true;
      }
    }
    return any ? sum : null;
  }, [chain]);

  return (
    <div className="space-y-4">
      {expiries.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <AnimatedTabs
            size="sm"
            value={effectiveExpiry ?? ''}
            onChange={setExpiry}
            tabs={expiries.map((e) => ({
              id: e,
              label: new Date(e + 'T12:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            }))}
          />
          <div className="flex flex-wrap items-center gap-3 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            {expiryNetGex !== null && (
              <span>
                Window Net GEX{' '}
                <span className={expiryNetGex >= 0 ? 'font-semibold text-[var(--positive)]' : 'font-semibold text-[var(--negative)]'}>
                  {fmtGexCompact(expiryNetGex)}
                </span>
              </span>
            )}
            {activeTrap?.available && (
              <span>
                Universe{' '}
                <span className={activeTrap.regime === 'long' ? 'text-[var(--positive)]' : activeTrap.regime === 'short' ? 'text-[var(--negative)]' : ''}>
                  {activeTrap.regime === 'long' ? 'Long γ' : activeTrap.regime === 'short' ? 'Short γ' : 'γ unknown'}
                </span>
                {activeTrap.gammaFlip?.flip != null ? ` · Flip $${activeTrap.gammaFlip.flip.toFixed(1)}` : ''}
              </span>
            )}
            <span>{chain?.dte ?? '—'} DTE · as-of last close</span>
          </div>
        </div>
      )}

      {loading ? (
        <div className="panel animate-pulse p-10 text-center font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
          Fetching chain…
        </div>
      ) : chain?.available ? (
        <FnoChainTable rows={chain.rows ?? []} spot={spot} atmStrike={atmStrike} />
      ) : (
        <div className="panel p-6 text-sm text-zinc-500 dark:text-zinc-400">
          {chain?.reason ?? 'Chain unavailable — Yahoo rejected or timed out.'}
        </div>
      )}

      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          IV computed from mid-quote Newton solve (never Yahoo&apos;s field); ATM row marked ●. IV from trade prints is noisy on illiquid strikes — a spread &gt; 25% of mid flags the row as unreliable. Strike-level Γ and $GEX use the assumed dealer squeeze convention (long calls +, short puts −) per 1% spot move; OI is as-of last close.
        </span>
      </div>
    </div>
  );
}
