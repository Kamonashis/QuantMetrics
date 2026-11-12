'use client';

import React, { useEffect, useState } from 'react';
import type { PanelPayload, ChainPayload } from './types';
import { FnoChainTable } from '@/components/charts/fno/FnoChainTable';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { Info } from 'lucide-react';

export function FnoChainTab({
  panel,
  symbol,
}: {
  panel: PanelPayload | null;
  symbol: string;
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
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            {chain?.dte ?? '—'} DTE · as-of last close
          </span>
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
          IV computed from mid-quote Newton solve (never Yahoo&apos;s field); ATM row marked ●. IV from trade prints is noisy on illiquid strikes — a spread &gt; 25% of mid flags the row as unreliable. OI is as-of last close.
        </span>
      </div>
    </div>
  );
}
