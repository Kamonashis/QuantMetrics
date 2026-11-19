'use client';

import React from 'react';
import { Database } from 'lucide-react';

/**
 * One provenance chip for every view. The seeded daily/intraday caches are
 * synthetic until a live fetch lands, so `null` is deliberately treated as
 * synthetic rather than as "live but unknown".
 */
export function ProvenanceBadge({
  daily,
  intraday,
  compact = false,
}: {
  daily?: string | null;
  intraday?: string | null;
  compact?: boolean;
}) {
  const live = daily === 'live_market';
  const title = `Daily bars: ${daily ?? 'synthetic_gbm (unfetched)'}${intraday ? ` · Intraday: ${intraday}` : ''}`;
  return (
    <span
      title={title}
      aria-label={title}
      className={`inline-flex shrink-0 cursor-help items-center gap-1.5 rounded-md border px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider ${
        live
          ? 'border-[var(--positive)]/40 bg-[var(--positive)]/10 text-[var(--positive)]'
          : 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300'
      }`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-[var(--positive)]' : 'bg-gold-500'}`} />
      {!compact && <Database className="h-3 w-3" aria-hidden="true" />}
      {compact ? (live ? 'LIVE' : 'SYN') : live ? 'Live data' : 'Synthetic data'}
    </span>
  );
}
