'use client';

import React from 'react';
import { Gauge } from 'lucide-react';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';

/** IV30 tile content — reads the single shared chain snapshot, no extra fetch. */
export function Iv30Chip() {
  const { chain, chainLoading } = useTickerSnapshot();
  const available = chain?.available === true && typeof chain.iv30 === 'number' && isFinite(chain.iv30);

  if (!available) {
    return chainLoading
      ? <span className="font-mono text-sm font-semibold text-zinc-400 animate-pulse">···</span>
      : <span className="font-mono text-sm font-semibold text-zinc-500">—</span>;
  }

  return (
    <span
      title="30-day implied volatility (interpolated from today's chain, w-linear on total variance). European BSM, ~15-min delayed, √365. See the Implied Vol tab for the full surface."
      className="inline-flex cursor-help items-center gap-1.5 font-mono text-sm font-semibold tabular-nums text-gold-800 dark:text-gold-300"
    >
      <Gauge className="h-3.5 w-3.5" aria-hidden="true" />
      {(chain!.iv30! * 100).toFixed(1)}%
    </span>
  );
}
