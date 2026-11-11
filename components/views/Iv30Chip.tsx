'use client';

import React, { useEffect, useState } from 'react';
import { Gauge } from 'lucide-react';

/**
 * Thin IV state chip for the Analysis tab header — one number, always visible.
 * Fetches the chain itself (server-cached 15 min). While a fetch for the current
 * ticker is in flight it renders a placeholder; on failure it hides silently.
 */
export function Iv30Chip({ ticker }: { ticker: string }) {
  const [result, setResult] = useState<{ ticker: string; iv30: number } | { ticker: string; error: true } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/options?mode=chain&symbol=${encodeURIComponent(ticker)}`)
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        if (d?.available && typeof d.iv30 === 'number') {
          setResult({ ticker, iv30: d.iv30 });
        } else {
          setResult({ ticker, error: true });
        }
      })
      .catch(() => {
        if (!cancelled) setResult({ ticker, error: true });
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  const current = result && result.ticker === ticker ? result : null;

  if (!current) {
    return (
      <span className="hidden md:inline-flex items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 font-mono text-[10px] tabular-nums text-zinc-400">
        IV <span className="animate-pulse">···</span>
      </span>
    );
  }
  if ('error' in current) return null;

  return (
    <span
      title="30-day implied volatility (interpolated from today's chain, w-linear on total variance). Click the Implied Vol tab for the full surface."
      className="hidden md:inline-flex cursor-help items-center gap-1.5 rounded-lg border border-gold-500/30 bg-gold-500/8 px-2 py-1 font-mono text-[10px] tabular-nums text-gold-800 dark:border-gold-500/25 dark:bg-gold-500/5 dark:text-gold-300"
    >
      <Gauge className="h-3 w-3" />
      IV30 <strong>{(current.iv30 * 100).toFixed(1)}%</strong>
    </span>
  );
}
