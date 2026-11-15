'use client';

import React, { useEffect, useState } from 'react';
import { Gauge } from 'lucide-react';

/**
 * IV30 tile content for the Analysis KPI strip — one number, always visible.
 * Fetches the chain itself (server-cached 15 min). While a fetch for the current
 * ticker is in flight it renders a pulse placeholder; on failure it renders "—".
 * The parent tile supplies the label and chrome.
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
    return <span className="font-mono text-sm font-semibold text-zinc-400 animate-pulse">···</span>;
  }
  if ('error' in current) {
    return <span className="font-mono text-sm font-semibold text-zinc-500">—</span>;
  }

  return (
    <span
      title="30-day implied volatility (interpolated from today's chain, w-linear on total variance). See the Implied Vol tab for the full surface."
      className="inline-flex cursor-help items-center gap-1.5 font-mono text-sm font-semibold tabular-nums text-gold-800 dark:text-gold-300"
    >
      <Gauge className="h-3.5 w-3.5" aria-hidden="true" />
      {(current.iv30 * 100).toFixed(1)}%
    </span>
  );
}
