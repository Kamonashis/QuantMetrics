'use client';

import { useCallback, useState } from 'react';
import { selectFrontSlice } from '@/lib/ticker-snapshot';

export interface WatchRow {
  symbol: string;
  available: boolean;
  iv30: number | null;
  rr25: number | null;
  trapState: string | null;
  trapScore: number | null;
}

export const WATCHLIST_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY'];

// One cache for the combined scan — T6's IV matrix and T7's trap scan used to
// fire twelve overlapping requests from two separate toolbars.
let cached: { at: number; rows: WatchRow[] } | null = null;

export function useWatchlist() {
  const [rows, setRows] = useState<WatchRow[]>(cached?.rows ?? []);
  const [loading, setLoading] = useState(false);

  const scan = useCallback(async (force = false) => {
    if (!force && cached && Date.now() - cached.at < 5 * 60 * 1000) {
      setRows(cached.rows);
      return;
    }
    setLoading(true);
    try {
      const trapRes = await fetch(`/api/options?mode=trap-scan&symbols=${WATCHLIST_UNIVERSE.join(',')}`)
        .then((r) => r.json())
        .catch(() => ({ rows: [] as Record<string, unknown>[] }));
      const trapRows: Record<string, unknown>[] = Array.isArray(trapRes?.rows) ? trapRes.rows : [];
      const chains = await Promise.all(WATCHLIST_UNIVERSE.map((sym) =>
        fetch(`/api/options?mode=chain&symbol=${encodeURIComponent(sym)}`)
          .then((r) => r.json())
          .catch(() => ({ available: false }))
      ));
      const next: WatchRow[] = WATCHLIST_UNIVERSE.map((sym, i) => {
        const trap = trapRows.find((r) => r && (r.symbol === sym || r.symbol === `${sym}`));
        const chain = chains[i] as { available?: boolean; iv30?: number | null; slices?: { dte: number; atmIv: number | null; rr25?: number | null }[] };
        const front = chain?.available ? selectFrontSlice(chain.slices ?? []) : null;
        return {
          symbol: sym,
          available: chain?.available === true || trap?.available === true,
          iv30: typeof chain?.iv30 === 'number' ? chain.iv30 : null,
          rr25: front?.rr25 ?? null,
          trapState: typeof trap?.state === 'string' ? trap.state : null,
          trapScore: typeof trap?.score === 'number' ? trap.score : null,
        };
      });
      cached = { at: Date.now(), rows: next };
      setRows(next);
    } finally {
      setLoading(false);
    }
  }, []);

  return { rows, loading, scan };
}
