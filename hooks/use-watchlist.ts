'use client';

import { useCallback, useState } from 'react';
import { selectFrontSlice } from '@/lib/ticker-snapshot';

export type WatchlistGammaRegime = 'long' | 'short' | 'unknown';

export interface WatchRow {
  symbol: string;
  available: boolean;
  iv30: number | null;
  rr25: number | null;
  trapState: string | null;
  trapScore: number | null;
  totalGex: number | null;
  gammaFlip: number | null;
  flipDistPct: number | null;
  regime: WatchlistGammaRegime | null;
  callWall: number | null;
  putWall: number | null;
  gammaCallWall: number | null;
  gammaPutWall: number | null;
  expiryConcentration7d: number | null;
  zeroDteShare: number | null;
  gexPerAdv: number | null;
}

const finiteNum = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

const extractStrike = (v: unknown): number | null => {
  const direct = finiteNum(v);
  if (direct !== null) return direct;
  if (v && typeof v === 'object') {
    return finiteNum((v as { strike?: unknown }).strike);
  }
  return null;
};

const extractRegime = (v: unknown): WatchlistGammaRegime | null =>
  v === 'long' || v === 'short' || v === 'unknown' ? v : null;

export const WATCHLIST_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY'];

// One cache for the combined scan — T6's IV matrix and T7's trap scan used to
// fire twelve overlapping requests from two separate toolbars.
let cached: { at: number; rows: WatchRow[] } | null = null;

interface WatchChainPayload {
  available?: boolean;
  iv30?: number | null;
  slices?: { dte: number; atmIv: number | null; rr25?: number | null }[];
  trap?: {
    available?: boolean;
    state?: string;
    score?: number;
    totalGex?: number;
    regime?: WatchlistGammaRegime;
    gammaFlip?: { flip?: number | null; regime?: WatchlistGammaRegime };
    flipDistPct?: number | null;
    walls?: { call?: { strike?: number } | null; put?: { strike?: number } | null };
    gammaWalls?: { call?: { strike?: number } | null; put?: { strike?: number } | null };
    expiryConcentration7d?: number;
    zeroDteShare?: number;
    gexPerAdv?: number | null;
  } | null;
}

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
        const chain = chains[i] as WatchChainPayload;
        const front = chain?.available ? selectFrontSlice(chain.slices ?? []) : null;
        const chainTrap = chain?.available && chain.trap?.available ? chain.trap : null;
        const trapOk = trap?.available === true;
        const rawTrap = trapOk ? trap : null;

        return {
          symbol: sym,
          available: chain?.available === true || trapOk,
          iv30: finiteNum(chain?.iv30),
          rr25: front?.rr25 ?? null,
          trapState: typeof rawTrap?.state === 'string'
            ? rawTrap.state
            : typeof chainTrap?.state === 'string'
              ? chainTrap.state
              : null,
          trapScore: finiteNum(rawTrap?.score) ?? finiteNum(chainTrap?.score),
          totalGex: finiteNum(rawTrap?.totalGex) ?? finiteNum(chainTrap?.totalGex),
          gammaFlip: finiteNum(rawTrap?.gammaFlip) ?? finiteNum(chainTrap?.gammaFlip?.flip),
          flipDistPct: finiteNum(rawTrap?.flipDistPct) ?? finiteNum(chainTrap?.flipDistPct),
          regime: extractRegime(rawTrap?.regime) ?? extractRegime(chainTrap?.regime ?? chainTrap?.gammaFlip?.regime),
          callWall: extractStrike(rawTrap?.callWall) ?? extractStrike(chainTrap?.walls?.call),
          putWall: extractStrike(rawTrap?.putWall) ?? extractStrike(chainTrap?.walls?.put),
          gammaCallWall: extractStrike(rawTrap?.gammaCallWall) ?? extractStrike(chainTrap?.gammaWalls?.call),
          gammaPutWall: extractStrike(rawTrap?.gammaPutWall) ?? extractStrike(chainTrap?.gammaWalls?.put),
          expiryConcentration7d: finiteNum(rawTrap?.expiryConcentration7d) ?? finiteNum(chainTrap?.expiryConcentration7d),
          zeroDteShare: finiteNum(rawTrap?.zeroDteShare) ?? finiteNum(chainTrap?.zeroDteShare),
          gexPerAdv: finiteNum(rawTrap?.gexPerAdv) ?? finiteNum(chainTrap?.gexPerAdv),
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
