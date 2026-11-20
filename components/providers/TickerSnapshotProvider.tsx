'use client';

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { PriceBar, RollingVolSeries } from '@/lib/quant-engine';
import { sessionsUntil } from '@/lib/earnings';
import { useWatchlist, type WatchRow } from '@/hooks/use-watchlist';
import { calculateReturns, fitGarch11 } from '@/lib/quant-engine';
import {
  canonicalRvLadder,
  canonicalRvSeries,
  curveVerdict,
  sessionDiff,
  vixContext,
  type RvReading,
} from '@/lib/ticker-snapshot';
import type { ChainPayload, IndicesPayload, IvHistoryEntry, PanelRowOut } from '@/components/views/volatility/types';
import type { PanelPayload } from '@/components/views/fno/types';

export interface TickerSnapshotValue {
  ticker: string;
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  ivHistory: IvHistoryEntry[];
  panel: PanelRowOut[];
  /** The /api/fno panel — one fetch shared by the whole options workstation. */
  fnoPanel: PanelPayload | null;
  fnoLoading: boolean;
  chainLoading: boolean;
  refresh: () => void;
  /** One RV ladder for the whole app: sample stdev of log returns × √252. */
  rvLadder: RvReading[];
  rvSeries: RollingVolSeries[];
  /** One GARCH(1,1) fit for the whole vol suite (T3/T8), same sample gate. */
  garchFit: ReturnType<typeof fitGarch11> | null;
  rv20: number | null;
  vix: { level: number | null; percentile: number | null; n: number };
  curve: 'CONTANGO' | 'FLAT' | 'INVERTED' | null;
  session: ReturnType<typeof sessionDiff>;
  /** Next earnings, lifted out of T8 so every setup card can gate on it. */
  earnings: { ticker: string; date: string; isEstimate: boolean; sessions: number } | null;
  /** All cached daily bars for cross-sectional reads. */
  allStockBars: Record<string, PriceBar[]>;
  /** One combined watchlist scan shared by the board and the workstation. */
  watchRows: WatchRow[];
  scanWatchlist: (force?: boolean) => Promise<void>;
  barsSource?: string | null;
  intradaySource?: string | null;
}

const SnapshotContext = createContext<TickerSnapshotValue | null>(null);

export function useTickerSnapshot(): TickerSnapshotValue {
  const ctx = useContext(SnapshotContext);
  if (!ctx) throw new Error('useTickerSnapshot must be used inside <TickerSnapshotProvider>');
  return ctx;
}

export function TickerSnapshotProvider({
  ticker,
  bars,
  barsSource = null,
  intradaySource = null,
  allStockBars = {},
  children,
}: {
  ticker: string;
  bars: PriceBar[];
  barsSource?: string | null;
  intradaySource?: string | null;
  allStockBars?: Record<string, PriceBar[]>;
  children: React.ReactNode;
}) {
  const [chain, setChain] = useState<ChainPayload | null>(null);
  const [indices, setIndices] = useState<IndicesPayload | null>(null);
  const [ivHistory, setIvHistory] = useState<IvHistoryEntry[]>([]);
  const [panel, setPanel] = useState<PanelRowOut[]>([]);
  const [fnoPanel, setFnoPanel] = useState<PanelPayload | null>(null);
  const [earnings, setEarnings] = useState<{ ticker: string; date: string; isEstimate: boolean; sessions: number } | null>(null);
  const watch = useWatchlist();
  const [fnoLoadedKey, setFnoLoadedKey] = useState<string | null>(null);
  const [chainLoading, setChainLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);

  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  // One chain/history/panel fetch for every view. Views no longer refetch the
  // same option chain from four separate places.
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(`/api/options?mode=chain&symbol=${encodeURIComponent(ticker)}`)
        .then((r) => r.json() as Promise<ChainPayload>)
        .catch(() => ({ available: false, reason: 'fetch failed' } as ChainPayload)),
      fetch(`/api/options?mode=history&symbol=${encodeURIComponent(ticker)}`)
        .then((r) => r.json() as Promise<{ history: IvHistoryEntry[] }>)
        .catch(() => ({ history: [] })),
      fetch(`/api/options?mode=panel&symbol=${encodeURIComponent(ticker)}`)
        .then((r) => r.json() as Promise<{ panel?: PanelRowOut[] }>)
        .catch(() => ({ panel: [] as PanelRowOut[] })),
      fetch(`/api/options?mode=earnings&symbol=${encodeURIComponent(ticker)}`)
        .then((r) => (r.ok ? r.json() as Promise<{ available: boolean; nextEarningsDate?: string | null; isEstimate?: boolean }> : null))
        .catch(() => null),
    ])
      .then(([chainData, historyData, panelData, earningsData]) => {
        if (cancelled) return;
        setChain({ ...chainData, requested: chainData.requested ?? ticker.toUpperCase() });
        setIvHistory(Array.isArray(historyData?.history) ? historyData.history : []);
        setPanel(Array.isArray(panelData?.panel) ? panelData.panel : []);
        if (earningsData?.available && earningsData.nextEarningsDate) {
          const today = new Date().toISOString().slice(0, 10);
          setEarnings({
            ticker,
            date: earningsData.nextEarningsDate,
            isEstimate: earningsData.isEstimate !== false,
            sessions: sessionsUntil(earningsData.nextEarningsDate, today),
          });
        } else {
          setEarnings(null);
        }
      })
      .finally(() => { if (!cancelled) setChainLoading(false); });
    return () => { cancelled = true; };
  }, [ticker, refreshKey]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/options?mode=indices')
      .then((r) => r.json() as Promise<IndicesPayload>)
      .then((d) => { if (!cancelled) setIndices(d); })
      .catch(() => { if (!cancelled) setIndices({ available: false, reason: 'fetch failed' }); });
    return () => { cancelled = true; };
  }, [refreshKey]);

  const fnoLoading = fnoLoadedKey !== `${ticker}:${refreshKey}`;
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/fno?mode=panel&symbol=${encodeURIComponent(ticker)}`)
      .then((r) => r.json() as Promise<PanelPayload>)
      .then((d) => { if (!cancelled) { setFnoPanel(d); setFnoLoadedKey(`${ticker}:${refreshKey}`); } })
      .catch(() => { if (!cancelled) { setFnoPanel({ available: false, reason: 'fetch failed' }); setFnoLoadedKey(`${ticker}:${refreshKey}`); } });
    return () => { cancelled = true; };
  }, [ticker, refreshKey]);

  const rvSeries = useMemo(() => canonicalRvSeries(bars.map((b) => b.close)), [bars]);
  const garchFit = useMemo(() => {
    const rets = calculateReturns(bars.map((b) => b.close), 'log');
    return rets.length > 50 ? fitGarch11(rets) : null;
  }, [bars]);
  const rvLadder = useMemo(() => canonicalRvLadder(bars.map((b) => b.close)), [bars]);
  const rv20 = useMemo(() => rvLadder.find((r) => r.window === 20)?.vol ?? null, [rvLadder]);
  const vix = useMemo(
    () => vixContext(indices?.series?.VIX?.bars ?? [], indices?.series?.VIX?.bars?.slice(-1)[0]?.close ?? null),
    [indices]
  );
  const curve = useMemo(() => curveVerdict(chain?.termStructure ?? []), [chain]);
  const session = useMemo(() => sessionDiff(panel), [panel]);

  const value = useMemo<TickerSnapshotValue>(() => ({
    ticker, chain, indices, ivHistory, panel, fnoPanel, fnoLoading, chainLoading, refresh,
    rvLadder, rvSeries, garchFit, rv20, vix, curve, session,
    earnings, allStockBars, watchRows: watch.rows, scanWatchlist: watch.scan,
    barsSource, intradaySource,
  }), [ticker, chain, indices, ivHistory, panel, fnoPanel, fnoLoading, chainLoading, refresh, rvLadder, rvSeries, garchFit, rv20, vix, curve, session, earnings, allStockBars, watch.rows, watch.scan, barsSource, intradaySource]);

  return <SnapshotContext.Provider value={value}>{children}</SnapshotContext.Provider>;
}
