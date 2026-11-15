'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Gauge,
  Waves,
  TrendingUp,
  Crosshair,
  PieChart,
  Network,
  Magnet,
  RefreshCw,
  AlertTriangle,
  Info,
} from 'lucide-react';
import type { PriceBar, IntradayBar } from '@/lib/quant-engine';
import { calculateReturns } from '@/lib/quant-engine';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { T1StateTab } from './volatility/T1StateTab';
import { T2SurfaceTab } from './volatility/T2SurfaceTab';
import { T3TermStructureTab } from './volatility/T3TermStructureTab';
import { T4ForecastTab } from './volatility/T4ForecastTab';
import { T5DensityTab } from './volatility/T5DensityTab';
import { T6CrossAssetTab } from './volatility/T6CrossAssetTab';
import { T7DealerTab } from './volatility/T7DealerTab';
import type { ChainPayload, IndicesPayload, IvHistoryEntry, PanelRowOut } from './volatility/types';

interface VolatilityViewProps {
  ticker: string;
  bars: PriceBar[];
  intradayBars: IntradayBar[];
  companyName: string;
  isLoading: boolean;
}

type SubTab = 't1' | 't2' | 't3' | 't4' | 't5' | 't6' | 't7';

export function VolatilityView({ ticker, bars, companyName, isLoading }: VolatilityViewProps) {
  const [chain, setChain] = useState<ChainPayload | null>(null);
  const [indices, setIndices] = useState<IndicesPayload | null>(null);
  const [ivHistory, setIvHistory] = useState<IvHistoryEntry[]>([]);
  const [panel, setPanel] = useState<PanelRowOut[]>([]);
  const [chainLoading, setChainLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const [subTab, setSubTab] = useState<SubTab>('t1');

  // Deep-link: ?vol=t3 restores the selection (replaceState keeps history clean).
  // Deferred 0ms so the setState doesn't cascade synchronously inside the effect.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const t = new URLSearchParams(window.location.search).get('vol');
      const valid: SubTab[] = ['t1', 't2', 't3', 't4', 't5', 't6', 't7'];
      if (t && (valid as string[]).includes(t)) {
        setSubTab(t as SubTab);
      }
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  const handleSubTabChange = (id: SubTab) => {
    setSubTab(id);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.searchParams.set('vol', id);
      window.history.replaceState(null, '', url.toString());
    }
  };

  // Fetch chain + accrued IV history for the active ticker
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(`/api/options?mode=chain&symbol=${encodeURIComponent(ticker)}`).then((r) => r.json() as Promise<ChainPayload>).catch(() => ({ available: false, reason: 'fetch failed' } as ChainPayload)),
      fetch(`/api/options?mode=history&symbol=${encodeURIComponent(ticker)}`).then((r) => r.json() as Promise<{ history: IvHistoryEntry[] }>).catch(() => ({ history: [] })),
      fetch(`/api/options?mode=panel&symbol=${encodeURIComponent(ticker)}`).then((r) => r.json() as Promise<{ panel?: PanelRowOut[] }>).catch(() => ({ panel: [] as PanelRowOut[] })),
    ])
      .then(([chainData, historyData, panelData]) => {
        if (cancelled) return;
        const requested = ticker.toUpperCase();
        setChain({ ...chainData, requested: chainData.requested ?? requested });
        setIvHistory(Array.isArray(historyData?.history) ? historyData.history : []);
        setPanel(Array.isArray(panelData?.panel) ? panelData.panel : []);
      })
      .finally(() => {
        if (!cancelled) setChainLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, refreshKey]);

  // Fetch vol-index family once (server-cached 15 min)
  useEffect(() => {
    let cancelled = false;
    fetch('/api/options?mode=indices')
      .then((r) => r.json() as Promise<IndicesPayload>)
      .then((d) => {
        if (!cancelled) setIndices(d);
      })
      .catch(() => {
        if (!cancelled) setIndices({ available: false, reason: 'fetch failed' });
      })

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  // Realized vol (20d, annualized) from the active ticker's bars
  const rv20 = useMemo(() => {
    if (!bars || bars.length < 25) return null;
    const closes = bars.map((b) => b.close);
    const rets = calculateReturns(closes, 'log');
    if (rets.length < 20) return null;
    const slice = rets.slice(-20);
    let sum = 0;
    for (const r of slice) sum += r * r;
    return Math.sqrt((sum / 20) * 252);
  }, [bars]);

  const substitution = chain?.substitution ?? null;
  const chainAvailable = chain?.available === true;
  const indicesAvailable = indices?.available === true;
  const indicesSynthetic = indices?.source === 'partial_or_fallback';

  const spot = chain?.quote?.price ?? null;
  const changePct = chain?.quote?.changePct ?? null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="kicker">Implied Volatility Suite</span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {chain?.quote?.name ?? companyName} ({ticker})
              </span>
              {substitution && (
                <span className="flex items-center gap-1 rounded border border-gold-500/40 bg-gold-500/10 px-1.5 py-0.5 font-mono text-[8.5px] font-semibold uppercase tracking-wider text-gold-800 dark:text-gold-200">
                  <Info className="h-3 w-3" /> Proxy
                </span>
              )}
            </div>
            <h2 className="mt-1 font-display text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              Implied Vol &amp; the Price of Uncertainty
            </h2>
            <p className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
              realized vol = the state it has been in · implied vol = the state it expects to be in
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {spot !== null && (
              <div className="text-right">
                <div className="font-mono text-xl font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
                  {spot.toFixed(2)}
                </div>
                {changePct !== null && (
                  <div className={`font-mono text-[11px] tabular-nums ${changePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
                    {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                  </div>
                )}
              </div>
            )}
            <button
              type="button"
              onClick={() => setRefreshKey((k) => k + 1)}
              disabled={chainLoading}
              className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[11px] font-semibold text-zinc-600 transition-colors hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-50 disabled:opacity-60"
              aria-label="Refresh chain"
            >
              <RefreshCw className={`h-4 w-4 ${chainLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Availability / fallback banners */}
        <div className="mt-3 flex flex-col gap-2 border-t border-[var(--hairline)] pt-3 font-mono text-[10px]">
          {chainAvailable ? (
            <span className="text-[var(--positive)]">
              ✓ Live chain{chain?.asOf ? ` · ${new Date(chain.asOf).toLocaleString()}` : ''} · as-of data (~15min delayed) · European BSM approximation
            </span>
          ) : chainLoading ? (
            <span className="text-zinc-400">Fetching option chain &amp; vol indices…</span>
          ) : (
            <span className="flex items-center gap-2 text-rose-500 dark:text-rose-400">
              <AlertTriangle className="h-3.5 w-3.5" /> Chain unavailable: {chain?.reason ?? 'unknown error'} — showing vol-index reference where possible.
            </span>
          )}
          {indicesSynthetic && (
            <span className="flex items-center gap-2 text-gold-700 dark:text-gold-300">
              <AlertTriangle className="h-3.5 w-3.5" /> Vol-index family fell back to synthetic levels — live Yahoo history unreachable right now.
            </span>
          )}
          {substitution && (
            <span className="text-zinc-500 dark:text-zinc-400">
              {substitution}. IV numbers refer to the proxy series, not the raw index.
            </span>
          )}
        </div>
      </div>

      {/* Sub-tab strip — scrollable chips, deep-linked via ?vol= */}
        <AnimatedTabs
          value={subTab}
          onChange={(id) => handleSubTabChange(id as SubTab)}
          tabs={[
            { id: 't1', label: 'State', icon: Gauge, meta: 'T1' },
            { id: 't2', label: 'Surface & Smile', icon: Waves, meta: 'T2' },
            { id: 't3', label: 'Term Structure & VRP', icon: TrendingUp, meta: 'T3' },
            { id: 't4', label: 'Forecast Validation', icon: Crosshair, meta: 'T4' },
            { id: 't5', label: 'Risk-Neutral Density', icon: PieChart, meta: 'T5' },
            { id: 't6', label: 'Cross-Asset & Regime', icon: Network, meta: 'T6' },
            { id: 't7', label: 'Dealer Positioning & Trap', icon: Magnet, meta: 'T7' },
          ]}
        />

      {/* Loading shimmer while the chain lands */}
      {chainLoading && !chain && (
        <div className="panel animate-pulse p-10 text-center font-mono text-xs uppercase tracking-[0.2em] text-zinc-400">
          Fetching live chain · computing SVI · pricing the tail…
        </div>
      )}

      {/* Sub-tab bodies */}
      {subTab === 't1' && <T1StateTab chain={chain} indices={indices} rv20={rv20} />}
      {subTab === 't2' && <T2SurfaceTab chain={chain} indices={indices} />}
      {subTab === 't3' && <T3TermStructureTab chain={chain} indices={indices} bars={bars} ticker={ticker} />}
      {subTab === 't4' && (
        <T4ForecastTab chain={chain} indices={indices} bars={bars} ticker={ticker} ivHistory={ivHistory} />
      )}
      {subTab === 't5' && <T5DensityTab chain={chain} bars={bars} ticker={ticker} />}
      {subTab === 't6' && <T6CrossAssetTab chain={chain} indices={indices} bars={bars} ticker={ticker} />}
      {subTab === 't7' && <T7DealerTab chain={chain} panel={panel} bars={bars} ticker={ticker} />}
    </div>
  );
}
