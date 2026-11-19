'use client';

import React, { useEffect, useState } from 'react';
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
  Scale,
} from 'lucide-react';
import type { PriceBar, IntradayBar } from '@/lib/quant-engine';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import { T1StateTab } from './volatility/T1StateTab';
import { T2SurfaceTab } from './volatility/T2SurfaceTab';
import { T3TermStructureTab } from './volatility/T3TermStructureTab';
import { VolatilityNowStrip } from './volatility/VolatilityNowStrip';
import { OptionsVerdictStrip } from './OptionsVerdictStrip';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { T4ForecastTab } from './volatility/T4ForecastTab';
import { T5DensityTab } from './volatility/T5DensityTab';
import { T6CrossAssetTab } from './volatility/T6CrossAssetTab';
import { T7DealerTab } from './volatility/T7DealerTab';
import { T8DecisionTab } from './volatility/T8DecisionTab';

interface VolatilityViewProps {
  ticker: string;
  bars: PriceBar[];
  intradayBars: IntradayBar[];
  intradaySource?: string | null;
  barsSource?: string | null;
  companyName: string;
  isLoading?: boolean;
}

type SubTab = 't1' | 't2' | 't3' | 't4' | 't5' | 't6' | 't7' | 't8';

export function VolatilityView({
  ticker,
  bars,
  intradayBars,
  intradaySource = null,
  barsSource = null,
  companyName,
}: VolatilityViewProps) {
  const snapshot = useTickerSnapshot();
  const { chain, indices, ivHistory, panel, chainLoading, refresh, rv20 } = snapshot;
  const [subTab, setSubTab] = useState<SubTab>('t1');

  // Deep-link: ?vol=t3 restores the selection (replaceState keeps history clean).
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const t = new URLSearchParams(window.location.search).get('vol');
      const valid: SubTab[] = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'];
      if (t && (valid as string[]).includes(t)) setSubTab(t as SubTab);
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

  const substitution = chain?.substitution ?? null;
  const chainAvailable = chain?.available === true;
  const indicesSynthetic = indices?.source === 'partial_or_fallback';
  const spot = chain?.quote?.price ?? null;
  const changePct = chain?.quote?.changePct ?? null;

  return (
    <div className="space-y-6">
      <div className="panel p-5">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="kicker">Implied Volatility Suite</span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {chain?.quote?.name ?? companyName} ({ticker})
              </span>
              <ProvenanceBadge daily={barsSource} intraday={intradaySource} />
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
                <div className="font-mono text-xl font-bold tabular-nums text-zinc-900 dark:text-zinc-50">{spot.toFixed(2)}</div>
                {changePct !== null && (
                  <div className={`font-mono text-[11px] tabular-nums ${changePct >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'}`}>
                    {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                  </div>
                )}
              </div>
            )}
            <button
              type="button"
              onClick={refresh}
              disabled={chainLoading}
              className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[11px] font-semibold text-zinc-600 transition-colors hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-50 disabled:opacity-60"
              aria-label="Refresh chain"
            >
              <RefreshCw className={`h-4 w-4 ${chainLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-col gap-2 border-t border-[var(--hairline)] pt-3 font-mono text-[10px]">
          {chainAvailable ? (
            <span className="text-[var(--positive)]">
              ✓ Live chain{chain?.asOf ? ` · fetched ${new Date(chain.asOf).toLocaleString()}` : ''} · ~15-min delayed quotes · European BSM approximation
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

      <OptionsVerdictStrip chain={chain} rv20={rv20} />

      <VolatilityNowStrip
        chain={chain}
        indices={indices}
        rv20={rv20}
        barsSource={barsSource}
        onJump={handleSubTabChange}
      />

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
          { id: 't8', label: 'Decision', icon: Scale, meta: 'T8' },
        ]}
      />

      {chainLoading && !chain && (
        <div className="panel animate-pulse p-10 text-center font-mono text-xs uppercase tracking-[0.2em] text-zinc-400">
          Fetching live chain · computing SVI · pricing the tail…
        </div>
      )}

      {subTab === 't1' && <T1StateTab chain={chain} indices={indices} rv20={rv20} vix={snapshot.vix} />}
      {subTab === 't2' && <T2SurfaceTab chain={chain} indices={indices} />}
      {subTab === 't3' && <T3TermStructureTab chain={chain} indices={indices} bars={bars} ticker={ticker} barsSource={barsSource} rvSeriesCanonical={snapshot.rvSeries} vixContextProp={snapshot.vix} garchFitProp={snapshot.garchFit} />}
      {subTab === 't4' && <T4ForecastTab chain={chain} indices={indices} bars={bars} ticker={ticker} ivHistory={ivHistory} />}
      {subTab === 't5' && <T5DensityTab chain={chain} bars={bars} ticker={ticker} />}
      {subTab === 't6' && <T6CrossAssetTab chain={chain} indices={indices} bars={bars} ticker={ticker} />}
      {subTab === 't7' && <T7DealerTab chain={chain} panel={panel} bars={bars} ticker={ticker} />}
      {subTab === 't8' && <T8DecisionTab chain={chain} bars={bars} ticker={ticker} ivHistory={ivHistory} intradayBars={intradayBars} intradaySource={intradaySource} rvSeries={snapshot.rvSeries} garchFitProp={snapshot.garchFit} />}
    </div>
  );
}
