'use client';

import React, { useCallback, useEffect, useState } from 'react';
import {
  LayoutList,
  Waves,
  Scale,
  Crosshair,
  TrendingUp,
  ArrowLeftRight,
  RefreshCw,
  AlertTriangle,
  ChevronDown,
} from 'lucide-react';
import type { PriceBar, IntradayBar } from '@/lib/quant-engine';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { US_UNIVERSE } from '@/lib/fno/config';
import { FnoChainTab } from './fno/FnoChainTab';
import { FnoOiTab } from './fno/FnoOiTab';
import { FnoPcrTab } from './fno/FnoPcrTab';
import { FnoMaxPainTab } from './fno/FnoMaxPainTab';
import { FnoTermStructureTab } from './fno/FnoTermStructureTab';
import { FnoExpectedMoveTab } from './fno/FnoExpectedMoveTab';
import type { PanelPayload } from './fno/types';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { OptionsVerdictStrip } from './OptionsVerdictStrip';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';

interface FnoViewProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
  bars: PriceBar[];
  intradayBars: IntradayBar[];
  barsSource?: string | null;
  companyName: string;
  isLoading: boolean;
}

type SubTab = 'chain' | 'oi' | 'pcr' | 'maxpain' | 'termstructure' | 'expectedmove';

export function FnoView({ ticker, onTickerChange, bars, barsSource, companyName, isLoading }: FnoViewProps) {
  const [panel, setPanel] = useState<PanelPayload | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [subTab, setSubTab] = useState<SubTab>('chain');
  const { chain, rv20 } = useTickerSnapshot();

  // Deep-link: ?fno=pcr restores the selection (replaceState keeps history clean).
  // Deferred 0ms so the setState doesn't cascade synchronously inside the effect.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const t = new URLSearchParams(window.location.search).get('fno');
      const valid: SubTab[] = ['chain', 'oi', 'pcr', 'maxpain', 'termstructure', 'expectedmove'];
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
      url.searchParams.set('fno', id);
      window.history.replaceState(null, '', url.toString());
    }
  };

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/fno?mode=panel&symbol=${encodeURIComponent(ticker.toUpperCase())}`)
      .then((r) => r.json() as Promise<PanelPayload>)
      .then((d) => {
        if (!cancelled) setPanel(d);
      })
      .catch(() => {
        if (!cancelled) setPanel({ available: false, reason: 'fetch failed' });
      })
      .finally(() => {
        if (!cancelled) setRefreshing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, refreshKey]);

  const panelLoading = panel === null;

  const substitution = panel?.substitution ?? null;
  const asOf = panel?.asOf ? new Date(panel.asOf).toLocaleString() : null;
  const spot = panel?.quote?.price ?? null;
  const changePct = panel?.quote?.changePct ?? null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="kicker">Options F&amp;O Suite · US Universe</span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">{companyName} ({ticker})</span>
              <ProvenanceBadge daily={barsSource} />
              {substitution && (
                <span className="flex items-center gap-1 rounded border border-gold-500/40 bg-gold-500/10 px-1.5 py-0.5 font-mono text-[8.5px] font-semibold uppercase tracking-wider text-gold-800 dark:text-gold-200">
                  <AlertTriangle className="h-3 w-3" /> Proxy
                </span>
              )}
            </div>
            <h2 className="mt-1 font-display text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              Where the Positioning Actually Sits
            </h2>
            <p className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
              OI · GEX walls · PCR · max pain · term structure · expected vs realized move
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <select
                value={ticker}
                onChange={(e) => onTickerChange(e.target.value)}
                aria-label="Universe symbol"
                className="appearance-none rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] py-2 pl-3 pr-8 font-mono text-xs font-bold uppercase tracking-wider text-zinc-800 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-gold-400"
              >
                {US_UNIVERSE.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-400" />
            </div>
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
              onClick={() => { setRefreshing(true); setRefreshKey((k) => k + 1); }}
              disabled={refreshing}
              className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[11px] font-semibold text-zinc-600 transition-colors hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-50 disabled:opacity-60"
              aria-label="Refresh chain"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        <div className="mt-3 flex flex-col gap-2 border-t border-[var(--hairline)] pt-3 font-mono text-[10px]">
          {panel?.available ? (
            <>
              <span className="text-[var(--positive)]">
                ✓ Live chain{asOf ? ` · fetched ${asOf}` : ''} · ~15-min delayed quotes · EOD open interest · European BSM approximation
              </span>
              <span className="text-zinc-500 dark:text-zinc-400">
                OI sessions accrued: {panel.sessions ?? 0}{panel.firstDate ? ` (since ${panel.firstDate})` : ''} · {panel.strikesUsed ?? '—'}/{panel.strikesTotal ?? '—'} contracts used
              </span>
            </>
          ) : panelLoading ? (
            <span className="text-zinc-400">Fetching chain &amp; accruing snapshot…</span>
          ) : (
            <span className="flex items-center gap-2 text-rose-500 dark:text-rose-400">
              <AlertTriangle className="h-3.5 w-3.5" /> {panel?.reason ?? 'Chain unavailable'} — try Refresh in a moment.
            </span>
          )}
        </div>
      </div>

      <OptionsVerdictStrip chain={chain} fnoPanel={panel} rv20={rv20} />

      {/* Sub-tab strip — scrollable chips, deep-linked via ?fno= */}
        <AnimatedTabs
          value={subTab}
          onChange={(id) => handleSubTabChange(id as SubTab)}
          tabs={[
            { id: 'chain', label: 'Option Chain', icon: LayoutList, meta: 'T1' },
            { id: 'oi', label: 'OI Positioning', icon: Waves, meta: 'T2' },
            { id: 'pcr', label: 'PCR', icon: Scale, meta: 'T3' },
            { id: 'maxpain', label: 'Max Pain', icon: Crosshair, meta: 'T4' },
            { id: 'termstructure', label: 'Term Structure', icon: TrendingUp, meta: 'T5' },
            { id: 'expectedmove', label: 'Expected vs Realized', icon: ArrowLeftRight, meta: 'T6' },
          ]}
        />

      {/* Loading shimmer */}
      {panelLoading && !panel && (
        <div className="panel animate-pulse p-10 text-center font-mono text-xs uppercase tracking-[0.2em] text-zinc-400">
          Fetching live chain · computing positioning…
        </div>
      )}

      {/* Panels */}
      {subTab === 'chain' && <FnoChainTab panel={panel} symbol={ticker.toUpperCase()} />}
      {subTab === 'oi' && <FnoOiTab panel={panel} symbol={ticker.toUpperCase()} />}
      {subTab === 'pcr' && <FnoPcrTab panel={panel} symbol={ticker.toUpperCase()} />}
      {subTab === 'maxpain' && <FnoMaxPainTab panel={panel} symbol={ticker.toUpperCase()} />}
      {subTab === 'termstructure' && <FnoTermStructureTab panel={panel} symbol={ticker.toUpperCase()} />}
      {subTab === 'expectedmove' && <FnoExpectedMoveTab panel={panel} bars={bars} symbol={ticker.toUpperCase()} />}
    </div>
  );
}
