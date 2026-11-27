'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowLeftRight,
  Crosshair,
  Gauge,
  Magnet,
  Network,
  PieChart,
  RefreshCw,
  Scale,
  Waves,
  Info,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { PriceBar, IntradayBar } from '@/lib/quant-engine';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { OptionsVerdictStrip } from './OptionsVerdictStrip';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import { T1StateTab } from './volatility/T1StateTab';
import { T2SurfaceTab } from './volatility/T2SurfaceTab';
import { T3TermStructureTab } from './volatility/T3TermStructureTab';
import { T4ForecastTab } from './volatility/T4ForecastTab';
import { T5DensityTab } from './volatility/T5DensityTab';
import { T5Calibration } from './volatility/T5Calibration';
import { buildRndResearch, MAX_SPOT_GAP, spotGap } from '@/lib/rnd-integrity';
import { useOptionsResearch } from '@/hooks/use-options-research';
import { trailingResearchRv } from '@/lib/options-research';
import { T6CrossAssetTab } from './volatility/T6CrossAssetTab';
import { T7DealerTab } from './volatility/T7DealerTab';
import { T8DecisionTab } from './volatility/T8DecisionTab';
import { FnoChainTab } from './fno/FnoChainTab';
import { FnoOiTab } from './fno/FnoOiTab';
import { FnoPcrTab } from './fno/FnoPcrTab';
import { FnoMaxPainTab } from './fno/FnoMaxPainTab';
import { FnoTermStructureTab } from './fno/FnoTermStructureTab';
import { FnoExpectedMoveTab } from './fno/FnoExpectedMoveTab';

type TabId = 'v1' | 'v2' | 'v3' | 'v4' | 'v5' | 'v6' | 'v7' | 'v8' | 'f1' | 'f2' | 'f3' | 'f4' | 'f5' | 'f6';

interface WorkstationTabDef {
  id: TabId;
  label: string;
  icon: LucideIcon;
  meta: string;
}

const VOL_TABS: WorkstationTabDef[] = [
  { id: 'v1', label: 'State', icon: Gauge, meta: 'V1' },
  { id: 'v2', label: 'Surface & Smile', icon: Waves, meta: 'V2' },
  { id: 'v3', label: 'Term Structure & VRP', icon: Activity, meta: 'V3' },
  { id: 'v4', label: 'Forecast Validation', icon: Crosshair, meta: 'V4' },
  { id: 'v5', label: 'Risk-Neutral Density', icon: PieChart, meta: 'V5' },
  { id: 'v6', label: 'Cross-Asset & Regime', icon: Network, meta: 'V6' },
  { id: 'v8', label: 'Decision', icon: Scale, meta: 'V8' },
];

const POSITION_TABS: WorkstationTabDef[] = [
  { id: 'v7', label: 'Gamma Exposure (GEX)', icon: Magnet, meta: 'GEX' },
  { id: 'f1', label: 'Option Chain', icon: Activity, meta: 'F1' },
  { id: 'f2', label: 'OI Positioning', icon: Waves, meta: 'F2' },
  { id: 'f3', label: 'PCR', icon: Scale, meta: 'F3' },
  { id: 'f4', label: 'Max Pain', icon: Crosshair, meta: 'F4' },
  { id: 'f5', label: 'F&O Term Structure', icon: Activity, meta: 'F5' },
  { id: 'f6', label: 'Expected vs Realized', icon: ArrowLeftRight, meta: 'F6' },
];

const FNO_PARAM_TO_TAB: Record<string, TabId> = {
  gex: 'v7',
  chain: 'f1',
  oi: 'f2',
  pcr: 'f3',
  maxpain: 'f4',
  termstructure: 'f5',
  expectedmove: 'f6',
};

const TAB_TO_FNO_PARAM: Partial<Record<TabId, string>> = {
  v7: 'gex',
  f1: 'chain',
  f2: 'oi',
  f3: 'pcr',
  f4: 'maxpain',
  f5: 'termstructure',
  f6: 'expectedmove',
};

function resolveTabFromLocation(initialGroup: 'vol' | 'fno', search: string): TabId {
  if (search) {
    const p = new URLSearchParams(search);
    const f = p.get('fno');
    if (f && FNO_PARAM_TO_TAB[f]) return FNO_PARAM_TO_TAB[f];
    const v = p.get('vol');
    if (v && /^t[1-8]$/.test(v)) return `v${v[1]}` as TabId;
  }
  return initialGroup === 'fno' ? 'v7' : 'v1';
}

/**
 * One options workstation. The former "Implied Vol" and "F&O" peer views are
 * sections of this single page: one header, one provenance read, one verdict,
 * one refresh. Both deep-link params (?vol=, ?fno=) still work.
 */
export function OptionsWorkstation({
  ticker,
  bars,
  intradayBars,
  intradaySource,
  barsSource,
  companyName,
  initialGroup = 'vol',
}: {
  ticker: string;
  bars: PriceBar[];
  intradayBars: IntradayBar[];
  intradaySource?: string | null;
  barsSource?: string | null;
  companyName: string;
  initialGroup?: 'vol' | 'fno';
}) {
  const s = useTickerSnapshot();
  const [tab, setTab] = useState<TabId>(() =>
    resolveTabFromLocation(initialGroup, typeof window !== 'undefined' ? window.location.search : '')
  );

  // Sync section switches (Options · Vol <-> Options · Position) and browser history
  // while honoring explicit ?fno= / ?vol= deep-link overrides.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const syncFromUrl = () => {
      setTab(resolveTabFromLocation(initialGroup, window.location.search));
    };
    const timer = setTimeout(syncFromUrl, 0);
    window.addEventListener('popstate', syncFromUrl);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('popstate', syncFromUrl);
    };
  }, [initialGroup]);

  const selectTab = (id: TabId) => {
    setTab(id);
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    url.searchParams.delete('vol');
    url.searchParams.delete('fno');
    const fnoSlug = TAB_TO_FNO_PARAM[id];
    if (fnoSlug) {
      url.searchParams.set('fno', fnoSlug);
    } else if (id.startsWith('v')) {
      url.searchParams.set('vol', `t${id[1]}`);
    }
    window.history.replaceState(null, '', url.toString());
  };

  const substitution = s.chain?.substitution ?? s.fnoPanel?.substitution ?? null;
  const chainQuote = s.chain?.quote;
  const fnoQuote = s.fnoPanel?.quote;
  const headerQuote = chainQuote ?? fnoQuote;
  const matchedFno = s.chain?.symbol && s.chain.symbol === s.fnoPanel?.symbol ? fnoQuote : undefined;
  const underlyingQuote = s.chain?.symbol === ticker.toUpperCase() ? s.latestQuotes[ticker.toUpperCase()] : undefined;
  const gap = spotGap(chainQuote, matchedFno);
  const underlyingGap = spotGap(chainQuote, underlyingQuote);
  const loading = s.chainLoading || s.fnoLoading;
  const rnd = useMemo(() => buildRndResearch(s.chain, matchedFno, underlyingQuote), [s.chain, matchedFno, underlyingQuote]);
  const rv20 = useMemo(() => barsSource === 'live_market' && !substitution ? trailingResearchRv(bars, chainQuote?.asOf) : null,
    [bars, barsSource, substitution, chainQuote?.asOf]);
  const ledger = useOptionsResearch({ chain: s.chain, research: rnd, rv20, loading, showLedger: tab === 'v5' });

  return (
    <div className="space-y-4">
      <div className="panel p-5">
        <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="kicker">Options Workstation</span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">{s.chain?.quote?.name ?? s.fnoPanel?.quote?.name ?? companyName} ({ticker})</span>
              <ProvenanceBadge daily={barsSource} intraday={intradaySource} />
              {substitution && (
                <span className="flex items-center gap-1 rounded border border-gold-500/40 bg-gold-500/10 px-1.5 py-0.5 font-mono text-[8.5px] font-semibold uppercase tracking-wider text-gold-800 dark:text-gold-200">
                  <Info className="h-3 w-3" /> Proxy
                </span>
              )}
            </div>
            <h2 className="mt-1 font-display text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
              Price, Positioning &amp; Decision — One Surface
            </h2>
            <p className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-400 dark:text-zinc-500">
              volatility panels V1–V6, V8 · positioning panels GEX &amp; F1–F6 · one refresh, one verdict
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {headerQuote && <div className="font-mono tabular-nums">
              <div className="text-[10px] text-zinc-500">{chainQuote ? 'Chain spot' : 'Positioning spot'}</div>
              <div className="text-xl font-bold text-zinc-900 dark:text-zinc-50">{headerQuote.price.toFixed(2)}</div>
              {headerQuote.prevClose != null && <div className="text-[10px] text-zinc-500">Prior close {headerQuote.prevClose.toFixed(2)}</div>}
            </div>}
            <button
              type="button"
              onClick={s.refresh}
              disabled={loading}
              aria-label="Refresh all options data"
              className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 font-mono text-[11px] font-semibold disabled:opacity-60"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
            </button>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-3 font-mono text-[10px]">
          <span className={s.chain?.available ? 'text-[var(--positive)]' : 'text-zinc-400'}>
            chain: {s.chain?.available ? `Yahoo${chainQuote?.asOf ? ` · quoted ${new Date(chainQuote.asOf).toLocaleString()}` : ' · quote time unavailable'}${s.chain.asOf ? ` · fetched ${new Date(s.chain.asOf).toLocaleString()}` : ''}` : 'unavailable'}
          </span>
          <span className={s.fnoPanel?.available ? 'text-[var(--positive)]' : 'text-zinc-400'}>positioning: {fnoQuote ? `${fnoQuote.price.toFixed(2)} · ${fnoQuote.asOf ? `quoted ${new Date(fnoQuote.asOf).toLocaleString()}` : 'quote time unavailable'} · EOD OI` : 'unavailable'}</span>
          {gap !== null && gap > MAX_SPOT_GAP && <span className="font-semibold text-amber-700 dark:text-amber-400">Chain and positioning spot differ by {(gap * 100).toFixed(2)}%</span>}
          {underlyingQuote && <span className={underlyingGap !== null && underlyingGap > MAX_SPOT_GAP ? 'font-semibold text-amber-700 dark:text-amber-400' : 'text-zinc-400'}>
            market-data spot: {underlyingQuote.price.toFixed(2)} · quoted {new Date(underlyingQuote.asOf).toLocaleString()}{underlyingGap !== null && underlyingGap > MAX_SPOT_GAP && ` · differs by ${(underlyingGap * 100).toFixed(2)}%`}
          </span>}
          <span className="text-zinc-400">quotes: ~15-min delayed European BSM · not executable</span>
          {substitution && <span className="text-zinc-400">{substitution}</span>}
        </div>
      </div>

      <OptionsVerdictStrip chain={s.chain} fnoPanel={s.fnoPanel} rv20={rv20} rndQuality={rnd?.quality} />

      <div className="space-y-2.5">
        <div className="space-y-1">
          <div className={`font-mono text-[9px] font-semibold uppercase tracking-[0.16em] ${tab !== 'v7' && tab.startsWith('v') ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-400 dark:text-zinc-500'}`}>
            Volatility · V1–V6, V8
          </div>
          <AnimatedTabs
            value={tab}
            onChange={(id) => selectTab(id as TabId)}
            tabs={VOL_TABS}
          />
        </div>
        <div className="space-y-1">
          <div className={`font-mono text-[9px] font-semibold uppercase tracking-[0.16em] ${tab === 'v7' || tab.startsWith('f') ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-400 dark:text-zinc-500'}`}>
            Positioning · GEX &amp; F1–F6
          </div>
          <AnimatedTabs
            value={tab}
            onChange={(id) => selectTab(id as TabId)}
            tabs={POSITION_TABS}
          />
        </div>
      </div>

      {tab === 'v1' && <T1StateTab chain={s.chain} indices={s.indices} rv20={s.rv20} vix={s.vix} />}
      {tab === 'v2' && <T2SurfaceTab chain={s.chain} indices={s.indices} />}
      {tab === 'v3' && <T3TermStructureTab chain={s.chain} indices={s.indices} bars={bars} ticker={ticker} barsSource={barsSource} rvSeriesCanonical={s.rvSeries} vixContextProp={s.vix} garchFitProp={s.garchFit} />}
      {tab === 'v4' && <T4ForecastTab chain={s.chain} indices={s.indices} bars={bars} ticker={ticker} ivHistory={s.ivHistory} />}
      {tab === 'v5' && <><T5DensityTab chain={s.chain} bars={bars} barsSource={barsSource} ticker={ticker} research={rnd} /><T5Calibration {...ledger} /></>}
      {tab === 'v6' && <T6CrossAssetTab chain={s.chain} indices={s.indices} bars={bars} ticker={ticker} />}
      {tab === 'v7' && <T7DealerTab chain={s.chain} panel={s.panel} bars={bars} ticker={ticker} />}
      {tab === 'v8' && <T8DecisionTab chain={s.chain} bars={bars} ticker={ticker} ivHistory={s.ivHistory} intradayBars={intradayBars} intradaySource={intradaySource} rvSeries={s.rvSeries} garchFitProp={s.garchFit} earningsProp={s.earnings} />}
      {tab === 'f1' && <FnoChainTab panel={s.fnoPanel} symbol={ticker} trap={s.fnoPanel?.trap ?? s.chain?.trap ?? null} />}
      {tab === 'f2' && <FnoOiTab panel={s.fnoPanel} symbol={ticker} trap={s.fnoPanel?.trap ?? s.chain?.trap ?? null} />}
      {tab === 'f3' && <FnoPcrTab panel={s.fnoPanel} symbol={ticker} />}
      {tab === 'f4' && <FnoMaxPainTab panel={s.fnoPanel} symbol={ticker} trap={s.fnoPanel?.trap ?? s.chain?.trap ?? null} />}
      {tab === 'f5' && <FnoTermStructureTab panel={s.fnoPanel} symbol={ticker} />}
      {tab === 'f6' && <FnoExpectedMoveTab panel={s.fnoPanel} bars={bars} symbol={ticker} />}
    </div>
  );
}
