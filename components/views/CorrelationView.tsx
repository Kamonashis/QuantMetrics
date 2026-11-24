'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, BookOpen, Layers, Network, PieChart, Plus, RefreshCw, ShieldAlert, X } from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import { MAX_ASSETS, buildCorrelationSample, correlationMatrix, summarizeBasket } from '@/lib/correlation-engine';
import { NowTab } from '@/components/views/correlation/NowTab';
import { OverTimeTab } from '@/components/views/correlation/OverTimeTab';
import { StressTab } from '@/components/views/correlation/StressTab';
import { PortfolioTab } from '@/components/views/correlation/PortfolioTab';
import { StructureTab } from '@/components/views/correlation/StructureTab';
import { MethodsTab } from '@/components/views/correlation/MethodsTab';
import type { Pair } from '@/components/views/correlation/shared';
import { Callout } from '@/components/ui/lab-ui';
import { TickerSuggest } from '@/components/ui/TickerSuggest';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { AnimatedTabs } from '@/components/ui/animated-tabs';

interface CorrelationViewProps {
  tickersInput: string;
  onTickersInputChange: (val: string) => void;
  startDate: string;
  onStartDateChange: (val: string) => void;
  endDate: string;
  onEndDateChange: (val: string) => void;
  returnType: 'simple' | 'log';
  allStockBars: { [ticker: string]: PriceBar[] };
  isLoading: boolean;
  onRunCorrelation: (tickers: string[]) => void | Promise<void>;
  sourcesByTicker?: Record<string, string>;
}

type TabId = 'now' | 'time' | 'stress' | 'portfolio' | 'structure' | 'methods';
const TAB_IDS: TabId[] = ['now', 'time', 'stress', 'portfolio', 'structure', 'methods'];
const LEGACY_TABS: Record<string, TabId> = { overview: 'now', dynamics: 'time', risk: 'portfolio' };
const SUGGESTIONS = ['SPY', 'QQQ', 'TLT', 'GLD', 'JPM', 'XOM'];
const FETCH_CHUNK = 10;

function parseTickers(raw: string): string[] {
  return [...new Set(raw.split(',').map((t) => t.trim().toUpperCase()).filter(Boolean))];
}

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

const dateClass =
  'w-full rounded-lg border border-zinc-200 bg-zinc-50 px-2 py-1.5 font-mono text-[11px] text-zinc-900 focus:outline-none focus:ring-1 focus:ring-gold-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100';

export function CorrelationView({
  tickersInput,
  onTickersInputChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  returnType,
  allStockBars,
  isLoading,
  onRunCorrelation,
  sourcesByTicker = {},
}: CorrelationViewProps) {
  const tickers = useMemo(() => parseTickers(tickersInput).slice(0, MAX_ASSETS), [tickersInput]);
  const [tab, setTab] = useState<TabId>('now');
  const [draft, setDraft] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [pairChoice, setPairChoice] = useState<Pair | null>(null);
  const [weightsByTicker, setWeightsByTicker] = useState<Record<string, number>>({});

  // Deep link: ?corr=time restores the tab (older ids are mapped).
  useEffect(() => {
    const timer = setTimeout(() => {
      const raw = new URLSearchParams(window.location.search).get('corr');
      const id = raw ? LEGACY_TABS[raw] ?? raw : null;
      if (id && (TAB_IDS as string[]).includes(id)) setTab(id as TabId);
    }, 0);
    return () => clearTimeout(timer);
  }, []);
  const changeTab = (id: TabId) => {
    setTab(id);
    const url = new URL(window.location.href);
    url.searchParams.set('corr', id);
    window.history.replaceState(null, '', url.toString());
  };

  // Fetch any series the basket needs but the cache has never seen.
  const missingBars = useMemo(() => tickers.filter((t) => !(allStockBars[t]?.length > 0)), [tickers, allStockBars]);
  const requestedRef = useRef<Set<string>>(new Set());
  const onRunRef = useRef(onRunCorrelation);
  useEffect(() => {
    onRunRef.current = onRunCorrelation;
  }, [onRunCorrelation]);
  const missingKey = missingBars.join(',');
  useEffect(() => {
    const toFetch = missingKey ? missingKey.split(',').filter((t) => !requestedRef.current.has(t)) : [];
    if (toFetch.length === 0) return;
    const timer = setTimeout(() => {
      toFetch.forEach((t) => requestedRef.current.add(t));
      void (async () => {
        for (const group of chunk(toFetch, FETCH_CHUNK)) await onRunRef.current(group);
      })();
    }, 0);
    return () => clearTimeout(timer);
  }, [missingKey]);

  const refreshAll = () => {
    void (async () => {
      for (const group of chunk(tickers, FETCH_CHUNK)) await onRunCorrelation(group);
    })();
  };

  const sampleResult = useMemo(
    () => buildCorrelationSample(allStockBars, tickers, { startDate, endDate, returnType }),
    [allStockBars, tickers, startDate, endDate, returnType]
  );
  const sample = sampleResult.ok ? sampleResult.sample : null;
  const corr = useMemo(() => (sample ? correlationMatrix(sample.returns) : null), [sample]);
  const summary = useMemo(() => (sample && corr ? summarizeBasket(corr, sample.tickers) : null), [sample, corr]);

  const pair: Pair =
    pairChoice && pairChoice[0] !== pairChoice[1] && tickers.includes(pairChoice[0]) && tickers.includes(pairChoice[1])
      ? pairChoice
      : [tickers[0] ?? '', tickers[1] ?? ''];

  const weightsPct = useMemo(() => tickers.map((t) => weightsByTicker[t] ?? 100 / Math.max(1, tickers.length)), [tickers, weightsByTicker]);
  const setWeights = (w: number[]) => setWeightsByTicker(Object.fromEntries(tickers.map((t, i) => [t, w[i] ?? 0])));

  const setBasket = (list: string[]) => onTickersInputChange(list.join(', '));
  const addTicker = (raw: string) => {
    const sym = parseTickers(raw)[0];
    if (!sym) return;
    if (tickers.includes(sym)) {
      setFormError(`${sym} is already in the basket.`);
      setDraft('');
      return;
    }
    if (tickers.length >= MAX_ASSETS) {
      setFormError(`The basket holds at most ${MAX_ASSETS} assets. Remove one first.`);
      setDraft('');
      return;
    }
    setFormError(null);
    setDraft('');
    setBasket([...tickers, sym]);
  };
  const removeTicker = (sym: string) => {
    setFormError(null);
    setBasket(tickers.filter((t) => t !== sym));
  };

  const syntheticTickers = tickers.filter((t) => (sourcesByTicker[t] ?? 'synthetic_gbm') !== 'live_market');
  const allLive = syntheticTickers.length === 0;
  // The data layer always lands bars (live or simulated fallback), so "missing" means "in flight".
  const fetching = missingBars.length > 0;
  const audit = sampleResult.ok ? sampleResult.sample.audit : sampleResult.audit;

  const tabs = [
    { id: 'now', label: 'Now', icon: Layers },
    { id: 'time', label: 'Over time', icon: Activity },
    { id: 'stress', label: 'Under stress', icon: ShieldAlert },
    { id: 'portfolio', label: 'Portfolio', icon: PieChart },
    { id: 'structure', label: 'Structure', icon: Network },
    { id: 'methods', label: 'Methods & data', icon: BookOpen },
  ];

  return (
    <div className="space-y-6">
      <div className="panel p-5">
        <div className="flex flex-col justify-between gap-3 border-b border-[var(--hairline)] pb-4 lg:flex-row lg:items-end">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="kicker">Correlation Lab</span>
              <ProvenanceBadge daily={allLive ? 'live_market' : 'synthetic_gbm'} />
            </div>
            <h2 className="mt-1 font-display text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">Do these assets really diversify each other?</h2>
            <p className="mt-1 max-w-2xl text-xs text-zinc-500 dark:text-zinc-400">
              Build a basket. The lab measures how the assets move together today, whether that is changing, what happens in sell-offs, and what
              it means for a portfolio.
            </p>
          </div>
          <button
            type="button"
            onClick={refreshAll}
            disabled={isLoading}
            className="inline-flex cursor-pointer items-center gap-1.5 self-start rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[11px] font-semibold text-zinc-700 hover:border-gold-500/50 disabled:opacity-50 dark:text-zinc-300 lg:self-auto"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh data
          </button>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div>
            <span className="mb-1.5 block text-xs font-medium text-zinc-700 dark:text-zinc-300">
              Basket <span className="font-normal text-zinc-400">({tickers.length} of up to {MAX_ASSETS})</span>
            </span>
            <ul className="flex flex-wrap items-center gap-1.5" aria-label="Assets in the basket">
              {tickers.map((t) => {
                const live = (sourcesByTicker[t] ?? 'synthetic_gbm') === 'live_market';
                return (
                  <li
                    key={t}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-gold-500/40 bg-gold-500/10 py-1 pl-2.5 pr-1 font-mono text-xs font-bold text-gold-900 dark:text-gold-100"
                  >
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-[var(--positive)]' : 'bg-amber-500'}`}
                      title={live ? 'Live data' : 'Simulated data'}
                      aria-label={live ? 'Live data' : 'Simulated data'}
                    />
                    {t}
                    <button
                      type="button"
                      onClick={() => removeTicker(t)}
                      disabled={tickers.length <= 2}
                      aria-label={`Remove ${t}`}
                      className="cursor-pointer rounded p-0.5 text-gold-800/70 hover:bg-gold-500/20 hover:text-gold-900 disabled:cursor-not-allowed disabled:opacity-30 dark:text-gold-200/70 dark:hover:text-gold-100"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </li>
                );
              })}
            </ul>
            <form
              className="mt-2.5 flex flex-wrap items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                addTicker(draft);
              }}
            >
              <div className="w-52">
                <TickerSuggest
                  id="correlation-add-ticker"
                  ariaLabel="Add an asset"
                  value={draft}
                  onChange={setDraft}
                  onCommit={addTicker}
                  placeholder="Add a ticker"
                  inputClassName="w-full px-3 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-gold-400"
                />
              </div>
              {SUGGESTIONS.filter((s) => !tickers.includes(s)).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => addTicker(s)}
                  className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-[var(--hairline)] px-2 py-1 font-mono text-[11px] text-zinc-600 transition-colors hover:border-gold-500/50 dark:text-zinc-300"
                >
                  <Plus className="h-3 w-3 text-zinc-400" />
                  {s}
                </button>
              ))}
            </form>
            {formError && <p className="mt-2 text-xs text-[var(--negative)]">{formError}</p>}
          </div>
          <div>
            <span className="mb-1.5 block text-xs font-medium text-zinc-700 dark:text-zinc-300">Date range</span>
            <div className="grid grid-cols-2 gap-1.5">
              <input type="date" aria-label="Start date" value={startDate} onChange={(e) => onStartDateChange(e.target.value)} className={dateClass} />
              <input type="date" aria-label="End date" value={endDate} onChange={(e) => onEndDateChange(e.target.value)} className={dateClass} />
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg bg-[var(--bg-inset)] px-3 py-2 font-mono text-[11px] text-zinc-600 dark:text-zinc-400">
          {fetching ? (
            <span className="inline-flex items-center gap-1.5">
              <RefreshCw className="h-3 w-3 animate-spin" /> Loading price history for {missingBars.join(', ')}…
            </span>
          ) : sample ? (
            <>
              <span>
                <strong className="text-zinc-900 dark:text-zinc-100">{audit.returnObs.toLocaleString()}</strong> daily returns
              </span>
              <span>
                {audit.start} → {audit.end}
              </span>
              <span>
                {returnType} returns · {audit.series.every((s) => s.basis === 'adjusted') ? 'dividend-adjusted' : 'some close-only prices'}
              </span>
              {audit.droppedSessions > 0 && <span>{audit.droppedSessions} sessions dropped to align calendars</span>}
              <button type="button" onClick={() => changeTab('methods')} className="cursor-pointer text-gold-700 underline-offset-2 hover:underline dark:text-gold-300">
                full audit
              </button>
            </>
          ) : (
            <span>No usable sample yet.</span>
          )}
        </div>
      </div>

      {!allLive && !fetching && (
        <Callout tone="warn" title="Some inputs are simulated">
          {syntheticTickers.join(', ')} {syntheticTickers.length === 1 ? 'is' : 'are'} using the offline price generator because live data has
          not loaded. Simulated prices do not reflect real co-movement, so every correlation involving them is meaningless. Use Refresh data to
          retry.
        </Callout>
      )}

      <AnimatedTabs tabs={tabs} value={tab} onChange={(id) => changeTab(id as TabId)} panelId="correlation-lab-panel" />

      <div id="correlation-lab-panel" role="tabpanel">
        {!sample || !corr || !summary ? (
          fetching ? (
            <div className="panel p-8 text-center text-xs text-zinc-500 dark:text-zinc-400">Fetching data…</div>
          ) : (
            <Callout tone="warn" title="Not enough data">
              {sampleResult.ok ? 'No sample.' : sampleResult.reason} Widen the date range, remove an asset with a short history, or use Refresh
              data.
            </Callout>
          )
        ) : (
          <>
            {tab === 'now' && <NowTab sample={sample} corr={corr} summary={summary} pair={pair} onPairChange={setPairChoice} />}
            {tab === 'time' && <OverTimeTab sample={sample} pair={pair} onPairChange={setPairChoice} />}
            {tab === 'stress' && <StressTab sample={sample} pair={pair} onPairChange={setPairChoice} weights={weightsPct} />}
            {tab === 'portfolio' && <PortfolioTab sample={sample} weightsPct={weightsPct} onWeightsChange={setWeights} />}
            {tab === 'structure' && <StructureTab sample={sample} corr={corr} pair={pair} onPairChange={setPairChoice} />}
            {tab === 'methods' && <MethodsTab sample={sample} sourcesByTicker={sourcesByTicker} />}
          </>
        )}
      </div>
    </div>
  );
}
