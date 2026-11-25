'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef, startTransition } from 'react';
import { TickerSnapshotProvider } from '@/components/providers/TickerSnapshotProvider';
import dynamic from 'next/dynamic';
import { PriceBar, IntradayBar } from '@/lib/quant-engine';
import {
  generateRealisticStockHistory,
  generateRealisticIntradayHistory,
  KNOWN_TICKERS,
  type LiveQuoteTick,
} from '@/lib/market-service';
import {
  buildStreamTickerList,
  isEndDateUpToPresent,
  reconcileBarsWithLiveTick,
  pruneTickerCache,
} from '@/lib/market-cache';
import { useMarketStream } from '@/hooks/use-market-stream';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';
import { HomeView } from '@/components/views/HomeView';

// Non-landing views are code-split: each loads only when first opened, so the
// Home view no longer parses every engine, tab and chart in the terminal.
function ViewLoading() {
  return (
    <div className="panel flex h-64 items-center justify-center">
      <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">Loading view…</span>
    </div>
  );
}
const EdgeView = dynamic(() => import('@/components/views/EdgeView').then((m) => m.EdgeView), { ssr: false, loading: ViewLoading });
const AnalysisView = dynamic(() => import('@/components/views/AnalysisView').then((m) => m.AnalysisView), { ssr: false, loading: ViewLoading });
const OptionsWorkstation = dynamic(() => import('@/components/views/OptionsWorkstation').then((m) => m.OptionsWorkstation), { ssr: false, loading: ViewLoading });
const ModelingView = dynamic(() => import('@/components/views/ModelingView').then((m) => m.ModelingView), { ssr: false, loading: ViewLoading });
const CorrelationView = dynamic(() => import('@/components/views/CorrelationView').then((m) => m.CorrelationView), { ssr: false, loading: ViewLoading });
const RegressionView = dynamic(() => import('@/components/views/RegressionView').then((m) => m.RegressionView), { ssr: false, loading: ViewLoading });

type AppView = 'now' | 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression' | 'volatility' | 'fno';
const VALID_VIEWS: AppView[] = ['now', 'home', 'analysis', 'modeling', 'correlation', 'regression', 'volatility', 'fno'];
const DEFAULT_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'] as const;
/** Looked-up tickers kept resident beyond the protected set (LRU). */
const MAX_EXTRA_DAILY_TICKERS = 6;
/** Intraday 1-minute series kept resident (most recently fetched). */
const MAX_INTRADAY_TICKERS = 3;
const NO_PROTECTED: ReadonlySet<string> = new Set<string>();

interface LiveQuoteEntry {
  price: number;
  changePct: number;
  asOf: string;
}

export default function App() {
  // Navigation — default landing page is Home (Edge remains available via the nav).
  const [activeView, setActiveView] = useState<AppView>('home');

  // Dark / Light Theme: default to true on both server and client for matching hydration
  const [isDark, setIsDark] = useState<boolean>(true);

  // Synchronize document theme class and saved preference asynchronously after hydration
  useEffect(() => {
    const timer = setTimeout(() => {
      const saved = localStorage.getItem('qt_theme');
      if (saved) {
        const darkPref = saved === 'dark';
        setIsDark(darkPref);
        if (darkPref) {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      } else if (window.matchMedia && !window.matchMedia('(prefers-color-scheme: dark)').matches) {
        setIsDark(false);
        document.documentElement.classList.remove('dark');
      }
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  const handleToggleTheme = () => {
    setIsDark((prev) => {
      const next = !prev;
      localStorage.setItem('qt_theme', next ? 'dark' : 'light');
      if (next) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      return next;
    });
  };

  // Deterministic initial date range for SSR/client parity. The initial end-date renders as this
  // constant on both server and client; a mount effect brings it forward to the real current day
  // (hydration-safe — same pattern as the theme sync).
  const todayStr = '2026-11-08';
  const defaultStartDateStr = '2020-01-15';

  // Shared Analysis Parameters — declared before navigation so URL state can reference them.
  const [activeTicker, setActiveTicker] = useState<string>('AAPL');
  const [resolution, setResolution] = useState<'daily' | 'intraday'>('daily');
  const [startDate, setStartDate] = useState<string>(defaultStartDateStr);
  const [endDate, setEndDate] = useState<string>(todayStr);
  const [returnType, setReturnType] = useState<'simple' | 'log'>('log');
  const [rollingWindows, setRollingWindows] = useState<string>('10,20,30');
  const [decompPeriod, setDecompPeriod] = useState<number>(21);
  const [decompModel, setDecompModel] = useState<'additive' | 'multiplicative'>('multiplicative');

  // Deep-link: ?view=analysis (plus tab params like ?tab=risk) restores the
  // top-level view on mount. Deferred 0ms so the setState doesn't cascade in the effect.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const p = new URLSearchParams(window.location.search);
      const v = p.get('view');
      if (v && (VALID_VIEWS as string[]).includes(v)) setActiveView(v as AppView);
      const t = p.get('ticker');
      if (t) setActiveTicker(t.toUpperCase());
      const s = p.get('start');
      if (s) setStartDate(s);
      const e = p.get('end');
      if (e) setEndDate(e);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  // Warm the code-split view chunks once the browser is idle after first paint.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const warm = () => {
      void import('@/components/views/EdgeView');
      void import('@/components/views/AnalysisView');
      void import('@/components/views/ModelingView');
      void import('@/components/views/OptionsWorkstation');
      void import('@/components/views/CorrelationView');
      void import('@/components/views/RegressionView');
    };
    if (typeof window.requestIdleCallback === 'function') {
      const id = window.requestIdleCallback(warm, { timeout: 6000 });
      return () => window.cancelIdleCallback(id);
    }
    const id = setTimeout(warm, 2500);
    return () => clearTimeout(id);
  }, []);

  const navigateView = useCallback((view: AppView, params?: Record<string, string>) => {
    // Low-priority render: the nav pill animation and input stay responsive
    // while the destination view mounts its charts.
    startTransition(() => setActiveView(view));
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      // Clear stale sub-tab keys so a bookmark always lands on the intended panel.
      ['tab', 'vol', 'fno', 'corr'].forEach((k) => url.searchParams.delete(k));
      url.searchParams.set('view', view);
      url.searchParams.set('ticker', activeTicker);
      url.searchParams.set('start', startDate);
      url.searchParams.set('end', endDate);
      Object.entries(params ?? {}).forEach(([k, v]) => url.searchParams.set(k, v));
      window.history.pushState(null, '', url.toString());
    }
  }, [activeTicker, startDate, endDate]);

  // Sync the default end date to the actual current day on mount (SSR/client parity preserved —
  // first paint uses the deterministic constant, the effect moves it to "today").
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const timer = setTimeout(() => {
      const now = new Date();
      const local = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      setEndDate((prev) => (prev === todayStr ? local : prev));
    }, 0);
    return () => clearTimeout(timer);
  }, [todayStr]);

  // Multi-stock parameters
  const [correlationTickers, setCorrelationTickers] = useState<string>(
    'AAPL, MSFT, GOOGL, NVDA, TSLA, SPY'
  );
  const [regressionTickers, setRegressionTickers] = useState<string>(
    'AAPL, SPY'
  );

  // Market Data In-Memory Cache with lazy instant initialization (defaulting to 10 years of history)
  const [stockBarsCache, setStockBarsCache] = useState<{ [ticker: string]: PriceBar[] }>(() => {
    const initialCache: { [ticker: string]: PriceBar[] } = {};
    DEFAULT_UNIVERSE.forEach((sym) => {
      initialCache[sym] = generateRealisticStockHistory(sym, 2600);
    });
    return initialCache;
  });

  // Dedicated latest live quotes map so TickerTape & Preset Universe always display
  // the latest live market prices even if the user slices endDate to a historical date.
  const [latestQuotes, setLatestQuotes] = useState<Record<string, LiveQuoteEntry>>({});
  const [initialLiveFetchDone, setInitialLiveFetchDone] = useState<boolean>(false);
  const tickerSeqRef = useRef<Record<string, number>>({});

  // 1-Minute Intraday Data In-Memory Cache (5-7 trading days). Only the active
  // ticker's intraday bars are ever read, so only the initial ticker is seeded
  // (others fall back to a memoized synthetic series) and the cache is bounded.
  const [intradayBarsCache, setIntradayBarsCache] = useState<{ [ticker: string]: IntradayBar[] }>(() => ({
    AAPL: generateRealisticIntradayHistory('AAPL', 7),
  }));
  // Source per ticker for the intraday cache — the seeded default is synthetic, always labelled as such.
  const [intradaySourceCache, setIntradaySourceCache] = useState<{ [ticker: string]: string }>(() => {
    const init: { [ticker: string]: string } = {};
    DEFAULT_UNIVERSE.forEach((s) => {
      init[s] = 'synthetic_gbm';
    });
    return init;
  });

  // Source per ticker for the daily-bar cache — seeded as synthetic until a live fetch lands.
  const [stockSourceCache, setStockSourceCache] = useState<{ [ticker: string]: string }>(() => {
    const init: { [ticker: string]: string } = {};
    DEFAULT_UNIVERSE.forEach((s) => {
      init[s] = 'synthetic_gbm';
    });
    return init;
  });

  const [companyNames, setCompanyNames] = useState<{ [ticker: string]: string }>(() => {
    const initialNames: { [ticker: string]: string } = {};
    DEFAULT_UNIVERSE.forEach((sym) => {
      initialNames[sym] = KNOWN_TICKERS[sym]?.name || `${sym} Equity`;
    });
    return initialNames;
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Cache bounds: symbols that must stay resident (default universe, active
  // ticker, both lab baskets) are protected; other looked-up tickers are kept
  // LRU up to a small cap so a long browsing session cannot grow memory
  // without bound (each ticker = up to 20y of daily bars).
  const protectedSymbolsRef = useRef<Set<string>>(new Set(DEFAULT_UNIVERSE));
  useEffect(() => {
    const parse = (s: string) => s.split(',').map((t) => t.trim().toUpperCase()).filter(Boolean);
    protectedSymbolsRef.current = new Set<string>([
      ...DEFAULT_UNIVERSE,
      activeTicker,
      ...parse(correlationTickers),
      ...parse(regressionTickers),
    ]);
  }, [activeTicker, correlationTickers, regressionTickers]);
  const boundDaily = useCallback(
    <T,>(next: Record<string, T>, touched: string[]) =>
      pruneTickerCache(next, touched, protectedSymbolsRef.current, MAX_EXTRA_DAILY_TICKERS),
    []
  );
  // Intraday is only read for the active ticker: keep the most recent few.
  const boundIntraday = useCallback(
    <T,>(next: Record<string, T>, touched: string[]) =>
      pruneTickerCache(next, touched, NO_PROTECTED, MAX_INTRADAY_TICKERS),
    []
  );

  // Fetch ticker data (daily and intraday) from API
  const fetchTickers = useCallback(
    async (
      tickersToFetch: string[],
      customRange: string = '10y',
      reqStartDate?: string,
      reqEndDate?: string,
      options?: { omitEndDate?: boolean; skipIntraday?: boolean; quotesOnly?: boolean }
    ) => {
      const syms = tickersToFetch
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0);
      if (syms.length === 0) return;

      const reqSeqBySym: Record<string, number> = {};
      if (!options?.quotesOnly) {
        syms.forEach((sym) => {
          const nextSeq = (tickerSeqRef.current[sym] ?? 0) + 1;
          tickerSeqRef.current[sym] = nextSeq;
          reqSeqBySym[sym] = nextSeq;
        });
      }

      if (!options?.quotesOnly) setIsLoading(true);
      try {
        const effectiveStart = reqStartDate ?? startDate;
        const effectiveEnd = options?.omitEndDate ? undefined : (reqEndDate ?? endDate);

        let rangeParam = customRange;
        if (effectiveStart) {
          const startYear = new Date(effectiveStart).getFullYear();
          const curYear = new Date().getFullYear();
          if (curYear - startYear > 10) {
            rangeParam = '20y';
          }
        }

        const params = new URLSearchParams({
          tickers: syms.join(','),
          range: rangeParam,
        });
        if (effectiveStart) params.set('startDate', effectiveStart);
        if (effectiveEnd) params.set('endDate', effectiveEnd);

        // 1. Fetch Daily Data (no-store so browser never returns stale cached quotes)
        const res = await fetch(`/api/market-data?${params.toString()}`, { cache: 'no-store' });
        if (res.ok) {
          const json = await res.json();
          if (json.success && json.data) {
            type ApiEntry = {
              bars: PriceBar[];
              name: string;
              source?: string;
              latestQuote?: { price: number; prevClose: number; changePct: number; asOf: string };
            };
            const entries = Object.entries(json.data as Record<string, ApiEntry>);

            setLatestQuotes((prev) => {
              const updated = { ...prev };
              for (const [sym, payload] of entries) {
                if (payload.source === 'live_market' && payload.latestQuote) {
                  updated[sym] = {
                    price: payload.latestQuote.price,
                    changePct: payload.latestQuote.changePct,
                    asOf: payload.latestQuote.asOf,
                  };
                }
              }
              return updated;
            });

            if (!options?.quotesOnly) {
              const isLatestForSym = (sym: string) =>
                reqSeqBySym[sym] === undefined || reqSeqBySym[sym] === tickerSeqRef.current[sym];

              setStockBarsCache((prev) => {
                const updated = { ...prev };
                const touched: string[] = [];
                for (const [sym, payload] of entries) {
                  if (isLatestForSym(sym) && payload.bars && payload.bars.length > 0) {
                    updated[sym] = payload.bars;
                    touched.push(sym);
                  }
                }
                return boundDaily(updated, touched);
              });
              setStockSourceCache((prev) => {
                const updated = { ...prev };
                for (const [sym, payload] of entries) {
                  if (isLatestForSym(sym) && payload.source && payload.bars && payload.bars.length > 0) {
                    updated[sym] = payload.source;
                  }
                }
                return updated;
              });
              setCompanyNames((prev) => {
                const updated = { ...prev };
                for (const [sym, payload] of entries) {
                  if (isLatestForSym(sym) && payload.name) {
                    updated[sym] = payload.name;
                  }
                }
                return updated;
              });
            }
          }
        }

        // 2. Fetch 1-Minute Intraday Data for primary ticker
        const primarySym = options?.skipIntraday ? null : syms[0];
        if (primarySym) {
          const intraRes = await fetch(
            `/api/market-data?ticker=${encodeURIComponent(primarySym)}&interval=1m&range=7d`,
            { cache: 'no-store' }
          );
          if (intraRes.ok) {
            const intraJson = await intraRes.json();
            if (intraJson.success && intraJson.data?.[primarySym]?.intradayBars) {
              const intraPayload = intraJson.data[primarySym];
              const intraSource = typeof intraPayload.source === 'string' ? intraPayload.source : null;
              setIntradayBarsCache((prev) =>
                boundIntraday({ ...prev, [primarySym]: intraPayload.intradayBars }, [primarySym])
              );
              setIntradaySourceCache((prev) => ({
                ...prev,
                [primarySym]: intraSource ?? 'synthetic_gbm',
              }));
            }
          }
        }
      } catch (err) {
        console.warn('Market data fetch error, using local quantitative generator:', err);
        setStockBarsCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => {
            if (!updated[sym]) {
              updated[sym] = generateRealisticStockHistory(sym, 2600, undefined, startDate);
            }
          });
          return boundDaily(updated, syms);
        });
        setStockSourceCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => { updated[sym] = 'synthetic_gbm'; });
          return updated;
        });
        // Intraday: no per-symbol seeding — the active ticker falls back to a
        // memoized synthetic series (labelled synthetic) until a fetch lands.
      } finally {
        if (!options?.quotesOnly) setIsLoading(false);
      }
    },
    [startDate, endDate, boundDaily, boundIntraday]
  );

  // Browser Back/Forward restores the full context (view + ticker + dates), not just a tab.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onPopState = () => {
      const p = new URLSearchParams(window.location.search);
      const v = p.get('view');
      if (v && (VALID_VIEWS as string[]).includes(v)) setActiveView(v as AppView);
      const t = p.get('ticker');
      if (t) {
        const next = t.toUpperCase();
        setActiveTicker(next);
        fetchTickers([next]);
      }
      const s = p.get('start');
      if (s) setStartDate(s);
      const e = p.get('end');
      if (e) setEndDate(e);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [fetchTickers]);

  // Automatically fetch full historical bars for all 10 universe symbols on mount
  // (omitting endDate so Yahoo returns up to the latest live price, avoiding the stale SSR
  // placeholder endDate race). Ongoing live updates arrive via the SSE stream below.
  const fetchTickersRef = useRef(fetchTickers);
  const endDateRef = useRef(endDate);
  const stockSourceRef = useRef(stockSourceCache);
  useEffect(() => {
    fetchTickersRef.current = fetchTickers;
  }, [fetchTickers]);
  useEffect(() => {
    endDateRef.current = endDate;
  }, [endDate]);
  useEffect(() => {
    stockSourceRef.current = stockSourceCache;
  }, [stockSourceCache]);

  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => {
      void fetchTickersRef
        .current([...DEFAULT_UNIVERSE], '10y', defaultStartDateStr, undefined, {
          omitEndDate: true,
        })
        .finally(() => {
          if (active) setInitialLiveFetchDone(true);
        });
    }, 0);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [defaultStartDateStr]);

  // Subscribe to Server-Sent Events live market stream (with visibility-aware pause/resume
  // and 60s HTTP polling fallback). Every live tick re-validates latestQuotes and patches/appends
  // into stockBarsCache (when viewing up-to-present) so all views & TickerSnapshotProvider update.
  const streamTickers = useMemo(
    () => buildStreamTickerList(DEFAULT_UNIVERSE, activeTicker),
    [activeTicker]
  );

  const handleStreamTicks = useCallback(
    (ticks: LiveQuoteTick[]) => {
      if (!ticks || ticks.length === 0) return;

      const now = new Date();
      const todayIso = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      const allowBarPatch = isEndDateUpToPresent(endDateRef.current, todayIso, todayStr);

      // 1. Propagate to latestQuotes (only allocate a new object if any value changed)
      setLatestQuotes((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const tick of ticks) {
          if (tick.source !== 'live_market') continue;
          const existing = prev[tick.ticker];
          if (
            !existing ||
            existing.price !== tick.price ||
            existing.changePct !== tick.changePct ||
            existing.asOf !== tick.asOf
          ) {
            next[tick.ticker] = {
              price: tick.price,
              changePct: tick.changePct,
              asOf: tick.asOf,
            };
            changed = true;
          }
        }
        return changed ? next : prev;
      });

      // 2. Re-validate & propagate latest OHLCV bar into stockBarsCache when viewing present horizon
      if (allowBarPatch) {
        setStockBarsCache((prev) => {
          let changed = false;
          const next = { ...prev };
          for (const tick of ticks) {
            const res = reconcileBarsWithLiveTick(prev[tick.ticker], tick, {
              allowPatchOrAppend: true,
              existingSource: stockSourceRef.current[tick.ticker],
            });
            if (res.changed && res.bars) {
              next[tick.ticker] = res.bars;
              changed = true;
            }
          }
          return changed ? next : prev;
        });
      }

      // 3. Keep company names populated for streamed symbols
      setCompanyNames((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const tick of ticks) {
          if (tick.source === 'live_market' && tick.name && !prev[tick.ticker]) {
            next[tick.ticker] = tick.name;
            changed = true;
          }
        }
        return changed ? next : prev;
      });

      setInitialLiveFetchDone(true);
    },
    [todayStr]
  );

  useMarketStream({
    tickers: streamTickers,
    intervalMs: 30_000,
    fallbackPollMs: 60_000,
    onTicks: handleStreamTicks,
  });

  // Re-validate 1-minute intraday bars every 60s when viewing intraday resolution
  useEffect(() => {
    if (resolution !== 'intraday' || typeof window === 'undefined') return;
    let active = true;

    const refreshIntraday = async () => {
      if (!active || document.visibilityState === 'hidden') return;
      try {
        const res = await fetch(
          `/api/market-data?ticker=${encodeURIComponent(activeTicker)}&interval=1m&range=7d`,
          { cache: 'no-store' }
        );
        if (!res.ok || !active) return;
        const json = await res.json();
        const payload = json?.data?.[activeTicker];
        if (json?.success && payload?.intradayBars?.length > 0 && active) {
          setIntradayBarsCache((prev) =>
            boundIntraday({ ...prev, [activeTicker]: payload.intradayBars }, [activeTicker])
          );
          setIntradaySourceCache((prev) => ({
            ...prev,
            [activeTicker]: typeof payload.source === 'string' ? payload.source : 'synthetic_gbm',
          }));
        }
      } catch {
        // Keep existing cached intraday bars on transient error
      }
    };

    const intervalId = setInterval(() => {
      void refreshIntraday();
    }, 60_000);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, [activeTicker, resolution, boundIntraday]);

  // Handle start date changes: automatically query earlier historical periods if date exceeds current cache
  const handleStartDateChange = useCallback(
    (newStart: string) => {
      setStartDate(newStart);
      const existingBars = stockBarsCache[activeTicker];
      if (existingBars && existingBars.length > 0 && newStart < existingBars[0].date) {
        fetchTickers([activeTicker], '20y', newStart, endDate);
      }
    },
    [activeTicker, stockBarsCache, fetchTickers, endDate]
  );

  // One ticker change for the whole terminal — called by the global navbar control.
  const lastRequestedTickerRef = useRef(activeTicker);
  const handleGlobalTickerChange = useCallback(
    (symbol: string) => {
      const next = symbol.trim().toUpperCase();
      if (!next || next === lastRequestedTickerRef.current) return;
      lastRequestedTickerRef.current = next;
      setActiveTicker(next);
      fetchTickers([next]);
      if (typeof window !== 'undefined') {
        const url = new URL(window.location.href);
        url.searchParams.set('ticker', next);
        window.history.pushState(null, '', url.toString());
      }
    },
    [fetchTickers]
  );

  // Build the ticker tape prioritizing the latest live quotes so TickerTape & Preset Universe
  // always display up-to-date market prices (and avoid flashing synthetic seed numbers on load).
  const tapeItems = useMemo(() => {
    const orderedSymbols: string[] = [...DEFAULT_UNIVERSE];
    for (const sym of Object.keys(stockBarsCache)) {
      if (!orderedSymbols.includes(sym)) orderedSymbols.push(sym);
    }

    return orderedSymbols
      .filter((sym) => {
        const bars = stockBarsCache[sym];
        return Boolean(latestQuotes[sym] || (bars && bars.length >= 2));
      })
      .map((symbol) => {
        const live = latestQuotes[symbol];
        if (live) {
          return {
            symbol,
            price: live.price,
            changePct: live.changePct,
            asOf: live.asOf,
            pending: false,
          };
        }
        const bars = stockBarsCache[symbol];
        const last = bars[bars.length - 1].close;
        const prev = bars[bars.length - 2].close;
        const isLiveSource = stockSourceCache[symbol] === 'live_market';
        return {
          symbol,
          price: last,
          changePct: prev > 0 ? ((last - prev) / prev) * 100 : 0,
          asOf: bars[bars.length - 1].date,
          pending: !initialLiveFetchDone && !isLiveSource,
        };
      })
      .slice(0, 12);
  }, [stockBarsCache, stockSourceCache, latestQuotes, initialLiveFetchDone]);

  // Active ticker bars
  // Fallbacks are generated once per ticker/start (not on every render) while a fetch is pending.
  const cachedBars = stockBarsCache[activeTicker];
  const fallbackBars = useMemo(
    () => (cachedBars ? null : generateRealisticStockHistory(activeTicker, 2600, undefined, startDate)),
    [cachedBars, activeTicker, startDate]
  );
  const currentBars = (cachedBars || fallbackBars) as PriceBar[];
  const cachedIntraday = intradayBarsCache[activeTicker];
  const fallbackIntraday = useMemo(
    () => (cachedIntraday ? null : generateRealisticIntradayHistory(activeTicker, 7)),
    [cachedIntraday, activeTicker]
  );
  const currentIntradayBars = (cachedIntraday || fallbackIntraday) as IntradayBar[];
  // Provenance follows the bars actually shown: an evicted or not-yet-fetched ticker renders
  // the memoized synthetic fallback, which must never carry a live label.
  const currentIntradaySource = cachedIntraday ? intradaySourceCache[activeTicker] ?? 'synthetic_gbm' : 'synthetic_gbm';
  const currentBarsSource = cachedBars ? stockSourceCache[activeTicker] ?? 'synthetic_gbm' : 'synthetic_gbm';
  const currentCompanyName =
    companyNames[activeTicker] ||
    KNOWN_TICKERS[activeTicker]?.name ||
    `${activeTicker} Equity`;
  const currentQuote = useMemo(() => {
    const live = latestQuotes[activeTicker];
    if (live) return { price: live.price, changePct: live.changePct };
    if (currentBars.length < 2) return { price: null as number | null, changePct: null as number | null };
    const last = currentBars[currentBars.length - 1].close;
    const prev = currentBars[currentBars.length - 2].close;
    return { price: last, changePct: prev > 0 ? ((last - prev) / prev) * 100 : null };
  }, [activeTicker, latestQuotes, currentBars]);

  // Inputs for the app-load TCN decomposition forecast (same as the Analysis view's).
  const decompositionParams = useMemo(
    () => ({ startDate, endDate, period: decompPeriod, model: decompModel }),
    [startDate, endDate, decompPeriod, decompModel]
  );

  return (
    <TickerSnapshotProvider
      ticker={activeTicker}
      bars={currentBars}
      barsSource={currentBarsSource}
      intradaySource={currentIntradaySource}
      allStockBars={stockBarsCache}
      latestQuotes={latestQuotes}
      decompositionParams={decompositionParams}
    >
    <div className="min-h-dvh bg-[#F5F4F1] dark:bg-[#08080A] text-zinc-900 dark:text-[#ECECF1] flex flex-col font-sans transition-colors pb-[calc(3.5rem+env(safe-area-inset-bottom,0px))] md:pb-0">
      {/* Top Navigation */}
      <Navbar
        activeView={activeView}
        onViewChange={navigateView}
        isDark={isDark}
        onToggleTheme={handleToggleTheme}
        activeTicker={activeTicker}
        onTickerChange={handleGlobalTickerChange}
        dailySource={currentBarsSource}
        intradaySource={currentIntradaySource}
        price={currentQuote.price}
        changePct={currentQuote.changePct}
      />

      {/* Main Content Area — fluid shell: 2.2vw gutters, capped only ≥1920px */}
      <main className="flex-1 w-full mx-auto px-[clamp(16px,2.2vw,48px)] pt-4 sm:pt-6 [@media(min-width:1920px)]:max-w-[2400px]">
        {activeView === 'now' && (
          <EdgeView
            ticker={activeTicker}
            companyName={currentCompanyName}
            bars={currentBars}
            tapeItems={tapeItems}
            onNavigate={navigateView}
          />
        )}

        {activeView === 'home' && (
          <HomeView
            onNavigate={navigateView}
            ticker={activeTicker}
            companyName={currentCompanyName}
            bars={currentBars}
            barsSource={currentBarsSource}
            tapeItems={tapeItems}
            onSelectPreset={handleGlobalTickerChange}
          />
        )}

        {activeView === 'analysis' && (
          <AnalysisView
            ticker={activeTicker}
            onTickerChange={(t) => {
              setActiveTicker(t);
              fetchTickers([t]);
            }}
            startDate={startDate}
            onStartDateChange={handleStartDateChange}
            endDate={endDate}
            onEndDateChange={setEndDate}
            returnType={returnType}
            onReturnTypeChange={setReturnType}
            rollingWindows={rollingWindows}
            onRollingWindowsChange={setRollingWindows}
            decompPeriod={decompPeriod}
            onDecompPeriodChange={setDecompPeriod}
            decompModel={decompModel}
            onDecompModelChange={setDecompModel}
            bars={currentBars}
            intradayBars={currentIntradayBars}
            intradaySource={currentIntradaySource}
            barsSource={currentBarsSource}
            resolution={resolution}
            onResolutionChange={setResolution}
            isLoading={isLoading}
            onRunAnalysis={() => fetchTickers([activeTicker], '10y', startDate, endDate)}
            companyName={currentCompanyName}
          />
        )}

        {(activeView === 'volatility' || activeView === 'fno') && (
          <OptionsWorkstation
            ticker={activeTicker}
            bars={currentBars}
            intradayBars={currentIntradayBars}
            intradaySource={currentIntradaySource}
            barsSource={currentBarsSource}
            companyName={currentCompanyName}
            initialGroup={activeView === 'fno' ? 'fno' : 'vol'}
          />
        )}

        {activeView === 'modeling' && (
          <ModelingView
            ticker={activeTicker}
            startDate={startDate}
            endDate={endDate}
            returnType={returnType}
            bars={currentBars}
            intradayBars={currentIntradayBars}
            barsSource={currentBarsSource}
            intradaySource={currentIntradaySource}
            resolution={resolution}
            onResolutionChange={setResolution}
            companyName={currentCompanyName}
            onNavigateToAnalysis={() => navigateView('analysis')}
          />
        )}

        {activeView === 'correlation' && (
          <CorrelationView
            tickersInput={correlationTickers}
            onTickersInputChange={setCorrelationTickers}
            startDate={startDate}
            onStartDateChange={handleStartDateChange}
            endDate={endDate}
            onEndDateChange={setEndDate}
            returnType={returnType}
            allStockBars={stockBarsCache}
            isLoading={isLoading}
            onRunCorrelation={(tickers) => fetchTickers(tickers, '10y', startDate, endDate, { skipIntraday: true })}
            sourcesByTicker={stockSourceCache}
          />
        )}

        {activeView === 'regression' && (
          <RegressionView
            regressionTickers={regressionTickers}
            onRegressionTickersChange={setRegressionTickers}
            startDate={startDate}
            onStartDateChange={handleStartDateChange}
            endDate={endDate}
            onEndDateChange={setEndDate}
            returnType={returnType}
            allStockBars={stockBarsCache}
            isLoading={isLoading}
            onRunRegression={(tickers) => fetchTickers(tickers, '10y', startDate, endDate, { skipIntraday: true })}
            barsSource={currentBarsSource}
            sourcesByTicker={stockSourceCache}
          />
        )}
      </main>

      <Footer />
    </div>
    </TickerSnapshotProvider>
  );
}
