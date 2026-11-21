'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { TickerSnapshotProvider } from '@/components/providers/TickerSnapshotProvider';
import { EdgeView } from '@/components/views/EdgeView';
import { PriceBar, IntradayBar } from '@/lib/quant-engine';
import {
  generateRealisticStockHistory,
  generateRealisticIntradayHistory,
  KNOWN_TICKERS,
} from '@/lib/market-service';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';
import { HomeView } from '@/components/views/HomeView';
import { AnalysisView } from '@/components/views/AnalysisView';
import { OptionsWorkstation } from '@/components/views/OptionsWorkstation';
import { ModelingView } from '@/components/views/ModelingView';
import { CorrelationView } from '@/components/views/CorrelationView';
import { RegressionView } from '@/components/views/RegressionView';

type AppView = 'now' | 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression' | 'volatility' | 'fno';
const VALID_VIEWS: AppView[] = ['now', 'home', 'analysis', 'modeling', 'correlation', 'regression', 'volatility', 'fno'];

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

  // Deterministic 1-year date range for SSR/client parity. The initial end-date renders as this
  // constant on both server and client; a mount effect brings it forward to the real current day
  // (hydration-safe — same pattern as the theme sync).
  const todayStr = '2026-11-08';
  const oneYearAgoStr = '2025-11-08';

  // Shared Analysis Parameters — declared before navigation so URL state can reference them.
  const [activeTicker, setActiveTicker] = useState<string>('AAPL');
  const [resolution, setResolution] = useState<'daily' | 'intraday'>('daily');
  const [startDate, setStartDate] = useState<string>(oneYearAgoStr);
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

  const navigateView = useCallback((view: AppView, params?: Record<string, string>) => {
    setActiveView(view);
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
    'AAPL, SPY, QQQ'
  );

  // Market Data In-Memory Cache with lazy instant initialization (defaulting to 10 years of history)
  const [stockBarsCache, setStockBarsCache] = useState<{ [ticker: string]: PriceBar[] }>(() => {
    const initialCache: { [ticker: string]: PriceBar[] } = {};
    const defaultSymbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'];
    defaultSymbols.forEach((sym) => {
      initialCache[sym] = generateRealisticStockHistory(sym, 2600);
    });
    return initialCache;
  });

  // 1-Minute Intraday Data In-Memory Cache (5-7 trading days)
  const [intradayBarsCache, setIntradayBarsCache] = useState<{ [ticker: string]: IntradayBar[] }>(() => {
    const initialCache: { [ticker: string]: IntradayBar[] } = {};
    const defaultSymbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'];
    defaultSymbols.forEach((sym) => {
      initialCache[sym] = generateRealisticIntradayHistory(sym, 7);
    });
    return initialCache;
  });
  // Source per ticker for the intraday cache — the seeded default is synthetic, always labelled as such.
  const [intradaySourceCache, setIntradaySourceCache] = useState<{ [ticker: string]: string }>(() => {
    const init: { [ticker: string]: string } = {};
    ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'].forEach((s) => {
      init[s] = 'synthetic_gbm';
    });
    return init;
  });

  // Source per ticker for the daily-bar cache — seeded as synthetic until a live fetch lands.
  const [stockSourceCache, setStockSourceCache] = useState<{ [ticker: string]: string }>(() => {
    const init: { [ticker: string]: string } = {};
    ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'].forEach((s) => {
      init[s] = 'synthetic_gbm';
    });
    return init;
  });

  const [companyNames, setCompanyNames] = useState<{ [ticker: string]: string }>(() => {
    const initialNames: { [ticker: string]: string } = {};
    const defaultSymbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'];
    defaultSymbols.forEach((sym) => {
      initialNames[sym] = KNOWN_TICKERS[sym]?.name || `${sym} Equity`;
    });
    return initialNames;
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch ticker data (daily and intraday) from API
  const fetchTickers = useCallback(
    async (
      tickersToFetch: string[],
      customRange: string = '10y',
      reqStartDate?: string,
      reqEndDate?: string
    ) => {
      const syms = tickersToFetch
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0);
      if (syms.length === 0) return;

      setIsLoading(true);
      try {
        const effectiveStart = reqStartDate || startDate;
        const effectiveEnd = reqEndDate || endDate;

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

        // 1. Fetch Daily Data
        const res = await fetch(`/api/market-data?${params.toString()}`);
        if (res.ok) {
          const json = await res.json();
          if (json.success && json.data) {
            setStockBarsCache((prev) => {
              const updated = { ...prev };
              for (const [sym, payload] of Object.entries(
                json.data as Record<string, { bars: PriceBar[]; name: string; source?: string }>
              )) {
                if (payload.bars && payload.bars.length > 0) {
                  updated[sym] = payload.bars;
                }
              }
              return updated;
            });
            setStockSourceCache((prev) => {
              const updated = { ...prev };
              for (const [sym, payload] of Object.entries(
                json.data as Record<string, { bars: PriceBar[]; name: string; source?: string }>
              )) {
                if (payload.source && payload.bars && payload.bars.length > 0) {
                  updated[sym] = payload.source;
                }
              }
              return updated;
            });
            setCompanyNames((prev) => {
              const updated = { ...prev };
              for (const [sym, payload] of Object.entries(
                json.data as Record<string, { bars: PriceBar[]; name: string; source?: string }>
              )) {
                if (payload.name) {
                  updated[sym] = payload.name;
                }
              }
              return updated;
            });
          }
        }

        // 2. Fetch 1-Minute Intraday Data for primary ticker
        const primarySym = syms[0];
        if (primarySym) {
          const intraRes = await fetch(
            `/api/market-data?ticker=${encodeURIComponent(primarySym)}&interval=1m&range=7d`
          );
          if (intraRes.ok) {
            const intraJson = await intraRes.json();
            if (intraJson.success && intraJson.data?.[primarySym]?.intradayBars) {
              const intraPayload = intraJson.data[primarySym];
              const intraSource = typeof intraPayload.source === 'string' ? intraPayload.source : null;
              setIntradayBarsCache((prev) => ({
                ...prev,
                [primarySym]: intraPayload.intradayBars,
              }));
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
          return updated;
        });
        setStockSourceCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => { updated[sym] = 'synthetic_gbm'; });
          return updated;
        });
        setIntradayBarsCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => {
            if (!updated[sym]) {
              updated[sym] = generateRealisticIntradayHistory(sym, 7);
            }
          });
          return updated;
        });
        setIntradaySourceCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => {
            if (!updated[sym]) updated[sym] = 'synthetic_gbm';
          });
          return updated;
        });
      } finally {
        setIsLoading(false);
      }
    },
    [startDate, endDate]
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

  // Automatically fetch fresh live market data on initial load
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchTickers(['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY']);
    }, 0);
    return () => clearTimeout(timer);
  }, [fetchTickers]);

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

  // Build the ticker tape from whatever market data is currently cached
  const tapeItems = useMemo(
    () =>
      Object.entries(stockBarsCache)
        .filter(([, bars]) => bars && bars.length >= 2)
        .map(([symbol, bars]) => {
          const last = bars[bars.length - 1].close;
          const prev = bars[bars.length - 2].close;
          return {
            symbol,
            price: last,
            changePct: prev > 0 ? ((last - prev) / prev) * 100 : 0,
          };
        })
        .slice(0, 12),
    [stockBarsCache]
  );

  // Active ticker bars
  const currentBars =
    stockBarsCache[activeTicker] ||
    generateRealisticStockHistory(activeTicker, 2600, undefined, startDate);
  const currentIntradayBars =
    intradayBarsCache[activeTicker] || generateRealisticIntradayHistory(activeTicker, 7);
  const currentIntradaySource = intradaySourceCache[activeTicker] ?? 'synthetic_gbm';
  const currentBarsSource = stockSourceCache[activeTicker] ?? 'synthetic_gbm';
  const currentCompanyName =
    companyNames[activeTicker] ||
    KNOWN_TICKERS[activeTicker]?.name ||
    `${activeTicker} Equity`;
  const currentQuote = useMemo(() => {
    if (currentBars.length < 2) return { price: null as number | null, changePct: null as number | null };
    const last = currentBars[currentBars.length - 1].close;
    const prev = currentBars[currentBars.length - 2].close;
    return { price: last, changePct: prev > 0 ? ((last - prev) / prev) * 100 : null };
  }, [currentBars]);

  return (
    <TickerSnapshotProvider
      ticker={activeTicker}
      bars={currentBars}
      barsSource={currentBarsSource}
      intradaySource={currentIntradaySource}
      allStockBars={stockBarsCache}
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
            bars={currentBars}
            barsSource={currentBarsSource}
            tapeItems={tapeItems}
            onSelectPreset={(ticker) => {
              setActiveTicker(ticker);
              fetchTickers([ticker]);
            }}
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
            onRunCorrelation={(tickers) => fetchTickers(tickers, '10y', startDate, endDate)}
            intradayBars={currentIntradayBars}
            barsSource={currentBarsSource}
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
            onRunRegression={(tickers) => fetchTickers(tickers, '10y', startDate, endDate)}
            barsSource={currentBarsSource}
          />
        )}
      </main>

      <Footer />
    </div>
    </TickerSnapshotProvider>
  );
}
