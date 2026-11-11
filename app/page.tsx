'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
import { ModelingView } from '@/components/views/ModelingView';
import { CorrelationView } from '@/components/views/CorrelationView';
import { RegressionView } from '@/components/views/RegressionView';

export default function App() {
  // Navigation
  const [activeView, setActiveView] = useState<
    'home' | 'analysis' | 'modeling' | 'correlation' | 'regression'
  >('home');

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

  // Deterministic 1-year date range for SSR/client parity
  const todayStr = '2026-11-08';
  const oneYearAgoStr = '2025-11-08';

  // Shared Analysis Parameters
  const [activeTicker, setActiveTicker] = useState<string>('AAPL');
  const [resolution, setResolution] = useState<'daily' | 'intraday'>('daily');
  const [startDate, setStartDate] = useState<string>(oneYearAgoStr);
  const [endDate, setEndDate] = useState<string>(todayStr);
  const [returnType, setReturnType] = useState<'simple' | 'log'>('log');
  const [rollingWindows, setRollingWindows] = useState<string>('30,60,90');
  const [decompPeriod, setDecompPeriod] = useState<number>(30);
  const [decompModel, setDecompModel] = useState<'additive' | 'multiplicative'>('additive');

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
                json.data as Record<string, { bars: PriceBar[]; name: string }>
              )) {
                if (payload.bars && payload.bars.length > 0) {
                  updated[sym] = payload.bars;
                }
              }
              return updated;
            });
            setCompanyNames((prev) => {
              const updated = { ...prev };
              for (const [sym, payload] of Object.entries(
                json.data as Record<string, { bars: PriceBar[]; name: string }>
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
              setIntradayBarsCache((prev) => ({
                ...prev,
                [primarySym]: intraJson.data[primarySym].intradayBars,
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
        setIntradayBarsCache((prev) => {
          const updated = { ...prev };
          syms.forEach((sym) => {
            if (!updated[sym]) {
              updated[sym] = generateRealisticIntradayHistory(sym, 7);
            }
          });
          return updated;
        });
      } finally {
        setIsLoading(false);
      }
    },
    [startDate, endDate]
  );

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
  const currentCompanyName =
    companyNames[activeTicker] ||
    KNOWN_TICKERS[activeTicker]?.name ||
    `${activeTicker} Equity`;

  return (
    <div className="min-h-screen bg-[#F5F4F1] dark:bg-[#08080A] text-zinc-900 dark:text-[#ECECF1] flex flex-col font-sans transition-colors">
      {/* Top Navigation */}
      <Navbar
        activeView={activeView}
        onViewChange={setActiveView}
        isDark={isDark}
        onToggleTheme={handleToggleTheme}
        activeTicker={activeTicker}
      />

      {/* Main Content Area */}
      <main className="flex-1 px-4 sm:px-6 py-6 max-w-7xl w-full mx-auto">
        {activeView === 'home' && (
          <HomeView
            onNavigate={setActiveView}
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
            resolution={resolution}
            onResolutionChange={setResolution}
            isLoading={isLoading}
            onRunAnalysis={() => fetchTickers([activeTicker], '10y', startDate, endDate)}
            companyName={currentCompanyName}
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
            resolution={resolution}
            onResolutionChange={setResolution}
            companyName={currentCompanyName}
            onNavigateToAnalysis={() => setActiveView('analysis')}
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
          />
        )}
      </main>

      <Footer />
    </div>
  );
}
