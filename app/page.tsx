'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { PriceBar, IntradayBar } from '@/lib/quant-engine';
import {
  generateRealisticStockHistory,
  generateRealisticIntradayHistory,
  KNOWN_TICKERS,
} from '@/lib/market-service';
import { Navbar } from '@/components/layout/Navbar';
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

  // Market Data In-Memory Cache with lazy instant initialization
  const [stockBarsCache, setStockBarsCache] = useState<{ [ticker: string]: PriceBar[] }>(() => {
    const initialCache: { [ticker: string]: PriceBar[] } = {};
    const defaultSymbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'];
    defaultSymbols.forEach((sym) => {
      initialCache[sym] = generateRealisticStockHistory(sym, 350);
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
  const fetchTickers = useCallback(async (tickersToFetch: string[]) => {
    const syms = tickersToFetch
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
    if (syms.length === 0) return;

    setIsLoading(true);
    try {
      // 1. Fetch Daily Data
      const res = await fetch(`/api/market-data?tickers=${encodeURIComponent(syms.join(','))}`);
      if (res.ok) {
        const json = await res.json();
        if (json.success && json.data) {
          setStockBarsCache((prev) => {
            const updated = { ...prev };
            for (const [sym, payload] of Object.entries(json.data as Record<string, { bars: PriceBar[]; name: string }>)) {
              if (payload.bars && payload.bars.length > 0) {
                updated[sym] = payload.bars;
              }
            }
            return updated;
          });
          setCompanyNames((prev) => {
            const updated = { ...prev };
            for (const [sym, payload] of Object.entries(json.data as Record<string, { bars: PriceBar[]; name: string }>)) {
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
        const intraRes = await fetch(`/api/market-data?ticker=${encodeURIComponent(primarySym)}&interval=1m&range=7d`);
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
            updated[sym] = generateRealisticStockHistory(sym, 350);
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
  }, []);

  // Active ticker bars
  const currentBars = stockBarsCache[activeTicker] || generateRealisticStockHistory(activeTicker, 350);
  const currentIntradayBars =
    intradayBarsCache[activeTicker] || generateRealisticIntradayHistory(activeTicker, 7);
  const currentCompanyName = companyNames[activeTicker] || KNOWN_TICKERS[activeTicker]?.name || `${activeTicker} Equity`;

  return (
    <div className="min-h-screen bg-[#faf8f5] dark:bg-[#121110] text-stone-900 dark:text-stone-100 flex flex-col font-sans transition-colors">
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
            onStartDateChange={setStartDate}
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
            onRunAnalysis={() => fetchTickers([activeTicker])}
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
            onStartDateChange={setStartDate}
            endDate={endDate}
            onEndDateChange={setEndDate}
            returnType={returnType}
            allStockBars={stockBarsCache}
            isLoading={isLoading}
            onRunCorrelation={(tickers) => fetchTickers(tickers)}
            intradayBars={currentIntradayBars}
          />
        )}

        {activeView === 'regression' && (
          <RegressionView
            regressionTickers={regressionTickers}
            onRegressionTickersChange={setRegressionTickers}
            startDate={startDate}
            onStartDateChange={setStartDate}
            endDate={endDate}
            onEndDateChange={setEndDate}
            returnType={returnType}
            allStockBars={stockBarsCache}
            isLoading={isLoading}
            onRunRegression={(tickers) => fetchTickers(tickers)}
          />
        )}
      </main>

      {/* Institutional Terminal Footer */}
      <footer className="border-t border-stone-200 dark:border-stone-800 bg-[#faf8f5] dark:bg-[#141311] py-4 text-xs text-stone-500 dark:text-stone-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-stone-800 dark:text-stone-200">
              QuantMetrics
            </span>
            <span>·</span>
            <span>GARCH(1,1), EWMA &amp; HAR-RV Volatility Engine</span>
            <span>·</span>
            <span>1-Min Intraday High-Frequency Analytics</span>
          </div>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span>Market Frequency: Daily &amp; 1-Min Intraday</span>
            <span>·</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Engine Operational</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
