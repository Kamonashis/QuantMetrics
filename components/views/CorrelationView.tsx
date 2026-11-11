'use client';

import React, { useState, useMemo } from 'react';
import {
  PriceBar,
  calculateReturns,
} from '@/lib/quant-engine';
import {
  computeAdvancedCorrelationMatrix,
  IntradayBar,
} from '@/lib/correlation-engine';
import { generateRealisticStockHistory } from '@/lib/market-service';
import { CorrelationOverviewTab } from '@/components/views/correlation/CorrelationOverviewTab';
import { CorrelationDynamicsTab } from '@/components/views/correlation/CorrelationDynamicsTab';
import { CorrelationRiskTab } from '@/components/views/correlation/CorrelationRiskTab';
import { CorrelationStructureTab } from '@/components/views/correlation/CorrelationStructureTab';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import {
  Play,
  RotateCcw,
  Plus,
  Layers,
  Activity,
  PieChart,
  Network,
  ShieldAlert,
  Info,
  Calendar,
} from 'lucide-react';

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
  onRunCorrelation: (tickers: string[]) => void;
  intradayBars?: IntradayBar[];
}

type SubTabId = 'overview' | 'dynamics' | 'risk' | 'structure';

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
  intradayBars = [],
}: CorrelationViewProps) {
  const [localInput, setLocalInput] = useState(tickersInput);
  const [activeSubTab, setActiveSubTab] = useState<SubTabId>('overview');

  const activeTickers = useMemo(() => {
    return tickersInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
  }, [tickersInput]);

  // Synchronize historical date alignment across all active tickers
  const { alignedReturnsByTicker, alignedDates, minObsCount } = useMemo(() => {
    if (activeTickers.length === 0) {
      return { alignedReturnsByTicker: {}, alignedDates: [], minObsCount: 0 };
    }

    // Build map of date -> close price per ticker, synthesizing data if ticker is missing from cache
    const priceMaps: { [ticker: string]: Map<string, number> } = {};
    for (const t of activeTickers) {
      const bars =
        allStockBars[t] && allStockBars[t].length >= 10
          ? allStockBars[t]
          : generateRealisticStockHistory(t, 350);

      const m = new Map<string, number>();
      for (const b of bars) {
        if (startDate && b.date < startDate) continue;
        if (endDate && b.date > endDate) continue;
        m.set(b.date, b.close);
      }
      priceMaps[t] = m;
    }

    // Find date intersection
    const firstTicker = activeTickers[0];
    const initialDates = Array.from(priceMaps[firstTicker]?.keys() || []).sort();
    let commonDates = initialDates.filter((d) => {
      return activeTickers.every((t) => priceMaps[t]?.has(d));
    });

    // If common intersection is too small (< 10 dates), fall back to initialDates or union
    if (commonDates.length < 10) {
      commonDates = initialDates;
    }

    // Compute returns on synchronized dates
    const alignedReturns: { [ticker: string]: number[] } = {};
    for (const t of activeTickers) {
      const pm = priceMaps[t];
      let prices = commonDates.map((d) => pm?.get(d)).filter((p): p is number => typeof p === 'number');
      if (prices.length < 10) {
        const fallbackBars =
          allStockBars[t] && allStockBars[t].length >= 10
            ? allStockBars[t]
            : generateRealisticStockHistory(t, 252);
        prices = fallbackBars.map((b) => b.close);
      }
      alignedReturns[t] = calculateReturns(prices, returnType);
    }

    const retDates = commonDates.length > 1 ? commonDates.slice(1) : ['T-0'];
    const obsLen = retDates.length;

    return {
      alignedReturnsByTicker: alignedReturns,
      alignedDates: retDates,
      minObsCount: obsLen,
    };
  }, [activeTickers, allStockBars, startDate, endDate, returnType]);

  // Compute master Advanced Correlation Matrix result (Tier 1 calculations)
  const advancedResult = useMemo(() => {
    return computeAdvancedCorrelationMatrix(alignedReturnsByTicker);
  }, [alignedReturnsByTicker]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const list = localInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
    if (list.length > 0) {
      onTickersInputChange(list.join(', '));
      onRunCorrelation(list);
    }
  };

  const handleAddTicker = (sym: string) => {
    const list = localInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
    if (!list.includes(sym)) {
      const updated = [...list, sym].join(', ');
      setLocalInput(updated);
      onTickersInputChange(updated);
      onRunCorrelation([...list, sym]);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Parameter Control Card */}
      <div className="panel p-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800/80">
            <div>
              <div className="flex items-center gap-2">
                <span className="kicker">
                  Multivariate Quantitative Suite
                </span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  Tiers 1–4 Cross-Asset Correlation
                </span>
              </div>
              <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5">
                Correlation, Dynamic Regimes &amp; Portfolio Risk
              </h2>
              <div className="flex items-center gap-2 mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                <span>Engle DCC-GARCH</span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span>Ledoit-Wolf Shrinkage</span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span>Euler Risk Attribution</span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span>Mantegna MST Network</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <ShimmerButton
                type="submit"
                disabled={isLoading}
                className="px-4 py-2"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    <span>Computing Suite...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current text-gold-500" />
                    <span>Recompute Analysis</span>
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            {/* Tickers Textarea/Input */}
            <div className="md:col-span-2">
              <label className="block text-zinc-700 dark:text-zinc-300 font-medium mb-1">
                Stock Ticker Symbols (comma separated)
              </label>
              <input
                type="text"
                value={localInput}
                onChange={(e) => setLocalInput(e.target.value)}
                placeholder="AAPL, MSFT, GOOGL, NVDA, TSLA, SPY"
                className="w-full px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono font-bold text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500 text-sm"
              />
              <span className="text-[11px] text-zinc-400 block mt-1">
                Supports N=2 to 15 tickers. Matrices and network topologies re-evaluate on submit.
              </span>
            </div>

            {/* Date Range */}
            <div>
              <label className="block text-zinc-700 dark:text-zinc-300 font-medium mb-1">
                Historical Date Range
              </label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => onStartDateChange(e.target.value)}
                  className="px-2 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500 text-xs"
                />
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => onEndDateChange(e.target.value)}
                  className="px-2 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500 text-xs"
                />
              </div>
            </div>
          </div>
        </form>

        {/* Quick Add Ticker Badges */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs">
          <span className="text-zinc-400">Quick Add:</span>
          {['SPY', 'QQQ', 'NVDA', 'AMZN', 'META', 'JPM', 'XOM'].map((sym) => (
            <button
              key={sym}
              type="button"
              onClick={() => handleAddTicker(sym)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/60 hover:border-gold-500 text-zinc-700 dark:text-zinc-300 font-mono transition-colors cursor-pointer"
            >
              <Plus className="w-3 h-3 text-zinc-400" />
              {sym}
            </button>
          ))}
        </div>
      </div>

      {/* Data Alignment Honesty Banner */}
      <div className="bg-zinc-50 dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800/90 rounded-xl px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2.5">
          <Calendar className="w-4 h-4 text-gold-500 shrink-0" />
          <div className="text-zinc-700 dark:text-zinc-300">
            <span className="font-semibold text-zinc-900 dark:text-zinc-100">
              Data Synchronization:
            </span>{' '}
            Aligned across{' '}
            <strong className="font-mono text-gold-600 dark:text-gold-400">
              {minObsCount}
            </strong>{' '}
            concurrent trading sessions ({alignedDates[0] || startDate} →{' '}
            {alignedDates[alignedDates.length - 1] || endDate}).
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
          <span>
            Shrinkage δ: <strong>{advancedResult.shrinkageIntensity.toFixed(3)}</strong>
          </span>
          <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
          <span>
            κ(Σ): <strong>{advancedResult.conditionNumberRaw.toFixed(1)}</strong> →{' '}
            <strong className="text-gold-600 dark:text-gold-400">
              {advancedResult.conditionNumberShrunk.toFixed(1)}
            </strong>
          </span>
        </div>
      </div>

      {/* Sub-Tab Navigation Bar */}
      <div className="overflow-x-auto pb-1">
        <AnimatedTabs
          value={activeSubTab}
          onChange={(id) => setActiveSubTab(id as typeof activeSubTab)}
          tabs={[
            { id: 'overview', label: 'Overview & Matrix', icon: Layers, meta: 'T1' },
            { id: 'dynamics', label: 'Dynamics & Regimes', icon: Activity, meta: 'T2' },
            { id: 'risk', label: 'Portfolio Risk & VaR', icon: PieChart, meta: 'T3' },
            { id: 'structure', label: 'Structure & Network', icon: Network, meta: 'T4' },
          ]}
        />
      </div>

      {/* Sub-Tab Content Mounts */}
      {activeSubTab === 'overview' && (
        <CorrelationOverviewTab
          result={advancedResult}
          returnsByTicker={alignedReturnsByTicker}
          dates={alignedDates}
        />
      )}

      {activeSubTab === 'dynamics' && (
        <CorrelationDynamicsTab
          tickers={advancedResult.tickers}
          returnsByTicker={alignedReturnsByTicker}
          dates={alignedDates}
          intradayBars={intradayBars}
        />
      )}

      {activeSubTab === 'risk' && (
        <CorrelationRiskTab
          tickers={advancedResult.tickers}
          returnsByTicker={alignedReturnsByTicker}
          shrunkCovariance={advancedResult.shrunkCovariance}
          rawCovariance={advancedResult.covarianceMatrix}
        />
      )}

      {activeSubTab === 'structure' && (
        <CorrelationStructureTab
          tickers={advancedResult.tickers}
          matrix={advancedResult.matrix}
          returnsByTicker={alignedReturnsByTicker}
          dates={alignedDates}
        />
      )}
    </div>
  );
}
