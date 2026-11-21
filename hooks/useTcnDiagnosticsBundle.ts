'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PriceBar } from '@/lib/quant-engine';
import { calculateReturns } from '@/lib/quant-engine';
import {
  clearTcnCache,
  getLatestCachedTcnArtifactByTicker,
  getTcnCacheStats,
  type TcnDecompositionForecast,
} from '@/lib/tcn-attention';
import {
  clearTcnMultiDomainCache,
  evaluateTcnConfluence,
  getCachedTcnRegressionForecast,
  getCachedTcnRiskVarForecast,
  getCachedTcnVolatilityForecast,
  getTcnMultiDomainCacheStats,
  runTcnMultiAssetRegression,
  runTcnRiskVarForecast,
  runTcnVolatilityForecast,
  type TcnConfluenceSummary,
  type TcnDomainKey,
  type TcnMultiAssetRegressionResult,
  type TcnRiskVarForecast,
  type TcnVolatilityForecast,
} from '@/lib/tcn-multi-domain';
import {
  selectDefaultRegressionFactors,
  type TcnModuleStatus,
  type TcnModuleStatuses,
} from '@/lib/tcn-diagnostics';

const yieldFrame = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export type TrainableTcnHead = 'volatility' | 'risk_var' | 'regression';

interface BundleState {
  ticker: string;
  volatility: TcnVolatilityForecast | null;
  riskVar: TcnRiskVarForecast | null;
  regression: TcnMultiAssetRegressionResult | null;
  decompForecast: TcnDecompositionForecast | null;
  statuses: TcnModuleStatuses;
  activeStepLabel: string | null;
  completedSteps: number;
  totalSteps: number;
  multiDomainCounts: Record<TcnDomainKey, number>;
  decompStats: { forecastCount: number; sensitivityCount: number; totalWeightTensors: number };
}

const EMPTY_COUNTS: Record<TcnDomainKey, number> = {
  volatility: 0,
  risk_var: 0,
  regression: 0,
  correlation: 0,
};

const EMPTY_DECOMP_STATS = {
  forecastCount: 0,
  sensitivityCount: 0,
  totalWeightTensors: 0,
};

function makeInitialState(ticker: string): BundleState {
  return {
    ticker: ticker.toUpperCase(),
    volatility: null,
    riskVar: null,
    regression: null,
    decompForecast: null,
    statuses: {
      volatility: 'idle',
      risk_var: 'idle',
      regression: 'idle',
      confluence: 'idle',
    },
    activeStepLabel: null,
    completedSteps: 0,
    totalSteps: 0,
    multiDomainCounts: EMPTY_COUNTS,
    decompStats: EMPTY_DECOMP_STATS,
  };
}

/**
 * React-19-safe sequential orchestrator for the Unified TCN-Attention Diagnostics Dashboard.
 * 1. Keeps initial SSR/client render deterministic (all artifacts null until mount).
 * 2. Hydrates any already-cached heads immediately on mount / ticker switch.
 * 3. Sequentially trains missing heads (Volatility -> Risk VaR -> Multi-Asset Regression)
 *    with non-blocking frame yields and cancellation safety on ticker change.
 */
export function useTcnDiagnosticsBundle(params: {
  ticker: string;
  bars: PriceBar[];
  allStockBars: Record<string, PriceBar[]>;
  iv30: number | null;
  rv20: number | null;
  primaryOrientation: 'short-premium' | 'long-gamma' | 'neutral';
}) {
  const { ticker, bars, allStockBars, iv30, rv20, primaryOrientation } = params;
  const upperTicker = ticker.trim().toUpperCase();
  const barsLen = bars?.length ?? 0;
  const firstDate = bars?.[0]?.date ?? '';
  const lastDate = bars?.[barsLen - 1]?.date ?? '';
  const lastClose = bars?.[barsLen - 1]?.close ?? 0;
  const stockCacheKeyCount = Object.keys(allStockBars ?? {}).length;

  const [bundle, setBundle] = useState<BundleState>(() => makeInitialState(upperTicker));
  const [runTrigger, setRunTrigger] = useState<{
    nonce: number;
    forceHeads: TrainableTcnHead[];
  }>({ nonce: 0, forceHeads: [] });

  const barsRef = useRef(bars);
  const allBarsRef = useRef(allStockBars);
  useEffect(() => {
    barsRef.current = bars;
    allBarsRef.current = allStockBars;
  }, [bars, allStockBars]);

  const retrainSingle = useCallback((head: TrainableTcnHead) => {
    setRunTrigger((prev) => ({ nonce: prev.nonce + 1, forceHeads: [head] }));
  }, []);

  const retrainAll = useCallback(() => {
    setRunTrigger((prev) => ({
      nonce: prev.nonce + 1,
      forceHeads: ['volatility', 'risk_var', 'regression'],
    }));
  }, []);

  const clearAllCaches = useCallback(() => {
    clearTcnMultiDomainCache();
    clearTcnCache();
    setRunTrigger((prev) => ({
      nonce: prev.nonce + 1,
      forceHeads: ['volatility', 'risk_var', 'regression'],
    }));
  }, []);

  const forceMask = runTrigger.forceHeads.join(',');
  const triggerNonce = runTrigger.nonce;

  useEffect(() => {
    let cancelled = false;
    const forced = new Set<TrainableTcnHead>(
      forceMask ? (forceMask.split(',') as TrainableTcnHead[]) : []
    );

    const timer = setTimeout(() => {
      if (cancelled) return;
      void (async () => {
        const curBars = barsRef.current ?? [];
        const curAllBars = allBarsRef.current ?? {};

        const hasEnoughBars = curBars.length >= 70;
        const dates = curBars.map((b) => b.date);
        const closes = curBars.map((b) => b.close);
        const returns = hasEnoughBars ? calculateReturns(closes, 'log') : [];
        const retDates = dates.slice(1);
        const regInputs = hasEnoughBars
          ? selectDefaultRegressionFactors(upperTicker, curBars, curAllBars)
          : null;

        // Step 1: Peek existing caches synchronously after mount
        const cachedDecomp = getLatestCachedTcnArtifactByTicker(upperTicker)?.forecast ?? null;
        const cachedVol =
          hasEnoughBars && !forced.has('volatility')
            ? getCachedTcnVolatilityForecast(upperTicker, curBars)
            : null;
        const cachedRisk =
          hasEnoughBars && !forced.has('risk_var')
            ? getCachedTcnRiskVarForecast(upperTicker, returns, retDates)
            : null;
        const cachedReg =
          regInputs && !forced.has('regression')
            ? getCachedTcnRegressionForecast(
                regInputs.dependentTicker,
                regInputs.independentTickers,
                regInputs.returnsByTicker,
                regInputs.dates
              )
            : null;

        const queue: TrainableTcnHead[] = [];
        if (hasEnoughBars && !cachedVol) queue.push('volatility');
        if (hasEnoughBars && !cachedRisk) queue.push('risk_var');
        if (regInputs && !cachedReg) queue.push('regression');

        const initialStatuses: TcnModuleStatuses = {
          volatility: !hasEnoughBars
            ? 'insufficient'
            : cachedVol
              ? 'cached'
              : 'training',
          risk_var: !hasEnoughBars
            ? 'insufficient'
            : cachedRisk
              ? 'cached'
              : queue[0] === 'risk_var'
                ? 'training'
                : 'idle',
          regression: !regInputs
            ? 'insufficient'
            : cachedReg
              ? 'cached'
              : queue[0] === 'regression'
                ? 'training'
                : 'idle',
          confluence: cachedVol ? 'ready' : hasEnoughBars ? 'idle' : 'insufficient',
        };

        const stepLabels: Record<TrainableTcnHead, string> = {
          volatility: 'Volatility QLIKE (1d/5d/22d)',
          risk_var: 'Risk VaR / ES Pinball (1d/5d/22d)',
          regression: 'Multi-Asset Regression (1d/5d/22d)',
        };

        if (!cancelled) {
          setBundle({
            ticker: upperTicker,
            volatility: cachedVol,
            riskVar: cachedRisk,
            regression: cachedReg,
            decompForecast: cachedDecomp,
            statuses: initialStatuses,
            activeStepLabel: queue.length > 0 ? stepLabels[queue[0]] : null,
            completedSteps: 0,
            totalSteps: queue.length,
            multiDomainCounts: getTcnMultiDomainCacheStats(),
            decompStats: getTcnCacheStats(),
          });
        }

        // Step 2: Sequentially train missing or forced heads with frame yields
        let done = 0;
        for (const head of queue) {
          if (cancelled) return;
          setBundle((prev) => ({
            ...prev,
            statuses: { ...prev.statuses, [head]: 'training' as TcnModuleStatus },
            activeStepLabel: stepLabels[head],
          }));
          await yieldFrame();
          if (cancelled) return;

          if (head === 'volatility') {
            const res = await runTcnVolatilityForecast(upperTicker, curBars, {
              forecastDays: 30,
              forceRetrain: forced.has('volatility'),
              yieldControl: yieldFrame,
              isCancelled: () => cancelled,
            });
            if (cancelled) return;
            done++;
            setBundle((prev) => ({
              ...prev,
              volatility: res,
              statuses: {
                ...prev.statuses,
                volatility: res ? (res.cacheHit ? 'cached' : 'ready') : 'insufficient',
                confluence: res ? 'ready' : 'insufficient',
              },
              completedSteps: done,
              multiDomainCounts: getTcnMultiDomainCacheStats(),
              decompStats: getTcnCacheStats(),
            }));
          } else if (head === 'risk_var') {
            const res = await runTcnRiskVarForecast(upperTicker, returns, retDates, {
              forceRetrain: forced.has('risk_var'),
              yieldControl: yieldFrame,
              isCancelled: () => cancelled,
            });
            if (cancelled) return;
            done++;
            setBundle((prev) => ({
              ...prev,
              riskVar: res,
              statuses: {
                ...prev.statuses,
                risk_var: res ? (res.cacheHit ? 'cached' : 'ready') : 'insufficient',
              },
              completedSteps: done,
              multiDomainCounts: getTcnMultiDomainCacheStats(),
              decompStats: getTcnCacheStats(),
            }));
          } else if (head === 'regression' && regInputs) {
            const res = await runTcnMultiAssetRegression(
              regInputs.dependentTicker,
              regInputs.independentTickers,
              regInputs.returnsByTicker,
              regInputs.dates,
              {
                forceRetrain: forced.has('regression'),
                yieldControl: yieldFrame,
                isCancelled: () => cancelled,
              }
            );
            if (cancelled) return;
            done++;
            setBundle((prev) => ({
              ...prev,
              regression: res,
              statuses: {
                ...prev.statuses,
                regression: res ? (res.cacheHit ? 'cached' : 'ready') : 'insufficient',
              },
              completedSteps: done,
              multiDomainCounts: getTcnMultiDomainCacheStats(),
              decompStats: getTcnCacheStats(),
            }));
          }
          await yieldFrame();
        }

        if (!cancelled) {
          setBundle((prev) => ({
            ...prev,
            activeStepLabel: null,
            multiDomainCounts: getTcnMultiDomainCacheStats(),
            decompStats: getTcnCacheStats(),
          }));
        }
      })();
    }, 0);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [upperTicker, barsLen, firstDate, lastDate, lastClose, stockCacheKeyCount, triggerNonce, forceMask]);

  const isCurrent = bundle.ticker === upperTicker;
  const activeVol = isCurrent ? bundle.volatility : null;
  const activeRisk = isCurrent ? bundle.riskVar : null;
  const activeReg = isCurrent ? bundle.regression : null;
  const activeDecomp = isCurrent ? bundle.decompForecast : null;
  const activeStatuses: TcnModuleStatuses = isCurrent
    ? bundle.statuses
    : { volatility: 'idle', risk_var: 'idle', regression: 'idle', confluence: 'idle' };

  const confluence = useMemo<TcnConfluenceSummary>(
    () =>
      evaluateTcnConfluence({
        iv30,
        rv20,
        volForecast: activeVol,
        decompForecast: activeDecomp,
        primaryOrientation,
      }),
    [iv30, rv20, activeVol, activeDecomp, primaryOrientation]
  );

  const isTrainingAny =
    activeStatuses.volatility === 'training' ||
    activeStatuses.risk_var === 'training' ||
    activeStatuses.regression === 'training';

  return {
    volatility: activeVol,
    riskVar: activeRisk,
    regression: activeReg,
    decompForecast: activeDecomp,
    confluence,
    statuses: activeStatuses,
    isTrainingAny,
    activeStepLabel: isCurrent ? bundle.activeStepLabel : null,
    completedSteps: isCurrent ? bundle.completedSteps : 0,
    totalSteps: isCurrent ? bundle.totalSteps : 0,
    multiDomainCounts: bundle.multiDomainCounts,
    decompStats: bundle.decompStats,
    retrainSingle,
    retrainAll,
    clearAllCaches,
  };
}
