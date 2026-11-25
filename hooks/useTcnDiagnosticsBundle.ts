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
  type TcnConfluenceSummary,
  type TcnDomainKey,
  type TcnMultiAssetRegressionResult,
  type TcnRiskVarForecast,
  type TcnVolatilityForecast,
} from '@/lib/tcn-multi-domain';
import {
  runTcnHeadSequence,
  subscribeTcnBootUpdates,
} from '@/lib/tcn-training-queue';
import type { TcnTrainableHead } from '@/lib/tcn-bootstrap-policy';
import {
  selectDefaultRegressionFactors,
  type TcnModuleStatus,
  type TcnModuleStatuses,
} from '@/lib/tcn-diagnostics';

import { yieldToMain } from '@/lib/yield-frame';

const yieldFrame = yieldToMain;

export type TrainableTcnHead = TcnTrainableHead;

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

interface BundleTrigger {
  nonce: number;
  forceHeads: TrainableTcnHead[];
  mode: 'train' | 'clear';
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

const STEP_LABELS: Record<TrainableTcnHead, string> = {
  volatility: 'Volatility QLIKE (1d/5d/22d)',
  risk_var: 'Risk VaR / ES Pinball (1d/5d/22d)',
  regression: 'Multi-Asset Regression (1d/5d/22d)',
};

const HEAD_ORDER: TrainableTcnHead[] = ['volatility', 'risk_var', 'regression'];

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
 *
 * This hook never starts a fit on its own: cache hydration follows ticker/data
 * changes (including live quote ticks) without training. Fits start in exactly
 * two places — the app-level bootstrap trainer (the first live-data fit per
 * ticker per session, which also notifies this hook to re-hydrate) and the
 * `retrainAll` / `retrainSingle` retrain buttons. Clearing caches only resets
 * to the untrained state.
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
  const [runTrigger, setRunTrigger] = useState<BundleTrigger>({
    nonce: 0,
    forceHeads: [],
    mode: 'train',
  });

  const barsRef = useRef(bars);
  const allBarsRef = useRef(allStockBars);
  const trainingRef = useRef(false);
  useEffect(() => {
    barsRef.current = bars;
    allBarsRef.current = allStockBars;
  }, [bars, allStockBars]);

  const retrainSingle = useCallback((head: TrainableTcnHead) => {
    setRunTrigger((prev) => ({
      nonce: prev.nonce + 1,
      forceHeads: [head],
      mode: 'train',
    }));
  }, []);

  const retrainAll = useCallback(() => {
    setRunTrigger((prev) => ({
      nonce: prev.nonce + 1,
      forceHeads: ['volatility', 'risk_var', 'regression'],
      mode: 'train',
    }));
  }, []);

  const clearAllCaches = useCallback(() => {
    clearTcnMultiDomainCache();
    clearTcnCache();
    setRunTrigger((prev) => ({ nonce: prev.nonce + 1, forceHeads: [], mode: 'clear' }));
  }, []);

  const forceMask = runTrigger.forceHeads.join(',');
  const triggerNonce = runTrigger.nonce;
  const triggerMode = runTrigger.mode;

  // The app-level bootstrap trainer writes artifacts into the shared engine
  // caches outside this hook; a completed head nudges hydration so the
  // fresh result surfaces immediately instead of waiting for the next bar
  // tick to re-run effect 1.
  const [bootBump, setBootBump] = useState(0);
  useEffect(
    () =>
      subscribeTcnBootUpdates((symbol) => {
        if (symbol === upperTicker) setBootBump((v) => v + 1);
      }),
    [upperTicker]
  );

  // 1. Cache hydration only — follows data/ticker changes, never trains.
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled || trainingRef.current) return;
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
      const cachedDecomp = getLatestCachedTcnArtifactByTicker(upperTicker)?.forecast ?? null;
      const cachedVol = hasEnoughBars ? getCachedTcnVolatilityForecast(upperTicker, curBars) : null;
      const cachedRisk = hasEnoughBars
        ? getCachedTcnRiskVarForecast(upperTicker, returns, retDates)
        : null;
      const cachedReg = regInputs
        ? getCachedTcnRegressionForecast(
            regInputs.dependentTicker,
            regInputs.independentTickers,
            regInputs.returnsByTicker,
            regInputs.dates
          )
        : null;

      setBundle((prev) => {
        // Live ticks change data signatures: keep the last fits visible (as 'ready')
        // until the user retrains, rather than flashing back to untrained.
        const sameTicker = prev.ticker === upperTicker;
        const vol = cachedVol ?? (sameTicker ? prev.volatility : null);
        const risk = cachedRisk ?? (sameTicker ? prev.riskVar : null);
        const reg = cachedReg ?? (sameTicker ? prev.regression : null);
        return {
          ...prev,
          ticker: upperTicker,
          volatility: vol,
          riskVar: risk,
          regression: reg,
          decompForecast: cachedDecomp,
          statuses: {
            volatility: !hasEnoughBars
              ? 'insufficient'
              : cachedVol
                ? 'cached'
                : vol
                  ? 'ready'
                  : 'idle',
            risk_var: !hasEnoughBars
              ? 'insufficient'
              : cachedRisk
                ? 'cached'
                : risk
                  ? 'ready'
                  : 'idle',
            regression: !regInputs
              ? 'insufficient'
              : cachedReg
                ? 'cached'
                : reg
                  ? 'ready'
                  : 'idle',
            confluence: vol ? 'ready' : hasEnoughBars ? 'idle' : 'insufficient',
          },
          activeStepLabel: null,
          completedSteps: 0,
          totalSteps: 0,
          multiDomainCounts: getTcnMultiDomainCacheStats(),
          decompStats: getTcnCacheStats(),
        };
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [upperTicker, barsLen, firstDate, lastDate, lastClose, stockCacheKeyCount, bootBump]);

  // 2. Explicit training only — started by the retrain buttons (and the
  // app-level bootstrap trainer outside this hook), never by data changes.
  useEffect(() => {
    if (triggerNonce === 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled) return;
      if (triggerMode === 'clear') {
        setBundle(makeInitialState(upperTicker));
        return;
      }

      void (async () => {
        const curBars = barsRef.current ?? [];
        const curAllBars = allBarsRef.current ?? {};
        const hasEnoughBars = curBars.length >= 70;
        const regInputs = hasEnoughBars
          ? selectDefaultRegressionFactors(upperTicker, curBars, curAllBars)
          : null;
        const forced = new Set<TrainableTcnHead>(
          forceMask ? (forceMask.split(',') as TrainableTcnHead[]) : []
        );
        const queue = HEAD_ORDER.filter(
          (head) => forced.has(head) && (head !== 'regression' || regInputs !== null)
        );

        trainingRef.current = true;
        setBundle((prev) => {
          const statuses: TcnModuleStatuses = { ...prev.statuses };
          for (const head of queue) statuses[head] = 'training';
          return {
            ...prev,
            ticker: upperTicker,
            statuses,
            activeStepLabel: queue.length > 0 ? STEP_LABELS[queue[0]] : null,
            completedSteps: 0,
            totalSteps: queue.length,
          };
        });

        let done = 0;
        await runTcnHeadSequence(
          queue,
          {
            ticker: upperTicker,
            bars: curBars,
            allStockBars: curAllBars,
            forceRetrain: true,
            yieldControl: yieldFrame,
            isCancelled: () => cancelled,
          },
          {
            onHeadStart: (head) => {
              if (cancelled) return;
              setBundle((prev) => ({
                ...prev,
                statuses: { ...prev.statuses, [head]: 'training' as TcnModuleStatus },
                activeStepLabel: STEP_LABELS[head],
              }));
            },
            onHeadDone: (head, result) => {
              if (cancelled) return;
              done++;
              setBundle((prev) => {
                const statuses: TcnModuleStatuses = { ...prev.statuses };
                if (head === 'volatility') {
                  statuses.volatility = result ? 'ready' : 'insufficient';
                  statuses.confluence = result ? 'ready' : 'insufficient';
                } else if (head === 'risk_var') {
                  statuses.risk_var = result ? 'ready' : 'insufficient';
                } else {
                  statuses.regression = result ? 'ready' : 'insufficient';
                }
                return {
                  ...prev,
                  volatility: head === 'volatility' ? (result as TcnVolatilityForecast | null) : prev.volatility,
                  riskVar: head === 'risk_var' ? (result as TcnRiskVarForecast | null) : prev.riskVar,
                  regression:
                    head === 'regression' ? (result as TcnMultiAssetRegressionResult | null) : prev.regression,
                  statuses,
                  completedSteps: done,
                  multiDomainCounts: getTcnMultiDomainCacheStats(),
                  decompStats: getTcnCacheStats(),
                };
              });
            },
          }
        );

        if (!cancelled) {
          setBundle((prev) => ({
            ...prev,
            activeStepLabel: null,
            multiDomainCounts: getTcnMultiDomainCacheStats(),
            decompStats: getTcnCacheStats(),
          }));
          trainingRef.current = false;
        }
      })();
    }, 0);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      trainingRef.current = false;
    };
  }, [upperTicker, triggerNonce, forceMask, triggerMode]);

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
