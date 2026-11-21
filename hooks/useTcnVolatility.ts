'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { PriceBar } from '@/lib/quant-engine';
import {
  getCachedTcnVolatilityForecast,
  runTcnVolatilityForecast,
  type TcnVolatilityForecast,
} from '@/lib/tcn-multi-domain';

const yieldFrame = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * Shared React-19-safe hook for the multi-horizon (1d / 5d / 22d) TCN-Attention
 * Volatility model. Reads synchronously from the cross-tab LRU cache on mount and
 * hydrates/trains asynchronously without blocking tab switches.
 */
export function useTcnVolatility(
  ticker: string,
  bars: PriceBar[],
  options?: { autoRun?: boolean; forecastDays?: number }
) {
  const autoRun = options?.autoRun !== false;
  const forecastDays = options?.forecastDays ?? 30;
  const barsLen = bars?.length ?? 0;
  const firstDate = bars?.[0]?.date ?? '';
  const lastDate = bars?.[barsLen - 1]?.date ?? '';
  const lastClose = bars?.[barsLen - 1]?.close ?? 0;

  const [result, setResult] = useState<TcnVolatilityForecast | null>(null);
  const [training, setTraining] = useState(false);
  const [retrainNonce, setRetrainNonce] = useState(0);
  const barsRef = useRef(bars);
  useEffect(() => {
    barsRef.current = bars;
  }, [bars]);

  const forceRetrain = useCallback(() => {
    setRetrainNonce((n) => n + 1);
  }, []);

  useEffect(() => {
    if (barsLen < 70) return;
    let cancelled = false;
    const force = retrainNonce > 0;
    const timer = setTimeout(() => {
      if (cancelled) return;
      const currentBars = barsRef.current;
      if (!force) {
        const cached = getCachedTcnVolatilityForecast(ticker, currentBars);
        if (cached) {
          setResult(cached);
          setTraining(false);
          return;
        }
        if (!autoRun) return;
      }
      setTraining(true);
      void runTcnVolatilityForecast(ticker, currentBars, {
        forecastDays,
        forceRetrain: force,
        yieldControl: yieldFrame,
        isCancelled: () => cancelled,
      }).then((res) => {
        if (!cancelled) {
          if (res) setResult(res);
          setTraining(false);
        }
      });
    }, 0);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ticker, barsLen, firstDate, lastDate, lastClose, autoRun, forecastDays, retrainNonce]);

  const isStale = result !== null && result.ticker !== ticker.toUpperCase();
  return {
    volForecast: isStale ? null : result,
    isTraining: training,
    retrain: forceRetrain,
  };
}
