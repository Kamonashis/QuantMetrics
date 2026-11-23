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
 * Volatility model.
 *
 * Training is manual-only: cache hydration follows the data/ticker, but fitting
 * only ever starts from an explicit `retrain()` call (a user pressing a button).
 * Live quote ticks patch bars and re-run hydration; they never start training.
 */
export function useTcnVolatility(
  ticker: string,
  bars: PriceBar[],
  options?: { forecastDays?: number }
) {
  const forecastDays = options?.forecastDays ?? 30;
  const barsLen = bars?.length ?? 0;
  const firstDate = bars?.[0]?.date ?? '';
  const lastDate = bars?.[barsLen - 1]?.date ?? '';
  const lastClose = bars?.[barsLen - 1]?.close ?? 0;

  const [result, setResult] = useState<TcnVolatilityForecast | null>(null);
  const [training, setTraining] = useState(false);
  // nonce 0 means no user request has been made yet.
  const [trainRequest, setTrainRequest] = useState<{ nonce: number; force: boolean }>({
    nonce: 0,
    force: false,
  });
  const barsRef = useRef(bars);
  const trainingRef = useRef(false);
  useEffect(() => {
    barsRef.current = bars;
  }, [bars]);

  const retrain = useCallback(() => {
    setTrainRequest((prev) => ({ nonce: prev.nonce + 1, force: true }));
  }, []);

  // Cache hydration only — follows the data, never starts a fit.
  useEffect(() => {
    if (barsLen < 70) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled || trainingRef.current) return;
      const cached = getCachedTcnVolatilityForecast(ticker, barsRef.current);
      setResult((prev) => {
        if (cached) return cached;
        // Live ticks change the signature: keep the last fit visible until retrained.
        return prev && prev.ticker === ticker.toUpperCase() ? prev : null;
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ticker, barsLen, firstDate, lastDate, lastClose, forecastDays]);

  // Explicit training only — started by the retrain button.
  useEffect(() => {
    if (trainRequest.nonce === 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      const currentBars = barsRef.current;
      if (cancelled || currentBars.length < 70) return;
      trainingRef.current = true;
      setTraining(true);
      void runTcnVolatilityForecast(ticker, currentBars, {
        forecastDays,
        forceRetrain: trainRequest.force,
        yieldControl: yieldFrame,
        isCancelled: () => cancelled,
      }).then((res) => {
        trainingRef.current = false;
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
  }, [trainRequest, ticker, forecastDays]);

  const isStale = result !== null && result.ticker !== ticker.toUpperCase();
  return {
    volForecast: isStale ? null : result,
    isTraining: training,
    retrain,
  };
}
