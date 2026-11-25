'use client';

import { useEffect, useRef, useState } from 'react';
import type { PriceBar } from '@/lib/quant-engine';
import { nextTcnBootstrapAction, type TcnTrainableHead } from '@/lib/tcn-bootstrap-policy';
import { notifyTcnBootUpdate, runTcnHeadSequence } from '@/lib/tcn-training-queue';
import { selectDefaultRegressionFactors } from '@/lib/tcn-diagnostics';

import { yieldToMain } from '@/lib/yield-frame';

const yieldFrame = yieldToMain;

/**
 * Tickers whose one-time bootstrap completed (or was fully answered from the
 * engine caches) this app session. Module scope: view navigation never
 * re-triggers a completed bootstrap; a page reload resets it by design.
 * "Once" is per ticker — switching back to a bootstrapped ticker reuses the
 * cache, and only the retrain buttons refresh it afterwards.
 */
const bootstrapped = new Set<string>();

export interface TcnBootStatus {
  ticker: string;
  state: 'idle' | 'training' | 'done';
  head: TcnTrainableHead | null;
}

const idle = (ticker: string): TcnBootStatus => ({ ticker, state: 'idle', head: null });

/**
 * App-level TCN bootstrap trainer: fits the multi-domain heads once per
 * ticker — at first load or on the first ticker switch — as soon as live
 * bars are available. Everything after that first fit is manual (the retrain
 * buttons); this hook never re-fires on live-tick data drift, refreshes or
 * remounts. Mounted inside TickerSnapshotProvider, so it runs regardless of
 * the active view, and its cache writes notify the view hooks for immediate
 * re-hydration.
 */
export function useTcnBootstrap({
  ticker,
  bars,
  barsSource,
  allStockBars,
}: {
  ticker: string;
  bars: PriceBar[];
  barsSource?: string | null;
  allStockBars: Record<string, PriceBar[]>;
}): TcnBootStatus {
  const upper = ticker.trim().toUpperCase();
  const barsLen = bars?.length ?? 0;
  const firstDate = bars?.[0]?.date ?? '';
  const lastDate = bars?.[barsLen - 1]?.date ?? '';
  const stockCacheKeyCount = Object.keys(allStockBars ?? {}).length;

  const [status, setStatus] = useState<TcnBootStatus>(() => idle(upper));
  const [
    request,
    setRequest,
  ] = useState<{ ticker: string; nonce: number; epoch: number; heads: TcnTrainableHead[] } | null>(null);
  // Re-arms the readiness watcher after a run settles (cancellation leaves
  // the ticker unregistered and nothing else would retry it).
  const [reval, setReval] = useState(0);
  const nonceRef = useRef(0);
  const epochRef = useRef(0);
  const trainingRef = useRef(false);
  const barsRef = useRef(bars);
  const allBarsRef = useRef(allStockBars);
  useEffect(() => {
    barsRef.current = bars;
    allBarsRef.current = allStockBars;
  }, [bars, allStockBars]);

  // A ticker change invalidates any pending or in-flight bootstrap fit for
  // the previous ticker (its epoch is now stale).
  useEffect(() => {
    epochRef.current += 1;
  }, [upper]);

  // Readiness watcher — wide deps: data loads asynchronously and the policy
  // decides. Completion lands in the session registry, so later re-arms on
  // tick drift resolve to "skip".
  useEffect(() => {
    if (!upper) return;
    const decision = nextTcnBootstrapAction({
      ticker: upper,
      bootstrapped,
      barsCount: barsRef.current?.length ?? 0,
      barsSource,
      regressionInputsReady:
        selectDefaultRegressionFactors(upper, barsRef.current ?? [], allBarsRef.current ?? {}) !== null,
      trainingInFlight: trainingRef.current,
    });
    if (decision.kind !== 'train') return;
    setRequest({
      ticker: decision.ticker,
      heads: decision.heads,
      nonce: ++nonceRef.current,
      epoch: epochRef.current,
    });
  }, [upper, barsLen, firstDate, lastDate, barsSource, stockCacheKeyCount, reval]);

  // The single boot fit for the requested ticker — narrow deps: it is never
  // restarted by data changes, only by a new request (a switch to a not-yet-
  // bootstrapped ticker) or killed by the epoch going stale.
  useEffect(() => {
    if (!request || bootstrapped.has(request.ticker)) return;
    let cancelled = false;
    // Start only once the browser is idle (first paint, hydration and the
    // initial view mount get the main thread first); hard cap at 4s.
    const ric = typeof window !== 'undefined' ? window.requestIdleCallback : undefined;
    const schedule = (fn: () => void): (() => void) => {
      if (typeof ric === 'function') {
        const id = ric(fn, { timeout: 4000 });
        return () => window.cancelIdleCallback(id);
      }
      const id = setTimeout(fn, 1500);
      return () => clearTimeout(id);
    };
    const cancelSchedule = schedule(() => {
      if (cancelled || epochRef.current !== request.epoch) return;
      const runTicker = request.ticker;
      trainingRef.current = true;
      setStatus({ ticker: runTicker, state: 'training', head: request.heads[0] ?? null });
      void runTcnHeadSequence(
        request.heads,
        {
          ticker: runTicker,
          bars: barsRef.current ?? [],
          allStockBars: allBarsRef.current ?? {},
          forceRetrain: false,
          yieldControl: yieldFrame,
          isCancelled: () => cancelled || epochRef.current !== request.epoch,
        },
        {
          onHeadStart: (head) => {
            if (cancelled || epochRef.current !== request.epoch) return;
            setStatus({ ticker: runTicker, state: 'training', head });
          },
          onHeadDone: (head) => {
            if (cancelled || epochRef.current !== request.epoch) return;
            notifyTcnBootUpdate(runTicker, head);
          },
        }
      ).finally(() => {
        const aborted = cancelled || epochRef.current !== request.epoch;
        trainingRef.current = false;
        if (aborted) {
          // Not registered: a later return to this ticker re-fits.
          setStatus((s) => idle(s.ticker));
        } else {
          bootstrapped.add(runTicker);
          setStatus({ ticker: runTicker, state: 'done', head: null });
        }
        setReval((v) => v + 1);
      });
    });
    return () => {
      cancelled = true;
      cancelSchedule();
    };
  }, [request]);

  return status;
}
