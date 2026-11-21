'use client';

import { useEffect, useRef } from 'react';
import type { LiveQuoteTick } from '@/lib/market-service';

export interface UseMarketStreamOptions {
  /** Sanitized list of tickers to subscribe to (max 12). */
  tickers: readonly string[];
  /** Server push cadence in milliseconds (server clamps to 15s–120s, default 30s). */
  intervalMs?: number;
  /** Fallback HTTP poll cadence in milliseconds if SSE is unavailable (default 60s). */
  fallbackPollMs?: number;
  /** Set false to pause streaming. */
  enabled?: boolean;
  /** Called whenever a batch of live quote ticks arrives from SSE or fallback polling. */
  onTicks: (ticks: LiveQuoteTick[]) => void;
}

/**
 * WebSocket-like live market data subscription hook powered by Server-Sent Events (SSE)
 * with automatic visibility-aware pause/resume, online reconnect, and 60s HTTP polling fallback.
 */
export function useMarketStream({
  tickers,
  intervalMs = 30_000,
  fallbackPollMs = 60_000,
  enabled = true,
  onTicks,
}: UseMarketStreamOptions): void {
  const onTicksRef = useRef(onTicks);
  useEffect(() => {
    onTicksRef.current = onTicks;
  }, [onTicks]);

  const tickersKey = tickers.join(',');

  useEffect(() => {
    if (!enabled || !tickersKey || typeof window === 'undefined') return;

    let disposed = false;
    let source: EventSource | null = null;
    let fallbackInterval: ReturnType<typeof setInterval> | null = null;
    let retryUpgradeTimer: ReturnType<typeof setTimeout> | null = null;
    let pollController: AbortController | null = null;
    let consecutiveSseErrors = 0;

    const closeSource = () => {
      if (source) {
        source.close();
        source = null;
      }
    };

    const stopFallbackPolling = () => {
      if (fallbackInterval) {
        clearInterval(fallbackInterval);
        fallbackInterval = null;
      }
      if (retryUpgradeTimer) {
        clearTimeout(retryUpgradeTimer);
        retryUpgradeTimer = null;
      }
      if (pollController) {
        pollController.abort();
        pollController = null;
      }
    };

    const pollQuotesOnce = async () => {
      if (disposed) return;
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;

      if (pollController) pollController.abort();
      const ctrl = new AbortController();
      pollController = ctrl;
      try {
        const params = new URLSearchParams({
          mode: 'quotes',
          tickers: tickersKey,
        });
        const res = await fetch(`/api/market-data?${params.toString()}`, {
          cache: 'no-store',
          signal: ctrl.signal,
        });
        if (!res.ok || disposed) return;
        const json = (await res.json()) as { success?: boolean; ticks?: LiveQuoteTick[] };
        if (!disposed && json.success && Array.isArray(json.ticks) && json.ticks.length > 0) {
          onTicksRef.current(json.ticks);
        }
      } catch {
        // Ignore transient network errors or intentional aborts
      }
    };

    const startFallbackPolling = () => {
      closeSource();
      if (fallbackInterval || disposed) return;
      void pollQuotesOnce();
      fallbackInterval = setInterval(() => {
        void pollQuotesOnce();
      }, fallbackPollMs);

      // Periodically attempt to re-upgrade back to SSE after transient proxy resets
      retryUpgradeTimer = setTimeout(() => {
        if (disposed) return;
        stopFallbackPolling();
        consecutiveSseErrors = 0;
        connectSse();
      }, 5 * 60_000);
    };

    const connectSse = () => {
      if (disposed) return;
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;

      if (typeof EventSource === 'undefined') {
        startFallbackPolling();
        return;
      }

      stopFallbackPolling();
      closeSource();

      const params = new URLSearchParams({
        tickers: tickersKey,
        intervalMs: String(intervalMs),
      });
      const es = new EventSource(`/api/market-data/stream?${params.toString()}`);
      source = es;

      es.addEventListener('open', () => {
        consecutiveSseErrors = 0;
      });

      es.addEventListener('quotes', (event: MessageEvent) => {
        if (disposed) return;
        consecutiveSseErrors = 0;
        try {
          const parsed = JSON.parse(String(event.data)) as { ticks?: LiveQuoteTick[] };
          if (Array.isArray(parsed.ticks) && parsed.ticks.length > 0) {
            onTicksRef.current(parsed.ticks);
          }
        } catch {
          // Ignore malformed event frames
        }
      });

      es.onerror = () => {
        if (disposed) return;
        consecutiveSseErrors += 1;
        if (consecutiveSseErrors >= 2) {
          startFallbackPolling();
        }
      };
    };

    const handleVisibilityChange = () => {
      if (disposed) return;
      if (document.visibilityState === 'hidden') {
        closeSource();
        stopFallbackPolling();
      } else {
        consecutiveSseErrors = 0;
        connectSse();
      }
    };

    const handleOnline = () => {
      if (disposed) return;
      consecutiveSseErrors = 0;
      connectSse();
    };

    connectSse();
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('online', handleOnline);

    return () => {
      disposed = true;
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('online', handleOnline);
      closeSource();
      stopFallbackPolling();
    };
  }, [enabled, tickersKey, intervalMs, fallbackPollMs]);
}
