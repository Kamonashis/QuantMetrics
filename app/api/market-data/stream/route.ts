import { NextRequest } from 'next/server';
import { fetchLiveQuoteSnapshot, sanitizeTickerSymbol } from '@/lib/market-service';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const MIN_INTERVAL_MS = 15_000;
const MAX_INTERVAL_MS = 120_000;
const DEFAULT_INTERVAL_MS = 30_000;
const HEARTBEAT_MS = 15_000;
const MAX_STREAM_LIFETIME_MS = 15 * 60_000;
const MAX_STREAM_TICKERS = 12;

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const tickersParam = searchParams.get('tickers') || 'NVDA';
  const rawInterval = Number(searchParams.get('intervalMs'));
  const intervalMs = Number.isFinite(rawInterval)
    ? Math.max(MIN_INTERVAL_MS, Math.min(MAX_INTERVAL_MS, Math.round(rawInterval)))
    : DEFAULT_INTERVAL_MS;

  const seen = new Set<string>();
  const tickers: string[] = [];
  for (const raw of tickersParam.split(',')) {
    const sym = sanitizeTickerSymbol(raw);
    if (sym && !seen.has(sym)) {
      seen.add(sym);
      tickers.push(sym);
      if (tickers.length >= MAX_STREAM_TICKERS) break;
    }
  }
  if (tickers.length === 0) tickers.push('NVDA');

  const encoder = new TextEncoder();
  let closed = false;
  let quoteTimer: ReturnType<typeof setInterval> | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  let lifetimeTimer: ReturnType<typeof setTimeout> | null = null;
  let inFlight = false;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const cleanup = () => {
        if (closed) return;
        closed = true;
        if (quoteTimer) clearInterval(quoteTimer);
        if (heartbeatTimer) clearInterval(heartbeatTimer);
        if (lifetimeTimer) clearTimeout(lifetimeTimer);
        quoteTimer = null;
        heartbeatTimer = null;
        lifetimeTimer = null;
        try {
          controller.close();
        } catch {
          // Stream already closed by consumer
        }
      };

      const sendEvent = (event: string, data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
        } catch {
          cleanup();
        }
      };

      const pushQuotes = async () => {
        if (closed || inFlight) return;
        inFlight = true;
        try {
          const ticks = await Promise.all(tickers.map((sym) => fetchLiveQuoteSnapshot(sym)));
          if (!closed) {
            sendEvent('quotes', {
              timestamp: new Date().toISOString(),
              ticks,
            });
          }
        } catch {
          // Transient upstream failures keep the stream open for the next scheduled push
        } finally {
          inFlight = false;
        }
      };

      // Hint EventSource reconnect delay
      try {
        controller.enqueue(encoder.encode('retry: 5000\n\n'));
      } catch {
        cleanup();
        return;
      }

      // Immediate initial push on connect
      void pushQuotes();

      quoteTimer = setInterval(() => {
        void pushQuotes();
      }, intervalMs);

      heartbeatTimer = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`: heartbeat ${Date.now()}\n\n`));
        } catch {
          cleanup();
        }
      }, HEARTBEAT_MS);

      // Graceful lifetime cap so long-lived server sockets rotate cleanly
      lifetimeTimer = setTimeout(() => {
        cleanup();
      }, MAX_STREAM_LIFETIME_MS);

      req.signal.addEventListener('abort', cleanup, { once: true });
    },
    cancel() {
      closed = true;
      if (quoteTimer) clearInterval(quoteTimer);
      if (heartbeatTimer) clearInterval(heartbeatTimer);
      if (lifetimeTimer) clearTimeout(lifetimeTimer);
      quoteTimer = null;
      heartbeatTimer = null;
      lifetimeTimer = null;
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-store, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
