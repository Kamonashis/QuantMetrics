import { NextRequest, NextResponse } from 'next/server';
import {
  fetchLiveStockData,
  fetchLiveIntradayData,
  fetchLiveQuoteSnapshot,
  sanitizeTickerSymbol,
} from '@/lib/market-service';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const tickersParam = searchParams.get('tickers') || searchParams.get('ticker') || 'NVDA';
  const mode = searchParams.get('mode') || 'bars';
  const range = searchParams.get('range') || '10y';
  const interval = searchParams.get('interval') || '1d';
  const startDate = searchParams.get('startDate') || undefined;
  const endDate = searchParams.get('endDate') || undefined;

  const maxSymbols = mode === 'quotes' ? 12 : 10;
  const seen = new Set<string>();
  const tickerList: string[] = [];
  for (const raw of tickersParam.split(',')) {
    const sym = sanitizeTickerSymbol(raw);
    if (sym && !seen.has(sym)) {
      seen.add(sym);
      tickerList.push(sym);
      if (tickerList.length >= maxSymbols) break;
    }
  }

  if (tickerList.length === 0) {
    tickerList.push('NVDA');
  }

  try {
    if (mode === 'quotes') {
      const ticks = await Promise.all(tickerList.map((ticker) => fetchLiveQuoteSnapshot(ticker)));
      return NextResponse.json(
        {
          success: true,
          timestamp: new Date().toISOString(),
          ticks,
        },
        {
          headers: {
            'Cache-Control': 'no-store, max-age=0',
          },
        }
      );
    }

    const results = await Promise.all(
      tickerList.map(async (ticker) => {
        if (interval === '1m') {
          const data = await fetchLiveIntradayData(ticker, range === '2y' || range === '10y' ? '7d' : range);
          return [ticker, data] as const;
        } else {
          const data = await fetchLiveStockData(ticker, range, startDate, endDate);
          return [ticker, data] as const;
        }
      })
    );

    const payload = Object.fromEntries(results);

    return NextResponse.json(
      {
        success: true,
        timestamp: new Date().toISOString(),
        data: payload,
      },
      {
        headers: {
          'Cache-Control': 'no-store, max-age=0',
        },
      }
    );
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Failed to retrieve market data';
    return NextResponse.json(
      { success: false, error: msg },
      { status: 500 }
    );
  }
}
