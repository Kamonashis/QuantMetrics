import { NextRequest, NextResponse } from 'next/server';
import { fetchLiveStockData, fetchLiveIntradayData } from '@/lib/market-service';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const tickersParam = searchParams.get('tickers') || searchParams.get('ticker') || 'AAPL';
  const range = searchParams.get('range') || '2y';
  const interval = searchParams.get('interval') || '1d';

  const tickerList = tickersParam
    .split(',')
    .map((t) => t.trim().toUpperCase())
    .filter((t) => t.length > 0)
    .slice(0, 10); // cap at 10 tickers per call

  if (tickerList.length === 0) {
    tickerList.push('AAPL');
  }

  try {
    const results = await Promise.all(
      tickerList.map(async (ticker) => {
        if (interval === '1m') {
          const data = await fetchLiveIntradayData(ticker, range === '2y' ? '7d' : range);
          return [ticker, data] as const;
        } else {
          const data = await fetchLiveStockData(ticker, range);
          return [ticker, data] as const;
        }
      })
    );

    const payload = Object.fromEntries(results);

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      data: payload,
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Failed to retrieve market data';
    return NextResponse.json(
      { success: false, error: msg },
      { status: 500 }
    );
  }
}
