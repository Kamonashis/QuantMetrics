import { NextRequest, NextResponse } from 'next/server';
import { readSignalRecords, writeSignalRecords } from '@/lib/signal-journal-store';
import type { SignalJournalRecord } from '@/lib/signal-outcomes';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** GET /api/signal-journal?symbol=AAPL — accrued daily setup snapshots. */
export async function GET(req: NextRequest) {
  const symbol = (new URL(req.url).searchParams.get('symbol') || '').trim().toUpperCase();
  if (!symbol) return NextResponse.json({ available: false, reason: 'symbol is required' }, { status: 200 });
  const records = readSignalRecords(symbol);
  return NextResponse.json(
    { available: true, mode: 'history', symbol, records, count: records.length },
    { headers: { 'Cache-Control': 'no-store' } }
  );
}

/** POST /api/signal-journal?symbol=AAPL — idempotent daily snapshot write. */
export async function POST(req: NextRequest) {
  const symbol = (new URL(req.url).searchParams.get('symbol') || '').trim().toUpperCase();
  if (!symbol) return NextResponse.json({ available: false, reason: 'symbol is required' }, { status: 200 });
  try {
    const body: unknown = await req.json();
    const records = (body && typeof body === 'object' && Array.isArray((body as { records?: unknown }).records))
      ? (body as { records: Partial<SignalJournalRecord>[] }).records
      : [];
    const result = writeSignalRecords(symbol, records);
    return NextResponse.json(
      { available: result.ok, mode: 'record', symbol, written: result.written, reused: result.reused, reason: result.error },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (err) {
    return NextResponse.json(
      { available: false, mode: 'record', symbol, reason: err instanceof Error ? err.message : String(err) },
      { status: 200 }
    );
  }
}
