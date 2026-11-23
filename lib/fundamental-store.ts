// Server-only accrual store for quoteSummary fundamentals and analyst snapshots.
// Yahoo exposes no filing timestamps and only ~4-5 statements, so history starts
// at first observation and can never be backfilled. Every consumer must label it.

import { mkdirSync, readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import type { FundamentalSnapshot } from './factors/fundamental-factors';
import type { AnalystSnapshot } from './factors/sentiment-options-factors';

const ROOT = path.join(process.cwd(), '.cache', 'fundamentals');

interface AccruedRecord {
  symbol: string;
  observedAt: string;
  firstObservedAt: string;
  snapshot: FundamentalSnapshot | null;
  analyst: AnalystSnapshot | null;
}

function symbolDir(symbol: string): string {
  const safe = symbol.toUpperCase().replace(/[^A-Z0-9^.=/-]/g, '');
  return path.join(ROOT, safe);
}

function ensureDir(dir: string): void {
  try {
    mkdirSync(dir, { recursive: true });
  } catch {
    // best effort
  }
}

/** Idempotently persist one snapshot per symbol/day. */
export function persistFundamentalSnapshot(
  symbol: string,
  observedAt: string,
  snapshot: FundamentalSnapshot | null,
  analyst: AnalystSnapshot | null
): void {
  const dir = symbolDir(symbol);
  ensureDir(dir);
  const file = path.join(dir, `${observedAt}.json`);
  if (existsSync(file)) return;
  const files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort();
  const firstObservedAt = files[0]?.slice(0, 10) ?? observedAt;
  const record: AccruedRecord = {
    symbol: symbol.toUpperCase(),
    observedAt,
    firstObservedAt,
    snapshot,
    analyst,
  };
  try {
    writeFileSync(file, JSON.stringify(record), 'utf8');
  } catch {
    // cache writes are best effort
  }
}

export interface FundamentalHistory {
  symbol: string;
  snapshots: FundamentalSnapshot[];
  analyst: AnalystSnapshot | null;
  sessions: number;
  firstDate: string | null;
  lastDate: string | null;
  /** True while the accrued history is too short for a fundamental backtest. */
  degraded: boolean;
}

export function readFundamentalHistory(symbol: string): FundamentalHistory {
  const dir = symbolDir(symbol);
  const empty: FundamentalHistory = {
    symbol: symbol.toUpperCase(),
    snapshots: [],
    analyst: null,
    sessions: 0,
    firstDate: null,
    lastDate: null,
    degraded: true,
  };
  if (!existsSync(dir)) return empty;
  try {
    const files = readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort();
    const snapshots: FundamentalSnapshot[] = [];
    let analyst: AnalystSnapshot | null = null;
    for (const file of files) {
      const record = JSON.parse(readFileSync(path.join(dir, file), 'utf8')) as AccruedRecord;
      if (record.snapshot) snapshots.push(record.snapshot);
      if (record.analyst) analyst = record.analyst;
    }
    const firstDate = files[0]?.slice(0, 10) ?? null;
    const lastDate = files[files.length - 1]?.slice(0, 10) ?? null;
    return {
      symbol: symbol.toUpperCase(),
      snapshots,
      analyst,
      sessions: files.length,
      firstDate,
      lastDate,
      degraded: files.length < 20,
    };
  } catch {
    return empty;
  }
}
