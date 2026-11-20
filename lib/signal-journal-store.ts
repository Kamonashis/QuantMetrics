// Server-only append-only signal journal. Mirrors iv-store's idempotence and
// corrupt-file tolerance. One file per symbol: .cache/iv/{SYMBOL}/journal.json
import fs from 'node:fs';
import path from 'node:path';
import type { SignalJournalRecord } from './signal-outcomes';

const root = () => path.join(process.cwd(), '.cache', 'iv');
const safeSymbol = (s: string) => s.toUpperCase().replace(/[^A-Z0-9._-]/g, '_').slice(0, 32);
const journalPath = (symbol: string) => path.join(root(), safeSymbol(symbol), 'journal.json');

// `recordedAt` is stamped by the server, so clients send it empty.
const REQUIRED: (keyof SignalJournalRecord)[] = ['date', 'symbol', 'setup', 'state'];

function valid(r: Partial<SignalJournalRecord> | null | undefined): r is SignalJournalRecord {
  if (!r) return false;
  return REQUIRED.every((k) => typeof r[k] === 'string' && (r[k] as string).length > 0)
    && /^\d{4}-\d{2}-\d{2}$/.test(r.date ?? '');
}

/** Idempotent per (date, symbol, setup, state): one daily snapshot per card. */
export function writeSignalRecords(
  symbol: string,
  records: Partial<SignalJournalRecord>[]
): { ok: boolean; error?: string; written: number; reused: number } {
  try {
    const clean: SignalJournalRecord[] = records.filter(valid).map((r) => ({
      ...r,
      symbol: safeSymbol(r.symbol),
      direction: r.direction === 'up' || r.direction === 'down' ? r.direction : 'none',
      recordedAt: new Date().toISOString(),
    }));
    if (clean.length === 0) return { ok: true, written: 0, reused: 0 };

    const file = journalPath(symbol);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    let existing: SignalJournalRecord[] = [];
    if (fs.existsSync(file)) {
      try {
        const parsed: unknown = JSON.parse(fs.readFileSync(file, 'utf8'));
        if (Array.isArray(parsed)) existing = parsed.filter(valid);
      } catch {
        existing = []; // corrupt — rebuild from incoming records
      }
    }

    const key = (r: SignalJournalRecord) => `${r.date}|${r.symbol}|${r.setup}|${r.state}`;
    const seen = new Set(existing.map(key));
    let reused = 0;
    const additions = clean.filter((r) => {
      const k = key(r);
      if (seen.has(k)) { reused++; return false; }
      seen.add(k);
      return true;
    });

    if (additions.length > 0) {
      const merged = [...existing, ...additions].sort(
        (a, b) => a.date.localeCompare(b.date) || a.setup.localeCompare(b.setup)
      );
      fs.writeFileSync(file, JSON.stringify(merged, null, 2), 'utf8');
    }
    return { ok: true, written: additions.length, reused };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err), written: 0, reused: 0 };
  }
}

/** Read the accrued journal; corrupt files degrade to an empty array. */
export function readSignalRecords(symbol: string): SignalJournalRecord[] {
  try {
    const file = journalPath(symbol);
    if (!fs.existsSync(file)) return [];
    const parsed: unknown = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(valid).sort((a, b) => a.date.localeCompare(b.date));
  } catch {
    return [];
  }
}
