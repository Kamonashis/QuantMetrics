// IV Snapshot Store: accretes one chain summary per symbol per day.
// Files live in .cache/iv/{SYMBOL}/{YYYY-MM-DD}.json (gitignored).
// Server-side only — never import from client components.
import fs from 'node:fs';
import path from 'node:path';

export interface IvSnapshot {
  symbol: string;
  date: string;          // YYYY-MM-DD (UTC)
  fetchedAt: string;     // ISO
  spot: number;
  rate: number;
  iv30: number | null;
  atmIv: number | null;
  skewRr25: number | null;
  totalVariance: number | null;
  nContractsUsed: number;
  minDte: number;
  source: string;        // 'live_chain' | 'cached_institutional'
}

const root = () => path.join(process.cwd(), '.cache', 'iv');

const safeSymbol = (s: string) => s.toUpperCase().replace(/[^A-Z0-9._-]/g, '_').slice(0, 32);

export function snapshotDir(symbol: string): string {
  return path.join(root(), safeSymbol(symbol));
}

export function writeSnapshot(snap: IvSnapshot): { ok: boolean; error?: string } {
  try {
    const dir = snapshotDir(snap.symbol);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${snap.date}.json`);
    fs.writeFileSync(file, JSON.stringify(snap, null, 2), 'utf8');
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

export function readHistory(symbol: string): IvSnapshot[] {
  try {
    const dir = snapshotDir(symbol);
    if (!fs.existsSync(dir)) return [];
    const files = fs.readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort();
    const out: IvSnapshot[] = [];
    for (const f of files) {
      try {
        const raw = fs.readFileSync(path.join(dir, f), 'utf8');
        const parsed = JSON.parse(raw) as IvSnapshot;
        if (parsed && typeof parsed.date === 'string') out.push(parsed);
      } catch {
        // skip corrupt files — the store must be self-healing
      }
    }
    return out;
  } catch {
    return [];
  }
}

export function historyStats(symbol: string): { count: number; firstDate: string | null; lastDate: string | null } {
  const hist = readHistory(symbol);
  if (hist.length === 0) return { count: 0, firstDate: null, lastDate: null };
  return { count: hist.length, firstDate: hist[0].date, lastDate: hist[hist.length - 1].date };
}
