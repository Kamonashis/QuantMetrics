// IV / OI Snapshot Store: accretes one full chain snapshot per symbol per day,
// plus a derived daily panel row so history is cheap to read later.
// Files live in .cache/iv/{SYMBOL}/{YYYY-MM-DD}.json and .cache/iv/{SYMBOL}/panel.json
// (gitignored). Server-side only — never import from client components.
import fs from 'node:fs';
import path from 'node:path';
import type { RawChainRow } from './options/providers/types';
import type { TrapState, TrapDirection } from './gex';

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
  /** v2: full raw rows (OI > 0, ≤ 90 DTE) — present only for v2 snapshots */
  rows?: RawChainRow[];
}

export interface WallSnapshot {
  strike: number;
  oi: number;
  share: number;
  distPct: number;
  distSigma: number;
}

export interface PanelRow {
  date: string;
  spot: number;
  atmIv30: number | null;
  expectedMove30: number | null;
  totalGex: number | null;
  gammaFlip: number | null;
  regime: 'long' | 'short' | 'unknown';
  callWall: WallSnapshot | null;
  putWall: WallSnapshot | null;
  expiryConcentration7d: number | null;
  /** Δ OI at the trapped wall (nearer of call/put), 1 session back */
  oiWallChange1d: number | null;
  oiWallChange5d: number | null;
  /** z-score of oiWallChange5d vs trailing 60 Δ5d values */
  oiWallChangeZ: number | null;
  trapScore: number | null;
  trapState: TrapState | null;
  trapDirection: TrapDirection | null;
  rv20: number | null;
  ret5d: number | null;
  computedAt: string;
}

const root = () => path.join(process.cwd(), '.cache', 'iv');

const safeSymbol = (s: string) => s.toUpperCase().replace(/[^A-Z0-9._-]/g, '_').slice(0, 32);

export function snapshotDir(symbol: string): string {
  return path.join(root(), safeSymbol(symbol));
}

const freshMinutes = 20;
const dateOf = (iso: string) => iso.split('T')[0];

/** Idempotent write: reuses today's file when < 20 min old; overwrites stale same-day files. */
export function writeSnapshot(snap: IvSnapshot): { ok: boolean; error?: string; reused?: boolean } {
  try {
    const dir = snapshotDir(snap.symbol);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${snap.date}.json`);
    if (fs.existsSync(file)) {
      try {
        const existing = JSON.parse(fs.readFileSync(file, 'utf8')) as IvSnapshot;
        const ageMin = (Date.now() - new Date(existing.fetchedAt).getTime()) / 60000;
        if (ageMin < freshMinutes && existing.spot === snap.spot) {
          return { ok: true, reused: true };
        }
      } catch {
        /* corrupt — overwrite */
      }
    }
    // Never persist an empty row set when we have rows to keep
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

const panelPath = (symbol: string) => path.join(snapshotDir(symbol), 'panel.json');

export function readPanel(symbol: string): PanelRow[] {
  try {
    const file = panelPath(symbol);
    if (!fs.existsSync(file)) return [];
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as PanelRow[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** Append or replace today's panel row; sorted by date. Returns the full series. */
export function writePanelRow(symbol: string, row: PanelRow): { ok: boolean; error?: string } {
  try {
    const dir = snapshotDir(symbol);
    fs.mkdirSync(dir, { recursive: true });
    const rows = readPanel(symbol);
    const idx = rows.findIndex((r) => r.date === row.date);
    if (idx >= 0) rows[idx] = row;
    else rows.push(row);
    rows.sort((a, b) => a.date.localeCompare(b.date));
    fs.writeFileSync(panelPath(symbol), JSON.stringify(rows, null, 2), 'utf8');
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/** Per-day OI at a given strike for the call or put side (for ΔOI tracking). */
export function oiAtStrikeByDate(
  history: IvSnapshot[],
  strike: number,
  type: 'call' | 'put',
  maxAgeDays: number
): { date: string; oi: number }[] {
  const out: { date: string; oi: number }[] = [];
  const today = new Date().toISOString().split('T')[0];
  const cutoff = new Date(Date.now() - maxAgeDays * 86400000).toISOString().split('T')[0];
  for (const snap of history) {
    if (snap.date >= today) continue; // exclude today — ΔOI must compare prior days
    if (snap.date < cutoff) continue;
    if (!snap.rows) continue;
    let oi = 0;
    for (const r of snap.rows) {
      if (r.type === type && r.strike === strike) oi += r.oi;
    }
    out.push({ date: snap.date, oi });
  }
  return out.sort((a, b) => a.date.localeCompare(b.date));
}
