// =============================================================
// QUANTMETRICS — EARNINGS PARSER (pure)
// =============================================================
// Parses Yahoo quoteSummary v10 (calendarEvents module) into the app's
// earnings shape. No I/O, no clock — deterministic for unit tests.
// Yahoo's earnings surface is the flakiest we touch; every consumer must
// treat null as "no earnings data", never as "no earnings".

export interface EarningsCalendar {
  /** Earliest earnings date on/after `nowYmd` when provided, else first parsed date. YYYY-MM-DD (UTC). */
  nextEarningsDate: string | null;
  /** All dates found, ascending, YYYY-MM-DD (UTC). */
  allDates: string[];
  /** Whether the date is Yahoo's estimate vs confirmed (from isEarningsDateEstimate). */
  isEstimate: boolean;
}

function toYmd(epochSec: unknown): string | null {
  const raw = typeof epochSec === 'number' ? epochSec : Number(epochSec);
  if (!Number.isFinite(raw) || raw <= 0) return null;
  const d = new Date(raw * 1000);
  return isNaN(d.getTime()) ? null : d.toISOString().slice(0, 10);
}

/**
 * Parse a quoteSummary JSON payload. Defensive across shapes:
 *  - calendarEvents.earnings.earningsDate: [{ raw, fmt }, …]
 *  - calendarEvents.earnings.earningsDateNext: { raw, fmt } (older nesting)
 * Returns null when the payload is missing/garbage — never throws.
 * @param json  Raw parsed JSON from the endpoint.
 * @param nowYmd  Optional YYYY-MM-DD lower bound; when given, `nextEarningsDate` is the first date >= it.
 */
export function parseQuoteSummaryEarnings(json: unknown, nowYmd?: string): EarningsCalendar | null {
  try {
    const root = (json as { quoteSummary?: { result?: unknown[] | null } } | null)?.quoteSummary;
    const result = root?.result?.[0] as
      | { calendarEvents?: { earnings?: Record<string, unknown> } }
      | undefined;
    const earnings = result?.calendarEvents?.earnings;
    if (!earnings || typeof earnings !== 'object') return null;

    const dates = new Set<string>();
    const collect = (node: unknown) => {
      if (!node) return;
      if (Array.isArray(node)) {
        for (const item of node) {
          const ymd = toYmd((item as Record<string, unknown>)?.raw);
          if (ymd) dates.add(ymd);
        }
      } else if (typeof node === 'object') {
        const ymd = toYmd((node as Record<string, unknown>)?.raw);
        if (ymd) dates.add(ymd);
      }
    };
    collect(earnings.earningsDate);
    collect((earnings as Record<string, unknown>).earningsDateNext);

    if (dates.size === 0) return null;
    const allDates = [...dates].sort();
    const isEstimate = (earnings as Record<string, unknown>).isEarningsDateEstimate !== false;

    let nextEarningsDate: string | null = allDates[0];
    if (nowYmd) {
      nextEarningsDate = allDates.find((d) => d >= nowYmd) ?? null;
    }
    return { nextEarningsDate, allDates, isEstimate };
  } catch {
    return null;
  }
}

/**
 * Rough trading sessions between two YYYY-MM-DD dates (calendar days × 5/7, floor).
 * For display only ("≈N sessions") — not an exchange calendar.
 */
export function sessionsUntil(targetYmd: string, fromYmd: string): number {
  const t = Date.parse(`${targetYmd}T00:00:00Z`);
  const f = Date.parse(`${fromYmd}T00:00:00Z`);
  if (!Number.isFinite(t) || !Number.isFinite(f)) return 0;
  const calDays = Math.max(0, Math.round((t - f) / 86_400_000));
  return Math.floor((calDays * 5) / 7);
}
