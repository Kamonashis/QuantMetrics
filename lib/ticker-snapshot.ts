// Canonical, pure reads shared by every workstation. One definition per quantity
// so the same number means the same thing wherever it is displayed.
// No I/O, no React.

import { calculateReturns, computeRollingAnnualizedVol, type RollingVolSeries } from './quant-engine';
import { percentileRank, termStructureVerdict } from './vol-analytics';

export interface RvReading {
  window: number;
  vol: number | null;
  n: number;
}

/** The canonical rolling series (sample stdev × √252) shared by every view. */
export function canonicalRvSeries(closes: number[], windows: number[] = [5, 10, 20, 30, 60]): RollingVolSeries[] {
  if (closes.length < 2) return windows.map((window) => ({ window, values: [] }));
  const dates = closes.map((_, i) => String(i));
  return computeRollingAnnualizedVol(dates, calculateReturns(closes, 'log'), windows);
}

/** RV ladder with one convention: sample stdev of log returns × √252. */
export function canonicalRvLadder(closes: number[], windows: number[] = [5, 10, 20, 30, 60]): RvReading[] {
  const series = canonicalRvSeries(closes, windows);
  return series.map((s) => {
    const valid = s.values.filter((v): v is { date: string; vol: number } => v.vol !== null && isFinite(v.vol));
    return { window: s.window, vol: valid.length ? valid[valid.length - 1].vol : null, n: valid.length };
  });
}

/** One front-slice rule for skew/RR tiles: 15–45 DTE, else nearest to 21. */
export function selectFrontSlice<T extends { dte: number; atmIv: number | null }>(slices: T[]): T | null {
  const valid = slices.filter((s) => s.atmIv !== null && isFinite(s.atmIv) && s.dte > 0);
  if (valid.length === 0) return null;
  const window = valid.filter((s) => s.dte >= 15 && s.dte <= 45);
  if (window.length > 0) return window.reduce((a, b) => (Math.abs(a.dte - 21) <= Math.abs(b.dte - 21) ? a : b));
  return valid.reduce((a, b) => (Math.abs(a.dte - 21) <= Math.abs(b.dte - 21) ? a : b));
}

/** VRP in variance points from one definition: IV² − RV² (both annualized). */
export function canonicalVrp(iv: number | null, rv: number | null): {
  variancePoints: number | null;
  premiumPoints: number | null;
} {
  if (iv === null || rv === null || !isFinite(iv) || !isFinite(rv)) {
    return { variancePoints: null, premiumPoints: null };
  }
  return { variancePoints: (iv * iv - rv * rv) * 1e4, premiumPoints: (iv - rv) * 100 };
}

/** Day-over-day changes from the accrued panel store (last two dated rows). */
export interface PanelLike {
  date: string;
  spot: number | null;
  atmIv30: number | null;
  trapScore: number | null;
  rv20: number | null;
}

export function sessionDiff(rows: PanelLike[]): (PanelLike & {
  dSpot: number | null;
  dIv30: number | null;
  dTrapScore: number | null;
  dRv20: number | null;
}) | null {
  const valid = rows.filter((r) => r && r.date).sort((a, b) => a.date.localeCompare(b.date));
  if (valid.length < 2) return null;
  const now = valid[valid.length - 1];
  const prev = valid[valid.length - 2];
  const delta = (a: number | null, b: number | null) =>
    a !== null && b !== null && isFinite(a) && isFinite(b) ? a - b : null;
  return {
    ...now,
    dSpot: delta(now.spot, prev.spot),
    dIv30: delta(now.atmIv30, prev.atmIv30),
    dTrapScore: delta(now.trapScore, prev.trapScore),
    dRv20: delta(now.rv20, prev.rv20),
  };
}

/** VIX level + one-year percentile, computed once for the whole app. */
export function vixContext(bars: { date: string; close: number }[], current: number | null): {
  level: number | null;
  percentile: number | null;
  n: number;
} {
  const values = bars.slice(-252).map((b) => b.close).filter((v) => isFinite(v));
  return {
    level: current,
    percentile: current !== null && values.length > 30 ? percentileRank(values, current) : null,
    n: values.length,
  };
}

/** Curve verdict from one term-structure rule. */
export function curveVerdict(term: { dte: number; iv: number | null }[]): 'CONTANGO' | 'FLAT' | 'INVERTED' | null {
  const valid = term.filter((p) => p.iv !== null && isFinite(p.iv) && p.iv > 0 && p.dte > 0);
  if (valid.length < 2) return null;
  const front = valid.reduce((a, b) => (a.dte <= b.dte ? a : b)).iv!;
  const back = valid.filter((p) => p.dte >= 45).length > 0
    ? valid.filter((p) => p.dte >= 45).reduce((a, b) => (a.dte >= b.dte ? a : b)).iv!
    : valid.reduce((a, b) => (a.dte >= b.dte ? a : b)).iv!;
  return termStructureVerdict(front, back);
}
