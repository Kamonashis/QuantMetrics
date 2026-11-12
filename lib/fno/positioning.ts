// Positioning primitives: OI ladder, put/call ratios, and the Option-Pain profile.
// Every function is total (returns null / empty structures on bad input) — never throws.
import type { RawChainRow, OiLadderPoint, PcrBundle, MaxPainResult, MaxPainPoint } from './types';

/** Dual-sided OI by strike. Keeps strikes within ±maxPct of spot. */
export function oiLadder(
  rows: RawChainRow[],
  spot: number,
  maxPct = 0.15
): OiLadderPoint[] {
  const map = new Map<number, OiLadderPoint>();
  const lo = spot * (1 - maxPct);
  const hi = spot * (1 + maxPct);
  for (const r of rows) {
    if (r.strike < lo || r.strike > hi) continue;
    const b = map.get(r.strike) ?? { strike: r.strike, callOi: 0, putOi: 0, totalOi: 0 };
    if (r.type === 'call') b.callOi += r.oi;
    else b.putOi += r.oi;
    b.totalOi = b.callOi + b.putOi;
    map.set(r.strike, b);
  }
  return Array.from(map.values()).sort((a, b) => a.strike - b.strike);
}

/** Put/call ratios — aggregate, by volume, and a ±5-strike window around spot. */
export function pcr(
  rows: RawChainRow[],
  spot: number
): PcrBundle {
  let callOi = 0, putOi = 0, callVol = 0, putVol = 0;
  for (const r of rows) {
    if (r.type === 'call') {
      callOi += r.oi;
      callVol += r.volume;
    } else {
      putOi += r.oi;
      putVol += r.volume;
    }
  }
  const aggregate = callOi > 0 ? putOi / callOi : null;
  const volume = callVol > 0 ? putVol / callVol : null;

  const strikes = Array.from(new Set(rows.map((r) => r.strike))).sort((a, b) => a - b);
  const atmIdx = strikes.reduce(
    (best, s, i) => (Math.abs(s - spot) < Math.abs(strikes[best] - spot) ? i : best),
    0
  );
  const lo = Math.max(0, atmIdx - 5);
  const hi = Math.min(strikes.length, atmIdx + 6);
  const windowSet = new Set(strikes.slice(lo, hi));
  let cAtm = 0, pAtm = 0;
  for (const r of rows) {
    if (!windowSet.has(r.strike)) continue;
    if (r.type === 'call') cAtm += r.oi;
    else pAtm += r.oi;
  }
  const atm = cAtm > 0 ? pAtm / cAtm : null;

  return { aggregate, volume, atm };
}

/**
 * Classic max-pain: strike K minimizing total payout to option buyers
 * Σ_i OI_i·max(K−K_i, 0) for puts + Σ_i OI_i·max(K_i−K, 0) for calls.
 * Evaluated only on strikes present in the chain with OI > 0.
 */
export function maxPain(
  rows: RawChainRow[],
  spot: number
): MaxPainResult {
  const byStrike = new Map<number, { call: number; put: number }>();
  for (const r of rows) {
    if (r.oi <= 0) continue;
    const b = byStrike.get(r.strike) ?? { call: 0, put: 0 };
    if (r.type === 'call') b.call += r.oi;
    else b.put += r.oi;
    byStrike.set(r.strike, b);
  }
  if (byStrike.size < 3) {
    return { maxPainStrike: null, pain: 0, distancePct: null, basinWidthPct: null, curve: [] };
  }
  const strikes = Array.from(byStrike.keys()).sort((a, b) => a - b);
  const lo = spot * 0.75;
  const hi = spot * 1.25;
  const candidateStrikes = strikes.filter((K) => K >= lo && K <= hi);

  const curve: MaxPainPoint[] = candidateStrikes.map((K) => {
    let payout = 0;
    for (const [Ki, o] of byStrike) {
      if (o.put > 0) payout += o.put * Math.max(K - Ki, 0);
      if (o.call > 0) payout += o.call * Math.max(Ki - K, 0);
    }
    return { strike: K, pain: payout };
  });
  const minPain = Math.min(...curve.map((p) => p.pain));
  const maxPainStrike = curve.find((p) => p.pain === minPain)?.strike ?? null;

  // Basin width: strikes within 5% of the minimum pain level, as % of spot
  const nearMin = curve.filter((p) => p.pain <= minPain * 1.05).map((p) => p.strike);
  const basinWidthPct =
    nearMin.length > 1 ? ((nearMin[nearMin.length - 1] - nearMin[0]) / spot) * 100 : 0;

  return {
    maxPainStrike,
    pain: minPain,
    distancePct: maxPainStrike !== null ? ((spot - maxPainStrike) / spot) * 100 : null,
    basinWidthPct,
    curve,
  };
}
