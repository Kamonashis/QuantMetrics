// Expiry ladder builder: groups provider rows by expiry, locates the near-ATM
// row(s), prices ATM IV via BSM Newton from the OTM mid price, and aggregates
// per-expiry open interest.
// Input rows must carry oi (or volume) > 0; rows with neither are dropped.

import {
  impliedForward,
  impliedVol,
  type OptionContract,
} from '../options';
import { MAX_DTE, TOP_EXPIRIES } from './config';
import type { ExpiryGroup, RawChainRow } from './types';

const toContract = (r: RawChainRow, spot: number): OptionContract => ({
  strike: r.strike,
  bid: r.bid,
  ask: r.ask,
  lastPrice: r.last,
  openInterest: r.oi,
  volume: r.volume,
  impliedVolatility: r.iv ?? 0,
  inTheMoney: r.type === 'call' ? r.strike < spot : r.strike > spot,
});

function expiryDatePrice(
  rows: RawChainRow[],
  spot: number,
  rate: number,
  dividendYield: number,
  t: number,
  dte: number
): { forward: number | null; atmStrike: number | null; atmIv: number | null } {
  const calls = rows.filter((r) => r.type === 'call').map((r) => toContract(r, spot));
  const puts = rows.filter((r) => r.type === 'put').map((r) => toContract(r, spot));
  const fwdRes = impliedForward(calls, puts, t, rate);
  const forward = fwdRes?.forward ?? null;

  // Candidate strikes: union of strikes seen in this expiry with usable prices
  const candidateRows = rows
    .filter((r) => (r.oi > 0 || r.volume > 0) && r.bid > 0 && r.ask >= r.bid)
    .map((r) => {
      const mid = (r.bid + r.ask) / 2;
      const k = forward ? Math.log(r.strike / forward) : Math.log(r.strike / spot);
      return { r, mid, k, absK: Math.abs(k) };
    })
    .sort((a, b) => a.absK - b.absK);

  const atmRow = candidateRows[0] ?? null;
  const atmStrike = atmRow?.r.strike ?? null;

  // ATM IV: Newton IV on the OTM row when it's an OTM call; else OTM put; else nearest OTM row IV
  let atmIv: number | null = null;
  if (atmRow && atmStrike !== null && forward) {
    const r = atmRow.r;
    const type = r.type;
    const iv = impliedVol(atmRow.mid, spot, r.strike, t, rate, dividendYield, type);
    if (iv !== null && isFinite(iv) && iv > 0.01 && iv < 3) {
      atmIv = iv;
    } else if (r.iv && r.iv > 0.01 && r.iv < 4) {
      atmIv = r.iv;
    }
  } else if (atmRow?.r.iv && atmRow.r.iv > 0.01 && atmRow.r.iv < 4) {
    atmIv = atmRow.r.iv;
  }

  return { forward, atmStrike, atmIv };
}

export function buildExpiryLadder(
  rows: RawChainRow[],
  spot: number,
  rate: number,
  dividendYield: number
): ExpiryGroup[] {
  if (!rows.length) return [];
  const byExpiry = new Map<string, RawChainRow[]>();
  for (const r of rows) {
    const arr = byExpiry.get(r.expiry) ?? [];
    arr.push(r);
    byExpiry.set(r.expiry, arr);
  }
  const groups: ExpiryGroup[] = [];
  for (const [expiry, rs] of byExpiry) {
    const d = Math.max(0.5, Math.min(...rs.map((r) => r.dte)));
    const t = d / 365;
    const { forward, atmStrike, atmIv } = expiryDatePrice(rs, spot, rate, dividendYield, t, d);
    let callOi = 0;
    let putOi = 0;
    for (const r of rs) {
      if (r.type === 'call') callOi += r.oi;
      else putOi += r.oi;
    }
    groups.push({
      expiry,
      dte: Math.round(d),
      t,
      forward,
      atmStrike,
      atmIv,
      callOi,
      putOi,
      totalOi: callOi + putOi,
      share7d: d <= 7 ? callOi + putOi : 0,
    });
  }
  const universeOi = groups.reduce((s, g) => s + g.totalOi, 0);
  const withShare = groups.map((g) => ({
    ...g,
    share7d: universeOi > 0 && g.dte <= 7 ? g.totalOi / universeOi : 0,
  }));
  return withShare
    .filter((g) => g.dte <= MAX_DTE)
    .sort((a, b) => a.dte - b.dte)
    .slice(0, TOP_EXPIRIES + 8); // 12 expiries — front for tables, back anchors for term structure
}
