// Term structure & carry: ATM IV across expiries, contango/backwardation verdict,
// forward carry between consecutive expiries, 25Δ risk reversal per expiry.
import {
  impliedForward,
  fitSvi,
  skewMetrics,
  cleanChain,
} from '../options';
import type { ExpiryGroup, TermStructurePoint, TermStructureVerdict, ForwardCarryPair, RawChainRow } from './types';

interface LadderWithRows extends ExpiryGroup {
  rows: RawChainRow[];
}

export function buildTermStructure(
  ladder: LadderWithRows[],
  spot: number,
  rate: number,
  dividendYield: number
): {
  points: TermStructurePoint[];
  verdict: TermStructureVerdict;
  forwardCarry: ForwardCarryPair[];
} {
  const points: TermStructurePoint[] = ladder.map((l) => {
    let rr25: number | null = null;
    try {
      const contracts = l.rows
        .filter((r) => r.bid > 0 && r.oi >= 10)
        .map((r) => ({
          strike: r.strike,
          bid: r.bid,
          ask: r.ask,
          lastPrice: r.last,
          openInterest: r.oi,
          volume: r.volume,
          impliedVolatility: r.iv ?? 0,
          inTheMoney: r.type === 'call' ? r.strike < spot : r.strike > spot,
        }));
      const calls = contracts.filter((c) => c.strike >= spot);
      const puts = contracts.filter((c) => c.strike < spot);
      if (calls.length >= 2 && puts.length >= 2) {
        const fwdRes = impliedForward(calls, puts, l.t, rate);
        const forward = fwdRes?.forward ?? spot;
        const clean = cleanChain(calls, puts, spot, forward, l.t, rate, dividendYield);
        const used = [...clean.calls, ...clean.puts];
        const fit = fitSvi(used);
        if (fit && used.length >= 5) {
          const ks = used.map((c) => c.k);
          const sk = skewMetrics(
            fit, spot, forward, l.t, rate, dividendYield,
            Math.max(-0.9, Math.min(...ks)), Math.min(0.9, Math.max(...ks))
          );
          rr25 = sk.rr25;
        }
      }
    } catch {
      rr25 = null;
    }
    return { dte: l.dte, expiry: l.expiry, iv: l.atmIv, rr25 };
  });

  const valid = points.filter((p) => p.iv !== null && (p.iv as number) > 0) as Array<TermStructurePoint & { iv: number }>;
  // Verdict anchors the front to ≥ 10 DTE — the front week is event-noise (0–7 DTE) and shouldn't set the regime.
  const frontForVerdict = valid.find((p) => p.dte >= 10) ?? valid[0] ?? null;
  const frontIv = frontForVerdict?.iv ?? null;
  const back = [...valid].reverse().find((p) => p.dte >= 60);
  const backIv = back?.iv ?? null;
  const ratio = frontIv !== null && backIv !== null && backIv > 0 ? frontIv / backIv : null;
  const verdict: TermStructureVerdict['verdict'] =
    ratio === null ? 'FLAT' : ratio > 1.04 ? 'BACKWARDATION' : ratio < 0.96 ? 'CONTANGO' : 'FLAT';

  const forwardCarry: ForwardCarryPair[] = [];
  for (let i = 0; i < ladder.length - 1; i++) {
    const a = ladder[i];
    const b = ladder[i + 1];
    const fA = parityForward(a.rows, spot, rate, a.t);
    const fB = parityForward(b.rows, spot, rate, b.t);
    if (fA && fB && fA > 0 && fB > 0) {
      const gapDays = Math.max(1, b.dte - a.dte);
      const carry = (fB / fA) ** (365 / gapDays) - 1;
      forwardCarry.push({ fromDte: a.dte, toDte: b.dte, carry });
    }
  }

  return { points, verdict: { verdict, frontIv, backIv, ratio }, forwardCarry };
}

function parityForward(
  rows: RawChainRow[],
  spot: number,
  rate: number,
  t: number
): number | null {
  const calls = rows
    .filter((r) => r.type === 'call')
    .map((r) => ({
      strike: r.strike, bid: r.bid, ask: r.ask, lastPrice: r.last, openInterest: r.oi,
      volume: r.volume, impliedVolatility: r.iv ?? 0, inTheMoney: r.strike < spot,
    }));
  const puts = rows
    .filter((r) => r.type === 'put')
    .map((r) => ({
      strike: r.strike, bid: r.bid, ask: r.ask, lastPrice: r.last, openInterest: r.oi,
      volume: r.volume, impliedVolatility: r.iv ?? 0, inTheMoney: r.strike > spot,
    }));
  const res = impliedForward(calls, puts, t, rate);
  return res?.forward ?? null;
}
