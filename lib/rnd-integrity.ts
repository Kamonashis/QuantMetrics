import { bsPrice, normCdf, riskNeutralProbabilities } from './options';
import { buildCallCurve, densityCdf, densityMean, extendWithSviTails, impliedDensityFromCalls, type DensityGrid, type SlicePointLike } from './rnd';

export const MAX_SPOT_GAP = 0.0025;
export const MFIV_REVIEW_POINTS = 2;
export type Quality = 'READY' | 'DEGRADED' | 'NOT_READY';

interface ResearchChain {
  available: boolean;
  quote?: { price: number; asOf?: string | null };
  rate?: number;
  dividendYield?: number;
  rnd?: (DensityGrid & { integrated: number; clippedFraction: number; forwardError: number; edgeMass: number }) | null;
  slices?: {
    dte: number; expirationDate: string; t: number; forward: number; atmIv: number | null; mfivAnnualized: number | null;
    svi?: { arbitrageFlags: string[] } | null;
    points: SlicePointLike[];
  }[];
}

export function lowerTouchGbm(barrierRatio: number, t: number, vol: number, carry: number) {
  if (!(barrierRatio > 0 && barrierRatio < 1 && t > 0 && vol > 0)) return null;
  const a = Math.log(barrierRatio), mu = carry - vol ** 2 / 2, scale = vol * Math.sqrt(t);
  const terminal = normCdf((a - mu * t) / scale);
  const reflection = Math.exp(2 * mu * a / vol ** 2) * normCdf((a + mu * t) / scale);
  return Number.isFinite(terminal + reflection) ? Math.min(1, Math.max(0, terminal + reflection)) : null;
}

export function spotGap(a?: { price: number }, b?: { price: number }) {
  return a && b && a.price > 0 && b.price > 0 && Number.isFinite(a.price + b.price)
    ? Math.abs(a.price - b.price) / a.price : null;
}

export function logMoments(grid: DensityGrid, spot: number) {
  const x = (k: number) => Math.log(k / spot);
  const mean = densityMean(grid, x);
  const variance = densityMean(grid, (k) => (x(k) - mean) ** 2);
  const std = Math.sqrt(variance);
  return { mean, std, skew: std > 0 ? densityMean(grid, (k) => ((x(k) - mean) / std) ** 3) : null,
    kurt: std > 0 ? densityMean(grid, (k) => ((x(k) - mean) / std) ** 4) - 3 : null };
}

export function sampleMoments(xs: number[]) {
  if (xs.length < 10) return null;
  const avg = (f: (x: number) => number) => xs.reduce((s, x) => s + f(x), 0) / xs.length;
  const mean = avg((x) => x);
  const std = Math.sqrt(avg((x) => (x - mean) ** 2));
  return { mean, std, skew: std > 0 ? avg((x) => ((x - mean) / std) ** 3) : null,
    kurt: std > 0 ? avg((x) => ((x - mean) / std) ** 4) - 3 : null };
}

const DAY = 86_400_000;
export function historicalHorizonReturns(bars: { date: string; close: number }[], days: number) {
  if (!Number.isInteger(days) || days < 1 || bars.length < 2) return [];
  const sorted = [...new Map(bars.map((b) => [b.date, b])).values()].sort((a, b) => a.date.localeCompare(b.date));
  const times = sorted.map((b) => Date.parse(b.date));
  const out: number[] = [];
  for (let i = 0, j = 0; i < sorted.length; i++) {
    const target = times[i] + days * DAY;
    if (target > times[times.length - 1]) break;
    while (j + 1 < sorted.length && times[j + 1] <= target) j++;
    if (j > i && target - times[j] <= 3 * DAY && sorted[i].close > 0 && sorted[j].close > 0) {
      out.push(Math.log(sorted[j].close / sorted[i].close));
    }
  }
  return out;
}

type ComparisonQuote = { price: number; asOf?: string | null };

export function buildRndResearch(chain: ResearchChain | null, comparison?: ComparisonQuote, underlying?: ComparisonQuote) {
  const slices = chain?.slices ?? [];
  const slice = slices.find((s) => s.dte >= 21) ?? slices[0];
  const spot = chain?.quote?.price ?? 0;
  const rate = chain?.rate;
  const q = chain?.dividendYield;
  if (!chain?.available || !slice || !(spot > 0 && slice.t > 0 && slice.forward > 0) ||
    rate === undefined || q === undefined || !Number.isFinite(rate + q)) return null;
  const { t, forward, atmIv } = slice;
  const curve = buildCallCurve(slice.points, { forward, t, rate });
  const market = curve && impliedDensityFromCalls(curve, { spot, forward, t, rate });
  const carryForward = spot * Math.exp((rate - q) * t);
  const forwardAgreement = Math.abs(carryForward / forward - 1) <= 0.02;
  const fit = chain.rnd;
  const fitted = fit?.cdf && !slice.svi?.arbitrageFlags.length && Math.abs(fit.integrated - 1) <= 0.02 &&
    fit.clippedFraction <= 0.005 && fit.forwardError <= 0.02 && fit.edgeMass <= 0.005 && forwardAgreement ? fit : null;
  const marketClean = market && market.rawArea >= 0.9 && market.rawArea <= 1.1 && market.clippedFraction <= 0.05 &&
    Math.abs(densityMean(market, (k) => k) / forward - 1) <= 0.02;
  const hybrid = marketClean && market && fitted ? extendWithSviTails(market, fitted) : null;
  const mfivDiff = slice.mfivAnnualized != null && atmIv != null ? (slice.mfivAnnualized - atmIv) * 100 : null;
  const gap = spotGap(chain.quote, comparison);
  const underlyingGap = spotGap(chain.quote, underlying);
  const quoteTime = chain.quote?.asOf;
  const timeGap = Math.abs(Date.parse(quoteTime ?? '') - Date.parse(comparison?.asOf ?? '')) / 60_000;
  const timeOkay = Number.isFinite(timeGap) && timeGap <= 45;
  const badSpot = (gap !== null && gap > MAX_SPOT_GAP) || (underlyingGap !== null && underlyingGap > MAX_SPOT_GAP);
  const supportReady = !!hybrid && !!market && market.strikeMin <= 0.8 * spot && market.strikeMax >= 1.2 * spot &&
    hybrid.leftTailMass + hybrid.rightTailMass <= 0.05;
  const mfivOkay = mfivDiff !== null && Math.abs(mfivDiff) <= MFIV_REVIEW_POINTS;
  const checks = [
    { label: 'Spot coherence', pass: !badSpot && gap !== null && timeOkay,
      detail: `${gap === null ? 'No comparable positioning quote.' : `${(gap * 100).toFixed(2)}% chain-to-positioning difference.${!Number.isFinite(timeGap) ? ' Vendor timestamp unavailable for at least one feed.' : ` Quote timestamps ${Math.round(timeGap)} min apart (45 min review threshold).`}`}${underlyingGap !== null ? ` ${(underlyingGap * 100).toFixed(2)}% chain-to-market quote difference.` : ''} Price review threshold: 0.25%.` },
    { label: 'Market density', pass: supportReady, detail: market ? `${market.rawArea.toFixed(2)} pre-normalization positive area; ${(market.clippedFraction * 100).toFixed(1)}% negative curvature removed; strikes ${market.strikeMin.toFixed(2)}–${market.strikeMax.toFixed(2)}.` : 'Insufficient usable strikes.' },
    { label: 'Fitted density', pass: !!fitted, detail: fitted ? `Unit mass; ${(fitted.edgeMass * 100).toFixed(2)}% estimated omitted mass; forward check passed.` : 'Missing fit or failed curvature, tail, mass, or forward check.' },
    { label: 'MFIV vs ATM IV', pass: mfivOkay, detail: mfivDiff === null ? 'Comparison unavailable.' : `${mfivDiff.toFixed(2)} vol pts. A >2 pt gap warrants review; skew alone can also cause it.` },
  ];
  const quality: Quality = badSpot || !fitted && !marketClean ? 'NOT_READY' : checks.every((c) => c.pass) ? 'READY' : 'DEGRADED';
  const atm = atmIv != null && atmIv > 0 ? atmIv : null;
  const disc = Math.exp(-rate * t);
  const usableFit = badSpot ? null : fitted;
  const usableHybrid = badSpot ? null : hybrid;
  const prob = (grid: DensityGrid | null, k: number, above: boolean) => grid && k > grid.strikes[0] && k < grid.strikes[grid.strikes.length - 1]
    ? (above ? 1 - densityCdf(grid, k) : densityCdf(grid, k)) : null;
  const digital = (multiple: number, above: boolean) => {
    const strike = multiple * spot;
    const bsAbove = atm !== null ? riskNeutralProbabilities(spot, strike, t, rate, q, atm).probAbove : null;
    return { strike, market: prob(usableHybrid, strike, above), fitted: prob(usableFit, strike, above), bs: bsAbove === null ? null : above ? bsAbove : 1 - bsAbove };
  };
  const tailShare = (payoff: (k: number) => number, breaks: number[], value: number | null) => {
    if (!usableHybrid || !market || value == null || value <= 0) return null;
    const tail = densityMean(usableHybrid, (k) => k < market.strikeMin || k > market.strikeMax ? payoff(k) : 0,
      [...breaks, market.strikeMin, market.strikeMax]);
    return Math.min(1, Math.max(0, tail / value));
  };
  const rows = [
    { label: 'Digital call · +10%', multiple: 1.1, above: true },
    { label: 'Digital put · −10%', multiple: 0.9, above: false },
    { label: 'Digital call · +20%', multiple: 1.2, above: true },
  ].map(({ label, multiple, above }) => {
    const d = digital(multiple, above);
    return { label, ...d, market: d.market === null ? null : disc * d.market, fitted: d.fitted === null ? null : disc * d.fitted,
      bs: d.bs === null ? null : disc * d.bs,
      tailShare: tailShare((k) => (above ? k > d.strike : k < d.strike) ? 1 : 0, [d.strike], d.market) };
  });
  const upper = spot * 1.1;
  const spreadPayoff = (k: number) => Math.min(Math.max(k - spot, 0), upper - spot);
  const spread = (grid: DensityGrid | null) => grid && upper < grid.strikes[grid.strikes.length - 1] && spot > grid.strikes[0]
    ? disc * densityMean(grid, spreadPayoff, [spot, upper]) : null;
  const call = (strike: number) => bsPrice({ spot, strike, t, rate, dividendYield: q, vol: atm!, type: 'call' });
  const hybridSpread = spread(usableHybrid);
  rows.push({ label: 'Call spread · spot to +10%', strike: upper, market: hybridSpread, fitted: spread(usableFit), bs: atm === null ? null : call(spot) - call(upper),
    tailShare: tailShare(spreadPayoff, [spot, upper], hybridSpread === null ? null : hybridSpread / disc) });
  const supported = (k: number) => market && k > market.strikeMin && k < market.strikeMax;
  return { spot, slice, market, fitted: usableFit, hybrid: usableHybrid, checks, quality,
    rows: rows.map((row, i) => ({ ...row, inRange: !!supported(row.strike) && (i !== 3 || !!supported(spot)) })),
    probabilities: [
      { label: 'Above spot +10%', ...digital(1.1, true) }, { label: 'Above spot', ...digital(1, true) },
      { label: 'Below spot −10%', ...digital(0.9, false) }, { label: 'Below spot −20%', ...digital(0.8, false) },
      { label: 'Above spot +20%', ...digital(1.2, true) },
    ],
    atmDispersion: atm === null ? null : atm * Math.sqrt(t),
    touchDown10: atm === null || badSpot ? null : lowerTouchGbm(0.9, t, atm, rate - q),
    marketMoments: usableHybrid && logMoments(usableHybrid, spot), fittedMoments: usableFit && logMoments(usableFit, spot),
  };
}
