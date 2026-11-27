// One decision object for the two options surfaces (/api/options and /api/fno).
// Every verdict below names its estimator so a "RICH" here cannot be confused
// with an unrelated "CHEAP" elsewhere. Pure — no I/O.

import { canonicalVrp, curveVerdict, selectFrontSlice } from './ticker-snapshot';

export type GexRegime = 'long' | 'short' | 'unknown';

export interface DecisionTrap {
  available?: boolean;
  state: string;
  score: number;
  spot?: number | null;
  totalGex?: number | null;
  regime?: string | null;
  gammaFlip?: { flip: number | null; regime?: string } | null;
  flipDistPct?: number | null;
  gexPerAdv?: number | null;
}

export interface DecisionChain {
  iv30?: number | null;
  available?: boolean;
  trap?: DecisionTrap | null;
  slices?: { dte: number; atmIv: number | null; rr25?: number | null; bf25?: number | null }[];
  termStructure?: { dte: number; iv: number | null }[];
  clean?: { used: number; total: number; medianRelSpread: number | null };
  snapshot?: { count: number };
}

export interface DecisionFnoPanel {
  maxPain?: { maxPainStrike: number | null } | null;
  pcr?: { aggregate: number | null; volume: number | null; atm: number | null } | null;
  strikesUsed?: number;
  strikesTotal?: number;
  iv30?: number | null;
  trap?: DecisionTrap | null;
}

export interface OptionsDecision {
  iv30: number | null;
  rv20: number | null;
  vrpVariancePoints: number | null;
  curve: 'CONTANGO' | 'FLAT' | 'INVERTED' | null;
  trap: { state: string; score: number } | null;
  gammaFlip: number | null;
  flipDistPct: number | null;
  totalGex: number | null;
  gexRegime: GexRegime;
  gexPerAdv: number | null;
  frontSkew: { dte: number; rr25: number | null; bf25: number | null } | null;
  positioning: {
    maxPainStrike: number | null;
    pcr: number | null;
    strikesUsed: number | null;
    strikesTotal: number | null;
  } | null;
  accruedSessions: number | null;
  verdict: 'RICH' | 'CHEAP' | 'NEUTRAL' | 'insufficient-data';
  method: string;
  notes: string[];
}

function normalizeRegime(raw: string | null | undefined, totalGex: number | null): GexRegime {
  if (raw === 'long' || raw === 'short' || raw === 'unknown') return raw;
  if (totalGex !== null && Math.abs(totalGex) >= 1e-8) {
    return totalGex > 0 ? 'long' : 'short';
  }
  return 'unknown';
}

/** Build the single read used by the Now view, Volatility T1/T3 and the F&O suite. */
export function buildOptionsDecision({
  chain,
  fnoPanel,
  rv20,
}: {
  chain?: DecisionChain | null;
  fnoPanel?: DecisionFnoPanel | null;
  rv20?: number | null;
}): OptionsDecision {
  const notes: string[] = [];
  const iv30 = chain?.iv30 ?? fnoPanel?.iv30 ?? null;
  const vrp = canonicalVrp(iv30, rv20 ?? null);
  const curve = chain ? curveVerdict(chain.termStructure ?? []) : null;
  const front = chain ? selectFrontSlice(chain.slices ?? []) : null;

  const rawTrap =
    chain?.trap && chain.trap.available !== false
      ? chain.trap
      : fnoPanel?.trap && fnoPanel.trap.available !== false
        ? fnoPanel.trap
        : null;

  const trap =
    rawTrap && typeof rawTrap.score === 'number'
      ? { state: rawTrap.state, score: rawTrap.score }
      : null;

  const gammaFlip =
    rawTrap?.gammaFlip && typeof rawTrap.gammaFlip.flip === 'number' && isFinite(rawTrap.gammaFlip.flip)
      ? rawTrap.gammaFlip.flip
      : null;

  const totalGex =
    typeof rawTrap?.totalGex === 'number' && isFinite(rawTrap.totalGex)
      ? rawTrap.totalGex
      : null;

  const gexRegime = rawTrap
    ? normalizeRegime(rawTrap.regime ?? rawTrap.gammaFlip?.regime, totalGex)
    : 'unknown';

  const flipDistPct =
    typeof rawTrap?.flipDistPct === 'number' && isFinite(rawTrap.flipDistPct)
      ? rawTrap.flipDistPct
      : gammaFlip !== null && typeof rawTrap?.spot === 'number' && isFinite(rawTrap.spot) && rawTrap.spot > 0
        ? gammaFlip / rawTrap.spot - 1
        : null;

  const gexPerAdv =
    typeof rawTrap?.gexPerAdv === 'number' && isFinite(rawTrap.gexPerAdv)
      ? rawTrap.gexPerAdv
      : null;

  const used = chain?.clean?.used ?? fnoPanel?.strikesUsed ?? null;
  const total = chain?.clean?.total ?? fnoPanel?.strikesTotal ?? null;
  const positioning = fnoPanel
    ? {
        maxPainStrike: fnoPanel.maxPain?.maxPainStrike ?? null,
        pcr: fnoPanel.pcr?.aggregate ?? null,
        strikesUsed: fnoPanel.strikesUsed ?? null,
        strikesTotal: fnoPanel.strikesTotal ?? null,
      }
    : null;

  let verdict: OptionsDecision['verdict'] = 'insufficient-data';
  if (vrp.variancePoints !== null) {
    verdict = vrp.variancePoints > 5e-4 ? 'RICH' : vrp.variancePoints < -5e-4 ? 'CHEAP' : 'NEUTRAL';
  }

  if (curve === 'INVERTED') notes.push('Front end inverted — event noise dominates the multi-day read.');
  if (trap && (trap.state === 'armed' || trap.state === 'triggered')) notes.push(`Dealer trap ${trap.state} (score ${trap.score}).`);
  if (totalGex !== null) {
    const sign = totalGex >= 0 ? '+' : '-';
    const gexStr = `${sign}$${(Math.abs(totalGex) / 1e6).toFixed(1)}M`;
    notes.push(
      `Net GEX ${gexStr} (${gexRegime} γ${gammaFlip !== null ? `, flip $${gammaFlip.toFixed(1)}` : ''}) — assumed squeeze convention.`
    );
  }
  if (used !== null && total !== null) notes.push(`Strikes used ${used}/${total}.`);
  if (iv30 !== null) notes.push('IV: European BSM on ~15-min delayed quotes, not executable.');
  if (rv20 !== null) notes.push('VRP = IV²−RV² (IV √365, RV √252), compared directly.');

  return {
    iv30,
    rv20: rv20 ?? null,
    vrpVariancePoints: vrp.variancePoints,
    curve,
    trap,
    gammaFlip,
    flipDistPct,
    totalGex,
    gexRegime,
    gexPerAdv,
    frontSkew: front ? { dte: front.dte, rr25: front.rr25 ?? null, bf25: front.bf25 ?? null } : null,
    positioning,
    accruedSessions: chain?.snapshot?.count ?? null,
    verdict,
    method: 'IV²−RV² variance points · 15–45d front slice · trap from accrued panel',
    notes,
  };
}
