// Trader-facing opportunity cards, assembled ONLY from signals the terminal
// already computes. No new alpha math, no I/O, no clock — every rule degrades
// to insufficient-data rather than inventing an edge. Pure and unit-testable.

import type { SignalRuleResult, SignalState } from './vol-signals';
import { BREAKEVEN_MULT } from './vol-signals';

export type OpportunityFamily =
  | 'premium-harvest'
  | 'long-gamma'
  | 'directional-trap'
  | 'event-straddle'
  | 'skew-tail'
  | 'cross-sectional';

export type OpportunityState = 'ACTIONABLE' | 'WATCH' | 'BLOCKED' | 'INSUFFICIENT_DATA';

export interface OpportunityTrigger {
  metric: string;
  measured: string | null;
  rule: string;
}

export interface OpportunityEdge {
  /** Human-readable, e.g. "+6.2 vol pts" or "68th pctile". */
  estimate: string | null;
  /** Sample size behind the estimate — honest, never fabricated. */
  n: number | null;
  provenance: string;
}

export interface OpportunitySize {
  scenario: string;
  basis: string;
  riskBudgetFraction: number;
}

export interface OpportunityEvidence {
  view: 'volatility' | 'fno' | 'analysis' | 'now';
  params: Record<string, string>;
  label: string;
}

export interface Opportunity {
  id: string;
  family: OpportunityFamily;
  title: string;
  thesis: string;
  state: OpportunityState;
  trigger: OpportunityTrigger;
  invalidation: string;
  edge: OpportunityEdge;
  size: OpportunitySize;
  horizon: string;
  evidence: OpportunityEvidence;
  confidence: 'low' | 'medium' | 'high';
  staleness: 'live' | 'delayed' | 'unknown';
  gates: string[];
}

export interface TrapForOpportunity {
  available?: boolean;
  state: string;
  direction: 'up' | 'down' | 'none';
  score: number | null;
  callWall: number | null;
  putWall: number | null;
  flip: number | null;
  regime?: 'long' | 'short' | 'unknown' | null;
  totalGex?: number | null;
  flipDistPct?: number | null;
  flipDistSigma?: number | null;
  gexPerAdv?: number | null;
  zeroDteShare?: number | null;
  backtestVerdict: string | null;
  backtestN: number | null;
  convention: string | null;
}

export interface ExpectedMoveForOpportunity {
  dte: number | null;
  impliedPct: number | null;
  verdict: string | null;
  percentile: number | null;
  n: number | null;
}

export interface CrossSectionalName {
  symbol: string;
  iv30: number | null;
  rv20: number | null;
  ivRank: number | null;
}

export interface OpportunityInputs {
  symbol: string;
  asOf: string | null;
  rules: SignalRuleResult[];
  iv30: number | null;
  spot: number | null;
  rv5: number | null;
  vrpVariancePoints: number | null;
  chainAvailable: boolean;
  fnoAvailable: boolean;
  barsSource: string | null;
  accruedSessions: number;
  ivRankReady: boolean;
  earningsSessions: number | null;
  earningsIsEstimate: boolean | null;
  term: { dte: number; iv: number | null }[];
  curve: 'CONTANGO' | 'FLAT' | 'INVERTED' | null;
  frontSkew: { dte: number; rr25: number | null; bf25: number | null } | null;
  vvix: number | null;
  vvixPercentile: number | null;
  skewPercentile: number | null;
  trap: TrapForOpportunity | null;
  expectedMove: ExpectedMoveForOpportunity | null;
  carry: { meanMonthlyBp2: number | null; hitRate: number | null; worstMonthBp2: number | null; maxDrawdownBp2: number | null; n: number | null } | null;
  crossSectional: CrossSectionalName[];
  riskBudgetFraction: number;
}

const pct = (v: number | null | undefined, d = 1) => v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const pts = (v: number | null | undefined, d = 1) => v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)} pts`;
const isNum = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v);

function fmtRegime(regime: TrapForOpportunity['regime']): string {
  if (regime === 'long') return 'Long γ';
  if (regime === 'short') return 'Short γ';
  return 'Unknown γ';
}

function fmtFlipContext(t: TrapForOpportunity): string | null {
  const parts: string[] = [];
  if (isNum(t.flipDistPct)) {
    parts.push(`${t.flipDistPct >= 0 ? '+' : ''}${(t.flipDistPct * 100).toFixed(1)}% vs flip`);
  }
  if (isNum(t.flipDistSigma)) {
    parts.push(`${t.flipDistSigma.toFixed(2)}σ`);
  }
  return parts.length > 0 ? parts.join(' / ') : null;
}

function fmtGexAdv(gexPerAdv: number | null | undefined): string | null {
  if (!isNum(gexPerAdv)) return null;
  return `GEX/ADV ${gexPerAdv >= 0 ? '+' : ''}${(gexPerAdv * 100).toFixed(1)}%`;
}

function buildGexReadout(t: TrapForOpportunity | null): string | null {
  if (!t || t.available !== true) return null;
  const parts: string[] = [];
  if (t.regime) parts.push(fmtRegime(t.regime));
  const flip = fmtFlipContext(t);
  if (flip) parts.push(flip);
  const adv = fmtGexAdv(t.gexPerAdv);
  if (adv) parts.push(adv);
  return parts.length > 0 ? parts.join(' · ') : null;
}

function ruleOf(inputs: OpportunityInputs, id: string): SignalRuleResult | null {
  return inputs.rules.find((r) => r.id === id) ?? null;
}

function stateFor(rule: SignalRuleResult | null, blockers: string[], needData: boolean): OpportunityState {
  if (needData) return 'INSUFFICIENT_DATA';
  if (blockers.length > 0) return 'BLOCKED';
  if (rule?.state === 'met') return 'ACTIONABLE';
  if (rule?.state === 'insufficient-data') return 'INSUFFICIENT_DATA';
  return 'WATCH';
}

function size(inputs: OpportunityInputs, basis: string, extra: string | null): OpportunitySize {
  const budget = Math.max(0, Math.min(0.1, inputs.riskBudgetFraction));
  return {
    riskBudgetFraction: budget,
    basis,
    scenario: `Scenario budget ${(budget * 100).toFixed(2)}% of book${extra ? ` · ${extra}` : ''}. Not a position recommendation.`,
  };
}

function stalenessOf(inputs: OpportunityInputs): Opportunity['staleness'] {
  if (!inputs.chainAvailable) return 'unknown';
  return inputs.barsSource === 'live_market' ? 'live' : 'delayed';
}

/** Premium harvest: IV rich vs the pre-spike baseline, with defined risk. */
function premiumHarvest(i: OpportunityInputs): Opportunity {
  const rule = ruleOf(i, 'short-premium-engage');
  const flat = ruleOf(i, 'short-premium-flat');
  const cut = ruleOf(i, 'hmm-crisis-cut');
  const cooldown = ruleOf(i, 'cusum-cooldown');
  const blockers: string[] = [];
  if (!i.chainAvailable) blockers.push('chain unavailable');
  if (i.earningsSessions !== null && i.earningsSessions <= 14) blockers.push(`earnings in ${i.earningsSessions} sessions`);
  if (flat?.state === 'met') blockers.push('no premium edge vs baseline');
  if (cut?.state === 'met') blockers.push('HMM crisis cut');
  if (cooldown?.state === 'met') blockers.push('CUSUM cooldown');

  const t = i.trap;
  const gexReady = Boolean(t && t.available === true && t.regime);
  const gexSummary = buildGexReadout(t);
  const gates = [...blockers];
  if (gexReady && t!.regime === 'short') {
    const flipCtx = fmtFlipContext(t!);
    const advCtx = fmtGexAdv(t!.gexPerAdv);
    const detail = [flipCtx, advCtx].filter(Boolean).join(', ');
    gates.push(`dealer short γ${detail ? ` (${detail})` : ''} amplifies realized swings — halve short-vol size`);
  }

  const needData = !rule || rule.state === 'insufficient-data' || !i.accruedSessions;
  const baseState = stateFor(rule, blockers, needData);
  const state: OpportunityState = baseState === 'ACTIONABLE' && gates.length > blockers.length ? 'WATCH' : baseState;
  const carry = i.carry;
  const measured = [rule?.measured ?? null, gexSummary].filter(Boolean).join(' · ') || null;
  const sizeExtra = [
    carry?.worstMonthBp2 != null ? `worst ${carry.worstMonthBp2.toFixed(0)} bp²` : null,
    gexReady ? (t!.regime === 'long' ? 'Long γ tailwind' : t!.regime === 'short' ? 'Short γ — cut size' : null) : null,
    gexReady ? fmtGexAdv(t!.gexPerAdv) : null,
  ].filter(Boolean).join(' · ') || null;
  const confidence: Opportunity['confidence'] = (carry?.n ?? 0) >= 30
    ? (gexReady && t!.regime === 'short' ? 'low' : gexReady && t!.regime === 'long' && isNum(t!.gexPerAdv) && t!.gexPerAdv >= 0.005 ? 'high' : 'medium')
    : 'low';

  return {
    id: 'premium-harvest',
    family: 'premium-harvest',
    title: 'Premium Harvest',
    thesis: 'Sell rich back-month variance only when IV exceeds the pre-spike baseline and the tail gates are quiet.',
    state,
    trigger: { metric: rule?.metric ?? 'IV30 − baseline', measured, rule: rule?.trigger ?? 'IV30 − variance baseline > +12 pts' },
    invalidation: `RV20 crosses IV30, HMM turns crisis, CUSUM cooldown fires, or earnings enters the window${gexReady && isNum(t?.flip) ? `; spot crosses gamma flip ${t!.flip.toFixed(2)} into short γ` : ''}.`,
    edge: {
      estimate: carry?.meanMonthlyBp2 !== null && carry?.meanMonthlyBp2 !== undefined ? `${carry.meanMonthlyBp2.toFixed(0)} bp² mean / ${((carry.hitRate ?? 0) * 100).toFixed(0)}% hit` : null,
      n: carry?.n ?? null,
      provenance: 'T3 short-vol carry study · 21-session overlapping windows · in-sample estimate',
    },
    size: size(i, 'carry worst month + max drawdown', sizeExtra),
    horizon: '30–45 DTE',
    evidence: { view: 'volatility', params: { vol: 't3' }, label: 'T3 · VRP & carry' },
    confidence,
    staleness: stalenessOf(i),
    gates,
  };
}

/** Long gamma: realized front-end vol must clear the cost-adjusted breakeven. */
function longGamma(i: OpportunityInputs): Opportunity {
  const rule = ruleOf(i, 'long-gamma');
  const exit = ruleOf(i, 'long-gamma-exit');
  const blockers: string[] = [];
  if (!i.chainAvailable) blockers.push('chain unavailable');
  if (i.earningsSessions !== null && i.earningsSessions <= 14) blockers.push(`earnings in ${i.earningsSessions} sessions — use the event card`);
  if (exit?.state === 'met') blockers.push('RV5/IV30 gap already closed');

  const t = i.trap;
  const gexReady = Boolean(t && t.available === true && t.regime);
  const gexSummary = buildGexReadout(t);
  const gates = [...blockers];
  if (gexReady && t!.regime === 'long') {
    const flipCtx = fmtFlipContext(t!);
    const advCtx = fmtGexAdv(t!.gexPerAdv);
    const detail = [flipCtx, advCtx].filter(Boolean).join(', ');
    gates.push(`dealer long γ${detail ? ` (${detail})` : ''} dampens moves — pinning headwind`);
  }

  const breakeven = i.iv30 !== null ? i.iv30 * BREAKEVEN_MULT : null;
  const baseMeasured = rule?.measured ?? (i.rv5 !== null ? `${pct(i.rv5)} vs ${pct(breakeven)}` : null);
  const measured = [baseMeasured, gexSummary].filter(Boolean).join(' · ') || null;
  const baseState = stateFor(rule, blockers, !rule || rule.state === 'insufficient-data');
  const state: OpportunityState = baseState === 'ACTIONABLE' && gates.length > blockers.length ? 'WATCH' : baseState;
  const sizeExtra = [
    i.iv30 !== null ? `1σ move ${pct(i.iv30 / Math.sqrt(252 / 21))}` : null,
    gexReady ? (t!.regime === 'short' ? 'Short γ amplification' : t!.regime === 'long' ? 'Long γ pin risk' : null) : null,
    gexReady ? fmtGexAdv(t!.gexPerAdv) : null,
  ].filter(Boolean).join(' · ') || null;
  const confidence: Opportunity['confidence'] = gexReady && t!.regime === 'long'
    ? 'low'
    : i.ivRankReady
      ? (gexReady && t!.regime === 'short' && isNum(t!.gexPerAdv) && Math.abs(t!.gexPerAdv) >= 0.005 ? 'high' : 'medium')
      : 'low';

  return {
    id: 'long-gamma',
    family: 'long-gamma',
    title: 'Long Front-End Gamma',
    thesis: 'Own realized movement when RV5 clears the delta-hedged straddle breakeven.',
    state,
    trigger: { metric: rule?.metric ?? `RV5 vs ${BREAKEVEN_MULT}×IV30`, measured, rule: rule?.trigger ?? `RV5 > ${BREAKEVEN_MULT}×IV30` },
    invalidation: `RV5 falls under the close threshold or IV30 collapses — exit the gap, do not ride to expiry${gexReady && isNum(t?.flip) ? `; reclaiming gamma flip ${t!.flip.toFixed(2)} restores long γ` : ''}.`,
    edge: {
      estimate: i.rv5 !== null && breakeven !== null ? pts(i.rv5 - breakeven) : null,
      n: i.accruedSessions || null,
      provenance: 'T8 pre-registered rule · breakeven is an uncalibrated cost heuristic',
    },
    size: size(i, 'expected-move cone', sizeExtra),
    horizon: '7–14 DTE',
    evidence: { view: 'volatility', params: { vol: 't8' }, label: 'T8 · decision rules' },
    confidence,
    staleness: stalenessOf(i),
    gates,
  };
}

/** Directional trap: only a base-rate-confirmed wall break earns an arrow. */
function directionalTrap(i: OpportunityInputs): Opportunity {
  const t = i.trap;
  const trapUnavailable = !t || t.available === false;
  const gexReady = Boolean(t && t.available === true);
  const blockers: string[] = [];
  if (trapUnavailable) blockers.push('trap snapshot unavailable');
  if (!trapUnavailable && t.state !== 'triggered') blockers.push(`trap state ${t.state ?? 'unknown'} — not triggered`);
  if (!trapUnavailable && t.direction === 'none') blockers.push('no directional lean');
  if (!trapUnavailable && t.backtestVerdict === 'insufficient_history') blockers.push('event study insufficient — no base-rate arrow');
  if (!trapUnavailable && t.backtestVerdict === 'no_reliable_follow_through') blockers.push('wall failed base-rate gate');
  if (gexReady && t!.regime === 'long') blockers.push('dealer long γ stabilizes price — opposes wall break');

  const gates = [...blockers];
  if (gexReady && blockers.length === 0) {
    if (isNum(t!.gexPerAdv) && Math.abs(t!.gexPerAdv) < 0.001) {
      gates.push('GEX/ADV < 0.1% — thin dealer footprint vs daily liquidity');
    }
    if (isNum(t!.flipDistSigma) && t!.flipDistSigma < 0.25) {
      gates.push('spot within 0.25σ of gamma flip — whip risk across regime boundary');
    }
  }

  const needData = trapUnavailable || t.backtestN === null;
  const baseState = stateFor({ id: 'trap', state: blockers.length === 0 && !needData ? 'met' : 'not-met' } as SignalRuleResult, blockers, needData);
  const state: OpportunityState = baseState === 'ACTIONABLE' && gates.length > blockers.length ? 'WATCH' : baseState;
  const wall = t && !trapUnavailable ? (t.direction === 'up' ? t.callWall ?? null : t.putWall ?? null) : null;
  const gexSummary = gexReady ? buildGexReadout(t) : null;
  const measured = t && !trapUnavailable
    ? [`${t.state.toUpperCase()} · ${t.direction}`, gexSummary, `score ${t.score ?? '—'}`, `n=${t.backtestN ?? '—'}`].filter(Boolean).join(' · ')
    : null;
  const flipCtx = gexReady && t ? fmtFlipContext(t) : null;
  const sizeExtra = [
    t && !trapUnavailable && isNum(t.score) ? `score ${t.score}/100` : null,
    gexReady && t ? fmtGexAdv(t.gexPerAdv) : null,
  ].filter(Boolean).join(' · ') || null;
  const confidence: Opportunity['confidence'] = (t?.backtestN ?? 0) >= 20
    ? (gexReady && t!.regime === 'short' && isNum(t!.gexPerAdv) && Math.abs(t!.gexPerAdv) >= 0.005 ? 'high' : 'medium')
    : 'low';

  return {
    id: 'directional-trap',
    family: 'directional-trap',
    title: 'Dealer Wall Ride',
    thesis: 'Follow a triggered wall break only after past triggers beat their base rate.',
    state,
    trigger: { metric: 'trap state · direction · GEX · event study', measured, rule: 'state=triggered AND short γ AND backtest verdict=above_base_rate' },
    invalidation: `gamma flip${isNum(t?.flip) ? ` ${t!.flip.toFixed(2)}` : ''}${flipCtx ? ` (${flipCtx})` : ''}; close back through the wall.`,
    edge: {
      estimate: wall !== null && wall !== undefined && i.spot && i.iv30
        ? `${(Math.abs(wall / i.spot - 1) / (i.iv30 * Math.sqrt(21 / 252))).toFixed(2)}σ to wall`
        : null,
      n: t && !trapUnavailable ? t.backtestN ?? null : null,
      provenance: 'GEX / T7 event study · GEX convention assumed, not observed · OI end-of-day',
    },
    size: size(i, 'wall distance in σ', sizeExtra),
    horizon: '2–10 sessions',
    evidence: { view: 'fno', params: { fno: 'gex' }, label: 'GEX · dealer positioning' },
    confidence,
    staleness: stalenessOf(i),
    gates,
  };
}

/** Event straddle: a priced move that is cheap versus the realized-move base rate. */
function eventStraddle(i: OpportunityInputs): Opportunity {
  const e = i.expectedMove;
  const blockers: string[] = [];
  if (!i.fnoAvailable) blockers.push('F&O panel unavailable');
  if (i.earningsSessions === null) blockers.push('no earnings date in window');
  else if (i.earningsSessions > 45) blockers.push(`earnings in ${i.earningsSessions} sessions — too far`);
  const gates = [...blockers];
  if (i.earningsIsEstimate === true) gates.push('earnings date is an estimate — verify before acting');
  const needData = !e || e.verdict === null || e.impliedPct === null;
  const met = e?.verdict === 'CHEAP' && gates.length === 0;
  return {
    id: 'event-straddle',
    family: 'event-straddle',
    title: 'Event Straddle',
    thesis: 'Buy the priced move when the straddle is cheap against the realized-move distribution.',
    state: needData ? 'INSUFFICIENT_DATA' : blockers.length > 0 ? 'BLOCKED' : met ? 'ACTIONABLE' : 'WATCH',

    trigger: { metric: 'implied vs realized |move|', measured: e ? `${pct(e.impliedPct)} · ${e.verdict ?? '—'} · ${e.percentile !== null ? `${e.percentile.toFixed(0)}th pctile` : '—'}` : null, rule: 'verdict=CHEAP and earnings within 45 sessions' },
    invalidation: 'No move through the straddle breakeven within the selected horizon; exit at 50% of expected move.',
    edge: { estimate: e?.percentile !== null && e?.percentile !== undefined ? `${(100 - e.percentile).toFixed(0)}th pctile cheapness` : null, n: e?.n ?? null, provenance: 'F&O expected-vs-realized study · historical |log move| base rate' },
    size: size(i, 'straddle breakeven', e?.impliedPct !== null && e?.impliedPct !== undefined ? `priced move ${pct(e.impliedPct)}` : null),
    horizon: e?.dte ? `${e.dte} DTE` : 'nearest expiry',
    evidence: { view: 'fno', params: { fno: 'expectedmove' }, label: 'F6 · expected vs realized' },
    confidence: (e?.n ?? 0) >= 100 ? 'medium' : 'low',
    staleness: stalenessOf(i),
    gates,
  };
}

/** Skew / tail relative value from the 25Δ tiles plus VIX-family context. */
function skewTail(i: OpportunityInputs): Opportunity {
  const crash = ruleOf(i, 'skew-crash');
  const callWing = ruleOf(i, 'rr25-call-wing');
  const primary = crash?.state === 'met' ? crash : callWing;
  const blockers: string[] = [];
  if (!i.chainAvailable) blockers.push('chain unavailable');
  if (i.earningsSessions !== null && i.earningsSessions <= 14) blockers.push('earnings inside skew window');
  const needData = !primary || primary.state === 'insufficient-data' || !i.frontSkew;
  return {
    id: 'skew-tail',
    family: 'skew-tail',
    title: 'Tail / Skew Relative Value',
    thesis: 'Trade the asymmetry when crash skew or call-wing richness is extreme versus its own history.',
    state: stateFor(primary, blockers, needData),
    trigger: { metric: primary?.metric ?? '25Δ RR / VVIX', measured: primary?.measured ?? (i.frontSkew ? `RR ${pts(i.frontSkew.rr25)} · BF ${pts(i.frontSkew.bf25)}` : null), rule: primary?.trigger ?? 'pre-registered skew rule' },
    invalidation: 'RR/BF reverts to its accrued median; VVIX and SKEW percentile context must agree.',
    edge: { estimate: i.skewPercentile !== null ? `${i.skewPercentile.toFixed(0)}th SKEW pctile · VVIX ${i.vvixPercentile !== null ? `${i.vvixPercentile.toFixed(0)}th` : '—'}` : null, n: i.accruedSessions || null, provenance: 'T1 skew tiles + T6 VIX-family percentiles' },
    size: size(i, 'defined-risk wings only', i.vvix !== null ? `VVIX ${i.vvix.toFixed(1)}` : null),
    horizon: '30–90 DTE',
    evidence: { view: 'volatility', params: { vol: 't1' }, label: 'T1 · skew & VIX context' },
    confidence: i.ivRankReady ? 'medium' : 'low',
    staleness: stalenessOf(i),
    gates: blockers,
  };
}

/** Cross-sectional rank of IV30 − RV20 across the watchlist. */
function crossSectional(i: OpportunityInputs): Opportunity {
  const rows = i.crossSectional
    .filter((r) => r.iv30 !== null && r.rv20 !== null)
    .map((r) => ({ ...r, gap: (r.iv30! - r.rv20!) * 100 }))
    .sort((a, b) => b.gap - a.gap);
  const needData = rows.length < 3;
  const longLeg = rows[rows.length - 1];
  const shortLeg = rows[0];
  const spread = longLeg && shortLeg ? shortLeg.gap - longLeg.gap : null;
  const blockers: string[] = [];
  if (!i.chainAvailable) blockers.push('watchlist scan unavailable');
  return {
    id: 'cross-sectional',
    family: 'cross-sectional',
    title: 'Cross-Sectional Vol RV',
    thesis: 'Fade the richest IV−RV gap against the cheapest one in the watchlist.',
    state: needData ? 'INSUFFICIENT_DATA' : blockers.length > 0 ? 'BLOCKED' : spread !== null && spread > 5 ? 'ACTIONABLE' : 'WATCH',
    trigger: { metric: 'IV30 − RV20 by name', measured: rows.length ? `${rows.map((r) => `${r.symbol} ${r.gap.toFixed(1)}`).join(' · ')}` : null, rule: 'rank gap > 5 vol pts with ≥3 names' },
    invalidation: 'Pairwise gap compresses below 5 vol pts; single-day snapshot has no IV rank yet.',
    edge: { estimate: spread !== null ? `${spread.toFixed(1)} vol pts spread` : null, n: rows.length || null, provenance: 'combined watchlist scan · single-day IV/RV snapshot, not accrued IV rank' },
    size: size(i, 'pairwise, market-neutral sizing', longLeg && shortLeg ? `long ${longLeg.symbol} / short ${shortLeg.symbol}` : null),
    horizon: '20 sessions',
    evidence: { view: 'volatility', params: { vol: 't6' }, label: 'T6 · cross-asset matrix' },
    confidence: 'low',
    staleness: stalenessOf(i),
    gates: blockers,
  };
}

/** Assemble all six cards. Order is the rank order on the board. */
export function evaluateOpportunities(inputs: OpportunityInputs): Opportunity[] {
  return [
    premiumHarvest(inputs),
    longGamma(inputs),
    directionalTrap(inputs),
    eventStraddle(inputs),
    skewTail(inputs),
    crossSectional(inputs),
  ];
}

export function rankOpportunities(ops: Opportunity[]): Opportunity[] {
  const weight: Record<OpportunityState, number> = { ACTIONABLE: 0, WATCH: 1, BLOCKED: 2, INSUFFICIENT_DATA: 3 };
  const conf = { high: 0, medium: 1, low: 2 } as const;
  return [...ops].sort((a, b) =>
    weight[a.state] - weight[b.state] || conf[a.confidence] - conf[b.confidence] || a.id.localeCompare(b.id)
  );
}

/** Rule state helper re-exported so the UI can render chips identically. */
export function isMet(state: SignalState | undefined): boolean {
  return state === 'met';
}
