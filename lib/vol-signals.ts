// =============================================================
// QUANTMETRICS — VOL DECISION LAYER
// =============================================================
// Pure, deterministic math behind the T8 "Decision" tab.
// No React, no I/O, no Date.now() — every function takes explicit inputs.
// All volatilities are annualized decimals (0.286 = 28.6%), returns are simple/log
// decimals, and horizons are in trading days (252/yr) unless noted.

// -------------------------------------------------------------
// CONFIG — tunable heuristics, never magic literals in the UI
// -------------------------------------------------------------

/** Delta-hedged long-straddle breakeven ≈ BREAKEVEN_MULT × IV. Heuristic from costs + hedge slippage; calibrate via backtest once IV accrual matures (~60 sessions). */
export const BREAKEVEN_MULT = 1.25;
/** Exit long-gamma when RV5 falls below this (edge is the RV–IV gap, not the calendar). */
export const RV5_CLOSE_THRESHOLD = 0.30;
/** Exit long-gamma when IV30 falls below this (now paying above realized decay). */
export const IV30_RICH_EXIT = 0.26;
/** Short-premium 2–12-pt edge band: IV30 − variance baseline, in decimal vol points. */
export const SHORT_PREMIUM_RICH_POINTS = 0.12;
export const SHORT_PREMIUM_FLAT_POINTS = 0.05;
/** Rolling 60d skew below zero ⇒ the realized jumps have been down, not up. */
export const SKEW_CRASH_THRESHOLD = 0;
/** Lo–MacKinlay VR z-stat below this ⇒ mean-reversion regime (condors, not straddles). */
export const VR_Z_MEAN_REVERSION = -1.0;
/** ADF on decomposition residuals above this p ⇒ decomposition unstable (drop seasonal tilt). */
export const ADF_RESIDUAL_UNSTABLE = 0.05;
/** No new short premium within this many sessions of a CUSUM-SQ break. */
export const CUSUM_COOLDOWN_SESSIONS = 5;
/** HMM posterior of the crisis state above this ⇒ halve short gamma. */
export const HMM_CRISIS_CUT = 0.25;
/** 25Δ RR (call IV − put IV) above this ⇒ call wings are priced; skip the upside asymmetry trade. */
export const RR25_CALL_RICH = 0.005;

// -------------------------------------------------------------
// ROLLING Z-SCORE
// -------------------------------------------------------------

/** z of the latest value vs the trailing window of `window` values (sample stdev). Null until the window fills; null when stdev = 0. */
export function rollingZ(values: (number | null)[], window: number): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i < window - 1) {
      out.push(null);
      continue;
    }
    const slice = values.slice(i - window + 1, i + 1);
    if (slice.some((v) => v === null || !isFinite(v as number))) {
      out.push(null);
      continue;
    }
    const nums = slice as number[];
    const n = nums.length;
    let s = 0;
    for (const v of nums) s += v;
    const m = s / n;
    let ss = 0;
    for (const v of nums) ss += (v - m) * (v - m);
    const sd = Math.sqrt(ss / (n - 1));
    out.push(Number.isFinite(sd) && sd > 1e-12 ? ((values[i] as number) - m) / sd : null);
  }
  return out;
}

// -------------------------------------------------------------
// VARIANCE DECOMPOSITION — forward realized vol & pre-spike baseline
// -------------------------------------------------------------

/**
 * Annualized forward realized vol implied by two trailing windows.
 * Total variance over N days ≈ N · RV(N)² / 252, so the segment between
 * tShort and tLong contributes (tLong·rvLong² − tShort·rvShort²)/252.
 * Returns null when the forward segment variance ≤ 0 (nothing left to extract).
 */
export function forwardRealizedVol(rvShort: number, rvLong: number, tShort: number, tLong: number): number | null {
  if (!isFinite(rvShort) || !isFinite(rvLong) || tLong <= tShort || rvShort < 0 || rvLong < 0) return null;
  const segVar = (tLong * rvLong * rvLong - tShort * rvShort * rvShort) / 252;
  const days = tLong - tShort;
  if (segVar <= 0 || days <= 0) return null;
  return Math.sqrt((segVar / days) * 252);
}

/** Pre-spike baseline: forward realized vol of the window outside the most recent `tRecent` days. E.g. (10, 30) isolates days −30…−10. */
export function varianceBaseline(rvRecent: number, rvLong: number, tRecent: number, tLong: number): number | null {
  return forwardRealizedVol(rvRecent, rvLong, tRecent, tLong);
}

// -------------------------------------------------------------
// IMPLIED FORWARD VOL — calendar decomposition of the term structure
// -------------------------------------------------------------

export interface TermPoint {
  dte: number;
  t: number; // years to expiry
  iv: number | null;
}

/**
 * Volatility of the [fromDte, toDte] calendar bucket, from w = σ²T differences:
 *   σ_fwd = sqrt( (w₂ − w₁) / (t₂ − t₁) ).
 * Picks the term point nearest each target DTE. Null when there are fewer than 2
 * usable points, the bucket is inverted, or forward variance comes out negative —
 * never interpolates a number out of thin air.
 */
export function forwardImpliedVol(termPoints: TermPoint[], fromDte: number, toDte: number): number | null {
  const usable = termPoints.filter((p) => p.iv !== null && isFinite(p.t) && p.t > 0 && isFinite(p.iv as number) && (p.iv as number) > 0);
  if (usable.length < 2) return null;
  const nearest = (target: number): TermPoint => {
    let best = usable[0];
    let bestDist = Math.abs(usable[0].dte - target);
    for (const p of usable) {
      const d = Math.abs(p.dte - target);
      if (d < bestDist) {
        best = p;
        bestDist = d;
      }
    }
    return best;
  };
  const p1 = nearest(fromDte);
  const p2 = nearest(toDte);
  if (p2.t <= p1.t) return null;
  const w1 = (p1.iv as number) * (p1.iv as number) * p1.t;
  const w2 = (p2.iv as number) * (p2.iv as number) * p2.t;
  const fwdVar = (w2 - w1) / (p2.t - p1.t);
  if (fwdVar <= 0) return null;
  return Math.sqrt(fwdVar);
}

// -------------------------------------------------------------
// RV TRIANGLE — shape of the realized-vol term structure
// -------------------------------------------------------------

export type RvTriangleShape = 'backwardation' | 'contango' | 'humped' | 'mixed';

export interface RvTriangle {
  shape: RvTriangleShape;
  /** RV10 − RV20, in decimal vol points (positive = front is rich). */
  spread1020: number;
  /** RV20 − RV30, in decimal vol points. */
  spread2030: number;
}

/** Classify the RV ladder: non-increasing = backwardation (front-rich spike), non-decreasing = contango (building), interior max = humped, otherwise mixed. */
export function rvTriangle(rv5: number | null, rv10: number | null, rv20: number | null, rv30: number | null): RvTriangle | null {
  if (rv10 === null || rv20 === null || rv30 === null) return null;
  const seq = [rv5, rv10, rv20, rv30].filter((v): v is number => v !== null && isFinite(v));
  const n = seq.length;
  let nonIncreasing = true;
  let nonDecreasing = true;
  let maxIdx = 0;
  for (let i = 1; i < n; i++) {
    if (seq[i - 1] < seq[i]) nonIncreasing = false;
    if (seq[i - 1] > seq[i]) nonDecreasing = false;
    if (seq[i] > seq[maxIdx]) maxIdx = i;
  }
  let shape: RvTriangleShape;
  if (nonIncreasing) shape = 'backwardation';
  else if (nonDecreasing) shape = 'contango';
  else if (maxIdx > 0 && maxIdx < n - 1) shape = 'humped';
  else shape = 'mixed';
  return { shape, spread1020: rv10 - rv20, spread2030: rv20 - rv30 };
}

/** 1.25× IV breakeven — the vol level a delta-hedged long straddle needs to clear costs. */
export function breakevenVol(iv30: number, mult: number = BREAKEVEN_MULT): number {
  return iv30 * mult;
}

// -------------------------------------------------------------
// SIGNAL RULES — the pre-registered table, encoded as data
// -------------------------------------------------------------

export type SignalState = 'met' | 'not-met' | 'insufficient-data';

export interface SignalRuleResult {
  id: string;
  label: string;
  /** The metric that drives the rule, e.g. 'RV5'. */
  metric: string;
  /** Formatted measured value for display, or null when data is missing. */
  measured: string | null;
  /** Human-readable trigger, e.g. '> 1.25 × IV30 (35.8%)'. */
  trigger: string;
  state: SignalState;
  /** What the pre-registered rule says to do when it fires. */
  action: string;
}

export interface VolSignalInputs {
  iv30: number | null;
  rv5: number | null;
  rv10: number | null;
  rv20: number | null;
  rv30: number | null;
  /** Latest rolling 60d skew (Lo 2002), or null when unavailable. */
  skew60: number | null;
  /** Skew95 band, when available (informational only). */
  skew60LowerCi: number | null;
  skew60UpperCi: number | null;
  /** Minimum Lo–MacKinlay VR z-stat across horizons (most-negative = strongest mean-reversion case). */
  vrMinZ: number | null;
  /** p-value of ADF on the decomposition residuals. */
  adfResidualP: number | null;
  /** Sessions since the most recent in-sample CUSUM-SQ break; null when none. */
  cusumBreakSessionsAgo: number | null;
  /** Posterior of the crisis state at the final bar (0..1); null when HMM unfit. */
  hmmCrisisProb: number | null;
  /** Dominant HMM state index at the final bar (0 = calmest); null when unfit. */
  hmmDominantState: number | null;
  /** 25Δ risk reversal (call IV − put IV), decimal; null when unfit. */
  rr25Front: number | null;
}

const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;
const pts = (v: number): string => `${(v * 100).toFixed(1)} pts`;
const fmtZ = (v: number): string => v.toFixed(2);

/**
 * Evaluate the pre-registered vol signal table. Pure: every rule degrades to
 * 'insufficient-data' when its inputs are missing — never defaults to 'met'.
 */
export function evaluateVolSignals(inp: VolSignalInputs): SignalRuleResult[] {
  const rules: SignalRuleResult[] = [];
  const baseline = (inp.rv10 !== null && inp.rv30 !== null)
    ? varianceBaseline(inp.rv10, inp.rv30, 10, 30)
    : null;
  const breakeven = inp.iv30 !== null ? breakevenVol(inp.iv30) : null;

  // 1 — Long-gamma engagement: RV5 clears the cost-adjusted breakeven
  {
    const ok = inp.rv5 !== null && breakeven !== null;
    const met = ok && (inp.rv5 as number) > (breakeven as number);
    rules.push({
      id: 'long-gamma',
      label: 'Long front-end gamma',
      metric: 'RV5 vs 1.25×IV30',
      measured: ok ? `${pct(inp.rv5 as number)} vs ${pct(breakeven as number)}` : null,
      trigger: `RV5 > 1.25 × IV30 (${breakeven !== null ? pct(breakeven) : '—'})`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Long 7–14 DTE ATM straddle, delta-hedge at ~0.75σ band. Size for ±$24 at 10 DTE (~$2.4k/lot).',
    });
  }

  // 2 — Long-gamma exit: the RV–IV gap closed
  {
    const ok = inp.rv5 !== null || inp.iv30 !== null;
    const rvMet = inp.rv5 !== null && inp.rv5 < RV5_CLOSE_THRESHOLD;
    const ivMet = inp.iv30 !== null && inp.iv30 < IV30_RICH_EXIT;
    const met = ok && (rvMet || ivMet);
    rules.push({
      id: 'long-gamma-exit',
      label: 'Close long gamma',
      metric: 'RV5 / IV30',
      measured: ok
        ? `RV5 ${inp.rv5 !== null ? pct(inp.rv5) : '—'} · IV30 ${inp.iv30 !== null ? pct(inp.iv30) : '—'}`
        : null,
      trigger: `RV5 < ${RV5_CLOSE_THRESHOLD * 100}% or IV30 < ${IV30_RICH_EXIT * 100}%`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Roll or close the straddle. Edge was the RV–IV gap, not the calendar. Exit fully; do not let it ride to expiry.',
    });
  }

  // 3a — Short-premium engagement: IV rich vs the pre-spike baseline
  {
    const ok = inp.iv30 !== null && baseline !== null;
    const spread = ok ? (inp.iv30 as number) - (baseline as number) : null;
    const met = spread !== null && spread > SHORT_PREMIUM_RICH_POINTS;
    rules.push({
      id: 'short-premium-engage',
      label: 'Short back-month strangle',
      metric: 'IV30 − baseline',
      measured: spread !== null ? pts(spread) : null,
      trigger: `IV30 − baseline > +${SHORT_PREMIUM_RICH_POINTS * 100} pts (IV30 > ${baseline !== null ? pct((baseline as number) + SHORT_PREMIUM_RICH_POINTS) : '—'})`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: '30–45 DTE ~16Δ strangle, ALWAYS defined-risk 5Δ wings. Half size if HMM ≠ calm. Never into earnings.',
    });
  }

  // 3b — Short-premium flat: no edge, vol merely looks rich vs history
  {
    const ok = inp.iv30 !== null && baseline !== null;
    const spread = ok ? (inp.iv30 as number) - (baseline as number) : null;
    const met = spread !== null && spread < SHORT_PREMIUM_FLAT_POINTS;
    rules.push({
      id: 'short-premium-flat',
      label: 'Short premium: stand aside',
      metric: 'IV30 − baseline',
      measured: spread !== null ? pts(spread) : null,
      trigger: `IV30 − baseline < +${SHORT_PREMIUM_FLAT_POINTS * 100} pts`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'No short-vol edge vs the 28–32% forward expectation. Stay flat; do not sell the 250d averaging artifact.',
    });
  }

  // 4a — HMM crisis posterior: cut short gamma
  {
    const ok = inp.hmmCrisisProb !== null;
    const met = ok && (inp.hmmCrisisProb as number) > HMM_CRISIS_CUT;
    rules.push({
      id: 'hmm-crisis-cut',
      label: 'HMM crisis risk-off',
      metric: 'P(crisis | data)',
      measured: ok ? pct(inp.hmmCrisisProb as number) : null,
      trigger: `Crisis posterior > ${HMM_CRISIS_CUT * 100}%`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Cut short gamma 50%. Buy back short strikes. Watch the transition matrix — a coin-flip calm→crisis state is not a hold signal.',
    });
  }

  // 4b — HMM non-calm dominant: half size on anything short
  {
    const ok = inp.hmmDominantState !== null;
    const met = ok && (inp.hmmDominantState as number) > 0;
    rules.push({
      id: 'hmm-half-size',
      label: 'HMM non-calm: half size',
      metric: 'Dominant state',
      measured: ok ? (inp.hmmDominantState === 0 ? 'Calm' : `State ${inp.hmmDominantState}`) : null,
      trigger: 'Dominant state ≠ calm',
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Half size on any short premium. Watchful is a coin flip to crisis — size for the transition, not the label.',
    });
  }

  // 5 — Fresh CUSUM-SQ break: stand aside on short vol
  {
    const ok = true; // null sessionsAgo = no break detected in-window, which is actionable information
    const met = inp.cusumBreakSessionsAgo !== null && inp.cusumBreakSessionsAgo <= CUSUM_COOLDOWN_SESSIONS;
    rules.push({
      id: 'cusum-cooldown',
      label: 'CUSUM break cooldown',
      metric: 'Sessions since break',
      measured: inp.cusumBreakSessionsAgo !== null ? `${inp.cusumBreakSessionsAgo} td` : 'none in window',
      trigger: `Break ≤ ${CUSUM_COOLDOWN_SESSIONS} sessions ago`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'No new short vol for 5 sessions. Re-fit every estimate on post-break data only.',
    });
  }

  // 6 — 60d skew below zero: downside jumps dominate
  {
    const ok = inp.skew60 !== null;
    const met = ok && (inp.skew60 as number) < SKEW_CRASH_THRESHOLD;
    rules.push({
      id: 'skew-crash',
      label: 'Skew flip: crash hazard',
      metric: 'Rolling 60d skew',
      measured: ok ? inp.skew60!.toFixed(2) : null,
      trigger: '60d skew < 0',
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Buy put wing (defined-risk verticals). Reduce naked/short delta. Positive-skew upside asymmetry is OFF the table.',
    });
  }

  // 7 — Variance ratio: flip from trend to mean-reversion
  {
    const ok = inp.vrMinZ !== null;
    const met = ok && (inp.vrMinZ as number) < VR_Z_MEAN_REVERSION;
    rules.push({
      id: 'vr-mean-reversion',
      label: 'VR regime flip',
      metric: 'Min VR z-stat',
      measured: ok ? fmtZ(inp.vrMinZ as number) : null,
      trigger: `min VR z < ${VR_Z_MEAN_REVERSION.toFixed(1)}`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Iron condors / short premium structures, not long straddles. Fade extremes; do not chase gamma.',
    });
  }

  // 8 — Decomposition residuals lack mean-reversion: drop seasonal tilt
  {
    const ok = inp.adfResidualP !== null;
    const met = ok && (inp.adfResidualP as number) > ADF_RESIDUAL_UNSTABLE;
    rules.push({
      id: 'adf-unstable',
      label: 'Decomposition unstable',
      metric: 'ADF p (residuals)',
      measured: ok ? (inp.adfResidualP as number).toFixed(4) : null,
      trigger: `ADF p > ${ADF_RESIDUAL_UNSTABLE}`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Drop the seasonal tilt. Trend-only. Do not fade residual-band excursions with defined-risk verticals.',
    });
  }

  // 9 — RR25: is the upside asymmetry already priced?
  {
    const ok = inp.rr25Front !== null;
    const met = ok && (inp.rr25Front as number) > RR25_CALL_RICH;
    rules.push({
      id: 'rr25-call-wing',
      label: 'Upside call wing priced?',
      metric: 'RR25 (front slice)',
      measured: ok ? pts(inp.rr25Front as number) : null,
      trigger: `RR25 > +${(RR25_CALL_RICH * 100).toFixed(1)} pts ⇒ calls richer than puts`,
      state: ok ? (met ? 'met' : 'not-met') : 'insufficient-data',
      action: 'Calls richer than puts — the +skew asymmetry is priced. Skip the long-call-wing trade. If RR25 ≤ 0, wings are not disqualified (size 10–15 Δ, defined-risk).',
    });
  }

  return rules;
}
