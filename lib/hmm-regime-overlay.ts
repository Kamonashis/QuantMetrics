// Pure TypeScript helpers & types for strictly causal HMM volatility regime
// overlays and per-regime out-of-sample forecast performance breakdowns.
// Zero React/DOM dependencies — relative imports only for node --test compatibility.

import { HMM_REGIME_LABELS, type HmmRegimeLabel } from './hmm';

export type HmmStateIdx = 0 | 1 | 2;

export interface HmmRegimeVisualToken {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  /** Primary stroke / marker color (high contrast on dark & light canvas) */
  color: string;
  /** RGB triplet for opacity-tuned SVG background bands */
  rgb: string;
  /** Tailwind text class (paired dark/light) */
  textClass: string;
  /** Tailwind swatch dot/bar fill class */
  bgClass: string;
}

/**
 * Harmonized 3-state HMM regime palette matching QuantMetrics' RegimeTimeline
 * and Obsidian Dark / Paper Light dual-theme design system.
 */
export const HMM_REGIME_TOKENS: readonly HmmRegimeVisualToken[] = [
  {
    state: 0,
    label: HMM_REGIME_LABELS[0], // Calm
    color: '#3FB27F',
    rgb: '63, 178, 127',
    textClass: 'text-emerald-700 dark:text-emerald-400',
    bgClass: 'bg-[#3FB27F]',
  },
  {
    state: 1,
    label: HMM_REGIME_LABELS[1], // Watchful
    color: '#DEC084',
    rgb: '222, 192, 132',
    textClass: 'text-gold-700 dark:text-gold-300',
    bgClass: 'bg-[#DEC084]',
  },
  {
    state: 2,
    label: HMM_REGIME_LABELS[2], // Crisis
    color: '#E0655F',
    rgb: '224, 101, 95',
    textClass: 'text-rose-600 dark:text-rose-400',
    bgClass: 'bg-[#E0655F]',
  },
] as const;

export function getHmmRegimeToken(state: number | null | undefined): HmmRegimeVisualToken {
  if (state === 1) return HMM_REGIME_TOKENS[1];
  if (state === 2) return HMM_REGIME_TOKENS[2];
  return HMM_REGIME_TOKENS[0];
}

export interface RegimeBandSpan {
  startIdx: number;
  endIdx: number;
  state: HmmStateIdx;
}

/**
 * Extracts dominant causal HMM state (0=Calm, 1=Watchful, 2=Crisis) and its
 * posterior probability per time step from forward-filtered posteriors [T][3].
 */
export function extractDominantHmmSeries(
  filteredPosteriors: number[][] | null | undefined
): { states: HmmStateIdx[]; probs: number[] } {
  if (!filteredPosteriors || filteredPosteriors.length === 0) {
    return { states: [], probs: [] };
  }
  const n = filteredPosteriors.length;
  const states: HmmStateIdx[] = new Array(n);
  const probs: number[] = new Array(n);
  for (let t = 0; t < n; t++) {
    const row = filteredPosteriors[t];
    const p0 = row?.[0] ?? 1 / 3;
    const p1 = row?.[1] ?? 1 / 3;
    const p2 = row?.[2] ?? 1 / 3;
    if (p2 >= p1 && p2 >= p0) {
      states[t] = 2;
      probs[t] = p2;
    } else if (p1 >= p0 && p1 >= p2) {
      states[t] = 1;
      probs[t] = p1;
    } else {
      states[t] = 0;
      probs[t] = p0;
    }
  }
  return { states, probs };
}

/**
 * Run-length merges contiguous sessions sharing the same dominant HMM regime
 * into [startIdx, endIdx] spans (inclusive). Eliminates SVG anti-aliasing seams
 * and minimizes DOM node counts for smooth 60fps hover scrubbing.
 */
export function buildRegimeBandSpans(
  states: readonly (HmmStateIdx | null | undefined)[]
): RegimeBandSpan[] {
  const spans: RegimeBandSpan[] = [];
  let current: RegimeBandSpan | null = null;

  for (let i = 0; i < states.length; i++) {
    const st = states[i];
    if (st !== 0 && st !== 1 && st !== 2) {
      if (current) {
        spans.push(current);
        current = null;
      }
      continue;
    }
    if (current && current.state === st) {
      current.endIdx = i;
    } else {
      if (current) spans.push(current);
      current = { startIdx: i, endIdx: i, state: st };
    }
  }
  if (current) spans.push(current);
  return spans;
}

// ---------------------------------------------------------------------------
// PER-REGIME OUT-OF-SAMPLE PERFORMANCE BREAKDOWN TYPES
// ---------------------------------------------------------------------------

export interface VolRegimePerfRow {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  count: number;
  meanQlike: number | null;
  meanMseLogVar: number | null;
  meanPredVol: number | null;
  meanRealizedVol: number | null;
}

export interface RiskRegimePerfRow {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  count: number;
  exceedance95Count: number;
  exceedance95RatePct: number | null;
  exceedance99Count: number;
  exceedance99RatePct: number | null;
  meanVar99: number | null;
  meanRealizedLoss: number | null;
}

export interface RegRegimePerfRow {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  count: number;
  oosRmse: number | null;
  olsOosRmse: number | null;
  hitRatePct: number | null;
  meanPredReturn: number | null;
  meanActualReturn: number | null;
}

export interface CorrRegimePerfRow {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  count: number;
  oosMae: number | null;
  oosRmse: number | null;
  meanPredCorr: number | null;
  meanActualCorr: number | null;
}

export interface BacktestRegimePerfRow {
  state: HmmStateIdx;
  label: HmmRegimeLabel;
  count: number;
  directionalHitRatePct: number | null;
  rmseDollar: number | null;
  meanNetReturnPct: number | null;
  cumNetReturnPct: number | null;
}
