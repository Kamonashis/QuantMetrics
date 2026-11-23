// Shared types for the cross-sectional factor library. Pure TS — no React, no DOM.

export type FactorFamily =
  | 'momentum'
  | 'reversal'
  | 'low_volatility'
  | 'low_beta'
  | 'downside_risk'
  | 'liquidity'
  | 'volatility_structure'
  | 'value'
  | 'quality'
  | 'investment'
  | 'size'
  | 'analyst'
  | 'options';

export type FactorDataRequirement = 'price' | 'fundamentals' | 'analyst' | 'options';

export interface FactorDefinition {
  /** Namespaced id, e.g. `price:MOM_12_1`. Used as the regression column key. */
  id: string;
  family: FactorFamily;
  label: string;
  /** Display formula, written in the same notation as the factor research brief. */
  formula: string;
  rationale: string;
  /** +1 when a higher raw value is attractive, -1 when a lower raw value is attractive. */
  direction: 1 | -1;
  requires: FactorDataRequirement;
  /** Minimum observations needed for a meaningful raw value. */
  minObservations: number;
}

export interface FactorSeries {
  id: string;
  /** One value per aligned date; null where data is insufficient. */
  values: (number | null)[];
}

export interface FactorObservation {
  symbol: string;
  date: string;
  value: number | null;
  /** Date at which the value legitimately becomes known (reporting lag for fundamentals). */
  availableAt: string;
  /** 0..1 share of required inputs that were present. */
  coverage: number;
}

export interface FactorPanelRow {
  date: string;
  symbol: string;
  sector: string;
  raw: Record<string, number | null>;
  z: Record<string, number | null>;
  composite: number | null;
  rank: number | null;
  decile: number | null;
  forwardReturn: number | null;
}

export interface FactorIcResult {
  id: string;
  label: string;
  observations: number;
  meanIc: number | null;
  rankIc: number | null;
  icStd: number | null;
  icIr: number | null;
  /** Two-sided t-stat of mean IC under iid assumption. */
  tStat: number | null;
  topMinusBottom: number | null;
}

export interface WinsorizeOptions {
  lower?: number;
  upper?: number;
}

export interface CrossSectionOptions extends WinsorizeOptions {
  /** Remove the sector mean from each z-score (sector neutrality). */
  sectorNeutral?: boolean;
  /** Weights for the composite; missing ids fall back to equal weight. */
  weights?: Record<string, number>;
  maxAbsoluteZ?: number;
}
