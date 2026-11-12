// Shared types for the F&O analytics module (US universe).

import type { RawChainRow } from '../options/providers/types';

export type { RawChainRow };

export interface ExpiryGroup {
  expiry: string;        // YYYY-MM-DD
  dte: number;
  t: number;             // years
  forward: number | null;
  atmStrike: number | null;
  atmIv: number | null;
  callOi: number;
  putOi: number;
  totalOi: number;
  share7d: number;       // share of universe OI in this bucket if dte <= 7 else 0
}

export interface OiLadderPoint {
  strike: number;
  callOi: number;
  putOi: number;
  totalOi: number;
}

export interface PcrBundle {
  aggregate: number | null;   // ΣPE OI / ΣCE OI
  volume: number | null;      // ΣPE vol / ΣCE vol
  atm: number | null;         // same, ±5 strikes around ATM
}

export interface MaxPainPoint {
  strike: number;
  pain: number;
}

export interface MaxPainResult {
  maxPainStrike: number | null;
  pain: number;
  distancePct: number | null;   // (spot - maxPain) / spot
  basinWidthPct: number | null; // width of strikes with Pain <= 1.05 × min, / spot
  curve: MaxPainPoint[];
}

export interface TermStructurePoint {
  dte: number;
  expiry: string;
  iv: number | null;
  rr25: number | null;
}

export interface TermStructureVerdict {
  verdict: 'CONTANGO' | 'FLAT' | 'BACKWARDATION';
  frontIv: number | null;
  backIv: number | null;
  ratio: number | null;
}

export interface ForwardCarryPair {
  fromDte: number;
  toDte: number;
  carry: number;         // F2/F1 annualized over the gap
}

export interface ExpectedMovePoint {
  dte: number;
  expiry: string;
  atmCall: number | null;
  atmPut: number | null;
  straddleMove: number | null;   // 0.8 × (C+P) premium as % of spot
  sigmaMove: number | null;      // atmIv × sqrt(T) as % of spot
  atmIv: number | null;
}

export interface StatusResult {
  available: boolean;
  symbol: string;
  sessions: number;
  firstDate: string | null;
  lastDate: string | null;
  staleDays: number | null;
}
