// Client-side types for the F&O tab payloads — mirrors app/api/fno response.

export interface ExpiryLadderEntry {
  expiry: string;
  dte: number;
  forward: number | null;
  atmStrike: number | null;
  atmIv: number | null;
  callOi: number;
  putOi: number;
  totalOi: number;
  share7d: number;
}

export interface OiLadderEntry {
  strike: number;
  callOi: number;
  putOi: number;
  totalOi: number;
}

export interface PcrPayload {
  aggregate: number | null;
  volume: number | null;
  atm: number | null;
}

export interface MaxPainPayload {
  maxPainStrike: number | null;
  pain: number;
  distancePct: number | null;
  basinWidthPct: number | null;
  curve: { strike: number; pain: number }[];
}

export interface TermStructurePointOut {
  dte: number;
  expiry: string;
  iv: number | null;
  rr25: number | null;
}

export interface TermStructurePayload {
  points: TermStructurePointOut[];
  verdict: { verdict: 'CONTANGO' | 'FLAT' | 'BACKWARDATION'; frontIv: number | null; backIv: number | null; ratio: number | null };
  forwardCarry: { fromDte: number; toDte: number; carry: number }[];
}

export interface ExpectedMovePayload {
  dte: number;
  expiry: string;
  atmCall: number | null;
  atmPut: number | null;
  straddleMove: number | null;
  sigmaMove: number | null;
  atmIv: number | null;
}

export interface PanelPayload {
  available: boolean;
  mode?: string;
  symbol?: string;
  requested?: string;
  substitution?: string | null;
  reason?: string;
  asOf?: string;
  quote?: { price: number; currency: string; name: string; changePct: number | null; marketCap: number | null; asOf?: string | null; prevClose?: number | null; source?: string };
  rate?: number;
  dividendYield?: number;
  iv30?: number | null;
  stale?: boolean;
  staleDays?: number | null;
  sessions?: number;
  firstDate?: string | null;
  lastDate?: string | null;
  expiryLadder?: ExpiryLadderEntry[];
  oiLadder?: OiLadderEntry[];
  pcr?: PcrPayload;
  maxPain?: MaxPainPayload;
  termStructure?: TermStructurePayload;
  expectedMoves?: ExpectedMovePayload[];
  concentration7d?: number;
  strikesUsed?: number;
  strikesTotal?: number;
  fetchedAt?: string;
}

export interface ChainSide {
  bid: number;
  ask: number;
  mid: number;
  iv: number | null;
  oi: number;
  volume: number;
  vega: number | null;
  delta: number | null;
  spreadPct: number | null;
}

export interface ChainRowOut {
  strike: number;
  call: ChainSide | null;
  put: ChainSide | null;
}

export interface ChainPayload {
  available: boolean;
  mode?: string;
  symbol?: string;
  expiry?: string;
  dte?: number;
  asOf?: string;
  quote?: { price: number };
  atmStrike?: number;
  rows?: ChainRowOut[];
  reason?: string;
}
