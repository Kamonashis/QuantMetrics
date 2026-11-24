// Shared client-side types for the Implied Vol tab payloads.

export interface SlicePoint {
  strike: number;
  k: number;
  iv: number;
  mid: number;
  relSpread: number;
  weight: number;
  type: 'call' | 'put';
}

export interface SviOut {
  a: number;
  b: number;
  rho: number;
  m: number;
  sigma: number;
  rmseIv: number;
  nUsed: number;
  arbitrageFlags: string[];
}

export interface SliceOut {
  expirationDate: string;
  dte: number;
  t: number;
  forward: number;
  atmIv: number | null;
  rr25: number | null;
  bf25: number | null;
  skew9010: number | null;
  mfivAnnualized: number | null;
  expectedMove: { oneSigmaPct: number; oneSigmaCurrency: number; straddleImpliedPct: number | null } | null;
  svi: SviOut | null;
  clean: { used: number; total: number; medianRelSpread: number | null };
  points: SlicePoint[];
}

export interface RndOut {
  strikes: number[];
  density: number[];
  minDensity: number;
  integrated: number;
  cdf: number[];
  clippedFraction: number;
  forwardError: number;
  edgeMass: number;
}

export interface ChainPayload {
  available: boolean;
  symbol?: string;
  requested?: string;
  substitution?: string | null;
  reason?: string;
  asOf?: string;
  quote?: { price: number; currency: string; name: string; changePct: number | null; marketCap?: number | null; asOf?: string | null; prevClose?: number | null; source?: string; securityType?: string };
  rate?: number;
  dividendYield?: number;
  iv30?: number | null;
  iv30Source?: string;
  termStructure?: { dte: number; t: number; iv: number | null }[];
  slices?: SliceOut[];
  rnd?: RndOut | null;
  pricedBand?: { oneSigmaPct: number; probDrawdown15: number; probTouch10: number } | null;
  snapshot?: {
    count: number;
    firstDate: string | null;
    lastDate: string | null;
    iv30HistoryCount: number;
    writeOk: boolean;
    snapshotError: string | null;
  };
  trap?: TrapPayload | null;
  ivHistory30?: number[];
  warnings?: string[];
  fetchedAt?: string;
}

export interface IndexSeries {
  symbol: string;
  key: 'VIX' | 'VIX9D' | 'VIX3M' | 'VIX6M' | 'VXN' | 'VVIX' | 'SKEW' | 'IRX';
  bars: { date: string; close: number }[];
  source: 'live_market' | 'synthetic_fallback';
}

export interface IndicesPayload {
  available: boolean;
  mode?: string;
  source?: string;
  reason?: string;
  series?: Record<string, IndexSeries>;
}

export interface IvHistoryEntry {
  symbol: string;
  date: string;
  iv30: number | null;
  atmIv: number | null;
  skewRr25: number | null;
  nContractsUsed: number;
  source: string;
}

// ---------------------------------------------------------------------------
// Trap (dealer positioning) payload
// ---------------------------------------------------------------------------

export interface WallSnapshot {
  strike: number;
  oi: number;
  share: number;
  distPct: number;
  distSigma: number;
}

export type TrapState = 'idle' | 'building' | 'armed' | 'triggered' | 'exhausted';
export type TrapDirection = 'up' | 'down' | 'none';

export interface TrapPayload {
  available: boolean;
  spot: number;
  forward: number;
  atmIv: number;
  headlineDte: number;
  atmIv30: number | null;
  totalGex: number;
  regime: 'long' | 'short' | 'unknown';
  gammaFlip: { flip: number | null; regime: 'long' | 'short' | 'unknown' };
  flipProfile: { spot: number; gex: number }[];
  byStrike: { strike: number; callOi: number; putOi: number; gex: number }[];
  maxAbsGex: number;
  walls: {
    call: WallSnapshot | null;
    put: WallSnapshot | null;
    runnerUpCall: WallSnapshot | null;
    runnerUpPut: WallSnapshot | null;
    totalOiLocal: number;
  };
  flow: { perPctUp: number; perPctDown: number; net: number };
  expiryConcentration7d: number;
  expectedMove30: number | null;
  rv20: number | null;
  ret5d: number | null;
  oiChange1d: number | null;
  oiChange5d: number | null;
  oiChange5dZ: number | null;
  convention: 'squeeze' | 'inverse';
  previousState: TrapState | null;
  score: number;
  state: TrapState;
  direction: TrapDirection;
  components: { wall: number; proximity: number; regime: number; velocity: number; coil: number };
  velocityDegraded: boolean;
  nearWall: 'call' | 'put' | 'none';
}

export interface PanelRowOut {
  date: string;
  spot: number;
  atmIv30: number | null;
  expectedMove30: number | null;
  totalGex: number | null;
  gammaFlip: number | null;
  regime: 'long' | 'short' | 'unknown';
  callWall: WallSnapshot | null;
  putWall: WallSnapshot | null;
  expiryConcentration7d: number | null;
  oiWallChange1d: number | null;
  oiWallChange5d: number | null;
  oiWallChangeZ: number | null;
  trapScore: number | null;
  trapState: TrapState | null;
  trapDirection: TrapDirection | null;
  rv20: number | null;
  ret5d: number | null;
  computedAt: string;
}
