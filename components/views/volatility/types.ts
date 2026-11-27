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

export interface GammaWallSnapshot {
  strike: number;
  oi: number;
  gex: number;
  absGex: number;
  share: number;
  distPct: number;
  distSigma: number;
}

export type GexExpiryBucketId = '0dte' | 'weekly' | 'monthly' | 'far';

export interface GexStrikeOut {
  strike: number;
  callOi: number;
  putOi: number;
  gex: number;
  callVolume?: number;
  putVolume?: number;
  callGex?: number;
  putGex?: number;
  callVolumeGex?: number;
  putVolumeGex?: number;
  volumeGex?: number;
  callCharm?: number;
  putCharm?: number;
  charm?: number;
  callVanna?: number;
  putVanna?: number;
  vanna?: number;
}

export interface GexFlipPointOut {
  spot: number;
  gex: number;
  callGex?: number;
  putGex?: number;
  gexSqueeze?: number;
  gexInverse?: number;
}

export interface GexExpiryBucketOut {
  bucket: GexExpiryBucketId;
  label: string;
  dteRange: string;
  oi: number;
  callOi: number;
  putOi: number;
  callGex: number;
  putGex: number;
  totalGex: number;
  absGex: number;
  shareOfAbsGex: number;
  totalVolumeGex: number;
  totalCharm: number;
  totalVanna: number;
  byStrike: GexStrikeOut[];
}

export interface GexStrikeExpiryMatrixOut {
  strikes: number[];
  expiries: {
    expiry: string;
    dte: number;
    bucket: GexExpiryBucketId;
    totalGex: number;
    callGex: number;
    putGex: number;
    totalOi: number;
  }[];
  netGex: number[][];
  callGex: number[][];
  putGex: number[][];
}

export type TrapState = 'idle' | 'building' | 'armed' | 'triggered' | 'exhausted';
export type TrapDirection = 'up' | 'down' | 'none';

export interface GexConventionSummaryOut {
  totalGex: number;
  totalCallGex: number;
  totalPutGex: number;
  totalVolumeGex: number;
  totalCharm: number;
  totalVanna: number;
  flip: number | null;
  regime: 'long' | 'short' | 'unknown';
  flow: { perPctUp: number; perPctDown: number; net: number };
  score: number;
  state: TrapState;
  direction: TrapDirection;
  components: { wall: number; proximity: number; regime: number; velocity: number; coil: number };
  nearWall: 'call' | 'put' | 'none';
}

export interface TrapPayload {
  available: boolean;
  spot: number;
  forward: number;
  atmIv: number;
  headlineDte: number;
  atmIv30: number | null;
  include0Dte?: boolean;
  totalGex: number;
  totalCallGex?: number;
  totalPutGex?: number;
  totalCallVolumeGex?: number;
  totalPutVolumeGex?: number;
  totalVolumeGex?: number;
  totalCallCharm?: number;
  totalPutCharm?: number;
  totalCharm?: number;
  totalCallVanna?: number;
  totalPutVanna?: number;
  totalVanna?: number;
  regime: 'long' | 'short' | 'unknown';
  gammaFlip: { flip: number | null; regime: 'long' | 'short' | 'unknown' };
  gammaFlipInverse?: { flip: number | null; regime: 'long' | 'short' | 'unknown' };
  flipDistPct?: number | null;
  flipDistSigma?: number | null;
  flipProfile: GexFlipPointOut[];
  byStrike: GexStrikeOut[];
  maxAbsGex: number;
  walls: {
    call: WallSnapshot | null;
    put: WallSnapshot | null;
    runnerUpCall: WallSnapshot | null;
    runnerUpPut: WallSnapshot | null;
    totalOiLocal: number;
  };
  gammaWalls?: {
    call: GammaWallSnapshot | null;
    put: GammaWallSnapshot | null;
    runnerUpCall: GammaWallSnapshot | null;
    runnerUpPut: GammaWallSnapshot | null;
    totalAbsGex: number;
  };
  expiryBuckets?: GexExpiryBucketOut[];
  strikeExpiryMatrix?: GexStrikeExpiryMatrixOut;
  flow: { perPctUp: number; perPctDown: number; net: number };
  expiryConcentration7d: number;
  zeroDteShare?: number;
  expectedMove30: number | null;
  rv20: number | null;
  ret5d: number | null;
  adv20Dollar?: number | null;
  gexPerAdv?: number | null;
  oiChange1d: number | null;
  oiChange5d: number | null;
  oiChange5dZ: number | null;
  convention: 'squeeze' | 'inverse';
  byConvention?: {
    squeeze: GexConventionSummaryOut;
    inverse: GexConventionSummaryOut;
  };
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
