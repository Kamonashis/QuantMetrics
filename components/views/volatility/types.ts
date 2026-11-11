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
}

export interface ChainPayload {
  available: boolean;
  symbol?: string;
  requested?: string;
  substitution?: string | null;
  reason?: string;
  asOf?: string;
  quote?: { price: number; currency: string; name: string; changePct: number | null; marketCap?: number | null };
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
