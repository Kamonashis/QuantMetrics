// Provider interface for option chains (Phase 0.3 of the Trap Indicator spec).
// A better provider (CBOE delayed chain with greeks, NSE bhavcopy for backfill)
// can drop in behind this interface without touching the compute or UI layers.

/** Normalized row — one option contract. Compact on purpose (goes to disk daily). */
export interface RawChainRow {
  /** YYYY-MM-DD of expiry */
  expiry: string;
  /** calendar days to expiry */
  dte: number;
  strike: number;
  type: 'call' | 'put';
  oi: number;
  volume: number;
  bid: number;
  ask: number;
  last: number;
  /** provider-computed IV — cross-check only, never trusted for pricing */
  iv: number | null;
  contractSymbol: string;
}

export interface ChainSnapshot {
  symbol: string;
  /** ISO timestamp of the fetch */
  asOf: string;
  spot: number;
  /** expiration-grouped raw rows, all expiries ≤ maxDte (default 90), OI > 0 */
  rows: RawChainRow[];
  provider: string;
  delayedMinutes: number;
}

export interface ChainProviderOptions {
  /** cap expiries fetched (default 12) */
  maxExpiries?: number;
  /** cap days-to-expiry of included rows (default 90) */
  maxDte?: number;
}

/** Per-expiry raw contract data as delivered by Yahoo v7 (used by the IV pipeline too). */
export interface ProviderExpiry {
  expirationEpoch: number;
  expirationDate: string;
  calls: RawChainRow[];
  puts: RawChainRow[];
}

/** Rich provider result — raw chain + quote + rate, shared by every consumer. */
export interface ProviderChainResult {
  symbol: string;
  substitution: string | null;
  asOf: string;
  spot: number;
  quoteAsOf?: string | null;
  prevClose?: number | null;
  currency: string;
  name: string;
  changePct: number | null;
  marketCap: number | null;
  quoteType: string;
  riskFreeRate: number;
  dividendYield: number;
  delayedMinutes: number;
  provider: 'yahoo';
  expirationDates: number[];
  expiries: ProviderExpiry[];
  /** flat rows across all included expiries (same data as expiries, de-grouped) */
  rows: RawChainRow[];
}

export interface ChainProvider {
  id: 'yahoo' | 'cboe' | 'nse';
  supports(symbol: string): boolean;
  fetchChain(symbol: string, opts?: ChainProviderOptions): Promise<ProviderChainResult | null>;
}
