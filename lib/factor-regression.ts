// Time-series factor regression lab — pure math, no React, no I/O.
//
// Pipeline: price bars → date-intersected return sample → rank-aware OLS (MGS-QR)
// → Newey–West HAC inference → attribution, variance shares, diagnostics,
// rolling exposures, nested-model comparison and a Huber robustness check.
//
// Factor returns are ETF long/short proxies (e.g. size = IWM − SPY). They are
// tradable approximations of the academic factors, not Ken French's portfolios,
// and every surface labels them as proxies.

import type { PriceBar } from './quant-engine';

export const TRADING_DAYS = 252;
export const MIN_OBSERVATIONS = 60;
export const CAUTION_OBSERVATIONS = 252;

// ---------------------------------------------------------------------------
// Factor proxy registry
// ---------------------------------------------------------------------------

export type StyleFactorId = 'SIZE' | 'VALUE' | 'MOMENTUM' | 'QUALITY' | 'LOWVOL' | 'NASDAQ';

export interface StyleFactorDef {
  id: StyleFactorId;
  label: string;
  short: string;
  long: string;
  shortLeg: string;
  /** What a +1 point move in the factor means, in plain English. */
  phrase: string;
  description: string;
}

export const STYLE_FACTORS: readonly StyleFactorDef[] = [
  {
    id: 'SIZE',
    label: 'Size',
    short: 'SMB',
    long: 'IWM',
    shortLeg: 'SPY',
    phrase: 'small caps (IWM) beat SPY by 1 pt',
    description: 'Small minus big: Russell 2000 minus S&P 500.',
  },
  {
    id: 'VALUE',
    label: 'Value',
    short: 'HML',
    long: 'IWD',
    shortLeg: 'IWF',
    phrase: 'value stocks (IWD) beat growth stocks (IWF) by 1 pt',
    description: 'High minus low: Russell 1000 Value minus Russell 1000 Growth.',
  },
  {
    id: 'MOMENTUM',
    label: 'Momentum',
    short: 'MOM',
    long: 'MTUM',
    shortLeg: 'SPY',
    phrase: 'momentum stocks (MTUM) beat SPY by 1 pt',
    description: 'Recent winners minus the market: MSCI USA Momentum minus S&P 500.',
  },
  {
    id: 'QUALITY',
    label: 'Quality',
    short: 'QMJ',
    long: 'QUAL',
    shortLeg: 'SPY',
    phrase: 'quality stocks (QUAL) beat SPY by 1 pt',
    description: 'Quality minus the market: MSCI USA Quality minus S&P 500.',
  },
  {
    id: 'LOWVOL',
    label: 'Low volatility',
    short: 'LVOL',
    long: 'USMV',
    shortLeg: 'SPY',
    phrase: 'low-volatility stocks (USMV) beat SPY by 1 pt',
    description: 'Low-vol minus the market: MSCI USA Min Vol minus S&P 500.',
  },
  {
    id: 'NASDAQ',
    label: 'Nasdaq tilt',
    short: 'NDX',
    long: 'QQQ',
    shortLeg: 'SPY',
    phrase: 'the Nasdaq-100 (QQQ) beat SPY by 1 pt',
    description: 'Tech/growth tilt: Nasdaq-100 minus S&P 500.',
  },
];

export const STYLE_FACTOR_BY_ID: Record<StyleFactorId, StyleFactorDef> = Object.fromEntries(
  STYLE_FACTORS.map((f) => [f.id, f])
) as Record<StyleFactorId, StyleFactorDef>;

export interface ModelPreset {
  id: string;
  label: string;
  hint: string;
  factors: StyleFactorId[];
}

export const MODEL_PRESETS: readonly ModelPreset[] = [
  { id: 'market', label: 'Market only', hint: 'Single-index beta', factors: [] },
  { id: 'ff3', label: 'Market + Size + Value', hint: 'Fama–French 3 style', factors: ['SIZE', 'VALUE'] },
  {
    id: 'style',
    label: 'Five style factors',
    hint: 'Market, size, value, momentum, quality, low vol',
    factors: ['SIZE', 'VALUE', 'MOMENTUM', 'QUALITY', 'LOWVOL'],
  },
];

export interface LabConfig {
  asset: string;
  benchmark: string;
  factors: StyleFactorId[];
  customTickers: string[];
  startDate?: string;
  endDate?: string;
  returnType: 'simple' | 'log';
}

/** Every ticker whose bars the configured model needs, asset first. */
export function requiredTickers(config: Pick<LabConfig, 'asset' | 'benchmark' | 'factors' | 'customTickers'>): string[] {
  const out: string[] = [];
  const push = (t: string) => {
    const s = t.trim().toUpperCase();
    if (s && !out.includes(s)) out.push(s);
  };
  push(config.asset);
  push(config.benchmark);
  for (const t of config.customTickers) push(t);
  for (const id of config.factors) {
    const def = STYLE_FACTOR_BY_ID[id];
    if (!def) continue;
    push(def.long);
    push(def.shortLeg);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sample construction — returns on the intersection of price dates
// ---------------------------------------------------------------------------

export type ColumnKind = 'market' | 'style' | 'custom';

export interface FactorColumn {
  key: string;
  label: string;
  short: string;
  kind: ColumnKind;
  phrase: string;
  description: string;
  values: number[];
}

export interface SeriesAudit {
  ticker: string;
  roles: string[];
  barsInWindow: number;
  firstDate: string | null;
  lastDate: string | null;
  basis: 'adjusted' | 'close';
}

export interface SampleAudit {
  start: string | null;
  end: string | null;
  series: SeriesAudit[];
  missing: string[];
  commonPriceDates: number;
  returnObs: number;
  assetBarsInWindow: number;
  /** Asset sessions dropped because another series has no bar that day. */
  droppedAssetSessions: number;
  /** Returns that span more than one asset session because of a calendar gap. */
  multiSessionReturns: number;
  returnType: 'simple' | 'log';
}

export interface FactorSample {
  asset: string;
  dates: string[];
  y: number[];
  columns: FactorColumn[];
  audit: SampleAudit;
}

export type SampleResult = { ok: true; sample: FactorSample } | { ok: false; reason: string; audit: SampleAudit };

function priceMap(bars: readonly PriceBar[], start?: string, end?: string): { map: Map<string, number>; basis: 'adjusted' | 'close'; dates: string[] } {
  const inWindow = bars.filter((b) => (!start || b.date >= start) && (!end || b.date <= end));
  const allAdjusted =
    inWindow.length > 0 &&
    inWindow.every((b) => {
      const adj = (b as PriceBar & { adjClose?: number }).adjClose;
      return typeof adj === 'number' && Number.isFinite(adj) && adj > 0;
    });
  const map = new Map<string, number>();
  for (const b of inWindow) {
    const px = allAdjusted ? (b as PriceBar & { adjClose?: number }).adjClose! : b.close;
    if (Number.isFinite(px) && px > 0) map.set(b.date, px);
  }
  const dates = [...map.keys()].sort();
  return { map, basis: allAdjusted ? 'adjusted' : 'close', dates };
}

function periodReturn(p0: number, p1: number, type: 'simple' | 'log'): number {
  return type === 'log' ? Math.log(p1 / p0) : p1 / p0 - 1;
}

export function buildFactorSample(bars: Record<string, readonly PriceBar[] | undefined>, config: LabConfig): SampleResult {
  const asset = config.asset.trim().toUpperCase();
  const benchmark = config.benchmark.trim().toUpperCase();
  const customs = [...new Set(config.customTickers.map((t) => t.trim().toUpperCase()).filter((t) => t && t !== asset && t !== benchmark))];
  const factors = [...new Set(config.factors)].filter((id) => STYLE_FACTOR_BY_ID[id]);
  const tickers = requiredTickers({ asset, benchmark, factors, customTickers: customs });

  const roles = new Map<string, string[]>();
  const addRole = (t: string, r: string) => roles.set(t, [...(roles.get(t) ?? []), r]);
  addRole(asset, 'asset');
  addRole(benchmark, 'benchmark');
  customs.forEach((t) => addRole(t, 'custom'));
  for (const id of factors) {
    const def = STYLE_FACTOR_BY_ID[id];
    addRole(def.long, `${def.short} long`);
    addRole(def.shortLeg, `${def.short} short`);
  }

  const maps = new Map<string, ReturnType<typeof priceMap>>();
  const missing: string[] = [];
  for (const t of tickers) {
    const pm = priceMap(bars[t] ?? [], config.startDate, config.endDate);
    maps.set(t, pm);
    if (pm.dates.length < 2) missing.push(t);
  }

  const assetDates = maps.get(asset)?.dates ?? [];
  const common = assetDates.filter((d) => tickers.every((t) => maps.get(t)!.map.has(d)));

  const audit: SampleAudit = {
    start: common[0] ?? null,
    end: common[common.length - 1] ?? null,
    series: tickers.map((t) => {
      const pm = maps.get(t)!;
      return {
        ticker: t,
        roles: roles.get(t) ?? [],
        barsInWindow: pm.dates.length,
        firstDate: pm.dates[0] ?? null,
        lastDate: pm.dates[pm.dates.length - 1] ?? null,
        basis: pm.basis,
      };
    }),
    missing,
    commonPriceDates: common.length,
    returnObs: Math.max(0, common.length - 1),
    assetBarsInWindow: assetDates.length,
    droppedAssetSessions: assetDates.length - common.length,
    multiSessionReturns: 0,
    returnType: config.returnType,
  };

  if (missing.length > 0) {
    return { ok: false, reason: `No price history in this window for ${missing.join(', ')}.`, audit };
  }

  const assetIndex = new Map(assetDates.map((d, i) => [d, i]));
  const returnsOf = (t: string): number[] => {
    const m = maps.get(t)!.map;
    const out: number[] = [];
    for (let i = 1; i < common.length; i++) out.push(periodReturn(m.get(common[i - 1])!, m.get(common[i])!, config.returnType));
    return out;
  };

  let multi = 0;
  for (let i = 1; i < common.length; i++) {
    if (assetIndex.get(common[i])! - assetIndex.get(common[i - 1])! > 1) multi++;
  }
  audit.multiSessionReturns = multi;

  if (audit.returnObs < MIN_OBSERVATIONS) {
    return {
      ok: false,
      reason: `Only ${audit.returnObs} overlapping daily returns — at least ${MIN_OBSERVATIONS} are needed for a usable fit.`,
      audit,
    };
  }

  const ret = new Map<string, number[]>();
  for (const t of tickers) ret.set(t, returnsOf(t));

  const columns: FactorColumn[] = [
    {
      key: 'MKT',
      label: `Market (${benchmark})`,
      short: 'MKT',
      kind: 'market',
      phrase: `${benchmark} rose 1%`,
      description: `Daily return of ${benchmark}.`,
      values: ret.get(benchmark)!,
    },
  ];
  for (const id of factors) {
    const def = STYLE_FACTOR_BY_ID[id];
    const l = ret.get(def.long)!;
    const s = ret.get(def.shortLeg)!;
    columns.push({
      key: def.id,
      label: def.label,
      short: def.short,
      kind: 'style',
      phrase: def.phrase,
      description: `${def.description} ETF proxy: ${def.long} − ${def.shortLeg}.`,
      values: l.map((v, i) => v - s[i]),
    });
  }
  for (const t of customs) {
    columns.push({
      key: `X:${t}`,
      label: t,
      short: t,
      kind: 'custom',
      phrase: `${t} rose 1%`,
      description: `Daily return of ${t}.`,
      values: ret.get(t)!,
    });
  }

  return {
    ok: true,
    sample: { asset, dates: common.slice(1), y: ret.get(asset)!, columns, audit },
  };
}

// ---------------------------------------------------------------------------
// Distribution functions (exact, via incomplete beta / gamma)
// ---------------------------------------------------------------------------

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function logGamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  const z = x - 1;
  let a = 0.99999999999980993;
  const t = z + 7.5;
  for (let i = 0; i < LANCZOS.length; i++) a += LANCZOS[i] / (z + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return h;
}

/** Regularized incomplete beta I_x(a, b). */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(lbt);
  if (x < (a + 1) / (a + b + 2)) return (bt * betaContinuedFraction(a, b, x)) / a;
  return 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
}

/** Regularized lower incomplete gamma P(a, x). */
export function incompleteGamma(a: number, x: number): number {
  if (x <= 0) return 0;
  const gln = logGamma(a);
  if (x < a + 1) {
    let ap = a;
    let sum = 1 / a;
    let del = sum;
    for (let n = 0; n < 500; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  const FPMIN = 1e-300;
  let b = x + 1 - a;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - gln) * h;
}

/** Two-sided p-value for a t statistic. */
export function tTwoSidedP(t: number, df: number): number {
  if (!Number.isFinite(t)) return Number.isNaN(t) ? NaN : 0;
  return incompleteBeta(df / (df + t * t), df / 2, 0.5);
}

/** Upper-tail p-value for an F statistic. */
export function fUpperP(f: number, d1: number, d2: number): number {
  if (!(f > 0)) return 1;
  return incompleteBeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
}

/** Upper-tail p-value for a chi-square statistic. */
export function chiSquareUpperP(x: number, k: number): number {
  if (!(x > 0)) return 1;
  return Math.max(0, 1 - incompleteGamma(k / 2, x / 2));
}

const tCritCache = new Map<string, number>();

/** Two-sided critical value: P(|T| > t) = alpha. */
export function tCritical(df: number, alpha = 0.05): number {
  const key = `${df}|${alpha}`;
  const hit = tCritCache.get(key);
  if (hit !== undefined) return hit;
  let lo = 0;
  let hi = 1000;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (tTwoSidedP(mid, df) > alpha) lo = mid;
    else hi = mid;
  }
  const v = (lo + hi) / 2;
  tCritCache.set(key, v);
  return v;
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

interface QrResult {
  ok: true;
  q: number[][]; // p columns of length n
  r: number[][]; // p × p upper triangular
}

/** Modified Gram–Schmidt with re-orthogonalization; flags columns in the span of earlier ones. */
function qrDecompose(cols: number[][]): QrResult | { ok: false; dependent: number } {
  const p = cols.length;
  const n = cols[0]?.length ?? 0;
  const q: number[][] = [];
  const r: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    const v = cols[j].slice();
    let norm0 = 0;
    for (let t = 0; t < n; t++) norm0 += v[t] * v[t];
    norm0 = Math.sqrt(norm0);
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < j; i++) {
        let dot = 0;
        const qi = q[i];
        for (let t = 0; t < n; t++) dot += qi[t] * v[t];
        r[i][j] += dot;
        for (let t = 0; t < n; t++) v[t] -= dot * qi[t];
      }
    }
    let norm = 0;
    for (let t = 0; t < n; t++) norm += v[t] * v[t];
    norm = Math.sqrt(norm);
    if (!(norm0 > 0) || norm <= 1e-7 * norm0) return { ok: false, dependent: j };
    r[j][j] = norm;
    for (let t = 0; t < n; t++) v[t] /= norm;
    q.push(v);
  }
  return { ok: true, q, r };
}

function invertUpper(r: number[][]): number[][] {
  const p = r.length;
  const inv: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    inv[j][j] = 1 / r[j][j];
    for (let i = j - 1; i >= 0; i--) {
      let s = 0;
      for (let k = i + 1; k <= j; k++) s += r[i][k] * inv[k][j];
      inv[i][j] = -s / r[i][i];
    }
  }
  return inv;
}

/** Gauss–Jordan inverse with partial pivoting; null when singular. */
export function invertSymmetric(a: number[][]): number[][] | null {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(m[k][i]) > Math.abs(m[piv][i])) piv = k;
    if (Math.abs(m[piv][i]) < 1e-300) return null;
    [m[i], m[piv]] = [m[piv], m[i]];
    const d = m[i][i];
    for (let j = 0; j < 2 * n; j++) m[i][j] /= d;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const f = m[k][i];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[k][j] -= f * m[i][j];
    }
  }
  return m.map((row) => row.slice(n));
}

export function neweyWestLags(n: number): number {
  return Math.max(1, Math.floor(4 * Math.pow(n / 100, 2 / 9)));
}

// ---------------------------------------------------------------------------
// Core OLS fit
// ---------------------------------------------------------------------------

export interface RegressorInput {
  key: string;
  label: string;
  values: number[];
}

interface CoreFit {
  n: number;
  p: number;
  beta: number[];
  residuals: number[];
  fitted: number[];
  rss: number;
  tss: number;
  xtxInv: number[][];
  leverage: number[];
  /** HAC (or classical) covariance of beta. */
  cov: number[][];
  covClassical: number[][];
  sigma2: number;
}

function coreOls(
  y: number[],
  regs: number[][],
  se: 'newey-west' | 'classical',
  lags: number,
  weights?: number[]
): CoreFit | { ok: false; dependent: number } {
  const n = y.length;
  const cols = [new Array(n).fill(1), ...regs];
  const sw = weights?.map((w) => Math.sqrt(w));
  const wCols = sw ? cols.map((c) => c.map((v, t) => v * sw[t])) : cols;
  const wy = sw ? y.map((v, t) => v * sw[t]) : y;
  const qr = qrDecompose(wCols);
  if ('dependent' in qr) return { ok: false, dependent: qr.dependent };
  const p = cols.length;
  const qty = qr.q.map((qc) => {
    let s = 0;
    for (let t = 0; t < n; t++) s += qc[t] * wy[t];
    return s;
  });
  const beta = new Array(p).fill(0);
  for (let i = p - 1; i >= 0; i--) {
    let s = qty[i];
    for (let k = i + 1; k < p; k++) s -= qr.r[i][k] * beta[k];
    beta[i] = s / qr.r[i][i];
  }
  const rInv = invertUpper(qr.r);
  const xtxInv: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0;
      for (let k = Math.max(i, j); k < p; k++) s += rInv[i][k] * rInv[j][k];
      xtxInv[i][j] = s;
      xtxInv[j][i] = s;
    }
  }

  const fitted = new Array(n);
  const residuals = new Array(n);
  let yMean = 0;
  for (let t = 0; t < n; t++) yMean += y[t];
  yMean /= n;
  let rss = 0;
  let tss = 0;
  for (let t = 0; t < n; t++) {
    let f = 0;
    for (let j = 0; j < p; j++) f += cols[j][t] * beta[j];
    fitted[t] = f;
    residuals[t] = y[t] - f;
    rss += residuals[t] * residuals[t];
    tss += (y[t] - yMean) ** 2;
  }
  const leverage = new Array(n).fill(0);
  for (const qc of qr.q) for (let t = 0; t < n; t++) leverage[t] += qc[t] * qc[t];

  const df = Math.max(1, n - p);
  const sigma2 = rss / df;
  const covClassical = xtxInv.map((row) => row.map((v) => v * sigma2));

  let cov = covClassical;
  if (se === 'newey-west' && !weights) {
    const u: number[][] = Array.from({ length: n }, (_, t) => cols.map((c) => c[t] * residuals[t]));
    const S: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let t = 0; t < n; t++) {
      const ut = u[t];
      for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) S[i][j] += ut[i] * ut[j];
    }
    const L = Math.min(lags, n - 1);
    for (let l = 1; l <= L; l++) {
      const w = 1 - l / (L + 1);
      for (let t = l; t < n; t++) {
        const ut = u[t];
        const us = u[t - l];
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) S[i][j] += w * (ut[i] * us[j] + us[i] * ut[j]);
        }
      }
    }
    const scale = n / df;
    const AS = xtxInv.map((row) => S[0].map((_, j) => row.reduce((s, v, k) => s + v * S[k][j], 0)));
    cov = AS.map((row) => xtxInv[0].map((_, j) => scale * row.reduce((s, v, k) => s + v * xtxInv[k][j], 0)));
  }

  return { n, p, beta, residuals, fitted, rss, tss, xtxInv, leverage, cov, covClassical, sigma2 };
}

// ---------------------------------------------------------------------------
// Public fit
// ---------------------------------------------------------------------------

export type Evidence = 'strong' | 'moderate' | 'weak' | 'none';

export function evidenceLevel(p: number): Evidence {
  if (!Number.isFinite(p)) return 'none';
  if (p < 0.01) return 'strong';
  if (p < 0.05) return 'moderate';
  if (p < 0.1) return 'weak';
  return 'none';
}

export interface CoefficientRow {
  key: string;
  label: string;
  estimate: number;
  se: number;
  seClassical: number;
  t: number;
  p: number;
  ciLow: number;
  ciHigh: number;
  vif: number | null;
  robustEstimate: number | null;
  evidence: Evidence;
}

export interface FactorFit {
  ok: true;
  n: number;
  p: number;
  df: number;
  seMethod: 'newey-west' | 'classical';
  hacLags: number;
  coefficients: CoefficientRow[];
  r2: number;
  adjR2: number;
  sigma: number;
  residuals: number[];
  fitted: number[];
  fStat: number | null;
  fP: number | null;
  /** HAC Wald test that every slope is zero. */
  waldAll: { stat: number; df: number; p: number } | null;
  aic: number;
  bic: number;
  durbinWatson: number;
  ljungBox: { q: number; lags: number; p: number };
  jarqueBera: { stat: number; p: number; skew: number; excessKurtosis: number };
  leverage: number[];
  cooks: number[];
  varianceShares: { key: string; label: string; share: number }[];
  residualShare: number;
  maxVif: number | null;
  cov: number[][];
}

export type FitOutcome = FactorFit | { ok: false; reason: string; dependent?: string };

export interface FitOptions {
  se?: 'newey-west' | 'classical';
  hacLags?: number;
  robust?: boolean;
}

function huberCoefficients(y: number[], regs: number[][], c = 1.345): number[] | null {
  const first = coreOls(y, regs, 'classical', 0);
  if (!('beta' in first)) return null;
  let fit: CoreFit = first;
  for (let iter = 0; iter < 30; iter++) {
    const res = fit.residuals;
    const med = median(res);
    const mad = median(res.map((r) => Math.abs(r - med))) / 0.6745;
    const scale = mad > 1e-12 ? mad : Math.sqrt(fit.sigma2) || 1e-12;
    const w = res.map((r) => {
      const u = Math.abs(r) / scale;
      return u <= c ? 1 : c / u;
    });
    const next = coreOls(y, regs, 'classical', 0, w);
    if (!('beta' in next)) return null;
    const delta = Math.max(...next.beta.map((b, i) => Math.abs(b - fit.beta[i])));
    fit = next;
    if (delta < 1e-10) break;
  }
  return fit.beta;
}

function median(v: number[]): number {
  if (v.length === 0) return NaN;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function residualDiagnostics(e: number[]) {
  const n = e.length;
  const mean = e.reduce((a, b) => a + b, 0) / n;
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const v of e) {
    const d = v - mean;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const skew = m2 > 0 ? m3 / Math.pow(m2, 1.5) : 0;
  const kurt = m2 > 0 ? m4 / (m2 * m2) : 3;
  const jb = (n / 6) * (skew * skew + ((kurt - 3) ** 2) / 4);

  let dwNum = 0;
  let ss = 0;
  for (let t = 0; t < n; t++) {
    ss += e[t] * e[t];
    if (t > 0) dwNum += (e[t] - e[t - 1]) ** 2;
  }
  const lags = Math.max(1, Math.min(10, Math.floor(n / 5)));
  let q = 0;
  const denom = n * m2;
  for (let k = 1; k <= lags; k++) {
    let c = 0;
    for (let t = k; t < n; t++) c += (e[t] - mean) * (e[t - k] - mean);
    const rho = denom > 0 ? c / denom : 0;
    q += (rho * rho) / (n - k);
  }
  q *= n * (n + 2);

  return {
    durbinWatson: ss > 0 ? dwNum / ss : 2,
    ljungBox: { q, lags, p: chiSquareUpperP(q, lags) },
    jarqueBera: { stat: jb, p: chiSquareUpperP(jb, 2), skew, excessKurtosis: kurt - 3 },
  };
}

function waldTest(beta: number[], cov: number[][], idx: number[]): { stat: number; df: number; p: number } | null {
  if (idx.length === 0) return null;
  const b = idx.map((i) => beta[i]);
  const V = idx.map((i) => idx.map((j) => cov[i][j]));
  const inv = invertSymmetric(V);
  if (!inv) return null;
  let stat = 0;
  for (let i = 0; i < b.length; i++) for (let j = 0; j < b.length; j++) stat += b[i] * inv[i][j] * b[j];
  return { stat, df: idx.length, p: chiSquareUpperP(stat, idx.length) };
}

export function fitFactorModel(y: number[], regressors: RegressorInput[], options: FitOptions = {}): FitOutcome {
  const n = y.length;
  const seMethod = options.se ?? 'newey-west';
  const p = regressors.length + 1;
  if (regressors.some((r) => r.values.length !== n)) return { ok: false, reason: 'Regressor lengths do not match the dependent series.' };
  if (n < p + 2) return { ok: false, reason: `Need more than ${p + 1} observations to fit ${p} coefficients.` };
  if (![...y, ...regressors.flatMap((r) => r.values)].every(Number.isFinite)) {
    return { ok: false, reason: 'Sample contains non-finite values.' };
  }

  const lags = options.hacLags ?? neweyWestLags(n);
  const regs = regressors.map((r) => r.values);
  const core = coreOls(y, regs, seMethod, lags);
  if (!('beta' in core)) {
    const dep = core.dependent === 0 ? 'Intercept' : regressors[core.dependent - 1]?.label ?? 'a regressor';
    return {
      ok: false,
      dependent: dep,
      reason:
        core.dependent === 0
          ? 'The intercept could not be estimated.'
          : `${dep} is (almost) an exact combination of the other regressors — the model cannot separate their effects. Remove it or one of the overlapping factors.`,
    };
  }

  const df = n - p;
  const tc = tCritical(df);
  const robust = options.robust === false ? null : huberCoefficients(y, regs);
  const names = [{ key: 'ALPHA', label: 'Alpha (intercept)' }, ...regressors.map((r) => ({ key: r.key, label: r.label }))];

  const vifs: (number | null)[] = [null];
  for (let j = 0; j < regressors.length; j++) {
    const v = regs[j];
    const m = v.reduce((a, b) => a + b, 0) / n;
    const sst = v.reduce((a, b) => a + (b - m) ** 2, 0);
    vifs.push(regressors.length > 1 ? core.xtxInv[j + 1][j + 1] * sst : 1);
  }

  const coefficients: CoefficientRow[] = names.map((nm, i) => {
    const est = core.beta[i];
    const se = Math.sqrt(Math.max(0, core.cov[i][i]));
    const seC = Math.sqrt(Math.max(0, core.covClassical[i][i]));
    const t = se > 0 ? est / se : 0;
    const pv = se > 0 ? tTwoSidedP(t, df) : 1;
    return {
      key: nm.key,
      label: nm.label,
      estimate: est,
      se,
      seClassical: seC,
      t,
      p: pv,
      ciLow: est - tc * se,
      ciHigh: est + tc * se,
      vif: vifs[i],
      robustEstimate: robust ? robust[i] : null,
      evidence: evidenceLevel(pv),
    };
  });

  const r2 = core.tss > 0 ? 1 - core.rss / core.tss : 0;
  const adjR2 = 1 - (1 - r2) * ((n - 1) / df);
  const k = p - 1;
  const fStat = k > 0 && 1 - r2 > 1e-15 ? (r2 / k) / ((1 - r2) / df) : null;
  const fP = fStat !== null ? fUpperP(fStat, k, df) : null;
  const waldAll = waldTest(core.beta, core.cov, Array.from({ length: k }, (_, i) => i + 1));

  const ll = -0.5 * n * (Math.log(2 * Math.PI) + Math.log(core.rss / n) + 1);
  const params = p + 1;
  const aic = 2 * params - 2 * ll;
  const bic = params * Math.log(n) - 2 * ll;

  const diag = residualDiagnostics(core.residuals);
  const cooks = core.residuals.map((e, t) => {
    const h = core.leverage[t];
    return h < 1 ? ((e * e) / (p * core.sigma2)) * (h / ((1 - h) ** 2)) : 0;
  });

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const varY = core.tss / n;
  const varianceShares = regressors.map((r, j) => {
    const xm = r.values.reduce((a, b) => a + b, 0) / n;
    let cov = 0;
    for (let t = 0; t < n; t++) cov += (r.values[t] - xm) * (y[t] - yMean);
    cov /= n;
    return { key: r.key, label: r.label, share: varY > 0 ? (core.beta[j + 1] * cov) / varY : 0 };
  });

  const vifOnly = vifs.filter((v): v is number => v !== null);
  return {
    ok: true,
    n,
    p,
    df,
    seMethod,
    hacLags: seMethod === 'newey-west' ? lags : 0,
    coefficients,
    r2,
    adjR2,
    sigma: Math.sqrt(core.sigma2),
    residuals: core.residuals,
    fitted: core.fitted,
    fStat,
    fP,
    waldAll,
    aic,
    bic,
    durbinWatson: diag.durbinWatson,
    ljungBox: diag.ljungBox,
    jarqueBera: diag.jarqueBera,
    leverage: core.leverage,
    cooks,
    varianceShares,
    residualShare: 1 - r2,
    maxVif: regressors.length > 1 && vifOnly.length > 0 ? Math.max(...vifOnly) : null,
    cov: core.cov,
  };
}

export function fitSample(sample: FactorSample, options: FitOptions = {}): FitOutcome {
  return fitFactorModel(
    sample.y,
    sample.columns.map((c) => ({ key: c.key, label: c.label, values: c.values })),
    options
  );
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

export interface AttributionComponent {
  key: string;
  label: string;
  /** Sum of daily contributions over the sample (not compounded). */
  total: number;
  /** Mean daily contribution × 252. */
  annualized: number;
  cumulative: number[];
}

export interface Attribution {
  dates: string[];
  actual: AttributionComponent;
  components: AttributionComponent[];
}

export function attributeReturns(sample: FactorSample, fit: FactorFit): Attribution {
  const n = sample.y.length;
  const build = (key: string, label: string, daily: (t: number) => number): AttributionComponent => {
    const cumulative: number[] = new Array(n);
    let s = 0;
    for (let t = 0; t < n; t++) {
      s += daily(t);
      cumulative[t] = s;
    }
    return { key, label, total: s, annualized: (s / n) * TRADING_DAYS, cumulative };
  };
  const alpha = fit.coefficients[0].estimate;
  const components = [
    ...sample.columns.map((c, j) => {
      const b = fit.coefficients[j + 1].estimate;
      return build(c.key, c.label, (t) => b * c.values[t]);
    }),
    build('ALPHA', 'Alpha (unexplained average)', () => alpha),
    build('RESID', 'Day-to-day noise', (t) => fit.residuals[t]),
  ];
  return {
    dates: sample.dates,
    actual: build('ACTUAL', `${sample.asset} actual`, (t) => sample.y[t]),
    components,
  };
}

// ---------------------------------------------------------------------------
// Rolling exposures
// ---------------------------------------------------------------------------

export interface RollingSeries {
  key: string;
  label: string;
  estimate: (number | null)[];
  low: (number | null)[];
  high: (number | null)[];
}

export interface RollingResult {
  window: number;
  step: number;
  dates: string[];
  r2: (number | null)[];
  series: RollingSeries[];
}

export function rollingFactorFit(sample: FactorSample, window: number, maxPoints = 400): RollingResult | null {
  const n = sample.y.length;
  const p = sample.columns.length + 1;
  if (window < p + 10 || n < window) return null;
  const step = Math.max(1, Math.ceil((n - window + 1) / maxPoints));
  const lags = neweyWestLags(window);
  const tc = tCritical(window - p);
  const dates: string[] = [];
  const r2: (number | null)[] = [];
  const series: RollingSeries[] = [
    { key: 'ALPHA', label: 'Alpha (intercept)', estimate: [], low: [], high: [] },
    ...sample.columns.map((c) => ({ key: c.key, label: c.label, estimate: [], low: [], high: [] })),
  ];
  const ends: number[] = [];
  for (let end = n; end >= window; end -= step) ends.push(end);
  ends.reverse();
  for (const end of ends) {
    const s = end - window;
    const fit = coreOls(
      sample.y.slice(s, end),
      sample.columns.map((c) => c.values.slice(s, end)),
      'newey-west',
      lags
    );
    dates.push(sample.dates[end - 1]);
    if (!('beta' in fit)) {
      r2.push(null);
      series.forEach((sr) => {
        sr.estimate.push(null);
        sr.low.push(null);
        sr.high.push(null);
      });
      continue;
    }
    r2.push(fit.tss > 0 ? 1 - fit.rss / fit.tss : null);
    series.forEach((sr, i) => {
      const se = Math.sqrt(Math.max(0, fit.cov[i][i]));
      sr.estimate.push(fit.beta[i]);
      sr.low.push(fit.beta[i] - tc * se);
      sr.high.push(fit.beta[i] + tc * se);
    });
  }
  return { window, step, dates, r2, series };
}

// ---------------------------------------------------------------------------
// Nested model comparison
// ---------------------------------------------------------------------------

export interface ModelComparison {
  restricted: { label: string; r2: number; adjR2: number; bic: number; k: number };
  full: { label: string; r2: number; adjR2: number; bic: number; k: number };
  /** HAC Wald test that the added coefficients are jointly zero. */
  wald: { stat: number; df: number; p: number } | null;
  addedKeys: string[];
  verdict: 'adds' | 'marginal' | 'no-gain';
}

export function compareWithMarketOnly(sample: FactorSample, full: FactorFit): ModelComparison | null {
  if (sample.columns.length < 2) return null;
  const restricted = fitFactorModel(
    sample.y,
    [sample.columns[0]].map((c) => ({ key: c.key, label: c.label, values: c.values })),
    { robust: false }
  );
  if (!restricted.ok) return null;
  const addedIdx = sample.columns.slice(1).map((_, i) => i + 2);
  const wald = waldTest(
    full.coefficients.map((c) => c.estimate),
    full.cov,
    addedIdx
  );
  const bicGain = restricted.bic - full.bic;
  const verdict: ModelComparison['verdict'] =
    wald && wald.p < 0.01 && bicGain > 0 ? 'adds' : wald && wald.p < 0.05 ? 'marginal' : 'no-gain';
  return {
    restricted: { label: 'Market only', r2: restricted.r2, adjR2: restricted.adjR2, bic: restricted.bic, k: 1 },
    full: { label: 'Selected model', r2: full.r2, adjR2: full.adjR2, bic: full.bic, k: sample.columns.length },
    wald,
    addedKeys: sample.columns.slice(1).map((c) => c.key),
    verdict,
  };
}

/** Pearson correlations between the regressor columns. */
export function columnCorrelations(columns: Pick<FactorColumn, 'values'>[]): number[][] {
  const stats = columns.map((c) => {
    const n = c.values.length;
    const m = c.values.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(c.values.reduce((a, b) => a + (b - m) ** 2, 0));
    return { m, sd };
  });
  return columns.map((a, i) =>
    columns.map((b, j) => {
      if (i === j) return 1;
      const { m: ma, sd: sa } = stats[i];
      const { m: mb, sd: sb } = stats[j];
      if (!(sa > 0 && sb > 0)) return 0;
      let s = 0;
      for (let t = 0; t < a.values.length; t++) s += (a.values[t] - ma) * (b.values[t] - mb);
      return s / (sa * sb);
    })
  );
}

// ---------------------------------------------------------------------------
// Influence & robustness
// ---------------------------------------------------------------------------

export interface InfluentialDate {
  date: string;
  actual: number;
  fitted: number;
  residual: number;
  cooks: number;
  leverage: number;
}

export function influentialDates(sample: FactorSample, fit: FactorFit, top = 6): InfluentialDate[] {
  return fit.cooks
    .map((c, t) => ({ c, t }))
    .sort((a, b) => b.c - a.c)
    .slice(0, top)
    .map(({ c, t }) => ({
      date: sample.dates[t],
      actual: sample.y[t],
      fitted: fit.fitted[t],
      residual: fit.residuals[t],
      cooks: c,
      leverage: fit.leverage[t],
    }));
}

/** Coefficients whose Huber estimate sits more than one HAC SE from OLS. */
export function outlierSensitiveCoefficients(fit: FactorFit): CoefficientRow[] {
  return fit.coefficients.filter(
    (c) => c.robustEstimate !== null && c.se > 0 && Math.abs(c.robustEstimate - c.estimate) > c.se
  );
}

// ---------------------------------------------------------------------------
// Plain-language readouts
// ---------------------------------------------------------------------------

export const EVIDENCE_LABEL: Record<Evidence, string> = {
  strong: 'Strong evidence',
  moderate: 'Moderate evidence',
  weak: 'Weak evidence',
  none: 'No clear evidence',
};

export function pct(v: number, digits = 2, signed = true): string {
  const s = (v * 100).toFixed(digits);
  return signed && v >= 0 ? `+${s}%` : `${s}%`;
}

/** “When SPY rose 1%, AAPL typically rose 1.18% (95% CI 1.05 to 1.31).” */
export function describeExposure(asset: string, column: Pick<FactorColumn, 'phrase' | 'kind'>, row: CoefficientRow, multiFactor: boolean): string {
  const b = row.estimate;
  const ci = `95% range ${row.ciLow.toFixed(2)} to ${row.ciHigh.toFixed(2)}`;
  const straddles = row.ciLow <= 0 && row.ciHigh >= 0;
  const held = multiFactor ? ', with the other factors held fixed' : '';
  if (straddles) {
    return `When ${column.phrase}, ${asset} showed no reliable response${held} (estimate ${b >= 0 ? '+' : ''}${b.toFixed(2)}%, ${ci}).`;
  }
  const verb = b >= 0 ? 'rose' : 'fell';
  return `When ${column.phrase}, ${asset} typically ${verb} ${Math.abs(b).toFixed(2)}%${held} (${ci}).`;
}

export function describeAlpha(asset: string, row: CoefficientRow): string {
  const ann = row.estimate * TRADING_DAYS;
  const lo = row.ciLow * TRADING_DAYS;
  const hi = row.ciHigh * TRADING_DAYS;
  const range = `95% range ${pct(lo, 1)} to ${pct(hi, 1)} per year`;
  if (lo <= 0 && hi >= 0) {
    return `After removing factor moves, ${asset}'s average leftover return of ${pct(ann, 1)}/yr is indistinguishable from zero (${range}).`;
  }
  return `After removing factor moves, ${asset} averaged ${pct(ann, 1)}/yr ${ann >= 0 ? 'more' : 'less'} than the factors imply (${range}).`;
}
