// Unit tests for lib/factor-regression.ts (compiled output in .cache/test-build).
// Fixtures: deterministic — no network, no clock.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const F = require(path.join(BUILD, 'factor-regression.js'));

const approx = (a, b, eps = 1e-3) => assert.ok(Math.abs(a - b) < eps, `expected ${b}, got ${a}`);

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gaussian(rand) {
  const u = Math.max(1e-12, rand());
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function dates(n) {
  const out = [];
  const d = new Date(Date.UTC(2020, 0, 1));
  while (out.length < n) {
    const day = d.getUTCDay();
    if (day !== 0 && day !== 6) out.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

function barsFromReturns(ds, rets, start = 100) {
  let p = start;
  const bars = [{ date: ds[0], open: p, high: p, low: p, close: p, volume: 0 }];
  for (let i = 1; i < ds.length; i++) {
    p *= 1 + rets[i - 1];
    bars.push({ date: ds[i], open: p, high: p, low: p, close: p, volume: 0 });
  }
  return bars;
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

test('distribution functions match textbook critical values', () => {
  approx(F.tTwoSidedP(2.228139, 10), 0.05, 1e-5);
  approx(F.tCritical(10), 2.228139, 1e-4);
  approx(F.tCritical(1e6), 1.959964, 1e-4);
  approx(F.chiSquareUpperP(3.841459, 1), 0.05, 1e-5);
  approx(F.chiSquareUpperP(18.307038, 10), 0.05, 1e-5);
  approx(F.fUpperP(4.964603, 1, 10), 0.05, 1e-5);
  approx(F.fUpperP(3.325835, 2, 10), 0.0775, 1e-3);
  assert.equal(F.tTwoSidedP(0, 20), 1);
});

// ---------------------------------------------------------------------------
// OLS
// ---------------------------------------------------------------------------

test('fitFactorModel recovers known coefficients and matches closed-form simple OLS', () => {
  const rand = lcg(7);
  const n = 800;
  const x1 = Array.from({ length: n }, () => 0.01 * gaussian(rand));
  const x2 = Array.from({ length: n }, () => 0.006 * gaussian(rand));
  const y = x1.map((v, i) => 0.0004 + 1.2 * v - 0.5 * x2[i] + 0.004 * gaussian(rand));
  const fit = F.fitFactorModel(y, [
    { key: 'A', label: 'A', values: x1 },
    { key: 'B', label: 'B', values: x2 },
  ]);
  assert.equal(fit.ok, true);
  approx(fit.coefficients[1].estimate, 1.2, 0.05);
  approx(fit.coefficients[2].estimate, -0.5, 0.08);
  assert.ok(fit.coefficients[1].ciLow < 1.2 && fit.coefficients[1].ciHigh > 1.2);
  assert.equal(fit.coefficients[1].evidence, 'strong');
  // Variance shares sum to R².
  approx(fit.varianceShares.reduce((a, s) => a + s.share, 0), fit.r2, 1e-9);
  // Orthogonal-ish regressors ⇒ VIF ≈ 1.
  assert.ok(fit.maxVif < 1.1);

  // Closed-form simple regression check (classical SE).
  const simple = F.fitFactorModel(y, [{ key: 'A', label: 'A', values: x1 }], { se: 'classical' });
  const mx = x1.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x1[i] - mx) * (y[i] - my);
    sxx += (x1[i] - mx) ** 2;
  }
  const beta = sxy / sxx;
  approx(simple.coefficients[1].estimate, beta, 1e-10);
  const rss = y.reduce((a, v, i) => a + (v - (my - beta * mx) - beta * x1[i]) ** 2, 0);
  approx(simple.coefficients[1].se, Math.sqrt(rss / (n - 2) / sxx), 1e-10);
  assert.equal(simple.seMethod, 'classical');
});

test('fitFactorModel refuses perfectly collinear regressors and names the culprit', () => {
  const rand = lcg(3);
  const x = Array.from({ length: 200 }, () => gaussian(rand));
  const z = Array.from({ length: 200 }, () => gaussian(rand));
  const dup = x.map((v, i) => 2 * v - z[i]);
  const y = x.map((v) => v + gaussian(rand));
  const fit = F.fitFactorModel(y, [
    { key: 'X', label: 'X', values: x },
    { key: 'Z', label: 'Z', values: z },
    { key: 'D', label: 'Dup', values: dup },
  ]);
  assert.equal(fit.ok, false);
  assert.equal(fit.dependent, 'Dup');
});

test('Newey–West SEs exceed classical SEs under positively autocorrelated errors', () => {
  const rand = lcg(11);
  const n = 1500;
  let ex = 0;
  let ee = 0;
  const x = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    ex = 0.8 * ex + gaussian(rand);
    ee = 0.8 * ee + gaussian(rand);
    x.push(ex);
    y.push(0.5 * ex + ee);
  }
  const fit = F.fitFactorModel(y, [{ key: 'X', label: 'X', values: x }]);
  assert.ok(fit.coefficients[1].se > 1.5 * fit.coefficients[1].seClassical);
  assert.ok(fit.durbinWatson < 1);
  assert.ok(fit.ljungBox.p < 0.001);
});

test('Huber estimate resists a single extreme outlier that drags OLS', () => {
  const rand = lcg(5);
  const n = 300;
  const x = Array.from({ length: n }, () => gaussian(rand));
  const y = x.map((v) => 1.0 * v + 0.1 * gaussian(rand));
  x[10] = 6;
  y[10] = -30;
  const fit = F.fitFactorModel(y, [{ key: 'X', label: 'X', values: x }]);
  const row = fit.coefficients[1];
  assert.ok(Math.abs(row.robustEstimate - 1) < 0.05);
  assert.ok(Math.abs(row.estimate - 1) > 0.2);
  assert.ok(F.outlierSensitiveCoefficients(fit).some((c) => c.key === 'X'));
  assert.equal(F.influentialDates({ dates: x.map((_, i) => `d${i}`), y }, fit, 1)[0].date, 'd10');
});

// ---------------------------------------------------------------------------
// Sample construction
// ---------------------------------------------------------------------------

test('buildFactorSample joins on dates, never on array position', () => {
  const ds = dates(120);
  const rand = lcg(9);
  const assetRet = Array.from({ length: 119 }, () => 0.01 * gaussian(rand));
  const benchRet = Array.from({ length: 119 }, () => 0.01 * gaussian(rand));
  const asset = barsFromReturns(ds, assetRet);
  const benchFull = barsFromReturns(ds, benchRet);
  // Benchmark misses day 50 (holiday mismatch) and starts 5 days later.
  const bench = benchFull.filter((_, i) => i !== 50 && i >= 5);

  const res = F.buildFactorSample({ AAA: asset, SPY: bench }, { asset: 'AAA', benchmark: 'SPY', factors: [], customTickers: [], returnType: 'simple' });
  assert.equal(res.ok, true);
  const s = res.sample;
  assert.equal(s.audit.commonPriceDates, 114);
  assert.equal(s.y.length, 113);
  assert.equal(s.audit.droppedAssetSessions, 6);
  assert.equal(s.audit.multiSessionReturns, 1);
  assert.equal(s.dates[0], ds[6]);
  // The return spanning the gap is the two-day compound return for both series.
  const gapIdx = s.dates.indexOf(ds[51]);
  approx(s.y[gapIdx], asset[51].close / asset[49].close - 1, 1e-12);
  approx(s.columns[0].values[gapIdx], benchFull[51].close / benchFull[49].close - 1, 1e-12);
  assert.ok(!s.dates.includes(ds[50]));
});

test('buildFactorSample builds long–short spreads and prefers adjusted closes', () => {
  const ds = dates(100);
  const rand = lcg(21);
  const mk = () => barsFromReturns(ds, Array.from({ length: 99 }, () => 0.01 * gaussian(rand)));
  const bars = { AAA: mk(), SPY: mk(), IWM: mk() };
  bars.AAA = bars.AAA.map((b) => ({ ...b, adjClose: b.close * 0.9 }));
  const res = F.buildFactorSample(bars, { asset: 'AAA', benchmark: 'SPY', factors: ['SIZE'], customTickers: [], returnType: 'simple' });
  assert.equal(res.ok, true);
  const size = res.sample.columns.find((c) => c.key === 'SIZE');
  approx(size.values[3], (bars.IWM[4].close / bars.IWM[3].close - 1) - (bars.SPY[4].close / bars.SPY[3].close - 1), 1e-12);
  assert.equal(res.sample.audit.series.find((s) => s.ticker === 'AAA').basis, 'adjusted');
  assert.equal(res.sample.audit.series.find((s) => s.ticker === 'SPY').basis, 'close');
});

test('buildFactorSample refuses missing series and short samples', () => {
  const ds = dates(40);
  const rand = lcg(2);
  const a = barsFromReturns(ds, Array.from({ length: 39 }, () => 0.01 * gaussian(rand)));
  const missing = F.buildFactorSample({ AAA: a, SPY: a }, { asset: 'AAA', benchmark: 'SPY', factors: ['VALUE'], customTickers: [], returnType: 'simple' });
  assert.equal(missing.ok, false);
  assert.deepEqual(missing.audit.missing, ['IWD', 'IWF']);
  const short = F.buildFactorSample({ AAA: a, SPY: a }, { asset: 'AAA', benchmark: 'SPY', factors: [], customTickers: [], returnType: 'simple' });
  assert.equal(short.ok, false);
  assert.match(short.reason, /at least 60/);
});

test('requiredTickers dedupes legs and keeps the asset first', () => {
  assert.deepEqual(F.requiredTickers({ asset: 'aapl', benchmark: 'SPY', factors: ['SIZE', 'MOMENTUM'], customTickers: ['spy'] }), ['AAPL', 'SPY', 'IWM', 'MTUM']);
});

// ---------------------------------------------------------------------------
// Attribution, rolling, comparison
// ---------------------------------------------------------------------------

test('attribution components sum exactly to the actual return path', () => {
  const ds = dates(300);
  const rand = lcg(31);
  const spyRet = Array.from({ length: 299 }, () => 0.01 * gaussian(rand));
  const iwmRet = spyRet.map((v) => v + 0.005 * gaussian(rand));
  const aRet = spyRet.map((v, i) => 0.0003 + 1.1 * v + 0.3 * (iwmRet[i] - v) + 0.008 * gaussian(rand));
  const bars = { AAA: barsFromReturns(ds, aRet), SPY: barsFromReturns(ds, spyRet), IWM: barsFromReturns(ds, iwmRet) };
  const res = F.buildFactorSample(bars, { asset: 'AAA', benchmark: 'SPY', factors: ['SIZE'], customTickers: [], returnType: 'simple' });
  const fit = F.fitSample(res.sample);
  const att = F.attributeReturns(res.sample, fit);
  const sum = att.components.reduce((a, c) => a + c.total, 0);
  approx(sum, att.actual.total, 1e-10);
  const last = att.components.reduce((a, c) => a + c.cumulative[c.cumulative.length - 1], 0);
  approx(last, att.actual.cumulative[att.actual.cumulative.length - 1], 1e-10);

  const roll = F.rollingFactorFit(res.sample, 126);
  assert.ok(roll.dates.length > 10);
  assert.equal(roll.dates[roll.dates.length - 1], res.sample.dates[res.sample.dates.length - 1]);
  const mkt = roll.series.find((s) => s.key === 'MKT');
  assert.ok(mkt.estimate.every((v) => v !== null && v > 0.6 && v < 1.6));

  const cmp = F.compareWithMarketOnly(res.sample, fit);
  assert.deepEqual(cmp.addedKeys, ['SIZE']);
  assert.ok(cmp.full.r2 >= cmp.restricted.r2);
});

test('plain-language readouts flag ranges that include zero', () => {
  const row = { estimate: 0.05, ciLow: -0.1, ciHigh: 0.2, se: 0.07, p: 0.5 };
  assert.match(F.describeExposure('AAA', { phrase: 'SPY rose 1%', kind: 'market' }, row, false), /no reliable response/);
  const strong = { estimate: 1.2, ciLow: 1.1, ciHigh: 1.3, se: 0.05, p: 0.0001 };
  assert.match(F.describeExposure('AAA', { phrase: 'SPY rose 1%', kind: 'market' }, strong, true), /typically rose 1\.20%, with the other factors held fixed/);
  assert.match(F.describeAlpha('AAA', { estimate: 0.0001, ciLow: -0.0002, ciHigh: 0.0004 }), /indistinguishable from zero/);
  assert.equal(F.evidenceLevel(0.003), 'strong');
  assert.equal(F.evidenceLevel(0.2), 'none');
});
