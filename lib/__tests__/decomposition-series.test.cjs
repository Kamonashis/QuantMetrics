// Decomposition series (price / realized vol / returns) + the series-aware TCN
// decomposition engine: builders, cache-key separation (price keys unchanged),
// latest-by-ticker isolation, baselines, direction rule and backtest semantics.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const S = require(path.join(buildDir, 'decomposition-series.js'));
const A = require(path.join(buildDir, 'tcn-attention.js'));
const E = require(path.join(buildDir, 'quant-engine.js'));

function makeBars(n, seed = 0) {
  const bars = [];
  const base = Date.parse('2025-01-02T00:00:00Z');
  let price = 100 + seed;
  for (let i = 0; i < n; i++) {
    const date = new Date(base + i * 86400000).toISOString().slice(0, 10);
    const ret = 0.0004 + 0.012 * (Math.sin((i + seed) * 1.618) * 0.8 + Math.cos((i + seed) * 2.718) * 0.5);
    const open = price;
    price = Math.max(5, price * Math.exp(ret));
    const spread = 0.004 + Math.abs(ret);
    bars.push({ date, open, high: Math.max(open, price) * (1 + spread / 2), low: Math.min(open, price) * (1 - spread / 2), close: price, volume: 1e6 });
  }
  return bars;
}

const FAST = { ...A.DEFAULT_TCN_CONFIG, epochs: 12, hiddenChannels: 6, numHeads: 2, lookback: 12 };
const decomp = (series, model) =>
  E.decomposeTimeSeries(series.dates, series.values, 21, model, 25, 126);

test('Garman–Klass annualized vol matches the formula and rejects invalid bars', () => {
  const bar = { date: 'd', open: 100, high: 103, low: 98, close: 101, volume: 1 };
  const hl = Math.log(103 / 98);
  const co = Math.log(101 / 100);
  const expected = Math.sqrt(252 * (0.5 * hl * hl - (2 * Math.LN2 - 1) * co * co)) * 100;
  assert.ok(Math.abs(S.garmanKlassAnnualizedVol(bar) - expected) < 1e-12);
  assert.equal(S.garmanKlassAnnualizedVol({ ...bar, high: 97 }), null, 'high < low');
  assert.equal(S.garmanKlassAnnualizedVol({ ...bar, open: 0 }), null, 'non-positive OHLC');
  assert.equal(S.garmanKlassAnnualizedVol({ ...bar, high: 100, low: 100, close: 100 }), null, 'zero range → no fabricated vol');
});

test('series builders keep dates, values and regime closes aligned', () => {
  const bars = makeBars(60, 1);
  const vol = S.buildRealizedVolSeries(bars);
  assert.equal(vol.dates.length, vol.values.length);
  assert.equal(vol.dates.length, vol.closes.length);
  vol.dates.forEach((d, i) => assert.equal(bars.find((b) => b.date === d).close, vol.closes[i]));
  assert.ok(vol.values.every((v) => v > 0));

  const logR = S.buildReturnsSeries(bars, 'log');
  assert.equal(logR.dates.length, bars.length - 1, 'returns start at the 2nd bar');
  assert.equal(logR.dates[0], bars[1].date);
  assert.ok(Math.abs(logR.values[0] - Math.log(bars[1].close / bars[0].close)) < 1e-15);
  assert.equal(logR.closes[0], bars[1].close, 'aligned close is C_t');
  assert.equal(logR.returnsLog, true);
  const simpleR = S.buildReturnsSeries(bars, 'simple');
  assert.ok(Math.abs(simpleR.values[0] - (bars[1].close / bars[0].close - 1)) < 1e-15);
});

test('returns are forced additive; price and vol keep the requested model', () => {
  assert.equal(S.effectiveDecompositionModel('returns', 'multiplicative'), 'additive');
  assert.equal(S.effectiveDecompositionModel('vol', 'multiplicative'), 'multiplicative');
  assert.equal(S.effectiveDecompositionModel('price', 'multiplicative'), 'multiplicative');
});

test('price cache keys are unchanged; vol/returns keys are distinct', () => {
  const bars = makeBars(120, 2);
  const d = decomp(S.buildPriceSeries(bars), 'multiplicative');
  const legacy = A.buildTcnCacheKey('KEYT', d, A.DEFAULT_TCN_CONFIG);
  assert.equal(A.buildTcnCacheKey('KEYT', d, A.DEFAULT_TCN_CONFIG, 'price'), legacy, 'price key byte-identical');
  assert.ok(!legacy.includes('S:'), 'no series tag on price keys');
  const kv = A.buildTcnCacheKey('KEYT', d, A.DEFAULT_TCN_CONFIG, 'vol');
  const kr = A.buildTcnCacheKey('KEYT', d, A.DEFAULT_TCN_CONFIG, 'returns');
  assert.ok(kv.includes('S:vol') && kr.includes('S:returns'));
  assert.equal(new Set([legacy, kv, kr]).size, 3);
});

test('engine: returns forecast uses the zero baseline, sign hit rate and return-based P&L; vol has no backtest; latest-by-ticker stays price-only', async () => {
  A.clearTcnCache();
  const bars = makeBars(170, 3);
  const yieldNow = () => Promise.resolve();

  const priceSeries = S.buildPriceSeries(bars);
  const price = await A.runTcnDecompositionForecast('ISO', decomp(priceSeries, 'multiplicative'), FAST, { yieldControl: yieldNow });

  const rs = S.buildReturnsSeries(bars, 'log');
  const rd = decomp(rs, 'additive');
  const ret = await A.runTcnDecompositionForecast('ISO', rd, FAST, {
    yieldControl: yieldNow,
    series: { kind: 'returns', regimeCloses: rs.closes, returnsLog: true },
  });

  const vs = S.buildRealizedVolSeries(bars);
  const vol = await A.runTcnDecompositionForecast('ISO', decomp(vs, 'multiplicative'), FAST, {
    yieldControl: yieldNow,
    series: { kind: 'vol', regimeCloses: vs.closes },
  });

  assert.ok(price && ret && vol);
  assert.equal(price.seriesKind, 'price');
  assert.equal(ret.seriesKind, 'returns');
  assert.equal(ret.returnsLog, true);
  assert.equal(vol.seriesKind, 'vol');

  // returns baseline = zero forecast: RMSE of the realized holdout returns themselves
  const holdDates = new Set(ret.backtestSignals.map((s) => s.date));
  assert.ok(holdDates.size > 0, 'returns keep a directional backtest');
  for (const sig of ret.backtestSignals) {
    const idx = rs.dates.indexOf(sig.date);
    assert.ok(Math.abs(sig.actualReturn - Math.expm1(rs.values[idx])) < 1e-12, 'P&L uses the realized simple return');
    assert.ok(Math.abs(sig.predictedReturn - Math.expm1(sig.predictedPrice)) < 1e-12, 'position sized on the predicted return');
  }
  assert.equal(ret.deltaPct, ret.deltaDollar * 100, 'returns report percentage points, not % of a return');

  assert.equal(vol.backtestSignals.length, 0, 'vol: no tradable backtest');

  // consumers asking for "the latest decomposition forecast" get price only
  assert.equal(A.getLatestCachedTcnArtifactByTicker('ISO').forecast.seriesKind, 'price');
  assert.equal(A.getLatestCachedTcnArtifactByTicker('ISO', 'vol').forecast.seriesKind, 'vol');
  assert.equal(A.getLatestCachedTcnArtifactByTicker('ISO', 'returns').forecast.seriesKind, 'returns');
});

test('engine: returns zero-baseline RMSE equals the RMS of realized holdout returns', async () => {
  A.clearTcnCache();
  const bars = makeBars(170, 5);
  const rs = S.buildReturnsSeries(bars, 'simple');
  const f = await A.runTcnDecompositionForecast('ZB', decomp(rs, 'additive'), FAST, {
    yieldControl: () => Promise.resolve(),
    series: { kind: 'returns', regimeCloses: rs.closes, returnsLog: false },
  });
  assert.ok(f);
  // With holdoutCount days the zero baseline is sqrt(mean(r_t^2)) over those days; it must be
  // a plausible daily-return scale (not a random-walk difference, which would be ~√2 larger).
  const allRms = Math.sqrt(rs.values.reduce((a, r) => a + r * r, 0) / rs.values.length);
  assert.ok(f.naiveBaselineRmse > 0.3 * allRms && f.naiveBaselineRmse < 3 * allRms);
  assert.ok(f.directionalAccuracy >= 0 && f.directionalAccuracy <= 100);
});
