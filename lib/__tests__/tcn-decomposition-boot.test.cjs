// App-load TCN decomposition bootstrap: cache-first lookup, default-parameter
// fit on a miss, cache-key parity with the Analysis view, and in-flight
// coalescing so the bootstrap and the chart never double-train.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const Q = require(path.join(buildDir, 'tcn-training-queue.js'));
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
    bars.push({ date, open, high: Math.max(open, price) * 1.004, low: Math.min(open, price) * 0.996, close: price, volume: 1e6 });
  }
  return bars;
}

const PARAMS = { startDate: '2025-01-10', endDate: '2025-12-31', period: 21, model: 'multiplicative' };
const yieldNow = () => Promise.resolve();

test('boot decomposition is byte-identical to the Analysis view decomposition (same cache key)', () => {
  const bars = makeBars(160, 1);
  const boot = Q.buildBootDecomposition(bars, PARAMS);
  // Analysis view: filter by dates, then decomposeTimeSeries(dates, prices, period, model, band.min, band.max)
  const filtered = bars.filter((b) => b.date >= PARAMS.startDate && b.date <= PARAMS.endDate);
  const band = E.DEFAULT_DECOMPOSITION_CYCLE_BAND;
  const view = E.decomposeTimeSeries(
    filtered.map((b) => b.date), filtered.map((b) => b.close), PARAMS.period, PARAMS.model, band.min, band.max
  );
  assert.equal(
    A.buildTcnCacheKey('BOOTDEC', boot, A.DEFAULT_TCN_CONFIG),
    A.buildTcnCacheKey('BOOTDEC', view, A.DEFAULT_TCN_CONFIG)
  );
});

test('too little history after the date filter yields no decomposition and no fit', async () => {
  const bars = makeBars(30, 2);
  assert.equal(Q.buildBootDecomposition(bars, PARAMS), null);
  const res = await Q.runTcnDecompositionBootstrap({
    ticker: 'BOOTSHORT', bars, params: PARAMS, yieldControl: yieldNow, isCancelled: () => false,
  });
  assert.deepEqual(res, { forecast: null, fromCache: false });
});

test('cache first: a miss trains with default parameters and caches; the next call is a cache hit', async () => {
  A.clearTcnCache();
  const bars = makeBars(140, 3);
  const ctx = { ticker: 'BOOTMISS', bars, params: PARAMS, yieldControl: yieldNow, isCancelled: () => false };

  const first = await Q.runTcnDecompositionBootstrap(ctx);
  assert.ok(first.forecast, 'a miss produces a trained forecast');
  assert.equal(first.fromCache, false);
  const key = A.buildTcnCacheKey('BOOTMISS', Q.buildBootDecomposition(bars, PARAMS), A.DEFAULT_TCN_CONFIG);
  assert.ok(A.getCachedTcnArtifact(key), 'the fit is cached under the default-config key');

  const second = await Q.runTcnDecompositionBootstrap(ctx);
  assert.equal(second.fromCache, true, 'second lookup is answered from cache');
  assert.deepEqual(second.forecast, first.forecast);
});

test('a cancelled bootstrap fit returns null and caches nothing', async () => {
  A.clearTcnCache();
  const bars = makeBars(140, 4);
  const res = await Q.runTcnDecompositionBootstrap({
    ticker: 'BOOTCANCEL', bars, params: PARAMS, yieldControl: yieldNow, isCancelled: () => true,
  });
  assert.equal(res.forecast, null);
  const key = A.buildTcnCacheKey('BOOTCANCEL', Q.buildBootDecomposition(bars, PARAMS), A.DEFAULT_TCN_CONFIG);
  assert.equal(A.getCachedTcnArtifact(key), null);
});

test('concurrent same-key decomposition fits join one run', async () => {
  A.clearTcnCache();
  const decomp = Q.buildBootDecomposition(makeBars(140, 5), PARAMS);
  const p1 = A.runTcnDecompositionForecast('BOOTJOIN', decomp, A.DEFAULT_TCN_CONFIG, { yieldControl: yieldNow });
  const p2 = A.runTcnDecompositionForecast('BOOTJOIN', decomp, A.DEFAULT_TCN_CONFIG, { yieldControl: yieldNow });
  assert.strictEqual(p1, p2, 'the second caller receives the in-flight promise');
  const r = await p1;
  assert.ok(r);
});
