// Unit tests for the TCN bootstrap trainer: pure policy decisions, the
// shared head sequencer, and the engine-level in-flight coalescing that
// keeps a startup bootstrap and a retrain button from double-training.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const P = require(path.join(buildDir, 'tcn-bootstrap-policy.js'));
const Q = require(path.join(buildDir, 'tcn-training-queue.js'));
const M = require(path.join(buildDir, 'tcn-multi-domain.js'));

// Deterministic synthetic OHLCV bars (same generator as the multi-domain tests)
function makeSyntheticBars(n = 240, seedOffset = 0) {
  const bars = [];
  const baseDate = new Date('2025-11-08T00:00:00Z');
  let price = 150 + seedOffset * 10;
  let sigma = 0.013;
  for (let i = 0; i < n; i++) {
    const d = new Date(baseDate.getTime() + i * 86400 * 1000);
    const date = d.toISOString().slice(0, 10);
    const z =
      Math.sin((i + 1 + seedOffset) * 1.6180339) * 0.85 +
      Math.cos((i + 1 + seedOffset) * 2.7182818) * 0.65;
    const shock = i % 47 === 20 ? -2.6 : z;
    const ret = 0.0003 + sigma * shock;
    sigma = Math.sqrt(0.00001 + 0.12 * ret * ret + 0.84 * sigma * sigma);
    const open = price;
    const close = Math.max(5, open * Math.exp(ret));
    const spread = Math.max(0.004, Math.abs(ret) * 1.25 + sigma * 0.6);
    const high = Math.max(open, close) * (1 + spread * 0.5);
    const low = Math.min(open, close) * Math.max(0.5, 1 - spread * 0.5);
    bars.push({ date, open, high, low, close, volume: 1_000_000 });
    price = close;
  }
  return bars;
}

const FAST_CFG = { epochs: 12, hiddenChannels: 6, numHeads: 2, lookback: 18 };

test('bootstrap policy: one fit per ticker per session, live bars only, defers instead of guessing', () => {
  const registry = new Set(['AAPL']);
  const base = {
    ticker: ' nvda ',
    bootstrapped: registry,
    barsCount: 250,
    barsSource: 'live_market',
    regressionInputsReady: false,
    trainingInFlight: false,
  };
  assert.deepEqual(P.nextTcnBootstrapAction(base), {
    kind: 'train',
    ticker: 'NVDA',
    heads: ['volatility', 'risk_var'],
  });
  assert.deepEqual(
    P.nextTcnBootstrapAction({ ...base, ticker: 'nvda', regressionInputsReady: true }).heads,
    ['volatility', 'risk_var', 'regression']
  );
  // once per session: completion (or a cache-only pass) registers the ticker
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, ticker: 'AAPL' }), {
    kind: 'skip',
    ticker: 'AAPL',
    reason: 'already-bootstrapped',
  });
  // an in-flight fit wins over every other decision for that ticker
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, trainingInFlight: true }), {
    kind: 'skip',
    ticker: 'NVDA',
    reason: 'training-in-flight',
  });
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, barsCount: 69 }), {
    kind: 'defer',
    ticker: 'NVDA',
    reason: 'insufficient-bars',
  });
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, barsSource: 'synthetic_gbm' }), {
    kind: 'defer',
    ticker: 'NVDA',
    reason: 'synthetic-data',
  });
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, barsSource: null, ticker: 'MSFT' }), {
    kind: 'defer',
    ticker: 'MSFT',
    reason: 'awaiting-live-bars',
  });
  assert.deepEqual(P.nextTcnBootstrapAction({ ...base, ticker: '' }), {
    kind: 'defer',
    ticker: '',
    reason: 'no-ticker',
  });
});

test('head sequencer: strict order, graceful regression skip, cancellation between heads', async () => {
  M.clearTcnMultiDomainCache();
  const bars = makeSyntheticBars(150, 3);
  const ctx = (extra = {}) => ({
    ticker: 'BOOTQ',
    bars,
    allStockBars: {}, // no peer series → regression head must resolve null, not throw
    forceRetrain: true,
    yieldControl: () => Promise.resolve(),
    isCancelled: () => false,
    config: FAST_CFG,
    ...extra,
  });

  const events = [];
  const volAndRisk = await Q.runTcnHeadSequence(['volatility', 'risk_var', 'regression'], ctx(), {
    onHeadStart: (h) => events.push(`start:${h}`),
    onHeadDone: (h, r) => events.push(`done:${h}:${r === null ? 'null' : 'ok'}`),
  });
  assert.equal(volAndRisk, undefined);
  assert.deepEqual(events, [
    'start:volatility',
    'done:volatility:ok',
    'start:risk_var',
    'done:risk_var:ok',
    'start:regression',
    'done:regression:null',
  ]);

  // cancellation lands between heads: the next head never starts (cache hits
  // from the run above keep this instant)
  const seen = [];
  let halt = false;
  await Q.runTcnHeadSequence(
    ['volatility', 'risk_var'],
    ctx({ forceRetrain: false, isCancelled: () => halt }),
    {
      onHeadStart: (h) => seen.push(h),
      onHeadDone: (h) => {
        if (h === 'volatility') halt = true;
      },
    }
  );
  assert.deepEqual(seen, ['volatility'], 'risk head must not start after cancellation');
});

test('engine coalescing: concurrent same-key fits join one run; different data fits separately', async () => {
  M.clearTcnMultiDomainCache();
  const bars = makeSyntheticBars(150, 7);

  const first = M.runTcnVolatilityForecast('BOOTC', bars, { config: FAST_CFG });
  const joined = M.runTcnVolatilityForecast('BOOTC', bars, { config: FAST_CFG });
  const forcedJoin = M.runTcnVolatilityForecast('BOOTC', bars, {
    config: FAST_CFG,
    forceRetrain: true,
  });
  const r1 = await first;
  const r2 = await joined;
  const r3 = await forcedJoin;
  assert.ok(r1 && r1.cacheHit === false, 'first run trains fresh');
  assert.strictEqual(r2, r1, 'concurrent non-forced caller joins the in-flight fit');
  assert.strictEqual(r3, r1, 'concurrent forced caller joins the same in-flight fit');
  const r4 = await M.runTcnVolatilityForecast('BOOTC', bars, { config: FAST_CFG });
  assert.ok(r4 && r4.cacheHit === true, 'after completion the cache serves the artifact');

  // different data (signature) → independent fits, never joined
  const otherBars = makeSyntheticBars(150, 11);
  const [a, b] = await Promise.all([
    M.runTcnVolatilityForecast('BOOTD', bars, { config: FAST_CFG }),
    M.runTcnVolatilityForecast('BOOTD', otherBars, { config: FAST_CFG }),
  ]);
  assert.ok(a && b && a !== b && a.cacheHit === false && b.cacheHit === false);
});

test('boot update subscription: listeners receive only what they subscribe to', () => {
  const seen = [];
  const off = Q.subscribeTcnBootUpdates((ticker, head) => seen.push([ticker, head]));
  Q.notifyTcnBootUpdate('AAPL', 'volatility');
  off();
  Q.notifyTcnBootUpdate('AAPL', 'risk_var');
  Q.notifyTcnBootUpdate('MSFT', 'regression');
  assert.deepEqual(seen, [['AAPL', 'volatility']]);
});
