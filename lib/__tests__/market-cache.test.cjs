const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { sanitizeTickerSymbol } = require(path.join(buildDir, 'market-service.js'));
const {
  isEndDateUpToPresent,
  buildStreamTickerList,
  reconcileBarsWithLiveTick,
  pruneTickerCache,
} = require(path.join(buildDir, "market-cache.js"));

test('sanitizeTickerSymbol accepts valid equity/index symbols and rejects unsafe inputs', () => {
  assert.equal(sanitizeTickerSymbol(' aapl '), 'AAPL');
  assert.equal(sanitizeTickerSymbol('brk.b'), 'BRK.B');
  assert.equal(sanitizeTickerSymbol('^gspc'), '^GSPC');
  assert.equal(sanitizeTickerSymbol('btc-usd'), 'BTC-USD');

  assert.equal(sanitizeTickerSymbol(''), null);
  assert.equal(sanitizeTickerSymbol('../etc/passwd'), null);
  assert.equal(sanitizeTickerSymbol('AAPL?foo=bar'), null);
  assert.equal(sanitizeTickerSymbol('AAPL&x=1'), null);
  assert.equal(sanitizeTickerSymbol('A'.repeat(20)), null);
});

test('buildStreamTickerList prioritizes active ticker, deduplicates, sanitizes, and caps length', () => {
  const universe = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'AMZN', 'META', 'JPM'];
  const list = buildStreamTickerList(universe, 'amd', ['../bad', 'MSFT', 'NFLX'], 12);

  assert.equal(list[0], 'AMD');
  assert.equal(list.length, 12);
  assert.equal(new Set(list).size, 12);
  assert.ok(list.includes('NFLX'));
  assert.ok(!list.includes('../BAD'));

  // When activeTicker is already in universe, it moves to index 0 without duplicating
  const list2 = buildStreamTickerList(universe, 'NVDA', [], 12);
  assert.equal(list2[0], 'NVDA');
  assert.equal(list2.length, 10);
});

test('isEndDateUpToPresent handles empty, SSR placeholder, present, and historical dates', () => {
  assert.equal(isEndDateUpToPresent(undefined, '2026-11-21'), true);
  assert.equal(isEndDateUpToPresent('2026-11-08', '2026-11-21', '2026-11-08'), true);
  assert.equal(isEndDateUpToPresent('2026-11-21', '2026-11-21'), true);
  assert.equal(isEndDateUpToPresent('2026-11-25', '2026-11-21'), true);
  assert.equal(isEndDateUpToPresent('2025-12-31', '2026-11-21'), false);
});

test('reconcileBarsWithLiveTick patches same-day bar immutably, appends new session, and skips no-op/historical updates', () => {
  const existing = [
    { date: '2026-11-19', open: 348, high: 350, low: 347, close: 349.21, volume: 50_000_000 },
    { date: '2026-11-20', open: 352.87, high: 354.34, low: 340.56, close: 342.56, volume: 80_000_000 },
  ];

  // 1. Identical tick -> exact same array reference (prevents unnecessary re-renders)
  const sameTick = {
    ticker: 'AAPL',
    name: 'Apple Inc.',
    source: 'live_market',
    price: 342.56,
    prevClose: 349.21,
    changePct: -1.9,
    asOf: '2026-11-20',
    bar: { date: '2026-11-20', open: 352.87, high: 354.34, low: 340.56, close: 342.56, volume: 80_000_000 },
    fetchedAt: '2026-11-21T00:00:00Z',
  };
  const noop = reconcileBarsWithLiveTick(existing, sameTick, {
    allowPatchOrAppend: true,
    existingSource: 'live_market',
  });
  assert.equal(noop.changed, false);
  assert.equal(noop.bars, existing);

  // 2. Updated intraday price on same date -> immutable patch preserving higher/lower intraday extremes
  const movedTick = {
    ...sameTick,
    price: 355.1,
    bar: { date: '2026-11-20', open: 352.87, high: 355.1, low: 342.0, close: 355.1, volume: 82_000_000 },
  };
  const patched = reconcileBarsWithLiveTick(existing, movedTick, {
    allowPatchOrAppend: true,
    existingSource: 'live_market',
  });
  assert.equal(patched.changed, true);
  assert.notEqual(patched.bars, existing);
  assert.equal(patched.bars.length, 2);
  assert.equal(patched.bars[1].close, 355.1);
  assert.equal(patched.bars[1].high, 355.1);
  assert.equal(patched.bars[1].low, 340.56); // kept lower low
  assert.equal(patched.bars[1].volume, 82_000_000);
  assert.equal(existing[1].close, 342.56); // original untouched

  // 3. New trading session date -> appends cleanly
  const nextDayTick = {
    ...sameTick,
    asOf: '2026-11-23',
    price: 346.0,
    bar: { date: '2026-11-23', open: 343.0, high: 347.2, low: 342.5, close: 346.0, volume: 12_000_000 },
  };
  const appended = reconcileBarsWithLiveTick(existing, nextDayTick, {
    allowPatchOrAppend: true,
    existingSource: 'live_market',
  });
  assert.equal(appended.changed, true);
  assert.equal(appended.bars.length, 3);
  assert.equal(appended.bars[2].date, '2026-11-23');
  assert.equal(appended.bars[2].close, 346.0);

  // 4. Historical backtest slice -> never patched/appended
  const blockedHist = reconcileBarsWithLiveTick(existing, nextDayTick, {
    allowPatchOrAppend: false,
    existingSource: 'live_market',
  });
  assert.equal(blockedHist.changed, false);
  assert.equal(blockedHist.bars, existing);

  // 5. Never splice a live bar onto synthetic_gbm seed series
  const blockedSynth = reconcileBarsWithLiveTick(existing, nextDayTick, {
    allowPatchOrAppend: true,
    existingSource: 'synthetic_gbm',
  });
  assert.equal(blockedSynth.changed, false);
  assert.equal(blockedSynth.bars, existing);
});

test('pruneTickerCache evicts oldest unprotected tickers and never protected ones', () => {
  const protectedSet = new Set(['AAPL', 'SPY']);
  let cache = { AAPL: 1, SPY: 2 };
  for (const sym of ['A', 'B', 'C', 'D']) {
    cache = pruneTickerCache({ ...cache, [sym]: sym }, [sym], protectedSet, 2);
  }
  assert.deepEqual(Object.keys(cache).sort(), ['AAPL', 'C', 'D', 'SPY']);
});

test('pruneTickerCache treats a touched key as most recently used', () => {
  let cache = { A: 1, B: 2 };
  cache = pruneTickerCache({ ...cache, A: 10 }, ['A'], new Set(), 2); // A refreshed
  cache = pruneTickerCache({ ...cache, C: 3 }, ['C'], new Set(), 2);  // B is now oldest
  assert.deepEqual(Object.keys(cache), ['A', 'C']);
  assert.equal(cache.A, 10);
});

test('pruneTickerCache does not mutate its input', () => {
  const input = { A: 1, B: 2, C: 3 };
  const out = pruneTickerCache(input, ['C'], new Set(), 1);
  assert.deepEqual(Object.keys(input), ['A', 'B', 'C']);
  assert.deepEqual(Object.keys(out), ['C']);
});
