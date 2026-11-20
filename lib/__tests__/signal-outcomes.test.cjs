// Unit tests for lib/signal-outcomes.ts — pre-registered journal resolutions.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const S = require(path.join(BUILD, 'signal-outcomes.js'));

const bar = (date, close, high = close, low = close) => ({ date, close, high, low, open: close, volume: 1000 });

function series(closes) {
  return closes.map((c, i) => bar(`2026-01-${String(i + 1).padStart(2, '0')}`, c, c * 1.01, c * 0.99));
}

const record = (overrides = {}) => ({
  date: '2026-01-03', symbol: 'AAPL', setup: 'premium-harvest', state: 'ACTIONABLE',
  iv30: 0.3, breakeven: 0.35, spot: 100, direction: 'none', wall: null, flip: null,
  impliedPct: 3, horizon: 21, edgeEstimate: '+12 bp²', pairLong: null, pairShort: null,
  recordedAt: '2026-01-03T00:00:00.000Z', ...overrides,
});

test('premium-harvest: resolves on forward RV20 vs IV30 at entry', () => {
  const quiet = series(Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i) * 0.2));
  const win = S.resolveSignal(record(), { AAPL: quiet });
  assert.equal(win.outcome, 'WIN');
  assert.ok(win.forwardValue < 0);
  const wild = series(Array.from({ length: 40 }, (_, i) => 100 + (i % 2 ? 8 : -8)));
  const loss = S.resolveSignal(record(), { AAPL: wild });
  assert.equal(loss.outcome, 'LOSS');
});

test('long-gamma: resolves on forward RV5 vs recorded breakeven', () => {
  const bars = series(Array.from({ length: 40 }, (_, i) => 100 + (i % 2 ? 6 : -6)));
  const r = S.resolveSignal(record({ setup: 'long-gamma' }), { AAPL: bars });
  assert.equal(r.outcome, 'WIN');
});

test('directional-trap: wall before flip wins, flip first loses, missing window unresolved', () => {
  const bars = series(Array.from({ length: 20 }, (_, i) => 100 + i * 0.1));
  bars[4] = bar(bars[4].date, 100.4, 107, 99);
  const up = S.resolveSignal(record({ setup: 'directional-trap', direction: 'up', wall: 106, flip: 98 }), { AAPL: bars });
  assert.equal(up.outcome, 'WIN');
  bars[4] = bar(bars[4].date, 100.4, 101, 97);
  const down = S.resolveSignal(record({ setup: 'directional-trap', direction: 'up', wall: 106, flip: 98 }), { AAPL: bars });
  assert.equal(down.outcome, 'LOSS');
  const short = S.resolveSignal(record({ setup: 'directional-trap', direction: 'up', wall: 106, flip: 98 }), { AAPL: bars.slice(0, 3) });
  assert.equal(short.outcome, 'UNRESOLVED');
});

test('event-straddle and cross-sectional resolve with pre-registered rules', () => {
  const flat = series(Array.from({ length: 40 }, (_, i) => 100 + i * 0.01));
  const ev = S.resolveSignal(record({ setup: 'event-straddle', impliedPct: 1, horizon: 5 }), { AAPL: flat });
  assert.equal(ev.outcome, 'LOSS');
  const wild = series(Array.from({ length: 40 }, (_, i) => 100 + (i % 2 ? 9 : -9)));
  const xs = S.resolveSignal(record({ setup: 'cross-sectional', pairLong: 'AAPL', pairShort: 'MSFT' }), { AAPL: wild, MSFT: flat });
  assert.equal(xs.outcome, 'WIN');
});

test('summarizeHitRates withholds small samples and counts resolved only', () => {
  const quiet = series(Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i) * 0.2));
  const rows = Array.from({ length: 11 }, (_, i) => record({ date: `2026-01-${String((i % 20) + 1).padStart(2, '0')}` }));
  const rate = S.summarizeHitRates(rows, { AAPL: quiet })[0];
  assert.equal(rate.setup, 'premium-harvest');
  assert.ok(rate.n >= 10);
  assert.equal(rate.hitRate, 1);
  const small = S.summarizeHitRates(rows.slice(0, 3), { AAPL: quiet })[0];
  assert.equal(small.hitRate, null);
});
