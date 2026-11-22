const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '..', '..', '.cache', 'test-build');
const {
  HOME_UNIVERSE_SYMBOLS,
  computeActiveCausalHmmSeries,
  computeTrailingRv,
  computeHomeUniverseCard,
  buildSparklineSvgPath,
} = require(path.join(buildDir, 'home-universe.js'));
const { generateRealisticStockHistory } = require(path.join(buildDir, 'market-service.js'));

test('HOME_UNIVERSE_SYMBOLS defines 10 distinct benchmark symbols', () => {
  assert.equal(HOME_UNIVERSE_SYMBOLS.length, 10);
  const unique = new Set(HOME_UNIVERSE_SYMBOLS.map((s) => s.ticker));
  assert.equal(unique.size, 10);
});

test('computeActiveCausalHmmSeries aligns points and produces valid causal regime probabilities', () => {
  const bars = generateRealisticStockHistory('AAPL', 300);
  const res = computeActiveCausalHmmSeries(bars, 252);
  assert.ok(res !== null);
  assert.equal(res.points.length, 252);
  assert.equal(res.points[res.points.length - 1].date, bars[bars.length - 1].date);
  assert.equal(res.points[res.points.length - 1].close, bars[bars.length - 1].close);
  assert.ok([0, 1, 2].includes(res.latestState));
  assert.ok(res.latestProb > 0 && res.latestProb <= 1.0001);
  const sum = res.probs[0] + res.probs[1] + res.probs[2];
  assert.ok(Math.abs(sum - 1) < 1e-4);
});

test('computeActiveCausalHmmSeries returns null for short histories', () => {
  const shortBars = generateRealisticStockHistory('AAPL', 30);
  assert.equal(computeActiveCausalHmmSeries(shortBars), null);
});

test('computeHomeUniverseCard prioritizes liveQuote and respects pending flag honesty', () => {
  const bars = generateRealisticStockHistory('NVDA', 180);
  const meta = HOME_UNIVERSE_SYMBOLS.find((m) => m.ticker === 'NVDA');

  const liveCard = computeHomeUniverseCard(
    meta,
    bars,
    { price: 142.5, changePct: 2.15, asOf: '2026-11-08' },
    { symbol: 'NVDA', price: 130.0, changePct: -1.0, pending: true }
  );
  assert.equal(liveCard.price, 142.5);
  assert.equal(liveCard.changePct, 2.15);
  assert.equal(liveCard.pending, false);
  assert.equal(liveCard.sparkline.length, 60);
  assert.ok(liveCard.rv20 !== null && liveCard.rv20 > 0);
  assert.ok([0, 1, 2].includes(liveCard.regimeState));

  const pendingCard = computeHomeUniverseCard(meta, bars, null, {
    symbol: 'NVDA',
    price: 130.0,
    changePct: -1.0,
    pending: true,
  });
  assert.equal(pendingCard.pending, true);
});

test('buildSparklineSvgPath returns valid SVG path strings', () => {
  const { linePath, areaPath } = buildSparklineSvgPath([100, 102, 101, 105, 108], 120, 32);
  assert.ok(linePath.startsWith('M0.0,'));
  assert.ok(areaPath.endsWith('Z'));
  const empty = buildSparklineSvgPath([100], 120, 32);
  assert.equal(empty.linePath, '');
  assert.equal(empty.areaPath, '');
});

test('computeTrailingRv returns annualized realized volatility', () => {
  const bars = generateRealisticStockHistory('SPY', 80);
  const rv = computeTrailingRv(bars, 20);
  assert.ok(rv !== null && rv > 0.02 && rv < 2.0);
});
