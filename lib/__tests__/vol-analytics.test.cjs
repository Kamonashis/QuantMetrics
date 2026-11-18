// Unit tests for lib/vol-analytics.ts (compiled output in .cache/test-build).
// Fixtures: deterministic — no network, no clock.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const V = require(path.join(BUILD, 'vol-analytics.js'));

const approx = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `expected ${b}, got ${a}`);
const deterministicReturns = (n) =>
  Array.from({ length: n }, (_, i) => 0.012 * Math.sin(i / 7) + 0.004 * Math.cos(i / 3));

test('computeRealizedVolCone: quantile ordering and sample counts', () => {
  const cone = V.computeRealizedVolCone(deterministicReturns(300), [10, 20, 30, 60, 90], 252);
  assert.deepEqual(cone.map((b) => b.dte), [10, 20, 30, 60, 90]);
  for (const b of cone) {
    assert.ok(b.n >= 30);
    assert.equal(b.samples.length, b.n);
    assert.ok(b.min <= b.p10 && b.p10 <= b.p25 && b.p25 <= b.median);
    assert.ok(b.median <= b.p75 && b.p75 <= b.p90 && b.p90 <= b.max);
    assert.ok(b.current > 0 && Number.isFinite(b.current));
  }
});

test('computeRealizedVolCone: constant returns collapse the distribution', () => {
  const cone = V.computeRealizedVolCone(new Array(200).fill(0.01), [10, 20], 252);
  assert.equal(cone.length, 2);
  const expected = Math.sqrt(0.0001 * 252); // √252 annualization of a constant daily variance
  for (const b of cone) {
    approx(b.min, expected);
    approx(b.median, expected);
    approx(b.max, expected);
    approx(b.current, expected);
  }
});

test('computeRealizedVolCone: degrades on short history and honours historyDays', () => {
  assert.deepEqual(V.computeRealizedVolCone(new Array(40).fill(0.01), [90], 252), []);
  const cone = V.computeRealizedVolCone(deterministicReturns(300), [10], 50);
  assert.equal(cone.length, 1);
  assert.equal(cone[0].n, 50);
});

test('ivAtConeTenor: exact, total-variance interpolation, clamped edges', () => {
  const term = [{ dte: 10, iv: 0.2 }, { dte: 30, iv: 0.22 }];
  approx(V.ivAtConeTenor(term, 10), 0.2);
  approx(V.ivAtConeTenor(term, 30), 0.22);
  approx(V.ivAtConeTenor(term, 5), 0.2);
  approx(V.ivAtConeTenor(term, 90), 0.22);
  const w10 = 0.2 * 0.2 * (10 / 365);
  const w30 = 0.22 * 0.22 * (30 / 365);
  approx(V.ivAtConeTenor(term, 20), Math.sqrt(((w10 + w30) / 2) / (20 / 365)));
  assert.equal(V.ivAtConeTenor([], 20), null);
  assert.equal(V.ivAtConeTenor([{ dte: 10, iv: null }], 20), null);
});

test('coneVerdict: thresholds, percentile, insufficient data', () => {
  const band = {
    dte: 20, min: 0.1, p10: 0.12, p25: 0.15, median: 0.18, p75: 0.21,
    p90: 0.25, max: 0.3, current: 0.18, n: 7,
    samples: [0.1, 0.12, 0.15, 0.18, 0.18, 0.21, 0.25, 0.3],
  };
  assert.equal(V.coneVerdict(0.26, band).verdict, 'RICH');
  assert.equal(V.coneVerdict(0.22, band).verdict, 'rich');
  assert.equal(V.coneVerdict(0.18, band).verdict, 'in range');
  assert.equal(V.coneVerdict(0.14, band).verdict, 'cheap');
  assert.equal(V.coneVerdict(0.11, band).verdict, 'CHEAP');
  assert.equal(V.coneVerdict(null, band).verdict, 'insufficient-data');
  assert.equal(V.coneVerdict(0.18, null).verdict, 'insufficient-data');
  const p = V.coneVerdict(0.18, band).percentile;
  assert.ok(p !== null && p > 0 && p < 100);
});
