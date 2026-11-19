// Unit tests for lib/forecast-trust.ts — walk-forward, no refit on the test tail.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const F = require(path.join(BUILD, 'forecast-trust.js'));

const clustered = (n) => {
  let v = 0.0002;
  let seed = 12345;
  const out = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const u = seed / 0x7fffffff - 0.5;
    v = 0.00001 + 0.08 * (u * u * 4) + 0.88 * v;
    out.push(u * Math.sqrt(v) * 2);
  }
  return out;
};

test('walkForwardForecastScores: scores all five models on a held-out tail', () => {
  const r = F.walkForwardForecastScores(clustered(400));
  assert.ok(r !== null);
  assert.equal(r.nTrain + r.nTest, 400);
  assert.ok(r.nTest >= 20);
  assert.equal(r.scores.length, 5);
  for (const s of r.scores) {
    assert.ok(s.qlike === null || Number.isFinite(s.qlike));
    assert.ok(s.n === 0 || s.n === r.nTest);
  }
  const ranked = r.scores.filter((s) => s.qlike !== null);
  for (let i = 1; i < ranked.length; i++) assert.ok(ranked[i - 1].qlike <= ranked[i].qlike);
});

test('walkForwardForecastScores: degrades honestly on short samples', () => {
  assert.equal(F.walkForwardForecastScores(new Array(60).fill(0.001)), null);
  assert.equal(F.walkForwardForecastScores([]), null);
});
