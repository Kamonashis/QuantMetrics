// Unit tests for lib/analysis-engine.ts (compiled output in .cache/test-build).
// Fixtures: deterministic — no network, no clock.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const A = require(path.join(BUILD, 'analysis-engine.js'));

// Deterministic pseudo-random generator (LCG) + Box–Muller for normal-ish variates
function makeLcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
function normalSampler(seed) {
  const rand = makeLcg(seed);
  return () => {
    const u1 = Math.max(1e-9, rand());
    const u2 = rand();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  };
}

test('computeInnovationDiagnostics: null below 40 aligned observations', () => {
  const r = new Array(30).fill(0.001);
  const cv = new Array(30).fill(0.0004);
  assert.equal(A.computeInnovationDiagnostics(r, cv), null);
});

test('computeInnovationDiagnostics: well-formed output on an iid-like series', () => {
  const randn = normalSampler(1234);
  const n = 250;
  const returns = Array.from({ length: n }, () => randn() * 0.02);
  const cv = new Array(n).fill(0.0004); // sd 0.02
  const d = A.computeInnovationDiagnostics(returns, cv);
  assert.ok(d !== null);
  assert.ok(d.hurstExponent > 0 && d.hurstExponent < 1);
  assert.ok(['geometric brownian motion (h = 0.5)', 'mean-reverting (h < 0.5)', 'trending / persistent (h > 0.5)']
    .includes(d.hurstRegime.toLowerCase()));
  assert.equal(d.varianceRatios.length, 4);
  assert.ok(d.vrMinZ <= d.varianceRatios[0].zStat);
  assert.ok(d.adfPValue >= 0 && d.adfPValue <= 1);
  assert.ok(Number.isFinite(d.skewness));
  assert.ok(Number.isFinite(d.excessKurtosis));
});

test('computeInnovationDiagnostics: zero/negative conditional variance floor does not NaN', () => {
  const randn = normalSampler(99);
  const n = 60;
  const returns = Array.from({ length: n }, () => randn() * 0.01);
  const cv = new Array(n).fill(0);
  const d = A.computeInnovationDiagnostics(returns, cv);
  assert.ok(d !== null);
  assert.ok(Number.isFinite(d.hurstExponent));
  assert.ok(d.varianceRatios.every((v) => Number.isFinite(v.zStat)));
});

test('computeInnovationDiagnostics: persistent residuals read as more persistent than iid', () => {
  const randn = normalSampler(777);
  const n = 300;
  // AR(1) with φ = 0.7 — genuine serial dependence
  let x = 0;
  const ar = [];
  for (let i = 0; i < n; i++) {
    x = 0.7 * x + randn();
    ar.push(x * 0.01);
  }
  const iidSampler = normalSampler(888);
  const iid = Array.from({ length: n }, () => iidSampler() * 0.01);
  const cv = new Array(n).fill(0.0001);
  const dAr = A.computeInnovationDiagnostics(ar, cv);
  const dIid = A.computeInnovationDiagnostics(iid, cv);
  assert.ok(dAr.hurstExponent > dIid.hurstExponent,
    `expected AR(1) H (${dAr.hurstExponent.toFixed(3)}) > iid H (${dIid.hurstExponent.toFixed(3)})`);
});
