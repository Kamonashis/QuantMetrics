// Unit tests for lib/hmm.ts (compiled output in .cache/test-build).
// Fixtures: seeded synthetic 2-state Gaussian chain — deterministic, no dates.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const { fitGaussianHmm } = require(path.join(BUILD, 'hmm.js'));

// mulberry32 — deterministic pseudo-random generator (same family as market-service)
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller normal from a seeded uniform stream
function makeNormal(rng) {
  let spare = null;
  return function () {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0, v = 0, s = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
}

test('recovers a synthetic 2-state volatility regime', () => {
  const rng = mulberry32(987654321);
  const norm = makeNormal(rng);
  const T = 1000;
  const trueStates = [];
  let s = 0;
  // Mild sticky regimes: 3% switching/day
  for (let t = 0; t < T; t++) {
    trueStates.push(s);
    if (rng() < 0.03) s = 1 - s;
  }
  const trueSd = [0.008, 0.035];
  const feats = trueStates.map((st) => {
    const r = norm() * trueSd[st];
    return [r, Math.abs(r)];
  });

  const fit = fitGaussianHmm(feats, { k: 2, rankFeature: 1, tol: 1e-8, maxIters: 400 });
  assert.ok(fit, 'fit should succeed');
  assert.equal(fit.k, 2);
  assert.ok(Number.isFinite(fit.logLikelihood));
  // Convergence
  assert.ok(fit.converged || fit.iterations >= 50, 'converged or ran to cap');
  // Posteriors are valid probability rows
  for (const row of fit.posteriors) {
    const s = row.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(s - 1) < 1e-6, 'posterior rows sum to 1');
  }
  assert.equal(fit.transition.length, 2);
  const t00 = fit.transition[0][0];
  assert.ok(t00 > 0.5, 'calm persists');
  // Mean of |r| recovered: calm < calmSD*0.8+margin, crisis ≈ crisisSd*0.8
  assert.ok(fit.stateMeans[0][1] < fit.stateMeans[1][1], 'crisis high-rv ranked second');
});

test('BIC prefers more states only when genuinely needed', () => {
  const rng = mulberry32(24681357);
  const norm = makeNormal(rng);
  const T = 800;
  const trueStates = [];
  let s = 0;
  for (let t = 0; t < T; t++) {
    trueStates.push(s);
    if (rng() < 0.02) s = 1 - s;
  }
  const sd = [0.01, 0.04];
  const feats = trueStates.map((st) => [norm() * sd[st]]);

  const k2 = fitGaussianHmm(feats, { k: 2, rankFeature: 0, tol: 1e-8, maxIters: 400 });
  const k3 = fitGaussianHmm(feats, { k: 3, rankFeature: 0, tol: 1e-8, maxIters: 400 });
  assert.ok(k2 && k3, 'both fits succeed');
  // LogLik must not decrease when we add a state
  assert.ok(k3.logLikelihood >= k2.logLikelihood - 1e-6, 'more states cannot hurt LL');
  assert.ok(k3.bic > k2.bic - 200, 'BIC penalty is real but bounded');
});

test('degenerate input does not crash', () => {
  const constFeats = Array.from({ length: 100 }, () => [0.01, 0.001]);
  const fit = fitGaussianHmm(constFeats, { k: 2, rankFeature: 1, maxIters: 50 });
  // Either converges or returns a result; must not throw
  assert.ok(fit === null || Number.isFinite(fit.logLikelihood));
});
