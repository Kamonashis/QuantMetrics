// Unit tests for lib/rnd.ts (compiled output in .cache/test-build).
// Anchor: a constant-σ Black–Scholes call curve must reproduce the analytic
// lognormal risk-neutral density — the only closed-form B–L fixture we have.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const O = require(path.join(BUILD, 'options.js'));
const R = require(path.join(BUILD, 'rnd.js'));

const SPOT = 100;
const VOL = 0.25;
const T = 30 / 365;
const RATE = 0.04;
const DIV = 0.0;
const FWD = SPOT * Math.exp((RATE - DIV) * T);

/** Black–Scholes call price via lib/options (the shared BS engine). */
function bsCall(K) {
  return O.bsPrice({ spot: SPOT, strike: K, t: T, rate: RATE, dividendYield: DIV, vol: VOL, type: 'call' });
}

/** Analytic risk-neutral density of S_T: lognormal with mean FWD. */
function analyticRnd(K) {
  const sigma = Math.sqrt(VOL * VOL * T);
  const m = Math.log(FWD) - 0.5 * sigma * sigma;
  return Math.exp(-((Math.log(K) - m) ** 2) / (2 * sigma * sigma)) / (K * sigma * Math.sqrt(2 * Math.PI));
}

/** Full synthetic slice: calls above F, puts below (dense, clean mids). */
function slicePoints({ minK = 60, maxK = 150, step = 5, spread = 0.001 } = {}) {
  const pts = [];
  for (let K = minK; K <= maxK + 1e-9; K += step) {
    const call = bsCall(K);
    const put = O.bsPrice({ spot: SPOT, strike: K, t: T, rate: RATE, dividendYield: DIV, vol: VOL, type: 'put' });
    pts.push({ strike: K, mid: call * (1 + spread), relSpread: 0.002, type: 'call' });
    pts.push({ strike: K, mid: put * (1 + spread), relSpread: 0.002, type: 'put' });
  }
  return pts;
}

test('buildCallCurve: direct calls above F, parity-synthesized below F', () => {
  const curve = R.buildCallCurve(slicePoints(), { forward: FWD, t: T, rate: RATE });
  assert.ok(curve, 'curve should build');
  const below = curve.points.filter((p) => p.strike < FWD);
  const above = curve.points.filter((p) => p.strike >= FWD);
  assert.ok(below.every((p) => p.parity), 'below-F points must come from parity');
  assert.ok(above.some((p) => !p.parity), 'above-F points must use direct calls');
  assert.equal(curve.dropped, 0);
  assert.ok(curve.nParity > 0 && curve.nDirect > 0);
  // parity identity: C = P + (F − K)e^{−rT}
  const K = curve.points.find((p) => p.strike < FWD).strike;
  const p = slicePoints().find((q) => q.strike === K && q.type === 'put');
  const expected = p.mid + Math.exp(-RATE * T) * (FWD - K);
  assert.ok(Math.abs(curve.points.find((q) => q.strike === K).call - expected) < 1e-12);
});

test('buildCallCurve: thin curve returns null instead of a fabricated density', () => {
  const thin = slicePoints().filter((p) => p.type === 'call' && p.strike >= FWD).slice(0, 3);
  assert.equal(R.buildCallCurve(thin, { forward: FWD, t: T, rate: RATE }), null);
  const junk = slicePoints().map((p) => ({ ...p, mid: 0 }));
  assert.equal(R.buildCallCurve(junk, { forward: FWD, t: T, rate: RATE }), null);
});

test('naturalCubicSpline: interpolates exactly; a straight line differentiates to zero', () => {
  const xs = [1, 2, 3, 4, 5, 6];
  const ys = xs.map((x) => 2 * x + 7);
  const s = R.naturalCubicSpline(xs, ys);
  for (const x of xs) assert.ok(Math.abs(s.value(x) - (2 * x + 7)) < 1e-12);
  for (const x of [1, 2.5, 3.5, 6]) assert.ok(Math.abs(s.second(x)) < 1e-9, 'line has zero second derivative');
  // nonlinear data: interpolates the nodes, second derivative is finite and positive here (convex data)
  const ys2 = xs.map((x) => x * x);
  const s2 = R.naturalCubicSpline(xs, ys2);
  for (const x of xs) assert.ok(Math.abs(s2.value(x) - x * x) < 1e-9);
  assert.ok(s2.second(3.5) > 0);
});

test('impliedDensityFromCalls: recovers the analytic lognormal density', () => {
  const curve = R.buildCallCurve(slicePoints(), { forward: FWD, t: T, rate: RATE });
  const rnd = R.impliedDensityFromCalls(curve, { spot: SPOT, forward: FWD, t: T, rate: RATE });
  assert.ok(rnd, 'density should build');
  // grid covers the observed strike range
  assert.ok(Math.abs(rnd.strikes[0] - curve.strikeMin) < 1e-12);
  assert.ok(Math.abs(rnd.strikes[rnd.strikes.length - 1] - curve.strikeMax) < 1e-12);
  // integrates to 1, raw area close to 1 for a clean smile
  let area = 0;
  for (let i = 1; i < rnd.strikes.length; i++) {
    area += ((rnd.density[i - 1] + rnd.density[i]) / 2) * (rnd.strikes[i] - rnd.strikes[i - 1]);
  }
  assert.ok(Math.abs(area - 1) < 1e-6, `normalized density must integrate to 1 (got ${area})`);
  assert.ok(Math.abs(rnd.rawArea - 1) < 0.02, `clean smile should have rawArea≈1 (got ${rnd.rawArea})`);
  assert.ok(rnd.clippedFraction < 0.01, `clean smile should barely clip (got ${rnd.clippedFraction})`);
  // shape: near-ATM match to the analytic density (wings carry spline error)
  for (let i = 0; i < rnd.strikes.length; i++) {
    const K = rnd.strikes[i];
    if (K > 0.85 * FWD && K < 1.15 * FWD) {
      const analytic = analyticRnd(K);
      const mine = rnd.density[i];
      const rel = Math.abs(mine - analytic) / analytic;
      assert.ok(rel < 0.15, `density mismatch at K=${K.toFixed(1)}: ${mine.toExponential(3)} vs ${analytic.toExponential(3)}`);
    }
  }
  // mean under the RN measure ≈ forward
  let mean = 0;
  for (let i = 1; i < rnd.strikes.length; i++) {
    const w = ((rnd.density[i - 1] + rnd.density[i]) / 2) * (rnd.strikes[i] - rnd.strikes[i - 1]);
    mean += w * (rnd.strikes[i] + rnd.strikes[i - 1]) / 2;
  }
  assert.ok(Math.abs(mean - FWD) / FWD < 0.02, `RN mean ≈ forward (got ${mean})`);
});

test('cdf: monotone non-decreasing, endpoints 0 → 1', () => {
  const curve = R.buildCallCurve(slicePoints(), { forward: FWD, t: T, rate: RATE });
  const rnd = R.impliedDensityFromCalls(curve, { spot: SPOT, forward: FWD, t: T, rate: RATE });
  assert.equal(rnd.cdf[0], 0);
  assert.equal(rnd.cdf[rnd.cdf.length - 1], 1);
  for (let i = 1; i < rnd.cdf.length; i++) {
    assert.ok(rnd.cdf[i] >= rnd.cdf[i - 1] - 1e-12, 'CDF must be monotone');
  }
});

test('pricePayoff: digital and call spread integrate against the density', () => {
  const curve = R.buildCallCurve(slicePoints(), { forward: FWD, t: T, rate: RATE });
  const rnd = R.impliedDensityFromCalls(curve, { spot: SPOT, forward: FWD, t: T, rate: RATE });
  const K10 = 1.1 * SPOT;
  const digital = R.pricePayoff(rnd, (K) => (K >= K10 ? 1 : 0), T, RATE);
  const d2 =
    (Math.log(SPOT / K10) + (RATE - DIV - 0.5 * VOL * VOL) * T) / (VOL * Math.sqrt(T));
  const bsDigital = Math.exp(-RATE * T) * O.normCdf(d2);
  assert.ok(Math.abs(digital - bsDigital) / bsDigital < 0.1, `digital ${digital} vs BS ${bsDigital}`);
  const spread = R.pricePayoff(
    rnd,
    (K) => Math.min(Math.max(K - SPOT, 0), 0.1 * SPOT),
    T,
    RATE,
  );
  const spreadBs =
    O.bsPrice({ spot: SPOT, strike: SPOT, t: T, rate: RATE, dividendYield: DIV, vol: VOL, type: 'call' }) -
    O.bsPrice({ spot: SPOT, strike: K10, t: T, rate: RATE, dividendYield: DIV, vol: VOL, type: 'call' });
  assert.ok(Math.abs(spread - spreadBs) / spreadBs < 0.1, `spread ${spread} vs BS ${spreadBs}`);
});

test('noisy mids: diagnostics report clipped mass instead of hiding it', () => {
  // adversarial: alternate ±4% mid noise — the smile is no longer guaranteed convex
  const noisy = slicePoints().map((p, i) => ({
    ...p,
    mid: p.mid * (i % 2 === 0 ? 1.04 : 0.96),
  }));
  const curve = R.buildCallCurve(noisy, { forward: FWD, t: T, rate: RATE });
  const rnd = R.impliedDensityFromCalls(curve, { spot: SPOT, forward: FWD, t: T, rate: RATE });
  assert.ok(rnd, 'density still builds');
  let area = 0;
  for (let i = 1; i < rnd.strikes.length; i++) {
    area += ((rnd.density[i - 1] + rnd.density[i]) / 2) * (rnd.strikes[i] - rnd.strikes[i - 1]);
  }
  assert.ok(Math.abs(area - 1) < 1e-6);
  assert.ok(rnd.minDensity < 0 || rnd.clippedFraction > 0, 'noise must show up in the diagnostics');
});
