// Unit tests for lib/vol-signals.ts (compiled output in .cache/test-build).
// Fixtures: deterministic — no network, no clock.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const V = require(path.join(BUILD, 'vol-signals.js'));

const approx = (a, b, eps = 1e-3) => assert.ok(Math.abs(a - b) < eps, `expected ${b}, got ${a}`);

// ---------------------------------------------------------------------------
// rollingZ
// ---------------------------------------------------------------------------

test('rollingZ: null until window fills, then sample-stdev z', () => {
  const z = V.rollingZ([1, 2, 3, 4, 5, 6], 3);
  assert.equal(z[0], null);
  assert.equal(z[1], null);
  approx(z[2], 1.0, 1e-9); // window [1,2,3]: mean 2, sample sd 1 ⇒ (3−2)/1 = 1
  assert.ok(Number.isFinite(z[3]));
  approx(z[5], 1.0, 1e-9); // mean(4,5,6)=5, sample sd=1, (6-5)/1 = 1
});

test('rollingZ: constant series yields null (sd = 0)', () => {
  const z = V.rollingZ([0.2, 0.2, 0.2, 0.2], 3);
  assert.equal(z[3], null);
});

test('rollingZ: propagates nulls in the trailing window', () => {
  const z = V.rollingZ([1, null, 3, 4, 5], 3);
  assert.equal(z[3], null); // window [null,3,4]
  assert.ok(Number.isFinite(z[4])); // window [3,4,5]
});

// ---------------------------------------------------------------------------
// forwardRealizedVol / varianceBaseline
// ---------------------------------------------------------------------------

test('forwardRealizedVol: recovers the pre-spike baseline from RV10/RV30', () => {
  // RV10 = 46.2%, RV30 = 30.3% ⇒ prior-20d baseline ≈ 17.6%.
  const base = V.varianceBaseline(0.462, 0.303, 10, 30);
  assert.ok(base !== null);
  approx(base, 0.176, 2e-3);
});

test('forwardRealizedVol: null when forward segment variance ≤ 0', () => {
  assert.equal(V.forwardRealizedVol(0.60, 0.30, 10, 30), null);
});

test('forwardRealizedVol: null on bad horizons', () => {
  assert.equal(V.forwardRealizedVol(0.3, 0.3, 30, 10), null);
});

// ---------------------------------------------------------------------------
// forwardImpliedVol
// ---------------------------------------------------------------------------

test('forwardImpliedVol: w-difference between two tenors', () => {
  const tp = [
    { dte: 10, t: 10 / 365, iv: 0.40 },
    { dte: 30, t: 30 / 365, iv: 0.286 },
    { dte: 60, t: 60 / 365, iv: 0.24 },
  ];
  const fv = V.forwardImpliedVol(tp, 10, 30);
  assert.ok(fv !== null);
  approx(fv, 0.207, 2e-3);
});

test('forwardImpliedVol: null when calendar variance is inverted', () => {
  const tp = [
    { dte: 10, t: 10 / 365, iv: 0.50 },
    { dte: 30, t: 30 / 365, iv: 0.05 }, // violently downward sloping → negative forward variance
  ];
  assert.equal(V.forwardImpliedVol(tp, 10, 30), null);
});

test('forwardImpliedVol: null with fewer than 2 usable points', () => {
  assert.equal(V.forwardImpliedVol([{ dte: 10, t: 10 / 365, iv: 0.3 }], 10, 30), null);
  assert.equal(V.forwardImpliedVol([{ dte: 10, t: 10 / 365, iv: null }, { dte: 30, t: 30 / 365, iv: null }], 10, 30), null);
});

// ---------------------------------------------------------------------------
// rvTriangle
// ---------------------------------------------------------------------------

test('rvTriangle: backwardation / contango / humped / mixed', () => {
  assert.equal(V.rvTriangle(0.46, 0.46, 0.35, 0.30).shape, 'backwardation');
  assert.equal(V.rvTriangle(0.10, 0.15, 0.25, 0.30).shape, 'contango');
  assert.equal(V.rvTriangle(0.20, 0.45, 0.30, 0.20).shape, 'humped');
  assert.equal(V.rvTriangle(0.35, 0.46, 0.20, 0.50).shape, 'mixed');
});

test('rvTriangle: falls back to the 10→20→30 ladder without RV5', () => {
  assert.equal(V.rvTriangle(null, 0.46, 0.35, 0.30).shape, 'backwardation');
  assert.equal(V.rvTriangle(null, 0.20, 0.30, 0.40).shape, 'contango');
});

test('rvTriangle: null without the core windows; spreads in vol points', () => {
  assert.equal(V.rvTriangle(0.3, null, 0.25, 0.30), null);
  const t = V.rvTriangle(0.46, 0.46, 0.35, 0.30);
  approx(t.spread1020, 0.11, 1e-9);
  approx(t.spread2030, 0.05, 1e-9);
});

test('breakevenVol: 1.25× IV by default', () => {
  approx(V.breakevenVol(0.286), 0.3575, 1e-9);
  approx(V.breakevenVol(0.286, 1.1), 0.3146, 1e-9);
});

test('varianceRatio: RV²/IV² under the direct convention', () => {
  approx(V.varianceRatio(0.303, 0.286), 1.122, 2e-3); // RV richer than IV
  approx(V.varianceRatio(0.286, 0.286), 1.0, 1e-12);  // parity
  assert.ok(V.varianceRatio(0.462, 0.286) > 2.5);     // front-window spike reads here
  assert.equal(V.varianceRatio(0.3, 0), null);
  assert.equal(V.varianceRatio(null, 0.286), null);
});

// ---------------------------------------------------------------------------
// evaluateVolSignals
// ---------------------------------------------------------------------------

const msftLike = {
  iv30: 0.286,
  rv5: 0.462,
  rv10: 0.462,
  rv20: 0.35,
  rv30: 0.303,
  skew60: 0.619,
  skew60LowerCi: -0.1,
  skew60UpperCi: 1.34,
  vrMinZ: 0.46,
  adfResidualP: 0.0021,
  cusumBreakSessionsAgo: 21,
  hmmCrisisProb: 0.11,
  hmmDominantState: 1,
  rr25Front: 0.02,
};

test('evaluateVolSignals: MSFT-like snapshot matches the pre-registered reads', () => {
  const rules = V.evaluateVolSignals(msftLike);
  const byId = Object.fromEntries(rules.map((r) => [r.id, r]));

  assert.equal(byId['long-gamma'].state, 'met'); // RV5 46.2% > 1.25×28.6% = 35.8%
  assert.equal(byId['long-gamma-exit'].state, 'not-met'); // RV5 and IV30 both above thresholds
  assert.equal(byId['short-premium-engage'].state, 'not-met'); // spread ≈ +11.0 pts < +12
  assert.equal(byId['short-premium-flat'].state, 'not-met'); // spread ≈ +11.0 pts > +5
  assert.equal(byId['hmm-crisis-cut'].state, 'not-met'); // 11% < 25%
  assert.equal(byId['hmm-half-size'].state, 'met'); // dominant state ≠ calm
  assert.equal(byId['cusum-cooldown'].state, 'not-met'); // 21 td since break
  assert.equal(byId['skew-crash'].state, 'not-met'); // +0.62 skew
  assert.equal(byId['vr-mean-reversion'].state, 'not-met'); // +0.46 z
  assert.equal(byId['adf-unstable'].state, 'not-met'); // residuals mean-revert
  assert.equal(byId['rr25-call-wing'].state, 'met'); // calls richer than puts
});

test('evaluateVolSignals: all-null inputs degrade honestly', () => {
  const rules = V.evaluateVolSignals({
    iv30: null, rv5: null, rv10: null, rv20: null, rv30: null,
    skew60: null, skew60LowerCi: null, skew60UpperCi: null,
    vrMinZ: null, adfResidualP: null, cusumBreakSessionsAgo: null,
    hmmCrisisProb: null, hmmDominantState: null, rr25Front: null,
  });
  for (const r of rules) {
    if (r.id === 'cusum-cooldown') {
      assert.equal(r.state, 'not-met'); // null = none in window, which is actionable
      assert.equal(r.measured, 'none in window');
    } else {
      assert.equal(r.state, 'insufficient-data', `${r.id} should be insufficient-data`);
    }
  }
});

test('evaluateVolSignals: exit rule fires on either leg', () => {
  const lowRv = V.evaluateVolSignals({ ...msftLike, rv5: 0.25, iv30: 0.3 });
  assert.equal(lowRv.find((r) => r.id === 'long-gamma-exit').state, 'met');
  const lowIv = V.evaluateVolSignals({ ...msftLike, rv5: 0.45, iv30: 0.25 });
  assert.equal(lowIv.find((r) => r.id === 'long-gamma-exit').state, 'met');
});

test('evaluateVolSignals: short-premium engagement requires the +12 pt gap', () => {
  const rich = V.evaluateVolSignals({ ...msftLike, iv30: 0.32 }); // spread ≈ +14.4
  assert.equal(rich.find((r) => r.id === 'short-premium-engage').state, 'met');
  const flat = V.evaluateVolSignals({ ...msftLike, iv30: 0.22 }); // spread ≈ +4.4
  assert.equal(flat.find((r) => r.id === 'short-premium-flat').state, 'met');
});

test('evaluateVolSignals: skew flip and VR flip fire their rules', () => {
  const flip = V.evaluateVolSignals({ ...msftLike, skew60: -0.15, vrMinZ: -1.4 });
  assert.equal(flip.find((r) => r.id === 'skew-crash').state, 'met');
  assert.equal(flip.find((r) => r.id === 'vr-mean-reversion').state, 'met');
});

test('evaluateVolSignals: fresh CUSUM break triggers cooldown', () => {
  const fresh = V.evaluateVolSignals({ ...msftLike, cusumBreakSessionsAgo: 2 });
  assert.equal(fresh.find((r) => r.id === 'cusum-cooldown').state, 'met');
});
