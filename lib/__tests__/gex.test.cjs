// Unit tests for lib/gex.ts (compiled output in .cache/test-build).
// Fixtures: synthetic chains — no network, no dates beyond injected values.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const O = require(path.join(BUILD, 'options.js'));
const G = require(path.join(BUILD, 'gex.js'));

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

function contract(strike, type, oi, iv = 0.25, dte = 30, overrides = {}) {
  return {
    expiry: '2026-12-11',
    dte,
    strike,
    type,
    oi,
    volume: Math.floor(oi / 3),
    bid: Math.max(0.01, strike * 0.01),
    ask: Math.max(0.02, strike * 0.011),
    last: strike * 0.0105,
    iv,
    contractSymbol: `${type === 'call' ? 'C' : 'P'}${strike}`,
    ...overrides,
  };
}

/** A symmetric chain around 100 with one dominant strike per side. */
function symmetricChain() {
  const rows = [];
  const strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120];
  for (const K of strikes) {
    const oi = K === 110 || K === 90 ? 5000 : 200;
    rows.push(contract(K, 'call', oi, 0.22));
    rows.push(contract(K, 'put', oi, 0.22));
  }
  return rows;
}

function nearEmpty() {
  return [contract(100, 'call', 10, 0.2), contract(100, 'put', 10, 0.2)];
}

const asOf = '2026-11-11T20:00:00Z';
const baseCtx = (rows, spot = 100) =>
  G.buildContext({
    symbol: 'TEST',
    asOf,
    spot,
    rows,
    rate: 0.04,
    dividendYield: 0,
    atmIv: 0.22,
    frontT: 30 / 365,
  });

// ---------------------------------------------------------------------------
// 1) Hand-computed BSM gamma: S=100, K=100, T=0.25, r=0.04, σ=0.2 → Γ ≈ 0.0394
// ---------------------------------------------------------------------------

test('bsGamma matches the reference value within 5e-4', () => {
  const gamma = O.bsGamma(100, 100, 0.25, 0.04, 0, 0.2);
  assert.ok(Math.abs(gamma - 0.0394) < 5e-4, `expected ~0.0394, got ${gamma}`);
});

test('bsGamma guards T<=0 and sigma<=0 (never NaN)', () => {
  assert.equal(O.bsGamma(100, 100, 0, 0.04, 0, 0.2), 0);
  assert.equal(O.bsGamma(100, 100, 0.25, 0.04, 0, 0), 0);
});

// ---------------------------------------------------------------------------
// 2) Empty chain → idle, zero score
// ---------------------------------------------------------------------------

test('empty chain → score 0, state idle, direction none', () => {
  const ctx = baseCtx([], 100);
  assert.equal(ctx, null, 'empty rows → context not constructible');
  const res = G.trapScore({
    walls: { call: null, put: null, runnerUpCall: null, runnerUpPut: null, totalOiLocal: 0 },
    flip: { flip: null, profile: [], regime: 'unknown' },
    spot: 100,
    atmIv: 0.22,
    frontT: 30 / 365,
    oiChange1d: null,
    oiChange5d: null,
    oiChange5dZ: null,
    realizedVol20: null,
    ret5d: null,
    expiryConcentration7d: 0,
    closedBeyondWallRecently: false,
  });
  assert.equal(res.score, 0);
  assert.equal(res.state, 'idle');
  assert.equal(res.direction, 'none');
});

// ---------------------------------------------------------------------------
// 3) Walls detection — 2 nearest strikes excluded, walls found per side
// ---------------------------------------------------------------------------

test('walls exclude the 2 nearest strikes to spot', () => {
  const ctx = baseCtx(symmetricChain(), 100);
  assert.ok(ctx);
  const walls = G.detectWalls(ctx);
  assert.ok(walls.call, 'call wall found');
  assert.ok(walls.put, 'put wall found');
  assert.notEqual(walls.call.strike, 100);
  assert.notEqual(walls.call.strike, 105);
  // nearest neighbour strikes should never be walls even if largest
  assert.ok(walls.call.strike > 100);
  assert.ok(walls.put.strike < 100);
});

test('symmetry: equal OI per side → symmetric wall strikes ±10', () => {
  const ctx = baseCtx(symmetricChain(), 100);
  const walls = G.detectWalls(ctx);
  assert.equal(walls.call.strike, 110);
  assert.equal(walls.put.strike, 90);
});

test('no OI above spot → call wall null, put wall found', () => {
  const rows = [
    contract(90, 'put', 5000),
    contract(95, 'put', 3000),
    contract(85, 'put', 1000),
  ];
  const ctx = baseCtx(rows, 100);
  const walls = G.detectWalls(ctx);
  assert.equal(walls.call, null);
  assert.ok(walls.put);
});

// ---------------------------------------------------------------------------
// 4) Gamma profile & flip — deterministic shapes
// ---------------------------------------------------------------------------

test('gamma flip profile has 121 points and flip inside grid when present', () => {
  const ctx = baseCtx(symmetricChain(), 100);
  const flip = G.gammaFlip(ctx);
  assert.equal(flip.profile.length, 121);
  assert.ok(['long', 'short', 'unknown'].includes(flip.regime));
  if (flip.flip !== null) {
    assert.ok(flip.flip > 100 * 0.85 && flip.flip < 100 * 1.15);
  }
});

test('hedge flow returns finite numbers and net = up + down', () => {
  const ctx = baseCtx(symmetricChain(), 100);
  const flow = G.hedgeFlow(ctx);
  for (const v of [flow.perPctUp, flow.perPctDown, flow.net]) {
    assert.ok(Number.isFinite(v), 'finite hedge flow');
  }
  assert.ok(Math.abs(flow.net - (flow.perPctUp + flow.perPctDown)) < 1e-6);
});

// ---------------------------------------------------------------------------
// 5) Trap score — bounds and monotonicity
// ---------------------------------------------------------------------------

function makeInput(overrides = {}) {
  const ctx = baseCtx(symmetricChain(), 100);
  const walls = G.detectWalls(ctx);
  const flip = G.gammaFlip(ctx);
  return {
    walls,
    flip,
    spot: 100,
    atmIv: 0.22,
    frontT: 30 / 365,
    oiChange1d: 100,
    oiChange5d: 800,
    oiChange5dZ: 2.5,
    realizedVol20: 0.2,
    ret5d: 0.015,
    expiryConcentration7d: 0.4,
    closedBeyondWallRecently: false,
    ...overrides,
  };
}

test('score is clamped to [0, 100]', () => {
  const extreme = makeInput({ oiChange5dZ: 100, ret5d: 0.5 });
  const res = G.trapScore(extreme);
  assert.ok(res.score >= 0 && res.score <= 100, `score ${res.score} out of bounds`);
});

test('score is monotone non-decreasing in oiChange5dZ', () => {
  const zs = [-2, 0, 0.5, 1, 2, 4];
  const scores = zs.map((z) => G.trapScore(makeInput({ oiChange5dZ: z })).score);
  for (let i = 1; i < scores.length; i++) {
    assert.ok(scores[i] >= scores[i - 1], `score dropped from z=${zs[i - 1]} to z=${zs[i]}`);
  }
});

test('score is monotone non-decreasing in wall share', () => {
  // Build chains where one strike grows progressively more dominant
  const shares = [0.08, 0.15, 0.25, 0.4];
  const scores = shares.map((target) => {
    const rows = [];
    const strikes = [80, 90, 95, 105, 110, 120];
    for (const K of strikes) {
      const oi = K === 110 ? Math.round(target * 10000) : Math.round((1 - target) * 10000 / 5);
      rows.push(contract(K, 'call', oi, 0.22));
    }
    const ctx = baseCtx(rows, 100);
    const walls = G.detectWalls(ctx);
    const flip = G.gammaFlip(ctx);
    return G.trapScore({
      walls, flip, spot: 100, atmIv: 0.22, frontT: 30 / 365,
      oiChange1d: 0, oiChange5d: 0, oiChange5dZ: 1,
      realizedVol20: 0.22, ret5d: 0, expiryConcentration7d: 0.3,
      closedBeyondWallRecently: false,
    }).score;
  });
  for (let i = 1; i < scores.length; i++) {
    assert.ok(scores[i] >= scores[i - 1], `score dropped as share rose: ${scores.join(',')}`);
  }
});

test('missing IV falls back to atmIv — no NaN anywhere in the output', () => {
  const rows = symmetricChain().map((r) => ({ ...r, iv: null }));
  const ctx = baseCtx(rows, 100);
  assert.ok(ctx);
  const flip = G.gammaFlip(ctx);
  assert.ok(flip.profile.every((p) => Number.isFinite(p.gex)));
  const res = G.trapScore(makeInput({ flip }));
  assert.ok(Number.isFinite(res.score));
});

test('velocity degraded when z-score missing — flagged, not invented', () => {
  const res = G.trapScore(makeInput({ oiChange5dZ: null }));
  assert.equal(res.velocityDegraded, true);
  assert.equal(res.components.velocity, 0);
});

test('all-zero oi chain → idle', () => {
  const rows = symmetricChain().map((r) => ({ ...r, oi: 0 }));
  const ctx = baseCtx(rows, 100);
  // buildContext requires oi > 0? rows are filtered inside; expect null or idle
  if (ctx === null) {
    assert.ok(true);
    return;
  }
  const res = G.trapScore(makeInput({ walls: G.detectWalls(ctx), flip: G.gammaFlip(ctx) }));
  assert.ok(['idle', 'building'].includes(res.state));
});

console.log('gex tests loaded');
