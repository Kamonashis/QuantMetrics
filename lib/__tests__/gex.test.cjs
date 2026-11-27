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

// ---------------------------------------------------------------------------
// 6) Second-order BSM Greeks (Vanna, Charm, Speed) + IV sanitization
// ---------------------------------------------------------------------------

test('bsVanna matches central finite differences of delta w.r.t. sigma and vega w.r.t. spot (with q)', () => {
  const S = 105, K = 100, T = 45 / 365, r = 0.045, q = 0.02, sigma = 0.28;
  const vanna = O.bsVanna(S, K, T, r, q, sigma);
  const hSig = 1e-5;
  const dDeltaDSig =
    (O.bsDelta(S, K, T, r, q, sigma + hSig, 'call') - O.bsDelta(S, K, T, r, q, sigma - hSig, 'call')) /
    (2 * hSig);
  const hS = 1e-3;
  const dVegaDS =
    (O.bsVega(S + hS, K, T, r, q, sigma) - O.bsVega(S - hS, K, T, r, q, sigma)) / (2 * hS);
  assert.ok(Math.abs(vanna - dDeltaDSig) < 1e-5, `vanna=${vanna} vs dDelta/dSig=${dDeltaDSig}`);
  assert.ok(Math.abs(vanna - dVegaDS) < 1e-5, `vanna=${vanna} vs dVega/dS=${dVegaDS}`);
  assert.equal(O.bsVanna(S, K, 0, r, q, sigma), 0);
});

test('bsCharm matches -dDelta/dT for calls and puts with continuous dividend yield q', () => {
  const S = 98, K = 105, T = 30 / 365, r = 0.04, q = 0.025, sigma = 0.24;
  const hT = 1e-5;
  for (const type of ['call', 'put']) {
    const charm = O.bsCharm(S, K, T, r, q, sigma, type);
    const num =
      -(O.bsDelta(S, K, T + hT, r, q, sigma, type) - O.bsDelta(S, K, T - hT, r, q, sigma, type)) /
      (2 * hT);
    assert.ok(Math.abs(charm - num) < 1e-4, `${type} charm=${charm} vs -dDelta/dT=${num}`);
    assert.ok(Math.abs(O.bsCharmPerDay(S, K, T, r, q, sigma, type) * 365 - charm) < 1e-12);
  }
  assert.equal(O.bsCharm(S, K, 0, r, q, sigma, 'call'), 0);
});

test('bsSpeed matches dGamma/dS with dividend yield q', () => {
  const S = 102, K = 100, T = 25 / 365, r = 0.04, q = 0.015, sigma = 0.25;
  const speed = O.bsSpeed(S, K, T, r, q, sigma);
  const hS = 1e-3;
  const num = (O.bsGamma(S + hS, K, T, r, q, sigma) - O.bsGamma(S - hS, K, T, r, q, sigma)) / (2 * hS);
  assert.ok(Math.abs(speed - num) < 1e-6, `speed=${speed} vs dGamma/dS=${num}`);
  assert.equal(O.bsSpeed(S, K, 0, r, q, sigma), 0);
});

test('sanitizeIv rejects degenerate, NaN, and extreme wide-quote IV outliers', () => {
  assert.equal(O.sanitizeIv(null, 0.22), 0.22);
  assert.equal(O.sanitizeIv(0.0001, 0.22), 0.22);
  assert.equal(O.sanitizeIv(21.41, 0.35), 0.35);
  assert.equal(O.sanitizeIv(3.41, 0.18), 0.18);
  assert.equal(O.sanitizeIv(2.4, 0.18, { bid: 0, ask: 0.05 }), 0.18);
  assert.equal(O.sanitizeIv(0.27, 0.22, { bid: 2.0, ask: 2.1 }), 0.27);
});

// ---------------------------------------------------------------------------
// 7) 0-DTE support, full-chain unification, convention split, buckets & matrix
// ---------------------------------------------------------------------------

test('include0Dte allows dte=0 rows with 0.5/365 floor and can be toggled off', () => {
  const rows = [
    contract(100, 'call', 2000, 0.25, 0, { expiry: '2026-11-11' }),
    contract(100, 'put', 1000, 0.25, 0, { expiry: '2026-11-11' }),
    contract(105, 'call', 1500, 0.22, 14, { expiry: '2026-11-25' }),
  ];
  const with0 = G.buildContext({
    symbol: 'TEST', asOf, spot: 100, rows, rate: 0.04, dividendYield: 0, atmIv: 0.22, frontT: 14 / 365, include0Dte: true,
  });
  const no0 = G.buildContext({
    symbol: 'TEST', asOf, spot: 100, rows, rate: 0.04, dividendYield: 0, atmIv: 0.22, frontT: 14 / 365, include0Dte: false,
  });
  assert.ok(with0 && no0);
  assert.equal(with0.rows.filter((r) => r.dte === 0).length, 2);
  assert.equal(no0.rows.filter((r) => r.dte === 0).length, 0);
  const pWith = G.gammaProfile(with0);
  const pNo = G.gammaProfile(no0);
  assert.ok(Number.isFinite(pWith.totalGex) && pWith.totalGex > pNo.totalGex);
});

test('gammaProfile and hedgeFlow unify full-chain (ITM + OTM) accounting', () => {
  // Include ITM calls (K < S) and ITM puts (K > S) within ±2% of spot=100
  const rows = [
    contract(99, 'call', 3000, 0.22, 20),
    contract(99, 'put', 1000, 0.22, 20),
    contract(101, 'call', 1000, 0.22, 20),
    contract(101, 'put', 3000, 0.22, 20),
  ];
  const ctx = baseCtx(rows, 100);
  const prof = G.gammaProfile(ctx, { maxDte: 45, convention: 'squeeze' });
  const flow = G.hedgeFlow(ctx, { bandPct: 0.02, maxDte: 45, convention: 'squeeze' });
  // Because all strikes are within ±2%, hedgeFlow.net must equal prof.totalGex
  assert.ok(Math.abs(flow.net - prof.totalGex) < 1e-6, `flow.net=${flow.net} vs totalGex=${prof.totalGex}`);
  // And ITM call at 99 contributed positive callGex, ITM put at 101 contributed negative putGex
  const b99 = prof.byStrike.find((b) => b.strike === 99);
  const b101 = prof.byStrike.find((b) => b.strike === 101);
  assert.ok(b99 && b99.callGex > 0 && b99.putGex < 0);
  assert.ok(b101 && b101.callGex > 0 && b101.putGex < 0);
  assert.ok(Math.abs((b99.callGex + b99.putGex) - b99.gex) < 1e-6);
});

test('gammaProfile & gammaFlip separate call/put GEX, volumeGex, charm, vanna and mirror under inverse', () => {
  const ctx = baseCtx(symmetricChain(), 100);
  const sq = G.gammaProfile(ctx, { convention: 'squeeze' });
  const inv = G.gammaProfile(ctx, { convention: 'inverse' });
  assert.ok(Math.abs(sq.totalGex + inv.totalGex) < 1e-6);
  assert.ok(Math.abs(sq.totalCallGex + inv.totalCallGex) < 1e-6);
  assert.ok(Math.abs(sq.totalPutGex + inv.totalPutGex) < 1e-6);
  assert.ok(Math.abs(sq.totalVolumeGex + inv.totalVolumeGex) < 1e-6);
  assert.ok(Math.abs(sq.totalCharm + inv.totalCharm) < 1e-6);
  assert.ok(Math.abs(sq.totalVanna + inv.totalVanna) < 1e-6);
  assert.ok(sq.totalCallGex > 0 && sq.totalPutGex < 0);

  const flipSq = G.gammaFlip(ctx, { convention: 'squeeze' });
  const flipInv = G.gammaFlip(ctx, { convention: 'inverse' });
  assert.equal(flipSq.profile.length, 121);
  assert.ok(flipSq.profile.every((p) => Number.isFinite(p.callGex) && Number.isFinite(p.putGex) && Math.abs(p.gexSqueeze + p.gexInverse) < 1e-6));
  assert.equal(flipSq.byConvention.squeeze.regime, flipSq.regime);
  assert.equal(flipSq.byConvention.inverse.regime, flipInv.regime);
});

test('expiryBuckets, strikeExpiryMatrix and detectGammaWalls decompose GEX cleanly across tenors', () => {
  const rows = [
    contract(100, 'call', 2000, 0.25, 0, { expiry: '2026-11-11' }),
    contract(95, 'put', 1800, 0.24, 5, { expiry: '2026-11-16' }),
    contract(110, 'call', 4500, 0.22, 37, { expiry: '2026-12-18' }), // 3rd Friday monthly OPEX
    contract(90, 'put', 4200, 0.23, 37, { expiry: '2026-12-18' }),
    contract(115, 'call', 2500, 0.21, 70, { expiry: '2027-01-20' }),
  ];
  const ctx = baseCtx(rows, 100);
  const buckets = G.expiryBuckets(ctx, { maxDte: 90, convention: 'squeeze' });
  assert.deepEqual(buckets.map((b) => b.bucket), ['0dte', 'weekly', 'monthly', 'far']);
  const shareSum = buckets.reduce((s, b) => s + b.shareOfAbsGex, 0);
  assert.ok(Math.abs(shareSum - 1) < 1e-6, `shareSum=${shareSum}`);
  assert.ok(buckets[0].oi > 0 && buckets[1].oi > 0 && buckets[2].oi > 0 && buckets[3].oi > 0);

  const matrix = G.strikeExpiryMatrix(ctx, { maxDte: 90 });
  assert.equal(matrix.expiries.length, 4);
  assert.equal(matrix.netGex.length, matrix.strikes.length);
  assert.equal(matrix.netGex[0].length, matrix.expiries.length);

  const gWalls = G.detectGammaWalls(ctx, { maxDte: 45, convention: 'squeeze' });
  assert.ok(gWalls.call && gWalls.put);
  assert.ok(gWalls.call.gex > 0 && gWalls.put.gex < 0);
});

test('raw Yahoo IV outlier (>300%) is sanitized instead of poisoning GEX', () => {
  const cleanRows = [
    contract(105, 'call', 4000, 0.22, 30),
    contract(110, 'call', 5000, 0.22, 30),
    contract(95, 'put', 2000, 0.22, 30),
  ];
  const dirtyRows = cleanRows.map((r) =>
    r.strike === 110 && r.type === 'call'
      ? { ...r, iv: 21.41, bid: 0, ask: 0.05 }
      : r
  );
  const profClean = G.gammaProfile(baseCtx(cleanRows, 100));
  const profDirty = G.gammaProfile(baseCtx(dirtyRows, 100));
  assert.ok(Number.isFinite(profDirty.totalGex) && profDirty.totalGex !== 0);
  assert.ok(Math.abs(profDirty.totalCallGex - profClean.totalCallGex) < 1e-6);
  assert.ok(profDirty.byStrike.every((b) => Number.isFinite(b.gex) && Number.isFinite(b.charm) && Number.isFinite(b.vanna)));
});

test('trapScore velocity works for downward put-wall buildup too', () => {
  const rows = [
    contract(98, 'put', 50, 0.24, 25),
    contract(102, 'call', 50, 0.24, 25),
    contract(92, 'put', 8000, 0.24, 25),
    contract(85, 'put', 1000, 0.24, 25),
    contract(115, 'call', 1000, 0.24, 25),
  ];
  const ctx = baseCtx(rows, 100);
  const walls = G.detectWalls(ctx);
  const res = G.trapScore({
    walls,
    flip: { flip: 105, regime: 'short' },
    spot: 100,
    atmIv: 0.24,
    frontT: 25 / 365,
    oiChange1d: 200,
    oiChange5d: 1200,
    oiChange5dZ: 2.5,
    realizedVol20: 0.2,
    ret5d: -0.015,
    expiryConcentration7d: 0.4,
    closedBeyondWallRecently: false,
  });
  assert.equal(res.nearWall, 'put');
  assert.ok(res.components.velocity > 0, `expected positive velocity for put-wall growth, got ${res.components.velocity}`);
});

console.log('gex tests loaded');
