// Unit tests for lib/fno/* (compiled output in .cache/test-build).
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const F = (f) => require(path.join(BUILD, f));
const { US_UNIVERSE, STRIKE_STEP } = F('fno/config.js');
const { buildExpiryLadder } = F('fno/expiry-ladder.js');
const { oiLadder, pcr, maxPain } = F('fno/positioning.js');
const { buildTermStructure } = F('fno/term-structure.js');
const { expectedMoves, historicalAbsMoves, percentileRankLocal, moveVerdict, moveDistributionSummary } = F('fno/expected-move.js');
const O = F('options.js');

// Build BS-consistent quotes for a given strike/expiry/type
function quotedContract(strike, type, oi, iv, dte, spot = 355, r = 0.04, q = 0) {
  const T = dte / 365;
  const px = O.bsPrice({ spot, strike, t: T, rate: r, dividendYield: q, vol: iv, type });
  const bid = Math.max(0.05, px * 0.97);
  const ask = px * 1.03;
  const expiry = new Date(Date.now() + dte * 86400000).toISOString().split('T')[0];
  seq++;
  return {
    expiry, dte, strike, type, oi, volume: 500,
    bid: +bid.toFixed(2), ask: +ask.toFixed(2), last: +px.toFixed(2),
    iv, contractSymbol: `T${seq}`,
  };
}

// Deterministic price series fixture (no network, no globalThis hacks)
function fixtureBars(n = 400) {
  let seed = 12345;
  const rand = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  const normal = () => {
    const u1 = Math.max(1e-7, rand()), u2 = Math.max(1e-7, rand());
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  const bars = [];
  let p = 100;
  const today = Date.now();
  for (let i = 0; i < n; i++) {
    p = Math.max(1, p * Math.exp(0.0004 + 0.018 * normal()));
    bars.push({ date: new Date(today - (n - i) * 86400000).toISOString().split('T')[0], open: p, high: p * 1.01, low: p * 0.99, close: p, volume: 1e6 });
  }
  return bars;
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

let seq = 0;
function contract(strike, type, oi, iv, dte, bid = 5, ask = 5.2, volume = 100) {
  seq++;
  const expiry = new Date(Date.now() + dte * 86400000).toISOString().split('T')[0];
  return {
    expiry, dte, strike, type, oi, volume,
    bid, ask, last: (bid + ask) / 2,
    iv, contractSymbol: `T${seq}`,
  };
}

function threeExpiryChain() {
  const rows = [];
  for (const [dte, iv] of [[21, 0.19], [45, 0.22], [90, 0.28]]) {
    for (const strike of [330, 340, 350, 355, 360, 380]) {
      const near = Math.abs(strike - 355) <= 10;
      rows.push(quotedContract(strike, 'call', near ? 4000 : 400, iv, dte));
      rows.push(quotedContract(strike, 'put', near ? 4000 : 400, iv, dte));
    }
  }
  return rows;
}

// ---------------------------------------------------------------------------
// 1) Universe config sanity
// ---------------------------------------------------------------------------

test('every universe symbol has a strike step', () => {
  for (const s of US_UNIVERSE) {
    assert.ok(STRIKE_STEP[s] > 0, `missing strike step for ${s}`);
  }
});

// ---------------------------------------------------------------------------
// 2) Expiry ladder groups by expiry and locates ATM
// ---------------------------------------------------------------------------

test('expiry ladder groups expiries and totals OI', () => {
  const rows = threeExpiryChain();
  const ladder = buildExpiryLadder(rows, 355, 0.04, 0);
  assert.ok(ladder.length >= 3);
  const sorted = [...ladder].sort((a, b) => a.dte - b.dte);
  for (const g of sorted) {
    assert.ok(g.callOi > 0 && g.putOi > 0);
    assert.ok(Number.isFinite(g.callOi + g.putOi));
  }
  assert.equal(sorted[0].dte, 21);
});

test('expiry ladder computes ATM IV from mid quotes', () => {
  const rows = threeExpiryChain();
  const ladder = buildExpiryLadder(rows, 355, 0.04, 0);
  const front = ladder.find((g) => g.dte === 21);
  assert.ok(front);
  assert.ok(front.atmIv !== null && front.atmIv > 0.05 && front.atmIv < 3,
    `atmIv expected sane, got ${front.atmIv}`);
});

// ---------------------------------------------------------------------------
// 3) Positioning — oiLadder, pcr, maxPain
// ---------------------------------------------------------------------------

test('oiLadder returns strikes within band with dual sides', () => {
  const rows = threeExpiryChain();
  const ladderOut = oiLadder(rows, 355, 0.1);
  assert.ok(ladderOut.length > 0);
  for (const p of ladderOut) {
    assert.ok(p.strike >= 355 * 0.9 && p.strike <= 355 * 1.1);
  }
});

test('pcr returns ratios near 1 for symmetric fixture', () => {
  const rows = threeExpiryChain();
  const p = pcr(rows, 355);
  assert.ok(p.aggregate !== null && Math.abs(p.aggregate - 1) < 0.001);
  assert.ok(p.atm !== null);
});

test('max pain lands at the heaviest strike', () => {
  const rows = [
    contract(100, 'call', 500, 0.2, 30, 1, 1.2),
    contract(100, 'put', 6000, 0.2, 30, 1, 1.2),   // heavy put concentration
    contract(110, 'call', 200, 0.2, 30, 1, 1.2),
    contract(110, 'put', 200, 0.2, 30, 1, 1.2),
    contract(120, 'call', 200, 0.2, 30, 1, 1.2),
    contract(120, 'put', 200, 0.2, 30, 1, 1.2),
  ];
  const res = maxPain(rows, 100);
  assert.equal(res.maxPainStrike, 100, 'max pain should coincide with the heavy strike');
  assert.ok(Array.isArray(res.curve) && res.curve.length > 0);
});

test('max pain returns nulls on empty chain', () => {
  const res = maxPain([], 100);
  assert.equal(res.maxPainStrike, null);
  assert.equal(res.curve.length, 0);
});

// ---------------------------------------------------------------------------
// 4) Term structure verdict
// ---------------------------------------------------------------------------

test('term structure verdict CONTANGO when back > front', () => {
  const rows = threeExpiryChain();
  const ladder = buildExpiryLadder(rows, 355, 0.04, 0);
  const ts = buildTermStructure(
    ladder.map((g) => ({ ...g, rows: rows.filter((r) => r.expiry === g.expiry) })),
    355, 0.04, 0
  );
  assert.equal(ts.verdict.verdict, 'CONTANGO');
});

// ---------------------------------------------------------------------------
// 5) Expected move vs realized
// ---------------------------------------------------------------------------

test('expectedMoves computes straddle and sigma moves', () => {
  const rows = threeExpiryChain();
  const ladder = buildExpiryLadder(rows, 355, 0.04, 0);
  const em = expectedMoves(ladder.map((g) => ({ ...g, rows: rows.filter((r) => r.expiry === g.expiry) })), 355);
  assert.ok(em.length >= 3);
  for (const e of em) {
    assert.ok(e.straddleMove === null || e.straddleMove > 0, `straddle ${e.straddleMove}`);
    assert.ok(e.sigmaMove === null || e.sigmaMove > 0, `sigma ${e.sigmaMove}`);
  }
});

test('historicalAbsMoves produces samples for the fixture series', () => {
  const bars = fixtureBars(400);
  const samples = historicalAbsMoves(bars, 21);
  assert.ok(samples.length > 200);
  assert.ok(samples.every((s) => Number.isFinite(s) && s >= 0));
});

test('percentileRankLocal computes ranks', () => {
  const samples = [0.01, 0.02, 0.03, 0.04, 0.05];
  assert.equal(Math.round(percentileRankLocal(samples, 0.03)), 50);
  assert.equal(Math.round(percentileRankLocal(samples, 0.01)), 10);
  assert.equal(Math.round(percentileRankLocal(samples, 0.05)), 90);
});

test('moveVerdict respects the 25/75 thresholds', () => {
  const samples = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
  assert.equal(moveVerdict(2, samples), 'CHEAP');
  assert.equal(moveVerdict(9, samples), 'RICH');
  assert.equal(moveVerdict(5, samples), 'FAIR');
});
