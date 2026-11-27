// Unit tests for lib/options-decision.ts — the single options verdict object.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const D = require(path.join(BUILD, 'options-decision.js'));

test('buildOptionsDecision: rich / cheap from one VRP definition', () => {
  const rich = D.buildOptionsDecision({ chain: { iv30: 0.3 }, rv20: 0.2 });
  assert.equal(rich.verdict, 'RICH');
  assert.ok(rich.vrpVariancePoints > 0);
  const cheap = D.buildOptionsDecision({ chain: { iv30: 0.15 }, rv20: 0.2 });
  assert.equal(cheap.verdict, 'CHEAP');
  assert.equal(D.buildOptionsDecision({}).verdict, 'insufficient-data');
});

test('buildOptionsDecision: curve, front slice, trap and F&O positioning', () => {
  const d = D.buildOptionsDecision({
    chain: {
      iv30: 0.25,
      available: true,
      trap: { state: 'armed', score: 72 },
      termStructure: [{ dte: 10, iv: 0.3 }, { dte: 60, iv: 0.2 }],
      slices: [{ dte: 21, atmIv: 0.25, rr25: 0.04, bf25: 0.01 }],
      clean: { used: 18, total: 24, medianRelSpread: 0.02 },
      snapshot: { count: 42 },
    },
    fnoPanel: { maxPain: { maxPainStrike: 210 }, pcr: { aggregate: 0.9 }, strikesUsed: 18, strikesTotal: 24 },
    rv20: 0.2,
  });
  assert.equal(d.curve, 'INVERTED');
  assert.equal(d.trap.state, 'armed');
  assert.equal(d.frontSkew.dte, 21);
  assert.equal(d.positioning.maxPainStrike, 210);
  assert.equal(d.accruedSessions, 42);
  assert.ok(d.notes.some((n) => n.includes('Strikes used 18/24')));
  assert.ok(d.notes.some((n) => n.toLowerCase().includes('inverted')));
});

test('buildOptionsDecision: F&O-only iv30 still produces a verdict', () => {
  const d = D.buildOptionsDecision({ fnoPanel: { iv30: 0.3, strikesUsed: 10, strikesTotal: 12 }, rv20: 0.2 });
  assert.equal(d.verdict, 'RICH');
  assert.equal(d.curve, null);
  assert.equal(d.trap, null);
  assert.equal(d.gammaFlip, null);
  assert.equal(d.flipDistPct, null);
  assert.equal(d.totalGex, null);
  assert.equal(d.gexRegime, 'unknown');
  assert.equal(d.gexPerAdv, null);
});

test('buildOptionsDecision: populates GEX metrics from chain trap or fnoPanel trap', () => {
  const d = D.buildOptionsDecision({
    chain: {
      iv30: 0.28,
      trap: {
        available: true,
        state: 'building',
        score: 48,
        spot: 140,
        totalGex: 18_500_000,
        regime: 'long',
        gammaFlip: { flip: 133, regime: 'long' },
        flipDistPct: -0.05,
        gexPerAdv: 0.012,
      },
    },
    rv20: 0.22,
  });
  assert.equal(d.gammaFlip, 133);
  assert.equal(d.flipDistPct, -0.05);
  assert.equal(d.totalGex, 18_500_000);
  assert.equal(d.gexRegime, 'long');
  assert.equal(d.gexPerAdv, 0.012);
  assert.ok(d.notes.some((n) => n.includes('Net GEX +$18.5M')));

  const fromFno = D.buildOptionsDecision({
    fnoPanel: {
      iv30: 0.32,
      trap: {
        available: true,
        state: 'armed',
        score: 65,
        spot: 100,
        totalGex: -9_200_000,
        gammaFlip: { flip: 105 },
      },
    },
    rv20: 0.2,
  });
  assert.equal(fromFno.trap.state, 'armed');
  assert.equal(fromFno.gammaFlip, 105);
  assert.ok(Math.abs(fromFno.flipDistPct - 0.05) < 1e-9);
  assert.equal(fromFno.totalGex, -9_200_000);
  assert.equal(fromFno.gexRegime, 'short');
  assert.equal(fromFno.gexPerAdv, null);
});

test('buildOptionsDecision: unavailable trap degrades cleanly to unknown/null GEX fields', () => {
  const d = D.buildOptionsDecision({
    chain: {
      iv30: 0.25,
      trap: {
        available: false,
        state: 'idle',
        score: 0,
        totalGex: 1_000_000,
        regime: 'long',
        gammaFlip: { flip: 120 },
      },
    },
    rv20: 0.2,
  });
  assert.equal(d.trap, null);
  assert.equal(d.gammaFlip, null);
  assert.equal(d.flipDistPct, null);
  assert.equal(d.totalGex, null);
  assert.equal(d.gexRegime, 'unknown');
  assert.equal(d.gexPerAdv, null);
});
