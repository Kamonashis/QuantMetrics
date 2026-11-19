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
});
