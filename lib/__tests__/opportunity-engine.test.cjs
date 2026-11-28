// Unit tests for lib/opportunity-engine.ts — six trader cards from existing signals.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const O = require(path.join(BUILD, 'opportunity-engine.js'));

const rule = (id, state, measured = 'x') => ({ id, label: id, metric: id, measured, trigger: 'rule', state, action: 'act' });

function base(overrides = {}) {
  return {
    symbol: 'AAPL', asOf: '2026-11-20', chainAvailable: true, fnoAvailable: true,
    barsSource: 'live_market', iv30: 0.25, spot: 100, rv5: 0.34,
    vrpVariancePoints: 200, accruedSessions: 80, ivRankReady: true,
    earningsSessions: 45, earningsIsEstimate: false,
    term: [{ dte: 10, iv: 0.3 }, { dte: 60, iv: 0.24 }], curve: 'INVERTED',
    frontSkew: { dte: 21, rr25: 0.03, bf25: 0.01 },
    vvix: 110, vvixPercentile: 55, skewPercentile: 60,
    trap: { state: 'triggered', direction: 'up', score: 70, callWall: 106, putWall: 94, flip: 99, backtestVerdict: 'above_base_rate', backtestN: 40, convention: 'squeeze' },
    expectedMove: { dte: 21, impliedPct: 4, verdict: 'CHEAP', percentile: 30, n: 300 },
    carry: { meanMonthlyBp2: 12, hitRate: 0.68, worstMonthBp2: -80, maxDrawdownBp2: -200, n: 120 },
    crossSectional: [
      { symbol: 'A', iv30: 0.3, rv20: 0.2, ivRank: null },
      { symbol: 'B', iv30: 0.26, rv20: 0.22, ivRank: null },
      { symbol: 'C', iv30: 0.2, rv20: 0.25, ivRank: null },
    ],
    riskBudgetFraction: 0.01,
    rules: [
      rule('short-premium-engage', 'met', '+14.0 pts'),
      rule('short-premium-flat', 'not-met'),
      rule('hmm-crisis-cut', 'not-met'),
      rule('cusum-cooldown', 'not-met'),
      rule('long-gamma', 'met', '34.0% vs 31.3%'),
      rule('long-gamma-exit', 'not-met'),
      rule('skew-crash', 'met', 'skew -0.4'),
      rule('rr25-call-wing', 'not-met'),
    ],
    ...overrides,
  };
}

test('evaluateOpportunities: six families with states, evidence and no empty fields', () => {
  const ops = O.evaluateOpportunities(base());
  assert.deepEqual(ops.map((o) => o.family), ['premium-harvest', 'long-gamma', 'directional-trap', 'event-straddle', 'skew-tail', 'cross-sectional']);
  for (const o of ops) {
    assert.ok(o.thesis && o.invalidation && o.trigger.rule && o.edge.provenance && o.size.basis);
    assert.ok(o.evidence.view && Object.keys(o.evidence.params).length > 0);
    assert.ok(['ACTIONABLE', 'WATCH', 'BLOCKED', 'INSUFFICIENT_DATA'].includes(o.state));
  }
});

test('gates: earnings and base-rate failure block the right cards', () => {
  const ops = O.evaluateOpportunities(base({ earningsSessions: 5 }));
  assert.equal(ops.find((o) => o.id === 'premium-harvest').state, 'BLOCKED');
  assert.equal(ops.find((o) => o.id === 'long-gamma').state, 'BLOCKED');
  const noRate = O.evaluateOpportunities(base({ trap: { ...base().trap, backtestVerdict: 'insufficient_history' } }));
  assert.equal(noRate.find((o) => o.id === 'directional-trap').state, 'BLOCKED');
  assert.ok(noRate.find((o) => o.id === 'directional-trap').gates.some((g) => g.includes('base-rate')));
});

test('degradation: missing rules/data never default to actionable', () => {
  const ops = O.evaluateOpportunities(base({ rules: [], trap: null, expectedMove: null, crossSectional: [] }));
  for (const id of ['premium-harvest', 'long-gamma', 'directional-trap', 'event-straddle', 'skew-tail', 'cross-sectional']) {
    assert.equal(ops.find((o) => o.id === id).state, 'INSUFFICIENT_DATA', id);
  }
});

test('event straddle: cheap + estimate earnings is blocked; confirmed near earning is actionable', () => {
  const est = O.evaluateOpportunities(base({ earningsIsEstimate: true, earningsSessions: 10 })).find((o) => o.id === 'event-straddle');
  assert.equal(est.state, 'WATCH');
  assert.ok(est.gates.some((g) => g.includes('estimate')));
  const ok = O.evaluateOpportunities(base({ earningsSessions: 10 })).find((o) => o.id === 'event-straddle');
  assert.equal(ok.state, 'ACTIONABLE');
});

test('rankOpportunities puts actionable first and degrades gracefully', () => {
  const ops = O.evaluateOpportunities(base());
  const ranked = O.rankOpportunities(ops);
  assert.equal(ranked.length, 6);
  assert.equal(ranked[0].state, 'ACTIONABLE');
  assert.equal(O.isMet('met'), true);
  assert.equal(O.isMet('insufficient-data'), false);
});

test('GEX enrichment: short gamma supports longGamma & directionalTrap while flagging short-vol caution', () => {
  const ops = O.evaluateOpportunities(base({
    trap: {
      ...base().trap,
      available: true,
      regime: 'short',
      totalGex: -1.8e8,
      flipDistPct: -0.024,
      flipDistSigma: 0.65,
      gexPerAdv: -0.018,
    },
  }));
  const dt = ops.find((o) => o.id === 'directional-trap');
  const lg = ops.find((o) => o.id === 'long-gamma');
  const ph = ops.find((o) => o.id === 'premium-harvest');

  assert.equal(dt.state, 'ACTIONABLE');
  assert.equal(dt.confidence, 'high');
  assert.equal(dt.evidence.view, 'options');
  assert.equal(dt.evidence.params.fno, 'gex');
  assert.match(dt.trigger.measured, /Short γ/);
  assert.match(dt.trigger.measured, /-2\.4% vs flip/);
  assert.match(dt.size.scenario, /GEX\/ADV -1\.8%/);

  assert.equal(lg.state, 'ACTIONABLE');
  assert.equal(lg.confidence, 'high');
  assert.match(lg.trigger.measured, /Short γ/);

  assert.equal(ph.state, 'WATCH');
  assert.equal(ph.confidence, 'low');
  assert.ok(ph.gates.some((g) => g.includes('dealer short γ')));
});

test('GEX enrichment: long gamma supports premiumHarvest, pins longGamma, and blocks directionalTrap', () => {
  const ops = O.evaluateOpportunities(base({
    trap: {
      ...base().trap,
      available: true,
      regime: 'long',
      totalGex: 2.2e8,
      flipDistPct: 0.032,
      flipDistSigma: 0.85,
      gexPerAdv: 0.022,
    },
  }));
  const dt = ops.find((o) => o.id === 'directional-trap');
  const lg = ops.find((o) => o.id === 'long-gamma');
  const ph = ops.find((o) => o.id === 'premium-harvest');

  assert.equal(ph.state, 'ACTIONABLE');
  assert.equal(ph.confidence, 'high');
  assert.match(ph.size.scenario, /Long γ tailwind/);

  assert.equal(lg.state, 'WATCH');
  assert.equal(lg.confidence, 'low');
  assert.ok(lg.gates.some((g) => g.includes('dealer long γ')));

  assert.equal(dt.state, 'BLOCKED');
  assert.ok(dt.gates.some((g) => g.includes('dealer long γ')));
});

test('GEX enrichment: trap.available=false degrades cleanly and thin GEX/ADV or flip proximity adds caution gates', () => {
  const unavail = O.evaluateOpportunities(base({
    trap: { ...base().trap, available: false, regime: 'short', gexPerAdv: -0.03 },
  }));
  assert.equal(unavail.find((o) => o.id === 'directional-trap').state, 'INSUFFICIENT_DATA');
  assert.equal(unavail.find((o) => o.id === 'premium-harvest').state, 'ACTIONABLE');
  assert.equal(unavail.find((o) => o.id === 'long-gamma').state, 'ACTIONABLE');

  const thinWhip = O.evaluateOpportunities(base({
    trap: {
      ...base().trap,
      available: true,
      regime: 'short',
      flipDistPct: -0.004,
      flipDistSigma: 0.15,
      gexPerAdv: -0.0004,
    },
  }));
  const dt = thinWhip.find((o) => o.id === 'directional-trap');
  assert.equal(dt.state, 'WATCH');
  assert.ok(dt.gates.some((g) => g.includes('GEX/ADV < 0.1%')));
  assert.ok(dt.gates.some((g) => g.includes('0.25σ')));
});
