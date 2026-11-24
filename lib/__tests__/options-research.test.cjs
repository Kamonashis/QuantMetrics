const test = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, rmSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const { join } = require('node:path');
const { tmpdir } = require('node:os');
const B = require('../../.cache/test-build/options-research.js');
const S = require('../../.cache/test-build/probability-scoring.js');
const C = require('../../.cache/test-build/credit-bridge.js');
const O = require('../../.cache/test-build/options.js');
const R = require('../../.cache/test-build/rnd.js');
const I = require('../../.cache/test-build/rnd-integrity.js');

const spot = 100, rate = 0.04, dividendYield = 0.01, t = 23 / 365, vol = 0.675;
const forward = spot * Math.exp((rate - dividendYield) * t);
const date = '2026-11-24T15:00:00.000Z';
const fit = { a: vol ** 2 * t, b: 0, rho: 0, m: 0, sigma: 0.1, arbitrageFlags: [], rmseIv: 0, nUsed: 150 };
const price = (strike, type) => O.bsPrice({ spot, rate, dividendYield, vol, t, strike, type });
const makeChain = (lo = 25, hi = 250) => ({ available: true, quote: { price: spot, asOf: date, prevClose: 105.58 }, rate, dividendYield,
  rnd: O.riskNeutralDensity(fit, spot, forward, t, rate, dividendYield),
  slices: [{ t, dte: 23, forward, expirationDate: '2026-12-17', atmIv: vol, mfivAnnualized: vol, svi: fit,
    points: Array.from({ length: hi - lo + 1 }, (_, i) => {
      const strike = lo + i, type = strike >= forward ? 'call' : 'put';
      return { strike, type, mid: price(strike, type), relSpread: 0.02 };
    }) }] });

test('23-calendar-day flat smile: log moments, digital, spread and normalization agree with BSM', () => {
  const chain = makeChain();
  const data = I.buildRndResearch(chain, chain.quote);
  assert.equal(data.quality, 'READY');
  assert.ok(Math.abs(R.densityMean(data.fitted, () => 1) - 1) < 1e-10);
  assert.ok(Math.abs(data.fittedMoments.std - vol * Math.sqrt(t)) < 0.0002);
  assert.ok(Math.abs(data.fittedMoments.mean - (rate - dividendYield - vol ** 2 / 2) * t) < 0.0002);
  assert.ok(Math.abs(data.atmDispersion - 0.169446517274) < 0.0001);
  const discount = Math.exp(-rate * t);
  for (const [i, idx] of [[0, 0], [1, 2], [2, 4]]) {
    const row = data.rows[i], prob = data.probabilities[idx];
    assert.equal(row.fitted, discount * prob.fitted);
    assert.ok(Math.abs(row.fitted - row.bs) < 0.0002);
  }
  assert.ok(Math.abs(data.rows[3].market - data.rows[3].bs) < 0.02);
  assert.ok(Math.abs(data.probabilities[1].bs - data.probabilities[2].bs) > 0.02, 'spot and −10% are separate barriers');
  assert.ok(data.touchDown10 >= data.probabilities[2].bs);
});

test('unit area is not coverage: truncated and noisy market curves are withheld separately from the fit', () => {
  const chain = makeChain(92, 112);
  const short = I.buildRndResearch(chain, chain.quote);
  assert.ok(short.market);
  assert.ok(Math.abs(R.densityMean(short.market, () => 1) - 1) < 1e-10);
  assert.ok(short.market.rawArea < 0.7);
  assert.equal(short.rows[2].market, null);
  assert.ok(short.rows[2].fitted > 0);
  assert.equal(short.rows[2].inRange, false);
  const noisy = makeChain(75, 140);
  noisy.slices[0].points.forEach((p, i) => { p.mid *= i % 2 ? 1.05 : 0.95; });
  const data = I.buildRndResearch(noisy, noisy.quote);
  assert.ok(data.market.clippedFraction > 0.05);
  assert.equal(data.hybrid, null);
  assert.notEqual(data.quality, 'READY');
  const gap = I.buildRndResearch(noisy, { price: 105.58, asOf: date });
  assert.equal(gap.quality, 'NOT_READY');
  assert.equal(gap.rows[0].fitted, null);
  const time = I.buildRndResearch(makeChain(), { price: 100, asOf: '2026-11-23T15:00:00.000Z' });
  assert.equal(time.checks[0].pass, false);
});

test('a valid but finite market range reports nonzero modeled wing payoff share', () => {
  const chain = makeChain(70, 138);
  const data = I.buildRndResearch(chain, chain.quote);
  assert.ok(data.hybrid);
  assert.ok(data.hybrid.rightTailMass > 0);
  assert.ok(data.rows[0].tailShare > 0);
  assert.ok(data.rows[0].tailShare < 1);
});

test('SVI rejects invalid variance and keeps negative curvature visible for gating', () => {
  assert.equal(O.riskNeutralDensity({ ...fit, a: -1 }, spot, forward, t, rate, dividendYield), null);
  assert.equal(O.riskNeutralDensity({ ...fit, b: 2 }, spot, forward, t, rate, dividendYield), null);
  const chain = makeChain();
  chain.rnd.clippedFraction = 0.45;
  assert.equal(I.buildRndResearch(chain, chain.quote).fitted, null);
});

test('historical calendar-window samples and completed session RV refuse insufficient data', () => {
  const bars = [
    { date: '2026-11-19', close: 100 }, { date: '2026-11-20', close: 110 },
    { date: '2026-11-23', close: 120 }, { date: '2026-11-24', close: 132 },
  ];
  assert.deepEqual(I.historicalHorizonReturns(bars, 4), [Math.log(1.2), Math.log(1.2)]);
  assert.equal(I.sampleMoments([0, 1]), null);
  assert.equal(B.trailingResearchRv(bars, date), null);
  assert.equal(B.usMarketDate('2026-11-24T01:00:00Z'), '2026-11-23');
  assert.equal(B.completedResearchBars(bars, '2026-11-24').length, 3);
});

test('panel reconstruction is dated, distinctly labelled and never invents digital forecasts', () => {
  const complete = { date: '2026-11-23', computedAt: '2026-11-23T19:00:00Z', spot: 100, atmIv30: 0.4, rv20: 0.8 };
  const rows = B.reconstructPanelObservations([complete, { ...complete, rv20: null }, { ...complete, computedAt: date }], '2026-11-24', date);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].origin, 'panel-reconstruction');
  assert.equal(rows[0].vrp.verdict, 'CHEAP');
  assert.equal(rows[0].digital, null);
  assert.equal(rows[0].quality, null);
});

test('research store freezes the first current-session record and does not overwrite prospective observations with panel rows', () => {
  const temp = mkdtempSync(join(tmpdir(), 'options-research-test-'));
  try {
    const store = require.resolve('../../.cache/test-build/options-research-store.js');
    const iv = require.resolve('../../.cache/test-build/iv-store.js');
    execFileSync(process.execPath, ['-e', `
      const assert = require('node:assert/strict');
      const { writeOptionsResearch, readOptionsResearch } = require(${JSON.stringify(store)});
      const { writePanelRow } = require(${JSON.stringify(iv)});
      const now = new Date('2026-11-24T15:02:00Z');
      const symbol = 'NVDA';
      assert.equal(writePanelRow(symbol, { date: '2026-11-23', computedAt: '2026-11-23T18:00:00Z', spot: 90, atmIv30: 0.6, rv20: 0.4 }).ok, true);
      const v = { asOf: '2026-11-24T15:00:00Z', spot: 100, quality: 'READY', vrp: { iv30: 0.55, rv20: 0.7 },
        digital: { expiry: '2026-12-18', svi: 0.23, atm: 0.2 } };
      assert.throws(() => writeOptionsResearch(symbol, { ...v, asOf: '2026-11-23T15:00:00Z' }, now));
      assert.deepEqual(readOptionsResearch(symbol), []);
      assert.equal(writeOptionsResearch(symbol, v, now).written, true);
      assert.equal(writeOptionsResearch(symbol, { ...v, spot: 90 }, now).written, false);
      const [old, fresh] = readOptionsResearch(symbol);
      assert.equal(old.origin, 'panel-reconstruction');
      assert.equal(old.digital, null);
      assert.equal(fresh.origin, 'prospective');
      assert.equal(fresh.spot, 100);
      assert.ok(Math.abs(fresh.digital.strike - 110) < 1e-10);
    `], { cwd: temp });
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});

test('Brier skill uses paired samples, exact expiry resolves, and small counts withhold calibration', () => {
  const rows = [{ probability: 0.8, outcome: 1, baseline: 0.5 }, { probability: 0.1, outcome: 0, baseline: null }];
  assert.ok(Math.abs(S.brierScore(rows) - 0.025) < 1e-12);
  assert.equal(S.brierSkillScore(rows).n, 1);
  assert.ok(Math.abs(S.brierSkillScore(rows).skill - 0.84) < 1e-12);
  assert.equal(S.reliabilityBins([{ probability: 1, outcome: 1 }]).at(-1).n, 1);
  const rec = { date: '2026-11-23', spot: 100, origin: 'prospective', digital: { expiry: '2026-11-25', strike: 110, svi: 0.8, atm: 0.5 } };
  const bars = [{ date: '2026-11-23', close: 100 }, { date: '2026-11-24', close: 130 }, { date: '2026-11-26', close: 130 }];
  assert.equal(B.scoreOptionsResearch([rec], bars).digitalPending, 1);
  const results = B.scoreOptionsResearch([rec], [bars[0], bars[1], { date: '2026-11-25', close: 100 }, bars[2]]);
  assert.equal(results.digitalN, 1);
  assert.equal(results.brier, null);
  assert.deepEqual(results.bins, []);
});

test('Merton proxy refuses fabricated inputs and ranks by model fragility under shared assumptions', () => {
  assert.equal(C.creditBridge(null, 0.6), null);
  assert.equal(C.creditBridge(100, 0.6, 0), null);
  const quiet = C.creditBridge(100, 0.4), stressed = C.creditBridge(100, 0.8);
  assert.equal(quiet.debt, 30);
  assert.ok(quiet.dd > stressed.dd);
  assert.ok(quiet.pd < stressed.pd);
});
