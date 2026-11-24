// Unit tests for lib/correlation-engine.ts (compiled output in .cache/test-build).
// Fixtures: deterministic — no network, no clock.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const C = require(path.join(BUILD, 'correlation-engine.js'));

const approx = (a, b, eps = 1e-3) => assert.ok(Math.abs(a - b) < eps, `expected ${b}, got ${a}`);

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
function gaussian(rand) {
  const u = Math.max(1e-12, rand());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
function dates(n) {
  const out = [];
  const d = new Date(Date.UTC(2020, 0, 1));
  while (out.length < n) {
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) out.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}
function barsFromReturns(ds, rets, start = 100) {
  let p = start;
  const bars = [{ date: ds[0], open: p, high: p, low: p, close: p, volume: 0 }];
  for (let i = 1; i < ds.length; i++) {
    p *= 1 + rets[i - 1];
    bars.push({ date: ds[i], open: p, high: p, low: p, close: p, volume: 0 });
  }
  return bars;
}
/** k assets driven by one common factor: r_i = load·f + sqrt(1−load²)·e_i, scaled by vol. */
function factorSample(seed, n, k, load, vols) {
  const rand = lcg(seed);
  const returns = Array.from({ length: k }, () => new Array(n));
  for (let t = 0; t < n; t++) {
    const f = gaussian(rand);
    for (let i = 0; i < k; i++) returns[i][t] = (vols?.[i] ?? 0.01) * (load * f + Math.sqrt(1 - load * load) * gaussian(rand));
  }
  return { tickers: Array.from({ length: k }, (_, i) => `A${i}`), dates: dates(n), returns, returnType: 'simple', audit: {} };
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

test('buildCorrelationSample joins on dates and refuses instead of simulating', () => {
  const ds = dates(140);
  const rand = lcg(4);
  const mk = () => barsFromReturns(ds, Array.from({ length: 139 }, () => 0.01 * gaussian(rand)));
  const a = mk();
  const bFull = mk();
  const b = bFull.filter((_, i) => i !== 70 && i >= 4);
  const ok = C.buildCorrelationSample({ AAA: a, BBB: b }, ['aaa', 'BBB'], { returnType: 'simple' });
  assert.equal(ok.ok, true);
  assert.equal(ok.sample.audit.commonPriceDates, 135);
  assert.equal(ok.sample.returns[0].length, 134);
  assert.equal(ok.sample.audit.droppedSessions, 5);
  const gap = ok.sample.dates.indexOf(ds[71]);
  approx(ok.sample.returns[1][gap], bFull[71].close / bFull[69].close - 1, 1e-12);

  const missing = C.buildCorrelationSample({ AAA: a }, ['AAA', 'ZZZ'], { returnType: 'simple' });
  assert.equal(missing.ok, false);
  assert.deepEqual(missing.audit.missing, ['ZZZ']);
  const short = C.buildCorrelationSample({ AAA: a.slice(0, 30), BBB: b.slice(0, 30) }, ['AAA', 'BBB'], { returnType: 'log' });
  assert.equal(short.ok, false);
  assert.equal(C.buildCorrelationSample({ AAA: a }, ['AAA'], { returnType: 'log' }).ok, false);
});

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

test('jacobiEigen reproduces known eigenpairs', () => {
  const two = C.jacobiEigen([[1, 0.6], [0.6, 1]]);
  approx(two.values[0], 1.6, 1e-10);
  approx(two.values[1], 0.4, 1e-10);
  const A = [[4, 1, 2], [1, 3, 0.5], [2, 0.5, 5]];
  const e = C.jacobiEigen(A);
  for (let k = 0; k < 3; k++) {
    const v = e.vectors[k];
    for (let i = 0; i < 3; i++) approx(A[i].reduce((s, a, j) => s + a * v[j], 0), e.values[k] * v[i], 1e-9);
  }
  approx(e.values.reduce((a, b) => a + b, 0), 12, 1e-9);
});

test('basket summary: effective bets spans 1 (identical) to N (independent)', () => {
  const ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const ones = [[1, 1, 1], [1, 1, 1], [1, 1, 1]];
  approx(C.effectiveBets(ident), 3, 1e-12);
  approx(C.effectiveBets(ones), 1, 1e-12);
  const half = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]];
  approx(C.effectiveBets(half), 3 / (1 + 2 * 0.5), 1e-12);
  const s = C.summarizeBasket(half, ['A', 'B', 'C']);
  approx(s.avgCorr, 0.5, 1e-12);
  approx(s.topFactorShare, 2 / 3, 1e-9);
});

test('ledoitWolf shrinks toward constant correlation with delta in [0, 1]', () => {
  // Two blocks of three: correlations are NOT constant, so the target is only partly right.
  const blocks = (seed, n) => {
    const rand = lcg(seed);
    const returns = Array.from({ length: 6 }, () => new Array(n));
    for (let t = 0; t < n; t++) {
      const f = [gaussian(rand), gaussian(rand)];
      for (let i = 0; i < 6; i++) returns[i][t] = 0.01 * (0.8 * f[i < 3 ? 0 : 1] + 0.6 * gaussian(rand));
    }
    return { returns };
  };
  const s = blocks(8, 120);
  const lw = C.ledoitWolf(s.returns);
  assert.ok(lw.delta > 0 && lw.delta < 1, `delta=${lw.delta}`);
  const raw = C.correlationMatrix(s.returns);
  const avg = C.averageCorrelation(raw);
  // Every shrunk correlation lies between the raw value and the average.
  for (let i = 0; i < 6; i++) {
    for (let j = i + 1; j < 6; j++) {
      const lo = Math.min(raw[i][j], avg) - 1e-9;
      const hi = Math.max(raw[i][j], avg) + 1e-9;
      assert.ok(lw.corr[i][j] >= lo && lw.corr[i][j] <= hi);
    }
  }
  // More data ⇒ less shrinkage.
  assert.ok(C.ledoitWolf(blocks(8, 3000).returns).delta < lw.delta / 5);
  // When correlations really are constant, the target is right and shrinkage is full.
  assert.ok(C.ledoitWolf(factorSample(8, 3000, 6, 0.6).returns).delta > 0.9);
  assert.equal(C.ledoitWolf(factorSample(8, 200, 2, 0.6).returns).delta, 0);
});

test('partial correlation removes a common driver', () => {
  const rand = lcg(12);
  const n = 4000;
  const m = Array.from({ length: n }, () => gaussian(rand));
  const a = m.map((v) => v + 0.5 * gaussian(rand));
  const b = m.map((v) => v + 0.5 * gaussian(rand));
  const corr = C.correlationMatrix([a, b, m]);
  assert.ok(corr[0][1] > 0.7);
  const partial = C.partialCorrelationMatrix(corr);
  assert.ok(Math.abs(partial[0][1]) < 0.06);
  assert.equal(C.partialCorrelationMatrix([[1, 1], [1, 1]]), null);
});

test('principal factors: one-factor data has one factor above the noise edge', () => {
  const s = factorSample(3, 1500, 6, 0.7);
  const { factors, noiseEdge } = C.principalFactors(C.correlationMatrix(s.returns), 1500);
  assert.equal(factors[0].aboveNoise, true);
  assert.equal(factors[0].sameSign, true);
  assert.equal(factors[1].aboveNoise, false);
  approx(noiseEdge, (1 + Math.sqrt(6 / 1500)) ** 2, 1e-12);
  approx(factors[0].share, (1 + 5 * 0.49) / 6, 0.04);
});

// ---------------------------------------------------------------------------
// Through time
// ---------------------------------------------------------------------------

test('jennrichTest: accepts equal matrices, rejects a correlation break', () => {
  const same = factorSample(21, 1200, 5, 0.6);
  const R1 = C.correlationMatrix(same.returns.map((r) => r.slice(0, 600)));
  const R2 = C.correlationMatrix(same.returns.map((r) => r.slice(600)));
  const t0 = C.jennrichTest(R1, 600, R2, 600);
  assert.equal(t0.df, 10);
  assert.ok(t0.p > 0.01, `p=${t0.p}`);
  const other = factorSample(22, 600, 5, 0.2);
  const t1 = C.jennrichTest(R1, 600, C.correlationMatrix(other.returns), 600);
  assert.ok(t1.p < 1e-6);
});

test('compareRecentToPrior and matrixDrift detect a regime change', () => {
  const calm = factorSample(31, 500, 4, 0.3);
  const tight = factorSample(32, 126, 4, 0.9);
  const s = { ...calm, dates: dates(626), returns: calm.returns.map((r, i) => [...r, ...tight.returns[i]]) };
  const cmp = C.compareRecentToPrior(s, 126);
  assert.ok(cmp.recentAvg > cmp.priorAvg + 0.5);
  assert.ok(cmp.test.p < 1e-6);
  assert.ok(cmp.pairs[0].p < 1e-6);
  const drift = C.matrixDrift(s, 126);
  const last = drift.drift[drift.drift.length - 1];
  assert.ok(C.percentileRank(drift.drift, last) > 0.9);
  assert.equal(C.compareRecentToPrior(calm, 300), null);
});

test('rollingPair matches a direct window correlation and brackets it', () => {
  const s = factorSample(41, 300, 2, 0.5);
  const roll = C.rollingPair(s.returns[0], s.returns[1], s.dates, 60);
  assert.equal(roll.corr.length, 241);
  approx(roll.corr[10], C.pearson(s.returns[0].slice(10, 70), s.returns[1].slice(10, 70)), 1e-9);
  assert.ok(roll.low[10] < roll.corr[10] && roll.corr[10] < roll.high[10]);
  const basket = C.rollingBasket(factorSample(42, 400, 4, 0.5), 126);
  assert.equal(basket.dates.length, basket.avgCorr.length);
  assert.ok(basket.effectiveBets.every((v) => v >= 1 && v <= 4.5));
});

test('fitDcc: flags time-varying correlation only when it exists', () => {
  const rand = lcg(77);
  const n = 1200;
  const x = [];
  const y = [];
  for (let t = 0; t < n; t++) {
    const rho = Math.floor(t / 200) % 2 === 0 ? 0.1 : 0.85;
    const a = gaussian(rand);
    const b = rho * a + Math.sqrt(1 - rho * rho) * gaussian(rand);
    x.push(0.01 * a);
    y.push(0.01 * b);
  }
  const fit = C.fitDcc(x, y);
  assert.ok(fit.a > 0 && fit.a + fit.b < 1);
  assert.ok(fit.lrP < 1e-4);
  const lowAvg = fit.path.slice(100, 200).reduce((a, b) => a + b, 0) / 100;
  const highAvg = fit.path.slice(300, 400).reduce((a, b) => a + b, 0) / 100;
  assert.ok(highAvg > lowAvg + 0.3);
  assert.ok(fit.path.every((v) => v > -1 && v < 1));

  const flat = factorSample(78, 1200, 2, 0.6);
  const constant = C.fitDcc(flat.returns[0], flat.returns[1]);
  assert.ok(constant.lrP > 0.01, `lrP=${constant.lrP}`);
  assert.equal(C.fitDcc(x.slice(0, 100), y.slice(0, 100)), null);
});

test('blockBootstrapInterval is deterministic and covers the estimate', () => {
  const s = factorSample(51, 800, 2, 0.6);
  const r = C.pearson(s.returns[0], s.returns[1]);
  const a = C.blockBootstrapInterval(s.returns[0], s.returns[1]);
  const b = C.blockBootstrapInterval(s.returns[0], s.returns[1]);
  assert.deepEqual(a, b);
  assert.ok(a.low < r && r < a.high);
  assert.ok(a.high - a.low < 0.2);
});

// ---------------------------------------------------------------------------
// Stress
// ---------------------------------------------------------------------------

test('bivariate normal benchmarks match closed forms and simulation', () => {
  approx(C.bivariateNormalLowerOrthant(-2, 0), 0.0227501 ** 2, 1e-6);
  approx(C.bivariateNormalLowerOrthant(0, 0.5), 0.25 + Math.asin(0.5) / (2 * Math.PI), 1e-5);
  approx(C.bivariateNormalLowerOrthant(-2, 0.64), 0.00619, 2e-4);
  approx(C.normalExceedanceCorrelation(0.2, 0), 0, 1e-6);
  // Monte Carlo cross-check at rho = 0.6, lower 30%.
  const rand = lcg(99);
  const xs = [];
  const ys = [];
  for (let i = 0; i < 400000; i++) {
    const a = gaussian(rand);
    const b = 0.6 * a + 0.8 * gaussian(rand);
    if (a < -0.5244 && b < -0.5244) {
      xs.push(a);
      ys.push(b);
    }
  }
  approx(C.normalExceedanceCorrelation(0.3, 0.6), C.pearson(xs, ys), 0.015);
  // Conditioning on a tail lowers correlation even when the true correlation is constant.
  assert.ok(C.normalExceedanceCorrelation(0.1, 0.6) < C.normalExceedanceCorrelation(0.5, 0.6));
  assert.ok(C.normalExceedanceCorrelation(0.5, 0.6) < 0.6);
});

test('exceedance correlation: normal data matches the benchmark, crash-coupled data exceeds it', () => {
  const normal = factorSample(61, 20000, 2, 0.6);
  const e = C.exceedanceCorrelations(normal.returns[0], normal.returns[1], [0.2, 0.5]);
  approx(e.points[0].lower, e.points[0].normal, 0.05);
  approx(e.points[0].upper, e.points[0].normal, 0.05);

  const rand = lcg(62);
  const x = [];
  const y = [];
  for (let t = 0; t < 20000; t++) {
    const crash = rand() < 0.08;
    const f = crash ? -2.5 - Math.abs(gaussian(rand)) : gaussian(rand);
    const w = crash ? 0.95 : 0.3;
    x.push(w * f + Math.sqrt(1 - w * w) * gaussian(rand));
    y.push(w * f + Math.sqrt(1 - w * w) * gaussian(rand));
  }
  const c = C.exceedanceCorrelations(x, y, [0.1]);
  assert.ok(c.points[0].lower > c.points[0].normal + 0.15);
  assert.ok(c.points[0].lower > c.points[0].upper + 0.15);
  assert.equal(C.exceedanceCorrelations(x.slice(0, 80), y.slice(0, 80), [0.1]).points[0].lower, null);
});

test('tail dependence: comonotonic = 1, independent ≈ q, normal data ≈ normal benchmark', () => {
  const rand = lcg(71);
  const x = Array.from({ length: 5000 }, () => gaussian(rand));
  approx(C.tailDependence(x, x.map((v) => 2 * v + 1), 0.05).lower, 1, 0.02);
  const ind = C.tailDependence(x, Array.from({ length: 5000 }, () => gaussian(rand)), 0.05);
  approx(ind.lower, 0.05, 0.04);
  approx(ind.normal, 0.05, 0.02);
  const s = factorSample(72, 20000, 2, 0.8);
  const td = C.tailDependence(s.returns[0], s.returns[1], 0.05);
  approx(td.lower, td.normal, 0.05);
  const pairs = C.basketTailDependence(factorSample(73, 2000, 4, 0.5), 0.05);
  assert.equal(pairs.length, 6);
});

// ---------------------------------------------------------------------------
// Portfolio
// ---------------------------------------------------------------------------

test('riskDecomposition annualises exactly once and contributions add up', () => {
  // 2% daily vol, uncorrelated.
  const cov = [[0.0004, 0], [0, 0.0004]];
  const rd = C.riskDecomposition([1, 1], cov);
  approx(rd.assets[0].vol, 0.02 * Math.sqrt(252), 1e-12);
  approx(rd.vol, (0.02 / Math.sqrt(2)) * Math.sqrt(252), 1e-12);
  approx(rd.effectiveBets, 2, 1e-9);
  const cov3 = [[0.0004, 0.0002, 0.0001], [0.0002, 0.0009, 0.0003], [0.0001, 0.0003, 0.0001]];
  const r3 = C.riskDecomposition([50, 30, 20], cov3);
  approx(r3.assets.reduce((a, x) => a + x.contribution, 0), r3.vol, 1e-12);
  approx(r3.assets.reduce((a, x) => a + x.riskShare, 0), 1, 1e-12);
  approx(r3.assets.reduce((a, x) => a + x.weight, 0), 1, 1e-12);
});

test('tailRisk components sum to VaR and expected shortfall', () => {
  const s = factorSample(81, 1500, 4, 0.5, [0.01, 0.02, 0.015, 0.03]);
  const cov = C.covarianceMatrix(s.returns);
  const tr = C.tailRisk(s, [0.4, 0.3, 0.2, 0.1], cov, 0.95);
  approx(tr.componentVaR.reduce((a, b) => a + b, 0), tr.parametricVaR, 1e-12);
  approx(tr.componentES.reduce((a, b) => a + b, 0), tr.expectedShortfall, 1e-12);
  assert.equal(tr.tailDays, 75);
  assert.ok(tr.expectedShortfall > tr.historicalVaR);
  approx(tr.parametricVaR, tr.historicalVaR, 0.004);
});

test('allocations: each satisfies its defining property', () => {
  const s = factorSample(91, 3000, 5, 0.5, [0.008, 0.012, 0.02, 0.03, 0.015]);
  const cov = C.covarianceMatrix(s.returns);
  const vol = (w) => C.riskDecomposition(w, cov).vol;

  const erc = C.equalRiskContribution(cov);
  const shares = C.riskDecomposition(erc, cov).assets.map((a) => a.riskShare);
  shares.forEach((v) => approx(v, 0.2, 1e-6));

  const mv = C.minVarianceLongOnly(cov);
  approx(mv.reduce((a, b) => a + b, 0), 1, 1e-12);
  assert.ok(mv.every((w) => w >= 0));
  for (const id of ['equal', 'inverseVol', 'erc', 'hrp', 'maxDiv']) {
    const w = C.allocationWeights(id, cov);
    approx(w.reduce((a, b) => a + b, 0), 1, 1e-9);
    assert.ok(w.every((x) => x >= 0));
    assert.ok(vol(w) >= vol(mv) - 1e-12, `${id} beat min variance`);
  }
  const md = C.maxDiversification(cov);
  const dr = (w) => C.riskDecomposition(w, cov).diversificationRatio;
  for (const id of ['equal', 'inverseVol', 'erc', 'hrp', 'minVar']) assert.ok(dr(md) >= dr(C.allocationWeights(id, cov)) - 1e-9);

  // Long-only constraint binds: the unconstrained solution would short asset 2.
  const tight = [[0.0001, 0.00019, 0], [0.00019, 0.0004, 0], [0, 0, 0.0004]];
  const w = C.minVarianceLongOnly(tight);
  approx(w[1], 0, 1e-12);
  approx(w[0], 0.8, 1e-9);
  approx(w[2], 0.2, 1e-9);

  // HRP on a diagonal covariance is inverse-variance.
  const diag = [[0.0001, 0, 0, 0], [0, 0.0004, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0004]];
  const hrp = C.hierarchicalRiskParity(diag);
  approx(hrp[0], 0.4, 1e-9);
  approx(hrp[1], 0.1, 1e-9);
});

test('correlationStress rises monotonically to the weighted-average volatility', () => {
  const s = factorSample(101, 2000, 4, 0.5, [0.01, 0.02, 0.015, 0.03]);
  const cov = C.covarianceMatrix(s.returns);
  const w = [0.25, 0.25, 0.25, 0.25];
  const rows = C.correlationStress(w, cov);
  const rd = C.riskDecomposition(w, cov);
  approx(rows[0].vol, rd.vol, 1e-12);
  approx(rows[rows.length - 1].vol, rd.weightedAvgVol, 1e-12);
  for (let i = 1; i < rows.length; i++) assert.ok(rows[i].vol > rows[i - 1].vol);
});

test('hedgeTable: variance reduction is rho squared', () => {
  const s = factorSample(111, 3000, 3, 0.7);
  const rows = C.hedgeTable(s, 0);
  assert.equal(rows.length, 2);
  approx(rows[0].varianceReduction, rows[0].corr ** 2, 1e-12);
  approx(rows[0].hedgedVol, rows[0].vol * Math.sqrt(1 - rows[0].corr ** 2), 1e-12);
});

test('walkForwardAllocations uses only past data and reports realised volatility', () => {
  const s = factorSample(121, 900, 4, 0.5, [0.008, 0.012, 0.02, 0.03]);
  const wf = C.walkForwardAllocations(s, 252, 21);
  assert.equal(wf.rebalances, Math.floor((900 - 252) / 21));
  assert.equal(wf.rows.length, 6);
  const byId = Object.fromEntries(wf.rows.map((r) => [r.id, r]));
  assert.ok(byId.minVar.realizedVol < byId.equal.realizedVol);
  assert.ok(wf.rows.every((r) => r.realizedVol > 0.05 && r.realizedVol < 0.5));
  assert.equal(C.walkForwardAllocations(factorSample(122, 280, 3, 0.5), 252, 21), null);
});

// ---------------------------------------------------------------------------
// Structure, intraday, export
// ---------------------------------------------------------------------------

test('residualCorrelation strips the market; MST links the tightest pairs', () => {
  const rand = lcg(131);
  const n = 4000;
  const m = Array.from({ length: n }, () => gaussian(rand));
  const sector = Array.from({ length: n }, () => gaussian(rand));
  const a = m.map((v, t) => v + 0.8 * sector[t] + 0.5 * gaussian(rand));
  const b = m.map((v, t) => v + 0.8 * sector[t] + 0.5 * gaussian(rand));
  const c = m.map((v) => v + 0.9 * gaussian(rand));
  const s = { tickers: ['A', 'B', 'C', 'MKT'], dates: dates(n), returns: [a, b, c, m], returnType: 'simple', audit: {} };
  const res = C.residualCorrelation(s, 3);
  assert.deepEqual(res.tickers, ['A', 'B', 'C']);
  assert.ok(res.matrix[0][1] > 0.6);
  assert.ok(Math.abs(res.matrix[0][2]) < 0.08);
  const pc = C.residualCorrelation(s, null);
  assert.equal(pc.tickers.length, 4);
  const tree = C.minimumSpanningTree(res.matrix, res.tickers);
  assert.equal(tree.edges.length, 2);
  assert.deepEqual([tree.edges[0].a, tree.edges[0].b], ['A', 'B']);
});

test('leadLag names a leader only when one asset predicts the other beyond its own past', () => {
  const rand = lcg(141);
  const n = 2000;
  const x = Array.from({ length: n }, () => gaussian(rand));
  const y = x.map((_, t) => (t >= 2 ? 0.6 * x[t - 2] : 0) + gaussian(rand));
  const ll = C.leadLag(x, y, 5);
  assert.equal(ll.leader, 'A');
  assert.equal(ll.aLeadsB.peakLag, 2);
  approx(ll.aLeadsB.peakCoef, 0.6, 0.06);
  assert.ok(ll.bLeadsA.p > 0.025);
  assert.ok(Math.abs(ll.lags.find((l) => l.lag === -2).corr) > 0.4);
  const none = C.leadLag(x, Array.from({ length: n }, () => gaussian(rand)), 5);
  assert.equal(none.leader, null);

  // A shared, mean-reverting driver makes lagged cross-correlations non-zero
  // in both directions; that is not one asset leading the other.
  let f = 0;
  const a = [];
  const b = [];
  for (let t = 0; t < 4000; t++) {
    const e = gaussian(rand);
    const common = e - 0.5 * f;
    f = e;
    a.push(common + 0.3 * gaussian(rand));
    b.push(common + 0.3 * gaussian(rand));
  }
  const shared = C.leadLag(a, b, 3);
  assert.ok(Math.abs(shared.lags.find((l) => l.lag === -1).corr) > 0.2);
  assert.notEqual(shared.leader, 'A');
  assert.notEqual(shared.leader, 'B');
});

test('intradayCorrelationBySampling matches on timestamp and shows the Epps effect', () => {
  const rand = lcg(151);
  const A = [];
  const B = [];
  for (let d = 0; d < 3; d++) {
    let pa = 100;
    let pb = 50;
    let prev = 0;
    for (let m = 0; m < 390; m++) {
      const f = 0.0005 * gaussian(rand);
      pa *= Math.exp(f);
      pb *= Math.exp(prev + 0.00005 * gaussian(rand)); // B reacts one minute late
      prev = f;
      const base = { date: `2026-01-0${d + 1}`, time: '', open: 0, high: 0, low: 0, volume: 0, timestamp: d * 1e6 + m };
      A.push({ ...base, minuteOfDay: m, close: pa });
      if (!(d === 0 && m < 10)) B.push({ ...base, minuteOfDay: m, close: pb });
    }
  }
  const res = C.intradayCorrelationBySampling(A, B, [1, 5, 30]);
  assert.equal(res.days, 3);
  assert.equal(res.overlappingBars, 1160);
  assert.ok(Math.abs(res.rows[0].corr) < 0.15);
  assert.ok(res.rows[1].corr > 0.6);
  assert.ok(res.rows[2].corr > res.rows[1].corr);
  const self = C.intradayCorrelationBySampling(A, A, [1]);
  approx(self.rows[0].corr, 1, 1e-12);
  assert.equal(C.intradayCorrelationBySampling(A.slice(0, 20), B.slice(0, 20), [5]).rows[0].corr, null);
});

test('exports: LaTeX math delimiters are balanced and CSV is rectangular', () => {
  const m = [[1, 0.5], [0.5, 1]];
  const tex = C.correlationLatex(['A', 'B'], m);
  assert.equal((tex.match(/\$/g) || []).length % 2, 0);
  assert.match(tex, /\$0\.50\$/);
  const csv = C.correlationCsv(['A', 'B'], m).split('\n');
  assert.equal(csv.length, 3);
  assert.ok(csv.every((r) => r.split(',').length === 3));
});

test('horizonCorrelations aggregates non-overlapping returns', () => {
  const s = factorSample(161, 1000, 3, 0.5);
  const rows = C.horizonCorrelations(s, [1, 5, 21, 63]);
  assert.deepEqual(rows.map((r) => r.days), [1, 5, 21]);
  assert.equal(rows[1].observations, 200);
  approx(rows[0].avgCorr, C.averageCorrelation(C.correlationMatrix(s.returns)), 1e-12);
});
