const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const {
  selectDefaultRegressionFactors,
  buildUnifiedOosScorecard,
  buildHorizonComparisonMatrix,
  aggregateCausalAttentionTimeline,
  summarizeCacheAndLatency,
} = require(path.join(buildDir, 'tcn-diagnostics.js'));

const {
  runTcnVolatilityForecast,
  runTcnRiskVarForecast,
  runTcnMultiAssetRegression,
  evaluateTcnConfluence,
  getTcnMultiDomainCacheStats,
  clearTcnMultiDomainCache,
} = require(path.join(buildDir, 'tcn-multi-domain.js'));

function makeBars(n = 200, offset = 0) {
  const bars = [];
  const base = new Date('2025-11-08T00:00:00Z');
  let p = 120 + offset * 15;
  let s = 0.014;
  for (let i = 0; i < n; i++) {
    const date = new Date(base.getTime() + i * 86400 * 1000).toISOString().slice(0, 10);
    const z = Math.sin((i + 1 + offset) * 1.414) * 0.9 + Math.cos((i + 2 + offset) * 2.236) * 0.6;
    const ret = 0.0002 + s * (i % 41 === 19 ? -2.5 : z);
    s = Math.sqrt(0.00001 + 0.11 * ret * ret + 0.85 * s * s);
    const open = p;
    const close = Math.max(5, open * Math.exp(ret));
    const spread = Math.max(0.004, Math.abs(ret) * 1.2 + s * 0.5);
    bars.push({
      date,
      open,
      high: Math.max(open, close) * (1 + spread * 0.5),
      low: Math.min(open, close) * Math.max(0.5, 1 - spread * 0.5),
      close,
      volume: 1_000_000,
    });
    p = close;
  }
  return bars;
}

const FAST_CFG = { epochs: 12, hiddenChannels: 6, numHeads: 2, lookback: 18 };

test('selectDefaultRegressionFactors aligns common dates and excludes dependent ticker', () => {
  const aapl = makeBars(180, 1);
  const spy = makeBars(180, 2);
  const qqq = makeBars(180, 3);
  const msft = makeBars(180, 4);

  const alignedAapl = selectDefaultRegressionFactors('AAPL', aapl, {
    AAPL: aapl,
    SPY: spy,
    QQQ: qqq,
    MSFT: msft,
  });
  assert.ok(alignedAapl);
  assert.equal(alignedAapl.dependentTicker, 'AAPL');
  assert.deepEqual(alignedAapl.independentTickers, ['SPY', 'QQQ']);
  assert.equal(alignedAapl.dates.length, 179);
  assert.equal(alignedAapl.returnsByTicker.AAPL.length, 179);
  assert.equal(alignedAapl.returnsByTicker.SPY.length, 179);

  // When SPY is the dependent ticker, it must not include itself
  const alignedSpy = selectDefaultRegressionFactors('SPY', spy, {
    AAPL: aapl,
    SPY: spy,
    QQQ: qqq,
    MSFT: msft,
  });
  assert.ok(alignedSpy);
  assert.deepEqual(alignedSpy.independentTickers, ['QQQ', 'MSFT']);
});

test('Unified diagnostics builders aggregate OOS scorecard, 1d/5d/22d matrix, causal attention & cache telemetry', async () => {
  clearTcnMultiDomainCache();
  const aapl = makeBars(200, 1);
  const spy = makeBars(200, 2);
  const qqq = makeBars(200, 3);

  const aligned = selectDefaultRegressionFactors('AAPL', aapl, {
    AAPL: aapl,
    SPY: spy,
    QQQ: qqq,
  });
  assert.ok(aligned);

  const vol = await runTcnVolatilityForecast('AAPL', aapl, { config: FAST_CFG });
  const risk = await runTcnRiskVarForecast('AAPL', aligned.returnsByTicker.AAPL, aligned.dates, {
    config: FAST_CFG,
  });
  const reg = await runTcnMultiAssetRegression(
    'AAPL',
    aligned.independentTickers,
    aligned.returnsByTicker,
    aligned.dates,
    { config: FAST_CFG }
  );
  assert.ok(vol && risk && reg);

  const iv30 = vol.horizons[22].annualizedVol + 0.05;
  const confluence = evaluateTcnConfluence({
    iv30,
    rv20: vol.horizons[22].annualizedVol,
    volForecast: vol,
    decompForecast: null,
    primaryOrientation: 'short-premium',
  });
  const statuses = {
    volatility: 'ready',
    risk_var: 'ready',
    regression: 'ready',
    confluence: 'ready',
  };

  // 1. OOS Scorecard
  const scorecard = buildUnifiedOosScorecard({
    volatility: vol,
    riskVar: risk,
    regression: reg,
    confluence,
    statuses,
  });
  assert.equal(scorecard.length, 4);
  assert.deepEqual(
    scorecard.map((r) => r.module),
    ['volatility', 'risk_var', 'regression', 'confluence']
  );
  assert.equal(scorecard[3].verdict, 'ALIGNED');
  assert.equal(scorecard[3].verdictTone, 'positive');

  // 2. 1d / 5d / 22d Horizon Matrix
  const matrix = buildHorizonComparisonMatrix({
    volatility: vol,
    riskVar: risk,
    regression: reg,
    iv30,
  });
  assert.equal(matrix.length, 3);
  assert.deepEqual(
    matrix.map((m) => m.horizon),
    [1, 5, 22]
  );
  for (const row of matrix) {
    assert.ok(row.volAnnualized > 0);
    assert.ok(row.es99 >= row.var99 && row.var99 >= row.var95 && row.var95 > 0);
    assert.ok(Number.isFinite(row.regReturn));
  }
  assert.ok(matrix[2].vrpNote.includes('IV30'));

  // 3. Causal Attention Timeline & Joint Shock Dates
  const timeline = aggregateCausalAttentionTimeline({
    volatility: vol,
    riskVar: risk,
    regression: reg,
  });
  assert.equal(timeline.activeModules.length, 3);
  assert.equal(timeline.lagCurves.length, timeline.maxLookback);
  // Consensus weights across all lags should sum closely to 1
  const consensusSum = timeline.lagCurves.reduce((acc, p) => acc + p.consensusWeight, 0);
  assert.ok(Math.abs(consensusSum - 1) < 1e-5, `Expected consensus attention sum ~1, got ${consensusSum}`);
  assert.ok(timeline.jointShockDates.length > 0);
  // Sorted descending by moduleCount then combinedWeight
  for (let i = 1; i < timeline.jointShockDates.length; i++) {
    const prev = timeline.jointShockDates[i - 1];
    const curr = timeline.jointShockDates[i];
    assert.ok(
      prev.moduleCount > curr.moduleCount ||
        (prev.moduleCount === curr.moduleCount && prev.combinedWeight >= curr.combinedWeight - 1e-9)
    );
  }

  // 4. Cache & Latency Telemetry
  const telemetry = summarizeCacheAndLatency({
    volatility: vol,
    riskVar: risk,
    regression: reg,
    decompForecast: null,
    statuses,
    multiDomainCounts: getTcnMultiDomainCacheStats(),
    decompStats: { forecastCount: 0, sensitivityCount: 0, totalWeightTensors: 0 },
  });
  assert.equal(telemetry.rows.length, 4);
  assert.equal(telemetry.readyTrainableCount, 3);
  assert.ok(telemetry.totalParameters > 0);
  assert.ok(telemetry.multiDomainTotal >= 3);
});
