const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const {
  HMM_REGIME_TOKENS,
  getHmmRegimeToken,
  extractDominantHmmSeries,
  buildRegimeBandSpans,
} = require(path.join(buildDir, 'hmm-regime-overlay.js'));

const {
  runTcnVolatilityForecast,
  runTcnRiskVarForecast,
  runTcnMultiAssetRegression,
  runTcnCorrelationForecast,
  clearTcnMultiDomainCache,
} = require(path.join(buildDir, 'tcn-multi-domain.js'));

const { decomposeTimeSeries } = require(path.join(buildDir, 'quant-engine.js'));
const {
  runTcnDecompositionForecast,
  computeHistoricalBacktest,
  clearTcnCache,
  DEFAULT_TCN_CONFIG,
} = require(path.join(buildDir, 'tcn-attention.js'));

const {
  selectDefaultRegressionFactors,
  aggregateCausalAttentionTimeline,
  buildCrossModuleRegimeBreakdown,
} = require(path.join(buildDir, 'tcn-diagnostics.js'));

function makeRegimeSwitchingBars(n = 220, seedOffset = 0) {
  const bars = [];
  const baseDate = new Date('2025-11-08T00:00:00Z');
  let price = 140 + seedOffset * 12;

  for (let i = 0; i < n; i++) {
    const d = new Date(baseDate.getTime() + i * 86400 * 1000);
    const date = d.toISOString().slice(0, 10);
    // Inject distinct Calm, Watchful, and Crisis volatility blocks
    const block = Math.floor(i / 35) % 3;
    const sigma = block === 0 ? 0.006 : block === 1 ? 0.016 : 0.038;
    const z =
      Math.sin((i + 1 + seedOffset) * 1.6180339) * 0.9 +
      Math.cos((i + 2 + seedOffset) * 2.7182818) * 0.6;
    const ret = (block === 2 ? -0.002 : 0.0005) + sigma * z;

    const open = price;
    const close = Math.max(5, open * Math.exp(ret));
    const spread = Math.max(0.003, Math.abs(ret) * 1.2 + sigma * 0.6);
    bars.push({
      date,
      open,
      high: Math.max(open, close) * (1 + spread * 0.5),
      low: Math.min(open, close) * Math.max(0.5, 1 - spread * 0.5),
      close,
      volume: Math.round(1_000_000 * (1 + block)),
    });
    price = close;
  }
  return bars;
}

function barsToReturnsAndDates(bars) {
  const returns = [];
  const dates = [];
  for (let i = 1; i < bars.length; i++) {
    returns.push(Math.log(bars[i].close / bars[i - 1].close));
    dates.push(bars[i].date);
  }
  return { returns, dates };
}

const FAST_CFG = { epochs: 10, hiddenChannels: 6, numHeads: 2, lookback: 18 };

test('extractDominantHmmSeries and buildRegimeBandSpans run-length merge states & handle gaps', () => {
  const empty = extractDominantHmmSeries([]);
  assert.deepEqual(empty.states, []);
  assert.deepEqual(empty.probs, []);

  const posteriors = [
    [0.8, 0.15, 0.05],
    [0.7, 0.2, 0.1],
    [0.1, 0.65, 0.25],
    [0.05, 0.15, 0.8],
    [0.02, 0.08, 0.9],
    [0.6, 0.3, 0.1],
  ];
  const { states, probs } = extractDominantHmmSeries(posteriors);
  assert.deepEqual(states, [0, 0, 1, 2, 2, 0]);
  assert.deepEqual(probs, [0.8, 0.7, 0.65, 0.8, 0.9, 0.6]);

  const spans = buildRegimeBandSpans([0, 0, 1, null, undefined, 2, 2, 0]);
  assert.deepEqual(spans, [
    { startIdx: 0, endIdx: 1, state: 0 },
    { startIdx: 2, endIdx: 2, state: 1 },
    { startIdx: 5, endIdx: 6, state: 2 },
    { startIdx: 7, endIdx: 7, state: 0 },
  ]);

  assert.equal(HMM_REGIME_TOKENS.length, 3);
  assert.equal(getHmmRegimeToken(0).label, 'Calm');
  assert.equal(getHmmRegimeToken(1).label, 'Watchful');
  assert.equal(getHmmRegimeToken(2).label, 'Crisis');
  assert.equal(getHmmRegimeToken(null).label, 'Calm');
});

test('Multi-domain TCN modules populate causal HMM state series and exhaustive OOS regimePerformance', async () => {
  clearTcnMultiDomainCache();
  const bA = makeRegimeSwitchingBars(220, 1);
  const bB = makeRegimeSwitchingBars(220, 2);
  const bC = makeRegimeSwitchingBars(220, 3);
  const { returns: rA, dates } = barsToReturnsAndDates(bA);
  const { returns: rB } = barsToReturnsAndDates(bB);
  const { returns: rC } = barsToReturnsAndDates(bC);

  // 1. Volatility TCN
  const vol = await runTcnVolatilityForecast('OVL_V', bA, { config: FAST_CFG });
  assert.ok(vol);
  assert.equal(vol.hmmStates.length, rA.length);
  assert.equal(vol.hmmProbs.length, rA.length);
  assert.equal(vol.regimePerformance.length, 3);
  const volCountSum = vol.regimePerformance.reduce((a, r) => a + r.count, 0);
  assert.equal(volCountSum, vol.horizons[1].holdoutCount);
  for (const row of vol.regimePerformance) {
    if (row.count === 0) {
      assert.equal(row.meanQlike, null);
      assert.equal(row.meanPredVol, null);
    } else {
      assert.ok(Number.isFinite(row.meanQlike));
      assert.ok(Number.isFinite(row.meanPredVol) && row.meanPredVol > 0);
      assert.ok(Number.isFinite(row.meanRealizedVol) && row.meanRealizedVol > 0);
    }
  }

  // 2. Risk VaR TCN
  const risk = await runTcnRiskVarForecast('OVL_R', rA, dates, { config: FAST_CFG });
  assert.ok(risk);
  assert.equal(risk.hmmStates.length, rA.length);
  assert.equal(risk.hmmProbs.length, rA.length);
  assert.equal(risk.dates.length, dates.length);
  assert.equal(risk.oosDailyLossSeries.length, rA.length);
  assert.equal(risk.regimePerformance.length, 3);
  const riskCountSum = risk.regimePerformance.reduce((a, r) => a + r.count, 0);
  assert.equal(riskCountSum, risk.horizons[1].holdoutCount);
  for (const row of risk.regimePerformance) {
    if (row.count === 0) {
      assert.equal(row.exceedance95RatePct, null);
      assert.equal(row.exceedance99RatePct, null);
    } else {
      assert.ok(row.exceedance95RatePct >= 0 && row.exceedance95RatePct <= 100);
      assert.ok(row.exceedance99RatePct >= 0 && row.exceedance99RatePct <= 100);
      assert.ok(Number.isFinite(row.meanVar99) && row.meanVar99 > 0);
    }
  }

  // 3. Multi-Asset Regression TCN
  const reg = await runTcnMultiAssetRegression(
    'OVL_A',
    ['OVL_B', 'OVL_C'],
    { OVL_A: rA, OVL_B: rB, OVL_C: rC },
    dates,
    { config: FAST_CFG }
  );
  assert.ok(reg);
  assert.equal(reg.hmmStates.length, rA.length);
  assert.equal(reg.hmmProbs.length, rA.length);
  assert.equal(reg.oosHistory.length, reg.horizons[1].holdoutCount);
  assert.equal(reg.regimePerformance.length, 3);
  const regCountSum = reg.regimePerformance.reduce((a, r) => a + r.count, 0);
  assert.equal(regCountSum, reg.horizons[1].holdoutCount);
  for (const pt of reg.oosHistory) {
    assert.ok([0, 1, 2].includes(pt.hmmState));
    assert.ok(pt.hmmProb >= 0 && pt.hmmProb <= 1);
    assert.ok(Number.isFinite(pt.predictedReturn) && Number.isFinite(pt.actualReturn));
  }

  // 4. Correlation Dynamics TCN
  const corr = await runTcnCorrelationForecast('OVL_A', 'OVL_B', rA, rB, dates, 22, { config: FAST_CFG });
  assert.ok(corr);
  assert.equal(corr.hmmStates.length, corr.dynamicCorrelations.length);
  assert.equal(corr.hmmProbs.length, corr.dynamicCorrelations.length);
  assert.equal(corr.regimePerformance.length, 3);
  const corrCountSum = corr.regimePerformance.reduce((a, r) => a + r.count, 0);
  assert.equal(corrCountSum, corr.horizons[1].holdoutCount);
  for (const pt of corr.dynamicCorrelations) {
    assert.ok([0, 1, 2].includes(pt.hmmState));
    assert.ok(pt.hmmProb >= 0 && pt.hmmProb <= 1);
  }
});

test('Decomposition backtest stratifies OOS signals strictly by prediction-origin causal HMM state', async () => {
  clearTcnCache();
  const bars = makeRegimeSwitchingBars(210, 4);
  const dates = bars.map((b) => b.date);
  const prices = bars.map((b) => b.close);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'multiplicative');

  const forecast = await runTcnDecompositionForecast('OVL_DEC', decomp, {
    ...DEFAULT_TCN_CONFIG,
    epochs: 10,
    lookback: 18,
    hiddenChannels: 6,
    numHeads: 2,
  });
  assert.ok(forecast);
  assert.ok(forecast.backtestSignals.length > 0);
  for (const sig of forecast.backtestSignals) {
    assert.ok([0, 1, 2].includes(sig.hmmState), `Expected valid hmmState on backtestSignal, got ${sig.hmmState}`);
    assert.ok(Number.isFinite(sig.hmmProb) && sig.hmmProb > 0 && sig.hmmProb <= 1);
  }

  const bt = computeHistoricalBacktest(forecast.backtestSignals, 5);
  assert.equal(bt.regimeBreakdown.length, 3);
  const totalCount = bt.regimeBreakdown.reduce((acc, r) => acc + r.count, 0);
  assert.equal(totalCount, bt.days.length);
  for (const row of bt.regimeBreakdown) {
    if (row.count === 0) {
      assert.equal(row.directionalHitRatePct, null);
      assert.equal(row.rmseDollar, null);
      assert.equal(row.cumNetReturnPct, null);
    } else {
      assert.ok(row.directionalHitRatePct >= 0 && row.directionalHitRatePct <= 100);
      assert.ok(Number.isFinite(row.rmseDollar) && row.rmseDollar >= 0);
      assert.ok(Number.isFinite(row.cumNetReturnPct));
    }
  }
});

test('Unified diagnostics attaches causal HMM regimes to attention lags and builds cross-module regime breakdown', async () => {
  clearTcnMultiDomainCache();
  const bA = makeRegimeSwitchingBars(200, 5);
  const bB = makeRegimeSwitchingBars(200, 6);
  const bC = makeRegimeSwitchingBars(200, 7);

  const aligned = selectDefaultRegressionFactors('OVL_U', bA, {
    OVL_U: bA,
    SPY: bB,
    QQQ: bC,
  });
  assert.ok(aligned);

  const vol = await runTcnVolatilityForecast('OVL_U', bA, { config: FAST_CFG });
  const risk = await runTcnRiskVarForecast('OVL_U', aligned.returnsByTicker.OVL_U, aligned.dates, {
    config: FAST_CFG,
  });
  const reg = await runTcnMultiAssetRegression(
    'OVL_U',
    aligned.independentTickers,
    aligned.returnsByTicker,
    aligned.dates,
    { config: FAST_CFG }
  );

  const timeline = aggregateCausalAttentionTimeline({
    volatility: vol,
    riskVar: risk,
    regression: reg,
  });
  assert.equal(timeline.lagCurves.length, timeline.maxLookback);
  for (const pt of timeline.lagCurves) {
    assert.ok([0, 1, 2].includes(pt.hmmState), `Expected causal hmmState on lag ${pt.lag}`);
    assert.ok(Number.isFinite(pt.hmmProb) && pt.hmmProb > 0 && pt.hmmProb <= 1);
  }

  const crossBreakdown = buildCrossModuleRegimeBreakdown({
    volatility: vol,
    riskVar: risk,
    regression: reg,
    decompBacktest: null,
  });
  assert.equal(crossBreakdown.length, 3);
  assert.deepEqual(
    crossBreakdown.map((r) => r.label),
    ['Calm', 'Watchful', 'Crisis']
  );
  const volSum = crossBreakdown.reduce((a, r) => a + r.volCount, 0);
  const riskSum = crossBreakdown.reduce((a, r) => a + r.riskCount, 0);
  const regSum = crossBreakdown.reduce((a, r) => a + r.regCount, 0);
  assert.equal(volSum, vol.horizons[1].holdoutCount);
  assert.equal(riskSum, risk.horizons[1].holdoutCount);
  assert.equal(regSum, reg.horizons[1].holdoutCount);
});
