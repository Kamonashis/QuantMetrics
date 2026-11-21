const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const {
  MULTI_HORIZONS,
  runTcnVolatilityForecast,
  runTcnRiskVarForecast,
  runTcnMultiAssetRegression,
  runTcnCorrelationForecast,
  interpolateTcnExpectedMove,
  classifyTcnMoveDislocation,
  evaluateTcnConfluence,
  getCachedTcnVolatilityForecast,
  getTcnMultiDomainCacheStats,
  clearTcnMultiDomainCache,
} = require(path.join(buildDir, 'tcn-multi-domain.js'));

const { decomposeTimeSeries } = require(path.join(buildDir, 'quant-engine.js'));
const {
  runTcnDecompositionForecast,
  getLatestCachedTcnArtifactByTicker,
  clearTcnCache,
  DEFAULT_TCN_CONFIG,
} = require(path.join(buildDir, 'tcn-attention.js'));

// Deterministic synthetic OHLCV bars with GARCH-like volatility clustering
function makeSyntheticBars(n = 240, seedOffset = 0) {
  const bars = [];
  const baseDate = new Date('2025-11-08T00:00:00Z');
  let price = 150 + seedOffset * 10;
  let sigma = 0.013;

  for (let i = 0; i < n; i++) {
    const d = new Date(baseDate.getTime() + i * 86400 * 1000);
    const date = d.toISOString().slice(0, 10);
    const z =
      Math.sin((i + 1 + seedOffset) * 1.6180339) * 0.85 +
      Math.cos((i + 1 + seedOffset) * 2.7182818) * 0.65;
    const shock = i % 47 === 20 ? -2.6 : z;
    const ret = 0.0003 + sigma * shock;
    sigma = Math.sqrt(0.00001 + 0.12 * ret * ret + 0.84 * sigma * sigma);

    const open = price;
    const close = Math.max(5, open * Math.exp(ret));
    const spread = Math.max(0.004, Math.abs(ret) * 1.25 + sigma * 0.6);
    const high = Math.max(open, close) * (1 + spread * 0.5);
    const low = Math.min(open, close) * Math.max(0.5, 1 - spread * 0.5);
    bars.push({
      date,
      open,
      high,
      low,
      close,
      volume: Math.round(1_000_000 * (1 + Math.abs(shock) * 0.5)),
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

const FAST_CFG = { epochs: 12, hiddenChannels: 6, numHeads: 2, lookback: 18 };

test('TCN Volatility (1d/5d/22d QLIKE head): positive vol, 99% bounds, strictly OOS series & LRU cache', async () => {
  clearTcnMultiDomainCache();
  const bars = makeSyntheticBars(220, 1);

  const res = await runTcnVolatilityForecast('TESTV', bars, { config: FAST_CFG });
  assert.ok(res, 'Expected non-null volatility forecast');
  assert.equal(res.ticker, 'TESTV');
  assert.equal(res.cacheHit, false);

  for (const h of MULTI_HORIZONS) {
    const hf = res.horizons[h];
    assert.ok(Number.isFinite(hf.annualizedVol) && hf.annualizedVol > 0, `h=${h} vol must be positive`);
    assert.ok(Number.isFinite(hf.dailyVariance) && hf.dailyVariance > 0, `h=${h} variance must be positive`);
    assert.ok(hf.lower99Vol > 0 && hf.lower99Vol <= hf.annualizedVol, `h=${h} lower99Vol <= point`);
    assert.ok(hf.upper99Vol >= hf.annualizedVol, `h=${h} upper99Vol >= point`);
    assert.ok(Number.isFinite(hf.holdoutQlike), `h=${h} OOS QLIKE must be finite`);
    assert.ok(hf.holdoutCount >= 1, `h=${h} must score at least 1 OOS holdout observation`);

    const oosSeries = res.oosDailyVarianceByHorizon[h];
    assert.equal(oosSeries.length, bars.length - 1);
    for (let i = 0; i < Math.min(res.holdoutStartIdx, oosSeries.length); i++) {
      assert.ok(Number.isNaN(oosSeries[i]), `h=${h} index ${i} before holdoutStartIdx must be NaN`);
    }
  }

  assert.ok(res.forecastCone.length >= 22);
  const attnSum = res.attentionWeights.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(attnSum - 1) < 1e-6, `Attention weights must sum to 1, got ${attnSum}`);
  assert.ok(res.hmmRegime, 'Expected causal HMM regime summary on Volatility TCN');
  assert.ok(['Calm', 'Watchful', 'Crisis'].includes(res.hmmRegime.dominantLabel));
  assert.ok(Math.abs(res.hmmRegime.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);

  const cached = getCachedTcnVolatilityForecast('TESTV', bars, FAST_CFG);
  assert.ok(cached, 'Expected synchronous cache hit');
  const res2 = await runTcnVolatilityForecast('TESTV', bars, { config: FAST_CFG });
  assert.equal(res2.cacheHit, true);
  assert.equal(res2.horizons[22].annualizedVol, res.horizons[22].annualizedVol);
  assert.ok(getTcnMultiDomainCacheStats().volatility >= 1);
});

test('TCN Risk VaR & ES (1d/5d/22d Pinball head): strict tail monotonicity & Basel/Kupiec/Christoffersen backtest', async () => {
  clearTcnMultiDomainCache();
  const bars = makeSyntheticBars(240, 2);
  const { returns, dates } = barsToReturnsAndDates(bars);

  const res = await runTcnRiskVarForecast('TESTR', returns, dates, { config: FAST_CFG });
  assert.ok(res, 'Expected non-null risk VaR forecast');
  assert.equal(res.ticker, 'TESTR');
  assert.equal(res.backtestRow.key, 'tcn-attention');

  for (const h of MULTI_HORIZONS) {
    const hf = res.horizons[h];
    assert.ok(hf.var95 > 0 && Number.isFinite(hf.var95), `h=${h} VaR95 > 0`);
    assert.ok(hf.var99 >= hf.var95, `h=${h} VaR99 >= VaR95`);
    assert.ok(hf.es95 >= hf.var95, `h=${h} ES95 >= VaR95`);
    assert.ok(hf.es99 >= hf.var99, `h=${h} ES99 >= VaR99`);
    assert.ok(hf.tailRatio >= 1 - 1e-9, `h=${h} ES99/VaR99 >= 1`);
  }

  assert.ok(res.horizons[1].holdoutCount >= 10);
  assert.ok(['Green', 'Yellow', 'Red'].includes(res.backtestRow.baselZone));
  assert.ok(Number.isFinite(res.backtestRow.kupiecPVal95));
  assert.ok(Number.isFinite(res.backtestRow.christoffersenPVal99));
  assert.equal(res.oosVar95Series.length, returns.length);
  assert.equal(res.oosVar99Series.length, returns.length);
  assert.ok(res.hmmRegime, 'Expected causal HMM regime summary on Risk VaR TCN');
  assert.ok(Math.abs(res.hmmRegime.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});

test('TCN Multi-Asset Regression (1d/5d/22d): OOS metrics, OLS lift & permutation importance sum to ~100%', async () => {
  clearTcnMultiDomainCache();
  const bDep = makeSyntheticBars(220, 3);
  const bF1 = makeSyntheticBars(220, 4);
  const bF2 = makeSyntheticBars(220, 5);
  const { returns: rF1, dates } = barsToReturnsAndDates(bF1);
  const { returns: rF2 } = barsToReturnsAndDates(bF2);
  const { returns: rNoise } = barsToReturnsAndDates(bDep);

  const rDep = rNoise.map((e, i) => {
    const lag1 = i > 0 ? rF1[i - 1] : 0;
    const lag2 = i > 1 ? rF2[i - 2] : 0;
    return 0.45 * lag1 - 0.25 * lag2 + 0.5 * e;
  });

  const res = await runTcnMultiAssetRegression(
    'DEP',
    ['FAC1', 'FAC2'],
    { DEP: rDep, FAC1: rF1, FAC2: rF2 },
    dates,
    { config: FAST_CFG }
  );
  assert.ok(res, 'Expected non-null multi-asset regression result');
  assert.equal(res.dependentTicker, 'DEP');
  assert.deepEqual(res.independentTickers, ['FAC1', 'FAC2']);

  for (const h of MULTI_HORIZONS) {
    const hf = res.horizons[h];
    assert.ok(Number.isFinite(hf.predictedReturn));
    assert.ok(hf.lower99 <= hf.predictedReturn && hf.upper99 >= hf.predictedReturn);
    assert.ok(Number.isFinite(hf.oosR2) && Number.isFinite(hf.olsOosR2));
    assert.ok(Math.abs(hf.nonlinearLiftR2 - (hf.oosR2 - hf.olsOosR2)) < 1e-9);
    assert.ok(hf.directionalAccuracyPct >= 0 && hf.directionalAccuracyPct <= 100);
  }

  assert.equal(res.factorImportance.length, 4);
  assert.ok(
    res.factorImportance.some((f) => f.ticker === 'HMM Regimes' && f.isHmmRegime === true),
    'Expected HMM Regimes group in regression permutation importance'
  );
  assert.ok(res.hmmRegime, 'Expected causal HMM regime summary on Regression TCN');
  assert.ok(Math.abs(res.hmmRegime.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  const impSum = res.factorImportance.reduce((s, f) => s + f.importancePct, 0);
  assert.ok(Math.abs(impSum - 100) < 0.5, `Permutation factor importance should sum to ~100%, got ${impSum}`);
});

test('TCN Correlation Dynamics (1d/5d/22d Fisher-z head): strict (-1, 1) bounds & spillover alert', async () => {
  clearTcnMultiDomainCache();
  const bA = makeSyntheticBars(220, 6);
  const bB = makeSyntheticBars(220, 7);
  const { returns: rA, dates } = barsToReturnsAndDates(bA);
  const { returns: rRaw } = barsToReturnsAndDates(bB);
  const rB = rA.map((v, i) => 0.55 * v + 0.65 * rRaw[i]);

  const res = await runTcnCorrelationForecast('AAA', 'BBB', rA, rB, dates, 22, { config: FAST_CFG });
  assert.ok(res, 'Expected non-null correlation forecast');
  assert.ok(res.currentCorr > -1 && res.currentCorr < 1);

  for (const h of MULTI_HORIZONS) {
    const hf = res.horizons[h];
    assert.ok(hf.predictedCorr > -1 && hf.predictedCorr < 1, `h=${h} corr must lie in (-1, 1)`);
    assert.ok(hf.lower99 > -1 && hf.lower99 <= hf.predictedCorr);
    assert.ok(hf.upper99 < 1 && hf.upper99 >= hf.predictedCorr);
    assert.ok(Number.isFinite(hf.oosMae) && hf.oosMae >= 0);
  }

  assert.equal(res.forecastCone.length, 22);
  for (const pt of res.forecastCone) {
    assert.ok(pt.correlation > -1 && pt.correlation < 1);
    assert.ok(pt.lower99 > -1 && pt.upper99 < 1);
    assert.ok(pt.lower99 <= pt.correlation && pt.correlation <= pt.upper99);
  }
  assert.ok(['ELEVATED', 'WATCH', 'CALM'].includes(res.spilloverAlert.level));
  assert.ok(res.hmmRegime, 'Expected causal joint HMM regime summary on Correlation TCN');
  assert.ok(Math.abs(res.hmmRegime.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);
});

test('F&O Variance-Time Interpolation, Dislocation Classifier & Edge/T8 Signal Confluence', async () => {
  clearTcnMultiDomainCache();
  clearTcnCache();
  const bars = makeSyntheticBars(220, 8);
  const volRes = await runTcnVolatilityForecast('CONF', bars, { config: FAST_CFG });
  assert.ok(volRes);

  const move5 = interpolateTcnExpectedMove(volRes, 5);
  assert.equal(move5.extrapolated, false);
  assert.ok(Math.abs(move5.annualizedVol - volRes.horizons[5].annualizedVol) < 1e-6);
  assert.ok(move5.move99Pct > move5.sigmaMovePct && move5.sigmaMovePct > move5.straddleEqMovePct);

  const move45 = interpolateTcnExpectedMove(volRes, 45);
  assert.equal(move45.extrapolated, true);
  assert.ok(move45.sigmaMovePct > move5.sigmaMovePct);

  assert.equal(classifyTcnMoveDislocation(5.0, 3.5).verdict, 'RICH vs TCN');
  assert.equal(classifyTcnMoveDislocation(2.5, 3.5).verdict, 'CHEAP vs TCN');
  assert.equal(classifyTcnMoveDislocation(3.6, 3.5).verdict, 'ALIGNED');
  assert.equal(classifyTcnMoveDislocation(null, 3.5).verdict, null);

  const dates = bars.map((b) => b.date);
  const prices = bars.map((b) => b.close);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'multiplicative');
  const decompForecast = await runTcnDecompositionForecast('CONF', decomp, {
    ...DEFAULT_TCN_CONFIG,
    epochs: 10,
    lookback: 18,
    hiddenChannels: 6,
    numHeads: 2,
  });
  const peeked = getLatestCachedTcnArtifactByTicker('CONF');
  assert.ok(peeked, 'Expected ticker-based decomposition cache lookup to succeed');
  assert.equal(peeked.forecast.ticker, 'CONF');

  const iv30High = volRes.horizons[22].annualizedVol + 0.08;
  const confShort = evaluateTcnConfluence({
    iv30: iv30High,
    rv20: volRes.horizons[22].annualizedVol,
    volForecast: volRes,
    decompForecast,
    primaryOrientation: 'short-premium',
  });
  assert.equal(confShort.volRegimeRead, 'SHORT-PREMIUM FAVORED');
  assert.equal(confShort.alignment, 'ALIGNED');
  assert.equal(confShort.hasVolArtifact, true);
  assert.equal(confShort.hasDecompArtifact, true);
  assert.ok(Math.abs(confShort.forwardVrp22 - (iv30High - volRes.horizons[22].annualizedVol)) < 1e-12);

  const confDivergent = evaluateTcnConfluence({
    iv30: iv30High,
    rv20: volRes.horizons[22].annualizedVol,
    volForecast: volRes,
    decompForecast,
    primaryOrientation: 'long-gamma',
  });
  assert.equal(confDivergent.alignment, 'DIVERGENT');
});
