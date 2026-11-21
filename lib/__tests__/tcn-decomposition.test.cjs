const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { decomposeTimeSeries } = require(path.join(buildDir, 'quant-engine.js'));
const {
  runTcnDecompositionForecast,
  runTcnLookbackSensitivitySweep,
  generateLookbackSweepGrid,
  computeCyclicalSignalToNoise,
  computeHistoricalBacktest,
  buildTcnCacheKey,
  buildTcnSensitivityCacheKey,
  getCachedTcnArtifact,
  getCachedTcnSensitivity,
  getTcnCacheStats,
  clearTcnCache,
  predictNextDayFromCachedWeights,
  DEFAULT_TCN_CONFIG,
  computeReceptiveField,
  computeParameterCount,
} = require(path.join(buildDir, 'tcn-attention.js'));

function makeSyntheticSeries(n = 252) {
  const dates = [];
  const prices = [];
  const baseDate = new Date('2025-11-08T00:00:00Z');
  for (let i = 0; i < n; i++) {
    const d = new Date(baseDate.getTime() + i * 86400 * 1000);
    dates.push(d.toISOString().slice(0, 10));
    // Known multiplicative structure: trend * 50d cycle * 21d seasonal * small noise
    const trend = 150 * Math.exp(0.0008 * i);
    const cycle = Math.exp(0.06 * Math.sin((2 * Math.PI * i) / 50));
    const seasonal = Math.exp(0.015 * Math.cos((2 * Math.PI * i) / 21));
    const noise = Math.exp(0.004 * Math.sin(i * 1.73) * Math.cos(i * 0.61));
    prices.push(trend * cycle * seasonal * noise);
  }
  return { dates, prices };
}

test('decomposeTimeSeries defaults to multiplicative and satisfies exact T*C*S*I = Y', () => {
  const { dates, prices } = makeSyntheticSeries(252);
  const res = decomposeTimeSeries(dates, prices);

  assert.equal(res.model, 'multiplicative');
  assert.equal(res.period, 21);
  assert.equal(res.cyclical.length, dates.length);
  assert.equal(res.trend.length, dates.length);
  assert.equal(res.seasonal.length, dates.length);
  assert.equal(res.residual.length, dates.length);

  for (let i = 0; i < dates.length; i++) {
    const reconstructed = res.trend[i] * res.cyclical[i] * res.seasonal[i] * res.residual[i];
    assert.ok(
      Math.abs(reconstructed - prices[i]) < 1e-8,
      `Multiplicative identity failed at i=${i}: ${reconstructed} vs ${prices[i]}`
    );
  }

  // Dominant Fourier cycle should closely detect the injected 50-day wave
  assert.ok(res.dominantCyclePeriod !== null, 'Expected dominantCyclePeriod to be detected');
  assert.ok(
    Math.abs(res.dominantCyclePeriod - 50) <= 10,
    `Expected dominant cycle near 50d, got ${res.dominantCyclePeriod}`
  );
  assert.ok(res.cyclePowerShare > 0 && res.cyclePowerShare <= 1);
});

test('decomposeTimeSeries additive satisfies exact T+C+S+I = Y', () => {
  const { dates, prices } = makeSyntheticSeries(200);
  const res = decomposeTimeSeries(dates, prices, 21, 'additive', 25, 126);

  assert.equal(res.model, 'additive');
  for (let i = 0; i < dates.length; i++) {
    const reconstructed = res.trend[i] + res.cyclical[i] + res.seasonal[i] + res.residual[i];
    assert.ok(
      Math.abs(reconstructed - prices[i]) < 1e-8,
      `Additive identity failed at i=${i}: ${reconstructed} vs ${prices[i]}`
    );
  }
});

test('TCN-Attention trains deterministically, converges, and reconstructs next-day multiplicative price', async () => {
  const { dates, prices } = makeSyntheticSeries(220);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'multiplicative', 25, 126);

  const rf = computeReceptiveField(DEFAULT_TCN_CONFIG.dilations, DEFAULT_TCN_CONFIG.kernelSize);
  assert.equal(rf, 15);
  assert.ok(computeParameterCount(DEFAULT_TCN_CONFIG) > 0);

  const forecast = await runTcnDecompositionForecast('TEST', decomp, DEFAULT_TCN_CONFIG);
  assert.ok(forecast !== null, 'Expected forecast object');
  assert.equal(forecast.model, 'multiplicative');

  // Exact multiplicative reconstruction
  const { trend, cyclical, seasonal, residual } = forecast.components;
  const prod =
    trend.predictedValue *
    cyclical.predictedValue *
    seasonal.predictedValue *
    residual.predictedValue;
  assert.ok(
    Math.abs(prod - forecast.predictedPrice) < 1e-9,
    `Reconstructed price mismatch: ${prod} vs ${forecast.predictedPrice}`
  );

  // Exact dollar & percentage attribution sums
  const sumDollar =
    trend.contributionDollar +
    cyclical.contributionDollar +
    seasonal.contributionDollar +
    residual.contributionDollar;
  const sumPct =
    trend.contributionPct +
    cyclical.contributionPct +
    seasonal.contributionPct +
    residual.contributionPct;
  assert.ok(Math.abs(sumDollar - forecast.deltaDollar) < 1e-6);
  assert.ok(Math.abs(sumPct - forecast.deltaPct) < 1e-6);

  // Strictly positive & ordered 99% prediction interval
  assert.ok(forecast.lower99 > 0);
  assert.ok(forecast.lower99 <= forecast.predictedPrice);
  assert.ok(forecast.predictedPrice <= forecast.upper99);

  // Cyclical Signal-to-Noise Ratio populated and finite
  assert.ok(forecast.signalToNoise !== undefined);
  assert.ok(Number.isFinite(forecast.signalToNoise.ratio) && forecast.signalToNoise.ratio >= 0);
  assert.ok(Number.isFinite(forecast.signalToNoise.db));
  assert.ok(
    ['Strong Signal', 'Moderate Signal', 'Noise-Dominated'].includes(
      forecast.signalToNoise.regime
    )
  );
  // Causal 3-state HMM volatility regime feature state populated
  assert.ok(forecast.hmmRegime, 'Expected causal HMM regime summary on Decomposition TCN');
  assert.ok(['Calm', 'Watchful', 'Crisis'].includes(forecast.hmmRegime.dominantLabel));
  assert.ok(Math.abs(forecast.hmmRegime.probs.reduce((a, b) => a + b, 0) - 1) < 1e-6);

  // Strictly OOS trailing <= 30-day backtest signals populated with bounded weights
  assert.ok(Array.isArray(forecast.backtestSignals));
  assert.ok(forecast.backtestSignals.length > 0 && forecast.backtestSignals.length <= 30);
  for (const sig of forecast.backtestSignals) {
    assert.ok(Number.isFinite(sig.weight) && sig.weight >= -1 && sig.weight <= 1);
    assert.ok(Number.isFinite(sig.predictedReturn) && Number.isFinite(sig.actualReturn));
  }

  // Attention weights sum to 1 and training loss decreases
  for (const key of ['trend', 'cyclical', 'seasonal', 'residual']) {
    const comp = forecast.components[key];
    const sumW = comp.attentionWeights.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sumW - 1) < 1e-6, `Attention weights for ${key} sum to ${sumW}`);
    assert.equal(comp.attentionByHead.length, DEFAULT_TCN_CONFIG.numHeads);
    for (const headW of comp.attentionByHead) {
      const hSum = headW.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(hSum - 1) < 1e-6);
    }
    const startLoss = comp.trainLossCurve[0];
    const minLoss = Math.min(...comp.trainLossCurve);
    assert.ok(
      minLoss <= startLoss,
      `Expected ${key} loss to not diverge: start=${startLoss}, min=${minLoss}`
    );
  }
});

test('TCN-Attention additive reconstruction & attribution identities hold', async () => {
  const { dates, prices } = makeSyntheticSeries(180);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'additive', 25, 126);

  const forecast = await runTcnDecompositionForecast('TEST_ADD', decomp, {
    ...DEFAULT_TCN_CONFIG,
    epochs: 25,
  });
  assert.ok(forecast !== null);
  assert.equal(forecast.model, 'additive');

  const { trend, cyclical, seasonal, residual } = forecast.components;
  const sumPred =
    trend.predictedValue +
    cyclical.predictedValue +
    seasonal.predictedValue +
    residual.predictedValue;
  assert.ok(Math.abs(sumPred - forecast.predictedPrice) < 1e-9);

  const sumDollar =
    trend.contributionDollar +
    cyclical.contributionDollar +
    seasonal.contributionDollar +
    residual.contributionDollar;
  assert.ok(Math.abs(sumDollar - forecast.deltaDollar) < 1e-6);
  assert.ok(forecast.lower99 <= forecast.predictedPrice && forecast.predictedPrice <= forecast.upper99);
});

test('generateLookbackSweepGrid builds symmetric, clamped, deduplicated 5-point lookback grids', () => {
  // Standard symmetric case: L=24, step=4, n=220 -> [16, 20, 24, 28, 32]
  assert.deepEqual(generateLookbackSweepGrid(24, 4, 220), [16, 20, 24, 28, 32]);
  assert.deepEqual(generateLookbackSweepGrid(24, 2, 220), [20, 22, 24, 26, 28]);
  assert.deepEqual(generateLookbackSweepGrid(24, 6, 220), [12, 18, 24, 30, 36]);

  // Lower bound clamping near minL=8: shifts interior so 5 distinct windows remain
  const lowGrid = generateLookbackSweepGrid(10, 4, 220);
  assert.equal(lowGrid.length, 5);
  assert.ok(lowGrid.includes(10));
  assert.ok(lowGrid.every((v) => v >= 8));
  assert.equal(new Set(lowGrid).size, 5);

  // Upper bound clamping near maxL = floor(120 * 0.45) = 54
  const highGrid = generateLookbackSweepGrid(48, 6, 120);
  assert.equal(highGrid.length, 5);
  assert.ok(highGrid.includes(48));
  assert.ok(highGrid.every((v) => v <= 54));
  assert.equal(new Set(highGrid).size, 5);
});

test('runTcnLookbackSensitivitySweep produces consistent 5-window stability stats and 100% component variance share', async () => {
  const { dates, prices } = makeSyntheticSeries(180);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'multiplicative', 25, 126);

  const sweep = await runTcnLookbackSensitivitySweep(
    'SENS',
    decomp,
    { ...DEFAULT_TCN_CONFIG, lookback: 24, epochs: 16 },
    4
  );

  assert.ok(sweep !== null);
  assert.equal(sweep.runs.length, 5);
  assert.equal(sweep.baseLookback, 24);
  assert.deepEqual(
    sweep.runs.map((r) => r.lookback),
    [16, 20, 24, 28, 32]
  );

  const baseRuns = sweep.runs.filter((r) => r.isBase);
  assert.equal(baseRuns.length, 1);
  assert.equal(baseRuns[0].forecast.predictedPrice, sweep.baseForecast.predictedPrice);

  const { summary, componentStability } = sweep;
  assert.ok(summary.minPrice <= summary.meanPrice && summary.meanPrice <= summary.maxPrice);
  assert.ok(summary.stdDevPrice >= 0);
  assert.ok(summary.dispersionToPiRatio >= 0);
  assert.equal(summary.upCount + summary.downCount, sweep.runs.length);
  assert.ok(sweep.runs.some((r) => r.lookback === summary.bestRmseLookback));
  assert.ok(['Stable', 'Moderate', 'Sensitive'].includes(summary.stabilityLabel));

  const shareSum = ['trend', 'cyclical', 'seasonal', 'residual'].reduce(
    (acc, key) => acc + componentStability[key].varianceSharePct,
    0
  );
  assert.ok(Math.abs(shareSum - 100) < 1e-6, `Expected variance share sum 100, got ${shareSum}`);
});

test('computeCyclicalSignalToNoise computes exact variance ratio, dB, and regime labels', () => {
  // Additive exact 4.0x ratio: signal = 2 * noise => Var(signal) = 4 * Var(noise) => +6.02 dB
  const noiseAdd = [-1, 1, -1, 1, -1, 1];
  const sigStrongAdd = noiseAdd.map((v) => v * 2);
  const strong = computeCyclicalSignalToNoise(sigStrongAdd, noiseAdd, 'additive');
  assert.ok(Math.abs(strong.ratio - 4.0) < 1e-9);
  assert.ok(Math.abs(strong.db - 10 * Math.log10(4)) < 1e-9);
  assert.ok(Math.abs(strong.signalSharePct - 80) < 1e-9);
  assert.equal(strong.regime, 'Strong Signal');

  // Additive 1.0x ratio => 0 dB => Moderate Signal
  const moderate = computeCyclicalSignalToNoise(noiseAdd, noiseAdd, 'additive');
  assert.ok(Math.abs(moderate.ratio - 1.0) < 1e-9);
  assert.ok(Math.abs(moderate.db) < 1e-9);
  assert.equal(moderate.regime, 'Moderate Signal');

  // Multiplicative noise-dominated: log-amplitude of cycle = 0.5 * log-amplitude of residual => 0.25x
  const residMult = [Math.exp(0.02), Math.exp(-0.02), Math.exp(0.02), Math.exp(-0.02)];
  const cycWeakMult = [Math.exp(0.01), Math.exp(-0.01), Math.exp(0.01), Math.exp(-0.01)];
  const weak = computeCyclicalSignalToNoise(cycWeakMult, residMult, 'multiplicative');
  assert.ok(Math.abs(weak.ratio - 0.25) < 1e-9);
  assert.ok(Math.abs(weak.db - 10 * Math.log10(0.25)) < 1e-9);
  assert.equal(weak.regime, 'Noise-Dominated');
});

test('computeHistoricalBacktest compounds signal-weighted P&L, friction drag, drawdown, and B&H accurately', () => {
  // 4-day analytical sequence: alternating +2% and -2% actual moves
  const perfectSignals = [
    { date: '2026-10-01', prevPrice: 100, actualPrice: 102, predictedPrice: 102, predictedReturn: 0.02, actualReturn: 0.02, weight: 0.5 },
    { date: '2026-10-02', prevPrice: 102, actualPrice: 99.96, predictedPrice: 99.96, predictedReturn: -0.02, actualReturn: -0.02, weight: -0.5 },
    { date: '2026-10-03', prevPrice: 99.96, actualPrice: 101.9592, predictedPrice: 101.9592, predictedReturn: 0.02, actualReturn: 0.02, weight: 0.5 },
    { date: '2026-10-04', prevPrice: 101.9592, actualPrice: 99.920016, predictedPrice: 99.920016, predictedReturn: -0.02, actualReturn: -0.02, weight: -0.5 },
  ];

  // Zero friction: each day yields +0.5 * 2% = +1.0% => (1.01)^4 - 1 = +4.060401%
  const bt0 = computeHistoricalBacktest(perfectSignals, 0);
  const expectedGrossPct = (Math.pow(1.01, 4) - 1) * 100;
  const expectedBenchPct = (Math.pow(1.02 * 0.98, 2) - 1) * 100;
  assert.equal(bt0.summary.tradingDays, 4);
  assert.ok(Math.abs(bt0.summary.strategyCumPct - expectedGrossPct) < 1e-9);
  assert.ok(Math.abs(bt0.summary.benchmarkCumPct - expectedBenchPct) < 1e-9);
  assert.equal(bt0.summary.winRatePct, 100);
  assert.equal(bt0.summary.maxDrawdownPct, 0);
  assert.equal(bt0.summary.profitFactor, null); // no losing days
  // Turnovers: |0.5-0|=0.5, |-0.5-0.5|=1.0, |0.5-(-0.5)|=1.0, |-0.5-0.5|=1.0 => avg 0.875
  assert.ok(Math.abs(bt0.summary.avgDailyTurnover - 0.875) < 1e-9);

  // Monotonic net return reduction as friction increases 0 -> 5 -> 10 -> 25 bps
  const bt5 = computeHistoricalBacktest(perfectSignals, 5);
  const bt10 = computeHistoricalBacktest(perfectSignals, 10);
  const bt25 = computeHistoricalBacktest(perfectSignals, 25);
  assert.ok(bt0.summary.strategyCumPct > bt5.summary.strategyCumPct);
  assert.ok(bt5.summary.strategyCumPct > bt10.summary.strategyCumPct);
  assert.ok(bt10.summary.strategyCumPct > bt25.summary.strategyCumPct);
  assert.ok(bt25.summary.totalCostDragPct > bt10.summary.totalCostDragPct);

  // Exact day-0 net return check at 10 bps: 0.01 - 0.5 * 0.001 = 0.0095
  assert.ok(Math.abs(bt10.days[0].netReturn - 0.0095) < 1e-12);
  // Exact day-1 net return check at 10 bps: 0.01 - 1.0 * 0.001 = 0.0090
  assert.ok(Math.abs(bt10.days[1].netReturn - 0.009) < 1e-12);

  // Inverted signals: every day loses -1%, 0% win rate, strictly negative drawdown
  const invertedSignals = perfectSignals.map((s) => ({ ...s, weight: -s.weight }));
  const btInv = computeHistoricalBacktest(invertedSignals, 0);
  assert.ok(btInv.summary.strategyCumPct < 0);
  assert.equal(btInv.summary.winRatePct, 0);
  assert.ok(btInv.summary.maxDrawdownPct < 0);
});

test('TCN model weights, results, and predictions cache eliminates redundant training and reproduces exact forward inference', async () => {
  clearTcnCache();
  assert.equal(getTcnCacheStats().forecastCount, 0);

  const { dates, prices } = makeSyntheticSeries(180);
  const decomp = decomposeTimeSeries(dates, prices, 21, 'multiplicative', 25, 126);
  const cfg = { ...DEFAULT_TCN_CONFIG, lookback: 24, epochs: 16 };
  const key = buildTcnCacheKey('CACHE_SYM', decomp, cfg);

  // First call trains and populates model weights + forecast cache
  let progressCallsFirst = 0;
  const first = await runTcnDecompositionForecast('CACHE_SYM', decomp, cfg, {
    onProgress: () => {
      progressCallsFirst++;
    },
  });
  assert.ok(first !== null);
  assert.ok(progressCallsFirst > 0, 'Expected training progress callbacks on cache miss');

  const artifact = getCachedTcnArtifact(key);
  assert.ok(artifact !== null, 'Expected trained artifact in cache');
  assert.equal(artifact.forecast.predictedPrice, first.predictedPrice);
  assert.ok(getTcnCacheStats().totalWeightTensors > 0, 'Expected Float64Array weight tensors cached');

  // Verify cached Float64Array weights run forward inference and reproduce exact t+1 predictions
  const inferred = predictNextDayFromCachedWeights(artifact, decomp);
  assert.ok(
    Math.abs(inferred.predictedPrice - first.predictedPrice) < 1e-10,
    `Cached-weight inference mismatch: ${inferred.predictedPrice} vs ${first.predictedPrice}`
  );
  for (const compKey of ['trend', 'cyclical', 'seasonal', 'residual']) {
    assert.ok(
      Math.abs(inferred.components[compKey] - first.components[compKey].predictedValue) < 1e-10
    );
  }

  // Second call with same inputs hits cache immediately with ZERO training epochs
  let progressCallsSecond = 0;
  const second = await runTcnDecompositionForecast('CACHE_SYM', decomp, cfg, {
    onProgress: () => {
      progressCallsSecond++;
    },
  });
  assert.equal(progressCallsSecond, 0, 'Expected zero training progress on cache hit');
  assert.equal(second, first, 'Expected exact cached forecast reference on hit');

  // Explicit forceRetrain bypasses cache and retrains cleanly
  let progressCallsRetrain = 0;
  const retrained = await runTcnDecompositionForecast('CACHE_SYM', decomp, cfg, {
    forceRetrain: true,
    onProgress: () => {
      progressCallsRetrain++;
    },
  });
  assert.ok(progressCallsRetrain > 0, 'Expected training progress when forceRetrain=true');
  assert.ok(Math.abs(retrained.predictedPrice - first.predictedPrice) < 1e-12);

  // Sensitivity sweep caches both the sweep result and all 5 individual window models
  const sweepKey = buildTcnSensitivityCacheKey('CACHE_SYM', decomp, cfg, 4);
  const sweep1 = await runTcnLookbackSensitivitySweep('CACHE_SYM', decomp, cfg, 4);
  assert.ok(sweep1 !== null);
  assert.ok(getCachedTcnSensitivity(sweepKey) !== null);
  assert.ok(getTcnCacheStats().forecastCount >= 5);

  let sweepHitProgress = 0;
  const sweep2 = await runTcnLookbackSensitivitySweep('CACHE_SYM', decomp, cfg, 4, {
    onProgress: () => {
      sweepHitProgress++;
    },
  });
  assert.equal(sweepHitProgress, 0, 'Expected zero compute on cached sensitivity sweep');
  assert.equal(sweep2, sweep1);

  // Changing a price invalidates the data signature cleanly
  const mutatedPrices = [...prices];
  mutatedPrices[mutatedPrices.length - 1] *= 1.03;
  const mutatedDecomp = decomposeTimeSeries(dates, mutatedPrices, 21, 'multiplicative', 25, 126);
  const mutatedKey = buildTcnCacheKey('CACHE_SYM', mutatedDecomp, cfg);
  assert.notEqual(mutatedKey, key);
  assert.equal(getCachedTcnArtifact(mutatedKey), null);

  clearTcnCache();
});
