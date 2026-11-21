const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { decomposeTimeSeries } = require(path.join(buildDir, 'quant-engine.js'));
const {
  runTcnDecompositionForecast,
  runTcnLookbackSensitivitySweep,
  generateLookbackSweepGrid,
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
