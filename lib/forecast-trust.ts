// Walk-forward forecast validation for the models the terminal actually offers.
// Parameters are fit on the train slice; each test variance is a one-step-ahead
// recursion (no refit), scored against the realized squared return. Pure.

import { computeEwmaVolatility, fitGarch11, fitGjrGarch } from './quant-engine';
import { fitGaussianHmm } from './hmm';
import { qlike, mseLogVariance } from './vol-analytics';

export interface ForecastScore {
  model: string;
  qlike: number | null;
  mseLog: number | null;
  n: number;
}

export interface ForecastTrust {
  nTrain: number;
  nTest: number;
  scores: ForecastScore[]; // best (lowest QLIKE) first
}

const meanOf = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

/** One held-out tail comparison: EWMA 0.94, GARCH(1,1), GJR-GARCH, HMM k=3, historical. */
export function walkForwardForecastScores(returns: number[], testFraction = 0.25): ForecastTrust | null {
  const clean = returns.filter((r) => isFinite(r));
  const n = clean.length;
  if (n < 200) return null;
  const split = Math.floor(n * (1 - Math.min(0.5, Math.max(0.1, testFraction))));
  const train = clean.slice(0, split);
  const test = clean.slice(split);
  if (train.length < 100 || test.length < 20) return null;

  const trainMean = meanOf(train);
  const epsTrain = train.map((r) => (r - trainMean) ** 2);
  const trainVar = meanOf(epsTrain);
  if (!(trainVar > 0)) return null;

  const garch = fitGarch11(train).summary;
  const gjr = fitGjrGarch(train).summary;
  const ewma = computeEwmaVolatility(train, 0.94).summary.lambda;

  const features = (rs: number[]) => rs.map((r) => [r, Math.abs(r), Math.sqrt(r * r)]);
  const hmm = fitGaussianHmm(features(train), { k: 3 });
  const hmmVar = hmm
    ? hmm.stationary.reduce((s, w, k) => s + w * (hmm.stateStdDevs[k]?.[0] ?? 0) ** 2, 0)
    : null;

  const ewmaVar = (() => {
    let s2 = trainVar;
    const out: number[] = [];
    for (let t = split; t < n; t++) {
      s2 = ewma * s2 + (1 - ewma) * (clean[t - 1] - trainMean) ** 2;
      out.push(Math.max(1e-12, s2));
    }
    return out;
  })();

  const garchVar = (() => {
    let s2 = Math.max(1e-12, trainVar);
    const out: number[] = [];
    for (let t = split; t < n; t++) {
      const eps2 = (clean[t - 1] - trainMean) ** 2;
      s2 = garch.omega + garch.alpha * eps2 + garch.beta * s2;
      out.push(Math.max(1e-12, s2));
    }
    return out;
  })();

  const gjrVar = (() => {
    let s2 = Math.max(1e-12, trainVar);
    const out: number[] = [];
    for (let t = split; t < n; t++) {
      const rPrev = clean[t - 1] - trainMean;
      const eps2 = rPrev * rPrev;
      s2 = gjr.omega + (gjr.alpha + (rPrev < 0 ? gjr.gamma : 0)) * eps2 + gjr.beta * s2;
      out.push(Math.max(1e-12, s2));
    }
    return out;
  })();

  const candidates: { model: string; forecasts: number[] | null }[] = [
    { model: 'EWMA λ=0.94', forecasts: ewmaVar },
    { model: 'GARCH(1,1)', forecasts: garchVar },
    { model: 'GJR-GARCH', forecasts: gjrVar },
    { model: 'HMM k=3', forecasts: hmmVar && hmmVar > 0 ? new Array(test.length).fill(hmmVar) : null },
    { model: 'Historical variance', forecasts: new Array(test.length).fill(trainVar) },
  ];

  const scores: ForecastScore[] = candidates.map(({ model, forecasts }) => {
    if (!forecasts || forecasts.length !== test.length) {
      return { model, qlike: null, mseLog: null, n: 0 };
    }
    let ql = 0;
    let ml = 0;
    let nOk = 0;
    for (let i = 0; i < test.length; i++) {
      if (!(forecasts[i] > 0)) continue;
      ql += qlike(test[i] ** 2, forecasts[i]);
      ml += mseLogVariance(test[i] ** 2, forecasts[i]);
      nOk++;
    }
    return nOk > 0
      ? { model, qlike: ql / nOk, mseLog: ml / nOk, n: nOk }
      : { model, qlike: null, mseLog: null, n: 0 };
  });

  scores.sort((a, b) => {
    if (a.qlike === null) return 1;
    if (b.qlike === null) return -1;
    return a.qlike - b.qlike;
  });
  return { nTrain: train.length, nTest: test.length, scores };
}
