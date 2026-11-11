// Distribution Engine: Fat-Tailed & Asymmetric Return Distribution Fitting
// Implements Maximum Likelihood Estimation (MLE) for:
// 1. Normal (Gaussian) baseline
// 2. Standardized Student-t distribution (tail parameter nu)
// 3. Hansen (1994) Skewed Student-t distribution (tail nu, skewness lambda)
// 4. Generalized Error Distribution (GED / Subbotin) (shape parameter kappa)
// Evaluates AIC, BIC, Delta-AIC, Log-Likelihood, and Kolmogorov-Smirnov (KS) statistic.

export interface FittedDistribution {
  name: string;
  key: 'normal' | 'student-t' | 'hansen-skew-t' | 'ged';
  numParams: number;
  logLikelihood: number;
  aic: number;
  bic: number;
  deltaAic: number;
  ksStatistic: number;
  rank: number;
  color: string;
  params: { [key: string]: number };
  paramDescriptions: string;
  pdf: (x: number) => number;
  cdf: (x: number) => number;
}

export interface DistributionFitResult {
  sampleSize: number;
  mean: number;
  stdDev: number;
  skewness: number;
  excessKurtosis: number;
  bestFitModel: string;
  models: FittedDistribution[];
  empiricalMin: number;
  empiricalMax: number;
}

/**
 * Lanczos approximation for log-gamma function ln(Gamma(z))
 */
export function logGamma(z: number): number {
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.138571095836524,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }

  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export function gamma(z: number): number {
  return Math.exp(logGamma(z));
}

/**
 * Standard Normal CDF approximation (Abramowitz & Stegun)
 */
export function standardNormalCdf(z: number): number {
  const b1 = 0.31938153;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;
  const p = 0.2316419;
  const c = 0.3989422804014337; // 1 / sqrt(2 * pi)

  if (z >= 0) {
    const t = 1.0 / (1.0 + p * z);
    return 1.0 - c * Math.exp(-0.5 * z * z) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
  } else {
    const t = 1.0 / (1.0 - p * z);
    return c * Math.exp(-0.5 * z * z) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
  }
}

/**
 * Normal Distribution PDF and CDF
 */
export function normalPdf(x: number, mean: number, std: number): number {
  if (std <= 1e-12) return 0;
  const z = (x - mean) / std;
  return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
}

export function normalCdf(x: number, mean: number, std: number): number {
  if (std <= 1e-12) return x >= mean ? 1 : 0;
  return standardNormalCdf((x - mean) / std);
}

/**
 * Standardized Student-t Distribution PDF and CDF (variance = std^2 for nu > 2)
 */
export function studentTPdf(x: number, mean: number, std: number, nu: number): number {
  if (nu <= 2 || std <= 1e-12) return normalPdf(x, mean, std);
  const z = (x - mean) / std;
  const logC = logGamma((nu + 1) / 2) - 0.5 * Math.log(Math.PI * (nu - 2)) - logGamma(nu / 2);
  const c = Math.exp(logC);
  const base = 1 + (z * z) / (nu - 2);
  return (c / std) * Math.pow(base, -(nu + 1) / 2);
}

/**
 * Hansen (1994) Skewed Student-t Distribution PDF
 * Reference: Hansen, B.E. (1994) "Autoregressive Conditional Heteroskedasticity with Asymmetric Student-t"
 */
export function hansenSkewTPdf(x: number, mean: number, std: number, nu: number, lambda: number): number {
  if (nu <= 2 || std <= 1e-12) return normalPdf(x, mean, std);
  if (Math.abs(lambda) < 1e-6) return studentTPdf(x, mean, std, nu);

  const logC = logGamma((nu + 1) / 2) - 0.5 * Math.log(Math.PI * (nu - 2)) - logGamma(nu / 2);
  const c = Math.exp(logC);
  const a = 4 * lambda * c * ((nu - 2) / (nu - 1));
  const bSq = 1 + 3 * lambda * lambda - a * a;
  const b = Math.sqrt(Math.max(1e-8, bSq));

  const z = (x - mean) / std;
  const threshold = -a / b;

  let term: number;
  if (z < threshold) {
    const q = (b * z + a) / (1 - lambda);
    term = Math.pow(1 + (q * q) / (nu - 2), -(nu + 1) / 2);
  } else {
    const q = (b * z + a) / (1 + lambda);
    term = Math.pow(1 + (q * q) / (nu - 2), -(nu + 1) / 2);
  }

  return (b * c / std) * term;
}

/**
 * Generalized Error Distribution (GED / Subbotin) PDF
 * kappa = 2: Normal, kappa = 1: Laplace (double exponential), kappa < 2: fat tails
 */
export function gedPdf(x: number, mean: number, std: number, kappa: number): number {
  if (kappa <= 0 || std <= 1e-12) return normalPdf(x, mean, std);

  const lg1 = logGamma(1 / kappa);
  const lg3 = logGamma(3 / kappa);
  // lambda = sqrt(2^(-2/kappa) * Gamma(1/kappa) / Gamma(3/kappa))
  const logLambda = 0.5 * (-(2 / kappa) * Math.LN2 + lg1 - lg3);
  const lam = Math.exp(logLambda);

  const z = Math.abs(x - mean) / (lam * std);
  const logNorm = Math.log(kappa) - Math.log(2 * lam * std) - lg1;

  return Math.exp(logNorm - Math.pow(z, kappa));
}

/**
 * Numerically integrate PDF to obtain CDF using trapezoidal rule
 */
function numericalCdf(
  x: number,
  pdf: (val: number) => number,
  minBound: number,
  steps: number = 80
): number {
  if (x <= minBound) return 0;
  const h = (x - minBound) / steps;
  let sum = 0.5 * (pdf(minBound) + pdf(x));
  for (let i = 1; i < steps; i++) {
    sum += pdf(minBound + i * h);
  }
  return Math.max(0, Math.min(1, sum * h));
}

/**
 * Compute Kolmogorov-Smirnov (KS) distance D between empirical distribution and fitted CDF
 */
function computeKsDistance(sortedReturns: number[], cdf: (val: number) => number): number {
  const n = sortedReturns.length;
  if (n === 0) return 0;

  let maxD = 0;
  for (let i = 0; i < n; i++) {
    const val = sortedReturns[i];
    const empiricalAbove = (i + 1) / n;
    const empiricalBelow = i / n;
    const theoretical = cdf(val);

    const d1 = Math.abs(empiricalAbove - theoretical);
    const d2 = Math.abs(empiricalBelow - theoretical);
    if (d1 > maxD) maxD = d1;
    if (d2 > maxD) maxD = d2;
  }

  return maxD;
}

/**
 * Master Distribution Fitting Engine
 * Fits Normal, Student-t, Hansen Skew-t, and GED distributions to returns series via MLE.
 */
export function fitParametricDistributions(returns: number[]): DistributionFitResult {
  const n = returns.length;
  if (n < 5) {
    const defaultModels: FittedDistribution[] = [
      {
        name: 'Normal (Gaussian)',
        key: 'normal',
        numParams: 2,
        logLikelihood: -100,
        aic: 204,
        bic: 211,
        deltaAic: 0,
        ksStatistic: 0.05,
        rank: 1,
        color: '#94a3b8',
        params: { mu: 0, sigma: 0.015 },
        paramDescriptions: 'μ = 0.00%, σ = 1.50%',
        pdf: (x) => normalPdf(x, 0, 0.015),
        cdf: (x) => normalCdf(x, 0, 0.015),
      },
    ];
    return {
      sampleSize: n,
      mean: 0,
      stdDev: 0.015,
      skewness: 0,
      excessKurtosis: 0,
      bestFitModel: 'Normal (Gaussian)',
      models: defaultModels,
      empiricalMin: -0.05,
      empiricalMax: 0.05,
    };
  }

  // Sample Moments
  let sum = 0;
  for (const r of returns) sum += r;
  const mean = sum / n;

  let sumSq = 0;
  let sumCube = 0;
  let sumQuad = 0;
  for (const r of returns) {
    const d = r - mean;
    sumSq += d * d;
    sumCube += d * d * d;
    sumQuad += d * d * d * d;
  }

  const variance = sumSq / (n - 1);
  const stdDev = Math.sqrt(Math.max(1e-12, variance));
  const skewness = variance > 0 ? (sumCube / n) / Math.pow(variance, 1.5) : 0;
  const excessKurtosis = variance > 0 ? (sumQuad / n) / Math.pow(variance, 2) - 3 : 0;

  const sortedReturns = [...returns].sort((a, b) => a - b);
  const minVal = sortedReturns[0];
  const maxVal = sortedReturns[n - 1];
  const boundLow = minVal - 3 * stdDev;

  // -------------------------------------------------------------
  // 1. Fit Normal Distribution (k = 2: mu, sigma)
  // -------------------------------------------------------------
  let normalLogL = 0;
  for (const r of returns) {
    const p = Math.max(1e-15, normalPdf(r, mean, stdDev));
    normalLogL += Math.log(p);
  }
  const normalAic = 2 * 2 - 2 * normalLogL;
  const normalBic = 2 * Math.log(n) - 2 * normalLogL;
  const normalKs = computeKsDistance(sortedReturns, (x) => normalCdf(x, mean, stdDev));

  // -------------------------------------------------------------
  // 2. Fit Student-t Distribution (k = 3: mu, sigma, nu)
  // -------------------------------------------------------------
  // Search nu in [2.1, 30] via 1D grid search (step 0.25)
  let bestNuT = 5.0;
  let bestLogLT = -Infinity;

  for (let nu = 2.1; nu <= 28; nu += 0.25) {
    let ll = 0;
    for (const r of returns) {
      const p = Math.max(1e-15, studentTPdf(r, mean, stdDev, nu));
      ll += Math.log(p);
    }
    if (ll > bestLogLT) {
      bestLogLT = ll;
      bestNuT = nu;
    }
  }

  const studentAic = 2 * 3 - 2 * bestLogLT;
  const studentBic = 3 * Math.log(n) - 2 * bestLogLT;
  const studentCdf = (x: number) => numericalCdf(x, (v) => studentTPdf(v, mean, stdDev, bestNuT), boundLow);
  const studentKs = computeKsDistance(sortedReturns, studentCdf);

  // -------------------------------------------------------------
  // 3. Fit Hansen (1994) Skewed Student-t (k = 4: mu, sigma, nu, lambda)
  // -------------------------------------------------------------
  // Coordinate search: nu in [2.2, 20] (step 0.4), lambda in [-0.6, 0.6] (step 0.05)
  let bestNuHansen = Math.min(15, Math.max(2.5, bestNuT));
  let bestLambdaHansen = Math.max(-0.5, Math.min(0.5, skewness * 0.25));
  let bestLogLHansen = -Infinity;

  // Grid search around initial estimate
  const nuCandidates = [2.2, 2.6, 3.0, 3.5, 4.0, 4.8, 5.8, 7.0, 9.0, 12.0, 18.0];
  const lambdaCandidates = [-0.45, -0.35, -0.25, -0.15, -0.08, 0.0, 0.08, 0.15, 0.25, 0.35, 0.45];

  for (const nu of nuCandidates) {
    for (const lam of lambdaCandidates) {
      let ll = 0;
      for (const r of returns) {
        const p = Math.max(1e-15, hansenSkewTPdf(r, mean, stdDev, nu, lam));
        ll += Math.log(p);
      }
      if (ll > bestLogLHansen) {
        bestLogLHansen = ll;
        bestNuHansen = nu;
        bestLambdaHansen = lam;
      }
    }
  }

  // Local refinement
  for (let dNu = -0.5; dNu <= 0.5; dNu += 0.25) {
    for (let dLam = -0.06; dLam <= 0.06; dLam += 0.03) {
      const nu = Math.max(2.15, bestNuHansen + dNu);
      const lam = Math.max(-0.6, Math.min(0.6, bestLambdaHansen + dLam));
      let ll = 0;
      for (const r of returns) {
        const p = Math.max(1e-15, hansenSkewTPdf(r, mean, stdDev, nu, lam));
        ll += Math.log(p);
      }
      if (ll > bestLogLHansen) {
        bestLogLHansen = ll;
        bestNuHansen = nu;
        bestLambdaHansen = lam;
      }
    }
  }

  const hansenAic = 2 * 4 - 2 * bestLogLHansen;
  const hansenBic = 4 * Math.log(n) - 2 * bestLogLHansen;
  const hansenCdf = (x: number) =>
    numericalCdf(x, (v) => hansenSkewTPdf(v, mean, stdDev, bestNuHansen, bestLambdaHansen), boundLow);
  const hansenKs = computeKsDistance(sortedReturns, hansenCdf);

  // -------------------------------------------------------------
  // 4. Fit Generalized Error Distribution (GED) (k = 3: mu, sigma, kappa)
  // -------------------------------------------------------------
  // Search kappa in [0.7, 2.8] via 1D grid search (step 0.05)
  let bestKappaGed = 1.4;
  let bestLogLGed = -Infinity;

  for (let kappa = 0.7; kappa <= 2.8; kappa += 0.05) {
    let ll = 0;
    for (const r of returns) {
      const p = Math.max(1e-15, gedPdf(r, mean, stdDev, kappa));
      ll += Math.log(p);
    }
    if (ll > bestLogLGed) {
      bestLogLGed = ll;
      bestKappaGed = kappa;
    }
  }

  const gedAic = 2 * 3 - 2 * bestLogLGed;
  const gedBic = 3 * Math.log(n) - 2 * bestLogLGed;
  const gedCdf = (x: number) => numericalCdf(x, (v) => gedPdf(v, mean, stdDev, bestKappaGed), boundLow);
  const gedKs = computeKsDistance(sortedReturns, gedCdf);

  // -------------------------------------------------------------
  // Construct Models Array & Rank by AIC
  // -------------------------------------------------------------
  const allAics = [studentAic, hansenAic, gedAic, normalAic];
  const minAic = Math.min(...allAics);

  const rawModels: Omit<FittedDistribution, 'rank' | 'deltaAic'>[] = [
    {
      name: 'Hansen Skewed Student-t',
      key: 'hansen-skew-t',
      numParams: 4,
      logLikelihood: bestLogLHansen,
      aic: hansenAic,
      bic: hansenBic,
      ksStatistic: hansenKs,
      color: '#a855f7', // Purple/Violet
      params: {
        mu: mean,
        sigma: stdDev,
        nu: bestNuHansen,
        lambda: bestLambdaHansen,
      },
      paramDescriptions: `ν = ${bestNuHansen.toFixed(1)}, λ = ${bestLambdaHansen > 0 ? '+' : ''}${bestLambdaHansen.toFixed(2)}`,
      pdf: (x) => hansenSkewTPdf(x, mean, stdDev, bestNuHansen, bestLambdaHansen),
      cdf: hansenCdf,
    },
    {
      name: 'Student-t (Standardized)',
      key: 'student-t',
      numParams: 3,
      logLikelihood: bestLogLT,
      aic: studentAic,
      bic: studentBic,
      ksStatistic: studentKs,
      color: '#10b981', // Emerald Green
      params: {
        mu: mean,
        sigma: stdDev,
        nu: bestNuT,
      },
      paramDescriptions: `ν = ${bestNuT.toFixed(1)} degrees of freedom`,
      pdf: (x) => studentTPdf(x, mean, stdDev, bestNuT),
      cdf: studentCdf,
    },
    {
      name: 'Generalized Error (GED)',
      key: 'ged',
      numParams: 3,
      logLikelihood: bestLogLGed,
      aic: gedAic,
      bic: gedBic,
      ksStatistic: gedKs,
      color: '#f59e0b', // Amber / Gold
      params: {
        mu: mean,
        sigma: stdDev,
        kappa: bestKappaGed,
      },
      paramDescriptions: `κ = ${bestKappaGed.toFixed(2)} (tail shape)`,
      pdf: (x) => gedPdf(x, mean, stdDev, bestKappaGed),
      cdf: gedCdf,
    },
    {
      name: 'Normal (Gaussian Baseline)',
      key: 'normal',
      numParams: 2,
      logLikelihood: normalLogL,
      aic: normalAic,
      bic: normalBic,
      ksStatistic: normalKs,
      color: '#94a3b8', // Slate dashed line
      params: {
        mu: mean,
        sigma: stdDev,
      },
      paramDescriptions: `μ = ${(mean * 100).toFixed(2)}%, σ = ${(stdDev * 100).toFixed(2)}%`,
      pdf: (x) => normalPdf(x, mean, stdDev),
      cdf: (x) => normalCdf(x, mean, stdDev),
    },
  ];

  // Sort by AIC ascending (lowest AIC is best)
  const sortedByAic = [...rawModels].sort((a, b) => a.aic - b.aic);

  const models: FittedDistribution[] = sortedByAic.map((m, index) => ({
    ...m,
    rank: index + 1,
    deltaAic: m.aic - minAic,
  }));

  return {
    sampleSize: n,
    mean,
    stdDev,
    skewness,
    excessKurtosis,
    bestFitModel: models[0]?.name || 'Student-t',
    models,
    empiricalMin: minVal,
    empiricalMax: maxVal,
  };
}
