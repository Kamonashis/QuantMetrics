// Pure TypeScript Dilated Causal Temporal Convolutional Network (TCN)
// with Causal Multi-Head Self-Attention for 4-Component Time-Series Forecasting.
// Zero React dependencies — runs deterministically via seeded PRNG, exact analytical
// backpropagation with preallocated scratch buffers, Adam optimizer, and
// chronological walk-forward holdout validation.

import type { DecompositionResult } from './quant-engine';

export type DecompositionComponentKey = 'trend' | 'cyclical' | 'seasonal' | 'residual';

export interface TcnAttentionConfig {
  /** Sliding lookback window length L (trading days) */
  lookback: number;
  /** Causal dilation schedule across residual TCN blocks, e.g. [1, 2, 4] */
  dilations: number[];
  /** 1D causal convolution kernel width k (2 or 3) */
  kernelSize: number;
  /** Number of self-attention heads (must divide hiddenChannels) */
  numHeads: number;
  /** Model hidden channels d_model */
  hiddenChannels: number;
  /** Number of training epochs per component */
  epochs: number;
  /** Initial Adam learning rate (cosine-annealed to 0.15 * lr) */
  learningRate: number;
  /** Chronological train/holdout split ratio (default 0.8) */
  trainSplit: number;
  /** Deterministic initialization seed */
  seed?: number;
}

export const DEFAULT_TCN_CONFIG: TcnAttentionConfig = {
  lookback: 24,
  dilations: [1, 2, 4],
  kernelSize: 3,
  numHeads: 2,
  hiddenChannels: 8,
  epochs: 40,
  learningRate: 0.01,
  trainSplit: 0.8,
  seed: 42,
};

export interface ComponentForecast {
  component: DecompositionComponentKey;
  label: string;
  currentValue: number;
  predictedValue: number;
  lower99: number;
  upper99: number;
  /** Exact signed dollar contribution to Ŷ_{t+1} - Y_t */
  contributionDollar: number;
  /** Exact signed % contribution to (Ŷ_{t+1} - Y_t) / Y_t * 100 */
  contributionPct: number;
  /** Per-epoch training Huber loss */
  trainLossCurve: number[];
  /** Per-epoch out-of-sample holdout MSE loss (in standardized space) */
  holdoutLossCurve: number[];
  /** Out-of-sample holdout RMSE in native component units */
  holdoutRmse: number;
  /** Out-of-sample holdout MAE in native component units */
  holdoutMae: number;
  /** Mean attention weights across heads for lag indices 0..L-1 (sums to 1) */
  attentionWeights: number[];
  /** Per-head attention weights [head][lag] */
  attentionByHead: number[][];
  /** Number of training + holdout windows used */
  trainWindows: number;
  holdoutWindows: number;
  /** Best epoch selected by holdout loss */
  bestEpoch: number;
}

export interface TcnDecompositionForecast {
  ticker: string;
  model: 'additive' | 'multiplicative';
  period: number;
  cycleMinPeriod: number;
  cycleMaxPeriod: number;
  config: TcnAttentionConfig;
  lastDate: string;
  nextDateLabel: string;
  lastPrice: number;
  predictedPrice: number;
  deltaDollar: number;
  deltaPct: number;
  lower99: number;
  upper99: number;
  /** Out-of-sample composite price holdout RMSE ($) */
  holdoutPriceRmse: number;
  /** Out-of-sample composite price holdout MAE ($) */
  holdoutPriceMae: number;
  /** Naive random-walk baseline RMSE ($) on identical holdout dates */
  naiveBaselineRmse: number;
  /** Percentage of holdout days with correct predicted move direction (0..100) */
  directionalAccuracy: number;
  /** Number of out-of-sample holdout trading sessions evaluated */
  holdoutCount: number;
  /** TCN causal receptive field in trading days */
  receptiveField: number;
  /** Trainable parameter count per component network */
  parameterCount: number;
  /** Wall-clock training time (ms) */
  trainingDurationMs: number;
  components: Record<DecompositionComponentKey, ComponentForecast>;
}

export interface TcnTrainingProgress {
  component: DecompositionComponentKey;
  componentIndex: number;
  totalComponents: number;
  epoch: number;
  totalEpochs: number;
  overallFraction: number;
  /** Optional sensitivity-sweep window context */
  windowIndex?: number;
  totalWindows?: number;
  windowLookback?: number;
}

export type SensitivityStep = 2 | 4 | 6;
export type SensitivityStabilityLabel = 'Stable' | 'Moderate' | 'Sensitive';

export interface SensitivityWindowRun {
  lookback: number;
  isBase: boolean;
  forecast: TcnDecompositionForecast;
}

export interface ComponentStabilityMetric {
  component: DecompositionComponentKey;
  label: string;
  baseValue: number;
  meanValue: number;
  stdDevValue: number;
  minValue: number;
  maxValue: number;
  minContribDollar: number;
  maxContribDollar: number;
  contribDollarSpread: number;
  contribDollarStdDev: number;
  /** Normalized share of cross-window dollar-attribution variance (0..100, sums to 100) */
  varianceSharePct: number;
}

export interface TcnSensitivitySummary {
  meanPrice: number;
  stdDevPrice: number;
  cvPct: number;
  minPrice: number;
  maxPrice: number;
  priceRangeDollar: number;
  priceRangePct: number;
  /** Cross-window price std dev divided by base 99% PI half-width */
  dispersionToPiRatio: number;
  upCount: number;
  downCount: number;
  dominantDirection: 'up' | 'down' | 'mixed';
  directionalAgreementPct: number;
  bestRmseLookback: number;
  bestRmseValue: number;
  stabilityLabel: SensitivityStabilityLabel;
  totalDurationMs: number;
}

export interface TcnSensitivityResult {
  ticker: string;
  model: 'additive' | 'multiplicative';
  baseLookback: number;
  step: SensitivityStep;
  runs: SensitivityWindowRun[];
  baseForecast: TcnDecompositionForecast;
  summary: TcnSensitivitySummary;
  componentStability: Record<DecompositionComponentKey, ComponentStabilityMetric>;
}

// -------------------------------------------------------------
// HELPERS: RECEPTIVE FIELD, PARAMETER COUNT, PRNG, T-QUANTILE
// -------------------------------------------------------------

export function computeReceptiveField(dilations: number[], kernelSize: number): number {
  const dilationSum = dilations.reduce((acc, d) => acc + Math.max(1, d), 0);
  return 1 + Math.max(0, kernelSize - 1) * dilationSum;
}

export function computeParameterCount(config: TcnAttentionConfig): number {
  const cIn = 3;
  const d = config.hiddenChannels;
  const k = config.kernelSize;
  const numBlocks = config.dilations.length;
  const inProj = cIn * d + d;
  const tcnBlocks = numBlocks * (k * d * d + d);
  const attn = 4 * d * d + d;
  const readout = d + 2;
  return inProj + tcnBlocks + attn + readout;
}

export function sanitizeTcnConfig(raw: TcnAttentionConfig, seriesLength: number): TcnAttentionConfig {
  const numHeads = Math.max(1, Math.min(4, Math.floor(raw.numHeads)));
  let hiddenChannels = Math.max(4, Math.min(24, Math.floor(raw.hiddenChannels)));
  if (hiddenChannels % numHeads !== 0) {
    hiddenChannels = Math.max(numHeads, Math.round(hiddenChannels / numHeads) * numHeads);
  }
  const maxSafeLookback = Math.max(8, Math.min(64, Math.floor(seriesLength * 0.45)));
  const lookback = Math.max(8, Math.min(maxSafeLookback, Math.floor(raw.lookback)));
  const kernelSize = raw.kernelSize === 2 ? 2 : 3;
  const dilations =
    raw.dilations.length > 0
      ? raw.dilations.map((d) => Math.max(1, Math.min(16, Math.floor(d))))
      : [1, 2, 4];
  const epochs = Math.max(15, Math.min(160, Math.floor(raw.epochs)));
  const learningRate = Math.max(0.0005, Math.min(0.05, raw.learningRate));
  const trainSplit = Math.max(0.65, Math.min(0.88, raw.trainSplit));
  return {
    lookback,
    dilations,
    kernelSize,
    numHeads,
    hiddenChannels,
    epochs,
    learningRate,
    trainSplit,
    seed: raw.seed ?? 42,
  };
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function studentT99(df: number): number {
  if (df <= 1) return 63.657;
  if (df <= 2) return 9.925;
  if (df <= 3) return 5.841;
  if (df <= 4) return 4.604;
  if (df <= 5) return 4.032;
  if (df <= 8) return 3.355;
  if (df <= 12) return 3.055;
  if (df <= 20) return 2.845;
  if (df <= 30) return 2.75;
  if (df <= 60) return 2.66;
  return 2.576;
}

function nextTradingDayLabel(isoDate: string): string {
  const parts = isoDate.split('-').map((p) => parseInt(p, 10));
  if (parts.length !== 3 || parts.some((p) => !Number.isFinite(p))) return 't+1';
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
  dt.setUTCDate(dt.getUTCDate() + 1);
  while (dt.getUTCDay() === 0 || dt.getUTCDay() === 6) {
    dt.setUTCDate(dt.getUTCDate() + 1);
  }
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const d = String(dt.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

// -------------------------------------------------------------
// TCN + CAUSAL MULTI-HEAD SELF-ATTENTION MODEL (ZERO-ALLOC LOOP)
// -------------------------------------------------------------

interface ParamBlock {
  w: Float64Array;
  g: Float64Array;
  m: Float64Array;
  v: Float64Array;
}

function makeParam(size: number, scale: number, rng: () => number): ParamBlock {
  const w = new Float64Array(size);
  if (scale > 0) {
    for (let i = 0; i < size; i++) {
      w[i] = (rng() * 2 - 1) * scale;
    }
  }
  return {
    w,
    g: new Float64Array(size),
    m: new Float64Array(size),
    v: new Float64Array(size),
  };
}

interface TcnLayerWeights {
  dilation: number;
  convW: ParamBlock;
  convB: ParamBlock;
}

interface TcnAttentionNet {
  cIn: number;
  d: number;
  k: number;
  numHeads: number;
  headDim: number;
  inW: ParamBlock;
  inB: ParamBlock;
  blocks: TcnLayerWeights[];
  wq: ParamBlock;
  wk: ParamBlock;
  wv: ParamBlock;
  wo: ParamBlock;
  bo: ParamBlock;
  outW: ParamBlock;
  outB: ParamBlock;
  skipW: ParamBlock;
  allParams: ParamBlock[];
}

function createNetwork(config: TcnAttentionConfig, seed: number): TcnAttentionNet {
  const rng = createRng(seed);
  const cIn = 3;
  const d = config.hiddenChannels;
  const k = config.kernelSize;
  const numHeads = config.numHeads;
  const headDim = d / numHeads;

  const inW = makeParam(cIn * d, Math.sqrt(2 / (cIn + d)), rng);
  const inB = makeParam(d, 0, rng);

  const blocks: TcnLayerWeights[] = config.dilations.map((dilation) => ({
    dilation,
    convW: makeParam(k * d * d, Math.sqrt(1.5 / (k * d + d)), rng),
    convB: makeParam(d, 0, rng),
  }));

  const attnScale = Math.sqrt(1.5 / (d + d));
  const wq = makeParam(d * d, attnScale, rng);
  const wk = makeParam(d * d, attnScale, rng);
  const wv = makeParam(d * d, attnScale, rng);
  const wo = makeParam(d * d, attnScale * 0.5, rng);
  const bo = makeParam(d, 0, rng);

  const outW = makeParam(d, Math.sqrt(1 / d) * 0.5, rng);
  const outB = makeParam(1, 0, rng);
  const skipW = makeParam(1, 0, rng);
  skipW.w[0] = 0.25;

  const allParams: ParamBlock[] = [inW, inB];
  for (const blk of blocks) {
    allParams.push(blk.convW, blk.convB);
  }
  allParams.push(wq, wk, wv, wo, bo, outW, outB, skipW);

  return {
    cIn,
    d,
    k,
    numHeads,
    headDim,
    inW,
    inB,
    blocks,
    wq,
    wk,
    wv,
    wo,
    bo,
    outW,
    outB,
    skipW,
    allParams,
  };
}

function cloneWeights(net: TcnAttentionNet): Float64Array[] {
  return net.allParams.map((p) => new Float64Array(p.w));
}

function restoreWeights(net: TcnAttentionNet, snapshot: Float64Array[]): void {
  for (let i = 0; i < net.allParams.length; i++) {
    net.allParams[i].w.set(snapshot[i]);
  }
}

interface ForwardCache {
  x: Float64Array;
  L: number;
  h0: Float64Array;
  blockInputs: Float64Array[];
  convPreAct: Float64Array[];
  convRes: Float64Array[];
  lnNorms: Float64Array[];
  lnInvStd: Float64Array[];
  hFinal: Float64Array;
  qLast: Float64Array;
  K: Float64Array;
  V: Float64Array;
  attnWeights: Float64Array;
  context: Float64Array;
  attnRes: Float64Array;
  finalLnNorm: Float64Array;
  finalLnInvStd: number;
  pred: number;
  gFinalLnOut: Float64Array;
  gAttnRes: Float64Array;
  gContext: Float64Array;
  gQLast: Float64Array;
  gH: Float64Array;
  gK: Float64Array;
  gV: Float64Array;
  gAlpha: Float64Array;
  gPrevA: Float64Array;
  gPrevB: Float64Array;
  gRes: Float64Array;
}

function createForwardWorkspace(net: TcnAttentionNet, L: number): ForwardCache {
  const { d, numHeads } = net;
  const numBlocks = net.blocks.length;
  const h0 = new Float64Array(L * d);
  const blockInputs: Float64Array[] = new Array(numBlocks);
  const convPreAct: Float64Array[] = new Array(numBlocks);
  const convRes: Float64Array[] = new Array(numBlocks);
  const lnNorms: Float64Array[] = new Array(numBlocks);
  const lnInvStd: Float64Array[] = new Array(numBlocks);

  for (let b = 0; b < numBlocks; b++) {
    blockInputs[b] = h0;
    convPreAct[b] = new Float64Array(L * d);
    convRes[b] = new Float64Array(L * d);
    lnNorms[b] = new Float64Array(L * d);
    lnInvStd[b] = new Float64Array(L);
  }

  return {
    x: new Float64Array(L * net.cIn),
    L,
    h0,
    blockInputs,
    convPreAct,
    convRes,
    lnNorms,
    lnInvStd,
    hFinal: numBlocks > 0 ? lnNorms[numBlocks - 1] : h0,
    qLast: new Float64Array(d),
    K: new Float64Array(L * d),
    V: new Float64Array(L * d),
    attnWeights: new Float64Array(numHeads * L),
    context: new Float64Array(d),
    attnRes: new Float64Array(d),
    finalLnNorm: new Float64Array(d),
    finalLnInvStd: 1,
    pred: 0,
    gFinalLnOut: new Float64Array(d),
    gAttnRes: new Float64Array(d),
    gContext: new Float64Array(d),
    gQLast: new Float64Array(d),
    gH: new Float64Array(L * d),
    gK: new Float64Array(L * d),
    gV: new Float64Array(L * d),
    gAlpha: new Float64Array(L),
    gPrevA: new Float64Array(L * d),
    gPrevB: new Float64Array(L * d),
    gRes: new Float64Array(L * d),
  };
}

const EPS_LN = 1e-5;

function silu(x: number): number {
  const xc = x < -20 ? -20 : x > 20 ? 20 : x;
  return xc / (1 + Math.exp(-xc));
}

function siluGrad(x: number): number {
  const xc = x < -20 ? -20 : x > 20 ? 20 : x;
  const sig = 1 / (1 + Math.exp(-xc));
  return sig * (1 + xc * (1 - sig));
}

function layerNormForwardVec(
  src: Float64Array,
  srcOffset: number,
  dstNorm: Float64Array,
  dstOffset: number,
  d: number
): number {
  let mean = 0;
  for (let c = 0; c < d; c++) mean += src[srcOffset + c];
  mean /= d;
  let varSum = 0;
  for (let c = 0; c < d; c++) {
    const diff = src[srcOffset + c] - mean;
    varSum += diff * diff;
  }
  const invStd = 1 / Math.sqrt(varSum / d + EPS_LN);
  for (let c = 0; c < d; c++) {
    dstNorm[dstOffset + c] = (src[srcOffset + c] - mean) * invStd;
  }
  return invStd;
}

function layerNormBackwardVec(
  gOut: Float64Array,
  gOutOffset: number,
  norm: Float64Array,
  normOffset: number,
  invStd: number,
  d: number,
  gIn: Float64Array,
  gInOffset: number
): void {
  let sumG = 0;
  let sumGN = 0;
  for (let c = 0; c < d; c++) {
    const g = gOut[gOutOffset + c];
    sumG += g;
    sumGN += g * norm[normOffset + c];
  }
  const invD = 1 / d;
  for (let c = 0; c < d; c++) {
    const g = gOut[gOutOffset + c];
    const nVal = norm[normOffset + c];
    gIn[gInOffset + c] += invStd * (g - sumG * invD - nVal * sumGN * invD);
  }
}

function forwardSample(
  net: TcnAttentionNet,
  x: Float64Array,
  L: number,
  ws: ForwardCache
): ForwardCache {
  const { cIn, d, k, numHeads, headDim } = net;
  ws.x = x;
  const h0 = ws.h0;
  const inW = net.inW.w;
  const inB = net.inB.w;

  // 1. Input linear projection
  for (let t = 0; t < L; t++) {
    const xBase = t * cIn;
    const hBase = t * d;
    const x0 = x[xBase];
    const x1 = x[xBase + 1];
    const x2 = x[xBase + 2];
    for (let c = 0; c < d; c++) {
      h0[hBase + c] = inB[c] + x0 * inW[c] + x1 * inW[d + c] + x2 * inW[2 * d + c];
    }
  }

  // 2. Dilated Causal TCN Residual Stack
  const numBlocks = net.blocks.length;
  let curr = h0;
  for (let b = 0; b < numBlocks; b++) {
    const blk = net.blocks[b];
    const dil = blk.dilation;
    const convW = blk.convW.w;
    const convB = blk.convB.w;
    ws.blockInputs[b] = curr;

    const pre = ws.convPreAct[b];
    const res = ws.convRes[b];
    const norm = ws.lnNorms[b];
    const invStd = ws.lnInvStd[b];

    for (let t = 0; t < L; t++) {
      const outBase = t * d;
      for (let co = 0; co < d; co++) {
        let sum = convB[co];
        for (let j = 0; j < k; j++) {
          const srcT = t - j * dil;
          if (srcT >= 0) {
            const srcBase = srcT * d;
            const wBase = j * d * d + co;
            for (let ci = 0; ci < d; ci++) {
              sum += curr[srcBase + ci] * convW[wBase + ci * d];
            }
          }
        }
        pre[outBase + co] = sum;
        res[outBase + co] = curr[outBase + co] + silu(sum);
      }
      invStd[t] = layerNormForwardVec(res, outBase, norm, outBase, d);
    }

    curr = norm;
  }

  ws.hFinal = curr;
  const lastBase = (L - 1) * d;

  // 3. Causal Multi-Head Self-Attention
  const qLast = ws.qLast;
  const wq = net.wq.w;
  const wk = net.wk.w;
  const wv = net.wv.w;
  for (let co = 0; co < d; co++) {
    let sum = 0;
    for (let ci = 0; ci < d; ci++) {
      sum += curr[lastBase + ci] * wq[ci * d + co];
    }
    qLast[co] = sum;
  }

  const K = ws.K;
  const V = ws.V;
  for (let t = 0; t < L; t++) {
    const hBase = t * d;
    for (let co = 0; co < d; co++) {
      let sumK = 0;
      let sumV = 0;
      for (let ci = 0; ci < d; ci++) {
        const hv = curr[hBase + ci];
        const wIdx = ci * d + co;
        sumK += hv * wk[wIdx];
        sumV += hv * wv[wIdx];
      }
      K[hBase + co] = sumK;
      V[hBase + co] = sumV;
    }
  }

  const attnWeights = ws.attnWeights;
  const context = ws.context;
  const invSqrtDk = 1 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    const hOff = h * headDim;
    const wOff = h * L;
    let maxScore = -Infinity;

    for (let t = 0; t < L; t++) {
      let dot = 0;
      const kBase = t * d + hOff;
      for (let m = 0; m < headDim; m++) {
        dot += qLast[hOff + m] * K[kBase + m];
      }
      const score = dot * invSqrtDk;
      attnWeights[wOff + t] = score;
      if (score > maxScore) maxScore = score;
    }

    let expSum = 0;
    for (let t = 0; t < L; t++) {
      const e = Math.exp(attnWeights[wOff + t] - maxScore);
      attnWeights[wOff + t] = e;
      expSum += e;
    }
    const invSum = expSum > 0 ? 1 / expSum : 1 / L;
    for (let t = 0; t < L; t++) {
      attnWeights[wOff + t] *= invSum;
    }

    for (let m = 0; m < headDim; m++) {
      let ctxVal = 0;
      for (let t = 0; t < L; t++) {
        ctxVal += attnWeights[wOff + t] * V[t * d + hOff + m];
      }
      context[hOff + m] = ctxVal;
    }
  }

  // Output projection + residual + final LayerNorm
  const attnRes = ws.attnRes;
  const wo = net.wo.w;
  const bo = net.bo.w;
  for (let co = 0; co < d; co++) {
    let sum = bo[co];
    for (let ci = 0; ci < d; ci++) {
      sum += context[ci] * wo[ci * d + co];
    }
    attnRes[co] = curr[lastBase + co] + sum;
  }

  ws.finalLnInvStd = layerNormForwardVec(attnRes, 0, ws.finalLnNorm, 0, d);

  // 4. Scalar Readout + Persistence Skip
  let pred = net.outB.w[0] + net.skipW.w[0] * x[(L - 1) * cIn];
  for (let c = 0; c < d; c++) {
    pred += ws.finalLnNorm[c] * net.outW.w[c];
  }
  ws.pred = pred;
  return ws;
}

function backwardSample(net: TcnAttentionNet, cache: ForwardCache, dPred: number): void {
  const { cIn, d, k, numHeads, headDim } = net;
  const { L, x, hFinal, qLast, K, V, attnWeights, context, finalLnNorm, finalLnInvStd } = cache;
  const lastBase = (L - 1) * d;

  // 1. Readout gradients
  net.outB.g[0] += dPred;
  net.skipW.g[0] += dPred * x[(L - 1) * cIn];

  const gFinalLnOut = cache.gFinalLnOut;
  for (let c = 0; c < d; c++) {
    net.outW.g[c] += dPred * finalLnNorm[c];
    gFinalLnOut[c] = dPred * net.outW.w[c];
  }

  // 2. Final LayerNorm backward
  const gAttnRes = cache.gAttnRes;
  gAttnRes.fill(0);
  layerNormBackwardVec(gFinalLnOut, 0, finalLnNorm, 0, finalLnInvStd, d, gAttnRes, 0);

  const gH = cache.gH;
  gH.fill(0);
  for (let c = 0; c < d; c++) {
    gH[lastBase + c] += gAttnRes[c];
    net.bo.g[c] += gAttnRes[c];
  }

  const gContext = cache.gContext;
  const woW = net.wo.w;
  const woG = net.wo.g;
  for (let ci = 0; ci < d; ci++) {
    let sum = 0;
    const cVal = context[ci];
    const rowOff = ci * d;
    for (let co = 0; co < d; co++) {
      const go = gAttnRes[co];
      woG[rowOff + co] += cVal * go;
      sum += go * woW[rowOff + co];
    }
    gContext[ci] = sum;
  }

  // 3. Multi-Head Self-Attention backward
  const gQLast = cache.gQLast;
  const gK = cache.gK;
  const gV = cache.gV;
  const gAlpha = cache.gAlpha;
  gQLast.fill(0);
  gK.fill(0);
  gV.fill(0);
  const invSqrtDk = 1 / Math.sqrt(headDim);

  for (let h = 0; h < numHeads; h++) {
    const hOff = h * headDim;
    const wOff = h * L;

    for (let t = 0; t < L; t++) {
      const a = attnWeights[wOff + t];
      const vBase = t * d + hOff;
      let dot = 0;
      for (let m = 0; m < headDim; m++) {
        const gc = gContext[hOff + m];
        dot += gc * V[vBase + m];
        gV[vBase + m] += a * gc;
      }
      gAlpha[t] = dot;
    }

    let sumAG = 0;
    for (let t = 0; t < L; t++) {
      sumAG += attnWeights[wOff + t] * gAlpha[t];
    }

    for (let t = 0; t < L; t++) {
      const gScore = attnWeights[wOff + t] * (gAlpha[t] - sumAG) * invSqrtDk;
      const kBase = t * d + hOff;
      for (let m = 0; m < headDim; m++) {
        gQLast[hOff + m] += gScore * K[kBase + m];
        gK[kBase + m] += gScore * qLast[hOff + m];
      }
    }
  }

  const wqW = net.wq.w;
  const wqG = net.wq.g;
  for (let ci = 0; ci < d; ci++) {
    const hLastVal = hFinal[lastBase + ci];
    const rowOff = ci * d;
    let sumQ = 0;
    for (let co = 0; co < d; co++) {
      const gq = gQLast[co];
      wqG[rowOff + co] += hLastVal * gq;
      sumQ += gq * wqW[rowOff + co];
    }
    gH[lastBase + ci] += sumQ;
  }

  const wkW = net.wk.w;
  const wkG = net.wk.g;
  const wvW = net.wv.w;
  const wvG = net.wv.g;
  for (let t = 0; t < L; t++) {
    const hBase = t * d;
    for (let ci = 0; ci < d; ci++) {
      const hv = hFinal[hBase + ci];
      const rowOff = ci * d;
      let sumKV = 0;
      for (let co = 0; co < d; co++) {
        const gk = gK[hBase + co];
        const gv = gV[hBase + co];
        wkG[rowOff + co] += hv * gk;
        wvG[rowOff + co] += hv * gv;
        sumKV += gk * wkW[rowOff + co] + gv * wvW[rowOff + co];
      }
      gH[hBase + ci] += sumKV;
    }
  }

  // 4. Dilated Causal TCN Residual Stack backward (ping-pong buffers)
  let gCurr = gH;
  const gRes = cache.gRes;
  let useA = true;
  for (let b = net.blocks.length - 1; b >= 0; b--) {
    const blk = net.blocks[b];
    const dil = blk.dilation;
    const inp = cache.blockInputs[b];
    const pre = cache.convPreAct[b];
    const norm = cache.lnNorms[b];
    const invStd = cache.lnInvStd[b];
    const convW = blk.convW.w;
    const convG = blk.convW.g;
    const convBG = blk.convB.g;

    const gPrev = useA ? cache.gPrevA : cache.gPrevB;
    useA = !useA;
    gPrev.fill(0);
    gRes.fill(0);

    for (let t = 0; t < L; t++) {
      const base = t * d;
      layerNormBackwardVec(gCurr, base, norm, base, invStd[t], d, gRes, base);
    }

    for (let t = 0; t < L; t++) {
      const outBase = t * d;
      for (let co = 0; co < d; co++) {
        const gr = gRes[outBase + co];
        gPrev[outBase + co] += gr;
        const gPre = gr * siluGrad(pre[outBase + co]);
        convBG[co] += gPre;

        for (let j = 0; j < k; j++) {
          const srcT = t - j * dil;
          if (srcT >= 0) {
            const srcBase = srcT * d;
            const wBase = j * d * d + co;
            for (let ci = 0; ci < d; ci++) {
              const wIdx = wBase + ci * d;
              convG[wIdx] += inp[srcBase + ci] * gPre;
              gPrev[srcBase + ci] += convW[wIdx] * gPre;
            }
          }
        }
      }
    }

    gCurr = gPrev;
  }

  // 5. Input Projection backward
  const inWG = net.inW.g;
  const inBG = net.inB.g;
  for (let t = 0; t < L; t++) {
    const xBase = t * cIn;
    const hBase = t * d;
    const x0 = x[xBase];
    const x1 = x[xBase + 1];
    const x2 = x[xBase + 2];
    for (let c = 0; c < d; c++) {
      const gh = gCurr[hBase + c];
      inBG[c] += gh;
      inWG[c] += x0 * gh;
      inWG[d + c] += x1 * gh;
      inWG[2 * d + c] += x2 * gh;
    }
  }
}

function zeroGrads(net: TcnAttentionNet): void {
  for (const p of net.allParams) {
    p.g.fill(0);
  }
}

function adamStep(
  net: TcnAttentionNet,
  batchSize: number,
  lr: number,
  step: number,
  weightDecay: number = 1e-4
): void {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const invB = 1 / Math.max(1, batchSize);

  let gradSq = 0;
  for (const p of net.allParams) {
    for (let i = 0; i < p.g.length; i++) {
      const g = p.g[i] * invB;
      gradSq += g * g;
    }
  }
  const gradNorm = Math.sqrt(gradSq);
  const clipScale = gradNorm > 2.0 ? 2.0 / gradNorm : 1.0;

  const bc1 = 1 - Math.pow(beta1, step);
  const bc2 = 1 - Math.pow(beta2, step);

  for (const p of net.allParams) {
    const w = p.w;
    const gArr = p.g;
    const mArr = p.m;
    const vArr = p.v;
    const isBias = w.length === 1 || p === net.inB || p === net.bo || net.blocks.some((b) => b.convB === p);

    for (let i = 0; i < w.length; i++) {
      const g = gArr[i] * invB * clipScale + (isBias ? 0 : weightDecay * w[i]);
      const m = beta1 * mArr[i] + (1 - beta1) * g;
      const v = beta2 * vArr[i] + (1 - beta2) * g * g;
      mArr[i] = m;
      vArr[i] = v;
      const mHat = m / bc1;
      const vHat = v / bc2;
      w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }
}

// -------------------------------------------------------------
// WINDOW CONSTRUCTION & COMPONENT TRAINING
// -------------------------------------------------------------

interface BuiltDataset {
  inputs: Float64Array[];
  targets: number[];
  targetIndices: number[];
  latestInput: Float64Array;
  mean: number;
  std: number;
  isDiffMode: boolean;
  anchors: number[];
  latestAnchor: number;
}

function buildComponentDataset(
  values: number[],
  lookback: number,
  isDiffMode: boolean,
  isLogMode: boolean,
  trainSplit: number
): BuiltDataset {
  const work: number[] = [];
  const workToPriceIdx: number[] = [];
  const anchors: number[] = [];

  if (isDiffMode) {
    for (let i = 1; i < values.length; i++) {
      const prev = values[i - 1];
      const curr = values[i];
      const v = isLogMode
        ? Math.log(Math.max(curr, 1e-6) / Math.max(prev, 1e-6))
        : curr - prev;
      work.push(v);
      workToPriceIdx.push(i);
      anchors.push(prev);
    }
  } else {
    for (let i = 0; i < values.length; i++) {
      const v = isLogMode ? Math.log(Math.max(values[i], 1e-6)) : values[i];
      work.push(v);
      workToPriceIdx.push(i);
      anchors.push(values[i]);
    }
  }

  const numSamples = work.length - lookback;
  const splitIdx = Math.max(4, Math.min(numSamples - 2, Math.floor(numSamples * trainSplit)));
  const statEnd = Math.min(work.length, splitIdx + lookback);
  let mean = 0;
  for (let i = 0; i < statEnd; i++) mean += work[i];
  mean /= Math.max(1, statEnd);

  let varSum = 0;
  for (let i = 0; i < statEnd; i++) {
    const diff = work[i] - mean;
    varSum += diff * diff;
  }
  const std = Math.sqrt(varSum / Math.max(1, statEnd - 1)) || 1e-4;

  const z = work.map((v) => (v - mean) / std);

  const makeWindow = (startIdx: number): Float64Array => {
    const buf = new Float64Array(lookback * 3);
    const denom = Math.max(1, lookback - 1);
    for (let t = 0; t < lookback; t++) {
      const zi = z[startIdx + t];
      const prevZ = t > 0 ? z[startIdx + t - 1] : zi;
      buf[t * 3] = zi;
      buf[t * 3 + 1] = zi - prevZ;
      buf[t * 3 + 2] = t / denom - 0.5;
    }
    return buf;
  };

  const inputs: Float64Array[] = [];
  const targets: number[] = [];
  const targetIndices: number[] = [];
  const targetAnchors: number[] = [];

  for (let s = 0; s < numSamples; s++) {
    inputs.push(makeWindow(s));
    targets.push(z[s + lookback]);
    targetIndices.push(workToPriceIdx[s + lookback]);
    targetAnchors.push(anchors[s + lookback]);
  }

  const latestInput = makeWindow(work.length - lookback);
  const latestAnchor = values[values.length - 1];

  return {
    inputs,
    targets,
    targetIndices,
    latestInput,
    mean,
    std,
    isDiffMode,
    anchors: targetAnchors,
    latestAnchor,
  };
}

function unscaleToNative(
  zPred: number,
  mean: number,
  std: number,
  isDiffMode: boolean,
  isLogMode: boolean,
  anchor: number
): number {
  const workVal = zPred * std + mean;
  if (isDiffMode) {
    return isLogMode
      ? anchor * Math.exp(Math.max(-0.5, Math.min(0.5, workVal)))
      : anchor + workVal;
  }
  return isLogMode ? Math.exp(Math.max(-2, Math.min(2, workVal))) : workVal;
}

interface TrainedComponentResult {
  forecast: Omit<ComponentForecast, 'contributionDollar' | 'contributionPct'>;
  holdoutPredByIndex: Map<number, number>;
}

async function trainSingleComponent(
  component: DecompositionComponentKey,
  label: string,
  values: number[],
  model: 'additive' | 'multiplicative',
  config: TcnAttentionConfig,
  seedOffset: number,
  onEpoch?: (epoch: number) => void,
  yieldControl?: () => Promise<void>,
  isCancelled?: () => boolean
): Promise<TrainedComponentResult> {
  const isLogMode = model === 'multiplicative';
  const isDiffMode = component === 'trend';
  const dataset = buildComponentDataset(values, config.lookback, isDiffMode, isLogMode, config.trainSplit);

  const totalSamples = dataset.inputs.length;
  const splitIdx = Math.max(4, Math.min(totalSamples - 2, Math.floor(totalSamples * config.trainSplit)));
  const trainCount = splitIdx;
  const holdoutCount = totalSamples - splitIdx;

  const net = createNetwork(config, (config.seed ?? 42) + seedOffset * 97);
  const ws = createForwardWorkspace(net, config.lookback);
  const shuffleRng = createRng((config.seed ?? 42) + seedOffset * 313 + 7);

  const trainOrder = new Int32Array(trainCount);
  for (let i = 0; i < trainCount; i++) trainOrder[i] = i;

  const batchSize = Math.min(16, Math.max(4, Math.floor(trainCount / 8)));
  const huberDelta = 1.5;

  const trainLossCurve: number[] = [];
  const holdoutLossCurve: number[] = [];

  let bestHoldoutLoss = Infinity;
  let bestWeights = cloneWeights(net);
  let bestEpoch = 1;
  let adamStepNum = 0;
  let lastHoldoutLoss = 1;

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    if (isCancelled?.()) break;

    for (let i = trainCount - 1; i > 0; i--) {
      const j = Math.floor(shuffleRng() * (i + 1));
      const tmp = trainOrder[i];
      trainOrder[i] = trainOrder[j];
      trainOrder[j] = tmp;
    }

    const progress = (epoch - 1) / Math.max(1, config.epochs - 1);
    const lr = config.learningRate * (0.15 + 0.85 * 0.5 * (1 + Math.cos(Math.PI * progress)));

    let epochTrainLoss = 0;
    for (let start = 0; start < trainCount; start += batchSize) {
      const end = Math.min(trainCount, start + batchSize);
      zeroGrads(net);

      for (let idx = start; idx < end; idx++) {
        const sIdx = trainOrder[idx];
        const cache = forwardSample(net, dataset.inputs[sIdx], config.lookback, ws);
        const err = cache.pred - dataset.targets[sIdx];
        const absErr = Math.abs(err);
        let loss: number;
        let dPred: number;
        if (absErr <= huberDelta) {
          loss = 0.5 * err * err;
          dPred = err;
        } else {
          loss = huberDelta * (absErr - 0.5 * huberDelta);
          dPred = huberDelta * (err > 0 ? 1 : -1);
        }
        epochTrainLoss += loss;
        backwardSample(net, cache, dPred);
      }

      adamStepNum++;
      adamStep(net, end - start, lr, adamStepNum);
    }

    const avgTrainLoss = epochTrainLoss / Math.max(1, trainCount);
    trainLossCurve.push(avgTrainLoss);

    if (epoch === 1 || epoch % 2 === 0 || epoch === config.epochs) {
      let holdoutLossSum = 0;
      for (let sIdx = splitIdx; sIdx < totalSamples; sIdx++) {
        const cache = forwardSample(net, dataset.inputs[sIdx], config.lookback, ws);
        const err = cache.pred - dataset.targets[sIdx];
        holdoutLossSum += err * err;
      }
      lastHoldoutLoss = holdoutLossSum / Math.max(1, holdoutCount);

      if (epoch >= Math.min(8, config.epochs) && lastHoldoutLoss < bestHoldoutLoss) {
        bestHoldoutLoss = lastHoldoutLoss;
        bestWeights = cloneWeights(net);
        bestEpoch = epoch;
      }
    }
    holdoutLossCurve.push(lastHoldoutLoss);

    if (onEpoch && (epoch === 1 || epoch % 5 === 0 || epoch === config.epochs)) {
      onEpoch(epoch);
    }
    if (yieldControl && epoch % 8 === 0) {
      await yieldControl();
    }
  }

  restoreWeights(net, bestWeights);

  const holdoutPredByIndex = new Map<number, number>();
  const holdoutNativeErrors: number[] = [];
  const holdoutWorkErrors: number[] = [];
  let sqErrSum = 0;
  let absErrSum = 0;

  for (let sIdx = splitIdx; sIdx < totalSamples; sIdx++) {
    const cache = forwardSample(net, dataset.inputs[sIdx], config.lookback, ws);
    const nativePred = unscaleToNative(
      cache.pred,
      dataset.mean,
      dataset.std,
      isDiffMode,
      isLogMode,
      dataset.anchors[sIdx]
    );
    const priceIdx = dataset.targetIndices[sIdx];
    const nativeActual = values[priceIdx];
    holdoutPredByIndex.set(priceIdx, nativePred);

    const diff = nativeActual - nativePred;
    holdoutNativeErrors.push(diff);
    sqErrSum += diff * diff;
    absErrSum += Math.abs(diff);

    if (isLogMode) {
      holdoutWorkErrors.push(
        Math.log(Math.max(nativeActual, 1e-6) / Math.max(nativePred, 1e-6))
      );
    } else {
      holdoutWorkErrors.push(diff);
    }
  }

  const nHold = Math.max(1, holdoutNativeErrors.length);
  const holdoutRmse = Math.sqrt(sqErrSum / nHold);
  const holdoutMae = absErrSum / nHold;

  const latestCache = forwardSample(net, dataset.latestInput, config.lookback, ws);
  const currentValue = values[values.length - 1];
  const predictedValue = unscaleToNative(
    latestCache.pred,
    dataset.mean,
    dataset.std,
    isDiffMode,
    isLogMode,
    dataset.latestAnchor
  );

  let workSqSum = 0;
  for (const e of holdoutWorkErrors) workSqSum += e * e;
  const workStd = Math.sqrt(workSqSum / Math.max(1, holdoutWorkErrors.length - 1));
  const tCrit = studentT99(Math.max(1, holdoutWorkErrors.length - 1));

  let lower99: number;
  let upper99: number;
  if (isLogMode) {
    lower99 = predictedValue * Math.exp(-tCrit * workStd);
    upper99 = predictedValue * Math.exp(tCrit * workStd);
  } else {
    lower99 = predictedValue - tCrit * workStd;
    upper99 = predictedValue + tCrit * workStd;
  }

  const attentionByHead: number[][] = [];
  const attentionWeights = new Array<number>(config.lookback).fill(0);
  for (let h = 0; h < config.numHeads; h++) {
    const headArr = new Array<number>(config.lookback);
    for (let t = 0; t < config.lookback; t++) {
      const wVal = latestCache.attnWeights[h * config.lookback + t];
      headArr[t] = wVal;
      attentionWeights[t] += wVal / config.numHeads;
    }
    attentionByHead.push(headArr);
  }

  return {
    forecast: {
      component,
      label,
      currentValue,
      predictedValue,
      lower99,
      upper99,
      trainLossCurve,
      holdoutLossCurve,
      holdoutRmse,
      holdoutMae,
      attentionWeights,
      attentionByHead,
      trainWindows: trainCount,
      holdoutWindows: holdoutCount,
      bestEpoch,
    },
    holdoutPredByIndex,
  };
}

// -------------------------------------------------------------
// PUBLIC ORCHESTRATOR: TRAIN ALL 4 COMPONENTS & RECONSTRUCT Ŷ_{t+1}
// -------------------------------------------------------------

const COMPONENT_ORDER: { key: DecompositionComponentKey; label: string }[] = [
  { key: 'trend', label: 'Trend (T)' },
  { key: 'cyclical', label: 'Cyclical (C)' },
  { key: 'seasonal', label: 'Seasonal (S)' },
  { key: 'residual', label: 'Residual (I)' },
];

export const MIN_BARS_FOR_TCN = 45;

export async function runTcnDecompositionForecast(
  ticker: string,
  decomposition: DecompositionResult,
  rawConfig: TcnAttentionConfig = DEFAULT_TCN_CONFIG,
  options?: {
    onProgress?: (progress: TcnTrainingProgress) => void;
    yieldControl?: () => Promise<void>;
    isCancelled?: () => boolean;
  }
): Promise<TcnDecompositionForecast | null> {
  const n = decomposition.observed.length;
  if (n < MIN_BARS_FOR_TCN) return null;

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const config = sanitizeTcnConfig(rawConfig, n);
  const { model, dates, observed } = decomposition;
  const neutral = model === 'multiplicative' ? 1 : 0;

  const seriesMap: Record<DecompositionComponentKey, number[]> = {
    trend: decomposition.trend.map((v, i) => (v !== null && Number.isFinite(v) ? v : observed[i])),
    cyclical: decomposition.cyclical.map((v) => (Number.isFinite(v) ? v : neutral)),
    seasonal: decomposition.seasonal.map((v) => (Number.isFinite(v) ? v : neutral)),
    residual: decomposition.residual.map((v) => (v !== null && Number.isFinite(v) ? v : neutral)),
  };

  const results = {} as Record<DecompositionComponentKey, TrainedComponentResult>;

  for (let idx = 0; idx < COMPONENT_ORDER.length; idx++) {
    if (options?.isCancelled?.()) return null;
    const { key, label } = COMPONENT_ORDER[idx];

    results[key] = await trainSingleComponent(
      key,
      label,
      seriesMap[key],
      model,
      config,
      idx + 1,
      (epoch) => {
        options?.onProgress?.({
          component: key,
          componentIndex: idx,
          totalComponents: COMPONENT_ORDER.length,
          epoch,
          totalEpochs: config.epochs,
          overallFraction: (idx + epoch / config.epochs) / COMPONENT_ORDER.length,
        });
      },
      options?.yieldControl,
      options?.isCancelled
    );

    if (options?.yieldControl) {
      await options.yieldControl();
    }
  }

  if (options?.isCancelled?.()) return null;

  const tPred = results.trend.forecast.predictedValue;
  const cPred = results.cyclical.forecast.predictedValue;
  const sPred = results.seasonal.forecast.predictedValue;
  const iPred = results.residual.forecast.predictedValue;

  const lastPrice = observed[n - 1];
  const predictedPrice =
    model === 'multiplicative'
      ? tPred * cPred * sPred * iPred
      : tPred + cPred + sPred + iPred;

  const deltaDollar = predictedPrice - lastPrice;
  const deltaPct = lastPrice !== 0 ? (deltaDollar / lastPrice) * 100 : 0;

  const sharedHoldoutIndices: number[] = [];
  for (const idx of results.trend.holdoutPredByIndex.keys()) {
    if (
      results.cyclical.holdoutPredByIndex.has(idx) &&
      results.seasonal.holdoutPredByIndex.has(idx) &&
      results.residual.holdoutPredByIndex.has(idx) &&
      idx > 0 &&
      idx < n
    ) {
      sharedHoldoutIndices.push(idx);
    }
  }
  sharedHoldoutIndices.sort((a, b) => a - b);

  let priceSqErr = 0;
  let priceAbsErr = 0;
  let naiveSqErr = 0;
  let dirHits = 0;
  const compositeWorkErrors: number[] = [];

  for (const idx of sharedHoldoutIndices) {
    const ht = results.trend.holdoutPredByIndex.get(idx)!;
    const hc = results.cyclical.holdoutPredByIndex.get(idx)!;
    const hs = results.seasonal.holdoutPredByIndex.get(idx)!;
    const hi = results.residual.holdoutPredByIndex.get(idx)!;
    const yHat = model === 'multiplicative' ? ht * hc * hs * hi : ht + hc + hs + hi;
    const yActual = observed[idx];
    const yPrev = observed[idx - 1];

    const err = yActual - yHat;
    priceSqErr += err * err;
    priceAbsErr += Math.abs(err);

    const naiveErr = yActual - yPrev;
    naiveSqErr += naiveErr * naiveErr;

    if ((yHat - yPrev) * (yActual - yPrev) >= 0) {
      dirHits++;
    }

    if (model === 'multiplicative') {
      compositeWorkErrors.push(Math.log(Math.max(yActual, 1e-6) / Math.max(yHat, 1e-6)));
    } else {
      compositeWorkErrors.push(err);
    }
  }

  const holdoutCount = Math.max(1, sharedHoldoutIndices.length);
  const holdoutPriceRmse = Math.sqrt(priceSqErr / holdoutCount);
  const holdoutPriceMae = priceAbsErr / holdoutCount;
  const naiveBaselineRmse = Math.sqrt(naiveSqErr / holdoutCount);
  const directionalAccuracy = (dirHits / holdoutCount) * 100;

  let compErrSq = 0;
  for (const e of compositeWorkErrors) compErrSq += e * e;
  const compWorkStd = Math.sqrt(compErrSq / Math.max(1, compositeWorkErrors.length - 1));
  const tCrit = studentT99(Math.max(1, compositeWorkErrors.length - 1));

  const lower99 =
    model === 'multiplicative'
      ? predictedPrice * Math.exp(-tCrit * compWorkStd)
      : predictedPrice - tCrit * compWorkStd;
  const upper99 =
    model === 'multiplicative'
      ? predictedPrice * Math.exp(tCrit * compWorkStd)
      : predictedPrice + tCrit * compWorkStd;

  const components = {} as Record<DecompositionComponentKey, ComponentForecast>;

  if (model === 'multiplicative') {
    const logChanges: Record<DecompositionComponentKey, number> = {
      trend: Math.log(Math.max(tPred, 1e-6) / Math.max(results.trend.forecast.currentValue, 1e-6)),
      cyclical: Math.log(Math.max(cPred, 1e-6) / Math.max(results.cyclical.forecast.currentValue, 1e-6)),
      seasonal: Math.log(Math.max(sPred, 1e-6) / Math.max(results.seasonal.forecast.currentValue, 1e-6)),
      residual: Math.log(Math.max(iPred, 1e-6) / Math.max(results.residual.forecast.currentValue, 1e-6)),
    };
    const totalLogChange =
      logChanges.trend + logChanges.cyclical + logChanges.seasonal + logChanges.residual;

    for (const { key } of COMPONENT_ORDER) {
      const base = results[key].forecast;
      const contribDollar =
        Math.abs(totalLogChange) > 1e-9
          ? deltaDollar * (logChanges[key] / totalLogChange)
          : lastPrice * logChanges[key];
      const contribPct = lastPrice !== 0 ? (contribDollar / lastPrice) * 100 : 0;
      components[key] = {
        ...base,
        contributionDollar: contribDollar,
        contributionPct: contribPct,
      };
    }
  } else {
    for (const { key } of COMPONENT_ORDER) {
      const base = results[key].forecast;
      const contribDollar = base.predictedValue - base.currentValue;
      const contribPct = lastPrice !== 0 ? (contribDollar / lastPrice) * 100 : 0;
      components[key] = {
        ...base,
        contributionDollar: contribDollar,
        contributionPct: contribPct,
      };
    }
  }

  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  return {
    ticker,
    model,
    period: decomposition.period,
    cycleMinPeriod: decomposition.cycleMinPeriod,
    cycleMaxPeriod: decomposition.cycleMaxPeriod,
    config,
    lastDate: dates[n - 1] || '',
    nextDateLabel: nextTradingDayLabel(dates[n - 1] || ''),
    lastPrice,
    predictedPrice,
    deltaDollar,
    deltaPct,
    lower99,
    upper99,
    holdoutPriceRmse,
    holdoutPriceMae,
    naiveBaselineRmse,
    directionalAccuracy,
    holdoutCount,
    receptiveField: computeReceptiveField(config.dilations, config.kernelSize),
    parameterCount: computeParameterCount(config),
    trainingDurationMs: Math.max(1, Math.round(t1 - t0)),
    components,
  };
}

/**
 * Generates an ordered, deduplicated grid of up to 5 valid lookback windows
 * centered around `baseLookback` with spacing `step`, safely clamped to
 * `[8, min(64, floor(0.45 * seriesLength))]`. If one side hits a bound, the
 * grid shifts toward the interior so distinct windows are preserved whenever
 * available history permits.
 */
export function generateLookbackSweepGrid(
  baseLookback: number,
  step: SensitivityStep,
  seriesLength: number
): number[] {
  const minL = 8;
  const maxL = Math.max(minL, Math.min(64, Math.floor(seriesLength * 0.45)));
  const center = Math.max(minL, Math.min(maxL, Math.floor(baseLookback)));
  const s = Math.max(1, Math.floor(step));

  const set = new Set<number>([center]);
  // Expand symmetrically outward: ±1s, ±2s, then shift inward if an edge hit a bound
  const offsets = [-s, s, -2 * s, 2 * s, -3 * s, 3 * s, -4 * s, 4 * s];
  for (const off of offsets) {
    if (set.size >= 5) break;
    const candidate = center + off;
    if (candidate >= minL && candidate <= maxL) {
      set.add(candidate);
    }
  }

  // Fallback fill if the feasible range is very narrow
  if (set.size < 5) {
    for (let delta = 1; delta <= maxL - minL && set.size < 5; delta++) {
      if (center - delta >= minL) set.add(center - delta);
      if (set.size < 5 && center + delta <= maxL) set.add(center + delta);
    }
  }

  return Array.from(set).sort((a, b) => a - b);
}

function sampleStdDev(values: number[], mean: number): number {
  if (values.length <= 1) return 0;
  let sq = 0;
  for (const v of values) {
    const diff = v - mean;
    sq += diff * diff;
  }
  return Math.sqrt(sq / (values.length - 1));
}

export async function runTcnLookbackSensitivitySweep(
  ticker: string,
  decomposition: DecompositionResult,
  rawConfig: TcnAttentionConfig = DEFAULT_TCN_CONFIG,
  step: SensitivityStep = 4,
  options?: {
    onProgress?: (progress: TcnTrainingProgress) => void;
    yieldControl?: () => Promise<void>;
    isCancelled?: () => boolean;
  }
): Promise<TcnSensitivityResult | null> {
  const n = decomposition.observed.length;
  if (n < MIN_BARS_FOR_TCN) return null;

  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const baseConfig = sanitizeTcnConfig(rawConfig, n);
  const lookbacks = generateLookbackSweepGrid(baseConfig.lookback, step, n);
  const totalWindows = lookbacks.length;

  const runs: SensitivityWindowRun[] = [];

  for (let wIdx = 0; wIdx < totalWindows; wIdx++) {
    if (options?.isCancelled?.()) return null;
    const lookback = lookbacks[wIdx];
    const winConfig: TcnAttentionConfig = {
      ...baseConfig,
      lookback,
    };

    const forecast = await runTcnDecompositionForecast(ticker, decomposition, winConfig, {
      onProgress: (p) => {
        options?.onProgress?.({
          ...p,
          windowIndex: wIdx,
          totalWindows,
          windowLookback: lookback,
          overallFraction: (wIdx + p.overallFraction) / totalWindows,
        });
      },
      yieldControl: options?.yieldControl,
      isCancelled: options?.isCancelled,
    });

    if (!forecast || options?.isCancelled?.()) return null;

    runs.push({
      lookback,
      isBase: lookback === baseConfig.lookback,
      forecast,
    });
  }

  if (runs.length === 0) return null;

  const baseRun = runs.find((r) => r.isBase) ?? runs[Math.floor(runs.length / 2)];
  const baseForecast = baseRun.forecast;
  const lastPrice = baseForecast.lastPrice;

  const prices = runs.map((r) => r.forecast.predictedPrice);
  const meanPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
  const stdDevPrice = sampleStdDev(prices, meanPrice);
  const cvPct = meanPrice !== 0 ? (stdDevPrice / Math.abs(meanPrice)) * 100 : 0;
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRangeDollar = maxPrice - minPrice;
  const priceRangePct = lastPrice !== 0 ? (priceRangeDollar / lastPrice) * 100 : 0;

  const baseHalfPi = Math.max((baseForecast.upper99 - baseForecast.lower99) / 2, 1e-6);
  const dispersionToPiRatio = stdDevPrice / baseHalfPi;

  let upCount = 0;
  let downCount = 0;
  let bestRmseRun = runs[0];
  for (const r of runs) {
    if (r.forecast.deltaDollar >= 0) upCount++;
    else downCount++;
    if (r.forecast.holdoutPriceRmse < bestRmseRun.forecast.holdoutPriceRmse) {
      bestRmseRun = r;
    }
  }

  const dominantDirection: 'up' | 'down' | 'mixed' =
    upCount > downCount ? 'up' : downCount > upCount ? 'down' : 'mixed';
  const directionalAgreementPct = (Math.max(upCount, downCount) / runs.length) * 100;

  let stabilityLabel: SensitivityStabilityLabel;
  if (directionalAgreementPct >= 80 && dispersionToPiRatio <= 0.25) {
    stabilityLabel = 'Stable';
  } else if (directionalAgreementPct >= 60 && dispersionToPiRatio <= 0.55) {
    stabilityLabel = 'Moderate';
  } else {
    stabilityLabel = 'Sensitive';
  }

  // Per-component stability and normalized dollar-attribution variance share
  const rawCompMetrics = {} as Record<
    DecompositionComponentKey,
    Omit<ComponentStabilityMetric, 'varianceSharePct'>
  >;
  let totalContribVar = 0;

  for (const { key, label } of COMPONENT_ORDER) {
    const vals = runs.map((r) => r.forecast.components[key].predictedValue);
    const contribs = runs.map((r) => r.forecast.components[key].contributionDollar);
    const meanValue = vals.reduce((a, b) => a + b, 0) / vals.length;
    const stdDevValue = sampleStdDev(vals, meanValue);
    const meanContrib = contribs.reduce((a, b) => a + b, 0) / contribs.length;
    const contribDollarStdDev = sampleStdDev(contribs, meanContrib);
    const minContribDollar = Math.min(...contribs);
    const maxContribDollar = Math.max(...contribs);

    totalContribVar += contribDollarStdDev * contribDollarStdDev;

    rawCompMetrics[key] = {
      component: key,
      label,
      baseValue: baseForecast.components[key].predictedValue,
      meanValue,
      stdDevValue,
      minValue: Math.min(...vals),
      maxValue: Math.max(...vals),
      minContribDollar,
      maxContribDollar,
      contribDollarSpread: maxContribDollar - minContribDollar,
      contribDollarStdDev,
    };
  }

  const componentStability = {} as Record<DecompositionComponentKey, ComponentStabilityMetric>;
  for (const { key } of COMPONENT_ORDER) {
    const raw = rawCompMetrics[key];
    const varianceSharePct =
      totalContribVar > 1e-12
        ? ((raw.contribDollarStdDev * raw.contribDollarStdDev) / totalContribVar) * 100
        : 25;
    componentStability[key] = {
      ...raw,
      varianceSharePct,
    };
  }

  const t1 = typeof performance !== 'undefined' ? performance.now() : 0;

  return {
    ticker,
    model: decomposition.model,
    baseLookback: baseRun.lookback,
    step,
    runs,
    baseForecast,
    summary: {
      meanPrice,
      stdDevPrice,
      cvPct,
      minPrice,
      maxPrice,
      priceRangeDollar,
      priceRangePct,
      dispersionToPiRatio,
      upCount,
      downCount,
      dominantDirection,
      directionalAgreementPct,
      bestRmseLookback: bestRmseRun.lookback,
      bestRmseValue: bestRmseRun.forecast.holdoutPriceRmse,
      stabilityLabel,
      totalDurationMs: Math.max(1, Math.round(t1 - t0)),
    },
    componentStability,
  };
}
