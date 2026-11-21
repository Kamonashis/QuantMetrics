'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Circle,
  Cpu,
  RefreshCw,
  Scale,
  Sigma,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import type { PriceBar, IntradayBar, RollingVolSeries } from '@/lib/quant-engine';
import {
  calculateReturns,
  decomposeTimeSeries,
  computeAdfTest,
  computeRealizedVolatilityAndJumps,
  fitHarRvModel,
} from '@/lib/quant-engine';
import {
  computeStructuralBreaks,
  computeHurstAndVarianceRatio,
  computeRollingMomentsWithBands,
  computeVarianceBudget,
  computeDiurnalRvReconciliation,
  computeInnovationDiagnostics,
} from '@/lib/analysis-engine';
import { fitGaussianHmm, type HmmFit } from '@/lib/hmm';
import {
  evaluateVolSignals,
  forwardImpliedVol,
  forwardRealizedVol,
  varianceBaseline,
  varianceRatio,
  rvTriangle,
  breakevenVol,
  BREAKEVEN_MULT,
  type SignalRuleResult,
  type SignalState,
} from '@/lib/vol-signals';
import { RollingVolatilityChart } from '@/components/charts/RollingVolatilityChart';
import { IvRvSpreadChart, type IvRvSpreadPoint } from '@/components/charts/IvRvSpreadChart';
import { ExpectedMoveConeChart } from '@/components/charts/ExpectedMoveConeChart';
import { ReconciliationLog } from './ReconciliationLog';
import { RegimeTimeline, regimeNames } from '@/components/charts/RegimeTimeline';
import { useTcnVolatility } from '@/hooks/useTcnVolatility';
import { evaluateTcnConfluence } from '@/lib/tcn-multi-domain';
import {
  getLatestCachedTcnArtifactByTicker,
  type TcnDecompositionForecast,
} from '@/lib/tcn-attention';
import type { ChainPayload, IvHistoryEntry } from './types';

interface T8DecisionTabProps {
  chain: ChainPayload | null;
  bars: PriceBar[];
  ticker: string;
  ivHistory: IvHistoryEntry[];
  intradayBars: IntradayBar[];
  /** 'live_market' | 'synthetic_gbm' | null — provenance of the intraday cache */
  intradaySource?: string | null;
  /** Canonical RV series from the shared snapshot; no local recomputation. */
  rvSeries?: RollingVolSeries[];
  garchFitProp?: ReturnType<typeof import('@/lib/quant-engine').fitGarch11> | null;
  /** Earnings from the shared snapshot — no second client fetch. */
  earningsProp?: { ticker: string; date: string; isEstimate: boolean; sessions: number } | null;
}

// ---------------------------------------------------------------------------
// Module-level pure helpers
// ---------------------------------------------------------------------------

function latestOf(series: RollingVolSeries[], window: number): number | null {
  const s = series.find((sv) => sv.window === window);
  const v = s?.values[s.values.length - 1]?.vol;
  return v === undefined ? null : v;
}

const pct = (v: number | null, digits = 1) => (v === null ? '—' : `${(v * 100).toFixed(digits)}%`);
const signedPct = (v: number | null, digits = 1) =>
  v === null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;

function Tile({ label, value, sub, tone = 'default' }: { label: string; value: string; sub?: string; tone?: 'default' | 'gold' | 'warn' }) {
  const border =
    tone === 'gold'
      ? 'border-gold-500/40 shadow-[0_0_0_1px_rgba(222,192,132,0.25)]'
      : tone === 'warn'
        ? 'border-amber-500/40'
        : 'border-[var(--hairline)]';
  return (
    <div className={`panel-inset flex flex-col justify-center gap-0.5 border px-3 py-2.5 ${border}`}>
      <span className="font-mono text-[9px] font-semibold uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">{label}</span>
      <span className="font-mono text-lg font-bold tabular-nums leading-none text-zinc-900 dark:text-zinc-50">{value}</span>
      {sub && <span className="font-mono text-[9px] tabular-nums text-zinc-500 dark:text-zinc-400">{sub}</span>}
    </div>
  );
}

function MeterRow({ label, value, hint, tone = 'gold' }: { label: string; value: number | null; hint?: string; tone?: 'gold' | 'warn' | 'zinc' }) {
  const pctValue = value === null ? null : Math.min(100, Math.max(0, value * 100));
  const bar = tone === 'warn' ? 'bg-amber-500/70' : tone === 'zinc' ? 'bg-zinc-500/60' : 'bg-gold-500/80';
  return (
    <div className="flex items-center gap-3">
      <div className="w-44 shrink-0">
        <div className="font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em] text-zinc-500 dark:text-zinc-400">{label}</div>
        {hint && <div className="font-mono text-[8px] text-zinc-400 dark:text-zinc-500">{hint}</div>}
      </div>
      <div className="h-3.5 flex-1 overflow-hidden rounded-sm bg-[var(--bg-inset)]">
        {pctValue !== null && <div className={`h-full rounded-sm ${bar}`} style={{ width: `${pctValue.toFixed(1)}%` }} />}
      </div>
      <span className="w-12 shrink-0 text-right font-mono text-[11px] font-bold tabular-nums text-zinc-800 dark:text-zinc-100">
        {pctValue !== null ? `${pctValue.toFixed(0)}%` : '—'}
      </span>
    </div>
  );
}

function StatePill({ state }: { state: SignalState }) {
  if (state === 'met')
    return (
      <span className="inline-flex items-center gap-1 rounded border border-gold-500/50 bg-gold-500/15 px-1.5 py-0.5 font-mono text-[9.5px] font-bold uppercase tracking-wider text-gold-800 dark:text-gold-200">
        <CheckCircle2 className="h-2.5 w-2.5" /> met
      </span>
    );
  if (state === 'not-met')
    return (
      <span className="inline-flex items-center gap-1 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-1.5 py-0.5 font-mono text-[9.5px] font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
        <Circle className="h-2.5 w-2.5" /> not met
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 rounded border border-amber-500/50 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[9.5px] font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300">
      <Circle className="h-2.5 w-2.5" /> no data
    </span>
  );
}

// ---------------------------------------------------------------------------
// T8 — Decision tab
// ---------------------------------------------------------------------------

export function T8DecisionTab({ chain, bars, ticker, ivHistory, intradayBars, intradaySource = null, rvSeries = [], garchFitProp = null, earningsProp = null }: T8DecisionTabProps) {
  const dates = useMemo(() => bars.map((b) => b.date), [bars]);
  const prices = useMemo(() => bars.map((b) => b.close), [bars]);
  const returns = useMemo(() => calculateReturns(prices, 'log'), [prices]);

  // Rolling RV ladder — one canonical series from the shared snapshot.
  const rolling = rvSeries;
  const rvL = useMemo(
    () => ({
      rv5: latestOf(rolling, 5),
      rv10: latestOf(rolling, 10),
      rv20: latestOf(rolling, 20),
      rv30: latestOf(rolling, 30),
      rv60: latestOf(rolling, 60),
    }),
    [rolling],
  );
  const { rv5, rv10, rv20, rv30, rv60 } = rvL;

  const iv30 = chain?.iv30 ?? null;
  const chainAvailable = chain?.available === true;
  const breakeven = iv30 !== null ? breakevenVol(iv30) : null;
  const varRatio30 = varianceRatio(rv30, iv30);

  // Forward vol: realized (variance decomposition) and implied (w-difference)
  const fwdReal10_30 = rv10 !== null && rv30 !== null ? forwardRealizedVol(rv10, rv30, 10, 30) : null;
  const fwdReal30_60 = rv30 !== null && rv60 !== null ? forwardRealizedVol(rv30, rv60, 30, 60) : null;
  const tsPoints = useMemo(() => chain?.termStructure ?? [], [chain]);
  const fwdImp10_30 = useMemo(() => forwardImpliedVol(tsPoints, 10, 30), [tsPoints]);
  const fwdImp30_60 = useMemo(() => forwardImpliedVol(tsPoints, 30, 60), [tsPoints]);
  const baseline = rv10 !== null && rv30 !== null ? varianceBaseline(rv10, rv30, 10, 30) : null;

  // HAR-RV next-day vol — only when intraday 1-min data exists for the session (never fabricated)
  const harNextDay = useMemo(() => {
    if (!intradayBars || intradayBars.length === 0) return null;
    try {
      const daily = computeRealizedVolatilityAndJumps(intradayBars);
      if (daily.length < 6) return null;
      return fitHarRvModel(daily).forecastNextDayVol ?? null;
    } catch {
      return null;
    }
  }, [intradayBars]);

  // Variance budget — what of the measured variance can a hedge even harvest
  const budget = useMemo(() => computeVarianceBudget(intradayBars, bars, 10), [intradayBars, bars]);
  const recon = useMemo(() => computeDiurnalRvReconciliation(intradayBars), [intradayBars]);
  const intradaySynthetic = intradaySource !== 'live_market';
  const reconRatio = recon.ratio;
  const reconSane = reconRatio !== null && reconRatio >= 0.5 && reconRatio <= 1.5;

  // RV triangle
  const triangle = rvTriangle(rv5, rv10, rv20, rv30);

  // IV-vs-RV aligned points for the spread chart (accrual store joined to RV series by date)
  const spreadPoints = useMemo<IvRvSpreadPoint[]>(() => {
    const rv20Series = rolling.find((s) => s.window === 20)?.values ?? [];
    const ivMap = new Map<string, number | null>();
    for (const h of ivHistory) ivMap.set(h.date, h.iv30);
    return rv20Series.map((v) => ({ date: v.date, iv30: ivMap.get(v.date) ?? null, rv: v.vol }));
  }, [rolling, ivHistory]);

  // HMM regime fit — same feature set/rank as ModelingView ([r, |r|, log GK-RV], rankFeature=2)
  // hmmModel.dates[i] is the session date for posterior i (features start at
  // bars[1] and skip invalid bars), so it must be collected in the same loop.
  const hmmModel = useMemo<{ fit: HmmFit | null; dates: string[] }>(() => {
    const hmmDates: string[] = [];
    if (!bars || bars.length < 60) return { fit: null, dates: hmmDates };
    const feats: number[][] = [];
    for (let i = 1; i < bars.length; i++) {
      const b = bars[i];
      const prev = bars[i - 1];
      const o = b.open, h = b.high, l = b.low, c = b.close;
      if (!(o > 0 && h > 0 && l > 0 && c > 0 && prev.close > 0)) continue;
      const r = Math.log(c / prev.close);
      const gk = 0.5 * Math.log(h / l) ** 2 - (2 * Math.LN2 - 1) * Math.log(c / o) ** 2;
      feats.push([r, Math.abs(r), Math.log(Math.max(gk, 1e-8))]);
      hmmDates.push(b.date);
    }
    return { fit: fitGaussianHmm(feats, { k: 3, rankFeature: 2 }), dates: hmmDates };
  }, [bars]);
  const hmm = hmmModel.fit;

  const hmmMeta = useMemo(() => {
    if (!hmm || hmm.posteriors.length === 0) return null;
    const K = hmm.posteriors[0].length;
    const names = regimeNames(K);
    const last = hmm.posteriors[hmm.posteriors.length - 1];
    let dominant = 0;
    for (let k = 1; k < K; k++) if ((last[k] ?? 0) > (last[dominant] ?? 0)) dominant = k;
    return {
      names,
      dominant,
      dominantName: names[dominant] ?? `State ${dominant}`,
      dominantPost: last[dominant] ?? 0,
      crisisProb: last[K - 1] ?? 0,
      crisisOccupancy: hmm.occupancy[K - 1] ?? 0,
      crisisDuration: hmm.expectedDurations[K - 1] ?? null,
      bic: hmm.bic,
    };
  }, [hmm]);

  // Rolling 60d skew (Lo 2002 bands)
  const skew60 = useMemo(() => {
    if (dates.length < 70) return null;
    const m = computeRollingMomentsWithBands(dates, returns, 60);
    const last = m.rollingSkew[m.rollingSkew.length - 1];
    return last ? { skew: last.skew, lowerCi: last.lowerCi, upperCi: last.upperCi } : null;
  }, [dates, returns]);

  // CUSUM-SQ breaks — sessions since the most recent one
  const cusumBreakSessionsAgo = useMemo(() => {
    if (dates.length < 20) return null;
    const res = computeStructuralBreaks(dates, returns, prices);
    const last = res.breakDates[res.breakDates.length - 1];
    if (!last) return null;
    const idx = dates.lastIndexOf(last.date);
    return idx >= 0 ? dates.length - 1 - idx : null;
  }, [dates, returns, prices]);

  // ADF on decomposition residuals — is the trend/seasonal decomposition stable?
  const adfResidualP = useMemo(() => {
    if (dates.length < 60) return null;
    const decomp = decomposeTimeSeries(dates, prices, 30, 'additive');
    const resid = decomp.residual.filter((v): v is number => v !== null && isFinite(v));
    if (resid.length < 30) return null;
    return computeAdfTest(resid).pValue;
  }, [dates, prices]);

  // Lo–MacKinlay VR — strongest mean-reversion case across horizons
  const vrMinZ = useMemo(() => {
    if (returns.length < 40) return null;
    const vr = computeHurstAndVarianceRatio(returns);
    return vr.varianceRatios.reduce((m, v) => Math.min(m, v.zStat), Infinity);
  }, [returns]);

  // Front-slice 25Δ RR — is the upside call wing already priced?
  const rr25Front = useMemo(() => {
    const slices = chain?.slices ?? [];
    const front = slices.find((s) => s.dte >= 15 && s.dte <= 45 && s.rr25 !== null);
    return front?.rr25 ?? slices.find((s) => s.rr25 !== null)?.rr25 ?? null;
  }, [chain]);

  // σ̂-standardized innovation diagnostics — kills the vol-clustering artifact before it becomes a "trend"
  const innovation = useMemo(() => {
    if (returns.length < 60) return null;
    try {
      return garchFitProp ? computeInnovationDiagnostics(returns, garchFitProp.conditionalVariance) : null;
    } catch {
      return null;
    }
  }, [returns, garchFitProp]);

  const earningsCurrent = earningsProp ?? null;

  // The pre-registered signal table, evaluated
  const signals = useMemo(
    () =>
      evaluateVolSignals({
        iv30,
        rv5: rvL.rv5,
        rv10: rvL.rv10,
        rv20: rvL.rv20,
        rv30: rvL.rv30,
        skew60: skew60?.skew ?? null,
        skew60LowerCi: skew60?.lowerCi ?? null,
        skew60UpperCi: skew60?.upperCi ?? null,
        vrMinZ,
        adfResidualP,
        cusumBreakSessionsAgo,
        hmmCrisisProb: hmmMeta?.crisisProb ?? null,
        hmmDominantState: hmmMeta?.dominant ?? null,
        rr25Front,
      }),
    [iv30, rvL, skew60, vrMinZ, adfResidualP, cusumBreakSessionsAgo, hmmMeta, rr25Front],
  );

  const signalSummary = useMemo(() => {
    let met = 0, notMet = 0, noData = 0;
    for (const s of signals) {
      if (s.state === 'met') met += 1;
      else if (s.state === 'not-met') notMet += 1;
      else noData += 1;
    }
    return { met, notMet, noData };
  }, [signals]);

  const longGamma = signals.find((s) => s.id === 'long-gamma');
  const entryMet = longGamma?.state === 'met';

  // TCN-Attention multi-horizon (1d / 5d / 22d) Volatility + cached Decomposition confluence
  const { volForecast: tcnVol, isTraining: tcnVolTraining, retrain: retrainTcnVol } = useTcnVolatility(
    ticker,
    bars,
    { autoRun: true }
  );
  const [cachedDecomp, setCachedDecomp] = useState<{ ticker: string; forecast: TcnDecompositionForecast } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled) return;
      const art = getLatestCachedTcnArtifactByTicker(ticker);
      setCachedDecomp(art ? { ticker: ticker.toUpperCase(), forecast: art.forecast } : null);
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [ticker]);

  const activeDecompForecast =
    cachedDecomp && cachedDecomp.ticker === ticker.toUpperCase() ? cachedDecomp.forecast : null;

  const tcnOrientation: 'short-premium' | 'long-gamma' | 'neutral' = entryMet
    ? 'long-gamma'
    : iv30 !== null && rv20 !== null && iv30 - rv20 >= 0.02
      ? 'short-premium'
      : 'neutral';

  const tcnConfluence = useMemo(
    () =>
      evaluateTcnConfluence({
        iv30,
        rv20,
        volForecast: tcnVol,
        decompForecast: activeDecompForecast,
        primaryOrientation: tcnOrientation,
      }),
    [iv30, rv20, tcnVol, activeDecompForecast, tcnOrientation]
  );

  // Series for the breakeven chart — RV windows vs IV30 and 1.25×IV30
  const breakevenSeries = useMemo(() => rolling.filter((s) => s.window === 5 || s.window === 10 || s.window === 30), [rolling]);

  const rollingRef = rolling.find((s) => s.window === 5);

  return (
    <div className="space-y-4">
      {/* Local banner — IV unavailable */}
      {!chainAvailable && (
        <div className="flex items-center gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2 font-mono text-[10px] uppercase tracking-[0.16em] text-amber-700 dark:text-amber-300">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> Chain unavailable ({chain?.reason ?? 'unknown'}) — IV panels show no-data; realized-vol panels unaffected.
        </div>
      )}

      {/* ── 1 ── Apples-to-apples strip: the honest comparison set ───────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Apples-to-Apples Strip
            </h3>
            <p className="mt-0.5 max-w-xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              IV30 against the trailing RV windows and the pre-spike variance decomposition — never against the 250-day
              blended annualized figure, which averages across CUSUM regime breaks and answers a different question.
            </p>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400">
            ticker {ticker}
          </span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
          <Tile label="RV 10d" value={pct(rv10)} sub="burst window" />
          <Tile label="RV 20d" value={pct(rv20)} sub="month" />
          <Tile label="RV 30d" value={pct(rv30)} sub="≈ term" />
          <Tile label="IV-30" value={pct(iv30)} sub={iv30 !== null ? 'BSM · √365 convention' : 'chain down'} tone={iv30 !== null ? 'gold' : 'warn'} />
          <Tile
            label="RV² / IV²"
            value={varRatio30 !== null ? `${varRatio30.toFixed(2)}×` : '—'}
            sub={varRatio30 !== null ? (varRatio30 >= 1 ? 'realized richer (30d)' : 'implied richer (30d)') : 'needs chain + history'}
          />
          <Tile label="Vol of spike vs rest" value={pct(baseline)} sub="RV30² minus RV10², t=10→30" />
          <Tile label="HAR 1d fwd" value={pct(harNextDay)} sub={harNextDay !== null ? 'intraday RV fit' : 'needs intraday bars'} />
        </div>
        <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500 dark:text-zinc-500">
          note — RV &gt; IV ⇒ the gap is in the front; the back already realized {pct(baseline)}. RV is √252, IV is √365 — annualizations of the same total variance, so they are compared directly and never rescaled; ANN VOL (250d) is shown nowhere on this tab on purpose.
        </p>
      </div>

      {/* ── 1a ── TCN-Attention Multi-Horizon VRP & Signal Confluence ─────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <Cpu className="h-3.5 w-3.5 text-gold-600 dark:text-gold-400" />
              <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
                TCN-Attention Forward VRP &amp; Signal Confluence
              </h3>
              <span
                className={`rounded border px-2 py-0.5 font-mono text-[9px] font-bold uppercase tracking-[0.16em] ${
                  tcnConfluence.alignment === 'ALIGNED'
                    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                    : tcnConfluence.alignment === 'DIVERGENT'
                      ? 'border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400'
                      : tcnConfluence.alignment === 'MIXED'
                        ? 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300'
                        : 'border-[var(--hairline)] bg-[var(--bg-inset)] text-zinc-500'
                }`}
              >
                {tcnConfluence.alignment}
              </span>
              {tcnConfluence.volRegimeRead !== 'UNAVAILABLE' && (
                <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500 dark:text-zinc-400">
                  · {tcnConfluence.volRegimeRead}
                </span>
              )}
            </div>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              Forward realized volatility projected by the shared dilated-causal TCN-Attention engine (1d · 5d · 22d QLIKE head) paired directly with live IV30 and any cached Decomposition price signal for {ticker}.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {tcnVol && (
              <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
                {tcnVol.cacheHit ? 'cached' : `${tcnVol.trainingDurationMs}ms`}
              </span>
            )}
            <button
              type="button"
              onClick={retrainTcnVol}
              disabled={tcnVolTraining || bars.length < 70}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1 font-mono text-[9.5px] font-semibold uppercase tracking-wider text-zinc-700 hover:border-gold-500/40 hover:text-gold-700 disabled:opacity-50 dark:text-zinc-200 dark:hover:text-gold-300"
            >
              <RefreshCw className={`h-3 w-3 ${tcnVolTraining ? 'animate-spin' : ''}`} />
              {tcnVolTraining ? 'Training…' : 'Retrain'}
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
          <Tile
            label="TCN RV 1d"
            value={pct(tcnConfluence.tcnRv1d)}
            sub={tcnVol ? `99% ${pct(tcnVol.horizons[1].lower99Vol, 0)}–${pct(tcnVol.horizons[1].upper99Vol, 0)}` : 'needs ≥70 bars'}
          />
          <Tile
            label="TCN RV 5d"
            value={pct(tcnConfluence.tcnRv5d)}
            sub={tcnVol ? `99% ${pct(tcnVol.horizons[5].lower99Vol, 0)}–${pct(tcnVol.horizons[5].upper99Vol, 0)}` : 'weekly anchor'}
          />
          <Tile
            label="TCN RV 22d"
            value={pct(tcnConfluence.tcnRv22d)}
            sub={tcnVol ? `OOS QLIKE ${tcnVol.horizons[22].holdoutQlike.toFixed(3)}` : 'monthly anchor'}
          />
          <Tile
            label="IV30 − TCN RV22"
            value={signedPct(tcnConfluence.forwardVrp22)}
            sub={
              tcnConfluence.forwardVrp22 !== null
                ? 'direct √365 vs √252 VRP'
                : 'needs live chain + TCN'
            }
            tone={tcnConfluence.forwardVrp22 !== null ? 'gold' : 'default'}
          />
          <Tile
            label="TCN 5d − 22d"
            value={signedPct(tcnConfluence.tcnVolSlope)}
            sub={
              tcnConfluence.tcnVolSlope === null
                ? 'term curve'
                : tcnConfluence.tcnVolSlope > 0.01
                  ? 'backwardation'
                  : tcnConfluence.tcnVolSlope < -0.01
                    ? 'contango'
                    : 'flat curve'
            }
          />
          <Tile
            label="Decomp t+1 Tilt"
            value={
              tcnConfluence.priceTiltPct !== null
                ? `${tcnConfluence.priceTiltPct >= 0 ? '+' : ''}${tcnConfluence.priceTiltPct.toFixed(2)}%`
                : '—'
            }
            sub={
              tcnConfluence.hasDecompArtifact
                ? `${tcnConfluence.cyclicalSnrRegime ?? 'SNR n/a'}${tcnConfluence.backtestSharpe30d !== null ? ` · Sh ${tcnConfluence.backtestSharpe30d.toFixed(2)}` : ''}`
                : 'run Analysis → Decomp'
            }
          />
        </div>

        {tcnConfluence.rationale.length > 0 && (
          <ul className="mt-2.5 space-y-0.5 border-t border-[var(--hairline)] pt-2 font-mono text-[9.5px] leading-snug text-zinc-500 dark:text-zinc-400">
            {tcnConfluence.rationale.map((line) => (
              <li key={line}>• {line}</li>
            ))}
          </ul>
        )}
      </div>

      {/* ── 1b ── The variance budget: what of this is harvestable ─────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Variance Budget — what a hedge can actually capture
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              RV charts display all of it; delta-hedging harvests only part of it. Bipower variation splits diffusion from jumps, the signature plateau bounds microstructure noise, and the close→open split measures gap variance — estimators on overlapping windows, not a partition.
            </p>
          </div>
          {intradaySynthetic && (
            <span className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-amber-500/50 bg-amber-500/10 px-2.5 py-1.5 font-mono text-[9px] font-bold uppercase tracking-[0.16em] text-amber-700 dark:text-amber-300">
              <AlertTriangle className="h-3.5 w-3.5" /> {intradaySource ?? 'not live'} · illustrative
            </span>
          )}
        </div>

        <div className="mt-3 space-y-2">
          <MeterRow label="Continuous diffusion" hint="bipower variation · the hedgeable part" value={budget.continuousShare} tone="gold" />
          <MeterRow label="Jump + residual" hint="RV − BV · incl. contamination" value={budget.jumpResidualShare} tone="zinc" />
          <MeterRow label="Microstructure noise" hint="signature plateau vs 1-min RV · LOWER BOUND" value={budget.noiseShareLowerBound} tone="warn" />
          <MeterRow label="Overnight gap" hint="close→open vs intraday · options-only or untradeable" value={budget.overnightShare} tone="warn" />
          <div className="border-t border-[var(--hairline)] pt-2">
            <MeterRow label="Harvestable intraday" hint="continuous × (1 − overnight)" value={budget.harvestableShare} tone="gold" />
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--hairline)] pt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
          <span>
            diurnal ÷ intraday RV:{' '}
            <span className={reconSane ? 'text-[var(--positive)]' : 'font-bold text-amber-600 dark:text-amber-300'}>
              {reconRatio !== null ? `${reconRatio.toFixed(2)}× — ${reconSane ? 'sane' : 'outside 0.5–1.5, do not size off the diurnal panel'}` : 'n/a'}
            </span>
          </span>
          <span>{budget.intradayDaysAvailable} intraday sessions · window 10 td</span>
        </div>
        {budget.notes.length > 0 && (
          <ul className="mt-2 list-disc space-y-0.5 pl-4 font-mono text-[9px] leading-snug text-zinc-400 dark:text-zinc-500">
            {budget.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
      </div>

      {/* ── 2 ── Breakeven chart: RV windows vs IV30 and 1.25× IV30 ─────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Breakeven Line — {BREAKEVEN_MULT}× IV30
              {breakeven !== null && <span className="ml-2 font-mono text-gold-700 dark:text-gold-300">{pct(breakeven)}</span>}
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              A delta-hedged long straddle clears costs only if realized beats ~{BREAKEVEN_MULT}× IV. The rule fires while
              RV5 &gt; {pct(breakeven)}. The {BREAKEVEN_MULT}× factor is a heuristic (costs + hedge slippage) — calibrate it via backtest once the IV store accrues ~60 sessions.
            </p>
          </div>
          <div
            className={`inline-flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.16em] ${
              entryMet
                ? 'border-gold-500/50 bg-gold-500/15 text-gold-800 dark:text-gold-200'
                : 'border-[var(--hairline)] bg-[var(--bg-inset)] text-zinc-500 dark:text-zinc-400'
            }`}
          >
            <Scale className="h-3.5 w-3.5" />
            {entryMet ? 'entry: RV5 clears it' : 'entry: not firing'}
          </div>
        </div>
        <div className="mt-3">
          <RollingVolatilityChart
            series={breakevenSeries}
            title="RV5 / RV10 / RV30 vs implied"
            referenceLines={
              iv30 !== null && breakeven !== null
                ? [
                    { value: iv30, label: `IV30 ${pct(iv30)}` },
                    { value: breakeven, label: `1.25× ${pct(breakeven)}` },
                  ]
                : []
            }
          />
        </div>
        <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[9px] text-zinc-400">
          <span>RV5 {pct(rv5)}</span>
          <span>RV10 {pct(rv10)}</span>
          <span>RV20 {pct(rv20)}</span>
          <span>RV30 {pct(rv30)}</span>
          <span>RV60 {pct(rv60)}</span>
          {rollingRef && <span className="text-zinc-500">window sample stdev · √252</span>}
        </p>
      </div>

      {/* ── 3 ── Spread & z-score ─────────────────────────────────────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Gap, Charted — IV30 − RV20 with its z-score
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              Accrued single-name IV versus RV — a different series from T3&apos;s market-proxy tape. Positive &amp; rising: implied is harvesting the spike premium. Negative: realized is outrunning the price — the only state in which buying gamma is free money, and it rarely lasts.
            </p>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400">z window 20 · needs ~15 sessions</span>
        </div>
        <div className="mt-3">
          <IvRvSpreadChart points={spreadPoints} ivLabel="IV30 (accrued)" rvLabel="RV 20d" />
        </div>
      </div>

      {/* ── 4 ── Expected-move cone ────────────────────────────────────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Price Cone — where spot can be, at IV30
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              One sigma at 10td and 21td is the whole front-end trade in two numbers: breakevens, strike selection, and whether the straddle price is in a sane region.
            </p>
          </div>
          <span className="flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400">
            <Sigma className="h-3 w-3" /> 1σ gold · 2σ faint
          </span>
        </div>
        <div className="mt-3">
          <ExpectedMoveConeChart bars={bars} iv30={iv30} />
        </div>
      </div>

      {/* ── 5 ── RV triangle + forward-vol decomposition ───────────────────── */}
      <div className="grid gap-4 xl:grid-cols-5">
        {/* triangle */}
        <div className="panel p-5 xl:col-span-2">
          <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            The RV Triangle — 5·10·20·30
          </h3>
          <p className="mt-0.5 text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
            Shape of realized vol across horizons. Front-rich ⇒ decaying spike; back-rich ⇒ building. The trade lives in the front when the triangle points down.
          </p>
          {triangle ? (
            <div className="mt-3 flex flex-col items-center gap-3">
              <div
                className={`inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-[0.2em] ${
                  triangle.shape === 'backwardation'
                    ? 'border-gold-500/50 bg-gold-500/15 text-gold-800 dark:text-gold-200'
                    : triangle.shape === 'contango'
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                      : 'border-[var(--hairline)] bg-[var(--bg-inset)] text-zinc-500 dark:text-zinc-400'
                }`}
              >
                {triangle.shape === 'backwardation' ? <TrendingDown className="h-3.5 w-3.5" /> : triangle.shape === 'contango' ? <TrendingUp className="h-3.5 w-3.5" /> : null}
                {triangle.shape}
              </div>
              {/* ladder */}
              <div className="grid w-full grid-cols-4 gap-1.5 text-center">
                {[
                  { label: 'RV5', v: rv5 },
                  { label: 'RV10', v: rv10 },
                  { label: 'RV20', v: rv20 },
                  { label: 'RV30', v: rv30 },
                ].map(({ label, v }) => {
                  const max = Math.max(rv5 ?? 0, rv10 ?? 0, rv20 ?? 0, rv30 ?? 0, 1e-9);
                  const h = v === null ? 4 : Math.max(6, (v / max) * 56);
                  return (
                    <div key={label} className="flex flex-col items-center gap-1">
                      <span className="font-mono text-[10px] font-bold tabular-nums text-zinc-800 dark:text-zinc-100">{pct(v)}</span>
                      <div className="flex h-16 w-full items-end rounded bg-[var(--bg-inset)] p-1">
                        <div className="w-full rounded-sm bg-gold-500/70" style={{ height: `${h}px` }} />
                      </div>
                      <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-500">{label}</span>
                    </div>
                  );
                })}
              </div>
              <div className="flex w-full flex-wrap items-center justify-center gap-4 font-mono text-[9px] text-zinc-500">
                <span>RV10 − RV20 {signedPct(triangle.spread1020)}</span>
                <span>RV20 − RV30 {signedPct(triangle.spread2030)}</span>
              </div>
            </div>
          ) : (
            <p className="mt-4 font-mono text-[10px] text-zinc-500">Insufficient history for the RV ladder.</p>
          )}
        </div>

        {/* forward-vol decomposition */}
        <div className="panel p-5 xl:col-span-3">
          <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            Forward Volatility — realized decomposition vs the quoted calendar
          </h3>
          <p className="mt-0.5 text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
            Left column: the variance the next leg of history must still contain, implied by the trailing windows. Right column: what the option ladder is charging for the same bucket (σ²T difference, never σ interpolation).
          </p>
          <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="panel-inset border border-[var(--hairline)] p-3">
              <p className="font-mono text-[9px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Realized forward (decomposition)</p>
              <div className="mt-2 divide-y divide-[var(--hairline)] font-mono text-[11px]">
                {[
                  { label: '10→30d (days 10–30 back)', v: fwdReal10_30, note: 'the spent segment' },
                  { label: '30→60d (days 30–60 back)', v: fwdReal30_60, note: 'older shell' },
                ].map(({ label, v, note }) => (
                  <div key={label} className="flex items-center justify-between gap-2 py-1.5">
                    <span className="text-zinc-500 dark:text-zinc-400">{label}<span className="block text-[9.5px] text-zinc-400 dark:text-zinc-500">{note}</span></span>
                    <span className="tabular-nums font-bold text-zinc-900 dark:text-zinc-100">{pct(v)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel-inset border border-[var(--hairline)] p-3">
              <p className="font-mono text-[9px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Implied forward (w-difference)</p>
              <div className="mt-2 divide-y divide-[var(--hairline)] font-mono text-[11px]">
                {[
                  { label: '10→30d calendar', v: fwdImp10_30 },
                  { label: '30→60d calendar', v: fwdImp30_60 },
                ].map(({ label, v }) => (
                  <div key={label} className="flex items-center justify-between gap-2 py-1.5">
                    <span className="text-zinc-500 dark:text-zinc-400">{label}</span>
                    <span className="tabular-nums font-bold text-zinc-900 dark:text-zinc-100">{pct(v)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 font-mono text-[9.5px] leading-snug text-zinc-400 dark:text-zinc-500">
                {tsPoints.length >= 2
                  ? `from ${tsPoints.length} quoted tenors · null if the bucket's forward variance is inverted`
                  : `needs ≥ 2 fitted slices in the ladder — chain returned ${tsPoints.length}`}
              </p>
            </div>
          </div>
          <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
            read — implied 10→30 fwd above realized 10→30 ⇒ the spike tail is still being paid for · below ⇒ decay is already priced
          </p>
        </div>
      </div>

      {/* ── 6 ── HMM regime strip ─────────────────────────────────────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              Return HMM (k=3 · features r, |r|, log GK-RV) — calm→crisis
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              Unsupervised occupancy, shaded by dominant-state posterior. Watch the transition matrix, not the label: calm here survives days, not weeks. HMM is a risk gate, never a timing edge.
            </p>
          </div>
          {hmmMeta && (
            <div className="flex flex-wrap items-center gap-2 font-mono text-[9px]">
              <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 uppercase tracking-wider text-zinc-400">
                now: <span className="font-bold text-gold-700 dark:text-gold-300">{hmmMeta.dominantName}</span> {(hmmMeta.dominantPost * 100).toFixed(0)}%
              </span>
              <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 uppercase tracking-wider text-zinc-400">
                P(crisis) <span className={hmmMeta.crisisProb > 0.25 ? 'font-bold text-rose-500' : 'font-bold text-zinc-200 dark:text-zinc-100'}>{(hmmMeta.crisisProb * 100).toFixed(0)}%</span>
              </span>
              <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 uppercase tracking-wider text-zinc-400">
                occupancy {(hmmMeta.crisisOccupancy * 100).toFixed(0)}%{hmmMeta.crisisDuration !== null ? ` · E[τ] ${hmmMeta.crisisDuration.toFixed(1)}d` : ''}
              </span>
            </div>
          )}
        </div>
        <div className="mt-3">
          {hmm ? (
            <RegimeTimeline posteriors={hmm.posteriors} labels={regimeNames(3)} height={54} maxBars={400} dates={hmmModel.dates} />
          ) : (
            <div className="flex h-14 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
              needs ≥ 60 sessions of bars
            </div>
          )}
        </div>
        <p className="mt-2 font-mono text-[9.5px] uppercase tracking-[0.16em] text-zinc-500">
          features [r, |r|, log GK-RV] · k=3 ranked by GK-RV · BIC {hmm?.bic !== undefined ? hmm.bic.toFixed(0) : '—'} · watchful ⇒ half size, always
        </p>
      </div>

      {/* ── 7 ── The signal checklist ─────────────────────────────────────── */}
      <div className="panel p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              The Checklist — pre-registered, inputs visible
            </h3>
            <p className="mt-0.5 max-w-2xl text-[11px] leading-snug text-zinc-500 dark:text-zinc-400">
              The table from the playbook, encoded as rules. A rule with missing inputs reports <span className="text-amber-600 dark:text-amber-300">no data</span> — it never silently flips to “met”. Your rules, not a recommendation; inputs are shown so you can disagree with them.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2 font-mono text-[9px] uppercase tracking-[0.18em]">
            <span className="rounded border border-gold-500/40 bg-gold-500/10 px-2 py-1 text-gold-800 dark:text-gold-200">{signalSummary.met} met</span>
            <span className="rounded border border-[var(--hairline)] bg-[var(--bg-inset)] px-2 py-1 text-zinc-500">{signalSummary.notMet} not</span>
            <span className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-amber-700 dark:text-amber-300">{signalSummary.noData} no data</span>
          </div>
        </div>

        {innovation && (
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
            <span className="text-gold-700 dark:text-gold-300">σ̂-standardized</span>
            <span>H {innovation.hurstExponent.toFixed(2)}</span>
            <span>min VR z {innovation.vrMinZ >= 0 ? '+' : ''}{innovation.vrMinZ.toFixed(2)}</span>
            <span>ADF p {innovation.adfPValue < 0.001 ? '<0.001' : innovation.adfPValue.toFixed(3)}</span>
            <span>→ trend overlay: <span className={innovation.hurstExponent >= 0.6 ? 'text-gold-700 dark:text-gold-300 font-bold' : 'text-zinc-400'}>{innovation.hurstExponent >= 0.6 ? 'defensible' : 'none — artifact'}</span></span>
          </div>
        )}

        {earningsCurrent && (
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-[var(--hairline)] pt-2 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
            <span className="text-gold-700 dark:text-gold-300">next earnings</span>
            <span>{earningsCurrent.date}{earningsCurrent.isEstimate ? ' (est.)' : ''}</span>
            <span>≈{earningsCurrent.sessions} sessions</span>
            {earningsCurrent.sessions <= 14 ? (
              <span className="font-bold text-amber-600 dark:text-amber-300">— do not hold short premium through it</span>
            ) : (
              <span>beyond the 14-session warning zone</span>
            )}
          </div>
        )}

        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[880px] border-collapse text-left font-mono text-[10.5px]">
            <thead>
              <tr className="border-b border-[var(--hairline)] text-[9.5px] uppercase tracking-[0.18em] text-zinc-400">
                <th className="py-1.5 pr-3 font-semibold">Rule</th>
                <th className="py-1.5 pr-3 font-semibold">Metric</th>
                <th className="py-1.5 pr-3 font-semibold">Trigger</th>
                <th className="py-1.5 pr-3 font-semibold">State</th>
                <th className="py-1.5 font-semibold">If met — do this</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--hairline)]">
              {signals.map((r) => (
                <tr key={r.id} className="align-top">
                  <td className="py-2 pr-3 font-semibold text-zinc-800 dark:text-zinc-100">{r.label}</td>
                  <td className="py-2 pr-3 tabular-nums text-zinc-500 dark:text-zinc-400">{r.measured ?? '—'}</td>
                  <td className="py-2 pr-3 text-zinc-500 dark:text-zinc-400">{r.trigger}</td>
                  <td className="py-2 pr-3"><StatePill state={r.state} /></td>
                  <td className="py-2 leading-snug text-zinc-600 dark:text-zinc-300">{r.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-[10px] leading-relaxed text-amber-800 dark:text-amber-200/90">
          <p className="font-mono font-semibold uppercase tracking-[0.18em]">Honesty chrome</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            <li>IV is a European BSM approximation on ~15-min-delayed Yahoo quotes. Greeks here are analytic, never executable.</li>
            <li>RV (√252) and IV (√365) are annualizations of the same total variance — the terminal compares them directly and never applies the cosmetic √(252/365) or √(365/252) rescale. If RV and IV disagree, that is information, not a basis error.</li>
            <li>The {BREAKEVEN_MULT}× breakeven is an uncalibrated heuristic — it has not been backtested (no IV history yet; revisit after ~60 accrued sessions).</li>
            <li>The IV store accrues one snapshot per day <em>you</em> open the chain. Panels with missing history say so instead of interpolating.</li>
            <li>These are one operator&rsquo;s pre-registered rules — signals, not orders. No broker connectivity exists in this app. Not investment advice.</li>
          </ul>
        </div>
      </div>

      {/* ── 8 ── Reconciliation log — which conclusion rests on which test ──── */}
      <ReconciliationLog n={returns.length} />
    </div>
  );
}
