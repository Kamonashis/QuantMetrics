'use client';

import React, { useMemo, useState } from 'react';
import { Target, Scale, Sigma, FlaskConical, Info } from 'lucide-react';
import {
  calculateReturns,
  fitGarch11,
  computeEwmaVolatility,
} from '@/lib/quant-engine';
import type { PriceBar } from '@/lib/quant-engine';
import {
  forwardRealizedVariance,
  qlike,
  mseLogVariance,
  mincerZarnowitz,
  dieboldMariano,
  encompassingWeights,
  fitGarchX,
} from '@/lib/vol-analytics';
import type { ChainPayload, IndicesPayload, IvHistoryEntry } from './types';
import { ForecastTrustCard } from '@/components/ui/ForecastTrustCard';

const f4 = (v: number | null, d = 4) => (v === null || !isFinite(v) ? '—' : v.toFixed(d));

export function T4ForecastTab({
  chain,
  indices,
  bars,
  ticker,
  ivHistory,
}: {
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  bars: PriceBar[];
  ticker: string;
  ivHistory: IvHistoryEntry[];
}) {
  const [horizon, setHorizon] = useState<1 | 5 | 22>(22);

  const closes = useMemo(() => (bars ?? []).map((b) => b.close), [bars]);
  const dates = useMemo(() => (bars ?? []).slice(1).map((b) => b.date), [bars]);
  const returns = useMemo(() => calculateReturns(closes, 'log'), [closes]);

  // ---- IV per date (annualized vol, decimal) -------------------------------
  const ivByDate = useMemo(() => {
    const vix = indices?.series?.VIX?.bars ?? [];
    const vxn = indices?.series?.VXN?.bars ?? [];
    const proxy = ticker === 'QQQ' ? vxn : vix;
    const isProxyEligible = ['SPY', 'QQQ', 'IWM', 'DIA'].includes(ticker) || chain?.substitution != null;
    if (isProxyEligible && proxy.length > 40) {
      return { byDate: new Map(proxy.map((b) => [b.date, b.close / 100])), label: `^VIX proxy (market)`, accrued: 0, available: true };
    }
    // Accrued single-name history only — small until 60 sessions accumulate
    const map = new Map<string, number>();
    for (const h of ivHistory) {
      if (h.iv30 && h.iv30 > 0) map.set(h.date, h.iv30);
    }
    return { byDate: map, label: `accrued ${ticker} IV`, accrued: map.size, available: map.size >= 40 };
  }, [indices, ticker, chain, ivHistory]);

  const ivDailyVarAligned = useMemo(
    () => dates.map((d) => {
      const v = ivByDate.byDate.get(d);
      return v !== undefined && isFinite(v) ? (v / Math.sqrt(252)) ** 2 : NaN;
    }),
    [dates, ivByDate]
  );

  // ---- Forecast models ------------------------------------------------------
  const garchFit = useMemo(() => (returns.length > 60 ? fitGarch11(returns) : null), [returns]);
  const ewma = useMemo(() => (returns.length > 30 ? computeEwmaVolatility(returns, 0.94) : null), [returns]);

  // Daily RV proxy: r²  (index aligned with returns)
  const rvDaily = useMemo(() => returns.map((r) => r * r), [returns]);

  // HAR-RV fitted on squared daily returns; in-sample 1-step-ahead forecasts
  const harForecasts = useMemo(() => {
    const n = returns.length;
    if (n < 30) return { preds: new Array(n).fill(NaN) as number[], coef: null as null | { b0: number; bd: number; bw: number; bm: number } };
    const rv = rvDaily;
    const X: number[][] = [];
    const Y: number[] = [];
    for (let t = 4; t < n - 1; t++) {
      const d = rv[t];
      const w = (rv[t] + rv[t - 1] + rv[t - 2] + rv[t - 3] + rv[t - 4]) / 5;
      const mLen = Math.min(22, t + 1);
      let m = 0;
      for (let j = 0; j < mLen; j++) m += rv[t - j];
      m /= mLen;
      X.push([1, d, w, m]);
      Y.push(rv[t + 1]);
    }
    if (X.length < 10) return { preds: new Array(n).fill(NaN) as number[], coef: null };
    // normal equations solve (4x4)
    const k = 4;
    const A: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    const yv = new Array(k).fill(0);
    for (let i = 0; i < X.length; i++) {
      for (let a = 0; a < k; a++) {
        yv[a] += X[i][a] * Y[i];
        for (let b = 0; b < k; b++) A[a][b] += X[i][a] * X[i][b];
      }
    }
    // Gaussian elimination
    const M = A.map((row, i) => [...row, yv[i]]);
    for (let col = 0; col < k; col++) {
      let piv = col;
      for (let row = col + 1; row < k; row++) if (Math.abs(M[row][col]) > Math.abs(M[piv][col])) piv = row;
      [M[col], M[piv]] = [M[piv], M[col]];
      const diag = M[col][col];
      if (Math.abs(diag) < 1e-14) continue;
      for (let row = 0; row < k; row++) {
        if (row === col) continue;
        const f = M[row][col] / diag;
        for (let j = col; j <= k; j++) M[row][j] -= f * M[col][j];
      }
    }
    const beta = new Array(k).fill(0);
    for (let i = 0; i < k; i++) beta[i] = Math.abs(M[i][i]) > 1e-14 ? M[i][k] / M[i][i] : 0;

    const preds = new Array(n).fill(NaN) as number[];
    for (let t = 4; t < n - 1; t++) {
      const d = rv[t];
      const w = (rv[t] + rv[t - 1] + rv[t - 2] + rv[t - 3] + rv[t - 4]) / 5;
      const mLen = Math.min(22, t + 1);
      let m = 0;
      for (let j = 0; j < mLen; j++) m += rv[t - j];
      m /= mLen;
      preds[t + 1] = Math.max(1e-8, beta[0] + beta[1] * d + beta[2] * w + beta[3] * m);
    }
    return { preds, coef: { b0: beta[0], bd: beta[1], bw: beta[2], bm: beta[3] } };
  }, [returns, rvDaily]);

  // ---- Horse race across horizons -------------------------------------------
  const race = useMemo(() => {
    const n = returns.length;
    if (n < 80 || !garchFit || !ewma) return null;
    const garchVar = garchFit.conditionalVariance;
    const ewmaVar = ewma.ewmaVariance;

    const ivUsable = ivByDate.available;
    const models: Record<string, { name: string; sub: string; fv: (i: number, h: number) => number | null }> = {
      ...(ivUsable
        ? {
            iv: {
              name: 'Implied Vol',
              sub: ivByDate.label,
              fv: (i: number, h: number) => {
                const v = ivByDate.byDate.get(dates[i]);
                if (v === undefined) return null;
                // flat h-horizon variance from today's IV (constant-maturity approximation)
                return (v / Math.sqrt(252)) ** 2 * h;
              },
            },
          }
        : {}),
      garch: {
        name: 'GARCH(1,1)',
        sub: 'Bollerslev MLE',
        fv: (i, h) => {
          let s = 0;
          for (let j = i; j < Math.min(i + h, garchVar.length); j++) s += garchVar[j];
          return s;
        },
      },
      ewma: {
        name: 'EWMA λ=0.94',
        sub: 'RiskMetrics',
        fv: (i, h) => {
          let s = 0;
          for (let j = i; j < Math.min(i + h, ewmaVar.length); j++) s += ewmaVar[j];
          return s;
        },
      },
      har: {
        name: 'HAR-RV',
        sub: 'Corsi daily/weekly/monthly',
        fv: (i, h) => {
          let s = 0;
          for (let j = i; j < Math.min(i + h, harForecasts.preds.length); j++) {
            const p = harForecasts.preds[j];
            if (!isFinite(p)) return null;
            s += p;
          }
          return s;
        },
      },
    };

    const results: Record<string, { qlike: number[]; mse: number[]; dmVsIv?: number; dmVsGarch?: number }> = {};
    for (const key of Object.keys(models)) {
      results[key] = { qlike: new Array(n).fill(NaN), mse: new Array(n).fill(NaN) };
    }

    for (let i = 0; i < n; i++) {
      // realized forward variance over h days from t=i+1 (annualized factor cancels in QLIKE ratios if consistent — keep total)
      const rvFwd = forwardRealizedVariance(returns, i, horizon);
      if (rvFwd === null || rvFwd <= 0) continue;
      const rvTotal = rvFwd * (horizon / 252); // de-annualize to total variance
      for (const key of Object.keys(models)) {
        const fv = models[key].fv(i, horizon);
        if (fv === null || fv <= 0 || !isFinite(fv)) continue;
        results[key].qlike[i] = qlike(rvTotal, fv);
        results[key].mse[i] = mseLogVariance(rvTotal, fv);
      }
    }

    // Encompassing: RV ~ a + b1*GARCH + b2*IV (only where IV exists)
    const gTot: number[] = [];
    const ivTot: number[] = [];
    const rvT: number[] = [];
    if (ivUsable) {
      for (let i = 0; i < n; i++) {
        const rvFwd = forwardRealizedVariance(returns, i, horizon);
        const fvG = models.garch.fv(i, horizon);
        const fvI = models.iv.fv(i, horizon);
        if (rvFwd !== null && fvG !== null && fvI !== null) {
          gTot.push(fvG);
          ivTot.push(fvI);
          rvT.push(rvFwd * (horizon / 252));
        }
      }
    }
    const weights = ivUsable ? encompassingWeights(gTot, ivTot, rvT) : null;

    // Blend losses
    const blendLoss: number[] = [];
    if (weights && ivUsable) {
      for (let i = 0; i < n; i++) {
        const rvFwd = forwardRealizedVariance(returns, i, horizon);
        const fvG = models.garch.fv(i, horizon);
        const fvI = models.iv.fv(i, horizon);
        if (rvFwd === null || fvG === null || fvI === null) { blendLoss.push(NaN); continue; }
        const wI = weights.wIv;
        const blend = wI * fvI + (1 - wI) * fvG;
        blendLoss.push(qlike(rvFwd * (horizon / 252), blend));
      }
    }

    // DM tests on the finite samples
    const dm = (a: (number | null)[], b: (number | null)[]): number | null => {
      const aa: number[] = [], bb: number[] = [];
      for (let i = 0; i < a.length; i++) {
        const x = a[i], y = b[i];
        if (x !== null && y !== null && isFinite(x) && isFinite(y)) { aa.push(x); bb.push(y); }
      }
      const r = dieboldMariano(aa, bb);
      return r ? r.dm : null;
    };
    const dmIvGarch = ivUsable ? dm(results.iv.qlike, results.garch.qlike) : null;
    const dmBlendGarch = blendLoss.length > 0 ? dm(blendLoss, results.garch.qlike) : null;

    const mean = (arr: number[]) => {
      const v = arr.filter((x) => isFinite(x));
      return v.length > 0 ? v.reduce((s, x) => s + x, 0) / v.length : NaN;
    };

    return {
      models: Object.entries(models).map(([key, m]) => ({
        key,
        name: m.name,
        sub: m.sub,
        qlike: mean(results[key].qlike),
        mse: mean(results[key].mse),
      })) as Array<{ key: string; name: string; sub: string; qlike: number; mse: number }>,
      blend: blendLoss.length > 0 ? mean(blendLoss) : null,
      weights,
      dmIvGarch,
      dmBlendGarch,
      ivUsable,
      n: n,
      ivCount: (() => { let c = 0; for (const v of ivDailyVarAligned) if (isFinite(v)) c++; return c; })(),
    };
  }, [returns, garchFit, ewma, harForecasts, horizon, ivByDate, dates, ivDailyVarAligned]);

  // ---- Mincer–Zarnowitz ------------------------------------------------------
  const mz = useMemo(() => {
    const n = returns.length;
    const iv: number[] = [];
    const rv: number[] = [];
    for (let i = 0; i < n; i++) {
      const rvFwd = forwardRealizedVariance(returns, i, horizon);
      const v = ivByDate.byDate.get(dates[i]);
      if (rvFwd !== null && v !== undefined && isFinite(v)) {
        iv.push(v * v);           // annualized IV² variance
        rv.push(rvFwd);           // annualized forward RV² variance
      }
    }
    return mincerZarnowitz(iv, rv, horizon);
  }, [returns, ivByDate, dates, horizon]);

  // ---- GARCH-X ----------------------------------------------------------------
  const garchX = useMemo(
    () => (returns.length > 60 ? fitGarchX(returns, ivDailyVarAligned) : null),
    [returns, ivDailyVarAligned]
  );

  // ---- Plain-English VRP measurement -------------------------------------------
  const vrpMeasured = mz && isFinite(mz.slope) ? 1 - mz.slope : null;

  const hOptions: (1 | 5 | 22)[] = [1, 5, 22];

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="panel flex flex-wrap items-center justify-between gap-3 p-4">
        <div className="flex items-center gap-3">
          <Target className="h-4 w-4 text-gold-600 dark:text-gold-400" />
          <div>
            <div className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Forecast validation laboratory</div>
            <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              h = {horizon}d forward · QLIKE scoring · IV leg: {ivByDate.label}{!ivByDate.available && ` (need ≥ 40 aligned sessions, have ${race?.ivCount ?? 0})`}
            </div>
          </div>
        </div>
        <div className="inline-flex items-center gap-1 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-1">
          {hOptions.map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`cursor-pointer rounded-md px-3 py-1.5 font-mono text-xs font-semibold transition-colors ${
                horizon === h ? 'bg-gold-500 text-zinc-950' : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
              }`}
            >
              h={h}d
            </button>
          ))}
        </div>
      </div>

      {/* Horse race table */}
      <div className="panel p-5">
        <div className="mb-3 border-b border-[var(--hairline)] pb-3">
          <p className="kicker mb-1 flex items-center gap-2"><FlaskConical className="h-3.5 w-3.5" /> Forecast Horse Race</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            Who forecasts variance best? QLIKE (log(σ²)+RV/σ²) and MSE on log-variance
          </h3>
        </div>
        {!race ? (
          <div className="text-xs text-zinc-400">Insufficient history for the horse race (need ≥ 80 returns).</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <th className="py-1.5 pr-3">Model</th>
                  <th className="py-1.5 pr-3 text-right">QLIKE ↓</th>
                  <th className="py-1.5 pr-3 text-right">MSE log-RV ↓</th>
                  <th className="py-1.5 text-right">DM vs IV</th>
                </tr>
              </thead>
              <tbody>
                {[...race.models].sort((a, b) => (isFinite(a.qlike) ? a.qlike : Infinity) - (isFinite(b.qlike) ? b.qlike : Infinity)).map((m, idx) => (
                  <tr key={m.key} className="border-t border-[var(--hairline)]">
                    <td className="py-1.5 pr-3">
                      <span className={`font-semibold ${idx === 0 ? 'text-gold-700 dark:text-gold-300' : 'text-zinc-800 dark:text-zinc-100'}`}>
                        {m.name}
                      </span>
                      <span className="block font-mono text-[8.5px] text-zinc-400">{m.sub}</span>
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-zinc-800 dark:text-zinc-100">
                      {f4(m.qlike, 4)}
                      {idx === 0 && isFinite(m.qlike) && <span className="ml-1.5 rounded bg-gold-500/20 px-1 py-0.5 text-[8px] font-bold text-gold-700 dark:text-gold-300">BEST</span>}
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-zinc-800 dark:text-zinc-100">{f4(m.mse, 4)}</td>
                    <td className="py-1.5 text-right font-mono tabular-nums text-zinc-500">
                      {m.key === 'iv' ? '—' : m.key === 'garch' && race.dmIvGarch !== null ? f4(race.dmIvGarch, 2) : '—'}
                    </td>
                  </tr>
                ))}
                {!race.ivUsable && (
                  <tr className="border-t border-[var(--hairline)] opacity-60">
                    <td className="py-1.5 pr-3">
                      <span className="font-semibold text-zinc-500">Implied Vol</span>
                      <span className="block font-mono text-[8.5px] text-zinc-400">{ivByDate.label}</span>
                    </td>
                    <td colSpan={3} className="py-1.5 text-right font-mono text-[10px] uppercase tracking-wider text-gold-700 dark:text-gold-300">
                      accruing — {race.ivCount}/40 aligned sessions
                    </td>
                  </tr>
                )}
                {race.blend !== null && (
                  <tr className="border-t-2 border-gold-500/30 bg-gold-500/5">
                    <td className="py-1.5 pr-3">
                      <span className="font-semibold text-gold-800 dark:text-gold-200">Blend (encompass)</span>
                      <span className="block font-mono text-[8.5px] text-zinc-400">
                        w(IV) = {race.weights ? (race.weights.wIv * 100).toFixed(0) : '—'}% · w(GARCH) = {race.weights ? (race.weights.wGarch * 100).toFixed(0) : '—'}%
                      </span>
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono tabular-nums font-bold text-gold-800 dark:text-gold-200">
                      {f4(race.blend, 4)}
                      <span className="ml-1.5 rounded bg-gold-500/30 px-1 py-0.5 text-[8px] font-bold text-gold-900 dark:text-gold-100">DELIVERABLE</span>
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono tabular-nums text-zinc-800 dark:text-zinc-100">—</td>
                    <td className="py-1.5 text-right font-mono tabular-nums text-zinc-500">{f4(race.dmBlendGarch, 2)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
          In-sample-fit evaluation over aligned windows; DM = mean(QLIKE_A − QLIKE_B)/√(LRV/T), Newey–West LRV. QLIKE penalizes under-prediction asymmetrically; MSE on log-RV controls tail inflation.
        </p>
      </div>

      <div className="panel p-5">
        <ForecastTrustCard returns={returns} compact />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Mincer–Zarnowitz */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1 flex items-center gap-2"><Scale className="h-3.5 w-3.5" /> Mincer–Zarnowitz</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              RV<sub>t+{horizon}d</sub> = a + b·IV<sub>t</sub> + ε — is IV informative and unbiased?
            </h3>
          </div>
          {mz ? (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Intercept a</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{f4(mz.intercept, 4)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">se {f4(mz.interceptSe, 4)}</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Slope b</div>
                  <div className="mt-1 font-mono text-lg font-bold text-gold-700 dark:text-gold-300">{f4(mz.slope, 3)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">se {f4(mz.slopeSe, 3)}</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">R²</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{mz.r2.toFixed(2)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">n = {mz.n} pairs</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">Joint F (a=0,b=1)</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{f4(mz.jointF, 2)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">p = {mz.jointP < 0.001 ? '<0.001' : mz.jointP.toFixed(3)}</div>
                </div>
              </div>
              <div className="mt-3 rounded-lg border border-gold-500/25 bg-gold-500/5 px-3 py-2 text-[11px] leading-relaxed text-gold-900 dark:text-gold-200">
                {mz.slope < 1
                  ? `b < 1 — IV is informative but biased HIGH by ~${vrpMeasured !== null ? (vrpMeasured * 100).toFixed(1) : '—'}% (the VRP, measured rather than asserted). The joint test ${mz.jointP < 0.05 ? 'rejects' : 'does not reject'} (a,b) = (0,1).`
                  : 'b ≥ 1 — IV is unbiased or conservative this sample; check horizon and sample alignment.'}
              </div>
            </>
          ) : (
            <div className="text-xs text-zinc-400">Not enough aligned IV/RV pairs ({mz === null ? 'need ≥ 20' : ''}). Single-name IV accrues daily.</div>
          )}
        </div>

        {/* GARCH-X */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2"><Sigma className="h-3.5 w-3.5" /> GARCH-X</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                σ²<sub>t</sub> = ω + αε²<sub>t-1</sub> + βσ²<sub>t-1</sub> + γ·IV²<sub>t-1</sub>
              </h3>
            </div>
            <span className="font-mono text-[8.5px] uppercase tracking-wider text-zinc-400">exogenous IV variance</span>
          </div>
          {garchX ? (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">γ (IV coef)</div>
                  <div className="mt-1 font-mono text-lg font-bold text-gold-700 dark:text-gold-300">{garchX.gamma.toFixed(4)}</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">α + β</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{(garchX.alpha + garchX.beta).toFixed(3)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">vs plain {garchFit ? (garchFit.summary.alpha + garchFit.summary.beta).toFixed(3) : '—'}</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">α</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{garchX.alpha.toFixed(3)}</div>
                </div>
                <div className="panel-inset p-3">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">β</div>
                  <div className="mt-1 font-mono text-lg font-bold text-zinc-900 dark:text-zinc-50">{garchX.beta.toFixed(3)}</div>
                </div>
              </div>
              <p className="mt-3 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                γ &gt; 0 with α+β falling versus plain GARCH means the market-priced variance absorbs part of the apparent
                clustering — the return process is less surprising once you listen to the options market.{' '}
                {garchX.converged ? '' : 'Not fully converged on this window.'}
              </p>
            </>
          ) : (
            <div className="text-xs text-zinc-400">GARCH-X needs ≥ 60 returns with an aligned IV series.</div>
          )}
        </div>
      </div>

      {/* Honest footnote */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Evaluation is in-sample-fit on aligned windows (GARCH conditional variance, HAR fitted on this sample). IV leg availability:{' '}
          {ivByDate.available
            ? `${ivByDate.label} — live.`
            : `${ivByDate.label} — accruing (${ivByDate.accrued}/60 required for single-name regression). Showing market proxy results instead.`}{' '}
          Single-name RV is the ticker&apos;s own; IV proxy (^VIX) measures SPX-risk pricing — interpret the IV leg for non-index names with care.
        </span>
      </div>
    </div>
  );
}
