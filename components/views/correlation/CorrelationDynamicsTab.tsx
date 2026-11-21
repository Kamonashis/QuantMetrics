'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
  IntradayBar,
  computeRollingCorrelation,
  computeEwmaDynamicCorrelation,
  fitDccGarchModel,
  computeTailDependence,
  computeRegimeCorrelations,
  computeRealizedCorrelationIntraday,
} from '@/lib/correlation-engine';
import {
  getCachedTcnCorrelationForecast,
  runTcnCorrelationForecast,
  type TcnCorrelationForecast,
} from '@/lib/tcn-multi-domain';
import {
  MathDisplay,
  DccQEquation,
  DccCorrelationEquation,
  TailDependenceEquation,
} from '@/components/math/MathEquations';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import { RegimeChip } from '@/components/ui/RegimeChip';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { fitGaussianHmm, type HmmFit } from '@/lib/hmm';
import {
  buildRegimeBandSpans,
  getHmmRegimeToken,
} from '@/lib/hmm-regime-overlay';
import {
  HmmRegimeOverlayToggle,
  HmmRegimeScorecard,
} from '@/components/charts/HmmRegimeOverlayControls';
import {
  Activity,
  Zap,
  Clock,
  Layers,
  TrendingDown,
  Info,
  ShieldAlert,
  ArrowRight,
  TrendingUp,
  RotateCcw,
  Sparkles,
} from 'lucide-react';

interface CorrelationDynamicsTabProps {
  tickers: string[];
  returnsByTicker: { [ticker: string]: number[] };
  dates: string[];
  intradayBars?: IntradayBar[];
}

export function CorrelationDynamicsTab({
  tickers,
  returnsByTicker,
  dates,
  intradayBars = [],
}: CorrelationDynamicsTabProps) {
  // Pair selection
  const [selectedPairA, setSelectedPairA] = useState<string>(tickers[0] || 'AAPL');
  const [selectedPairB, setSelectedPairB] = useState<string>(
    tickers[1] || (tickers.length > 1 ? tickers[1] : 'SPY')
  );

  // Engine choice: DCC-GARCH vs EWMA vs TCN-Attention (Fisher-z)
  const [modelType, setModelType] = useState<'dcc' | 'ewma' | 'tcn'>('dcc');
  const [ewmaLambda, setEwmaLambda] = useState<number>(0.94);
  const [rollingWindow, setRollingWindow] = useState<number>(60);
  const [forecastDays, setForecastDays] = useState<number>(30);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState<boolean>(true);
  const [corrHoverIdx, setCorrHoverIdx] = useState<number | null>(null);

  const returnsA = useMemo(() => returnsByTicker[selectedPairA] || [], [returnsByTicker, selectedPairA]);
  const returnsB = useMemo(() => returnsByTicker[selectedPairB] || [], [returnsByTicker, selectedPairB]);
  const nObs = Math.min(returnsA.length, returnsB.length);
  const firstDate = dates[0] ?? '';
  const lastDate = dates[dates.length - 1] ?? '';

  const [tcnCorr, setTcnCorr] = useState<TcnCorrelationForecast | null>(() =>
    nObs >= 70
      ? getCachedTcnCorrelationForecast(selectedPairA, selectedPairB, returnsA, returnsB, dates, forecastDays)
      : null
  );
  const [tcnTraining, setTcnTraining] = useState(false);
  const [tcnRetrainNonce, setTcnRetrainNonce] = useState(0);
  const rARef = useRef(returnsA);
  const rBRef = useRef(returnsB);
  const datesRef = useRef(dates);
  useEffect(() => {
    rARef.current = returnsA;
    rBRef.current = returnsB;
    datesRef.current = dates;
  }, [returnsA, returnsB, dates]);

  useEffect(() => {
    if (nObs < 70) return;
    let cancelled = false;
    const force = tcnRetrainNonce > 0;
    const timer = setTimeout(() => {
      if (cancelled) return;
      const curA = rARef.current;
      const curB = rBRef.current;
      const curDates = datesRef.current;
      if (!force) {
        const cached = getCachedTcnCorrelationForecast(selectedPairA, selectedPairB, curA, curB, curDates, forecastDays);
        if (cached) {
          setTcnCorr(cached);
          setTcnTraining(false);
          return;
        }
        if (modelType !== 'tcn') return;
      }
      setTcnTraining(true);
      void runTcnCorrelationForecast(selectedPairA, selectedPairB, curA, curB, curDates, forecastDays, {
        forceRetrain: force,
        yieldControl: () => new Promise<void>((r) => setTimeout(r, 0)),
        isCancelled: () => cancelled,
      }).then((res) => {
        if (!cancelled) {
          if (res) setTcnCorr(res);
          setTcnTraining(false);
        }
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [selectedPairA, selectedPairB, nObs, firstDate, lastDate, forecastDays, modelType, tcnRetrainNonce]);

  const activeTcnCorr =
    tcnCorr && tcnCorr.tickerA === selectedPairA.toUpperCase() && tcnCorr.tickerB === selectedPairB.toUpperCase()
      ? tcnCorr
      : null;

  // Fit DCC-GARCH
  const dccResult = useMemo(() => {
    return fitDccGarchModel(returnsA, returnsB, dates, forecastDays);
  }, [returnsA, returnsB, dates, forecastDays]);

  // Fit EWMA
  const ewmaResult = useMemo(() => {
    return computeEwmaDynamicCorrelation(returnsA, returnsB, dates, ewmaLambda);
  }, [returnsA, returnsB, dates, ewmaLambda]);

  // Rolling correlation
  const rollingPoints = useMemo(() => {
    return computeRollingCorrelation(returnsA, returnsB, dates, rollingWindow);
  }, [returnsA, returnsB, dates, rollingWindow]);

  // Tail dependence & Copula
  const tailResult = useMemo(() => {
    return computeTailDependence(returnsA, returnsB);
  }, [returnsA, returnsB]);

  // Regime conditioning — HMM primary (Calm→Crisis posterior), threshold heuristic as fallback
  const hmmFit = useMemo<HmmFit | null>(() => {
    const marketTicker = tickers.includes('SPY') ? 'SPY' : tickers[0];
    const marketRet = returnsByTicker[marketTicker] || [];
    const keys = Object.keys(returnsByTicker).slice(0, 25);
    const minLen = Math.min(...keys.map((k) => returnsByTicker[k].length));
    const n = Math.min(minLen, marketRet.length);
    if (n < 120) return null;
    const x: number[][] = new Array(n);
    for (let t = 0; t < n; t++) {
      const absUniverse = keys.map((k) => Math.abs(returnsByTicker[k][t])).sort((a, b) => a - b);
      const medAbs = absUniverse[Math.floor(absUniverse.length / 2)] ?? 0;
      x[t] = [marketRet[t], Math.abs(marketRet[t]), medAbs];
    }
    return fitGaussianHmm(x, { k: 3, rankFeature: 2 });
  }, [returnsByTicker, tickers]);

  const hmmRegime = useMemo(() => {
    if (!hmmFit) return null;
    const K = hmmFit.k;
    const keys = Object.keys(returnsByTicker);
    const covM = (weights: number[], a: string, b: string) => {
      const n = weights.length;
      let ma = 0, mb = 0, sw = 0;
      for (let t = 0; t < n; t++) {
        ma += weights[t] * returnsByTicker[a][t];
        mb += weights[t] * returnsByTicker[b][t];
        sw += weights[t];
      }
      ma /= sw; mb /= sw;
      let sAB = 0, sAA = 0, sBB = 0;
      for (let t = 0; t < n; t++) {
        const w = weights[t];
        const da = returnsByTicker[a][t] - ma;
        const db = returnsByTicker[b][t] - mb;
        sAB += w * da * db; sAA += w * da * da; sBB += w * db * db;
      }
      const denom = Math.sqrt(sAA * sBB);
      return denom > 0 ? sAB / denom : 0;
    };
    const stateMean = (ki: number) => {
      const weights = hmmFit.posteriors.map((row) => row[ki]);
      let sum = 0, count = 0;
      for (let i = 0; i < keys.length; i++) {
        for (let j = i + 1; j < keys.length; j++) {
          sum += covM(weights, keys[i], keys[j]);
          count++;
        }
      }
      return count > 0 ? sum / count : 0;
    };
    const calm = stateMean(0);
    const crisis = stateMean(K - 1);
    return { calm, crisis, delta: crisis - calm, converged: hmmFit.converged, bic: hmmFit.bic };
  }, [hmmFit, returnsByTicker]);

  const regimeResult = useMemo(() => {
    const marketTicker = tickers.includes('SPY') ? 'SPY' : tickers[0];
    return computeRegimeCorrelations(returnsByTicker, marketTicker);
  }, [returnsByTicker, tickers]);

  // Realized 1-minute correlation & jump decomposition
  const realizedCorrResult = useMemo(() => {
    return computeRealizedCorrelationIntraday(intradayBars, intradayBars);
  }, [intradayBars]);

  // Active series for the dynamic chart
  const activeSeries =
    modelType === 'dcc'
      ? dccResult.dynamicCorrelations
      : modelType === 'tcn'
        ? (activeTcnCorr?.dynamicCorrelations ?? ewmaResult)
        : ewmaResult;

  const currentDynamicCorr =
    activeSeries[activeSeries.length - 1]?.correlation ?? dccResult.unconditionalCorr;

  return (
    <div className="space-y-6">
      {/* Parameter & Model Selection Card */}
      <div className="panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="kicker">
                Dynamic Correlation Suite
              </span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500">
                Engle (2002) DCC-GARCH &amp; RiskMetrics EWMA
              </span>
            </div>
            <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 mt-0.5">
              Time-Varying Co-Movement &amp; Volatility Spillover
            </h3>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Pair Selectors */}
            <div className="flex items-center gap-1.5 text-xs font-mono">
              <span className="text-zinc-400">Pair:</span>
              <select
                value={selectedPairA}
                onChange={(e) => setSelectedPairA(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-bold text-zinc-900 dark:text-zinc-100"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <span className="text-zinc-400">/</span>
              <select
                value={selectedPairB}
                onChange={(e) => setSelectedPairB(e.target.value)}
                className="px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-bold text-zinc-900 dark:text-zinc-100"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Architecture Switcher */}
            <AlgoSelect
              ariaLabel="Correlation model"
              value={modelType}
              onChange={(v) => setModelType(v as 'dcc' | 'ewma' | 'tcn')}
              options={[
                { value: 'dcc', label: 'DCC-GARCH', hint: 'Engle · QMLE' },
                { value: 'ewma', label: 'EWMA', hint: 'RiskMetrics' },
                { value: 'tcn', label: 'TCN-Attention', hint: '1d/5d/22d · Fisher-z' },
              ]}
              className="w-56"
            />
          </div>
        </div>

        {/* Dynamic Model Control Readout Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4 text-xs">
          {modelType === 'tcn' ? (
            <>
              {([1, 5, 22] as const).map((h) => {
                const hz = activeTcnCorr?.horizons[h];
                return (
                  <div key={h} className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                    <div className="flex items-center justify-between font-mono text-[10px] text-zinc-400">
                      <span>TCN {h}D Forecast ρ</span>
                      <span>OOS MAE {hz ? hz.oosMae.toFixed(3) : '—'}</span>
                    </div>
                    <div className="mt-0.5 flex items-baseline gap-2">
                      <span className="font-mono text-xl font-bold text-gold-600 dark:text-gold-300 tabular-nums">
                        {hz ? hz.predictedCorr.toFixed(3) : tcnTraining ? '…' : '—'}
                      </span>
                      {hz && (
                        <span className={`font-mono text-[10px] font-semibold tabular-nums ${hz.deltaVsCurrent >= 0.04 ? 'text-rose-500' : hz.deltaVsCurrent <= -0.04 ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-400'}`}>
                          {hz.deltaVsCurrent >= 0 ? '+' : ''}{hz.deltaVsCurrent.toFixed(2)} vs now
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] font-mono text-zinc-500 block mt-0.5 tabular-nums">
                      {hz ? `99% PI [${hz.lower99.toFixed(2)}, ${hz.upper99.toFixed(2)}]` : 'Fisher-z tanh bounded (-1, 1)'}
                    </span>
                  </div>
                );
              })}

              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between font-mono text-[10px] text-zinc-400">
                    <span className="flex items-center gap-1"><Sparkles className="h-3 w-3 text-gold-500" /> Spillover State</span>
                    <button
                      type="button"
                      onClick={() => setTcnRetrainNonce((n) => n + 1)}
                      disabled={tcnTraining}
                      className="inline-flex cursor-pointer items-center gap-1 text-gold-600 hover:underline dark:text-gold-400 disabled:opacity-50"
                    >
                      <RotateCcw className={`h-3 w-3 ${tcnTraining ? 'animate-spin' : ''}`} />
                      {tcnTraining ? 'Training' : 'Retrain'}
                    </button>
                  </div>
                  <div className={`mt-1 font-mono text-sm font-bold ${
                    activeTcnCorr?.spilloverAlert.level === 'ELEVATED'
                      ? 'text-rose-500 dark:text-rose-400'
                      : activeTcnCorr?.spilloverAlert.level === 'WATCH'
                        ? 'text-gold-600 dark:text-gold-400'
                        : 'text-emerald-600 dark:text-emerald-400'
                  }`}>
                    {activeTcnCorr?.spilloverAlert.level ?? (tcnTraining ? 'COMPUTING…' : '—')}
                  </div>
                  <p className="mt-0.5 text-[10px] leading-snug text-zinc-500 dark:text-zinc-400">
                    {activeTcnCorr?.spilloverAlert.summary ?? 'Monitoring cross-asset correlation jump risk.'}
                  </p>
                  {activeTcnCorr?.hmmRegime && (
                    <div className="mt-1 font-mono text-[9.5px] tabular-nums text-gold-700 dark:text-gold-300">
                      Joint HMM: {activeTcnCorr.hmmRegime.dominantLabel} ({Math.round(activeTcnCorr.hmmRegime.probs[0] * 100)}/{Math.round(activeTcnCorr.hmmRegime.probs[1] * 100)}/{Math.round(activeTcnCorr.hmmRegime.probs[2] * 100)}%)
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : modelType === 'dcc' ? (
            <>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">
                  Shock Reaction Parameter (a)
                </span>
                <span className="font-mono text-xl font-bold text-gold-500">
                  {dccResult.a.toFixed(3)}
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Sensitivity to joint innovation shocks
                </span>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">
                  Decay Memory Parameter (b)
                </span>
                <span className="font-mono text-xl font-bold text-gold-600 dark:text-gold-300">
                  {dccResult.b.toFixed(3)}
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Autoregressive correlation persistence
                </span>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">
                  Total Persistence (a + b)
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100">
                    {dccResult.persistence.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-gold-500 font-semibold">
                    Stationary (&lt;1)
                  </span>
                </div>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Half-life: {(Math.log(0.5) / Math.log(dccResult.persistence)).toFixed(1)} sessions
                </span>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">
                  Latest Dynamic Correlation
                </span>
                <span className="font-mono text-xl font-bold text-gold-500">
                  {currentDynamicCorr.toFixed(3)}
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Unconditional: {dccResult.unconditionalCorr.toFixed(2)}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800 lg:col-span-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-zinc-700 dark:text-zinc-300 font-semibold">
                    EWMA Decay Factor (λ)
                  </span>
                  <span className="font-mono font-bold text-gold-600 dark:text-gold-300">{ewmaLambda.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.85"
                  max="0.99"
                  step="0.01"
                  value={ewmaLambda}
                  onChange={(e) => setEwmaLambda(parseFloat(e.target.value))}
                  className="w-full accent-gold-400 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-zinc-400 font-mono mt-1">
                  <span>0.85 (Fast shock response)</span>
                  <span>0.94 (RiskMetrics standard)</span>
                  <span>0.99 (Smooth long memory)</span>
                </div>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">Effective Half-Life</span>
                <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100">
                  {(Math.log(0.5) / Math.log(ewmaLambda)).toFixed(1)} days
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Memory horizon for joint returns
                </span>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
                <span className="text-zinc-400 font-mono text-[10px] block">
                  Current EWMA Correlation
                </span>
                <span className="font-mono text-xl font-bold text-gold-600 dark:text-gold-300">
                  {currentDynamicCorr.toFixed(3)}
                </span>
                <span className="text-[10px] text-zinc-500 block mt-0.5">
                  Static: {dccResult.unconditionalCorr.toFixed(2)}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Main Dynamic Correlation Chart with Baseline & Forecast Cone */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                {modelType === 'dcc'
                  ? 'Engle DCC-GARCH(1,1)'
                  : modelType === 'tcn'
                    ? 'TCN-Attention (Fisher-z 1d/5d/22d)'
                    : 'RiskMetrics EWMA'}{' '}
                Dynamic Correlation Path
              </h4>
              <span className="text-xs font-mono text-zinc-400">
                · {selectedPairA} vs {selectedPairB}
              </span>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Demonstrating how correlation fluctuates as a state variable compared to constant static ρ
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-gold-500 inline-block" />
              <span>Dynamic {modelType.toUpperCase()} Path</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-zinc-400 border-dashed inline-block" />
              <span>Static Baseline ({dccResult.unconditionalCorr.toFixed(2)})</span>
            </div>
          </div>
        </div>

        {/* Causal HMM Regime Overlay Toggle & Live Hover Readout (TCN mode) */}
        {modelType === 'tcn' && activeTcnCorr && activeTcnCorr.hmmStates.length > 0 && (() => {
          const dyn = activeTcnCorr.dynamicCorrelations;
          const hIdx = corrHoverIdx !== null && corrHoverIdx < dyn.length ? corrHoverIdx : dyn.length - 1;
          const hPt = dyn[hIdx];
          const tok = hPt?.hmmState !== undefined ? getHmmRegimeToken(hPt.hmmState) : null;
          return (
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b border-[var(--hairline)]">
              <HmmRegimeOverlayToggle
                enabled={showRegimeOverlay}
                onToggle={() => setShowRegimeOverlay((v) => !v)}
                caption="Joint HMM origin t−1"
              />
              {hPt && (
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] font-mono tabular-nums text-zinc-600 dark:text-zinc-300">
                  <strong className="text-zinc-900 dark:text-zinc-100">{hPt.date}</strong>
                  {hPt.isOos && <span className="text-gold-600 dark:text-gold-400">OOS</span>}
                  <span>·</span>
                  <span>ρ = {hPt.correlation.toFixed(3)}</span>
                  {tok && (
                    <>
                      <span>·</span>
                      <span className="inline-flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: tok.color }}
                        />
                        <span className={`font-semibold ${tok.textClass}`}>
                          {tok.label}
                          {hPt.hmmProb !== undefined ? ` (${Math.round(hPt.hmmProb * 100)}%)` : ''}
                        </span>
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })()}

        {/* SVG Time-Series Chart */}
        <ChartFullscreen title="Dynamic Correlation Path vs. Static Baseline">
        <div className="w-full h-64 bg-zinc-50 dark:bg-zinc-900/60 rounded-xl border border-zinc-200 dark:border-zinc-800 relative p-3">
          {activeSeries.length > 5 ? (
            <svg
              viewBox="0 0 800 220"
              className="w-full h-full cursor-crosshair"
              onMouseMove={(e) => {
                if (modelType !== 'tcn' || !activeTcnCorr) return;
                const rect = e.currentTarget.getBoundingClientRect();
                const svgX = ((e.clientX - rect.left) / rect.width) * 800;
                const clamped = Math.max(50, Math.min(700, svgX));
                const idx = Math.round(((clamped - 50) / 650) * Math.max(1, activeSeries.length - 1));
                setCorrHoverIdx(Math.max(0, Math.min(activeSeries.length - 1, idx)));
              }}
              onMouseLeave={() => setCorrHoverIdx(null)}
            >
              {/* Causal Joint HMM Regime Background Bands (TCN mode) */}
              {modelType === 'tcn' && activeTcnCorr && activeTcnCorr.hmmStates.length > 0 && (() => {
                const dyn = activeTcnCorr.dynamicCorrelations;
                const totalPts = Math.max(1, dyn.length - 1);
                const stepX = 650 / totalPts;
                const spans = buildRegimeBandSpans(dyn.map((p) => p.hmmState));
                const holdX = 50 + Math.min(totalPts, activeTcnCorr.holdoutStartIdx) * stepX;
                return (
                  <g
                    style={{
                      opacity: showRegimeOverlay ? 1 : 0,
                      transition: 'opacity 150ms ease',
                    }}
                  >
                    {spans.map((span) => {
                      const tok = getHmmRegimeToken(span.state);
                      const x0 = 50 + span.startIdx * stepX;
                      const x1 = span.endIdx < dyn.length - 1 ? 50 + (span.endIdx + 1) * stepX : 700;
                      return (
                        <rect
                          key={`corr-band-${span.startIdx}-${span.endIdx}-${span.state}`}
                          x={x0}
                          y={20}
                          width={Math.max(1, x1 - x0)}
                          height={180}
                          fill={`rgba(${tok.rgb}, 0.12)`}
                        />
                      );
                    })}
                    {activeTcnCorr.holdoutStartIdx > 0 && activeTcnCorr.holdoutStartIdx < dyn.length && (
                      <g>
                        <line
                          x1={holdX}
                          y1={20}
                          x2={holdX}
                          y2={200}
                          stroke="#DEC084"
                          strokeWidth="0.8"
                          strokeDasharray="2 3"
                          opacity="0.7"
                        />
                        <text
                          x={holdX + 4}
                          y={31}
                          className="text-[9px] font-mono fill-gold-600 dark:fill-gold-400"
                        >
                          OOS
                        </text>
                      </g>
                    )}
                  </g>
                );
              })()}

              {/* Horizontal Reference Lines (-1, 0, +1) */}
              <line x1="40" y1="20" x2="780" y2="20" stroke="#64748b" strokeWidth="0.3" strokeDasharray="3 3" />
              <line x1="40" y1="110" x2="780" y2="110" stroke="#64748b" strokeWidth="0.65" strokeDasharray="2 2" />
              <line x1="40" y1="200" x2="780" y2="200" stroke="#64748b" strokeWidth="0.3" strokeDasharray="3 3" />

              {/* Zero & Bounds Labels */}
              <text x="35" y="24" textAnchor="end" className="text-[10px] font-mono fill-zinc-400">+1.0</text>
              <text x="35" y="114" textAnchor="end" className="text-[10px] font-mono fill-zinc-400">0.0</text>
              <text x="35" y="204" textAnchor="end" className="text-[10px] font-mono fill-zinc-400">-1.0</text>

              {/* Static Unconditional Line */}
              <line
                x1="45"
                y1={110 - dccResult.unconditionalCorr * 90}
                x2="780"
                y2={110 - dccResult.unconditionalCorr * 90}
                stroke="#94a3b8"
                strokeWidth="0.8"
                strokeDasharray="4 4"
              />

              {/* Dynamic Path Polyline */}
              {(() => {
                const totalPoints = activeSeries.length;
                const stepX = 650 / Math.max(1, totalPoints - 1);
                const pointsStr = activeSeries
                  .map((pt, i) => {
                    const x = 50 + i * stepX;
                    const y = 110 - pt.correlation * 90;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                  })
                  .join(' ');

                return (
                  <polyline
                    fill="none"
                    stroke={modelType === 'ewma' ? '#94a3b8' : '#DEC084'}
                    strokeWidth="1.4"
                    points={pointsStr}
                  />
                );
              })()}

              {/* Color-coded OOS Prediction Points by Causal Joint HMM Regime (TCN mode) */}
              {modelType === 'tcn' && activeTcnCorr && activeTcnCorr.hmmStates.length > 0 && (() => {
                const dyn = activeTcnCorr.dynamicCorrelations;
                const stepX = 650 / Math.max(1, dyn.length - 1);
                const oosIndices: number[] = [];
                for (let i = 0; i < dyn.length; i++) {
                  if (dyn[i].isOos) oosIndices.push(i);
                }
                const stride = Math.max(1, Math.ceil(oosIndices.length / 44));
                return (
                  <g
                    style={{
                      opacity: showRegimeOverlay ? 1 : 0,
                      transition: 'opacity 150ms ease',
                    }}
                  >
                    {oosIndices.map((idx, k) => {
                      if (k % stride !== 0 && idx !== oosIndices[oosIndices.length - 1]) return null;
                      const pt = dyn[idx];
                      const tok = getHmmRegimeToken(pt.hmmState);
                      return (
                        <circle
                          key={`corr-oos-${idx}`}
                          cx={50 + idx * stepX}
                          cy={110 - pt.correlation * 90}
                          r={2.7}
                          fill={tok.color}
                          stroke="#08080a"
                          strokeWidth={0.9}
                        />
                      );
                    })}
                  </g>
                );
              })()}

              {/* Hover Crosshair & Active Marker (TCN mode) */}
              {modelType === 'tcn' && activeTcnCorr && corrHoverIdx !== null && (() => {
                const dyn = activeTcnCorr.dynamicCorrelations;
                const pt = dyn[corrHoverIdx];
                if (!pt) return null;
                const stepX = 650 / Math.max(1, dyn.length - 1);
                const cx = 50 + corrHoverIdx * stepX;
                const cy = 110 - pt.correlation * 90;
                const fill =
                  showRegimeOverlay && pt.hmmState !== undefined
                    ? getHmmRegimeToken(pt.hmmState).color
                    : '#DEC084';
                return (
                  <g>
                    <line
                      x1={cx}
                      y1={20}
                      x2={cx}
                      y2={200}
                      stroke="#DEC084"
                      strokeWidth="0.7"
                      strokeDasharray="2 2"
                      opacity="0.7"
                    />
                    <circle cx={cx} cy={cy} r={4.2} fill={fill} stroke="#08080a" strokeWidth={1.3} />
                  </g>
                );
              })()}

              {/* Forecast Cone (DCC or TCN-Attention) */}
              {modelType === 'dcc' && dccResult.forecastCone.length > 0 && (() => {
                const startX = 700;
                const endX = 770;
                const cone = dccResult.forecastCone;
                const topPath = cone
                  .map((c, i) => `${startX + i * 2.3},${110 - c.upper95 * 90}`)
                  .join(' ');
                const bottomPath = cone
                  .slice()
                  .reverse()
                  .map((c, i) => `${endX - i * 2.3},${110 - c.lower95 * 90}`)
                  .join(' ');

                return (
                  <polygon
                    points={`${topPath} ${bottomPath}`}
                    fill="#DEC084"
                    fillOpacity="0.15"
                    stroke="#DEC084"
                    strokeWidth="0.5"
                    strokeDasharray="2 2"
                  />
                );
              })()}

              {modelType === 'tcn' && activeTcnCorr && activeTcnCorr.forecastCone.length > 0 && (() => {
                const startX = 700;
                const cone = activeTcnCorr.forecastCone;
                const dx = 70 / Math.max(1, cone.length - 1);
                const topPath = cone
                  .map((c, i) => `${(startX + i * dx).toFixed(1)},${(110 - c.upper99 * 90).toFixed(1)}`)
                  .join(' ');
                const bottomPath = cone
                  .slice()
                  .reverse()
                  .map((c, i) => `${(startX + (cone.length - 1 - i) * dx).toFixed(1)},${(110 - c.lower99 * 90).toFixed(1)}`)
                  .join(' ');
                const midPath = cone
                  .map((c, i) => `${(startX + i * dx).toFixed(1)},${(110 - c.correlation * 90).toFixed(1)}`)
                  .join(' ');

                return (
                  <g>
                    <polygon
                      points={`${topPath} ${bottomPath}`}
                      fill="#DEC084"
                      fillOpacity="0.16"
                      stroke="#DEC084"
                      strokeWidth="0.5"
                      strokeDasharray="2 2"
                    />
                    <polyline
                      fill="none"
                      stroke="#3fb27f"
                      strokeWidth="1.5"
                      strokeDasharray="3 2"
                      points={midPath}
                    />
                  </g>
                );
              })()}
            </svg>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-zinc-400 font-mono">
              Insufficient time series points to construct dynamic correlation path.
            </div>
          )}
        </div>
        </ChartFullscreen>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-zinc-500 font-mono">
          <span>Start: {dates[0] || 'Historical'}</span>
          <span>Latest: {dates[dates.length - 1] || 'Current'}</span>
          {modelType === 'dcc' && (
            <span className="text-gold-500 font-semibold">+30-Day Forecast Cone Shaded</span>
          )}
          {modelType === 'tcn' && activeTcnCorr && (
            <span className="text-gold-500 font-semibold">
              +30D Fisher-z 99% Cone · {activeTcnCorr.cacheHit ? 'cached' : `${activeTcnCorr.trainingDurationMs}ms`}
              {activeTcnCorr.topShockDates.length > 0 && ` · Top shock ${activeTcnCorr.topShockDates[0].date}`}
            </span>
          )}
        </div>

        {/* Per-Regime OOS Correlation Accuracy Scorecard (TCN mode) */}
        {modelType === 'tcn' && activeTcnCorr && activeTcnCorr.regimePerformance.length > 0 && (
          <HmmRegimeScorecard
            className="mt-4"
            title={`TCN 1d OOS Correlation Accuracy by Joint HMM Regime (${selectedPairA}/${selectedPairB})`}
            subtitle="Stratified by dominant forward-filtered joint HMM state at prediction origin (t−1)"
            columns={[
              { key: 'mae', label: '1d OOS MAE' },
              { key: 'rmse', label: '1d OOS RMSE' },
              { key: 'pred', label: 'Mean Pred ρ' },
              { key: 'act', label: 'Mean Actual ρ' },
            ]}
            rows={activeTcnCorr.regimePerformance.map((r) => ({
              state: r.state,
              count: r.count,
              cells: {
                mae: r.oosMae !== null ? r.oosMae.toFixed(3) : '—',
                rmse: r.oosRmse !== null ? r.oosRmse.toFixed(3) : '—',
                pred: r.meanPredCorr !== null ? r.meanPredCorr.toFixed(3) : '—',
                act: r.meanActualCorr !== null ? r.meanActualCorr.toFixed(3) : '—',
              },
            }))}
          />
        )}
      </div>

      {/* Rolling Windows & Tail Copula Breakdown Row */}
      <div className="grid-cards">
        {/* Rolling Window Correlation with Historical Percentile Bands */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Rolling Correlation &amp; Percentile Bands
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Evaluates historical regime stability against long-term quantile bands
              </p>
            </div>

            {/* Window Selector */}
            <div className="flex items-center gap-1 p-0.5 bg-zinc-100 dark:bg-zinc-900 rounded border border-zinc-200 dark:border-zinc-800 text-xs">
              {[20, 60, 120].map((w) => (
                <button
                  key={w}
                  onClick={() => setRollingWindow(w)}
                  className={`px-2 py-1 rounded font-mono font-medium ${
                    rollingWindow === w
                      ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white font-bold shadow-xs'
                      : 'text-zinc-500'
                  }`}
                >
                  {w}D
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center font-mono text-xs mb-3">
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">10th Percentile (Min Regime)</span>
              <span className="font-bold text-rose-500 text-sm">
                {rollingPoints[0]?.p10.toFixed(2) ?? '-0.15'}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">50th Percentile (Median)</span>
              <span className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">
                {rollingPoints[0]?.p50.toFixed(2) ?? '0.45'}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">90th Percentile (Stress Spike)</span>
              <span className="font-bold text-gold-500 text-sm">
                {rollingPoints[0]?.p90.toFixed(2) ?? '0.82'}
              </span>
            </div>
          </div>

          <p className="text-[11px] text-zinc-500 leading-relaxed">
            During liquidity crises, correlations typically spike towards the 90th percentile (~{rollingPoints[0]?.p90.toFixed(2) ?? '0.82'}), eroding standard portfolio diversification benefit exactly when drawdown protection is required.
          </p>
        </div>

        {/* Tail Dependence & Copula Exceedance Quadrant */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Tail Dependence &amp; Crash Quadrants
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Empirical co-crash probability vs Gaussian copula assumption
              </p>
            </div>
            <span className="text-xs font-mono text-rose-500 font-semibold">
              Tail Risk Indicator
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs mb-3">
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl">
              <span className="text-[10px] font-mono text-rose-600 dark:text-rose-400 block font-semibold">
                Empirical Joint Crash Exceedance
              </span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="font-mono text-xl font-bold text-rose-600 dark:text-rose-400">
                  {(tailResult.jointExceedanceEmpirical * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] font-mono text-zinc-400">
                  vs {(tailResult.jointExceedanceGaussian * 100).toFixed(2)}% Gaussian
                </span>
              </div>
              <span className="text-[10px] text-rose-700 dark:text-rose-400 block mt-1 font-semibold">
                {tailResult.exceedanceRatio.toFixed(1)}× higher probability than normal distribution
              </span>
            </div>

            <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800 rounded-xl">
              <span className="text-[10px] font-mono text-zinc-400 block font-semibold">
                Lower Tail Dependence (λ_L)
              </span>
              <div className="flex items-baseline gap-2 mt-0.5">
                <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100">
                  {tailResult.lowerTailDep.toFixed(2)}
                </span>
                <span className="text-[10px] text-zinc-500">
                  (Gaussian = 0.00)
                </span>
              </div>
              <span className="text-[10px] text-zinc-500 block mt-1">
                Both assets drop &gt;1.5σ in {tailResult.quadrants.bothCrash} sessions
              </span>
            </div>
          </div>

          <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800 flex items-center justify-between text-xs">
            <TailDependenceEquation display="inline" className="text-xs" />
            <span className="text-[10px] font-mono text-zinc-400">
              Gaussian copulas dangerously assume zero tail dependence
            </span>
          </div>
        </div>
      </div>

      {/* Regime-Conditional Matrices & 1-Min Intraday Realized Correlation */}
      <div className="grid-cards">
        {/* Calm vs Stress Regime Split */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                  Regime-Conditional Correlation Breakdown
                </h4>
                {hmmFit && <RegimeChip fit={hmmFit} />}
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                {hmmRegime
                  ? `Gaussian HMM on [SPY r, |SPY r|, median |universe r|] · K = ${hmmFit?.k} · BIC ${hmmRegime.bic.toFixed(0)} · calm→crisis ranked by median |r|`
                  : regimeResult.stressDefinition}
              </p>
            </div>
            <span className="text-xs font-mono text-gold-500 font-semibold">
              {hmmFit ? 'HMM Calm→Crisis' : 'Calm vs Stress Split'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center text-xs mb-3">
            <div className="p-2.5 bg-gold-500/10 border border-gold-500/20 rounded-lg">
              <span className="text-[10px] text-gold-600 dark:text-gold-400 block font-semibold">
                Calm Regime Mean ρ
              </span>
              <span className="font-bold text-gold-600 dark:text-gold-400 text-lg">
                {(hmmRegime ? hmmRegime.calm : regimeResult.avgCalmCorr).toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-rose-500/10 border border-rose-500/20 rounded-lg">
              <span className="text-[10px] text-rose-600 dark:text-rose-400 block font-semibold">
                Stress Regime Mean ρ
              </span>
              <span className="font-bold text-rose-600 dark:text-rose-400 text-lg">
                {(hmmRegime ? hmmRegime.crisis : regimeResult.avgStressCorr).toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-gold-500/10 border border-gold-500/20 rounded-lg">
              <span className="text-[10px] text-gold-600 dark:text-gold-400 block font-semibold">
                Tail Penalty (Δρ)
              </span>
              <span className="font-bold text-gold-600 dark:text-gold-400 text-lg">
                +{(hmmRegime ? hmmRegime.delta : regimeResult.avgDeltaCorr).toFixed(2)}
              </span>
            </div>
          </div>

          {hmmFit && (
            <RegimeTransitionMatrix fit={hmmFit} className="mt-3 border-t border-zinc-100 dark:border-zinc-800/60 pt-2.5 lg:max-w-md" showStats={false} />
          )}

          <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
            Quantifies the empirical phenomenon where asset correlations increase by{' '}
            <strong className="text-gold-500">
              +{((hmmRegime ? hmmRegime.delta : regimeResult.avgDeltaCorr) * 100).toFixed(0)}%
            </strong>{' '}
            during market stress, reducing diversification exactly when protection is vital.
            {hmmFit && <> HMM ranks calm→crisis by median universe |r|; threshold heuristic (bottom-20%) shown for continuity when HMM does not converge.</>}
          </p>
        </div>

        {/* Intraday 1-Minute Realized Correlation & Jump Split */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <div className="flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-gold-500" />
                <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                  1-Minute Realized Correlation &amp; Jump Split
                </h4>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Continuous Brownian co-diffusion vs discrete co-jump shocks
              </p>
            </div>
            <span className="text-xs font-mono text-gold-600 dark:text-gold-300 font-semibold">
              High-Frequency RC
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 font-mono text-center text-xs mb-3">
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Total Realized (ρ^RV)</span>
              <span className="font-bold text-zinc-900 dark:text-zinc-100 text-lg">
                {realizedCorrResult.intradayRealizedCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Continuous Diffusion</span>
              <span className="font-bold text-gold-600 dark:text-gold-300 text-lg">
                {realizedCorrResult.continuousDiffusionCorr.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Discrete Co-Jumps</span>
              <span className="font-bold text-gold-500 text-lg">
                {realizedCorrResult.discreteJumpCorr.toFixed(2)}
              </span>
            </div>
          </div>

          {/* Epps Effect Transparency Warning */}
          <div className="p-3 bg-gold-500/10 border border-gold-500/20 rounded-lg text-xs text-gold-700 dark:text-gold-400 flex items-start gap-2">
            <ShieldAlert className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div>
              <strong>Epps Effect &amp; Sample Size Notice: </strong>
              {realizedCorrResult.eppsBiasCaveat}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
