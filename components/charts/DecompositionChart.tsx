'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { DecompositionResult, AdfTestResult } from '@/lib/quant-engine';
import {
  DEFAULT_TCN_CONFIG,
  MIN_BARS_FOR_TCN,
  runTcnDecompositionForecast,
  runTcnLookbackSensitivitySweep,
  type SensitivityStep,
  type TcnAttentionConfig,
  type TcnDecompositionForecast,
  type TcnSensitivityResult,
  type TcnTrainingProgress,
} from '@/lib/tcn-attention';
import {
  DecompositionAdditiveEquation,
  DecompositionMultiplicativeEquation,
  AdfEquation,
} from '@/components/math/MathEquations';
import { TcnForecastPanel } from '@/components/charts/TcnForecastPanel';

interface DecompositionChartProps {
  ticker?: string;
  decomposition: DecompositionResult;
  adfTest: AdfTestResult;
}

type ComponentFilter = 'all' | 'trend' | 'cyclical' | 'seasonal' | 'residual';

interface NextDayPointOverlay {
  predicted: number;
  lower99: number;
  upper99: number;
  dateLabel: string;
}

export function DecompositionChart({
  ticker = 'ASSET',
  decomposition,
  adfTest,
}: DecompositionChartProps) {
  const [activeComponent, setActiveComponent] = useState<ComponentFilter>('all');
  const [tcnConfig, setTcnConfig] = useState<TcnAttentionConfig>(() => ({
    ...DEFAULT_TCN_CONFIG,
  }));
  const [forecast, setForecast] = useState<TcnDecompositionForecast | null>(null);
  const [trainedSignature, setTrainedSignature] = useState<string | null>(null);
  const [sensitivityEnabled, setSensitivityEnabled] = useState<boolean>(false);
  const [sensitivityStep, setSensitivityStep] = useState<SensitivityStep>(4);
  const [sensitivityResult, setSensitivityResult] = useState<TcnSensitivityResult | null>(null);
  const [sensitivitySignature, setSensitivitySignature] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progress, setProgress] = useState<TcnTrainingProgress | null>(null);

  const runIdRef = useRef(0);
  const tcnConfigRef = useRef<TcnAttentionConfig>(tcnConfig);
  const decompositionRef = useRef<DecompositionResult>(decomposition);
  const tickerRef = useRef<string>(ticker);

  useEffect(() => {
    tcnConfigRef.current = tcnConfig;
  }, [tcnConfig]);

  useEffect(() => {
    decompositionRef.current = decomposition;
    tickerRef.current = ticker;
  }, [decomposition, ticker]);

  const {
    dates,
    observed,
    trend,
    cyclical,
    seasonal,
    residual,
    model,
    period,
    cycleMinPeriod,
    cycleMaxPeriod,
    dominantCyclePeriod,
    cyclePowerShare,
  } = decomposition;
  const n = dates.length;

  const dataSignature = `${ticker}|${model}|${period}|${cycleMinPeriod}-${cycleMaxPeriod}|${n}|${dates[n - 1] ?? ''}`;
  const fullSignature = `${dataSignature}|${tcnConfig.lookback}|${tcnConfig.dilations.join(',')}|${tcnConfig.kernelSize}|${tcnConfig.numHeads}|${tcnConfig.hiddenChannels}|${tcnConfig.epochs}|${tcnConfig.learningRate}`;
  const fullSensitivitySignature = `${fullSignature}|step:${sensitivityStep}`;
  const isStale = trainedSignature !== null && trainedSignature !== fullSignature;
  const isSensitivityStale =
    sensitivityEnabled && sensitivitySignature !== fullSensitivitySignature;

  const executeTraining = useCallback(
    async (
      cfg: TcnAttentionConfig,
      runSweep: boolean,
      step: SensitivityStep
    ) => {
      const decomp = decompositionRef.current;
      const sym = tickerRef.current;
      if (decomp.observed.length < MIN_BARS_FOR_TCN) {
        setForecast(null);
        setSensitivityResult(null);
        setIsTraining(false);
        setProgress(null);
        return;
      }

      const myRunId = ++runIdRef.current;
      setIsTraining(true);
      setProgress(null);

      const sig = `${sym}|${decomp.model}|${decomp.period}|${decomp.cycleMinPeriod}-${decomp.cycleMaxPeriod}|${decomp.dates.length}|${decomp.dates[decomp.dates.length - 1] ?? ''}|${cfg.lookback}|${cfg.dilations.join(',')}|${cfg.kernelSize}|${cfg.numHeads}|${cfg.hiddenChannels}|${cfg.epochs}|${cfg.learningRate}`;

      if (runSweep) {
        const sweep = await runTcnLookbackSensitivitySweep(sym, decomp, cfg, step, {
          onProgress: (p) => {
            if (runIdRef.current === myRunId) {
              setProgress(p);
            }
          },
          yieldControl: () =>
            new Promise<void>((resolve) => {
              setTimeout(resolve, 0);
            }),
          isCancelled: () => runIdRef.current !== myRunId,
        });

        if (runIdRef.current === myRunId) {
          if (sweep) {
            setForecast(sweep.baseForecast);
            setTrainedSignature(sig);
            setSensitivityResult(sweep);
            setSensitivitySignature(`${sig}|step:${step}`);
          }
          setIsTraining(false);
          setProgress(null);
        }
        return;
      }

      const result = await runTcnDecompositionForecast(sym, decomp, cfg, {
        onProgress: (p) => {
          if (runIdRef.current === myRunId) {
            setProgress(p);
          }
        },
        yieldControl: () =>
          new Promise<void>((resolve) => {
            setTimeout(resolve, 0);
          }),
        isCancelled: () => runIdRef.current !== myRunId,
      });

      if (runIdRef.current === myRunId) {
        setForecast(result);
        setTrainedSignature(result ? sig : null);
        setIsTraining(false);
        setProgress(null);
      }
    },
    []
  );

  // Auto-train the single base lookback window on mount and whenever ticker / decomposition changes.
  // Sensitivity Analysis sweeps are intentionally triggered only on explicit Train/Retrain click.
  useEffect(() => {
    if (n < MIN_BARS_FOR_TCN) return;
    const timer = setTimeout(() => {
      void executeTraining(tcnConfigRef.current, false, 4);
    }, 30);
    return () => {
      clearTimeout(timer);
      runIdRef.current += 1;
    };
  }, [dataSignature, n, executeTraining]);

  if (n === 0) return null;

  const isMult = model === 'multiplicative';
  const refBaseline = isMult ? 1 : 0;

  const renderSubplot = (
    title: string,
    subtitle: string | null,
    data: (number | null)[],
    color: string,
    formatVal: (v: number) => string,
    baseline: number | null,
    overlay?: NextDayPointOverlay
  ) => {
    const valid = data.filter((v): v is number => v !== null && !Number.isNaN(v));
    if (valid.length === 0) return null;

    let min = Math.min(...valid);
    let max = Math.max(...valid);
    if (overlay) {
      min = Math.min(min, overlay.lower99, overlay.predicted);
      max = Math.max(max, overlay.upper99, overlay.predicted);
    }
    const range = max - min || 1;
    const padRange = range * 0.08;
    const yMin = min - padRange;
    const yMax = max + padRange;
    const ySpan = yMax - yMin || 1;

    const width = 760;
    const height = 114;
    const padLeft = 58;
    const padRight = overlay ? 48 : 16;
    const padTop = 14;
    const padBottom = 18;

    const chartW = width - padLeft - padRight;
    const chartH = height - padTop - padBottom;

    // Reserve rightmost slot for t+1 prediction overlay
    const histW = overlay ? chartW * 0.94 : chartW;
    const getX = (i: number) => padLeft + (i / Math.max(1, n - 1)) * histW;
    const getY = (val: number) => padTop + chartH - ((val - yMin) / ySpan) * chartH;

    const points: string[] = [];
    let lastValidIdx = -1;
    let lastValidVal = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v !== null && !Number.isNaN(v)) {
        points.push(`${getX(i).toFixed(1)},${getY(v).toFixed(1)}`);
        lastValidIdx = i;
        lastValidVal = v;
      }
    }

    const nextX = padLeft + chartW;

    return (
      <div className="panel-inset p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 min-w-0">
            <span className="font-semibold text-zinc-800 dark:text-zinc-200">{title}</span>
            {subtitle && (
              <span className="font-mono text-[10px] text-zinc-500 dark:text-zinc-400 tabular-nums">
                · {subtitle}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-500 dark:text-zinc-400 tabular-nums">
            <span>Min {formatVal(min)}</span>
            <span>·</span>
            <span>Max {formatVal(max)}</span>
            {overlay && (
              <>
                <span>·</span>
                <span className="text-gold-600 dark:text-gold-400 font-semibold">
                  t+1: {formatVal(overlay.predicted)}
                </span>
              </>
            )}
          </div>
        </div>

        <div className="w-full aspect-[7.6/1.14]">
          <svg viewBox={`0 0 ${width} ${height}`} className="block h-full w-full">
            {/* Y-axis min/max labels */}
            <text
              x={padLeft - 6}
              y={padTop + 4}
              textAnchor="end"
              className="text-[9px] font-mono fill-zinc-400 dark:fill-zinc-500 tabular-nums"
            >
              {formatVal(max)}
            </text>
            <text
              x={padLeft - 6}
              y={padTop + chartH}
              textAnchor="end"
              className="text-[9px] font-mono fill-zinc-400 dark:fill-zinc-500 tabular-nums"
            >
              {formatVal(min)}
            </text>

            {/* Reference baseline (0 for additive, 1.0 for multiplicative) */}
            {baseline !== null && baseline > yMin && baseline < yMax && (
              <line
                x1={padLeft}
                y1={getY(baseline)}
                x2={width - padRight}
                y2={getY(baseline)}
                stroke="currentColor"
                className="text-zinc-300 dark:text-zinc-700"
                strokeDasharray="2 3"
                strokeWidth={0.8}
              />
            )}

            {/* Historical series path */}
            <path
              d={`M ${points.join(' L ')}`}
              fill="none"
              stroke={color}
              strokeWidth="1.35"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Next-day t+1 forecast overlay */}
            {overlay && lastValidIdx >= 0 && (
              <g>
                {/* Vertical divider separating history from t+1 */}
                <line
                  x1={getX(lastValidIdx)}
                  y1={padTop}
                  x2={getX(lastValidIdx)}
                  y2={padTop + chartH}
                  stroke="currentColor"
                  className="text-zinc-300/60 dark:text-zinc-700/60"
                  strokeDasharray="2 2"
                  strokeWidth={0.75}
                />

                {/* Dashed connector from t to t+1 */}
                <line
                  x1={getX(lastValidIdx)}
                  y1={getY(lastValidVal)}
                  x2={nextX}
                  y2={getY(overlay.predicted)}
                  stroke={color}
                  strokeDasharray="3 2"
                  strokeWidth={1.3}
                />

                {/* 99% confidence whisker */}
                <line
                  x1={nextX}
                  y1={getY(overlay.lower99)}
                  x2={nextX}
                  y2={getY(overlay.upper99)}
                  stroke="#DEC084"
                  strokeWidth={1.6}
                  strokeOpacity={0.85}
                />
                <line
                  x1={nextX - 4}
                  y1={getY(overlay.lower99)}
                  x2={nextX + 4}
                  y2={getY(overlay.lower99)}
                  stroke="#DEC084"
                  strokeWidth={1.2}
                />
                <line
                  x1={nextX - 4}
                  y1={getY(overlay.upper99)}
                  x2={nextX + 4}
                  y2={getY(overlay.upper99)}
                  stroke="#DEC084"
                  strokeWidth={1.2}
                />

                {/* Forecast dot */}
                <circle
                  cx={nextX}
                  cy={getY(overlay.predicted)}
                  r={3.5}
                  fill="#DEC084"
                  stroke="#08080a"
                  strokeWidth={1}
                />

                {/* t+1 label */}
                <text
                  x={nextX}
                  y={height - 4}
                  textAnchor="middle"
                  className="text-[8px] font-mono fill-gold-500 tabular-nums"
                >
                  t+1
                </text>
              </g>
            )}
          </svg>
        </div>
      </div>
    );
  };

  const fmtPrice = (v: number) => `$${v.toFixed(2)}`;
  const fmtFactor = (v: number) => (isMult ? `${v.toFixed(4)}×` : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`);

  const activeForecastForPlot =
    forecast && forecast.model === model && forecast.period === period ? forecast : null;

  const activeSensitivityForPlot =
    sensitivityResult &&
    sensitivityResult.ticker === ticker &&
    sensitivityResult.model === model &&
    sensitivityResult.baseForecast.period === period
      ? sensitivityResult
      : null;

  return (
    <div>
      {/* Header row inside the parent panel */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-zinc-500 dark:text-zinc-400 font-mono tabular-nums">
            <span className="font-sans font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
              {model.toUpperCase()}
            </span>
            <span>·</span>
            <span>Seasonal m = {period}d</span>
            <span>·</span>
            <span>
              Fourier Band {cycleMinPeriod}–{cycleMaxPeriod}d
            </span>
            {dominantCyclePeriod !== null && (
              <>
                <span>·</span>
                <span className="text-gold-600 dark:text-gold-400">
                  Dominant Cycle {dominantCyclePeriod.toFixed(1)}d ({(cyclePowerShare * 100).toFixed(1)}%
                  spectral power)
                </span>
              </>
            )}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            <span>Model:</span>
            {model === 'additive' ? (
              <DecompositionAdditiveEquation
                display="inline"
                className="text-zinc-700 dark:text-zinc-300"
              />
            ) : (
              <DecompositionMultiplicativeEquation
                display="inline"
                className="text-zinc-700 dark:text-zinc-300"
              />
            )}
          </div>
        </div>

        {/* Component filter segmented strip */}
        <div className="flex items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-lg self-start sm:self-auto">
          {(['all', 'trend', 'cyclical', 'seasonal', 'residual'] as const).map((comp) => (
            <button
              key={comp}
              type="button"
              onClick={() => setActiveComponent(comp)}
              className={`px-2.5 py-1 text-xs font-medium rounded-md capitalize transition-colors cursor-pointer ${
                activeComponent === comp
                  ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              {comp}
            </button>
          ))}
        </div>
      </div>

      {/* 5 Decomposition Subplots (with t+1 forecast overlay when trained) */}
      <div className="space-y-3">
        {activeComponent === 'all' &&
          renderSubplot(
            'Observed Price (Y_t)',
            `${n} trading sessions`,
            observed,
            '#DEC084',
            fmtPrice,
            null,
            activeForecastForPlot
              ? {
                  predicted: activeForecastForPlot.predictedPrice,
                  lower99: activeForecastForPlot.lower99,
                  upper99: activeForecastForPlot.upper99,
                  dateLabel: activeForecastForPlot.nextDateLabel,
                }
              : undefined
          )}

        {(activeComponent === 'all' || activeComponent === 'trend') &&
          renderSubplot(
            'Secular Trend (T_t)',
            'Fourier low-pass secular component',
            trend,
            '#3fb27f',
            fmtPrice,
            null,
            activeForecastForPlot
              ? {
                  predicted: activeForecastForPlot.components.trend.predictedValue,
                  lower99: activeForecastForPlot.components.trend.lower99,
                  upper99: activeForecastForPlot.components.trend.upper99,
                  dateLabel: activeForecastForPlot.nextDateLabel,
                }
              : undefined
          )}

        {(activeComponent === 'all' || activeComponent === 'cyclical') &&
          renderSubplot(
            'Cyclical Component (C_t)',
            dominantCyclePeriod
              ? `Fourier bandpass ${cycleMinPeriod}–${cycleMaxPeriod}d · peak ${dominantCyclePeriod.toFixed(1)}d`
              : `Fourier bandpass ${cycleMinPeriod}–${cycleMaxPeriod}d`,
            cyclical,
            '#d4a853',
            fmtFactor,
            refBaseline,
            activeForecastForPlot
              ? {
                  predicted: activeForecastForPlot.components.cyclical.predictedValue,
                  lower99: activeForecastForPlot.components.cyclical.lower99,
                  upper99: activeForecastForPlot.components.cyclical.upper99,
                  dateLabel: activeForecastForPlot.nextDateLabel,
                }
              : undefined
          )}

        {(activeComponent === 'all' || activeComponent === 'seasonal') &&
          renderSubplot(
            'Seasonal Component (S_t)',
            `Repeating periodic factor (m = ${period}d)`,
            seasonal,
            '#a1a1aa',
            fmtFactor,
            refBaseline,
            activeForecastForPlot
              ? {
                  predicted: activeForecastForPlot.components.seasonal.predictedValue,
                  lower99: activeForecastForPlot.components.seasonal.lower99,
                  upper99: activeForecastForPlot.components.seasonal.upper99,
                  dateLabel: activeForecastForPlot.nextDateLabel,
                }
              : undefined
          )}

        {(activeComponent === 'all' || activeComponent === 'residual') &&
          renderSubplot(
            'Irregular Residual (I_t)',
            'Isolated remainder after T, C, S',
            residual,
            '#e0655f',
            fmtFactor,
            refBaseline,
            activeForecastForPlot
              ? {
                  predicted: activeForecastForPlot.components.residual.predictedValue,
                  lower99: activeForecastForPlot.components.residual.lower99,
                  upper99: activeForecastForPlot.components.residual.upper99,
                  dateLabel: activeForecastForPlot.nextDateLabel,
                }
              : undefined
          )}
      </div>

      {/* Inline TCN + Multi-Head Self-Attention Forecasting Lab */}
      <TcnForecastPanel
        ticker={ticker}
        decomposition={decomposition}
        config={tcnConfig}
        onConfigChange={setTcnConfig}
        forecast={activeForecastForPlot}
        sensitivityEnabled={sensitivityEnabled}
        onSensitivityEnabledChange={setSensitivityEnabled}
        sensitivityStep={sensitivityStep}
        onSensitivityStepChange={setSensitivityStep}
        sensitivityResult={activeSensitivityForPlot}
        isSensitivityStale={isSensitivityStale}
        isTraining={isTraining}
        progress={progress}
        isStale={isStale}
        onTrain={() => {
          void executeTraining(tcnConfig, sensitivityEnabled, sensitivityStep);
        }}
      />

      {/* Augmented Dickey-Fuller (ADF) Test on Residuals Box */}
      <div className="mt-5 pt-4 border-t border-zinc-200/80 dark:border-zinc-800/80">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-700 dark:text-zinc-300 block">
              Augmented Dickey-Fuller (ADF) Test on Irregular Residuals
            </span>
            <div className="mt-0.5">
              <AdfEquation display="inline" className="text-xs text-zinc-600 dark:text-zinc-300" />
            </div>
          </div>
          <span
            className={`text-xs font-semibold ${
              adfTest.isStationary ? 'text-[#3fb27f]' : 'text-gold-600 dark:text-gold-400'
            }`}
          >
            {adfTest.isStationary ? 'Stationary (p < 0.05)' : 'Non-Stationary (Unit Root)'}
          </span>
        </div>

        <div className="panel-inset grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 text-xs">
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">Test Statistic (τ):</div>
            <div className="font-mono font-bold text-zinc-900 dark:text-zinc-100 text-sm tabular-nums">
              {adfTest.testStatistic.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">MacKinnon p-value:</div>
            <div className="font-mono font-bold text-zinc-900 dark:text-zinc-100 text-sm tabular-nums">
              {adfTest.pValue < 0.001 ? '< 0.001' : adfTest.pValue.toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">1% Critical Value:</div>
            <div className="font-mono text-zinc-700 dark:text-zinc-300 tabular-nums">
              {adfTest.criticalValues['1%'].toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-zinc-500 dark:text-zinc-400">5% Critical Value:</div>
            <div className="font-mono text-zinc-700 dark:text-zinc-300 tabular-nums">
              {adfTest.criticalValues['5%'].toFixed(3)}
            </div>
          </div>
        </div>

        <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">{adfTest.conclusion}</p>
      </div>
    </div>
  );
}
