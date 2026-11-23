'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { Data, Layout, Shape } from 'plotly.js';
import type { DecompositionResult, AdfTestResult } from '@/lib/quant-engine';
import {
  DEFAULT_TCN_CONFIG,
  MIN_BARS_FOR_TCN,
  buildTcnCacheKey,
  buildTcnDataSignature,
  buildTcnSensitivityCacheKey,
  getCachedTcnArtifact,
  getCachedTcnSensitivity,
  getLatestCachedTcnArtifactForData,
  hydrateTcnCacheFromSessionStorage,
  runTcnDecompositionForecast,
  runTcnLookbackSensitivitySweep,
  type SensitivityStep,
  type TcnAttentionConfig,
  type TcnTrainingProgress,
} from '@/lib/tcn-attention';
import {
  DecompositionAdditiveEquation,
  DecompositionMultiplicativeEquation,
  AdfEquation,
} from '@/components/math/MathEquations';
import { TcnForecastPanel } from '@/components/charts/TcnForecastPanel';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

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

const persistedChartUiState: {
  activeComponent: ComponentFilter;
  tcnConfig: TcnAttentionConfig;
  sensitivityEnabled: boolean;
  sensitivityStep: SensitivityStep;
} = {
  activeComponent: 'all',
  tcnConfig: { ...DEFAULT_TCN_CONFIG },
  sensitivityEnabled: false,
  sensitivityStep: 4,
};

function nextBusinessDay(dateStr: string): string {
  const ms = Date.parse(`${dateStr}T00:00:00Z`);
  if (!Number.isFinite(ms)) return `${dateStr} (t+1)`;
  let next = ms + 86400000;
  const d = new Date(next);
  if (d.getUTCDay() === 6) next += 2 * 86400000;
  else if (d.getUTCDay() === 0) next += 86400000;
  return new Date(next).toISOString().slice(0, 10);
}

interface DecompositionSubplotProps {
  title: string;
  subtitle: string | null;
  dates: string[];
  data: (number | null)[];
  color: string;
  formatVal: (v: number) => string;
  baseline: number | null;
  overlay?: NextDayPointOverlay;
  showSlider?: boolean;
  isDark: boolean;
  uirevisionKey: string;
}

function DecompositionSubplot({
  title,
  subtitle,
  dates,
  data,
  color,
  formatVal,
  baseline,
  overlay,
  showSlider = false,
  isDark,
  uirevisionKey,
}: DecompositionSubplotProps) {
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const { traces, layout, minVal, maxVal, hasData } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
    minVal: number;
    maxVal: number;
    hasData: boolean;
  }>(() => {
    const valid = data.filter((v): v is number => v !== null && !Number.isNaN(v));
    if (valid.length === 0) {
      return { traces: [], layout: {}, minVal: 0, maxVal: 0, hasData: false };
    }
    let min = Math.min(...valid);
    let max = Math.max(...valid);
    if (overlay) {
      min = Math.min(min, overlay.lower99, overlay.predicted);
      max = Math.max(max, overlay.upper99, overlay.predicted);
    }

    const xs = dates;
    const ys = data.map((v) => (v !== null && !Number.isNaN(v) ? v : null));

    let lastValidIdx = -1;
    let lastValidVal = 0;
    for (let i = ys.length - 1; i >= 0; i--) {
      if (ys[i] !== null) {
        lastValidIdx = i;
        lastValidVal = ys[i] as number;
        break;
      }
    }

    const dataTraces: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: title,
        x: xs,
        y: ys,
        line: { color, width: 1.7 },
        connectgaps: false,
        hovertemplate: '%{y:.4~f}<extra>' + title + '</extra>',
      },
    ];

    const shapes: Partial<Shape>[] = [];

    if (baseline !== null) {
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: baseline,
        y1: baseline,
        line: { color: tokens.zeroLine, width: 1, dash: 'dot' },
      });
    }

    if (overlay && lastValidIdx >= 0) {
      const lastDate = xs[lastValidIdx];
      const nextDate = nextBusinessDay(lastDate);
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: lastDate,
        x1: lastDate,
        y0: 0,
        y1: 1,
        line: { color: tokens.textMuted, width: 0.85, dash: 'dot' },
      });
      dataTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: 't→t+1',
        x: [lastDate, nextDate],
        y: [lastValidVal, overlay.predicted],
        line: { color, width: 1.4, dash: 'dot' },
        showlegend: false,
        hoverinfo: 'skip',
      });
      dataTraces.push({
        type: 'scatter',
        mode: 'markers',
        name: 't+1 TCN Forecast',
        x: [nextDate],
        y: [overlay.predicted],
        error_y: {
          type: 'data',
          symmetric: false,
          array: [Math.max(0, overlay.upper99 - overlay.predicted)],
          arrayminus: [Math.max(0, overlay.predicted - overlay.lower99)],
          color: QUANT_COLORS.goldLight,
          thickness: 1.6,
          width: 4,
        },
        marker: {
          color: QUANT_COLORS.goldLight,
          size: 7.5,
          line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1.2 },
        },
        hovertemplate: `t+1: ${formatVal(overlay.predicted)}<br>99% PI [${formatVal(
          overlay.lower99
        )}, ${formatVal(overlay.upper99)}]<extra>TCN Forecast</extra>`,
      });
    }

    return {
      traces: dataTraces,
      layout: {
        margin: { l: 54, r: 16, t: 8, b: showSlider ? 20 : 24 },
        shapes,
        yaxis: {
          gridcolor: tokens.grid,
          zeroline: false,
          nticks: 4,
        },
      },
      minVal: min,
      maxVal: max,
      hasData: true,
    };
  }, [data, dates, title, color, baseline, overlay, showSlider, isDark, tokens, formatVal]);

  if (!hasData) return null;

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
          <span>Min {formatVal(minVal)}</span>
          <span>·</span>
          <span>Max {formatVal(maxVal)}</span>
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

      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        rangeSlider={showSlider}
        hovermode="x unified"
        height={showSlider ? 220 : 155}
        exportFilename={`quantmetrics-decomp-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`}
        uirevisionKey={uirevisionKey}
      />
    </div>
  );
}

export function DecompositionChart({
  ticker = 'ASSET',
  decomposition,
  adfTest,
}: DecompositionChartProps) {
  const isDark = usePlotlyDarkTheme();
  const [activeComponent, setActiveComponentState] = useState<ComponentFilter>(
    () => persistedChartUiState.activeComponent
  );
  const [tcnConfig, setTcnConfigState] = useState<TcnAttentionConfig>(() => ({
    ...persistedChartUiState.tcnConfig,
    dilations: [...persistedChartUiState.tcnConfig.dilations],
  }));
  const [sensitivityEnabled, setSensitivityEnabledState] = useState<boolean>(
    () => persistedChartUiState.sensitivityEnabled
  );
  const [sensitivityStep, setSensitivityStepState] = useState<SensitivityStep>(
    () => persistedChartUiState.sensitivityStep
  );
  const [cacheRevision, setCacheRevision] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [progress, setProgress] = useState<TcnTrainingProgress | null>(null);

  const setActiveComponent = useCallback((next: ComponentFilter) => {
    persistedChartUiState.activeComponent = next;
    setActiveComponentState(next);
  }, []);

  const setTcnConfig = useCallback((next: TcnAttentionConfig) => {
    persistedChartUiState.tcnConfig = { ...next, dilations: [...next.dilations] };
    setTcnConfigState(next);
  }, []);

  const setSensitivityEnabled = useCallback((enabled: boolean) => {
    persistedChartUiState.sensitivityEnabled = enabled;
    setSensitivityEnabledState(enabled);
  }, []);

  const setSensitivityStep = useCallback((step: SensitivityStep) => {
    persistedChartUiState.sensitivityStep = step;
    setSensitivityStepState(step);
  }, []);

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

  const dataSignature = buildTcnDataSignature(ticker, decomposition);
  const fullCacheKey = buildTcnCacheKey(ticker, decomposition, tcnConfig);
  const fullSensitivityKey = buildTcnSensitivityCacheKey(
    ticker,
    decomposition,
    tcnConfig,
    sensitivityStep
  );

  const exactCachedArtifact = cacheRevision >= 0 ? getCachedTcnArtifact(fullCacheKey) : null;
  const fallbackCachedArtifact =
    exactCachedArtifact ?? getLatestCachedTcnArtifactForData(dataSignature);
  const forecast = fallbackCachedArtifact?.forecast ?? null;
  const exactCachedSweep =
    cacheRevision >= 0 ? getCachedTcnSensitivity(fullSensitivityKey) : null;
  const sensitivityResult = exactCachedSweep?.result ?? null;

  const isStale = forecast !== null && exactCachedArtifact === null;
  const isSensitivityStale = sensitivityEnabled && exactCachedSweep === null;

  const executeTraining = useCallback(
    async (
      cfg: TcnAttentionConfig,
      runSweep: boolean,
      step: SensitivityStep,
      forceRetrain: boolean = false
    ) => {
      const decomp = decompositionRef.current;
      const sym = tickerRef.current;
      if (decomp.observed.length < MIN_BARS_FOR_TCN) {
        setIsTraining(false);
        setProgress(null);
        return;
      }

      if (!forceRetrain) {
        if (runSweep) {
          const sweepKey = buildTcnSensitivityCacheKey(sym, decomp, cfg, step);
          if (getCachedTcnSensitivity(sweepKey)) {
            setCacheRevision((v) => v + 1);
            setIsTraining(false);
            setProgress(null);
            return;
          }
        } else {
          const key = buildTcnCacheKey(sym, decomp, cfg);
          if (getCachedTcnArtifact(key)) {
            setCacheRevision((v) => v + 1);
            setIsTraining(false);
            setProgress(null);
            return;
          }
        }
      }

      const myRunId = ++runIdRef.current;
      setIsTraining(true);
      setProgress(null);

      if (runSweep) {
        await runTcnLookbackSensitivitySweep(sym, decomp, cfg, step, {
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
          useCache: true,
          forceRetrain,
        });

        if (runIdRef.current === myRunId) {
          setCacheRevision((v) => v + 1);
          setIsTraining(false);
          setProgress(null);
        }
        return;
      }

      await runTcnDecompositionForecast(sym, decomp, cfg, {
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
        useCache: true,
        forceRetrain,
      });

      if (runIdRef.current === myRunId) {
        setCacheRevision((v) => v + 1);
        setIsTraining(false);
        setProgress(null);
      }
    },
    []
  );

  // Hydrate a previously fitted model from sessionStorage on mount. Training itself
  // is manual-only — the Train button below is the sole trigger (live quote ticks
  // must never start a fit).
  useEffect(() => {
    const timer = setTimeout(() => {
      const hydrated = hydrateTcnCacheFromSessionStorage();
      if (hydrated) {
        setCacheRevision((v) => v + 1);
      }
    }, 15);
    return () => clearTimeout(timer);
  }, [executeTraining]);

  const isMult = model === 'multiplicative';
  const refBaseline = isMult ? 1 : 0;

  const fmtPrice = useCallback((v: number) => `$${v.toFixed(2)}`, []);
  const fmtFactor = useCallback(
    (v: number) => (isMult ? `${v.toFixed(4)}×` : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`),
    [isMult]
  );

  if (n === 0) return null;

  const activeForecastForPlot =
    forecast && forecast.model === model && forecast.period === period ? forecast : null;

  const activeSensitivityForPlot =
    sensitivityResult &&
    sensitivityResult.ticker === ticker &&
    sensitivityResult.model === model &&
    sensitivityResult.baseForecast.period === period
      ? sensitivityResult
      : null;

  const baseRev = `${ticker}-${model}-${period}-${n}`;

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
                  Dominant Cycle {dominantCyclePeriod.toFixed(1)}d (
                  {(cyclePowerShare * 100).toFixed(1)}% spectral power)
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
              className={`px-2.5 py-1 text-xs font-medium rounded-md capitalize transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
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

      {/* 5 Interactive Plotly Decomposition Subplots (with t+1 forecast overlay when trained) */}
      <div className="space-y-3">
        {activeComponent === 'all' && (
          <DecompositionSubplot
            title="Observed Price (Y_t)"
            subtitle={`${n} trading sessions · Wheel: Pan X · Ctrl+Wheel: Zoom`}
            dates={dates}
            data={observed}
            color="#DEC084"
            formatVal={fmtPrice}
            baseline={null}
            overlay={
              activeForecastForPlot
                ? {
                    predicted: activeForecastForPlot.predictedPrice,
                    lower99: activeForecastForPlot.lower99,
                    upper99: activeForecastForPlot.upper99,
                    dateLabel: activeForecastForPlot.nextDateLabel,
                  }
                : undefined
            }
            showSlider
            isDark={isDark}
            uirevisionKey={`obs-${baseRev}`}
          />
        )}

        {(activeComponent === 'all' || activeComponent === 'trend') && (
          <DecompositionSubplot
            title="Secular Trend (T_t)"
            subtitle="Fourier low-pass secular component"
            dates={dates}
            data={trend}
            color="#3fb27f"
            formatVal={fmtPrice}
            baseline={null}
            overlay={
              activeForecastForPlot
                ? {
                    predicted: activeForecastForPlot.components.trend.predictedValue,
                    lower99: activeForecastForPlot.components.trend.lower99,
                    upper99: activeForecastForPlot.components.trend.upper99,
                    dateLabel: activeForecastForPlot.nextDateLabel,
                  }
                : undefined
            }
            showSlider={activeComponent === 'trend'}
            isDark={isDark}
            uirevisionKey={`trend-${baseRev}`}
          />
        )}

        {(activeComponent === 'all' || activeComponent === 'cyclical') && (
          <DecompositionSubplot
            title="Cyclical Component (C_t)"
            subtitle={
              dominantCyclePeriod
                ? `Fourier bandpass ${cycleMinPeriod}–${cycleMaxPeriod}d · peak ${dominantCyclePeriod.toFixed(1)}d`
                : `Fourier bandpass ${cycleMinPeriod}–${cycleMaxPeriod}d`
            }
            dates={dates}
            data={cyclical}
            color="#d4a853"
            formatVal={fmtFactor}
            baseline={refBaseline}
            overlay={
              activeForecastForPlot
                ? {
                    predicted: activeForecastForPlot.components.cyclical.predictedValue,
                    lower99: activeForecastForPlot.components.cyclical.lower99,
                    upper99: activeForecastForPlot.components.cyclical.upper99,
                    dateLabel: activeForecastForPlot.nextDateLabel,
                  }
                : undefined
            }
            showSlider={activeComponent === 'cyclical'}
            isDark={isDark}
            uirevisionKey={`cyc-${baseRev}`}
          />
        )}

        {(activeComponent === 'all' || activeComponent === 'seasonal') && (
          <DecompositionSubplot
            title="Seasonal Component (S_t)"
            subtitle={`Repeating periodic factor (m = ${period}d)`}
            dates={dates}
            data={seasonal}
            color="#a1a1aa"
            formatVal={fmtFactor}
            baseline={refBaseline}
            overlay={
              activeForecastForPlot
                ? {
                    predicted: activeForecastForPlot.components.seasonal.predictedValue,
                    lower99: activeForecastForPlot.components.seasonal.lower99,
                    upper99: activeForecastForPlot.components.seasonal.upper99,
                    dateLabel: activeForecastForPlot.nextDateLabel,
                  }
                : undefined
            }
            showSlider={activeComponent === 'seasonal'}
            isDark={isDark}
            uirevisionKey={`seas-${baseRev}`}
          />
        )}

        {(activeComponent === 'all' || activeComponent === 'residual') && (
          <DecompositionSubplot
            title="Irregular Residual (I_t)"
            subtitle="Isolated remainder after T, C, S"
            dates={dates}
            data={residual}
            color="#e0655f"
            formatVal={fmtFactor}
            baseline={refBaseline}
            overlay={
              activeForecastForPlot
                ? {
                    predicted: activeForecastForPlot.components.residual.predictedValue,
                    lower99: activeForecastForPlot.components.residual.lower99,
                    upper99: activeForecastForPlot.components.residual.upper99,
                    dateLabel: activeForecastForPlot.nextDateLabel,
                  }
                : undefined
            }
            showSlider={activeComponent === 'residual'}
            isDark={isDark}
            uirevisionKey={`res-${baseRev}`}
          />
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
          const shouldForceRetrain = sensitivityEnabled
            ? exactCachedSweep !== null
            : exactCachedArtifact !== null;
          void executeTraining(tcnConfig, sensitivityEnabled, sensitivityStep, shouldForceRetrain);
        }}
      />

      {/* Augmented Dickey-Fuller (ADF) Test on Residuals Box */}
      <div className="mt-5 pt-4 border-t border-[var(--hairline)]">
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
