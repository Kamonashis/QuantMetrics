'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  PriceBar,
  calculateReturns,
  performRegressionAnalysis,
} from '@/lib/quant-engine';
import {
  getCachedTcnRegressionForecast,
  runTcnMultiAssetRegression,
  type TcnMultiAssetRegressionResult,
} from '@/lib/tcn-multi-domain';
import {
  ALL_FACTORS,
  FACTOR_BY_ID,
  computePriceFactorSeries,
  computeFundamentalFactors,
  computeAnalystFactors,
  computeOptionsFactors,
  type FactorDefinition,
} from '@/lib/factors';
import { useFactorData } from '@/hooks/use-factor-data';
import { FactorResearchPanel } from '@/components/views/regression/FactorResearchPanel';
import { ResidualDiagnosticsChart } from '@/components/charts/ResidualDiagnosticsChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import {
  RegressionModelEquation,
  OlsMatrixEquation,
  RobustIrlsEquation,
  VifEquation,
  ExpectedReturnEquation,
  MathSymbol,
} from '@/components/math/MathEquations';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { TickerSuggest } from '@/components/ui/TickerSuggest';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import type { Data, Layout } from 'plotly.js';
import { getHmmRegimeToken } from '@/lib/hmm-regime-overlay';
import {
  HmmRegimeOverlayToggle,
  HmmRegimeScorecard,
} from '@/components/charts/HmmRegimeOverlayControls';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import {
  buildHmmRegimeShapes,
  getPlotlyThemeTokens,
  QUANT_COLORS,
} from '@/lib/plotly-theme';
import { Play, RotateCcw, HelpCircle, ArrowUpRight, TrendingUp, Info, Sparkles } from 'lucide-react';

function TcnRegressionOosChart({ tcnReg }: { tcnReg: TcnMultiAssetRegressionResult }) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [showRegimeOverlay, setShowRegimeOverlay] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const hist = tcnReg.oosHistory;
  const series = useMemo(() => {
    if (!hist || hist.length === 0) return [];
    let cumAct = 0;
    let cumPred = 0;
    let cumOls = 0;
    return hist.map((pt) => {
      cumAct += pt.actualReturn * 100;
      cumPred += pt.predictedReturn * 100;
      cumOls += pt.olsPredictedReturn * 100;
      const hit =
        (pt.predictedReturn >= 0 && pt.actualReturn >= 0) ||
        (pt.predictedReturn < 0 && pt.actualReturn < 0);
      return {
        ...pt,
        cumAct,
        cumPred,
        cumOls,
        hit,
      };
    });
  }, [hist]);

  const { traces, layout } = useMemo<{
    traces: Data[];
    layout: Partial<Layout>;
  }>(() => {
    if (series.length < 3) return { traces: [], layout: {} };

    const xs = series.map((s) => s.date);
    const regimeColors = series.map((s) => getHmmRegimeToken(s.hmmState).color);
    const stride = Math.max(1, Math.ceil(series.length / 44));

    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Lag-Matched OLS',
        x: xs,
        y: series.map((s) => s.cumOls),
        line: { color: '#64748b', width: 1.2, dash: 'dot' },
        hovertemplate: 'OLS Cum: %{y:+.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Realized Cum Return',
        x: xs,
        y: series.map((s) => s.cumAct),
        line: { color: '#94a3b8', width: 1.6, dash: 'dash' },
        hovertemplate: 'Realized Cum: %{y:+.2f}%<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'TCN Cum Pred',
        x: xs,
        y: series.map((s) => s.cumPred),
        line: { color: QUANT_COLORS.gold, width: 2 },
        hovertemplate: 'TCN Cum Pred: %{y:+.2f}%<extra></extra>',
      },
    ];

    if (showRegimeOverlay) {
      const mX: string[] = [];
      const mY: number[] = [];
      const mColor: string[] = [];
      const mText: string[] = [];
      for (let i = 0; i < series.length; i++) {
        if (i % stride !== 0 && i !== series.length - 1) continue;
        const s = series[i];
        const tok = getHmmRegimeToken(s.hmmState);
        mX.push(s.date);
        mY.push(s.cumPred);
        mColor.push(regimeColors[i]);
        mText.push(`${tok.label} (${Math.round(s.hmmProb * 100)}%)`);
      }
      data.push({
        type: 'scatter',
        mode: 'markers',
        name: 'HMM Origin Regime',
        x: mX,
        y: mY,
        marker: {
          size: 6,
          color: mColor,
          line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 },
        },
        text: mText,
        hovertemplate: 'Regime: %{text}<extra></extra>',
        showlegend: false,
      });
    }

    const shapes: Layout['shapes'] = [
      ...buildHmmRegimeShapes(
        xs,
        series.map((s) => s.hmmState),
        showRegimeOverlay
      ),
      {
        type: 'line',
        xref: 'paper',
        x0: 0,
        x1: 1,
        yref: 'y',
        y0: 0,
        y1: 0,
        line: { color: tokens.zeroLine, width: 1 },
      },
    ];

    return {
      traces: data,
      layout: {
        margin: { l: 50, r: 16, t: 10, b: 36 },
        shapes,
        yaxis: {
          ticksuffix: '%',
          tickfont: { size: 10, color: tokens.textMuted },
        },
      },
    };
  }, [series, showRegimeOverlay, isDark, tokens]);

  if (series.length < 3) return null;

  const n = series.length;
  const activeIdx = hoverIdx !== null && hoverIdx < n ? hoverIdx : n - 1;
  const activePt = series[activeIdx];
  const activeTok = getHmmRegimeToken(activePt.hmmState);

  return (
    <div className="mt-4 pt-4 border-t border-[var(--hairline)] space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
            Strictly OOS 1d Cumulative Forecast vs. Realized Return ({tcnReg.dependentTicker})
          </h4>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400">
            Cumulative 1d predicted path (TCN-Attention vs. lag-matched OLS) color-coded by causal HMM regime at origin t−1.
          </p>
        </div>
        <HmmRegimeOverlayToggle
          enabled={showRegimeOverlay}
          onToggle={() => setShowRegimeOverlay((v) => !v)}
          caption="origin t−1 regime"
        />
      </div>

      {/* Live Hover Readout Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <strong className="text-zinc-900 dark:text-zinc-100">{activePt.date}</strong>
          <span>·</span>
          <span className="inline-flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: activeTok.color }}
            />
            <strong className={activeTok.textClass}>
              {activeTok.label} ({Math.round(activePt.hmmProb * 100)}%)
            </strong>
          </span>
          <span>·</span>
          <span>
            TCN r̂{' '}
            <strong className={activePt.predictedReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
              {activePt.predictedReturn >= 0 ? '+' : ''}
              {(activePt.predictedReturn * 100).toFixed(2)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            Actual r{' '}
            <strong className={activePt.actualReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
              {activePt.actualReturn >= 0 ? '+' : ''}
              {(activePt.actualReturn * 100).toFixed(2)}%
            </strong>
          </span>
          <span>·</span>
          <span>
            OLS {activePt.olsPredictedReturn >= 0 ? '+' : ''}
            {(activePt.olsPredictedReturn * 100).toFixed(2)}%
          </span>
        </div>

        <span
          className={`font-semibold ${
            activePt.hit
              ? 'text-emerald-700 dark:text-emerald-400'
              : 'text-rose-600 dark:text-rose-400'
          }`}
        >
          {activePt.hit ? 'Direction Hit' : 'Direction Miss'}
        </span>
      </div>

      <div className="w-full rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)]/40 p-1.5">
        <PlotlyChart
          data={traces}
          layout={layout}
          plotType="2d"
          rangeSlider
          hovermode="x unified"
          showLegend={false}
          height={240}
          showWheelHint
          exportFilename={`${tcnReg.dependentTicker}-tcn-regression-oos`}
          uirevisionKey={`tcn-reg-oos-${tcnReg.dependentTicker}-${showRegimeOverlay}`}
          onHover={(ev) => {
            const pt = (ev as { points?: Array<{ x?: unknown }> })?.points?.[0];
            if (pt?.x !== undefined) {
              const idx = series.findIndex((s) => s.date === String(pt.x));
              if (idx >= 0) setHoverIdx(idx);
            }
          }}
          onUnhover={() => setHoverIdx(null)}
        />
      </div>

      {/* Legend Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[#DEC084] inline-block" />
            TCN Cum Pred
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dashed border-[#94a3b8] inline-block" />
            Realized Cum Return
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 border-t border-dotted border-[#64748b] inline-block" />
            Lag-Matched OLS
          </span>
        </div>
        <span>Hover sessions to compare TCN vs. OLS across HMM regimes</span>
      </div>

      {/* Per-Regime Regression Accuracy Scorecard */}
      {tcnReg.regimePerformance && tcnReg.regimePerformance.length > 0 && (
        <HmmRegimeScorecard
          title={`TCN 1d OOS Regression Accuracy by Causal HMM Regime (${tcnReg.dependentTicker})`}
          subtitle="Stratified by dominant HMM state at t−1 prediction origin"
          columns={[
            { key: 'hit', label: 'Dir Hit Rate' },
            { key: 'tcnRmse', label: 'TCN OOS RMSE' },
            { key: 'olsRmse', label: 'OLS OOS RMSE' },
            { key: 'means', label: 'Mean Pred / Act' },
          ]}
          rows={tcnReg.regimePerformance.map((r) => ({
            state: r.state,
            count: r.count,
            cells: {
              hit:
                r.hitRatePct !== null ? (
                  <span
                    className={
                      r.hitRatePct >= 52
                        ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                        : 'text-zinc-800 dark:text-zinc-200'
                    }
                  >
                    {r.hitRatePct.toFixed(0)}%
                  </span>
                ) : (
                  '—'
                ),
              tcnRmse: r.oosRmse !== null ? `${(r.oosRmse * 100).toFixed(2)}%` : '—',
              olsRmse: r.olsOosRmse !== null ? `${(r.olsOosRmse * 100).toFixed(2)}%` : '—',
              means:
                r.meanPredReturn !== null && r.meanActualReturn !== null
                  ? `${r.meanPredReturn >= 0 ? '+' : ''}${(r.meanPredReturn * 100).toFixed(2)}% / ${r.meanActualReturn >= 0 ? '+' : ''}${(r.meanActualReturn * 100).toFixed(2)}%`
                  : '—',
            },
          }))}
        />
      )}
    </div>
  );
}

interface RegressionViewProps {
  regressionTickers: string;
  onRegressionTickersChange: (val: string) => void;
  startDate: string;
  onStartDateChange: (val: string) => void;
  endDate: string;
  onEndDateChange: (val: string) => void;
  returnType: 'simple' | 'log';
  allStockBars: { [ticker: string]: PriceBar[] };
  isLoading: boolean;
  onRunRegression: (tickers: string[]) => void;
  barsSource?: string | null;
}

export function RegressionView({
  regressionTickers,
  onRegressionTickersChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  returnType,
  allStockBars,
  isLoading,
  onRunRegression,
  barsSource,
}: RegressionViewProps) {
  const [localInput, setLocalInput] = useState(regressionTickers);
  const [regressionType, setRegressionType] = useState<
    'Standard OLS' | 'Robust Regression' | 'Rolling Regression' | 'TCN-Attention Sequence'
  >('Standard OLS');
  const [rollingWindow, setRollingWindow] = useState<number>(60);
  const [selectedFactorIds, setSelectedFactorIds] = useState<string[]>([]);
  const [universeSymbols, setUniverseSymbols] = useState<string[]>(() =>
    Object.keys(allStockBars ?? {}).slice(0, 12)
  );
  const factorData = useFactorData(universeSymbols, true);

  const parsedTickers = React.useMemo(() => {
    return regressionTickers
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
  }, [regressionTickers]);

  const dependentTicker = parsedTickers[0] || 'AAPL';
  const independentTickers = React.useMemo(() => {
    return parsedTickers.slice(1).length > 0 ? parsedTickers.slice(1) : ['SPY'];
  }, [parsedTickers]);

  // Align dates and returns, then inject any selected factor series as named
  // regressor columns. Factor columns use an `F:` prefix so they can never
  // collide with a real ticker symbol.
  const {
    returnsByTicker,
    dates,
    factorColumnKeys,
    factorColumnCoverage,
  } = React.useMemo(() => {
    const map: { [ticker: string]: number[] } = {};
    let commonDates: string[] = [];
    let dependentBars: PriceBar[] = [];

    for (const t of [dependentTicker, ...independentTickers]) {
      const raw = allStockBars[t] || [];
      const filtered = raw.filter((b) => {
        if (startDate && b.date < startDate) return false;
        if (endDate && b.date > endDate) return false;
        return true;
      });
      if (t === dependentTicker) dependentBars = filtered;
      const prices = filtered.map((b) => b.close);
      map[t] = calculateReturns(prices, returnType);
      if (filtered.length > commonDates.length) {
        commonDates = filtered.map((b) => b.date);
      }
    }

    const keys: string[] = [];
    const coverage: { id: string; available: boolean; observations: number }[] = [];
    if (selectedFactorIds.length > 0 && dependentBars.length > 1) {
      const benchmark = allStockBars['SPY'] ?? dependentBars;
      const priceSeries = computePriceFactorSeries(dependentBars, benchmark);
      const barIndexByDate = new Map(dependentBars.map((b, i) => [b.date, i]));
      const returnDates = dependentBars.slice(1).map((b) => b.date);
      const recentReturns = calculateReturns(dependentBars.map((b) => b.close), 'log').slice(-20);
      const rv20 =
        recentReturns.length >= 10
          ? Math.sqrt(
              Math.max(
                0,
                recentReturns.reduce((a, b) => a + (b - recentReturns.reduce((x, y) => x + y, 0) / recentReturns.length) ** 2, 0) /
                  (recentReturns.length - 1) *
                  252
              )
            )
          : null;

      const usable: { def: FactorDefinition; values: (number | null)[] }[] = [];
      for (const id of selectedFactorIds) {
        const def = FACTOR_BY_ID[id];
        if (!def) continue;
        let values: (number | null)[];
        if (def.requires === 'price') {
          values = returnDates.map((d) => {
            const idx = barIndexByDate.get(d);
            return idx === undefined ? null : priceSeries[id]?.[idx] ?? null;
          });
        } else {
          let snapshotValue: number | null = null;
          if (def.requires === 'fundamentals') {
            const snap = factorData.fundamentalsBySymbol[dependentTicker]?.[0] ?? null;
            if (snap && endDate) snapshotValue = computeFundamentalFactors(snap, endDate)[id] ?? null;
          } else if (def.requires === 'analyst') {
            snapshotValue = computeAnalystFactors(factorData.analystBySymbol[dependentTicker] ?? null)[id] ?? null;
          } else if (def.requires === 'options') {
            const snap = factorData.optionsBySymbol[dependentTicker]?.[0] ?? null;
            snapshotValue = computeOptionsFactors(snap ? { ...snap, rv20: snap.rv20 ?? rv20 } : null)[id] ?? null;
          }
          values = returnDates.map(() => snapshotValue);
        }
        if (values.every((v) => v === null)) {
          coverage.push({ id, available: false, observations: 0 });
          continue;
        }
        usable.push({ def, values });
      }

      if (usable.length > 0) {
        const firstValid = returnDates.findIndex((_, i) => usable.every((u) => u.values[i] !== null));
        if (firstValid >= 0) {
          const observations = returnDates.length - firstValid;
          for (const u of usable) {
            const key = `F:${u.def.id}`;
            map[key] = u.values.slice(firstValid).map((v) => v ?? 0);
            keys.push(key);
            coverage.push({ id: u.def.id, available: true, observations });
          }
        }
      }
    }

    return { returnsByTicker: map, dates: commonDates, factorColumnKeys: keys, factorColumnCoverage: coverage };
  }, [dependentTicker, independentTickers, allStockBars, startDate, endDate, returnType, selectedFactorIds, factorData]);

  // Execute linear regression baseline (used directly and as OLS comparison baseline)
  const linearEngineType: 'Standard OLS' | 'Robust Regression' | 'Rolling Regression' =
    regressionType === 'TCN-Attention Sequence' ? 'Standard OLS' : regressionType;
  const regressionIndependentTickers = React.useMemo(
    () => [...independentTickers, ...factorColumnKeys],
    [independentTickers, factorColumnKeys]
  );
  const regressionSummary = React.useMemo(() => {
    return performRegressionAnalysis(
      returnsByTicker,
      dates,
      dependentTicker,
      regressionIndependentTickers,
      linearEngineType,
      rollingWindow
    );
  }, [returnsByTicker, dates, dependentTicker, regressionIndependentTickers, linearEngineType, rollingWindow]);

  // Multi-asset TCN-Attention Sequence model (1d / 5d / 22d, cached)
  const indepKey = independentTickers.join(',');
  const yLen = returnsByTicker[dependentTicker]?.length ?? 0;
  const lastDate = dates[dates.length - 1] ?? '';
  const firstDate = dates[0] ?? '';

  const [tcnReg, setTcnReg] = useState<TcnMultiAssetRegressionResult | null>(() =>
    yLen >= 70
      ? getCachedTcnRegressionForecast(dependentTicker, independentTickers, returnsByTicker, dates)
      : null
  );
  const [tcnTraining, setTcnTraining] = useState(false);
  // nonce 0 means no user request yet; a request always forces a fresh fit.
  const [trainRequest, setTrainRequest] = useState<{ nonce: number; force: boolean }>({
    nonce: 0,
    force: false,
  });
  const returnsRef = useRef(returnsByTicker);
  const datesRef = useRef(dates);
  const indepRef = useRef(independentTickers);
  const regressionTypeRef = useRef(regressionType);
  const trainingRef = useRef(false);
  useEffect(() => {
    returnsRef.current = returnsByTicker;
    datesRef.current = dates;
    indepRef.current = independentTickers;
  }, [returnsByTicker, dates, independentTickers]);
  useEffect(() => {
    regressionTypeRef.current = regressionType;
  }, [regressionType]);

  // Cache hydration follows data/ticker changes — it never starts a fit.
  useEffect(() => {
    if (yLen < 70) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled || trainingRef.current) return;
      const cached = getCachedTcnRegressionForecast(
        dependentTicker,
        indepRef.current,
        returnsRef.current,
        datesRef.current
      );
      setTcnReg((prev) => {
        if (cached) return cached;
        // Live ticks change the signature: keep the last fit visible until retrained.
        return prev &&
          prev.dependentTicker === dependentTicker.toUpperCase() &&
          prev.independentTickers.join(',') === indepRef.current.map((t) => t.toUpperCase()).join(',')
          ? prev
          : null;
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [dependentTicker, indepKey, yLen, firstDate, lastDate]);

  // Training is manual-only — started by the Retrain Sequence Head button.
  useEffect(() => {
    if (trainRequest.nonce === 0) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled || regressionTypeRef.current !== 'TCN-Attention Sequence') return;
      trainingRef.current = true;
      setTcnTraining(true);
      void runTcnMultiAssetRegression(
        dependentTicker,
        indepRef.current,
        returnsRef.current,
        datesRef.current,
        {
          forceRetrain: trainRequest.force,
          yieldControl: () => new Promise<void>((r) => setTimeout(r, 0)),
          isCancelled: () => cancelled,
        }
      ).then((res) => {
        trainingRef.current = false;
        if (!cancelled) {
          if (res) setTcnReg(res);
          setTcnTraining(false);
        }
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      trainingRef.current = false;
    };
  }, [trainRequest, dependentTicker, indepKey]);

  const activeTcnReg =
    tcnReg &&
    tcnReg.dependentTicker === dependentTicker.toUpperCase() &&
    tcnReg.independentTickers.join(',') === independentTickers.slice(0, 6).map((t) => t.toUpperCase()).join(',')
      ? tcnReg
      : null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const list = localInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0);
    if (list.length >= 2) {
      onRegressionTickersChange(list.join(', '));
      onRunRegression(list);
    }
  };

  const isMultiple = independentTickers.length > 1;

  return (
    <div className="space-y-6">
      {/* Parameter Control Card */}
      <div className="panel p-5">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800/80">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-gold-700 dark:text-gold-300">
                  Econometric Factor Analysis
                </span>
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                <span className="text-xs text-zinc-500 dark:text-zinc-400">
                  {isMultiple ? 'Multiple Linear Regression' : 'Simple Linear Regression (CAPM Beta)'}
                </span>
                <ProvenanceBadge daily={barsSource} />
              </div>
              <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5">
                Factor Sensitivity &amp; Residual Econometrics
              </h2>
              <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-zinc-600 dark:text-zinc-300">
                <RegressionModelEquation display="inline" />
                <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
                {regressionType === 'Robust Regression' ? (
                  <RobustIrlsEquation display="inline" />
                ) : (
                  <OlsMatrixEquation display="inline" />
                )}
              </div>
            </div>


            <div className="flex items-center gap-2">
              <ShimmerButton
                type="submit"
                disabled={isLoading}
                className="px-4 py-2"
              >
                {isLoading ? (
                  <>
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                    <span>Fitting Model...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current text-gold-500" />
                    <span>Run Regression</span>
                  </>
                )}
              </ShimmerButton>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            {/* Tickers Input */}
            <div className="lg:col-span-2">
              <label className="block text-zinc-700 dark:text-zinc-300 font-medium mb-1">
                Tickers: <strong className="text-gold-700 dark:text-gold-300">Y (Dependent)</strong>, followed by <strong className="text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300">X₁, X₂, ... (Independent)</strong>
              </label>
              <TickerSuggest
                multiple
                ariaLabel="Regression tickers"
                id="regression-tickers-input"
                value={localInput}
                onChange={setLocalInput}
                placeholder="Search U.S. tickers — Y then X₁, X₂… e.g. AAPL, SPY"
                inputClassName="w-full px-3 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono font-bold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-gold-400"
              />
              <span className="text-[11px] text-zinc-400 block mt-1">
                Active: Dependent <strong>{dependentTicker}</strong> ~ Intercept + {independentTickers.join(' + ')}
              </span>
            </div>

            {/* Regression Type */}
            <AlgoSelect
              label="Regression Engine"
              ariaLabel="Regression engine"
              value={regressionType}
              onChange={(v) =>
                setRegressionType(
                  v as 'Standard OLS' | 'Robust Regression' | 'Rolling Regression' | 'TCN-Attention Sequence'
                )
              }
              options={[
                { value: 'Standard OLS', label: 'Standard OLS', hint: 'closed form' },
                { value: 'Robust Regression', label: 'Robust Regression', hint: 'Huber IRLS' },
                { value: 'Rolling Regression', label: 'Rolling Regression', hint: 'windowed OLS' },
                { value: 'TCN-Attention Sequence', label: 'TCN-Attention Sequence', hint: '1d/5d/22d · nonlinear' },
              ]}
              className="w-full"
            />

            {/* Rolling Window / Settings */}
            <div>
              <label className="block text-zinc-700 dark:text-zinc-300 font-medium mb-1">
                {regressionType === 'Rolling Regression' ? 'Rolling Window (days)' : 'Date Range Filter'}
              </label>
              {regressionType === 'Rolling Regression' ? (
                <input
                  type="number"
                  min="20"
                  max="180"
                  value={rollingWindow}
                  onChange={(e) => setRollingWindow(Math.max(20, parseInt(e.target.value, 10) || 60))}
                  className="w-full px-3 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-gold-400"
                />
              ) : (
                <div className="grid grid-cols-2 gap-1.5">
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => onStartDateChange(e.target.value)}
                    className="px-2 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-zinc-900 dark:text-zinc-100 text-[11px]"
                  />
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => onEndDateChange(e.target.value)}
                    className="px-2 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-zinc-900 dark:text-zinc-100 text-[11px]"
                  />
                </div>
              )}
            </div>
          </div>
        </form>

        {/* Model Presets */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs">
          <span className="text-zinc-400">Benchmark Factor Presets:</span>
          {[
            { label: 'Single Factor (CAPM): AAPL ~ SPY', val: 'AAPL, SPY' },
            { label: 'Two-Factor Tech: NVDA ~ SPY, QQQ', val: 'NVDA, SPY, QQQ' },
            { label: 'Cloud Beta: MSFT ~ GOOGL, SPY', val: 'MSFT, GOOGL, SPY' },
          ].map((preset) => (
            <button
              key={preset.label}
              onClick={() => {
                setLocalInput(preset.val);
                onRegressionTickersChange(preset.val);
                onRunRegression(preset.val.split(',').map((s) => s.trim()));
              }}
              className="px-2.5 py-1 rounded-md border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/60 hover:border-gold-500 text-zinc-700 dark:text-zinc-300 font-mono text-[11px] transition-colors"
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Factor regressors — injected as named columns alongside ticker returns. */}
        <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800/60">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <span className="text-zinc-400 text-xs">Factor Regressors (add as X columns):</span>
            <span className="text-[10px] font-mono text-zinc-500">
              {factorColumnKeys.length} active · {ALL_FACTORS.length} available
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5 mt-2">
            {ALL_FACTORS.map((f) => {
              const active = selectedFactorIds.includes(f.id);
              const unavailable =
                active &&
                factorColumnCoverage.some((c) => c.id === f.id && !c.available);
              return (
                <button
                  key={f.id}
                  type="button"
                  title={`${f.formula} — ${f.rationale}${unavailable ? ' (no data for this symbol)' : ''}`}
                  onClick={() =>
                    setSelectedFactorIds((prev) =>
                      prev.includes(f.id) ? prev.filter((x) => x !== f.id) : [...prev, f.id]
                    )
                  }
                  className={`px-2 py-0.5 rounded font-mono text-[10px] border transition-colors ${
                    active
                      ? unavailable
                        ? 'bg-amber-500/10 border-amber-500/40 text-amber-700 dark:text-amber-400'
                        : 'bg-gold-500/15 border-gold-500/50 text-gold-700 dark:text-gold-300'
                      : 'bg-zinc-50 dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 text-zinc-500 hover:border-gold-500'
                  }`}
                >
                  {f.label}
                </button>
              );
            })}
          </div>
          {factorColumnCoverage.some((c) => !c.available) && (
            <p className="mt-1.5 text-[10px] text-amber-600 dark:text-amber-400">
              Some selected factors have no data for {dependentTicker} in this range and were omitted. Fundamental
              history begins at first observation; analyst/option inputs are current snapshots.
            </p>
          )}
        </div>
      </div>

      {/* Observation Availability Notice */}
      {regressionSummary.nObservations === 0 && (
        <div className="p-4 bg-gold-500/10 border border-gold-500/20 rounded-xl text-xs text-gold-700 dark:text-gold-400 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 flex-shrink-0" />
            <span>
              {isLoading
                ? `Retrieving and aligning price series for ${dependentTicker} and ${independentTickers.join(', ')}...`
                : `Insufficient overlapping price observations for ${dependentTicker} and ${independentTickers.join(', ')} in selected date range. Click 'Run Regression' to fetch price bars.`}
            </span>
          </div>
        </div>
      )}

      {/* Multi-Asset Nonlinear Sequence Panel (TCN-Attention 1d / 5d / 22d) */}
      {(regressionType === 'TCN-Attention Sequence' || activeTcnReg) && (
        <div className="panel p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
            <div>
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-gold-500" />
                <span className="kicker">TCN-Attention Multi-Asset Sequence Forecasting</span>
                {activeTcnReg && (
                  <span className="font-mono text-[10px] text-zinc-400 tabular-nums">
                    · {activeTcnReg.cacheHit ? 'cached' : `${activeTcnReg.trainingDurationMs}ms`} · RF {activeTcnReg.receptiveField}d · {activeTcnReg.parameterCount} params
                    {activeTcnReg.hmmRegime
                      ? ` · HMM ${activeTcnReg.hmmRegime.dominantLabel} (${Math.round(activeTcnReg.hmmRegime.probs[0] * 100)}/${Math.round(activeTcnReg.hmmRegime.probs[1] * 100)}/${Math.round(activeTcnReg.hmmRegime.probs[2] * 100)}%)`
                      : ''}
                  </span>
                )}
              </div>
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm mt-0.5">
                Nonlinear Cross-Asset Sequence Forecast: {dependentTicker} ← [{independentTickers.join(', ')}] (1d · 5d · 22d)
              </h3>
            </div>
            <button
              type="button"
              onClick={() => setTrainRequest((prev) => ({ nonce: prev.nonce + 1, force: true }))}
              disabled={tcnTraining || yLen < 70}
              className="inline-flex cursor-pointer items-center gap-1.5 self-start sm:self-auto rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[10px] font-semibold text-zinc-700 hover:border-gold-500/50 dark:text-zinc-300 disabled:opacity-50"
            >
              <RotateCcw className={`h-3.5 w-3.5 ${tcnTraining ? 'animate-spin' : ''}`} />
              {tcnTraining ? 'Training…' : activeTcnReg ? 'Retrain Sequence Head' : 'Train Sequence Head'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {([1, 5, 22] as const).map((h) => {
              const hz = activeTcnReg?.horizons[h];
              const lift = hz ? hz.nonlinearLiftR2 : null;
              return (
                <div key={h} className="panel-inset p-3.5 border border-[var(--hairline)]">
                  <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-zinc-400">
                    <span className="font-semibold text-gold-700 dark:text-gold-300">{h}D Cumulative Return</span>
                    <span>OOS n={hz?.holdoutCount ?? 0}</span>
                  </div>
                  <div className="mt-1.5 flex items-baseline justify-between font-mono tabular-nums">
                    <span className={`text-xl font-bold ${hz && hz.predictedReturn >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400'}`}>
                      {hz ? `${hz.predictedReturn >= 0 ? '+' : ''}${(hz.predictedReturn * 100).toFixed(2)}%` : tcnTraining ? '…' : '—'}
                    </span>
                    <span className="text-[10px] text-zinc-500 dark:text-zinc-400">
                      {hz ? `99% [${(hz.lower99 * 100).toFixed(1)}%, ${(hz.upper99 * 100).toFixed(1)}%]` : '—'}
                    </span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-[var(--hairline)] grid grid-cols-3 gap-2 font-mono text-[10px] tabular-nums">
                    <div>
                      <span className="text-zinc-400 block">OOS R²</span>
                      <span className="font-semibold text-zinc-800 dark:text-zinc-100">{hz ? `${(hz.oosR2 * 100).toFixed(1)}%` : '—'}</span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block">Lift vs OLS</span>
                      <span className={`font-semibold ${lift !== null && lift >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500'}`}>
                        {lift !== null ? `${lift >= 0 ? '+' : ''}${(lift * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div>
                      <span className="text-zinc-400 block">Dir Hit</span>
                      <span className="font-semibold text-gold-700 dark:text-gold-300">{hz ? `${hz.directionalAccuracyPct.toFixed(0)}%` : '—'}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {activeTcnReg && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="panel-inset p-3.5 border border-[var(--hairline)]">
                <div className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 mb-2">
                  Cross-Asset Permutation Factor Importance (Holdout RMSE Impact)
                </div>
                <div className="space-y-2">
                  {activeTcnReg.factorImportance.map((f) => (
                    <div key={f.ticker} className="flex items-center gap-3 font-mono text-[11px]">
                      <span
                        className={`w-28 shrink-0 font-semibold truncate ${
                          f.isHmmRegime
                            ? 'text-gold-700 dark:text-gold-300'
                            : 'text-zinc-800 dark:text-zinc-200'
                        }`}
                      >
                        {f.ticker}{f.isDependentLag ? ' (lag)' : f.isHmmRegime ? ' (3s)' : ''}
                      </span>
                      <div className="h-2.5 flex-1 overflow-hidden rounded-sm bg-zinc-200/70 dark:bg-zinc-800">
                        <div
                          className={`h-full rounded-sm ${f.isHmmRegime ? 'bg-emerald-500/80' : 'bg-gold-500/80'}`}
                          style={{ width: `${Math.max(3, f.importancePct).toFixed(1)}%` }}
                        />
                      </div>
                      <span className="w-24 shrink-0 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                        {f.importancePct.toFixed(1)}% <span className="text-zinc-400">(+{f.rmseIncreaseBps.toFixed(1)}bp)</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel-inset p-3.5 border border-[var(--hairline)]">
                <div className="font-mono text-[10px] uppercase tracking-wider text-zinc-400 mb-2">
                  Causal Self-Attention Transmission Dates
                </div>
                <div className="space-y-1.5 font-mono text-[11px]">
                  {activeTcnReg.topTransmissionDates.map((d) => (
                    <div key={d.date} className="flex items-center justify-between border-b border-[var(--hairline)] last:border-0 pb-1 last:pb-0 tabular-nums">
                      <span className="text-zinc-700 dark:text-zinc-200 font-semibold">
                        {d.date} <span className="font-normal text-zinc-400">(lag t-{d.lag})</span>
                      </span>
                      <span className="text-gold-700 dark:text-gold-300 font-semibold">
                        attn {(d.weight * 100).toFixed(1)}% · {d.metricLabel} {d.metricValue >= 0 ? '+' : ''}{d.metricValue.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-2 font-mono text-[9.5px] text-zinc-400">
                  Evaluated strictly out-of-sample on the trailing chronological holdout split against a lag-matched OLS linear baseline.
                </p>
              </div>
            </div>
          )}

          {activeTcnReg && <TcnRegressionOosChart tcnReg={activeTcnReg} />}
        </div>
      )}

      {/* Model Fit Diagnostics Summary Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
          <span className="text-xs text-zinc-500 dark:text-zinc-400 block">
            R-Squared (<MathSymbol name="r_squared" />)
          </span>
          <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {(regressionSummary.rSquared * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            Variance explained
          </span>
        </div>

        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
          <span className="text-xs text-zinc-500 dark:text-zinc-400 block">
            Adjusted <MathSymbol name="adj_r_squared" />
          </span>
          <span className="font-mono text-xl font-bold text-gold-600 dark:text-gold-300 tabular-nums">
            {(regressionSummary.adjRSquared * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            Degrees of freedom adj.
          </span>
        </div>

        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
          <span className="text-xs text-zinc-500 dark:text-zinc-400 block">F-Statistic</span>
          <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {regressionSummary.fStatistic.toFixed(1)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            Joint significance · p-value not computed
          </span>
        </div>

        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
          <span className="text-xs text-zinc-500 dark:text-zinc-400 block">Akaike / Bayesian</span>
          <span className="font-mono text-sm font-bold text-zinc-900 dark:text-zinc-100 tabular-nums">
            {regressionSummary.aic.toFixed(0)} / {regressionSummary.bic.toFixed(0)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            Model selection AIC/BIC
          </span>
        </div>

        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl">
          <span className="text-xs text-zinc-500 dark:text-zinc-400 block">Durbin-Watson</span>
          <span className="font-mono text-xl font-bold text-gold-500 tabular-nums">
            {regressionSummary.durbinWatson.toFixed(2)}
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            {regressionSummary.durbinWatson >= 1.5 && regressionSummary.durbinWatson <= 2.5
              ? 'near 2 · no strong serial correlation'
              : 'departs from 2 · inspect residuals'}
          </span>
        </div>

        <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 p-4 rounded-xl bg-gold-500/20 dark:bg-gold-500/20 border-gold-500 dark:border-gold-500/40">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gold-700 dark:text-gold-300 block font-semibold">
              {factorColumnKeys.length > 0 ? 'Next-Day Forecast (factor inputs held at t)' : 'Next-Day Forecast'}
            </span>
            <ExpectedReturnEquation display="inline" className="text-[10px]" />
          </div>
          <span
            className={`font-mono text-xl font-bold tabular-nums ${
              regressionSummary.forecastNextDay.expectedReturn >= 0
                ? 'text-gold-500'
                : 'text-rose-500'
            }`}
          >
            {(regressionSummary.forecastNextDay.expectedReturn * 100).toFixed(2)}%
          </span>
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400 block mt-0.5">
            95% CI: [{(regressionSummary.forecastNextDay.predictionInterval95[0] * 100).toFixed(1)}%,{' '}
            {(regressionSummary.forecastNextDay.predictionInterval95[1] * 100).toFixed(1)}%]
          </span>
          {factorColumnKeys.length > 0 && (
            <span className="text-[10px] text-amber-700 dark:text-amber-400 block mt-1">
              Factor columns are levels, not returns — this forecast plugs their latest values into the fitted
              coefficients and is not a return prediction.
            </span>
          )}
        </div>
      </div>

      {/* Regression Coefficients & VIF Table */}
      <div className="panel p-5 overflow-x-auto">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
              Parameter Estimates, Significance &amp; Multicollinearity (VIF)
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Dependent Variable: {dependentTicker} · Benchmark: {independentTickers.join(', ')} · Observations: {regressionSummary.nObservations} · aligned on available dates
            </p>
          </div>
          <span className="text-xs font-mono text-zinc-500 dark:text-zinc-400">
            Regression Error s: {(regressionSummary.standardErrorOfRegression * 100).toFixed(3)}%
          </span>
        </div>

        <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-800 text-xs font-mono">
          <thead>
            <tr>
              <th className="py-2.5 px-3 text-left font-semibold text-zinc-500">Variable</th>
              <th className="py-2.5 px-3 text-right font-semibold text-zinc-500">
                Coefficient (<MathSymbol name="beta" />)
              </th>
              <th className="py-2.5 px-3 text-right font-semibold text-zinc-500">Std Error</th>
              <th className="py-2.5 px-3 text-right font-semibold text-zinc-500">t-Statistic</th>
              <th className="py-2.5 px-3 text-right font-semibold text-zinc-500">p-Value (P&gt;|t|)</th>
              <th className="py-2.5 px-3 text-right font-semibold text-zinc-500">
                VIF (<VifEquation display="inline" className="text-[10px]" />)
              </th>
            </tr>
          </thead>

          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
            {regressionSummary.coefficients.map((row) => {
              const isSignificant = row.pValue < 0.05;
              const hasHighVif = row.vif !== null && row.vif > 5;

              return (
                <tr key={row.variable} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                  <td className="py-2.5 px-3 font-bold text-zinc-900 dark:text-zinc-100">
                    {row.variable.startsWith('F:') ? (
                      <span className="inline-flex items-center gap-1.5">
                        <span>{FACTOR_BY_ID[row.variable.slice(2)]?.label ?? row.variable.slice(2)}</span>
                        <span className="text-[9px] font-mono text-gold-600 dark:text-gold-400">factor</span>
                      </span>
                    ) : (
                      row.variable
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-right font-bold tabular-nums text-zinc-900 dark:text-zinc-100">
                    {row.coefficient.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums text-zinc-600 dark:text-zinc-400">
                    {row.stdError.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-semibold text-zinc-800 dark:text-zinc-200">
                    {row.tStatistic.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-bold">
                    <span
                      className={
                        isSignificant
                          ? 'text-gold-600 dark:text-gold-400'
                          : 'text-zinc-400'
                      }
                    >
                      {row.pValue < 0.0001 ? '< 0.0001 ***' : `${row.pValue.toFixed(4)} ${isSignificant ? '**' : ''}`}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right tabular-nums font-semibold">
                    {row.vif !== null ? (
                      <span
                        className={
                          hasHighVif
                            ? 'text-rose-500 font-bold'
                            : 'text-zinc-700 dark:text-zinc-300'
                        }
                      >
                        {row.vif.toFixed(2)} {hasHighVif ? '(High Collinearity)' : ''}
                      </span>
                    ) : (
                      <span className="text-zinc-400">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 mt-3 pt-2 border-t border-zinc-100 dark:border-zinc-800">
          <span>* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</span>
          <span>VIF &lt; 5 indicates healthy absence of severe multicollinearity</span>
        </div>
      </div>

      {/* Residual Analysis Diagnostics Charts */}
      <ChartFullscreen title="Regression Residual Diagnostics">
        <ResidualDiagnosticsChart summary={regressionSummary} />
      </ChartFullscreen>

      {/* Cross-sectional factor research — rankings, z-scores, composite and IC */}
      <FactorResearchPanel
        symbols={universeSymbols}
        allStockBars={allStockBars}
        factorData={factorData}
        onSymbolsChange={setUniverseSymbols}
        startDate={startDate}
        endDate={endDate}
        barsSource={barsSource}
      />
    </div>
  );
}
