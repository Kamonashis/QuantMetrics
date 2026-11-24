'use client';

import React, { useMemo, useState } from 'react';
import {
  Activity,
  ArrowRight,
  BarChart2,
  CandlestickChart,
  ChevronDown,
  LineChart,
  Network,
  Radar,
  Sigma,
  Waves,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { MovingBorderButton } from '@/components/ui/moving-border';
import { TickerTape, type TapeItem } from '@/components/ui/ticker-tape';
import { GarchEquation, CorrelationEquation } from '@/components/math/MathEquations';
import { HomeHeroRegimeChart } from '@/components/charts/HomeHeroRegimeChart';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import {
  calculateReturns,
  computeEwmaVolatility,
  fitGarch11,
  type PriceBar,
} from '@/lib/quant-engine';
import {
  buildSparklineSvgPath,
  computeActiveCausalHmmSeries,
  computeHomeUniverseCard,
  HOME_UNIVERSE_SYMBOLS,
} from '@/lib/home-universe';
import { getHmmRegimeToken } from '@/lib/hmm-regime-overlay';

export type HomeDestinationView =
  | 'now'
  | 'analysis'
  | 'modeling'
  | 'volatility'
  | 'fno'
  | 'correlation'
  | 'regression';

interface HomeViewProps {
  onNavigate: (view: HomeDestinationView) => void;
  onSelectPreset: (ticker: string) => void;
  ticker: string;
  companyName?: string;
  bars: PriceBar[];
  barsSource?: string | null;
  tapeItems: TapeItem[];
}

interface WorkstationItem {
  num: string;
  view: HomeDestinationView;
  title: string;
  subtitle: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  featured?: boolean;
}

const WORKSTATIONS: readonly WorkstationItem[] = [
  {
    num: '01',
    view: 'now',
    title: 'Edge & TCN Diagnostics',
    subtitle: 'Multi-Domain TCN-Attention · Causal HMM Regimes · Tactical Signal Board',
    description:
      'Live tactical summary combining causal 3-state volatility regimes, multi-domain TCN-Attention forecasts, out-of-sample regime scorecards, and options edge triggers.',
    icon: Radar,
    featured: true,
  },
  {
    num: '02',
    view: 'analysis',
    title: 'Single-Stock Analysis',
    subtitle: 'Returns · Tail Risk · Stationarity',
    description:
      'ADF, KPSS, and Hurst diagnostics paired with five-model VaR/Expected Shortfall backtesting, EVT peaks-over-threshold, and MLE distribution fits.',
    icon: BarChart2,
  },
  {
    num: '03',
    view: 'modeling',
    title: 'Volatility Modeling',
    subtitle: 'GARCH(1,1) · EWMA · HAR-RV',
    description:
      'Realized volatility estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang), GARCH persistence calibration, and multi-horizon variance cones.',
    icon: Activity,
  },
  {
    num: '04',
    view: 'volatility',
    title: 'Implied Volatility Suite',
    subtitle: 'SVI Surface · Term Structure · VRP',
    description:
      'European BSM implied volatility smiles, SVI parameterization, model-free IV, variance risk premium spread, and regime-conditioned vol signals.',
    icon: Waves,
  },
  {
    num: '05',
    view: 'fno',
    title: 'Options F&O Positioning',
    subtitle: 'GEX · Dealer Flip · OI Walls',
    description:
      'Dealer gamma exposure profiles, assumed squeeze/inverse zero-gamma flip levels, call/put open-interest walls, and trap state-machine telemetry.',
    icon: CandlestickChart,
  },
  {
    num: '06',
    view: 'correlation',
    title: 'Correlation & Covariance',
    subtitle: 'Ledoit–Wolf · RMT · GMV',
    description:
      'Marchenko–Pastur eigenvalue denoising, shrinkage covariance estimation, minimum spanning trees, and global minimum-variance portfolio weights.',
    icon: Network,
  },
  {
    num: '07',
    view: 'regression',
    title: 'Factor & Regression Lab',
    subtitle: 'Style Factors · Rolling Exposures · Newey–West',
    description:
      'Explain any stock’s returns with market and style-factor exposures, see what drove its risk and return, test whether the exposures held up, and keep forecasting in a separate out-of-sample lab.',
    icon: LineChart,
  },
];

export function HomeView({
  onNavigate,
  onSelectPreset,
  ticker,
  companyName,
  bars,
  barsSource,
  tapeItems,
}: HomeViewProps) {
  const [equationsOpen, setEquationsOpen] = useState(false);
  const { allStockBars, latestQuotes, rv20, vix, chain, chainLoading } = useTickerSnapshot();

  // Core econometric summary on active ticker's daily bars
  const activeStats = useMemo(() => {
    if (!bars || bars.length < 60) return null;
    const rets = calculateReturns(
      bars.map((b) => b.close),
      'log'
    );
    if (rets.length < 50) return null;
    const ewma = computeEwmaVolatility(rets, 0.94).summary.latestAnnualizedVol;
    const garch = fitGarch11(rets);
    const sorted = [...rets].sort((a, b) => a - b);
    const q01 = sorted[Math.floor(0.01 * (sorted.length - 1))] ?? 0;
    return {
      ewma,
      persistence: garch.summary.persistence,
      historicalVar99: Math.max(0, -q01),
    };
  }, [bars]);

  // Strictly causal 3-state HMM series for the interactive Hero chart & Pulse Strip
  const activeHmm = useMemo(() => computeActiveCausalHmmSeries(bars, 504), [bars]);

  const tapeBySymbol = useMemo(() => {
    const map = new Map<string, TapeItem>();
    for (const item of tapeItems) {
      map.set(item.symbol.toUpperCase(), item);
    }
    return map;
  }, [tapeItems]);

  const activeTickerUpper = ticker.toUpperCase();
  const activeLiveQuote = latestQuotes[activeTickerUpper];
  const activeTape = tapeBySymbol.get(activeTickerUpper);
  const activePending = !activeLiveQuote && Boolean(activeTape?.pending);
  const resolvedCompanyName =
    companyName ||
    HOME_UNIVERSE_SYMBOLS.find((m) => m.ticker === activeTickerUpper)?.name ||
    `${activeTickerUpper} Equity`;

  const activeSpotPrice = !activePending
    ? activeLiveQuote?.price ??
      activeTape?.price ??
      (bars.length > 0 ? bars[bars.length - 1].close : null)
    : null;
  const activeSpotChangePct = !activePending
    ? activeLiveQuote?.changePct ??
      activeTape?.changePct ??
      (bars.length >= 2 && bars[bars.length - 2].close > 0
        ? ((bars[bars.length - 1].close - bars[bars.length - 2].close) /
            bars[bars.length - 2].close) *
          100
        : null)
    : null;

  // Guard IV30 against cross-ticker staleness while options chain loads for a newly selected symbol
  const chainMatchesActive = Boolean(
    chain &&
      chain.available &&
      (chain.requested?.toUpperCase() === activeTickerUpper ||
        chain.symbol?.toUpperCase() === activeTickerUpper)
  );
  const iv30 =
    chainMatchesActive && typeof chain?.iv30 === 'number' && Number.isFinite(chain.iv30)
      ? chain.iv30
      : null;

  // Direct basis comparison (RV sqrt(252) vs IV sqrt(365) — never rescaled)
  const volSpreadPts = iv30 !== null && rv20 !== null ? (iv30 - rv20) * 100 : null;
  const vrpBp2 = iv30 !== null && rv20 !== null ? (iv30 * iv30 - rv20 * rv20) * 10000 : null;

  // 10-ticker live watchboard cards (sparkline + RV20 + causal HMM state)
  const universeCards = useMemo(() => {
    return HOME_UNIVERSE_SYMBOLS.map((meta) => {
      const symbolBars =
        allStockBars[meta.ticker] ??
        (meta.ticker === activeTickerUpper ? bars : undefined);
      const liveQuote = latestQuotes[meta.ticker];
      const tapeItem = tapeBySymbol.get(meta.ticker);
      return computeHomeUniverseCard(meta, symbolBars, liveQuote, tapeItem);
    });
  }, [allStockBars, activeTickerUpper, bars, latestQuotes, tapeBySymbol]);

  const activeHmmToken = activeHmm ? getHmmRegimeToken(activeHmm.latestState) : null;

  return (
    <div className="space-y-7 pb-10">
      {/* =====================================================================
          BAND 1 — EDITORIAL HERO SPLIT + INTERACTIVE LIVE PRICE & HMM CHART
      ===================================================================== */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch pt-1">
        {/* Left: Editorial Framing & Primary Entry Points */}
        <div className="lg:col-span-5 flex flex-col justify-between gap-6 py-1">
          <div className="space-y-4">
            <div className="kicker">
              Live SSE Feeds · GARCH &amp; TCN-Attention · 3-State HMM
            </div>

            <h1 className="font-display font-semibold tracking-tight text-balance text-zinc-950 dark:text-zinc-50 text-[clamp(2.1rem,3.4vw,3.25rem)] leading-[1.06]">
              Quantitative risk, computed in the open.
            </h1>

            <p className="text-sm sm:text-[15px] leading-relaxed text-zinc-600 dark:text-zinc-400 max-w-[46ch]">
              Real-time equity and volatility analytics — strictly causal HMM regime filtering,
              GARCH(1,1) persistence, implied volatility surfaces, and five-model VaR backtesting.
            </p>

            <div className="pt-2 flex flex-wrap items-center gap-3">
              <MovingBorderButton
                onClick={() => onNavigate('now')}
                className="text-xs sm:text-sm font-medium whitespace-nowrap shrink-0"
              >
                <span>Open Edge Workstation</span>
                <ArrowRight className="w-4 h-4 shrink-0" />
              </MovingBorderButton>

              <button
                type="button"
                onClick={() => onNavigate('analysis')}
                className="inline-flex h-11 items-center gap-2 px-5 rounded-xl text-[13px] font-medium border border-[var(--hairline-strong)] bg-[var(--bg-panel)] text-zinc-800 dark:text-zinc-200 hover:border-gold-500/50 hover:text-zinc-950 dark:hover:text-gold-300 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <span>Single-Stock Analysis</span>
                <ArrowRight className="w-3.5 h-3.5 shrink-0" />
              </button>
            </div>
          </div>

          {/* Quiet Typographic Proof Strip (zero boxed pills) */}
          <div className="pt-4 border-t border-[var(--hairline)] flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
            <span>28 Econometric Tests</span>
            <span className="text-zinc-300 dark:text-zinc-700">/</span>
            <span>5-Model VaR</span>
            <span className="text-zinc-300 dark:text-zinc-700">/</span>
            <span>7 Live Workstations</span>
          </div>
        </div>

        {/* Right: Live Interactive Price & Causal HMM Volatility Regime Chart */}
        <div className="lg:col-span-7 flex">
          <HomeHeroRegimeChart
            ticker={activeTickerUpper}
            companyName={resolvedCompanyName}
            bars={bars}
            barsSource={barsSource}
            livePrice={activeLiveQuote?.price ?? activeTape?.price ?? null}
            liveChangePct={activeLiveQuote?.changePct ?? activeTape?.changePct ?? null}
            liveAsOf={activeLiveQuote?.asOf ?? activeTape?.asOf ?? null}
            pending={activePending}
            hmmSeries={activeHmm}
            garchPersistence={activeStats?.persistence ?? null}
            ewmaVol={activeStats?.ewma ?? null}
            var99={activeStats?.historicalVar99 ?? null}
          />
        </div>
      </section>

      {/* =====================================================================
          BAND 2 — LIVE TICKER TAPE + 5-METRIC CROSS-ASSET PULSE STRIP
      ===================================================================== */}
      <section className="space-y-3" aria-label="Live market strip and cross-asset pulse">
        <TickerTape items={tapeItems} />

        <div className="panel overflow-hidden">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 divide-y sm:divide-y-0 sm:divide-x divide-[var(--hairline)]">
            {/* 1. Active Spot */}
            <div className="px-4 py-3.5">
              <div className="text-[10px] font-mono uppercase tracking-[0.09em] text-zinc-500 dark:text-zinc-400">
                Active Spot · {activeTickerUpper}
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {activeSpotPrice != null ? `$${activeSpotPrice.toFixed(2)}` : '—'}
                </span>
                {activeSpotChangePct != null && (
                  <span
                    className={`font-mono text-xs font-medium tabular-nums ${
                      activeSpotChangePct >= 0
                        ? 'text-[var(--positive)]'
                        : 'text-[var(--negative)]'
                    }`}
                  >
                    {activeSpotChangePct >= 0 ? '+' : ''}
                    {activeSpotChangePct.toFixed(2)}%
                  </span>
                )}
              </div>
              <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 truncate">
                {resolvedCompanyName}
              </div>
            </div>

            {/* 2. RV20 vs IV30 Spread */}
            <div className="px-4 py-3.5">
              <div className="text-[10px] font-mono uppercase tracking-[0.09em] text-zinc-500 dark:text-zinc-400">
                RV20 vs. IV30 Spread
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {volSpreadPts !== null
                    ? `${volSpreadPts >= 0 ? '+' : ''}${volSpreadPts.toFixed(1)} pts`
                    : rv20 !== null
                      ? `${(rv20 * 100).toFixed(1)}% RV`
                      : '—'}
                </span>
                {vrpBp2 !== null && (
                  <span
                    className={`font-mono text-[11px] tabular-nums ${
                      vrpBp2 >= 0 ? 'text-gold-700 dark:text-gold-300' : 'text-[var(--negative)]'
                    }`}
                  >
                    {vrpBp2 >= 0 ? '+' : ''}
                    {Math.round(vrpBp2)} bp²
                  </span>
                )}
              </div>
              <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
                {rv20 !== null ? `RV ${(rv20 * 100).toFixed(1)}%` : 'RV —'}
                {' · '}
                {iv30 !== null
                  ? `IV ${(iv30 * 100).toFixed(1)}%`
                  : chainLoading
                    ? 'IV loading…'
                    : 'IV n/a'}
              </div>
            </div>

            {/* 3. CBOE VIX */}
            <div className="px-4 py-3.5">
              <div className="text-[10px] font-mono uppercase tracking-[0.09em] text-zinc-500 dark:text-zinc-400">
                CBOE VIX Index
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-mono text-lg font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                  {vix.level != null ? vix.level.toFixed(2) : '—'}
                </span>
                {vix.percentile != null && (
                  <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400 tabular-nums">
                    P{Math.round(vix.percentile)}
                  </span>
                )}
              </div>
              <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
                {vix.percentile != null
                  ? `${vix.n}-session 1Y percentile`
                  : 'Implied 30d S&P 500 vol'}
              </div>
            </div>

            {/* 4. Dominant Causal HMM Regime */}
            <div className="px-4 py-3.5">
              <div className="text-[10px] font-mono uppercase tracking-[0.09em] text-zinc-500 dark:text-zinc-400">
                Causal HMM Regime
              </div>
              <div className="mt-1 flex items-center gap-2">
                {activeHmmToken ? (
                  <>
                    <span
                      className="inline-block w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: activeHmmToken.color }}
                    />
                    <span className={`font-mono text-lg font-semibold ${activeHmmToken.textClass}`}>
                      {activeHmmToken.label}
                    </span>
                  </>
                ) : (
                  <span className="font-mono text-lg font-semibold text-zinc-400">—</span>
                )}
              </div>
              <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
                {activeHmm
                  ? `${(activeHmm.latestProb * 100).toFixed(1)}% forward posterior`
                  : 'Requires ≥45 daily bars'}
              </div>
            </div>

            {/* 5. 1d 99% Historical VaR */}
            <div className="px-4 py-3.5 col-span-2 sm:col-span-1">
              <div className="text-[10px] font-mono uppercase tracking-[0.09em] text-zinc-500 dark:text-zinc-400">
                1d 99% Hist. VaR
              </div>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="font-mono text-lg font-semibold text-[var(--negative)] tabular-nums">
                  {activeStats ? `-${(activeStats.historicalVar99 * 100).toFixed(2)}%` : '—'}
                </span>
              </div>
              <div className="mt-0.5 text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
                {activeStats
                  ? `EWMA ${(activeStats.ewma * 100).toFixed(1)}% ann.`
                  : '1% empirical tail loss'}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* =====================================================================
          BAND 3 — INTERACTIVE 10-TICKER MARKET GRID (SPARKLINES + REGIMES)
      ===================================================================== */}
      <section className="space-y-3" aria-label="10-ticker live market grid">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="kicker">Market Universe</div>
            <h2 className="mt-0.5 font-display text-lg sm:text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              Live 10-benchmark watchboard
            </h2>
          </div>
          <p className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
            Select any card to update the Hero chart · Analyze opens Single-Stock view
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {universeCards.map((card) => {
            const isActive = card.ticker === activeTickerUpper;
            const spark = buildSparklineSvgPath(card.sparkline, 120, 32, 3);
            const strokeColor = card.sparklinePositive ? 'var(--positive)' : 'var(--negative)';
            const regimeTok =
              card.regimeState !== null ? getHmmRegimeToken(card.regimeState) : null;
            const sparkGradId = `home-spark-${card.ticker}`;

            return (
              <div
                key={card.ticker}
                role="button"
                tabIndex={0}
                onClick={() => onSelectPreset(card.ticker)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectPreset(card.ticker);
                  }
                }}
                aria-pressed={isActive}
                className={`panel p-3.5 text-left transition-colors cursor-pointer flex flex-col justify-between gap-2.5 group focus:outline-none focus-visible:ring-1 focus-visible:ring-gold-500 ${
                  isActive
                    ? '!border-gold-500/70 dark:!border-gold-400/60 bg-gold-500/[0.03] dark:bg-gold-500/[0.04]'
                    : 'hover:border-[var(--hairline-strong)]'
                }`}
              >
                {/* Row 1: Symbol + Live Price */}
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                        {card.ticker}
                      </span>
                      {isActive && (
                        <span className="font-mono text-[9.5px] uppercase tracking-wider text-gold-700 dark:text-gold-300">
                          Active
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-zinc-500 dark:text-zinc-400 truncate">
                      {card.name}
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <div className="font-mono text-sm font-semibold text-zinc-900 dark:text-zinc-100 tabular-nums">
                      {!card.pending && card.price != null ? `$${card.price.toFixed(2)}` : '—'}
                    </div>
                    <div
                      className={`font-mono text-[11px] font-medium tabular-nums ${
                        card.pending || card.changePct == null
                          ? 'text-zinc-400'
                          : card.changePct >= 0
                            ? 'text-[var(--positive)]'
                            : 'text-[var(--negative)]'
                      }`}
                    >
                      {!card.pending && card.changePct != null
                        ? `${card.changePct >= 0 ? '+' : ''}${card.changePct.toFixed(2)}%`
                        : '—'}
                    </div>
                  </div>
                </div>

                {/* Row 2: 60-Session Real Sparkline */}
                <div className="h-8 w-full">
                  {spark.linePath ? (
                    <svg
                      viewBox="0 0 120 32"
                      preserveAspectRatio="none"
                      className="w-full h-full block overflow-visible"
                      aria-hidden="true"
                    >
                      <defs>
                        <linearGradient id={sparkGradId} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={strokeColor} stopOpacity="0.2" />
                          <stop offset="100%" stopColor={strokeColor} stopOpacity="0" />
                        </linearGradient>
                      </defs>
                      <path d={spark.areaPath} fill={`url(#${sparkGradId})`} />
                      <path
                        d={spark.linePath}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : (
                    <div className="h-full border-b border-dashed border-[var(--hairline)]" />
                  )}
                </div>

                {/* Row 3: RV20 + Causal HMM Regime + Explicit Analyze Link */}
                <div className="pt-2 border-t border-[var(--hairline)] flex items-center justify-between gap-2 text-[10.5px] font-mono">
                  <div className="flex items-center gap-2 min-w-0 truncate">
                    <span className="text-zinc-500 dark:text-zinc-400 tabular-nums shrink-0">
                      {card.rv20 != null ? `RV ${(card.rv20 * 100).toFixed(0)}%` : 'RV —'}
                    </span>
                    {regimeTok && (
                      <>
                        <span className="text-zinc-300 dark:text-zinc-700">·</span>
                        <span className={`truncate font-medium ${regimeTok.textClass}`}>
                          {regimeTok.label}
                        </span>
                      </>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectPreset(card.ticker);
                      onNavigate('analysis');
                    }}
                    className="inline-flex items-center gap-0.5 text-zinc-600 dark:text-zinc-300 hover:text-gold-700 dark:hover:text-gold-300 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <span>Analyze</span>
                    <ArrowRight className="w-3 h-3 shrink-0" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* =====================================================================
          BAND 4 — UNIFIED 7-WORKSTATION LAUNCHER (ASYMMETRIC EDITORIAL GRID)
      ===================================================================== */}
      <section className="space-y-3" aria-label="Analytical workstations">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="kicker">Workstations</div>
            <h2 className="mt-0.5 font-display text-lg sm:text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
              Seven analytical surfaces
            </h2>
          </div>
          <p className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400">
            Shared ticker context across every view
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {WORKSTATIONS.map((ws) => {
            const Icon = ws.icon;
            return (
              <button
                key={ws.view}
                type="button"
                onClick={() => onNavigate(ws.view)}
                className={`panel p-5 text-left transition-colors hover:border-gold-500/50 dark:hover:border-gold-400/45 group flex flex-col justify-between gap-4 cursor-pointer focus:outline-none focus-visible:ring-1 focus-visible:ring-gold-500 ${
                  ws.featured ? 'md:col-span-2' : ''
                }`}
              >
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2.5">
                      <span className="font-mono text-[11px] font-medium text-gold-700 dark:text-gold-400 tabular-nums">
                        {ws.num}
                      </span>
                      <span className="text-zinc-300 dark:text-zinc-700">/</span>
                      <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                        {ws.subtitle}
                      </span>
                    </div>
                    <Icon className="w-4 h-4 text-zinc-400 dark:text-zinc-500 group-hover:text-gold-600 dark:group-hover:text-gold-300 transition-colors shrink-0" />
                  </div>

                  <h3 className="font-display text-base sm:text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100 group-hover:text-gold-700 dark:group-hover:text-gold-300 transition-colors">
                    {ws.title}
                  </h3>

                  <p className="text-xs sm:text-[13px] leading-relaxed text-zinc-600 dark:text-zinc-400 max-w-[60ch]">
                    {ws.description}
                  </p>
                </div>

                <div className="flex items-center gap-1.5 text-xs font-mono font-medium text-zinc-700 dark:text-zinc-300 group-hover:text-gold-700 dark:group-hover:text-gold-300 transition-colors">
                  <span>Open {ws.title.split(' ')[0]}</span>
                  <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5 shrink-0" />
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* =====================================================================
          BAND 5 — COLLAPSIBLE ECONOMETRIC EQUATIONS REFERENCE
      ===================================================================== */}
      <section aria-label="Econometric equations reference">
        <div className="panel overflow-hidden">
          <button
            type="button"
            onClick={() => setEquationsOpen((prev) => !prev)}
            aria-expanded={equationsOpen}
            className="w-full px-5 py-4 flex items-center justify-between gap-4 text-left hover:bg-[var(--bg-inset)]/60 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-3 min-w-0">
              <Sigma className="w-4 h-4 text-gold-700 dark:text-gold-400 shrink-0" />
              <div className="min-w-0">
                <div className="font-display text-sm sm:text-base font-semibold text-zinc-900 dark:text-zinc-100">
                  Econometric Formulations Reference
                </div>
                <div className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 truncate">
                  GARCH(1,1) conditional variance · Marchenko–Pastur random matrix denoising
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <span className="hidden sm:inline font-mono text-[11px] text-zinc-500 dark:text-zinc-400 whitespace-nowrap">
                {equationsOpen ? 'Hide equations' : 'Show equations'}
              </span>
              <ChevronDown
                className={`w-4 h-4 text-zinc-500 dark:text-zinc-400 transition-transform duration-200 ${
                  equationsOpen ? 'rotate-180' : ''
                }`}
              />
            </div>
          </button>

          <AnimatePresence initial={false}>
            {equationsOpen && (
              <motion.div
                key="equations-body"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.22, ease: 'easeInOut' }}
                className="overflow-hidden border-t border-[var(--hairline)]"
              >
                <div className="p-5 grid grid-cols-1 lg:grid-cols-2 gap-4 bg-[var(--bg-inset)]/40">
                  <div className="panel p-4 flex flex-col justify-between gap-4">
                    <GarchEquation />
                    <div className="pt-2 border-t border-[var(--hairline)] flex justify-end">
                      <button
                        type="button"
                        onClick={() => onNavigate('modeling')}
                        className="inline-flex items-center gap-1.5 text-xs font-mono text-gold-700 dark:text-gold-300 hover:underline whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        <span>Open Volatility Modeling</span>
                        <ArrowRight className="w-3.5 h-3.5 shrink-0" />
                      </button>
                    </div>
                  </div>

                  <div className="panel p-4 flex flex-col justify-between gap-4">
                    <CorrelationEquation />
                    <div className="pt-2 border-t border-[var(--hairline)] flex justify-end">
                      <button
                        type="button"
                        onClick={() => onNavigate('correlation')}
                        className="inline-flex items-center gap-1.5 text-xs font-mono text-gold-700 dark:text-gold-300 hover:underline whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        <span>Open Correlation Lab</span>
                        <ArrowRight className="w-3.5 h-3.5 shrink-0" />
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>
    </div>
  );
}
