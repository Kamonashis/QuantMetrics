'use client';

import React, { useRef } from 'react';
import {
  TrendingUp,
  Activity,
  Layers,
  ShieldAlert,
  ArrowRight,
  Database,
  Cpu,
  BarChart2,
  LineChart,
  Sigma,
  Network,
  LineChart as RegressionIcon,
  ScanSearch,
  Braces,
} from 'lucide-react';
import { AnimatedBeam } from '@/components/ui/AnimatedBeam';
import { NumberTicker } from '@/components/ui/NumberTicker';
import { Spotlight } from '@/components/ui/spotlight';
import { BackgroundBeams } from '@/components/ui/background-beams';
import { Meteors } from '@/components/ui/meteors';
import { MovingBorderButton } from '@/components/ui/moving-border';
import { BentoGrid, BentoGridItem } from '@/components/ui/bento-grid';
import { GlowingEffect } from '@/components/ui/glowing-effect';
import { TextGenerateEffect } from '@/components/ui/text-generate-effect';
import { FlipWords } from '@/components/ui/flip-words';
import { TickerTape, type TapeItem } from '@/components/ui/ticker-tape';
import {
  GarchEquation,
  CorrelationEquation,
} from '@/components/math/MathEquations';

interface HomeViewProps {
  onNavigate: (view: 'analysis' | 'modeling' | 'correlation' | 'regression') => void;
  onSelectPreset: (ticker: string) => void;
  tapeItems: TapeItem[];
}

export function HomeView({ onNavigate, onSelectPreset, tapeItems }: HomeViewProps) {
  // References for the AnimatedBeam data pipeline
  const containerRef = useRef<HTMLDivElement>(null);
  const feedRef = useRef<HTMLDivElement>(null);
  const returnsRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<HTMLDivElement>(null);
  const riskRef = useRef<HTMLDivElement>(null);

  const presets = [
    { ticker: 'AAPL', name: 'Apple Inc.', category: 'Consumer Tech', vol: 21.4, beta: 1.12 },
    { ticker: 'NVDA', name: 'NVIDIA Corp.', category: 'Semiconductors / AI', vol: 44.8, beta: 1.95 },
    { ticker: 'MSFT', name: 'Microsoft Corp.', category: 'Enterprise Cloud', vol: 19.8, beta: 0.98 },
    { ticker: 'SPY', name: 'S&P 500 ETF', category: 'Broad Market Benchmark', vol: 13.6, beta: 1.00 },
    { ticker: 'TSLA', name: 'Tesla Inc.', category: 'High Beta Momentum', vol: 52.3, beta: 2.15 },
  ];

  return (
    <div className="space-y-6 pb-10">
      {/* ════════════ HERO ════════════ */}
      <section className="relative overflow-hidden rounded-3xl border border-white/[0.08] bg-[#08080A] text-zinc-100 shadow-[0_40px_80px_-40px_rgba(0,0,0,0.8)]">
        <BackgroundBeams />
        <Spotlight />
        <div className="bg-grid mask-radial-fade pointer-events-none absolute inset-0 opacity-[0.35]" />

        <div className="relative z-10 flex flex-col gap-10 px-6 pb-10 pt-12 sm:px-10 sm:pt-16 lg:flex-row lg:items-center lg:gap-14">
          <div className="max-w-2xl flex-1">
            {/* Kicker */}
            <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[10px] font-medium uppercase tracking-[0.22em] text-gold-300/90">
              <span className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--positive)] animate-ping-dot" />
                Continuous Feeds
              </span>
              <span className="text-zinc-700" aria-hidden="true">/</span>
              <span>Institutional Econometrics</span>
              <span className="text-zinc-700" aria-hidden="true">/</span>
              <span className="text-zinc-500">Zero-AI Mathematics</span>
            </div>

            {/* Headline */}
            <h1 className="font-display text-[2.6rem] font-semibold leading-[1.04] tracking-[-0.02em] text-zinc-50 sm:text-6xl">
              <TextGenerateEffect words="The mathematics" as="span" className="block" />
              <span className="mt-1 block">
                of <span className="text-gradient-gold">market risk</span>.
              </span>
            </h1>

            <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-zinc-400">
              Institutional-grade econometrics in one terminal —{' '}
              <FlipWords
                words={['GARCH volatility term structures', 'Hansen skewed-t tails', 'Ledoit–Wolf covariance shrinkage', 'Basel-grade VaR backtesting']}
                className="font-medium text-zinc-200"
              />
              , stationarity batteries, EVT tail models — every number traceable, every test reproducible.
            </p>

            {/* CTAs */}
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <MovingBorderButton onClick={() => onNavigate('analysis')} aria-label="Launch Single-Stock Analysis">
                Launch the Terminal
                <ArrowRight className="h-4 w-4" />
              </MovingBorderButton>
              <button
                type="button"
                onClick={() => onNavigate('modeling')}
                className="group inline-flex h-11 cursor-pointer items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-5 text-[13px] font-semibold text-zinc-200 backdrop-blur-sm transition-all duration-200 hover:border-gold-400/40 hover:text-gold-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60"
              >
                <Activity className="h-4 w-4 text-gold-400 transition-transform duration-200 group-hover:scale-110" />
                GARCH / EWMA Modeling
              </button>
            </div>

            {/* Proof strip */}
            <div className="mt-9 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-500">
              <span><strong className="text-zinc-200">28</strong> Econometric Tests</span>
              <span className="text-zinc-800" aria-hidden="true">◆</span>
              <span><strong className="text-zinc-200">5</strong>-Model VaR · 99% ES</span>
              <span className="text-zinc-800" aria-hidden="true">◆</span>
              <span><strong className="text-zinc-200">N×N</strong> Correlation RMT</span>
            </div>
          </div>

          {/* Floating terminal telemetry card (decorative) */}
          <div className="relative mx-auto w-full max-w-sm shrink-0 lg:mx-0">
            <div
              aria-hidden="true"
              className="absolute -inset-6 rounded-[28px] bg-[radial-gradient(ellipse_at_center,rgba(222,192,132,0.14),transparent_70%)] blur-2xl"
            />
            <div className="relative rounded-2xl border border-white/[0.09] bg-[#101014]/80 p-5 shadow-2xl backdrop-blur-md">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                <div className="flex items-center gap-2">
                  <span className="flex h-2 w-2 gap-1">
                    <span className="h-2 w-2 rounded-full bg-zinc-700" />
                    <span className="h-2 w-2 rounded-full bg-zinc-700" />
                    <span className="h-2 w-2 rounded-full bg-gold-400/80" />
                  </span>
                  <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">
                    quant-engine · AAPL
                  </span>
                </div>
                <span className="rounded border border-gold-500/30 bg-gold-400/10 px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider text-gold-300">
                  MLE
                </span>
              </div>

              {/* Sparkline */}
              <svg viewBox="0 0 260 64" className="mt-4 w-full" role="presentation">
                <defs>
                  <linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#DEC084" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#DEC084" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path
                  d="M0 46 L18 42 L34 48 L52 36 L70 40 L88 28 L104 34 L122 22 L138 30 L156 18 L172 26 L190 14 L208 20 L226 10 L244 16 L260 8"
                  fill="none" stroke="#DEC084" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                />
                <path
                  d="M0 46 L18 42 L34 48 L52 36 L70 40 L88 28 L104 34 L122 22 L138 30 L156 18 L172 26 L190 14 L208 20 L226 10 L244 16 L260 8 L260 64 L0 64 Z"
                  fill="url(#spark-fill)" stroke="none"
                />
              </svg>

              {/* Telemetry rows */}
              <dl className="mt-4 space-y-2 font-mono text-[11px]">
                {[
                  ['σ²(t+1) forecast', '0.000412'],
                  ['α + β persistence', '0.9741 · stationary'],
                  ['VaR 99% · 1d', '-3.42%'],
                  ['ARCH-LM (12)', 'p = 0.000 · reject H₀'],
                ].map(([k, v]) => (
                  <div key={k} className="flex items-baseline justify-between gap-4 border-b border-white/[0.05] pb-2 last:border-0 last:pb-0">
                    <dt className="text-zinc-500">{k}</dt>
                    <dd className="tabular-nums text-zinc-100">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      </section>

      {/* ════════════ TICKER TAPE ════════════ */}
      {tapeItems.length > 0 && <TickerTape items={tapeItems} />}

      {/* ════════════ PIPELINE ════════════ */}
      <section className="panel p-6">
        <div className="mb-6 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-5 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1.5 flex items-center gap-2">
              <Cpu className="h-3.5 w-3.5" /> Signal Flow
            </p>
            <h2 className="font-display text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              From raw ticks to risk attribution
            </h2>
            <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              Live computational path of every number shown in the terminal
            </p>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">
            AnimatedBeam Architecture
          </span>
        </div>

        <div
          ref={containerRef}
          className="relative flex flex-col items-stretch justify-between gap-4 rounded-2xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-4 py-6 md:flex-row md:items-center md:gap-6 md:px-8"
        >
          {[
            { ref: feedRef, icon: Database, title: 'Market Feed', sub: 'OHLCV · Yahoo Finance', accent: false },
            { ref: returnsRef, icon: BarChart2, title: 'Returns', sub: 'r_t = ln(P_t / P_{t-1})', accent: false },
            { ref: engineRef, icon: Cpu, title: 'Quant Engine', sub: 'GARCH · MLE · EVT', accent: true },
            { ref: riskRef, icon: ShieldAlert, title: 'Risk Output', sub: 'VaR · ES · Euler CCTR', accent: false },
          ].map(({ ref, icon: Icon, title, sub, accent }) => (
            <div
              key={title}
              ref={ref}
              className={`z-10 flex flex-col items-center justify-center gap-1 rounded-2xl border px-4 py-5 text-center shadow-sm transition-colors ${
                accent
                  ? 'border-gold-500/50 bg-gradient-to-b from-gold-400/15 to-gold-600/5 dark:from-gold-400/10 dark:to-transparent'
                  : 'border-[var(--hairline)] bg-[var(--bg-panel)]'
              }`}
            >
              <Icon className={`h-5 w-5 ${accent ? 'text-gold-600 dark:text-gold-300 animate-pulse' : 'text-zinc-400 dark:text-zinc-500'}`} strokeWidth={1.7} />
              <span className="font-display text-[13px] font-semibold text-zinc-900 dark:text-zinc-50">{title}</span>
              <span className="font-mono text-[10px] text-zinc-400 dark:text-zinc-500">{sub}</span>
            </div>
          ))}

          <AnimatedBeam containerRef={containerRef} fromRef={feedRef} toRef={returnsRef} duration={2.4} pathColor="#63636d" gradientStartColor="#C69A48" gradientStopColor="#DEC084" />
          <AnimatedBeam containerRef={containerRef} fromRef={returnsRef} toRef={engineRef} duration={2.4} delay={0.35} pathColor="#63636d" gradientStartColor="#C69A48" gradientStopColor="#DEC084" />
          <AnimatedBeam containerRef={containerRef} fromRef={engineRef} toRef={riskRef} duration={2.4} delay={0.7} pathColor="#63636d" gradientStartColor="#DEC084" gradientStopColor="#3fb27f" />
        </div>
      </section>

      {/* ════════════ WORKSATION BENTO ════════════ */}
      <section>
        <div className="mb-4 flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1.5 flex items-center gap-2">
              <Sigma className="h-3.5 w-3.5" /> The Workstations
            </p>
            <h2 className="font-display text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              Four instruments, one instrument panel
            </h2>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">
            Select a module to enter
          </span>
        </div>

        <BentoGrid className="md:grid-cols-3">
          <BentoGridItem
            className="md:col-span-2"
            onClick={() => onNavigate('analysis')}
            icon={<ScanSearch className="h-6 w-6" strokeWidth={1.6} />}
            title="Single-Stock Analysis Battery"
            description="ADF · PP · KPSS stationarity across price, log-return and squared-return lenses. ARCH-LM, Ljung-Box, CUSUM-SQ structural breaks, Hurst exponent, Roll & Corwin-Schultz spread estimators, and a 5-model VaR with Kupiec + Christoffersen backtests."
          />
          <BentoGridItem
            onClick={() => onNavigate('modeling')}
            icon={<Activity className="h-6 w-6" strokeWidth={1.6} />}
            title="Volatility Modeling"
            description="GARCH(1,1) by MLE with 68/95% forecast cones, RiskMetrics EWMA λ=0.94, Corsi HAR-RV cascade on RV components, news-impact curves and BNS bipower jump decomposition."
          />
          <BentoGridItem
            onClick={() => onNavigate('correlation')}
            icon={<Network className="h-6 w-6" strokeWidth={1.6} />}
            title="Correlation & Covariance"
            description="Pearson · Spearman · Kendall matrices, Ledoit–Wolf shrinkage, Marchenko–Pastur eigenvalue clipping, MST topology and GMV frontier with Euler risk attribution."
          />
          <BentoGridItem
            onClick={() => onNavigate('regression')}
            icon={<RegressionIcon className="h-6 w-6" strokeWidth={1.6} />}
            title="Factor & Regression Lab"
            description="OLS single-index CAPM with Jensen's alpha, Durbin–Watson diagnostics, residual QQ inspection and next-day return forecasting with confidence bands."
          />
          <BentoGridItem
            icon={<Braces className="h-6 w-6" strokeWidth={1.6} />}
            title="Traceable by construction"
            description="No LLM reasoning anywhere in the chain. Deterministic generators, seeded paths, analytic MLE, closed-form test statistics — every figure reproducible offline."
            header={
              <span className="font-mono text-[10px] uppercase leading-relaxed tracking-[0.14em] text-zinc-400 dark:text-zinc-500">
                ln L · AIC · BIC<br />χ² · F · z-scores<br />Kupiec · Christoffersen
              </span>
            }
          />
        </BentoGrid>
      </section>

      {/* ════════════ KPI BAND ════════════ */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {[
          { label: 'S&P 500 Realized Vol', node: <NumberTicker value={13.62} decimalPlaces={2} suffix="%" />, sub: 'Annualized · σ × √252', tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'RiskMetrics EWMA λ', node: <NumberTicker value={0.94} decimalPlaces={2} />, sub: 'Half-life ≈ 11.2 sessions', tone: 'text-gold-700 dark:text-gold-300' },
          { label: 'GARCH(1,1) Persistence', node: <NumberTicker value={0.978} decimalPlaces={3} />, sub: 'α + β < 1 · mean-reverting', tone: 'text-gold-700 dark:text-gold-300' },
          { label: 'POT-GPD 99% Tail VaR', node: <NumberTicker value={3.42} decimalPlaces={2} suffix="%" />, sub: 'Fréchet extreme quantile', tone: 'text-[var(--negative)]' },
        ].map((s) => (
          <div key={s.label} className="panel-inset flex flex-col justify-between gap-2 p-4">
            <div className="text-[11px] font-medium leading-snug text-zinc-500 dark:text-zinc-400">{s.label}</div>
            <div className={`font-mono text-2xl font-semibold tabular-nums tracking-tight ${s.tone}`}>{s.node}</div>
            <div className="font-mono text-[9.5px] uppercase tracking-wide text-zinc-400 dark:text-zinc-500">{s.sub}</div>
          </div>
        ))}
      </section>

      {/* ════════════ FRAMEWORK EQUATIONS ════════════ */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="panel p-6">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-gold-600 dark:text-gold-300" />
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Parametric Volatility &amp; Jump Filters
              </h3>
            </div>
            <button type="button" onClick={() => onNavigate('modeling')} className="cursor-pointer text-[11px] font-semibold text-gold-700 dark:text-gold-300 hover:underline">
              Open Modeling →
            </button>
          </div>
          <p className="mb-4 text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
            Bollerslev GARCH(1,1) and RiskMetrics EWMA variance memory, Corsi HAR-RV cascade over
            realized-volatility components, Barndorff-Nielsen &amp; Shephard bipower variation separating
            diffusion from jumps.
          </p>
          <div className="panel-inset flex items-center justify-center px-4 py-5 font-mono text-xs">
            <GarchEquation />
          </div>
        </div>

        <div className="panel p-6">
          <div className="mb-3 flex items-center justify-between border-b border-[var(--hairline)] pb-3">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-gold-600 dark:text-gold-300" />
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Shrinkage Covariance &amp; RMT Filtering
              </h3>
            </div>
            <button type="button" onClick={() => onNavigate('correlation')} className="cursor-pointer text-[11px] font-semibold text-gold-700 dark:text-gold-300 hover:underline">
              Open Correlation →
            </button>
          </div>
          <p className="mb-4 text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
            Ledoit–Wolf analytical shrinkage for ill-conditioned N-asset covariance matrices.
            Marchenko–Pastur spectral clipping isolates true systemic modes from sampling noise.
          </p>
          <div className="panel-inset flex items-center justify-center px-4 py-5 font-mono text-xs">
            <CorrelationEquation />
          </div>
        </div>
      </section>

      {/* ════════════ PRESET UNIVERSE ════════════ */}
      <section className="panel p-6">
        <div className="mb-4 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-4 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1.5">Pre-Calibrated Universe</p>
            <h3 className="font-display text-base font-semibold text-zinc-900 dark:text-zinc-50">
              Five liquid benchmarks, instantly wired
            </h3>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">
            Jump into any workstation
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {presets.map((item) => (
            <button
              key={item.ticker}
              type="button"
              onClick={() => {
                onSelectPreset(item.ticker);
                onNavigate('analysis');
              }}
              className="group relative cursor-pointer overflow-hidden rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] p-4 text-left transition-all duration-200 hover:border-gold-500/45 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60"
            >
              <GlowingEffect borderWidth={1} proximity={36} />
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold tracking-wide text-zinc-900 transition-colors group-hover:text-gold-700 dark:text-zinc-50 dark:group-hover:text-gold-300">
                  {item.ticker}
                </span>
                <span className="font-mono text-[10px] font-semibold text-gold-700 dark:text-gold-300">
                  β {item.beta.toFixed(2)}
                </span>
              </div>
              <div className="mt-1.5 truncate text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                {item.name}
              </div>
              <div className="mt-3 flex items-center justify-between font-mono text-[9px] uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
                <span className="truncate">{item.category}</span>
                <span className="shrink-0">σ {item.vol}%</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* ════════════ FINAL CTA ════════════ */}
      <section className="relative overflow-hidden rounded-3xl border border-gold-500/25 bg-gradient-to-br from-[#141310] via-[#100f0c] to-[#08080a] px-6 py-12 text-center dark:from-[#171510] dark:via-[#100f0c] dark:to-[#08080a]">
        <Meteors count={16} />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(222,192,132,0.12),transparent_65%)]"
        />
        <div className="relative z-10 mx-auto max-w-xl">
          <p className="kicker mb-3 justify-center">Ready when you are</p>
          <h2 className="font-display text-2xl font-semibold leading-tight tracking-tight text-zinc-50 sm:text-4xl">
            Interrogate the tails.
            <br />
            <span className="text-gradient-gold">Trust every decimal.</span>
          </h2>
          <p className="mx-auto mt-4 max-w-md text-xs leading-relaxed text-zinc-400">
            Twelve years of daily bars, minute-level intraday seasonality, closed-form MLE —
            all recomputed on your machine, deterministic end to end.
          </p>
          <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
            <MovingBorderButton onClick={() => onNavigate('analysis')}>
              Enter the Terminal
              <ArrowRight className="h-4 w-4" />
            </MovingBorderButton>
            <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-500">
              Synthetic fallback · Nothing leaves your machine
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
