'use client';

import React, { useRef } from 'react';
import {
  TrendingUp,
  Activity,
  Layers,
  GitBranch,
  ShieldAlert,
  ArrowRight,
  Database,
  Cpu,
  BarChart2,
  LineChart,
} from 'lucide-react';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { AnimatedBeam } from '@/components/ui/AnimatedBeam';
import { InteractiveGridPattern } from '@/components/ui/InteractiveGridPattern';
import { NumberTicker } from '@/components/ui/NumberTicker';
import {
  GarchEquation,
  EwmaEquation,
  HarRvEquation,
  CorrelationEquation,
  MathSymbol,
} from '@/components/math/MathEquations';

interface HomeViewProps {
  onNavigate: (view: 'analysis' | 'modeling' | 'correlation' | 'regression') => void;
  onSelectPreset: (ticker: string) => void;
}

export function HomeView({ onNavigate, onSelectPreset }: HomeViewProps) {
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
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Hero Canvas with Interactive Grid Pattern */}
      <div className="relative overflow-hidden rounded-2xl bg-white dark:bg-[#161513] border border-stone-200 dark:border-stone-800/90 p-8 sm:p-10 shadow-sm transition-colors">
        {/* Subtle Interactive Grid Canvas */}
        <InteractiveGridPattern
          width={32}
          height={32}
          squares={[36, 16]}
          className="opacity-40 dark:opacity-30"
        />

        <div className="relative z-10 max-w-3xl">
          {/* Unboxed inline metadata */}
          <div className="flex items-center gap-2 text-xs font-mono tracking-wider text-amber-700 dark:text-amber-400 uppercase mb-3">
            <span>Institutional Financial Econometrics</span>
            <span className="text-stone-300 dark:text-stone-700" aria-hidden="true">·</span>
            <span>Continuous Market Feeds</span>
            <span className="text-stone-300 dark:text-stone-700" aria-hidden="true">·</span>
            <span>Zero-AI-Slop Architecture</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-stone-900 dark:text-stone-100 mb-3 text-balance">
            Quantitative Time-Series, Volatility Modeling &amp; Extreme Risk Terminal
          </h1>

          <p className="text-xs sm:text-sm text-stone-600 dark:text-stone-300 leading-relaxed mb-6 max-w-2xl">
            Evaluate fat-tailed return distributions with Student-$t$ and Hansen Skew-$t$, detect volatility clustering via Engle ARCH-LM and Ljung-Box batteries, calibrate GARCH(1,1) &amp; EWMA term structures, run Ledoit-Wolf shrinkage, and backtest 5-model Value-at-Risk under Basel standards.
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <ShimmerButton
              onClick={() => onNavigate('analysis')}
              className="px-5 py-2.5 text-xs font-semibold"
            >
              <span>Launch Single-Stock Analysis</span>
              <ArrowRight className="w-4 h-4 text-amber-500" />
            </ShimmerButton>

            <button
              type="button"
              onClick={() => onNavigate('modeling')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-stone-100 hover:bg-stone-200 dark:bg-stone-800/80 dark:hover:bg-stone-800 text-stone-800 dark:text-stone-200 font-semibold text-xs rounded-lg transition-all border border-stone-200 dark:border-stone-700 cursor-pointer"
            >
              <Activity className="w-3.5 h-3.5 text-amber-500" />
              <span>GARCH / EWMA Modeling</span>
            </button>

            <button
              type="button"
              onClick={() => onNavigate('correlation')}
              className="inline-flex items-center gap-2 px-4 py-2 bg-stone-100 hover:bg-stone-200 dark:bg-stone-800/80 dark:hover:bg-stone-800 text-stone-800 dark:text-stone-200 font-semibold text-xs rounded-lg transition-all border border-stone-200 dark:border-stone-700 cursor-pointer"
            >
              <Layers className="w-3.5 h-3.5 text-amber-500" />
              <span>Correlation &amp; Shrinkage</span>
            </button>
          </div>
        </div>
      </div>

      {/* Purposeful Magic UI AnimatedBeam: Econometric Data Pipeline */}
      <div className="bg-white dark:bg-[#161513] border border-stone-200 dark:border-stone-800/90 rounded-2xl p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-stone-100 dark:border-stone-800/60 gap-2 mb-6">
          <div>
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <h2 className="font-bold text-sm text-stone-900 dark:text-stone-100 tracking-tight">
                Real-Time Econometric Data Pipeline
              </h2>
            </div>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
              Live computational signal flow from raw tick feeds to regulatory risk attribution
            </p>
          </div>

          <span className="text-[11px] font-mono text-stone-400">
            AnimatedBeam Architecture
          </span>
        </div>

        {/* Node container */}
        <div
          ref={containerRef}
          className="relative flex flex-col md:flex-row items-center justify-between gap-6 py-6 px-4 bg-stone-50/70 dark:bg-[#11100e] rounded-xl border border-stone-200/80 dark:border-stone-800/60"
        >
          {/* Node 1: Market Feed */}
          <div
            ref={feedRef}
            className="z-10 flex flex-col items-center justify-center w-28 h-28 rounded-xl border border-stone-200 dark:border-stone-700/80 bg-white dark:bg-[#1c1b18] shadow-xs text-center p-2"
          >
            <Database className="w-5 h-5 text-amber-600 dark:text-amber-400 mb-1" />
            <span className="text-xs font-bold text-stone-900 dark:text-stone-100">Market Feed</span>
            <span className="text-[10px] font-mono text-stone-400">OHLCV Bars</span>
          </div>

          {/* Node 2: Return Innovations */}
          <div
            ref={returnsRef}
            className="z-10 flex flex-col items-center justify-center w-28 h-28 rounded-xl border border-stone-200 dark:border-stone-700/80 bg-white dark:bg-[#1c1b18] shadow-xs text-center p-2"
          >
            <BarChart2 className="w-5 h-5 text-amber-600 dark:text-amber-400 mb-1" />
            <span className="text-xs font-bold text-stone-900 dark:text-stone-100">Returns</span>
            <span className="text-[10px] font-mono text-stone-400">ln(P_t / P_t-1)</span>
          </div>

          {/* Node 3: Econometric Engine */}
          <div
            ref={engineRef}
            className="z-10 flex flex-col items-center justify-center w-32 h-32 rounded-xl border-2 border-amber-600/50 dark:border-amber-500/50 bg-white dark:bg-[#1c1b18] shadow-sm text-center p-2"
          >
            <Cpu className="w-6 h-6 text-amber-600 dark:text-amber-400 mb-1 animate-pulse" />
            <span className="text-xs font-bold text-stone-900 dark:text-stone-100">Quant Engine</span>
            <span className="text-[10px] font-mono text-amber-700 dark:text-amber-400 font-semibold">GARCH / MLE / EVT</span>
          </div>

          {/* Node 4: Risk Diagnostics */}
          <div
            ref={riskRef}
            className="z-10 flex flex-col items-center justify-center w-28 h-28 rounded-xl border border-stone-200 dark:border-stone-700/80 bg-white dark:bg-[#1c1b18] shadow-xs text-center p-2"
          >
            <ShieldAlert className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mb-1" />
            <span className="text-xs font-bold text-stone-900 dark:text-stone-100">Risk Output</span>
            <span className="text-[10px] font-mono text-stone-400">VaR · ES · CCTR</span>
          </div>

          {/* Animated Beams connecting nodes */}
          <AnimatedBeam
            containerRef={containerRef}
            fromRef={feedRef}
            toRef={returnsRef}
            duration={2.5}
            pathColor="#78716c"
            gradientStartColor="#d97706"
            gradientStopColor="#f59e0b"
          />
          <AnimatedBeam
            containerRef={containerRef}
            fromRef={returnsRef}
            toRef={engineRef}
            duration={2.5}
            delay={0.4}
            pathColor="#78716c"
            gradientStartColor="#d97706"
            gradientStopColor="#f59e0b"
          />
          <AnimatedBeam
            containerRef={containerRef}
            fromRef={engineRef}
            toRef={riskRef}
            duration={2.5}
            delay={0.8}
            pathColor="#78716c"
            gradientStartColor="#d97706"
            gradientStopColor="#10b981"
          />
        </div>
      </div>

      {/* KPI Stats Strip with NumberTicker */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-[#161513] shadow-2xs">
          <div className="text-xs text-stone-500 dark:text-stone-400 font-medium">S&amp;P 500 Historical Volatility</div>
          <div className="mt-2 text-2xl font-bold text-stone-900 dark:text-stone-100">
            <NumberTicker value={13.62} decimalPlaces={2} suffix="%" />
          </div>
          <div className="mt-1 text-[11px] text-stone-400 font-mono">
            Annualized daily σ × √252
          </div>
        </div>

        <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-[#161513] shadow-2xs">
          <div className="text-xs text-stone-500 dark:text-stone-400 font-medium">RiskMetrics EWMA Decay (λ)</div>
          <div className="mt-2 text-2xl font-bold text-amber-700 dark:text-amber-400">
            <NumberTicker value={0.94} decimalPlaces={2} />
          </div>
          <div className="mt-1 text-[11px] text-stone-400 font-mono">
            Half-life = 11.2 trading days
          </div>
        </div>

        <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-[#161513] shadow-2xs">
          <div className="text-xs text-stone-500 dark:text-stone-400 font-medium">GARCH(1,1) Persistence (α + β)</div>
          <div className="mt-2 text-2xl font-bold text-emerald-700 dark:text-emerald-400">
            <NumberTicker value={0.978} decimalPlaces={3} />
          </div>
          <div className="mt-1 text-[11px] text-stone-400 font-mono">
            Stationary mean-reverting (α+β &lt; 1)
          </div>
        </div>

        <div className="p-4 rounded-xl border border-stone-200 dark:border-stone-800 bg-white dark:bg-[#161513] shadow-2xs">
          <div className="text-xs text-stone-500 dark:text-stone-400 font-medium">POT-GPD 99% Tail VaR</div>
          <div className="mt-2 text-2xl font-bold text-rose-600 dark:text-rose-400">
            <NumberTicker value={3.42} decimalPlaces={2} suffix="%" />
          </div>
          <div className="mt-1 text-[11px] text-stone-400 font-mono">
            Heavy-tail Fréchet extreme quantile
          </div>
        </div>
      </div>

      {/* Preset Universe Selection Card */}
      <div className="bg-white dark:bg-[#161513] border border-stone-200 dark:border-stone-800/90 rounded-2xl p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-stone-100 dark:border-stone-800/60 gap-2 mb-4">
          <div>
            <h3 className="font-bold text-sm text-stone-900 dark:text-stone-100 tracking-tight">
              Pre-Calibrated Asset Universe
            </h3>
            <p className="text-xs text-stone-500 dark:text-stone-400 mt-0.5">
              Select an asset to jump directly into single-stock or multi-factor workstations
            </p>
          </div>
          <span className="text-[11px] font-mono text-stone-400">5 Representative Assets</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {presets.map((item) => (
            <button
              key={item.ticker}
              type="button"
              onClick={() => {
                onSelectPreset(item.ticker);
                onNavigate('analysis');
              }}
              className="text-left p-3.5 rounded-xl border border-stone-200 dark:border-stone-800 bg-stone-50/50 dark:bg-[#191816] hover:border-amber-500 hover:bg-white dark:hover:bg-[#201e1a] transition-all cursor-pointer group shadow-2xs"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold text-sm text-stone-900 dark:text-stone-100 group-hover:text-amber-600 dark:group-hover:text-amber-400 transition-colors">
                  {item.ticker}
                </span>
                <span className="font-mono text-xs text-amber-700 dark:text-amber-400 font-semibold">
                  β {item.beta.toFixed(2)}
                </span>
              </div>
              <div className="text-[11px] font-medium text-stone-700 dark:text-stone-300 mt-1 truncate">
                {item.name}
              </div>
              <div className="flex items-center justify-between text-[10px] text-stone-400 mt-2 font-mono">
                <span>{item.category}</span>
                <span>σ {item.vol}%</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Econometric Framework Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Module 1: Volatility & Jumps */}
        <div className="p-5 rounded-2xl border border-stone-200 dark:border-stone-800/90 bg-white dark:bg-[#161513] shadow-sm">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100 dark:border-stone-800/60 mb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <h4 className="font-bold text-sm text-stone-900 dark:text-stone-100">
                Parametric Volatility &amp; Jump Filters
              </h4>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('modeling')}
              className="text-xs font-semibold text-amber-700 dark:text-amber-400 hover:underline cursor-pointer"
            >
              Open Modeling &rarr;
            </button>
          </div>
          <p className="text-xs text-stone-600 dark:text-stone-300 mb-3 leading-relaxed">
            Estimate persistent variance memory with Bollerslev GARCH(1,1) and RiskMetrics EWMA. Asymptotically separate continuous Brownian diffusion from abnormal discrete jump shocks via Corsi HAR-RV and Barndorff-Nielsen Bipower Variation.
          </p>
          <div className="space-y-2 pt-1 font-mono text-xs">
            <GarchEquation />
          </div>
        </div>

        {/* Module 2: High-Dimensional Covariance */}
        <div className="p-5 rounded-2xl border border-stone-200 dark:border-stone-800/90 bg-white dark:bg-[#161513] shadow-sm">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100 dark:border-stone-800/60 mb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <h4 className="font-bold text-sm text-stone-900 dark:text-stone-100">
                Shrinkage Covariance &amp; RMT Filtering
              </h4>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('correlation')}
              className="text-xs font-semibold text-amber-700 dark:text-amber-400 hover:underline cursor-pointer"
            >
              Open Correlation &rarr;
            </button>
          </div>
          <p className="text-xs text-stone-600 dark:text-stone-300 mb-3 leading-relaxed">
            Condition ill-conditioned sample covariance matrices for N &approx; 10 assets using Ledoit-Wolf analytical shrinkage. Filter empirical eigenvalues against the Marchenko-Pastur noise envelope to isolate true systemic market modes.
          </p>
          <div className="space-y-2 pt-1 font-mono text-xs">
            <CorrelationEquation />
          </div>
        </div>
      </div>
    </div>
  );
}
