'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { MicrostructureResult } from '@/lib/analysis-engine';
import {
  RollSpreadEquation,
  AmihudEquation,
  RealizedSemivarianceEquation,
} from '@/components/math/MathEquations';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens } from '@/lib/plotly-theme';
import {
  Waves,
  Clock,
  Info,
} from 'lucide-react';

interface AnalysisMicrostructureTabProps {
  microstructure: MicrostructureResult;
  ticker: string;
}

export function AnalysisMicrostructureTab({
  microstructure,
  ticker,
}: AnalysisMicrostructureTabProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [spreadHoverIdx, setSpreadHoverIdx] = useState<number | null>(null);

  const curveData = microstructure.intradaySpreadCurve;
  const nCurve = curveData.length;
  const activeIdx = spreadHoverIdx !== null && spreadHoverIdx < nCurve ? spreadHoverIdx : 0;
  const activePoint = curveData[activeIdx] || { timeStr: '09:30', estimatedSpreadBps: 10 };

  const { spreadTraces, spreadLayout } = useMemo<{
    spreadTraces: Data[];
    spreadLayout: Partial<Layout>;
  }>(() => {
    if (curveData.length === 0) return { spreadTraces: [], spreadLayout: {} };
    return {
      spreadTraces: [
        {
          type: 'scatter',
          mode: 'lines',
          name: 'Estimated Spread',
          x: curveData.map((c) => c.timeStr),
          y: curveData.map((c) => c.estimatedSpreadBps),
          fill: 'tozeroy',
          fillcolor: 'rgba(16,185,129,0.14)',
          line: { color: '#10b981', width: 2 },
          hovertemplate: '%{x}: <b>%{y:.1f} bps</b><extra></extra>',
        },
      ],
      spreadLayout: {
        margin: { l: 54, r: 16, t: 10, b: 36 },
        xaxis: {
          type: 'category',
          nticks: 6,
          tickfont: { size: 10, color: tokens.textMuted },
        },
        yaxis: {
          title: { text: 'Spread (bps)', font: { size: 10, color: tokens.textMuted } },
          ticksuffix: ' bps',
          rangemode: 'tozero',
        },
      },
    };
  }, [curveData, tokens]);

  return (
    <div className="space-y-6">
      {/* Bid-Ask Bounce & Return Autocorrelation */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Waves className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                1-Minute Serial Autocorrelation &amp; Bid-Ask Bounce (Roll, 1984)
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              High-frequency order bounce between bid and ask quotes creates negative lag-1 return autocorrelation
            </p>
          </div>

          <RollSpreadEquation className="hidden lg:inline" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <div className="text-xs text-zinc-500 dark:text-zinc-400">1-Min Return Autocorrelation (ρ₁)</div>
            <div className={`mt-2 text-2xl font-bold font-mono ${microstructure.lag1Autocorr < 0 ? 'text-gold-500' : 'text-zinc-900 dark:text-zinc-100'}`}>
              {microstructure.lag1Autocorr.toFixed(3)}
            </div>
            <div className="mt-2 text-[11px] font-mono text-zinc-400">
              {microstructure.bidAskBounceDetected
                ? 'Negative ρ₁: Significant bid-ask bounce detected'
                : 'ρ₁ ≥ 0: Diffusive or momentum-driven'}
            </div>
          </div>

          <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <div className="text-xs text-zinc-500 dark:text-zinc-400">Roll (1984) Effective Spread</div>
            <div className="mt-2 text-2xl font-bold font-mono text-gold-600 dark:text-gold-400">
              {(microstructure.rollEffectiveSpreadPct * 10000).toFixed(1)} bps
            </div>
            <div className="mt-2 text-[11px] font-mono text-zinc-400">
              {(microstructure.rollEffectiveSpreadPct * 100).toFixed(3)}% of asset price
            </div>
          </div>

          <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <div className="text-xs text-zinc-500 dark:text-zinc-400">Corwin-Schultz (2012) High-Low Spread</div>
            <div className="mt-2 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
              {(microstructure.corwinSchultzSpreadPct * 10000).toFixed(1)} bps
            </div>
            <div className="mt-2 text-[11px] font-mono text-zinc-400">
              Derived from 2-day high-to-low price volatility
            </div>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span>
            Under the Roll model, successive market orders alternating between dealer bid and dealer ask generate negative
            serial covariance Cov(Δp_t, Δp_t-1) = -s²/4. This allows extracting implicit execution costs without requiring
            Level 2 order-book tick data.
          </span>
        </div>
      </div>

      {/* Intraday Spread Smile Curve */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Intraday Effective Bid-Ask Spread Dynamics (Smile Curve)
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Evolution of liquidity provision costs across market open, continuous midday, and market-on-close
            </p>
          </div>
          <div className="font-mono text-xs text-zinc-400">
            {activePoint.timeStr}: <strong className="text-gold-500">{activePoint.estimatedSpreadBps.toFixed(1)} bps</strong>
          </div>
        </div>

        <div className="mt-4">
          <ChartFullscreen title="Estimated Intraday Spread Curve">
            <PlotlyChart
              data={spreadTraces}
              layout={spreadLayout}
              plotType="categorical"
              hovermode="x unified"
              showLegend={false}
              height={240}
              showWheelHint
              exportFilename={`${ticker}-intraday-spread-smile`}
              uirevisionKey={`spread-smile-${ticker}-${nCurve}`}
              onHover={(ev) => {
                const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                if (pt && typeof pt.pointIndex === 'number') setSpreadHoverIdx(pt.pointIndex);
              }}
              onUnhover={() => setSpreadHoverIdx(null)}
            />
          </ChartFullscreen>
        </div>

        <div className="mt-3 text-[11px] text-zinc-400">
          Spreads widen sharply at market open (09:30–10:00 EST) due to overnight asymmetric information processing,
          compress to a tight baseline during midday institutional crossing, and tighten further into the closing auction.
        </div>
      </div>

      {/* Realized Semivariance & Realized Quarticity (RQ) Panel */}
      <div className="grid-cards">
        {/* Realized Semivariance */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Realized Semivariance Decomposition (Barndorff-Nielsen et al., 2010)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Separating continuous variance into downside crash shocks vs. upside rallies
              </p>
            </div>
            <RealizedSemivarianceEquation className="hidden md:inline" />
          </div>

          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Downside Semivariance (RS⁻)</div>
              <div className="mt-1 text-lg font-bold font-mono text-rose-600 dark:text-rose-400">
                {(microstructure.realizedSemivarianceDownside * 10000).toFixed(1)} bps²
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Upside Semivariance (RS⁺)</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {(microstructure.realizedSemivarianceUpside * 10000).toFixed(1)} bps²
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Realized Skewness (RS⁺ - RS⁻)</div>
              <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {(microstructure.realizedSkewness * 10000).toFixed(1)} bps²
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30 flex items-center justify-between text-xs font-mono">
            <span>Downside Variance Share:</span>
            <strong className="text-rose-600 dark:text-rose-400">
              {(
                (microstructure.realizedSemivarianceDownside /
                  (microstructure.realizedSemivarianceDownside + microstructure.realizedSemivarianceUpside || 1)) *
                100
              ).toFixed(1)}
              %
            </strong>
          </div>
        </div>

        {/* Amihud Illiquidity & Realized Quarticity (RQ) */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Amihud Illiquidity &amp; Realized Quarticity (RQ) Error Bars
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Price impact per $1M volume &amp; finite-sample volatility standard error
              </p>
            </div>
            <AmihudEquation className="hidden md:inline" />
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Amihud Illiquidity (ILLIQ)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {microstructure.amihudIlliquidity.toFixed(3)}
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                Price impact per $1M turnover
              </div>
            </div>

            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Realized Quarticity SE(RV)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-gold-600 dark:text-gold-400">
                ±{(microstructure.rvStandardErrorAnn * 100).toFixed(2)}%
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                Asymptotic 95% error band on RV
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>
              Realized Quarticity (RQ) measures the variance of realized variance itself. Higher quarticity indicates
              intermittent burstiness in trading activity and widens the standard error around the true integrated volatility.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
