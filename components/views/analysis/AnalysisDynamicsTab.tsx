'use client';

import React, { useState, useEffect, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  NewsImpactPoint,
  VolatilitySignaturePoint,
  OvernightIntradaySplitDay,
  DiurnalRvReconciliation,
} from '@/lib/analysis-engine';
import { DailyRealizedVol, DiurnalPoint } from '@/lib/quant-engine';
import { IntradayDiurnalChart } from '@/components/charts/IntradayDiurnalChart';
import { JumpDecompositionChart } from '@/components/charts/JumpDecompositionChart';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import {
  NewsImpactEquation,
  SignaturePlotEquation,
} from '@/components/math/MathEquations';
import {
  Zap,
  Clock,
  Info,
  TriangleAlert,
} from 'lucide-react';

interface AnalysisDynamicsTabProps {
  newsImpactPoints: NewsImpactPoint[];
  volSignaturePoints: VolatilitySignaturePoint[];
  overnightIntradaySplit: OvernightIntradaySplitDay[];
  dailyRealizedVol: DailyRealizedVol[];
  diurnalProfile: DiurnalPoint[];
  /** diurnal mean ÷ intraday 1-min RV — the sanity gate for the diurnal panel */
  diurnalReconciliation: DiurnalRvReconciliation | null;
  /** 'live_market' | 'synthetic_gbm' — provenance of the intraday bars */
  intradaySource?: string | null;
  /** GJR fit on open→close returns — does γ survive removing the gap? */
  newsImpactOcPoints?: NewsImpactPoint[] | null;
  /** fitted γ close→close and open→close, for the Δγ readout */
  gjrGammaCloseClose?: number | null;
  gjrGammaOpenClose?: number | null;
  ticker: string;
}

export function AnalysisDynamicsTab({
  newsImpactPoints,
  volSignaturePoints,
  overnightIntradaySplit,
  dailyRealizedVol,
  diurnalProfile,
  diurnalReconciliation,
  intradaySource = null,
  newsImpactOcPoints = null,
  gjrGammaCloseClose = null,
  gjrGammaOpenClose = null,
  ticker,
}: AnalysisDynamicsTabProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);
  const [newsHoverIdx, setNewsHoverIdx] = useState<number | null>(null);
  const [sigHoverIdx, setSigHoverIdx] = useState<number | null>(null);

  // Next earnings — fetched lazily, server-cached 15 min. "No data" is the designed answer when Yahoo's calendar is unreachable.
  const [earnings, setEarnings] = useState<{ ticker: string; date: string; isEstimate: boolean } | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/options?mode=earnings&symbol=${encodeURIComponent(ticker)}`)
      .then((r) => (r.ok ? (r.json() as Promise<{ available: boolean; nextEarningsDate?: string | null; isEstimate?: boolean }>) : null))
      .then((d) => {
        if (cancelled) return;
        if (d?.available && d.nextEarningsDate) {
          setEarnings({ ticker, date: d.nextEarningsDate, isEstimate: d.isEstimate !== false });
        } else {
          setEarnings(null);
        }
      })
      .catch(() => {
        if (!cancelled) setEarnings(null);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);
  const earningsCurrent = earnings && earnings.ticker === ticker ? earnings : null;

  // Aggregate overnight share
  const avgOvernightPct = React.useMemo(() => {
    if (overnightIntradaySplit.length === 0) return 0;
    const sum = overnightIntradaySplit.reduce((acc, d) => acc + d.overnightRatioPct, 0);
    return sum / overnightIntradaySplit.length;
  }, [overnightIntradaySplit]);

  // Jump statistics
  const jumpStats = React.useMemo(() => {
    if (dailyRealizedVol.length === 0) return { jumpDays: 0, totalDays: 0, maxZ: 0 };
    const significantDays = dailyRealizedVol.filter((d) => d.hasSignificantJump).length;
    const maxZ = Math.max(...dailyRealizedVol.map((d) => d.jumpZStat));
    return {
      jumpDays: significantDays,
      totalDays: dailyRealizedVol.length,
      maxZ: isFinite(maxZ) ? maxZ : 0,
    };
  }, [dailyRealizedVol]);

  const nNews = newsImpactPoints.length;
  const activeNewsIdx = newsHoverIdx !== null && newsHoverIdx < nNews ? newsHoverIdx : Math.floor(nNews / 2);
  const activeNewsPoint = newsImpactPoints[activeNewsIdx] || { shockStd: 0, symmetricVariance: 0, asymmetricVarianceGjr: 0 };

  const { newsTraces, newsLayout } = useMemo<{
    newsTraces: Data[];
    newsLayout: Partial<Layout>;
  }>(() => {
    if (newsImpactPoints.length === 0) return { newsTraces: [], newsLayout: {} };
    const xs = newsImpactPoints.map((p) => p.shockStd);
    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Symmetric GARCH(1,1)',
        x: xs,
        y: newsImpactPoints.map((p) => p.symmetricVariance),
        line: { color: '#64748b', width: 1.5, dash: 'dash' },
        hovertemplate: 'Symmetric: %{y:.4f}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'GJR close→close',
        x: xs,
        y: newsImpactPoints.map((p) => p.asymmetricVarianceGjr),
        line: { color: '#10b981', width: 2.2 },
        hovertemplate: 'GJR c→c: %{y:.4f}<extra></extra>',
      },
    ];
    if (newsImpactOcPoints && newsImpactOcPoints.length > 1) {
      data.push({
        type: 'scatter',
        mode: 'lines',
        name: 'GJR open→close',
        x: newsImpactOcPoints.map((p) => p.shockStd),
        y: newsImpactOcPoints.map((p) => p.asymmetricVarianceGjr),
        line: { color: QUANT_COLORS.gold, width: 1.9, dash: 'dashdot' },
        hovertemplate: 'GJR o→c: %{y:.4f}<extra></extra>',
      });
    }
    return {
      newsTraces: data,
      newsLayout: {
        margin: { l: 56, r: 18, t: 10, b: 36 },
        shapes: [
          {
            type: 'line',
            xref: 'x',
            x0: 0,
            x1: 0,
            yref: 'paper',
            y0: 0,
            y1: 1,
            line: { color: tokens.zeroLine, width: 1, dash: 'dot' },
          },
        ],
        xaxis: {
          ticksuffix: 'σ',
          title: { text: 'Return Shock (σ)', font: { size: 10, color: tokens.textMuted } },
        },
        yaxis: {
          title: { text: 'Variance', font: { size: 10, color: tokens.textMuted } },
          rangemode: 'tozero',
        },
      },
    };
  }, [newsImpactPoints, newsImpactOcPoints, tokens]);

  const nSig = volSignaturePoints.length;
  const activeSigIdx = sigHoverIdx !== null && sigHoverIdx < nSig ? sigHoverIdx : 0;
  const activeSigPoint = volSignaturePoints[activeSigIdx] || { samplingMinutes: 1, annualizedRv: 0 };

  const { sigTraces, sigLayout } = useMemo<{
    sigTraces: Data[];
    sigLayout: Partial<Layout>;
  }>(() => {
    if (volSignaturePoints.length === 0) return { sigTraces: [], sigLayout: {} };
    return {
      sigTraces: [
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Annualized RV',
          x: volSignaturePoints.map((p) => `${p.samplingMinutes}m`),
          y: volSignaturePoints.map((p) => p.annualizedRv * 100),
          line: { color: QUANT_COLORS.gold, width: 2 },
          marker: { size: 7, color: QUANT_COLORS.gold, line: { color: isDark ? '#08080A' : '#FFFFFF', width: 1 } },
          hovertemplate: '%{x}: <b>%{y:.2f}%</b><extra></extra>',
        },
      ],
      sigLayout: {
        margin: { l: 54, r: 16, t: 10, b: 36 },
        xaxis: {
          type: 'category',
          title: { text: 'Sampling Interval', font: { size: 10, color: tokens.textMuted } },
        },
        yaxis: {
          title: { text: 'Ann. RV', font: { size: 10, color: tokens.textMuted } },
          ticksuffix: '%',
        },
      },
    };
  }, [volSignaturePoints, isDark, tokens]);

  const nSplit = overnightIntradaySplit.length;
  const { splitTraces, splitLayout } = useMemo<{
    splitTraces: Data[];
    splitLayout: Partial<Layout>;
  }>(() => {
    if (overnightIntradaySplit.length === 0) return { splitTraces: [], splitLayout: {} };
    const xs = overnightIntradaySplit.map((d) => d.date);
    const hasEarningsInWindow =
      earningsCurrent !== null && overnightIntradaySplit.some((d) => d.date === earningsCurrent.date);

    const shapes: Layout['shapes'] = hasEarningsInWindow && earningsCurrent
      ? [
          {
            type: 'line',
            xref: 'x',
            x0: earningsCurrent.date,
            x1: earningsCurrent.date,
            yref: 'paper',
            y0: 0,
            y1: 1,
            line: { color: QUANT_COLORS.gold, width: 1.4, dash: 'dash' },
          },
        ]
      : [];

    const annotations: Layout['annotations'] = hasEarningsInWindow && earningsCurrent
      ? [
          {
            x: earningsCurrent.date,
            y: 0.98,
            xref: 'x',
            yref: 'paper',
            text: 'EARNINGS',
            showarrow: false,
            font: { size: 9, color: QUANT_COLORS.gold, family: 'IBM Plex Mono, monospace' },
          },
        ]
      : [];

    return {
      splitTraces: [
        {
          type: 'bar',
          name: 'Intraday RV',
          x: xs,
          y: overnightIntradaySplit.map((d) => d.intradayRv),
          marker: { color: 'rgba(16,185,129,0.82)' },
          hovertemplate: 'Intraday RV: %{y:.5f}<extra></extra>',
        },
        {
          type: 'bar',
          name: 'Overnight Gap Var',
          x: xs,
          y: overnightIntradaySplit.map((d) => d.overnightVar),
          marker: { color: 'rgba(222,192,132,0.85)' },
          hovertemplate: 'Overnight Gap Var: %{y:.5f}<extra></extra>',
        },
      ],
      splitLayout: {
        barmode: 'stack',
        margin: { l: 54, r: 16, t: 10, b: 36 },
        shapes,
        annotations,
        yaxis: {
          title: { text: 'Daily Variance', font: { size: 10, color: tokens.textMuted } },
          rangemode: 'tozero',
        },
      },
    };
  }, [overnightIntradaySplit, earningsCurrent, tokens]);

  return (
    <div className="space-y-6">
      {/* GJR-GARCH News Impact Curve */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                News Impact Curve (Symmetric GARCH vs. Asymmetric GJR-GARCH)
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Empirical asymmetric volatility response: negative return shocks trigger larger conditional variance spikes
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-zinc-400">
            <span>
              Shock: <strong>{activeNewsPoint.shockStd > 0 ? `+${activeNewsPoint.shockStd}` : activeNewsPoint.shockStd}σ</strong>
            </span>
            <span className="text-zinc-300 dark:text-zinc-700">·</span>
            <span>
              GJR Var: <strong className="text-gold-500">{activeNewsPoint.asymmetricVarianceGjr.toFixed(2)}</strong>
            </span>
            <NewsImpactEquation className="hidden lg:inline" />
          </div>
        </div>

        <div className="mt-2">
          <ChartFullscreen title="News Impact Curve (Symmetric GARCH vs. Asymmetric GJR-GARCH)">
            <PlotlyChart
              data={newsTraces}
              layout={newsLayout}
              plotType="2d"
              hovermode="x unified"
              showLegend={false}
              height={240}
              showWheelHint
              exportFilename={`${ticker}-news-impact-curve`}
              uirevisionKey={`nic-${ticker}-${nNews}`}
              onHover={(ev) => {
                const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                if (pt && typeof pt.pointIndex === 'number') setNewsHoverIdx(pt.pointIndex);
              }}
              onUnhover={() => setNewsHoverIdx(null)}
            />
          </ChartFullscreen>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-zinc-400 font-mono mt-1">
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-zinc-400 border-b border-dashed inline-block" /> Symmetric GARCH(1,1)
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-emerald-500 inline-block" /> GJR close→close (γ {gjrGammaCloseClose !== null && gjrGammaCloseClose !== undefined ? gjrGammaCloseClose.toFixed(3) : '—'})
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0 border-t-2 border-dashed border-gold-400 inline-block" /> GJR open→close (γ {gjrGammaOpenClose !== null && gjrGammaOpenClose !== undefined ? gjrGammaOpenClose.toFixed(3) : 'n/a'})
            </span>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>
              Two curves, two mechanisms — the Black (1976) leverage hypothesis, tested honestly.
              γ survives removal of the overnight gap ⇒ asymmetry is <em>continuous intraday flow</em> — positioning/liquidity, persistent only while the imbalance lasts.
              γ collapses when the gap is removed ⇒ the leverage effect is an <em>event/gap phenomenon</em> — news between sessions.
              {gjrGammaCloseClose !== null && gjrGammaCloseClose !== undefined && gjrGammaOpenClose !== null && gjrGammaOpenClose !== undefined
                ? ` Δγ = ${(gjrGammaOpenClose - gjrGammaCloseClose) >= 0 ? '+' : ''}${(gjrGammaOpenClose - gjrGammaCloseClose).toFixed(3)}.`
                : ' Open→close fit needs ≥ 60 sessions — currently unavailable.'}
            </span>
          </div>
        </div>
      </div>

      {/* Volatility Signature Plot & Overnight vs Intraday Split */}
      <div className="grid-cards">
        {/* Volatility Signature Plot */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Volatility Signature Plot (Andersen et al., 2000)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Sampling frequency vs. Realized Volatility: isolates microstructure noise bias
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              {activeSigPoint.samplingMinutes}m: <strong className="text-gold-600 dark:text-gold-300">{(activeSigPoint.annualizedRv * 100).toFixed(2)}%</strong>
            </div>
          </div>

          <div className="mt-2">
            <ChartFullscreen title="Volatility Signature Plot — Realized Vol vs. Sampling Interval">
              <PlotlyChart
                data={sigTraces}
                layout={sigLayout}
                plotType="categorical"
                hovermode="x unified"
                showLegend={false}
                height={230}
                showWheelHint
                exportFilename={`${ticker}-volatility-signature`}
                uirevisionKey={`vol-sig-${ticker}-${nSig}`}
                onHover={(ev) => {
                  const pt = (ev as { points?: Array<{ pointIndex?: number }> })?.points?.[0];
                  if (pt && typeof pt.pointIndex === 'number') setSigHoverIdx(pt.pointIndex);
                }}
                onUnhover={() => setSigHoverIdx(null)}
              />
            </ChartFullscreen>
          </div>

          <div className="mt-3 text-[11px] text-zinc-400">
            Microstructure noise inflates 1-minute RV; the curve stabilizes around the 5–15 min frequency, which provides
            the optimal bias-variance tradeoff for intraday pricing.
          </div>
        </div>

        {/* Overnight vs Intraday Split */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Overnight vs. Intraday Variance Decomposition
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Earnings releases &amp; macro news break outside market hours
              </p>
            </div>
            <div className="flex items-center gap-3 font-mono text-xs">
              {earningsCurrent && (
                <span
                  title="Per Yahoo quoteSummary (estimated) — drifts; not investment advice"
                  className="rounded border border-gold-500/40 bg-gold-500/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-gold-800 dark:text-gold-200"
                >
                  Next earnings {earningsCurrent.date}{earningsCurrent.isEstimate ? ' (est.)' : ''}
                </span>
              )}
              <div className="text-gold-600 dark:text-gold-400">
                Avg Overnight Share: <strong>{avgOvernightPct.toFixed(1)}%</strong>
              </div>
            </div>
          </div>

          <div className="mt-2">
            <ChartFullscreen title="Overnight vs. Intraday Variance Split">
              <PlotlyChart
                data={splitTraces}
                layout={splitLayout}
                plotType="2d"
                rangeSlider
                hovermode="x unified"
                showLegend={false}
                height={230}
                showWheelHint
                exportFilename={`${ticker}-overnight-intraday-split`}
                uirevisionKey={`split-${ticker}-${nSplit}`}
              />
            </ChartFullscreen>
          </div>

          <div className="flex items-center justify-center text-[11px] text-zinc-400 font-mono mt-1">
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-[#10b981] rounded-xs inline-block" /> Intraday RV
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-[#DEC084] rounded-xs inline-block" /> Overnight Gap Var
              </span>
            </div>
          </div>

          <div className="mt-3 text-[11px] text-zinc-400">
            Overnight variance represents gap risk unhedged by standard intraday stops. High overnight ratios favor option
            straddles over delta-neutral spot market-making.
          </div>
        </div>
      </div>

      {/* Intraday 390-min Diurnal Profile & Jump Decomposition */}
      <div className="grid-cards">
        {/* Diurnal U-Shape Profile */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                390-Minute Intraday Diurnal Volatility Profile
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                U-shaped intraday volatility smile: open rush → midday lull → market-on-close (MOC)
              </p>
            </div>
            <Clock className="w-4 h-4 text-gold-500" />
          </div>
          {(() => {
            const ratio = diurnalReconciliation?.ratio ?? null;
            const badRatio = ratio !== null && (ratio > 1.5 || ratio < 0.5);
            const synthetic = intradaySource !== 'live_market';
            if (!badRatio && !synthetic) return null;
            return (
              <div className="mt-2 flex flex-wrap items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.14em] text-amber-700 dark:text-amber-300">
                <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {synthetic && <span>Intraday bars: {intradaySource ?? 'not live'} — panel is illustrative.</span>}
                {badRatio && (
                  <span>Diurnal mean ÷ intraday RV = {ratio !== null ? `${ratio.toFixed(2)}×` : 'n/a'} — outside the 0.5–1.5 sane band. Do not size off this panel until it reconciles.</span>
                )}
              </div>
            );
          })()}
          <div className="mt-4">
            <ChartFullscreen title="390-Minute Intraday Diurnal Volatility Profile" subtitle={ticker}>
              <IntradayDiurnalChart points={diurnalProfile} ticker={ticker} />
            </ChartFullscreen>
          </div>
        </div>

        {/* Jump Decomposition */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Bipower Variation &amp; Jump Shock Decomposition
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Barndorff-Nielsen &amp; Shephard: RV = Continuous BV + Discrete Jump J_t
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              Significant Jumps: <strong className="text-gold-500">{jumpStats.jumpDays}</strong> / {jumpStats.totalDays} days
            </div>
          </div>

          <div className="mt-4">
            <ChartFullscreen title="Bipower Variation & Jump Shock Decomposition" subtitle={ticker}>
              <JumpDecompositionChart dailyRV={dailyRealizedVol} ticker={ticker} />
            </ChartFullscreen>
          </div>
        </div>
      </div>
    </div>
  );
}
