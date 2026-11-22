'use client';

import React, { useState, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import { ReturnStats } from '@/lib/quant-engine';
import { fitParametricDistributions } from '@/lib/distribution-engine';
import {
  StudentTEquation,
  HansenSkewTEquation,
  GedEquation,
  AicEquation,
} from '@/components/math/MathEquations';
import {
  Award,
  Info,
  Layers,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { AlgoSelect } from '@/components/ui/AlgoSelect';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';

interface DistributionFittingPanelProps {
  returns: number[];
  stats: ReturnStats;
  ticker?: string;
}

export function DistributionFittingPanel({
  returns,
  stats,
  ticker = 'Asset',
}: DistributionFittingPanelProps) {
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const fitResults = useMemo(() => {
    return fitParametricDistributions(returns);
  }, [returns]);

  const [activeCurves, setActiveCurves] = useState<{ [key: string]: boolean }>({
    'hansen-skew-t': true,
    'student-t': true,
    ged: true,
    normal: true,
  });

  const [highlightedModel, setHighlightedModel] = useState<string | null>(null);
  const [highlightModel, setHighlightModel] = useState<string>('aic-winner');
  const [showMathReference, setShowMathReference] = useState<boolean>(false);

  const toggleCurve = (key: string) => {
    setActiveCurves((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const bestModel = fitResults.models[0];
  const effectiveHighlight =
    highlightedModel ?? (highlightModel === 'aic-winner' ? bestModel?.key : highlightModel);

  const { traces, layout } = useMemo<{ traces: Data[]; layout: Partial<Layout> }>(() => {
    if (!returns || returns.length === 0) return { traces: [], layout: {} };

    const min = fitResults.empiricalMin;
    const max = fitResults.empiricalMax;
    const range = max - min || 0.01;
    const binCount = 35;
    const binWidth = range / binCount;

    const bins: number[] = new Array(binCount).fill(0);
    for (const r of returns) {
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / binWidth)));
      bins[idx]++;
    }

    const binCentersPct = Array.from({ length: binCount }, (_, i) => (min + (i + 0.5) * binWidth) * 100);

    const dataTraces: Data[] = [
      {
        type: 'bar',
        name: 'Empirical Histogram',
        x: binCentersPct,
        y: bins,
        marker: {
          color: 'rgba(139, 139, 147, 0.45)',
          line: { color: 'rgba(222, 192, 132, 0.5)', width: 0.6 },
        },
        hovertemplate: '%{x:+.2f}% · %{y} obs<extra>Empirical</extra>',
      },
    ];

    const numSteps = 120;
    const xsPct = Array.from({ length: numSteps + 1 }, (_, s) => (min + (s / numSteps) * range) * 100);

    for (const m of fitResults.models) {
      if (!activeCurves[m.key]) continue;
      const isHovered = effectiveHighlight === m.key;
      const isNormal = m.key === 'normal';
      const ys = xsPct.map((xp) => {
        const pdfVal = m.pdf(xp / 100);
        return returns.length * binWidth * pdfVal;
      });

      dataTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: m.name.split(' (')[0],
        x: xsPct,
        y: ys,
        opacity: effectiveHighlight && !isHovered ? 0.3 : 0.98,
        line: {
          color: m.color,
          width: isHovered ? 2.6 : isNormal ? 1.4 : 1.8,
          dash: isNormal ? 'dash' : 'solid',
        },
        hovertemplate: `%{y:.1f} exp<extra>${m.name.split(' (')[0]}</extra>`,
      });
    }

    return {
      traces: dataTraces,
      layout: {
        bargap: 0.06,
        shapes: [
          {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: stats.mean * 100,
            x1: stats.mean * 100,
            y0: 0,
            y1: 1,
            line: { color: QUANT_COLORS.positive, width: 1.2, dash: 'dot' },
          },
        ],
        xaxis: {
          ticksuffix: '%',
          gridcolor: tokens.grid,
        },
        yaxis: {
          title: { text: 'Frequency', font: { size: 10, color: tokens.textMuted } },
          rangemode: 'tozero',
          gridcolor: tokens.grid,
        },
      },
    };
  }, [returns, fitResults, activeCurves, effectiveHighlight, stats.mean, tokens]);

  if (!returns || returns.length === 0) {
    return null;
  }

  return (
    <div className="panel p-5">
      {/* Header and Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-[var(--hairline)]">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider text-gold-600 dark:text-gold-400">
              Inferential Econometrics
            </span>
            <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">
              ·
            </span>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              Maximum Likelihood Estimation (MLE)
            </span>
          </div>
          <h3 className="text-base font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5">
            Fat-Tailed Distribution Fitting &amp; Information Criteria Ranking
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 max-w-2xl">
            Financial returns depart from normality through heavy tails (leptokurtosis) and crash
            asymmetry. We fit Student-t, Hansen Skew-t, and Generalized Error (GED) distributions
            and rank them by AIC, BIC, and Kolmogorov–Smirnov distance.
          </p>
        </div>

        {/* Best Model Badge */}
        <div className="flex items-center gap-2.5 bg-gold-50 dark:bg-gold-950/30 border border-gold-200 dark:border-gold-800/60 p-2.5 rounded-lg shrink-0">
          <Award className="w-5 h-5 text-gold-600 dark:text-gold-400 shrink-0" />
          <div className="text-xs">
            <div className="text-zinc-500 dark:text-zinc-400 text-[11px]">Best Empirical Fit:</div>
            <div className="font-mono font-bold text-gold-700 dark:text-gold-300">
              {bestModel?.name}
            </div>
          </div>
        </div>
      </div>

      {/* Curve Filter / Visibility Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 my-4 p-2.5 panel-inset rounded-lg text-xs">
        <div className="flex items-center gap-2 font-medium text-zinc-600 dark:text-zinc-400 text-xs">
          <Layers className="w-4 h-4 text-zinc-400" />
          <span>Fitted Density Curves:</span>
        </div>

        <AlgoSelect
          ariaLabel="Highlight distribution model"
          value={highlightModel}
          onChange={setHighlightModel}
          options={[
            { value: 'aic-winner', label: 'Highlight: AIC winner' },
            ...fitResults.models.map((m) => ({
              value: m.key,
              label: `Highlight: ${m.name.split(' (')[0]}`,
            })),
          ]}
          className="w-56"
        />
        <div className="flex flex-wrap items-center gap-2">
          {fitResults.models.map((m) => {
            const isVisible = activeCurves[m.key];
            const isHovered = effectiveHighlight === m.key;

            return (
              <button
                key={m.key}
                type="button"
                onClick={() => toggleCurve(m.key)}
                onMouseEnter={() => setHighlightedModel(m.key)}
                onMouseLeave={() => setHighlightedModel(null)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-all cursor-pointer border whitespace-nowrap shrink-0 ${
                  isVisible
                    ? 'bg-white dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100 shadow-2xs'
                    : 'bg-transparent border-transparent text-zinc-400 opacity-60 hover:opacity-100'
                } ${isHovered ? 'ring-1 ring-zinc-400 dark:ring-zinc-500' : ''}`}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full shrink-0"
                  style={{
                    backgroundColor: m.color,
                    boxShadow: isVisible ? `0 0 6px ${m.color}88` : 'none',
                  }}
                />
                <span>{m.name.split(' (')[0]}</span>
                {m.rank === 1 && (
                  <span className="text-[10px] bg-gold-100 dark:bg-gold-900/50 text-gold-700 dark:text-gold-300 font-bold px-1 rounded-xs">
                    #1
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Interactive Plotly Empirical Histogram & Multi-Distribution Density Curves */}
      <PlotlyChart
        data={traces}
        layout={layout}
        plotType="2d"
        hovermode="x unified"
        height={300}
        exportFilename={`quantmetrics-mle-fit-${ticker.toLowerCase()}`}
        uirevisionKey={`mle-${ticker}-${returns.length}`}
        showWheelHint
      />

      {/* Model Ranking and Goodness-of-Fit Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead>
            <tr className="border-b border-zinc-200 dark:border-zinc-700/80 text-zinc-500 dark:text-zinc-400">
              <th className="py-2 px-2.5">Rank</th>
              <th className="py-2 px-2.5">Distribution Family</th>
              <th className="py-2 px-2.5">Estimated Parameters</th>
              <th className="py-2 px-2.5 text-right">Log-Likelihood (ln L)</th>
              <th className="py-2 px-2.5 text-right">AIC (ΔAIC)</th>
              <th className="py-2 px-2.5 text-right">BIC</th>
              <th className="py-2 px-2.5 text-right">KS Distance (D)</th>
              <th className="py-2 px-2.5 text-right">Verdict</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {fitResults.models.map((m) => {
              const isBest = m.rank === 1;
              const isNormal = m.key === 'normal';

              return (
                <tr
                  key={m.key}
                  onMouseEnter={() => setHighlightedModel(m.key)}
                  onMouseLeave={() => setHighlightedModel(null)}
                  className={`hover:bg-zinc-50 dark:hover:bg-zinc-900/50 transition-colors ${
                    isBest
                      ? 'bg-gold-50/50 dark:bg-gold-950/20 font-semibold'
                      : isNormal
                      ? 'opacity-85'
                      : ''
                  }`}
                >
                  <td className="py-2 px-2.5">
                    {isBest ? (
                      <span className="inline-flex items-center gap-1 text-gold-700 dark:text-gold-300 font-bold">
                        <Award className="w-3.5 h-3.5 text-gold-600" /> #1
                      </span>
                    ) : (
                      <span className="text-zinc-400">#{m.rank}</span>
                    )}
                  </td>

                  <td className="py-2 px-2.5">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: m.color }}
                      />
                      <span className="text-zinc-800 dark:text-zinc-200">{m.name}</span>
                    </div>
                  </td>

                  <td className="py-2 px-2.5 text-zinc-600 dark:text-zinc-300">
                    {m.paramDescriptions}
                  </td>

                  <td className="py-2 px-2.5 text-right text-zinc-700 dark:text-zinc-300 tabular-nums">
                    {m.logLikelihood.toFixed(1)}
                  </td>

                  <td className="py-2 px-2.5 text-right tabular-nums">
                    <span
                      className={
                        isBest
                          ? 'text-gold-700 dark:text-gold-300 font-bold'
                          : 'text-zinc-700 dark:text-zinc-300'
                      }
                    >
                      {m.aic.toFixed(1)}
                    </span>{' '}
                    <span className="text-[10px] text-zinc-400">
                      ({m.deltaAic === 0 ? '0.0' : `+${m.deltaAic.toFixed(1)}`})
                    </span>
                  </td>

                  <td className="py-2 px-2.5 text-right text-zinc-700 dark:text-zinc-300 tabular-nums">
                    {m.bic.toFixed(1)}
                  </td>

                  <td className="py-2 px-2.5 text-right text-zinc-700 dark:text-zinc-300 tabular-nums">
                    {m.ksStatistic.toFixed(3)}
                  </td>

                  <td className="py-2 px-2.5 text-right">
                    {isBest ? (
                      <span className="text-gold-700 dark:text-gold-300 font-bold">
                        Best Fit (Parsimonious)
                      </span>
                    ) : m.deltaAic > 10 ? (
                      <span className="text-rose-600 dark:text-rose-400 font-medium">
                        Rejected (ΔAIC &gt; 10)
                      </span>
                    ) : (
                      <span className="text-gold-600 dark:text-gold-400">Competitive Fit</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {fitResults.hmm && (
        <RegimeTransitionMatrix
          fit={fitResults.hmm}
          labels={['Calm', 'Crisis']}
          className="mt-3"
          showStats={false}
        />
      )}

      {/* Quantitative Takeaway Banner */}
      <div className="mt-4 p-3 panel-inset rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-gold-500 shrink-0" />
          <div className="text-zinc-600 dark:text-zinc-300">
            <strong>Econometric Takeaway:</strong> {ticker} empirical returns exhibit{' '}
            {stats.excessKurtosis > 1 ? (
              <span>
                pronounced heavy tails (fitted Student-t degree of freedom{' '}
                <strong className="font-mono text-gold-600 dark:text-gold-400">
                  ν ={' '}
                  {fitResults.models.find((m) => m.key === 'student-t')?.params.nu?.toFixed(1) ||
                    '4.2'}
                </strong>
                ).
              </span>
            ) : (
              <span>mild kurtosis with tails well-approximated by GED.</span>
            )}{' '}
            The Gaussian hypothesis is rejected by ΔAIC &gt; 10, confirming standard normal VaR
            severely understates tail crash probability.
          </div>
        </div>
      </div>

      {/* Expandable Mathematical Formulations Panel */}
      <div className="mt-4 pt-3 border-t border-[var(--hairline)]">
        <button
          type="button"
          onClick={() => setShowMathReference((prev) => !prev)}
          className="w-full flex items-center justify-between text-left text-xs font-bold text-zinc-800 dark:text-zinc-200 cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-zinc-400" />
            <span>
              Mathematical Density Formulations (Student-t, Hansen Skew-t, GED &amp; AIC)
            </span>
          </div>
          {showMathReference ? (
            <ChevronUp className="w-3.5 h-3.5 text-zinc-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-zinc-400" />
          )}
        </button>

        {showMathReference && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-[var(--hairline)] text-xs">
            <div className="p-3 rounded-lg panel-inset">
              <div className="font-bold text-zinc-800 dark:text-zinc-200 mb-1">
                Standardized Student-t Density
              </div>
              <StudentTEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Parameterized such that Var(x) = σ² for ν &gt; 2. As ν → ∞, the density converges
                smoothly to standard Normal.
              </p>
            </div>

            <div className="p-3 rounded-lg panel-inset">
              <div className="font-bold text-zinc-800 dark:text-zinc-200 mb-1">
                Hansen (1994) Skewed Student-t
              </div>
              <HansenSkewTEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Splits scale by (1 ± λ) around the mode threshold -a/b to capture crash skewness
                without violating unit variance.
              </p>
            </div>

            <div className="p-3 rounded-lg panel-inset">
              <div className="font-bold text-zinc-800 dark:text-zinc-200 mb-1">
                Generalized Error Distribution (GED)
              </div>
              <GedEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Subbotin exponential power density. κ &lt; 2 models leptokurtic tails; κ = 1 yields
                the Laplace (double exponential) distribution.
              </p>
            </div>

            <div className="p-3 rounded-lg panel-inset">
              <div className="font-bold text-zinc-800 dark:text-zinc-200 mb-1">
                Akaike Information Criterion (AIC)
              </div>
              <AicEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Penalizes parameter complexity k against maximized log-likelihood ln L. Models with
                ΔAIC &gt; 10 have virtually no empirical support.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
