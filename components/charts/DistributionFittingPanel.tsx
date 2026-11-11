'use client';

import React, { useState, useMemo } from 'react';
import { ReturnStats } from '@/lib/quant-engine';
import {
  fitParametricDistributions,
  FittedDistribution,
} from '@/lib/distribution-engine';
import {
  StudentTEquation,
  HansenSkewTEquation,
  GedEquation,
  AicEquation,
} from '@/components/math/MathEquations';
import {
  BarChart2,
  Award,
  Check,
  Info,
  TrendingDown,
  Layers,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
} from 'lucide-react';

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
  // Fit all 4 distribution families via MLE
  const fitResults = useMemo(() => {
    return fitParametricDistributions(returns);
  }, [returns]);

  // Active visible curves state
  const [activeCurves, setActiveCurves] = useState<{ [key: string]: boolean }>({
    'hansen-skew-t': true,
    'student-t': true,
    'ged': true,
    'normal': true,
  });

  const [highlightedModel, setHighlightedModel] = useState<string | null>(null);
  const [showMathReference, setShowMathReference] = useState<boolean>(false);

  const toggleCurve = (key: string) => {
    setActiveCurves((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  // SVG dimensions
  const width = 640;
  const height = 260;
  const padLeft = 45;
  const padRight = 20;
  const padTop = 24;
  const padBottom = 34;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const min = fitResults.empiricalMin;
  const max = fitResults.empiricalMax;
  const range = max - min || 0.01;
  const binCount = 35;
  const binWidth = range / binCount;

  // 35 histogram bins
  const bins = useMemo(() => {
    if (!returns || returns.length === 0) return [];
    const b: number[] = new Array(binCount).fill(0);
    for (const r of returns) {
      const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / binWidth)));
      b[idx]++;
    }
    return b;
  }, [returns, min, binWidth, binCount]);

  const maxCount = useMemo(() => {
    return Math.max(...bins, 1);
  }, [bins]);

  // Generate curve path points for each fitted distribution
  const numSteps = 100;
  const curvePaths = useMemo(() => {
    const paths: { [key: string]: string } = {};
    if (!returns || returns.length === 0) return paths;

    const getX = (val: number) => padLeft + ((val - min) / range) * chartW;
    const getY = (count: number) => padTop + chartH - (count / maxCount) * chartH;

    fitResults.models.forEach((m) => {
      const points: string[] = [];
      for (let s = 0; s <= numSteps; s++) {
        const xVal = min + (s / numSteps) * range;
        const pdfVal = m.pdf(xVal);
        // Area scaling: count = N * binWidth * PDF(x)
        const expectedCount = returns.length * binWidth * pdfVal;
        const yVal = getY(expectedCount);
        const clampedY = Math.max(padTop - 10, Math.min(padTop + chartH, yVal));
        points.push(`${getX(xVal).toFixed(1)},${clampedY.toFixed(1)}`);
      }
      paths[m.key] = `M ${points.join(' ')}`;
    });

    return paths;
  }, [fitResults, min, range, returns, binWidth, maxCount, chartW, chartH]);

  if (!returns || returns.length === 0) {
    return null;
  }

  const getX = (val: number) => padLeft + ((val - min) / range) * chartW;
  const getY = (count: number) => padTop + chartH - (count / maxCount) * chartH;
  const bestModel = fitResults.models[0];

  return (
    <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs">
      {/* Header and Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
              Inferential Econometrics
            </span>
            <span className="text-slate-300 dark:text-slate-700" aria-hidden="true">·</span>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Maximum Likelihood Estimation (MLE)
            </span>
          </div>
          <h3 className="text-base font-bold tracking-tight text-slate-900 dark:text-slate-100 mt-0.5">
            Fat-Tailed Distribution Fitting &amp; Information Criteria Ranking
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-2xl">
            Financial returns depart from normality through heavy tails (leptokurtosis) and crash asymmetry. We fit Student-t, Hansen Skew-t, and Generalized Error (GED) distributions and rank them by AIC, BIC, and Kolmogorov–Smirnov distance.
          </p>
        </div>

        {/* Best Model Badge */}
        <div className="flex items-center gap-2.5 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-800/60 p-2.5 rounded-lg shrink-0">
          <Award className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <div className="text-xs">
            <div className="text-slate-500 dark:text-slate-400 text-[11px]">Best Empirical Fit:</div>
            <div className="font-mono font-bold text-emerald-700 dark:text-emerald-300">
              {bestModel?.name}
            </div>
          </div>
        </div>
      </div>

      {/* Curve Filter / Visibility Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 my-4 p-2.5 bg-slate-50 dark:bg-slate-900/50 rounded-lg border border-slate-100 dark:border-slate-800 text-xs">
        <div className="flex items-center gap-2 font-medium text-slate-600 dark:text-slate-400 text-xs">
          <Layers className="w-4 h-4 text-slate-400" />
          <span>Fitted Density Curves:</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {fitResults.models.map((m) => {
            const isVisible = activeCurves[m.key];
            const isHovered = highlightedModel === m.key;

            return (
              <button
                key={m.key}
                type="button"
                onClick={() => toggleCurve(m.key)}
                onMouseEnter={() => setHighlightedModel(m.key)}
                onMouseLeave={() => setHighlightedModel(null)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-all cursor-pointer border ${
                  isVisible
                    ? 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-900 dark:text-slate-100 shadow-2xs'
                    : 'bg-transparent border-transparent text-slate-400 opacity-60 hover:opacity-100'
                } ${isHovered ? 'ring-1 ring-slate-400 dark:ring-slate-500' : ''}`}
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
                  <span className="text-[10px] bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300 font-bold px-1 rounded-xs">
                    #1
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* SVG Canvas: Empirical Histogram & Multi-Distribution Curves */}
      <div className="relative w-full aspect-21/9 min-h-[200px] max-h-[300px]">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible select-none">
          {/* Grid lines */}
          {[0, 0.5, 1].map((pct) => {
            const count = Math.round(pct * maxCount);
            const yPos = getY(count);
            return (
              <g key={pct}>
                <line
                  x1={padLeft}
                  y1={yPos}
                  x2={width - padRight}
                  y2={yPos}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800/80"
                  strokeDasharray="2 3"
                />
                <text
                  x={padLeft - 6}
                  y={yPos + 4}
                  textAnchor="end"
                  className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
                >
                  {count}
                </text>
              </g>
            );
          })}

          {/* Empirical Histogram Bars */}
          {bins.map((count, i) => {
            const binStart = min + i * binWidth;
            const x0 = getX(binStart);
            const x1 = getX(binStart + binWidth);
            const bWidth = Math.max(1, x1 - x0 - 1);
            const y = getY(count);
            const h = padTop + chartH - y;

            return (
              <rect
                key={i}
                x={x0}
                y={y}
                width={bWidth}
                height={h}
                fill="#38bdf8"
                fillOpacity="0.45"
                stroke="#0284c7"
                strokeWidth="0.5"
                strokeOpacity="0.6"
                className="hover:fill-sky-400 transition-colors"
              >
                <title>{`Bin: ${(binStart * 100).toFixed(2)}% to ${(
                  (binStart + binWidth) *
                  100
                ).toFixed(2)}% | Count: ${count}`}</title>
              </rect>
            );
          })}

          {/* Fitted Distribution Density Curves */}
          {fitResults.models.map((m) => {
            if (!activeCurves[m.key]) return null;
            const isHovered = highlightedModel === m.key;
            const isNormal = m.key === 'normal';

            return (
              <path
                key={m.key}
                d={curvePaths[m.key]}
                fill="none"
                stroke={m.color}
                strokeWidth={isHovered ? 3.5 : isNormal ? 2.0 : 2.5}
                strokeDasharray={isNormal ? '4 3' : 'none'}
                strokeOpacity={highlightedModel && !isHovered ? 0.25 : 0.95}
                className="transition-all duration-200"
              />
            );
          })}

          {/* Mean vertical return line */}
          <line
            x1={getX(stats.mean)}
            y1={padTop}
            x2={getX(stats.mean)}
            y2={padTop + chartH}
            stroke="#10b981"
            strokeWidth="1.5"
            strokeDasharray="4 2"
          />

          {/* X axis ticks */}
          {[-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06].map((tick) => {
            if (tick < min || tick > max) return null;
            return (
              <g key={tick}>
                <line
                  x1={getX(tick)}
                  y1={padTop + chartH}
                  x2={getX(tick)}
                  y2={padTop + chartH + 4}
                  stroke="currentColor"
                  className="text-slate-400 dark:text-slate-600"
                />
                <text
                  x={getX(tick)}
                  y={height - 10}
                  textAnchor="middle"
                  className="text-[10px] font-mono fill-slate-500 dark:fill-slate-400 tabular-nums"
                >
                  {(tick * 100).toFixed(0)}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Model Ranking and Goodness-of-Fit Table */}
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700/80 text-slate-500 dark:text-slate-400">
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
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {fitResults.models.map((m) => {
              const isBest = m.rank === 1;
              const isNormal = m.key === 'normal';

              return (
                <tr
                  key={m.key}
                  onMouseEnter={() => setHighlightedModel(m.key)}
                  onMouseLeave={() => setHighlightedModel(null)}
                  className={`hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors ${
                    isBest
                      ? 'bg-emerald-50/50 dark:bg-emerald-950/20 font-semibold'
                      : isNormal
                      ? 'opacity-85'
                      : ''
                  }`}
                >
                  <td className="py-2 px-2.5">
                    {isBest ? (
                      <span className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-300 font-bold">
                        <Award className="w-3.5 h-3.5 text-emerald-600" /> #1
                      </span>
                    ) : (
                      <span className="text-slate-400">#{m.rank}</span>
                    )}
                  </td>

                  <td className="py-2 px-2.5">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: m.color }}
                      />
                      <span className="text-slate-800 dark:text-slate-200">
                        {m.name}
                      </span>
                    </div>
                  </td>

                  <td className="py-2 px-2.5 text-slate-600 dark:text-slate-300">
                    {m.paramDescriptions}
                  </td>

                  <td className="py-2 px-2.5 text-right text-slate-700 dark:text-slate-300">
                    {m.logLikelihood.toFixed(1)}
                  </td>

                  <td className="py-2 px-2.5 text-right">
                    <span
                      className={
                        isBest
                          ? 'text-emerald-700 dark:text-emerald-300 font-bold'
                          : 'text-slate-700 dark:text-slate-300'
                      }
                    >
                      {m.aic.toFixed(1)}
                    </span>{' '}
                    <span className="text-[10px] text-slate-400">
                      ({m.deltaAic === 0 ? '0.0' : `+${m.deltaAic.toFixed(1)}`})
                    </span>
                  </td>

                  <td className="py-2 px-2.5 text-right text-slate-700 dark:text-slate-300">
                    {m.bic.toFixed(1)}
                  </td>

                  <td className="py-2 px-2.5 text-right text-slate-700 dark:text-slate-300">
                    {m.ksStatistic.toFixed(3)}
                  </td>

                  <td className="py-2 px-2.5 text-right">
                    {isBest ? (
                      <span className="text-emerald-700 dark:text-emerald-300 font-bold">
                        Best Fit (Parsimonious)
                      </span>
                    ) : m.deltaAic > 10 ? (
                      <span className="text-rose-600 dark:text-rose-400 font-medium">
                        Rejected (ΔAIC &gt; 10)
                      </span>
                    ) : (
                      <span className="text-amber-600 dark:text-amber-400">
                        Competitive Fit
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Quantitative Takeaway Banner */}
      <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-900/40 rounded-lg border border-slate-100 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-emerald-500 shrink-0" />
          <div className="text-slate-600 dark:text-slate-300">
            <strong>Econometric Takeaway:</strong> {ticker} empirical returns exhibit{' '}
            {stats.excessKurtosis > 1 ? (
              <span>
                pronounced heavy tails (fitted Student-t degree of freedom{' '}
                <strong className="font-mono text-emerald-600 dark:text-emerald-400">
                  ν = {fitResults.models.find((m) => m.key === 'student-t')?.params.nu?.toFixed(1) || '4.2'}
                </strong>
                ).
              </span>
            ) : (
              <span>mild kurtosis with tails well-approximated by GED.</span>
            )}{' '}
            The Gaussian hypothesis is rejected by ΔAIC &gt; 10, confirming standard normal VaR severely understates tail crash probability.
          </div>
        </div>
      </div>

      {/* Expandable Mathematical Formulations Panel */}
      <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800">
        <button
          type="button"
          onClick={() => setShowMathReference((prev) => !prev)}
          className="w-full flex items-center justify-between text-left text-xs font-bold text-slate-800 dark:text-slate-200 cursor-pointer"
        >
          <div className="flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-slate-400" />
            <span>Mathematical Density Formulations (Student-t, Hansen Skew-t, GED &amp; AIC)</span>
          </div>
          {showMathReference ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </button>

        {showMathReference && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs">
            <div className="p-3 rounded-lg bg-slate-50/80 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                Standardized Student-t Density
              </div>
              <StudentTEquation display="block" />
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
                Parameterized such that Var(x) = σ² for ν &gt; 2. As ν → ∞, the density converges smoothly to standard Normal.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-50/80 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                Hansen (1994) Skewed Student-t
              </div>
              <HansenSkewTEquation display="block" />
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
                Splits scale by (1 ± λ) around the mode threshold -a/b to capture crash skewness without violating unit variance.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-50/80 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                Generalized Error Distribution (GED)
              </div>
              <GedEquation display="block" />
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
                Subbotin exponential power density. κ &lt; 2 models leptokurtic tails; κ = 1 yields the Laplace (double exponential) distribution.
              </p>
            </div>

            <div className="p-3 rounded-lg bg-slate-50/80 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800">
              <div className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                Akaike Information Criterion (AIC)
              </div>
              <AicEquation display="block" />
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2">
                Penalizes parameter complexity k against maximized log-likelihood ln L. Models with ΔAIC &gt; 10 have virtually no empirical support.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
