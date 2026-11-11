'use client';

import React, { useState } from 'react';
import { RegressionSummary } from '@/lib/quant-engine';

interface ResidualDiagnosticsChartProps {
  summary: RegressionSummary;
}

export function ResidualDiagnosticsChart({
  summary,
}: ResidualDiagnosticsChartProps) {
  const [activeTab, setActiveTab] = useState<'qq' | 'fitted' | 'histogram' | 'rolling'>('qq');

  const { residuals, rollingBetas, regressionType } = summary;

  if (!residuals || residuals.length === 0) return null;

  // Q-Q Plot points
  const qqPoints = residuals.map((r) => ({
    theoretical: r.theoreticalQuantile,
    sample: r.standardizedResidual,
  }));

  // Max bounds for Q-Q plot
  const maxQ = Math.max(
    3.5,
    Math.max(...qqPoints.map((p) => Math.abs(p.sample))),
    Math.max(...qqPoints.map((p) => Math.abs(p.theoretical)))
  );

  return (
    <div className="panel p-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm tracking-tight">
              Model Diagnostics & Residual Analysis
            </h3>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              · {regressionType}
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
            Evaluate normality, homoskedasticity, and time-varying parameter stability
          </p>
        </div>

        {/* Tab switch control */}
        <div className="flex items-center gap-1 p-1 bg-zinc-100 dark:bg-zinc-800/80 rounded-lg text-xs">
          <button
            onClick={() => setActiveTab('qq')}
            className={`px-3 py-1 font-medium rounded-md transition-colors ${
              activeTab === 'qq'
                ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            Normal Q-Q Plot
          </button>
          <button
            onClick={() => setActiveTab('fitted')}
            className={`px-3 py-1 font-medium rounded-md transition-colors ${
              activeTab === 'fitted'
                ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            Residual vs Fitted
          </button>
          <button
            onClick={() => setActiveTab('histogram')}
            className={`px-3 py-1 font-medium rounded-md transition-colors ${
              activeTab === 'histogram'
                ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            Residual Distribution
          </button>
          {rollingBetas && rollingBetas.length > 0 && (
            <button
              onClick={() => setActiveTab('rolling')}
              className={`px-3 py-1 font-medium rounded-md transition-colors ${
                activeTab === 'rolling'
                  ? 'bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-xs'
                  : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              Rolling Betas
            </button>
          )}
        </div>
      </div>

      {/* Viewport for Q-Q Plot */}
      {activeTab === 'qq' && (
        <div>
          <div className="w-full aspect-[2/1] max-h-[300px] min-h-[200px]">
            <svg viewBox="0 0 600 300" className="w-full h-full overflow-visible">
              {/* Axes */}
              <line x1="60" y1="150" x2="560" y2="150" stroke="#475569" strokeWidth="0.8" strokeDasharray="3 3" />
              <line x1="310" y1="20" x2="310" y2="270" stroke="#475569" strokeWidth="0.8" strokeDasharray="3 3" />

              {/* 45-degree reference line */}
              <line
                x1="80"
                y1="250"
                x2="540"
                y2="40"
                stroke="#DEC084"
                strokeWidth="1.8"
                strokeDasharray="4 2"
              />

              {/* Q-Q scatter points */}
              {qqPoints.map((pt, i) => {
                const cx = 310 + (pt.theoretical / maxQ) * 230;
                const cy = 150 - (pt.sample / maxQ) * 115;
                return (
                  <circle
                    key={i}
                    cx={cx}
                    cy={cy}
                    r="2.8"
                    fill="#38bdf8"
                    opacity="0.75"
                    className="hover:r-4 hover:fill-gold-300"
                  />
                );
              })}

              {/* Labels */}
              <text x="560" y="145" textAnchor="end" className="text-[10px] font-mono fill-zinc-500">
                Theoretical Quantiles N(0, 1) →
              </text>
              <text x="315" y="30" textAnchor="start" className="text-[10px] font-mono fill-zinc-500">
                Standardized Residuals (e_i / s) ↑
              </text>
            </svg>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
            Points conforming tightly to the diagonal line indicate normal residual distribution. Deviations in tails reflect market kurtosis/fat tails.
          </p>
        </div>
      )}

      {/* Viewport for Residual vs Fitted */}
      {activeTab === 'fitted' && (
        <div>
          <div className="w-full aspect-[2/1] max-h-[300px] min-h-[200px]">
            <svg viewBox="0 0 600 300" className="w-full h-full overflow-visible">
              <line x1="60" y1="150" x2="560" y2="150" stroke="#10b981" strokeWidth="1.2" strokeDasharray="3 3" />

              {residuals.map((r, i) => {
                const cx = 310 + r.fitted * 2500;
                const cy = 150 - r.residual * 2500;
                return (
                  <circle
                    key={i}
                    cx={Math.max(65, Math.min(555, cx))}
                    cy={Math.max(25, Math.min(275, cy))}
                    r="2.5"
                    fill="#a855f7"
                    opacity="0.65"
                  />
                );
              })}

              <text x="560" y="145" textAnchor="end" className="text-[10px] font-mono fill-zinc-500">
                Fitted Value (Ŷ_t) →
              </text>
              <text x="65" y="30" textAnchor="start" className="text-[10px] font-mono fill-zinc-500">
                Residual (e_t) ↑
              </text>
            </svg>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 text-center">
            Residuals randomly scattered around the zero line without a funnel shape signify homoskedasticity and absence of structural misspecification.
          </p>
        </div>
      )}

      {/* Viewport for Residual Histogram */}
      {activeTab === 'histogram' && (
        <div className="p-2">
          <div className="flex items-center justify-around text-xs mb-3 font-mono text-zinc-600 dark:text-zinc-400">
            <span>Durbin-Watson: <strong className="text-zinc-900 dark:text-zinc-100">{summary.durbinWatson.toFixed(3)}</strong> (2.0 = no serial corr)</span>
            <span>Regression SE: <strong className="text-zinc-900 dark:text-zinc-100">{(summary.standardErrorOfRegression * 100).toFixed(3)}%</strong></span>
          </div>
          <div className="w-full aspect-[2/0.9] max-h-[260px] flex items-center justify-center text-xs text-zinc-500">
            <svg viewBox="0 0 600 220" className="w-full h-full overflow-visible">
              {/* Binned bar distribution of residuals */}
              {(() => {
                const resVals = residuals.map((r) => r.residual);
                const minR = Math.min(...resVals);
                const maxR = Math.max(...resVals);
                const rng = maxR - minR || 0.01;
                const bCount = 28;
                const bWidth = rng / bCount;
                const counts = new Array(bCount).fill(0);
                resVals.forEach((val) => {
                  const idx = Math.min(bCount - 1, Math.max(0, Math.floor((val - minR) / bWidth)));
                  counts[idx]++;
                });
                const maxC = Math.max(...counts, 1);

                return counts.map((c, idx) => {
                  const x = 60 + (idx / bCount) * 480;
                  const h = (c / maxC) * 160;
                  const y = 190 - h;
                  return (
                    <rect
                      key={idx}
                      x={x}
                      y={y}
                      width={480 / bCount - 2}
                      height={h}
                      fill="#38bdf8"
                      opacity="0.8"
                    />
                  );
                });
              })()}
              <line x1="60" y1="190" x2="540" y2="190" stroke="#64748b" strokeWidth="1" />
            </svg>
          </div>
        </div>
      )}

      {/* Viewport for Rolling Betas */}
      {activeTab === 'rolling' && rollingBetas && (
        <div>
          <div className="w-full aspect-[2/1] max-h-[300px] min-h-[200px]">
            <svg viewBox="0 0 700 260" className="w-full h-full overflow-visible">
              {/* Grid */}
              <line x1="50" y1="130" x2="660" y2="130" stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />
              <text x="45" y="133" textAnchor="end" className="text-[9px] font-mono fill-zinc-500">β=0</text>

              {/* Render beta lines for each independent ticker */}
              {summary.independentTickers.map((ticker, tIdx) => {
                const color = ['#0ea5e9', '#DEC084', '#10b981', '#f43f5e'][tIdx % 4];
                const points = rollingBetas.map((rb, i) => {
                  const x = 60 + (i / Math.max(1, rollingBetas.length - 1)) * 590;
                  const bVal = rb.betas[ticker] ?? 1;
                  // Center at 1.0 or 0
                  const y = 130 - (bVal - 1.0) * 80;
                  return `${x},${Math.max(20, Math.min(240, y))}`;
                });

                return (
                  <path
                    key={ticker}
                    d={`M ${points.join(' L ')}`}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                );
              })}

              {/* First and last date */}
              <text x="60" y="250" className="text-[9px] font-mono fill-zinc-500">
                {rollingBetas[0]?.date}
              </text>
              <text x="650" y="250" textAnchor="end" className="text-[9px] font-mono fill-zinc-500">
                {rollingBetas[rollingBetas.length - 1]?.date}
              </text>
            </svg>
          </div>
          <div className="flex items-center justify-center gap-4 text-xs mt-2">
            {summary.independentTickers.map((ticker, tIdx) => (
              <span key={ticker} className="flex items-center gap-1.5">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block"
                  style={{ backgroundColor: ['#0ea5e9', '#DEC084', '#10b981', '#f43f5e'][tIdx % 4] }}
                />
                Beta ({ticker})
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
