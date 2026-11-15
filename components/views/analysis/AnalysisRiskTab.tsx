'use client';

import React, { useState } from 'react';
import {
  VaRBacktestModelResult,
  DrawdownResult,
  DownsideBenchmarkResult,
} from '@/lib/analysis-engine';
import {
  KupiecPofEquation,
  ChristoffersenEquation,
  DrawdownEquation,
  PsrEquation,
} from '@/components/math/MathEquations';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { RegimeChip } from '@/components/ui/RegimeChip';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { regimeNames } from '@/components/charts/RegimeTimeline';
import { useChartSize } from '@/lib/use-chart-size';
import type { HmmFit } from '@/lib/hmm';
import {
  ShieldAlert,
  TrendingDown,
  Scale,
  Info,
} from 'lucide-react';

interface AnalysisRiskTabProps {
  varBacktestModels: VaRBacktestModelResult[];
  drawdownResult: DrawdownResult;
  downsideResult: DownsideBenchmarkResult;
  ticker: string;
  hmmFit?: HmmFit | null;
}

export function AnalysisRiskTab({
  varBacktestModels,
  drawdownResult,
  downsideResult,
  ticker,
  hmmFit,
}: AnalysisRiskTabProps) {
  const [ddHoverIdx, setDdHoverIdx] = useState<number | null>(null);
  const [betaHoverIdx, setBetaHoverIdx] = useState<number | null>(null);
  const { ref: ddRef, size: ddSize } = useChartSize<HTMLDivElement>();
  const { ref: betaRef, size: betaSize } = useChartSize<HTMLDivElement>();

  // Underwater curve pure SVG scaling
  const ddData = drawdownResult.underwaterCurve;
  const nDd = ddData.length;
  let maxDd = 0;
  for (const d of ddData) {
    if (Math.abs(d.drawdownPct) > maxDd) maxDd = Math.abs(d.drawdownPct);
  }
  maxDd = Math.max(5, maxDd * 1.1);

  const ddW = ddSize.width;
  const ddH = ddSize.height;
  const ddReady = ddW > 0;
  const dPadL = 46;
  const dPadR = 14;
  const dPadT = 14;
  const dPadB = 28;
  const dcW = Math.max(1, ddW - dPadL - dPadR);
  const dcH = Math.max(1, ddH - dPadT - dPadB);
  const ddXTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => Math.round(pct * Math.max(0, nDd - 1)));

  const getDdX = (i: number) => dPadL + (i / Math.max(1, nDd - 1)) * dcW;
  const getDdY = (ddPct: number) => dPadT + (Math.abs(ddPct) / maxDd) * dcH;

  const activeDdIdx = ddHoverIdx !== null ? ddHoverIdx : nDd - 1;
  const activeDdPoint = ddData[activeDdIdx] || { date: '', drawdownPct: 0 };

  // Rolling Beta pure SVG scaling
  const betaData = downsideResult.rollingBetaSeries;
  const nBeta = betaData.length;
  let minBeta = 0;
  let maxBeta = 2;
  if (nBeta > 0) {
    for (const b of betaData) {
      if (b.beta < minBeta) minBeta = b.beta;
      if (b.beta > maxBeta) maxBeta = b.beta;
    }
  }
  minBeta = Math.min(-0.2, minBeta - 0.1);
  maxBeta = Math.max(1.8, maxBeta + 0.1);
  const betaRange = maxBeta - minBeta || 1;

  const bW = betaSize.width;
  const bH = betaSize.height;
  const betaReady = bW > 0;
  const bPadL = 38;
  const bPadR = 12;
  const bPadT = 10;
  const bPadB = 24;
  const bcW = Math.max(1, bW - bPadL - bPadR);
  const bcH = Math.max(1, bH - bPadT - bPadB);
  const betaXTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => Math.round(pct * Math.max(0, nBeta - 1)));

  const getBetaX = (i: number) => bPadL + (i / Math.max(1, nBeta - 1)) * bcW;
  const getBetaY = (val: number) => bPadT + bcH - ((val - minBeta) / betaRange) * bcH;

  const activeBetaIdx = betaHoverIdx !== null ? betaHoverIdx : nBeta - 1;
  const activeBetaPoint = betaData[activeBetaIdx] || { date: '', beta: 1 };

  return (
    <div className="space-y-6">
      {/* Regime-Conditioned VaR Attribution — HMM posterior tags each violation day */}
      {hmmFit && (
        <div className="panel p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-1">
            <div>
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-gold-500" />
                <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                  Regime-Conditioned VaR Attribution (HMM)
                </h3>
                <RegimeChip fit={hmmFit} />
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Percentage of each model&apos;s 99% exceedances that landed in each HMM state. Stress-heavy clustering ⇒ regime mixture, not a parameter error — the exceedance clustering Christoffersen already flags.
              </p>
            </div>
          </div>

          <RegimeTransitionMatrix fit={hmmFit} labels={regimeNames(hmmFit.k)} className="mt-3 border-t border-zinc-100 dark:border-zinc-800/60 pt-3 lg:max-w-md" showStats={false} />

          <div className="mt-3 overflow-x-auto">
            <table className="table-sticky-first w-full text-xs text-left">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 font-mono">
                  <th className="py-2 px-2">Model</th>
                  <th className="py-2 px-2">99% Hits</th>
                  {regimeNames(hmmFit.k).map((name) => (
                    <th key={name} className="py-2 px-2 text-right">{name}</th>
                  ))}
                  <th className="py-2 px-2 text-right">Stress share</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60 font-mono">
                {varBacktestModels.map((m) => {
                  const hits = m.exceedanceIdx99.length;
                  const byState = new Array(hmmFit.k).fill(0);
                  for (const idx of m.exceedanceIdx99) {
                    const v = hmmFit.viterbi[idx];
                    if (v !== undefined && v >= 0) byState[v]++;
                  }
                  const stressShare = hits > 0 ? (byState[hmmFit.k - 1] / hits) * 100 : null;
                  return (
                    <tr key={m.key} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                      <td className="py-2 px-2 font-semibold text-zinc-800 dark:text-zinc-200">{m.modelName}</td>
                      <td className="py-2 px-2">{hits}</td>
                      {byState.map((c, si) => (
                        <td key={si} className="py-2 px-2 text-right tabular-nums">
                          {hits > 0 ? c : '—'}
                          {hits > 0 && <span className="text-zinc-400 pl-1">({((c / hits) * 100).toFixed(0)}%)</span>}
                        </td>
                      ))}
                      <td className="py-2 px-2 text-right">
                        <span className={stressShare !== null && stressShare > 50 ? 'text-rose-500 font-semibold' : stressShare !== null && stressShare > 33 ? 'text-gold-600 dark:text-gold-400' : 'text-zinc-500'}>
                          {stressShare !== null ? `${stressShare.toFixed(0)}%` : '—'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 5-Model VaR & ES Backtesting Comparison Table */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-rose-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                5-Model Value-at-Risk (VaR) &amp; Expected Shortfall (ES) Backtesting Suite
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Kupiec (1995) unconditional coverage &amp; Christoffersen (1998) exceedance clustering tests with Basel traffic lights
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <KupiecPofEquation className="hidden lg:inline" />
            <ChristoffersenEquation className="hidden lg:inline" />
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="table-sticky-first w-full text-xs text-left">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 font-mono">
                <th className="py-2.5 px-3">Model Specification</th>
                <th className="py-2.5 px-3">95% 1d VaR</th>
                <th className="py-2.5 px-3">99% 1d VaR</th>
                <th className="py-2.5 px-3">99% 1d ES (CVaR)</th>
                <th className="py-2.5 px-3">99% Violations (Act/Exp)</th>
                <th className="py-2.5 px-3">Kupiec POF Test</th>
                <th className="py-2.5 px-3">Christoffersen Independence</th>
                <th className="py-2.5 px-3">Basel Zone</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60 font-mono">
              {varBacktestModels.map((m) => (
                <tr key={m.key} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                  <td className="py-3 px-3 font-semibold text-zinc-900 dark:text-zinc-100">
                    {m.modelName}
                  </td>
                  <td className="py-3 px-3 text-zinc-700 dark:text-zinc-300">
                    {(m.var95Pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 font-semibold text-rose-600 dark:text-rose-400">
                    {(m.var99Pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-rose-700 dark:text-rose-300 font-semibold">
                    {(m.es99Pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3">
                    <span className="font-semibold">{m.exceedances99}</span> / {m.expectedExceedances99.toFixed(1)}
                  </td>
                  <td className="py-3 px-3">
                    <div className="text-zinc-800 dark:text-zinc-200">
                      LR = {m.kupiecLr99.toFixed(2)}
                    </div>
                    <div className="text-[11px] text-zinc-400">
                      p = {m.kupiecPVal99.toFixed(3)}{' '}
                      <span className={m.kupiecVerdict99 === 'Pass' ? 'text-gold-500 font-medium' : 'text-rose-500 font-semibold'}>
                        [{m.kupiecVerdict99}]
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-3">
                    <div className="text-zinc-800 dark:text-zinc-200">
                      LR = {m.christoffersenLr99.toFixed(2)}
                    </div>
                    <div className="text-[11px] text-zinc-400">
                      p = {m.christoffersenPVal99.toFixed(3)}{' '}
                      <span className={m.christoffersenVerdict99.startsWith('Pass') ? 'text-gold-500 font-medium' : 'text-gold-500 font-semibold'}>
                        [{m.christoffersenVerdict99.startsWith('Pass') ? 'Independent' : 'Clustered'}]
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded font-mono font-semibold text-xs ${
                        m.baselZone === 'Green'
                          ? 'bg-gold-100 dark:bg-gold-950/40 text-gold-700 dark:text-gold-300'
                          : m.baselZone === 'Yellow'
                          ? 'bg-gold-100 dark:bg-gold-950/40 text-gold-700 dark:text-gold-300'
                          : 'bg-rose-100 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          m.baselZone === 'Green'
                            ? 'bg-gold-500'
                            : m.baselZone === 'Yellow'
                            ? 'bg-gold-500'
                            : 'bg-rose-500'
                        }`}
                      />
                      {m.baselZone}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          <span>
            Under Basel Committee standards, 99% VaR with fewer than 5 exceptions over 250 days falls into the Green zone
            (no regulatory capital multiplier penalty). Notice that Parametric Normal underestimates tail exceedances
            relative to Cornish-Fisher and EVT-GPD.
          </span>
        </div>
      </div>

      {/* Underwater Drawdown Curve & Analytics */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-rose-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Underwater Drawdown Curve &amp; Peak-to-Trough Analytics
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Continuous percentage decline from running peak price
            </p>
          </div>

          <div className="font-mono text-xs text-zinc-400">
            {activeDdPoint.date}: <strong className="text-rose-500">{activeDdPoint.drawdownPct.toFixed(2)}%</strong>
          </div>
        </div>

        {/* Drawdown Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Current Drawdown</div>
            <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
              {drawdownResult.currentDrawdownPct.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Maximum Drawdown (MDD)</div>
            <div className="mt-1 text-lg font-bold font-mono text-rose-600 dark:text-rose-400">
              {drawdownResult.maxDrawdownPct.toFixed(2)}%
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Longest Drawdown</div>
            <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
              {drawdownResult.maxDrawdownDurationDays} days
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Calmar Ratio</div>
            <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
              {drawdownResult.calmarRatio.toFixed(2)}
            </div>
          </div>

          <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
            <div className="text-[11px] text-zinc-400">Ulcer Index</div>
            <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
              {drawdownResult.ulcerIndex.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Underwater Chart */}
        <div className="mt-4">
          <ChartFullscreen title="Underwater Drawdown Curve">
          <div ref={ddRef} className="chart-frame">
          {ddReady && (
          <svg
            width={ddW}
            height={ddH}
            className="block h-full w-full select-none"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const mouseX = e.clientX - rect.left;
              const ratio = Math.max(0, Math.min(1, (mouseX - dPadL) / dcW));
              const idx = Math.round(ratio * (nDd - 1));
              setDdHoverIdx(idx);
            }}
            onMouseLeave={() => setDdHoverIdx(null)}
          >
            {/* Gridlines */}
            {[0, 0.33, 0.66, 1].map((pct) => {
              const yVal = pct * maxDd;
              const yPos = getDdY(yVal);
              return (
                <g key={pct}>
                  <line x1={dPadL} y1={yPos} x2={ddW - dPadR} y2={yPos} stroke="#334155" strokeDasharray="2 2" strokeWidth={0.5} opacity={0.3} />
                  <text x={dPadL - 8} y={yPos + 3} textAnchor="end" fontSize="10" fill="#64748b" className="font-mono">
                    -{yVal.toFixed(1)}%
                  </text>
                </g>
              );
            })}

            {/* Zero top baseline */}
            <line x1={dPadL} y1={dPadT} x2={ddW - dPadR} y2={dPadT} stroke="#64748b" strokeWidth={0.65} />

            {/* Underwater Area */}
            {nDd > 1 && (
              <path
                d={`
                  M ${getDdX(0)} ${dPadT}
                  ${ddData.map((d, i) => `L ${getDdX(i)} ${getDdY(d.drawdownPct)}`).join(' ')}
                  L ${getDdX(nDd - 1)} ${dPadT}
                  Z
                `}
                fill="#f43f5e"
                fillOpacity={0.25}
              />
            )}

            {/* Underwater Line */}
            {nDd > 1 && (
              <path
                d={ddData.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getDdX(i)} ${getDdY(d.drawdownPct)}`).join(' ')}
                fill="none"
                stroke="#f43f5e"
                strokeWidth={1.2}
              />
            )}

            {/* Y-axis title */}
            <text
              x={10}
              y={dPadT + dcH / 2}
              textAnchor="middle"
              transform={`rotate(-90 10 ${dPadT + dcH / 2})`}
              className="text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 uppercase"
            >
              DD %
            </text>

            {/* X-axis baseline + date ticks */}
            <line
              x1={dPadL}
              y1={ddH - dPadB}
              x2={ddW - dPadR}
              y2={ddH - dPadB}
              stroke="currentColor"
              className="text-zinc-200 dark:text-zinc-800"
              strokeWidth={1}
            />
            {ddXTicks.map((idx, i) =>
              ddData[idx] ? (
                <text
                  key={`xtick-${i}`}
                  x={getDdX(idx)}
                  y={ddH - dPadB + 13}
                  textAnchor={i === 0 ? 'start' : i === ddXTicks.length - 1 ? 'end' : 'middle'}
                  className="text-[8px] sm:text-[9px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
                >
                  {ddData[idx].date}
                </text>
              ) : null
            )}

            {/* Hover marker */}
            {activeDdIdx >= 0 && activeDdIdx < nDd && (
              <g>
                <line x1={getDdX(activeDdIdx)} y1={dPadT} x2={getDdX(activeDdIdx)} y2={ddH - dPadB} stroke="#fda4af" strokeDasharray="2 2" strokeWidth={0.65} />
                <circle cx={getDdX(activeDdIdx)} cy={getDdY(activeDdPoint.drawdownPct)} r="4" fill="#f43f5e" />
              </g>
            )}
          </svg>
          )}
          </div>
          </ChartFullscreen>
        </div>
      </div>

      {/* Downside Risk & Asymmetric Beta to SPY Benchmark */}
      <div className="grid-cards">
        {/* Downside Risk Metrics */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Downside Risk &amp; Benchmark Asymmetry (vs. SPY)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Separating downside co-crash exposure from upside market participation
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              Beta: <strong className="text-gold-500">{activeBetaPoint.beta.toFixed(2)}</strong>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Semi-Deviation (Ann.)</div>
              <div className="mt-1 text-lg font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {(downsideResult.semiDeviationAnn * 100).toFixed(2)}%
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Sortino Ratio</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.sortinoRatio.toFixed(2)}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Downside Beta (β⁻)</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.downsideBeta.toFixed(2)}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">SPY Down-days</div>
            </div>

            <div className="p-3 rounded-lg border border-zinc-100 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30">
              <div className="text-[11px] text-zinc-400">Upside Beta (β⁺)</div>
              <div className="mt-1 text-lg font-bold font-mono text-gold-600 dark:text-gold-400">
                {downsideResult.upsideBeta.toFixed(2)}
              </div>
              <div className="text-[10px] text-zinc-400 mt-0.5">SPY Up-days</div>
            </div>
          </div>

          {/* Rolling Beta Chart */}
          <div className="mt-4">
            <ChartFullscreen title="60-Day Rolling Beta to SPY">
            <div ref={betaRef} className="chart-frame-sm">
            {betaReady && (
            <svg
              width={bW}
              height={bH}
              className="block h-full w-full select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, (mouseX - bPadL) / bcW));
                const idx = Math.round(ratio * (nBeta - 1));
                setBetaHoverIdx(idx);
              }}
              onMouseLeave={() => setBetaHoverIdx(null)}
            >
              {/* Beta = 1 reference line */}
              <line x1={bPadL} y1={getBetaY(1)} x2={bW - bPadR} y2={getBetaY(1)} stroke="#64748b" strokeDasharray="3 3" strokeWidth={0.65} opacity={0.5} />

              {/* Beta Path */}
              {nBeta > 1 && (
                <path
                  d={betaData.map((b, i) => `${i === 0 ? 'M' : 'L'} ${getBetaX(i)} ${getBetaY(b.beta)}`).join(' ')}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth={1.3}
                />
              )}

              {/* X-axis baseline + date ticks */}
              <line
                x1={bPadL}
                y1={bH - bPadB}
                x2={bW - bPadR}
                y2={bH - bPadB}
                stroke="currentColor"
                className="text-zinc-200 dark:text-zinc-800"
                strokeWidth={1}
              />
              {betaXTicks.map((idx, i) =>
                betaData[idx] ? (
                  <text
                    key={`xtick-${i}`}
                    x={getBetaX(idx)}
                    y={bH - bPadB + 13}
                    textAnchor={i === 0 ? 'start' : i === betaXTicks.length - 1 ? 'end' : 'middle'}
                    className="text-[8px] font-mono fill-zinc-500 dark:fill-zinc-400 tabular-nums"
                  >
                    {betaData[idx].date}
                  </text>
                ) : null
              )}

              {/* Hover dot */}
              {activeBetaIdx >= 0 && activeBetaIdx < nBeta && (
                <circle cx={getBetaX(activeBetaIdx)} cy={getBetaY(activeBetaPoint.beta)} r="4" fill="#10b981" />
              )}
            </svg>
            )}
            </div>
            </ChartFullscreen>
          </div>

        </div>

        {/* Probabilistic & Deflated Sharpe Ratio (Bailey & López de Prado) */}
        <div className="panel p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Probabilistic Sharpe Ratio (PSR) &amp; Deflated Sharpe (DSR)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Bailey &amp; López de Prado (2012): correcting point Sharpe for non-normality and selection bias
              </p>
            </div>
            <PsrEquation className="hidden md:inline" />
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Point Sharpe Ratio (Ann.)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-zinc-900 dark:text-zinc-100">
                {downsideResult.sharpeRatioAnn.toFixed(2)}
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                HAC SE: ±{downsideResult.sharpeStdErrorHac.toFixed(3)}
              </div>
            </div>

            <div className="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Probabilistic Sharpe (PSR)</div>
              <div className="mt-2 text-2xl font-bold font-mono text-gold-600 dark:text-gold-400">
                {(downsideResult.probabilisticSharpeRatio * 100).toFixed(1)}%
              </div>
              <div className="mt-2 text-[11px] font-mono text-zinc-400">
                Confidence that SR &gt; 0
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-zinc-500 dark:text-zinc-400">Deflated Sharpe Ratio (DSR)</div>
                <div className="text-lg font-bold font-mono text-gold-600 dark:text-gold-400 mt-1">
                  {(downsideResult.deflatedSharpeRatio * 100).toFixed(1)}%
                </div>
              </div>
              <div className="text-right text-[11px] text-zinc-400 max-w-xs">
                Penalizes for multiple testing / backtest overfitting across hyperparameter search trials.
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center gap-2">
            <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>
              Standard i.i.d. Sharpe standard errors assume zero skewness and kurtosis = 3. In the presence of negative
              skew and fat tails, PSR proves that the observed risk-adjusted return is statistically genuine rather than
              sampling luck.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
