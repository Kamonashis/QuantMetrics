'use client';

import React from 'react';
import {
  StationarityTestRow,
  ArchLmLjungBoxResult,
  EvtTailResult,
} from '@/lib/analysis-engine';
import { ReturnStats } from '@/lib/quant-engine';
import { DistributionFittingPanel } from '@/components/charts/DistributionFittingPanel';
import {
  AdfEquation,
  ArchLmEquation,
  LjungBoxEquation,
  HillEstimatorEquation,
  GpdVarEquation,
} from '@/components/math/MathEquations';
import {
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Info,
  Scale,
  Flame,
  Binary,
} from 'lucide-react';

interface AnalysisStationarityTabProps {
  stationarityBattery: StationarityTestRow[];
  archLmLjungBox: ArchLmLjungBoxResult;
  evtTail: EvtTailResult;
  returns: number[];
  stats: ReturnStats;
  ticker: string;
}

export function AnalysisStationarityTab({
  stationarityBattery,
  archLmLjungBox,
  evtTail,
  returns,
  stats,
  ticker,
}: AnalysisStationarityTabProps) {
  return (
    <div className="space-y-6">
      {/* Stationarity Battery Verdict Table */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Scale className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Stationarity &amp; Unit-Root Econometric Battery (ADF · PP · KPSS)
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Testing Price Level P_t, Return Innovations r_t, and Variance Shocks r_t² across three asymptotic frameworks
            </p>
          </div>

          <AdfEquation className="hidden lg:inline" />
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 font-mono">
                <th className="py-2.5 px-3">Variable Tested</th>
                <th className="py-2.5 px-3">ADF Test (t-stat · p-val)</th>
                <th className="py-2.5 px-3">Phillips-Perron (Z_t · p-val)</th>
                <th className="py-2.5 px-3">KPSS (LM-stat · p-val)</th>
                <th className="py-2.5 px-3">Integration Order</th>
                <th className="py-2.5 px-3">Econometric Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
              {stationarityBattery.map((row) => (
                <tr key={row.variable} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30">
                  <td className="py-3 px-3 font-semibold text-slate-900 dark:text-slate-100">
                    <div>{row.seriesName}</div>
                    <div className="text-[11px] text-slate-400 font-normal">{row.variable}</div>
                  </td>

                  {/* ADF */}
                  <td className="py-3 px-3">
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      t = {row.adfStat.toFixed(2)}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      p = {row.adfPValue < 0.001 ? '< 0.001' : row.adfPValue.toFixed(3)} (
                      <span className={row.adfPValue < 0.05 ? 'text-emerald-500' : 'text-amber-500'}>
                        {row.adfVerdict}
                      </span>
                      )
                    </div>
                  </td>

                  {/* PP */}
                  <td className="py-3 px-3">
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      Z = {row.ppStat.toFixed(2)}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      p = {row.ppPValue < 0.001 ? '< 0.001' : row.ppPValue.toFixed(3)} (
                      <span className={row.ppPValue < 0.05 ? 'text-emerald-500' : 'text-amber-500'}>
                        {row.ppVerdict}
                      </span>
                      )
                    </div>
                  </td>

                  {/* KPSS */}
                  <td className="py-3 px-3">
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      LM = {row.kpssStat.toFixed(3)}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      p = {row.kpssPValue.toFixed(2)} (
                      <span className={row.kpssPValue >= 0.05 ? 'text-emerald-500' : 'text-amber-500'}>
                        {row.kpssPValue >= 0.05 ? 'H0 Not Rejected' : 'H0 Rejected'}
                      </span>
                      )
                    </div>
                  </td>

                  {/* Integration Order */}
                  <td className="py-3 px-3">
                    <span
                      className={`px-2 py-0.5 rounded font-mono font-semibold text-xs ${
                        row.integratedOrder === 'I(0)'
                          ? 'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300'
                          : row.integratedOrder === 'I(1)'
                          ? 'bg-amber-100 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300'
                          : 'bg-purple-100 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300'
                      }`}
                    >
                      {row.integratedOrder}
                    </span>
                  </td>

                  {/* Verdict */}
                  <td className="py-3 px-3 text-slate-700 dark:text-slate-300 font-sans max-w-xs">
                    {row.interpretation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span>
            ADF and Phillips-Perron test H0: Unit root (non-stationary); KPSS tests H0: Stationarity. Concurrence between
            ADF (p &lt; 0.05) and KPSS (p &gt; 0.05) confirms rigorous covariance stationarity without structural ambiguity.
          </span>
        </div>
      </div>

      {/* ARCH-LM & Ljung-Box Joint Clustering Battery */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Binary className="w-4 h-4 text-emerald-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Conditional Heteroskedasticity &amp; Autocorrelation Diagnostics (ARCH-LM · Ljung-Box)
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Falsifiable proof of volatility clustering across lags 5, 10, and 22 trading sessions
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <ArchLmEquation className="hidden md:inline" />
            <LjungBoxEquation className="hidden md:inline" />
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 font-mono">
                <th className="py-2 px-3">Lag Horizon (k)</th>
                <th className="py-2 px-3">Ljung-Box Q(r_t) [Returns]</th>
                <th className="py-2 px-3">Ljung-Box Q(r_t²) [Squared Returns]</th>
                <th className="py-2 px-3">Engle ARCH-LM [T·R²]</th>
                <th className="py-2 px-3">ARCH Effect Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
              {archLmLjungBox.lags.map((lag, idx) => {
                const lbR = archLmLjungBox.ljungBoxReturns[idx];
                const lbSq = archLmLjungBox.ljungBoxSquared[idx];
                const lm = archLmLjungBox.archLm[idx];

                return (
                  <tr key={lag} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30">
                    <td className="py-2.5 px-3 font-semibold text-slate-900 dark:text-slate-100">
                      k = {lag} ({lag === 5 ? '1 Week' : lag === 10 ? '2 Weeks' : '1 Month'})
                    </td>

                    {/* Ljung-Box on Returns */}
                    <td className="py-2.5 px-3">
                      <div className="font-semibold text-slate-700 dark:text-slate-300">
                        Q = {lbR.stat.toFixed(1)}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        p = {lbR.pValue.toFixed(3)}{' '}
                        <span className={lbR.isSignificant ? 'text-amber-500' : 'text-slate-400'}>
                          ({lbR.isSignificant ? 'Serial Corr' : 'White Noise'})
                        </span>
                      </div>
                    </td>

                    {/* Ljung-Box on Squared Returns */}
                    <td className="py-2.5 px-3">
                      <div className="font-semibold text-slate-700 dark:text-slate-300">
                        Q² = {lbSq.stat.toFixed(1)}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        p = {lbSq.pValue < 0.001 ? '< 0.001' : lbSq.pValue.toFixed(3)}{' '}
                        <span className={lbSq.isSignificant ? 'text-emerald-500 font-semibold' : 'text-slate-400'}>
                          ({lbSq.isSignificant ? 'Clustering Confirmed' : 'No Memory'})
                        </span>
                      </div>
                    </td>

                    {/* ARCH LM */}
                    <td className="py-2.5 px-3">
                      <div className="font-semibold text-emerald-600 dark:text-emerald-400">
                        LM = {lm.stat.toFixed(1)}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        p = {lm.pValue < 0.001 ? '< 0.001' : lm.pValue.toFixed(3)}
                      </div>
                    </td>

                    {/* ARCH Verdict */}
                    <td className="py-2.5 px-3">
                      {lm.isSignificant ? (
                        <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium font-sans">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Significant ARCH Effects (p &lt; 0.001)
                        </span>
                      ) : (
                        <span className="text-slate-500 font-sans">No significant ARCH effects</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 flex items-center justify-between text-xs">
          <div className="text-slate-700 dark:text-slate-300 font-medium">
            <strong>Modeling Justification:</strong> {archLmLjungBox.clusteringVerdict}
          </div>
          {archLmLjungBox.justifiesGarch && (
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 font-semibold">
              GARCH(1,1) Specification Validated
            </span>
          )}
        </div>
      </div>

      {/* Parametric Distribution Fitting Panel (Student-t, Hansen Skew-t, GED vs Normal) */}
      <DistributionFittingPanel returns={returns} stats={stats} ticker={ticker} />

      {/* Extreme Value Theory (EVT) Heavy-Tail Panel */}
      <div className="bg-white dark:bg-[#0c1222] border border-slate-200 dark:border-slate-800/90 rounded-xl p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-rose-500" />
              <h3 className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                Extreme Value Theory (EVT): Hill Tail Index &amp; Peaks-Over-Threshold (POT-GPD)
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Semi-parametric asymptotic modeling of catastrophic left-tail crash quantiles exceeding threshold u
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <HillEstimatorEquation className="hidden md:inline" />
            <GpdVarEquation className="hidden md:inline" />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
          {/* Hill Estimator */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-xs text-slate-500 dark:text-slate-400">Hill Tail Index (ξ̂)</div>
            <div className="mt-2 text-2xl font-bold font-mono text-slate-900 dark:text-slate-100">
              {evtTail.hillTailIndex.toFixed(3)}
            </div>
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              Pareto α = {evtTail.alphaPareto.toFixed(2)} (Heavy tail)
            </div>
          </div>

          {/* GPD Shape Parameter */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-xs text-slate-500 dark:text-slate-400">GPD Shape (ξ) &amp; Scale (β)</div>
            <div className="mt-2 text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
              ξ = {evtTail.gpdXi.toFixed(3)}
            </div>
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              β = {(evtTail.gpdBeta * 100).toFixed(2)}% | k = {evtTail.tailExtremeObservationsCount} exceedances
            </div>
          </div>

          {/* EVT-VaR 99% */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-xs text-slate-500 dark:text-slate-400">POT-GPD 99% 1-Day VaR</div>
            <div className="mt-2 text-2xl font-bold font-mono text-rose-600 dark:text-rose-400">
              {(evtTail.evtVaR99 * 100).toFixed(2)}%
            </div>
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              95% VaR: {(evtTail.evtVaR95 * 100).toFixed(2)}%
            </div>
          </div>

          {/* EVT Expected Shortfall */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
            <div className="text-xs text-slate-500 dark:text-slate-400">POT 99% Expected Shortfall (ES)</div>
            <div className="mt-2 text-2xl font-bold font-mono text-rose-700 dark:text-rose-300">
              {(evtTail.evtExpectedShortfall99 * 100).toFixed(2)}%
            </div>
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              Threshold u = {(evtTail.thresholdP95 * 100).toFixed(2)}%
            </div>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span>
            {evtTail.tailRiskVerdict} Pickands-Balkema-de Haan theorem guarantees that excesses over high threshold u
            converge to Generalized Pareto (GPD), capturing catastrophic black swan risk where Gaussian models underestimate.
          </span>
        </div>
      </div>
    </div>
  );
}
