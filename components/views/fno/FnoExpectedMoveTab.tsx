'use client';

import React, { useMemo, useState } from 'react';
import { ArrowLeftRight, Info, RefreshCw, Sparkles } from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import { historicalAbsMoves, percentileRankLocal, moveVerdict, moveDistributionSummary } from '@/lib/fno/expected-move';
import { interpolateTcnExpectedMove, classifyTcnMoveDislocation } from '@/lib/tcn-multi-domain';
import { useTcnVolatility } from '@/hooks/useTcnVolatility';
import { MoveDistributionChart } from '@/components/charts/fno/MoveDistributionChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { AnimatedTabs } from '@/components/ui/animated-tabs';
import type { PanelPayload } from './types';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${v.toFixed(d)}%`;

export function FnoExpectedMoveTab({
  panel,
  bars,
  symbol,
}: {
  panel: PanelPayload | null;
  bars: PriceBar[];
  symbol: string;
}) {
  const moves = panel?.expectedMoves ?? [];
  const closest = moves[0] ?? null;
  const [horizonIdx, setHorizonIdx] = useState<number>(0);
  const selected = moves[horizonIdx] ?? closest;

  const { volForecast: tcnVol, isTraining: tcnTraining, retrain: retrainTcn } = useTcnVolatility(symbol, bars);

  const effectiveHorizon = selected?.dte ?? 21;
  const realizedSamples = useMemo(() => historicalAbsMoves(bars, effectiveHorizon), [bars, effectiveHorizon]);
  const summary = useMemo(() => moveDistributionSummary(realizedSamples), [realizedSamples]);
  const implied = selected?.straddleMove ?? selected?.sigmaMove ?? null;
  const isStraddleBasis = selected?.straddleMove !== null && selected?.straddleMove !== undefined;
  const verdict = moveVerdict(implied, realizedSamples);
  const pctRank = implied !== null ? percentileRankLocal(realizedSamples, implied / 100) : null;

  const tcnEstimate = useMemo(
    () => (tcnVol ? interpolateTcnExpectedMove(tcnVol, effectiveHorizon) : null),
    [tcnVol, effectiveHorizon]
  );
  const tcnComparablePct = tcnEstimate
    ? isStraddleBasis
      ? tcnEstimate.straddleEqMovePct
      : tcnEstimate.sigmaMovePct
    : null;
  const tcnDislocation = useMemo(
    () => classifyTcnMoveDislocation(implied, tcnComparablePct),
    [implied, tcnComparablePct]
  );

  const verdictColor =
    verdict === 'CHEAP' ? 'text-emerald-600 dark:text-emerald-400'
    : verdict === 'RICH' ? 'text-rose-500 dark:text-rose-400'
    : verdict === 'FAIR' ? 'text-zinc-500'
    : 'text-zinc-400';

  return (
    <div className="space-y-4">
      {/* Verdict hero */}
      <div className="panel p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-2xl">
            <p className="kicker mb-1.5 flex items-center gap-2"><ArrowLeftRight className="h-3.5 w-3.5" /> Expected vs Realized Move</p>
            <h3 className="font-display text-lg font-semibold leading-snug text-zinc-900 dark:text-zinc-50">
              The market pays{' '}
              <span className="text-gold-700 dark:text-gold-300">
                ±{implied !== null ? implied.toFixed(1) : '—'}%
              </span>{' '}
              for a {effectiveHorizon}-session move; history says{' '}
              <span className="text-zinc-700 dark:text-zinc-300">
                {summary ? `median ${summary.median.toFixed(1)}%` : '—'}
              </span>.
            </h3>
            <p className="mt-1 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
              Implied = 0.8×ATM straddle (or ATM IV × √T). Realized = |log return| over the same horizon, sampled from {bars.length.toLocaleString()} sessions of {symbol} price history.
              {tcnEstimate && (
                <>
                  {' '}TCN-Attention projects <strong className="font-mono text-zinc-800 dark:text-zinc-200">±{tcnComparablePct?.toFixed(1)}%</strong> (99% bound ±{tcnEstimate.move99Pct.toFixed(1)}%{tcnEstimate.extrapolated ? ', flat ≥22d' : ''}).
                </>
              )}
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={retrainTcn}
              disabled={tcnTraining || bars.length < 70}
              title={tcnTraining ? 'TCN volatility head is training' : 'Fit the TCN volatility head on demand'}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gold-500/40 bg-gold-500/10 px-2.5 py-1.5 font-mono text-[10px] text-gold-700 transition-colors hover:bg-gold-500/20 disabled:cursor-not-allowed disabled:opacity-50 dark:text-gold-300"
            >
              <RefreshCw className={`h-3 w-3 ${tcnTraining ? 'animate-spin' : ''}`} />
              {tcnTraining ? 'Training…' : tcnVol ? 'Retrain TCN' : 'Train TCN'}
            </button>
            {pctRank !== null && (
              <div className="text-right">
                <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">vs history</div>
                <div className="font-mono text-lg font-bold text-zinc-800 dark:text-zinc-100 tabular-nums">
                  {pctRank.toFixed(0)}th <span className="text-[10px] text-zinc-400">pctile</span>
                </div>
              </div>
            )}
            <div className="rounded-xl border border-gold-500/30 bg-[var(--bg-inset)] px-3.5 py-2.5 text-center">
              <div className={`font-mono text-sm font-bold tabular-nums ${
                tcnDislocation.verdict === 'RICH vs TCN'
                  ? 'text-rose-500 dark:text-rose-400'
                  : tcnDislocation.verdict === 'CHEAP vs TCN'
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-gold-700 dark:text-gold-300'
              }`}>
                {tcnDislocation.verdict ?? (tcnTraining ? 'TCN…' : '—')}
              </div>
              <div className="font-mono text-[9px] tabular-nums text-zinc-500 dark:text-zinc-400">
                {tcnComparablePct !== null && tcnDislocation.diffPct !== null
                  ? `TCN ±${tcnComparablePct.toFixed(1)}% (${tcnDislocation.diffPct >= 0 ? '+' : ''}${tcnDislocation.diffPct.toFixed(1)}pt)`
                  : 'TCN 1d/5d/22d'}
              </div>
            </div>
            <div className={`rounded-xl border px-4 py-2.5 text-center ${
              verdict === 'CHEAP' ? 'border-emerald-500/40 bg-emerald-500/10'
              : verdict === 'RICH' ? 'border-rose-500/40 bg-rose-500/10'
              : 'border-zinc-300 dark:border-white/[0.08] bg-[var(--bg-inset)]'
            }`}>
              <div className={`font-mono text-2xl font-bold ${verdictColor}`}>{verdict ?? '—'}</div>
              <div className="font-mono text-[8.5px] uppercase tracking-[0.18em] text-zinc-400">straddle verdict</div>
            </div>
          </div>
        </div>
      </div>

      {/* Horizon selector + distribution */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <AnimatedTabs
            size="sm"
            value={String(horizonIdx)}
            onChange={(id) => setHorizonIdx(Number(id))}
            tabs={moves.map((m, i) => ({ id: String(i), label: `${m.dte}d` }))}
          />
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            {moves[horizonIdx]?.expiry ?? '—'} · histogram of {realizedSamples.length.toLocaleString()} historical windows
          </span>
        </div>
        <ChartFullscreen title="Historical |Move| Distribution vs Implied & TCN" subtitle={`${effectiveHorizon}-session windows`}>
          <MoveDistributionChart
            samples={realizedSamples}
            impliedPct={implied}
            tcnMovePct={tcnComparablePct}
            horizonLabel={`${effectiveHorizon}-session |move| distribution`}
          />
        </ChartFullscreen>
      </div>

      {/* Per-expiry table */}
      <div className="panel overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] px-5 py-4">
          <p className="kicker flex items-center gap-2"><ArrowLeftRight className="h-3.5 w-3.5" /> Implied Move by Expiry vs TCN-Attention</p>
          {tcnVol && (
            <span className="inline-flex items-center gap-1 font-mono text-[9.5px] text-zinc-400">
              <Sparkles className="h-3 w-3 text-gold-500" />
              TCN RV 1d {(tcnVol.horizons[1].annualizedVol * 100).toFixed(1)}% · 5d {(tcnVol.horizons[5].annualizedVol * 100).toFixed(1)}% · 22d {(tcnVol.horizons[22].annualizedVol * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[780px] text-xs">
            <thead>
              <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                <th className="px-5 py-2">Expiry</th>
                <th className="px-3 py-2">DTE</th>
                <th className="px-3 py-2 text-right">ATM IV</th>
                <th className="px-3 py-2 text-right">σ-move</th>
                <th className="px-3 py-2 text-right">Straddle</th>
                <th className="px-3 py-2 text-right">Prem %</th>
                <th className="px-3 py-2 text-right">TCN Move</th>
                <th className="px-3 py-2 text-right">Implied vs TCN</th>
                <th className="px-5 py-2 text-right">Implied vs realized</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {moves.map((m) => {
                const samples = historicalAbsMoves(bars, m.dte);
                const imp = m.straddleMove ?? m.sigmaMove;
                const v = moveVerdict(imp, samples);
                const pctRankRow = imp !== null ? percentileRankLocal(samples, imp / 100) : null;
                const tcnRow = tcnVol ? interpolateTcnExpectedMove(tcnVol, m.dte) : null;
                const tcnPctRow = tcnRow ? (m.straddleMove !== null ? tcnRow.straddleEqMovePct : tcnRow.sigmaMovePct) : null;
                const disl = classifyTcnMoveDislocation(imp, tcnPctRow);
                return (
                  <tr key={m.expiry} className="border-t border-[var(--hairline)]">
                    <td className="px-5 py-2 text-zinc-700 dark:text-zinc-200">{m.expiry}</td>
                    <td className="px-3 py-2 tabular-nums text-zinc-500">{m.dte}d</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{m.atmIv !== null ? `${(m.atmIv * 100).toFixed(1)}%` : '—'}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{pct(m.sigmaMove)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{m.straddleMove !== null && m.atmCall !== null && m.atmPut !== null ? (m.atmCall + m.atmPut).toFixed(2) : '—'}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-gold-700 dark:text-gold-300">{pct(m.straddleMove)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-zinc-800 dark:text-zinc-100">
                      {tcnPctRow !== null ? `±${tcnPctRow.toFixed(1)}%` : '—'}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {disl.verdict && disl.diffPct !== null ? (
                        <span className={
                          disl.verdict === 'RICH vs TCN'
                            ? 'font-semibold text-rose-500 dark:text-rose-400'
                            : disl.verdict === 'CHEAP vs TCN'
                              ? 'font-semibold text-emerald-600 dark:text-emerald-400'
                              : 'text-zinc-500'
                        }>
                          {disl.diffPct >= 0 ? '+' : ''}{disl.diffPct.toFixed(1)}pt · {disl.verdict.replace(' vs TCN', '')}
                        </span>
                      ) : '—'}
                    </td>
                    <td className="px-5 py-2 text-right">
                      {v === null ? '—' : (
                        <span className={`rounded px-2 py-0.5 text-[9px] font-bold tracking-wider ${
                          v === 'CHEAP' ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                          : v === 'RICH' ? 'bg-rose-500/15 text-rose-500 dark:text-rose-400'
                          : 'bg-zinc-200/50 text-zinc-500 dark:bg-white/[0.06] dark:text-zinc-400'
                        }`}>
                          {v} {pctRankRow !== null ? `${pctRankRow.toFixed(0)}th` : ''}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {moves.length === 0 && (
                <tr><td colSpan={9} className="px-5 py-4 text-zinc-400">No chain data available.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex items-start gap-2 border-t border-[var(--hairline)] px-5 py-3 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
          <span>Historical IV is not fetchable — this compares today&apos;s price to realized distribution, not to backtested straddle P&amp;L. &quot;RICH&quot; means the market pays more than realized base rates; &quot;CHEAP&quot; means the opposite. Event-week expiries (earnings, FOMC) read RICH by design — don&apos;t fade them blindly.</span>
        </div>
      </div>
    </div>
  );
}
