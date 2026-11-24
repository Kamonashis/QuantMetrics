'use client';

import React, { useMemo, useState } from 'react';
import {
  ALLOCATIONS,
  allocationWeights,
  correlationStress,
  covarianceMatrix,
  hedgeTable,
  ledoitWolf,
  normalizeWeights,
  riskDecomposition,
  tailRisk,
  walkForwardAllocations,
  type CorrelationSample,
} from '@/lib/correlation-engine';
import { Callout, SectionHeader, SegmentedControl, StatTile } from '@/components/ui/lab-ui';
import { DataTable, fmtPct, td, tdClass, thClass } from './shared';

export function PortfolioTab({
  sample,
  weightsPct,
  onWeightsChange,
}: {
  sample: CorrelationSample;
  weightsPct: number[];
  onWeightsChange: (w: number[]) => void;
}) {
  const [covKind, setCovKind] = useState<'shrunk' | 'sample'>('shrunk');
  const [level, setLevel] = useState<0.95 | 0.99>(0.95);
  const [hedgeChoice, setHedgeTicker] = useState<string>('SPY');
  const hedgeTicker = sample.tickers.includes(hedgeChoice) ? hedgeChoice : sample.tickers[0];

  const shrunk = useMemo(() => ledoitWolf(sample.returns), [sample]);
  const sampleCov = useMemo(() => covarianceMatrix(sample.returns), [sample]);
  const cov = covKind === 'shrunk' ? shrunk.cov : sampleCov;

  const weights = useMemo(() => normalizeWeights(weightsPct), [weightsPct]);
  const risk = useMemo(() => riskDecomposition(weights, cov), [weights, cov]);
  const tail = useMemo(() => tailRisk(sample, weights, cov, level), [sample, weights, cov, level]);
  const stress = useMemo(() => correlationStress(weights, cov), [weights, cov]);
  const allocations = useMemo(
    () =>
      ALLOCATIONS.map((a) => {
        const w = allocationWeights(a.id, cov);
        return { ...a, weights: w, vol: riskDecomposition(w, cov).vol };
      }),
    [cov]
  );
  const hedgeIndex = Math.max(0, sample.tickers.indexOf(hedgeTicker));
  const hedges = useMemo(() => hedgeTable(sample, hedgeIndex), [sample, hedgeIndex]);
  const walk = useMemo(() => walkForwardAllocations(sample), [sample]);

  const totalInput = weightsPct.reduce((a, b) => a + (Number.isFinite(b) && b > 0 ? b : 0), 0);
  const top = risk.assets.map((a, i) => ({ ...a, ticker: sample.tickers[i] })).sort((a, b) => b.riskShare - a.riskShare)[0];
  const maxShare = Math.max(...risk.assets.map((a) => a.riskShare), 1e-9);
  const pctLabel = `${Math.round(level * 100)}%`;
  const walkBest = walk ? [...walk.rows].sort((a, b) => a.realizedVol - b.realizedVol)[0] : null;
  const walkEqual = walk?.rows.find((r) => r.id === 'equal') ?? null;

  const setWeight = (i: number, v: number) => {
    const next = [...weightsPct];
    next[i] = Number.isFinite(v) && v > 0 ? v : 0;
    onWeightsChange(next);
  };

  return (
    <div className="space-y-6">
      <section className="panel p-5">
        <div className="mb-4 flex flex-col gap-3 border-b border-[var(--hairline)] pb-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="kicker mb-1">Your mix</p>
            <p className="max-w-3xl font-display text-lg leading-snug text-zinc-900 dark:text-zinc-100">
              This mix has an annual volatility of <strong className="tabular-nums text-gold-600 dark:text-gold-300">{fmtPct(risk.vol)}</strong>.
              Without diversification it would be {fmtPct(risk.weightedAvgVol)}. The largest source of risk is <strong>{top.ticker}</strong>:{' '}
              {fmtPct(top.weight, 0)} of the capital and {fmtPct(top.riskShare, 0)} of the risk.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              ariaLabel="Covariance estimate"
              value={covKind}
              onChange={setCovKind}
              options={[
                { value: 'shrunk', label: 'Shrunk estimate' },
                { value: 'sample', label: 'Raw sample' },
              ]}
            />
            <SegmentedControl
              ariaLabel="Confidence level"
              value={level}
              onChange={setLevel}
              options={[
                { value: 0.95 as const, label: '95%' },
                { value: 0.99 as const, label: '99%' },
              ]}
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile label="Annual volatility" value={fmtPct(risk.vol)} tone="gold" sub={`${fmtPct(risk.weightedAvgVol)} if everything moved together`} />
          <StatTile label="Independent bets" value={risk.effectiveBets.toFixed(1)} sub={`Diversification ratio ${risk.diversificationRatio.toFixed(2)}`} />
          <StatTile label={`1-day loss, ${pctLabel} of days are better`} value={fmtPct(tail.historicalVaR, 2)} sub={`Historical · normal model ${fmtPct(tail.parametricVaR, 2)}`} />
          <StatTile label={`Average loss on the worst ${Math.round((1 - level) * 100)}% of days`} value={fmtPct(tail.expectedShortfall, 2)} sub={`${tail.tailDays} days in the sample`} />
        </div>
      </section>

      <section className="panel p-5">
        <SectionHeader
          kicker="Risk budget"
          title="Where the risk comes from"
          description="Capital weight and risk share differ when assets have different volatility or correlation. Loss columns add up to the portfolio totals."
        />
        <DataTable>
          <thead>
            <tr>
              <th className={thClass}>Asset</th>
              <th className={thClass}>Weight %</th>
              <th className={thClass}>Own volatility</th>
              <th className={`${thClass} min-w-40`}>Share of risk</th>
              <th className={thClass} title="Risk share divided by capital weight">Risk ÷ capital</th>
              <th className={thClass} title={`Contribution to the ${pctLabel} one-day loss under a normal model`}>Normal-model loss</th>
              <th className={thClass} title="Contribution to the average loss on the portfolio's worst days">Worst-days loss</th>
            </tr>
          </thead>
          <tbody>
            {risk.assets.map((a, i) => {
              const ratio = a.weight > 0 ? a.riskShare / a.weight : 0;
              return (
                <tr key={sample.tickers[i]}>
                  <td className={td('text-zinc-900 dark:text-zinc-100', 'font-semibold')}>{sample.tickers[i]}</td>
                  <td className={tdClass}>
                    <input
                      type="number"
                      min={0}
                      step={1}
                      aria-label={`${sample.tickers[i]} weight in percent`}
                      value={Number((weightsPct[i] ?? 0).toFixed(2))}
                      onChange={(e) => setWeight(i, parseFloat(e.target.value))}
                      className="w-20 rounded-md border border-zinc-200 bg-zinc-50 px-2 py-1 text-right font-mono text-xs font-bold text-zinc-900 focus:outline-none focus:ring-1 focus:ring-gold-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
                    />
                  </td>
                  <td className={tdClass}>{fmtPct(a.vol)}</td>
                  <td className={tdClass}>
                    <div className="flex items-center justify-end gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-[var(--bg-inset)]">
                        <div className="h-full rounded-full bg-gold-500/80" style={{ width: `${(Math.max(0, a.riskShare) / maxShare) * 100}%` }} />
                      </div>
                      <span className="w-12">{fmtPct(a.riskShare, 0)}</span>
                    </div>
                  </td>
                  <td className={td(ratio > 1.25 && 'text-amber-600 dark:text-amber-400')}>{a.weight > 0 ? `${ratio.toFixed(2)}×` : '—'}</td>
                  <td className={tdClass}>{fmtPct(tail.componentVaR[i], 2)}</td>
                  <td className={tdClass}>{fmtPct(tail.componentES[i], 2)}</td>
                </tr>
              );
            })}
            <tr className="font-semibold">
              <td className={td('text-zinc-900 dark:text-zinc-100')}>Portfolio</td>
              <td className={td(Math.abs(totalInput - 100) > 0.5 && 'text-amber-600 dark:text-amber-400')}>{totalInput.toFixed(1)}</td>
              <td className={tdClass}>{fmtPct(risk.vol)}</td>
              <td className={tdClass}>100%</td>
              <td className={tdClass}>1.00×</td>
              <td className={tdClass}>{fmtPct(tail.parametricVaR, 2)}</td>
              <td className={tdClass}>{fmtPct(tail.expectedShortfall, 2)}</td>
            </tr>
          </tbody>
        </DataTable>
        {Math.abs(totalInput - 100) > 0.5 && (
          <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-400">
            Weights add up to {totalInput.toFixed(1)}%. They are rescaled to 100% for every figure on this page.
          </p>
        )}

        <div className="mt-5 border-t border-[var(--hairline)] pt-4">
          <p className="mb-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">Try a standard mix</p>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {allocations.map((a) => (
              <button
                key={a.id}
                type="button"
                onClick={() => onWeightsChange(a.weights.map((w) => w * 100))}
                className="cursor-pointer rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] p-3 text-left transition-colors hover:border-gold-500/50"
              >
                <span className="flex items-baseline justify-between gap-2">
                  <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{a.label}</span>
                  <span className="font-mono text-xs tabular-nums text-gold-700 dark:text-gold-300">{fmtPct(a.vol)} vol</span>
                </span>
                <span className="mt-0.5 block text-[11px] text-zinc-500 dark:text-zinc-400">{a.description}</span>
                <span className="mt-1.5 block font-mono text-[10px] tabular-nums text-zinc-500 dark:text-zinc-400">
                  {a.weights.map((w, i) => `${sample.tickers[i]} ${(w * 100).toFixed(0)}`).join(' · ')}
                </span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            All long-only, computed from the {covKind === 'shrunk' ? `shrunk covariance (shrinkage ${(shrunk.delta * 100).toFixed(0)}%)` : 'raw sample covariance'}.
          </p>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="panel p-5">
          <SectionHeader
            kicker="What if"
            title="If correlations rise"
            description="Volatility of this mix if every pair became more correlated while each asset's own volatility stayed the same."
          />
          <DataTable>
            <thead>
              <tr>
                <th className={thClass}>Scenario</th>
                <th className={thClass}>Average correlation</th>
                <th className={thClass}>Annual volatility</th>
                <th className={thClass}>vs today</th>
              </tr>
            </thead>
            <tbody>
              {stress.map((r, i) => (
                <tr key={r.label}>
                  <td className={tdClass}>{r.label}</td>
                  <td className={tdClass}>{r.avgCorr.toFixed(2)}</td>
                  <td className={tdClass}>{fmtPct(r.vol)}</td>
                  <td className={td(i > 0 && 'text-[var(--negative)]')}>{i === 0 ? '—' : `+${((r.vol / stress[0].vol - 1) * 100).toFixed(0)}%`}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            In practice volatility rises too when correlations do, so these are lower bounds on a real stress.
          </p>
        </section>

        <section className="panel p-5">
          <SectionHeader
            kicker="Hedging"
            title={`How much does ${hedgeTicker} hedge each asset?`}
            description="The hedge ratio is the amount to short per $1 of the asset to remove the most variance."
            right={
              <select
                aria-label="Hedge instrument"
                value={hedgeTicker}
                onChange={(e) => setHedgeTicker(e.target.value)}
                className="cursor-pointer rounded-lg border border-zinc-200 bg-zinc-50 px-2 py-1 font-mono text-xs font-bold text-zinc-900 focus:outline-none focus:ring-1 focus:ring-gold-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              >
                {sample.tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            }
          />
          <DataTable>
            <thead>
              <tr>
                <th className={thClass}>Asset</th>
                <th className={thClass}>Hedge ratio</th>
                <th className={thClass}>Variance removed</th>
                <th className={thClass}>Volatility</th>
                <th className={thClass}>After hedge</th>
              </tr>
            </thead>
            <tbody>
              {hedges.map((h) => (
                <tr key={h.ticker}>
                  <td className={td('text-zinc-900 dark:text-zinc-100', 'font-semibold')}>{h.ticker}</td>
                  <td className={tdClass}>${h.hedgeRatio.toFixed(2)}</td>
                  <td className={tdClass}>{fmtPct(h.varianceReduction, 0)}</td>
                  <td className={tdClass}>{fmtPct(h.vol)}</td>
                  <td className={tdClass}>{fmtPct(h.hedgedVol)}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            Full-sample, in-sample figures. Hedge ratios drift; see Over time for how stable each pair is.
          </p>
        </section>
      </div>

      <section className="panel p-5">
        <SectionHeader
          kicker="Reality check"
          title="Did the standard mixes deliver?"
          description="Each mix was set using only the previous year of data, held for a month, then reset. Realised volatility is what actually happened next."
        />
        {walk && walkBest && walkEqual ? (
          <>
            <DataTable>
              <thead>
                <tr>
                  <th className={thClass}>Mix</th>
                  <th className={thClass}>Promised volatility</th>
                  <th className={thClass}>Realised volatility</th>
                  <th className={thClass}>Realised vs promised</th>
                  <th className={thClass}>vs equal weight</th>
                </tr>
              </thead>
              <tbody>
                {walk.rows.map((r) => (
                  <tr key={r.id}>
                    <td className={td('text-zinc-900 dark:text-zinc-100', 'font-semibold')}>{r.label}</td>
                    <td className={tdClass}>{fmtPct(r.expectedVol)}</td>
                    <td className={tdClass}>{fmtPct(r.realizedVol)}</td>
                    <td className={td(r.realizedVol > r.expectedVol * 1.1 && 'text-amber-600 dark:text-amber-400')}>
                      {r.realizedVol >= r.expectedVol ? '+' : ''}
                      {((r.realizedVol / r.expectedVol - 1) * 100).toFixed(0)}%
                    </td>
                    <td className={tdClass}>{r.id === 'equal' ? '—' : `${r.realizedVol >= walkEqual.realizedVol ? '+' : ''}${((r.realizedVol / walkEqual.realizedVol - 1) * 100).toFixed(0)}%`}</td>
                  </tr>
                ))}
              </tbody>
            </DataTable>
            <p className="mt-3 text-xs text-zinc-600 dark:text-zinc-300">
              Lowest realised volatility: <strong>{walkBest.label}</strong> at {fmtPct(walkBest.realizedVol)}, against {fmtPct(walkEqual.realizedVol)} for
              equal weight. {walk.rebalances} monthly resets since {walk.firstDate}. This measures risk only; it says nothing about returns.
            </p>
          </>
        ) : (
          <Callout>Needs about 15 months of history (a one-year lookback plus three monthly holding periods).</Callout>
        )}
      </section>
    </div>
  );
}
