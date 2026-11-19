'use client';

import React, { useMemo } from 'react';
import { Scale } from 'lucide-react';
import {
  buildOptionsDecision,
  type DecisionChain,
  type DecisionFnoPanel,
} from '@/lib/options-decision';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

/** The one options verdict, rendered identically in the Volatility and F&O suites. */
export function OptionsVerdictStrip({
  chain,
  fnoPanel,
  rv20,
}: {
  chain?: DecisionChain | null;
  fnoPanel?: DecisionFnoPanel | null;
  rv20?: number | null;
}) {
  const d = useMemo(() => buildOptionsDecision({ chain, fnoPanel, rv20 }), [chain, fnoPanel, rv20]);
  const positive = d.verdict === 'RICH';
  const negative = d.verdict === 'CHEAP';

  return (
    <div className="panel-inset flex flex-wrap items-center gap-x-4 gap-y-1 p-3 font-mono text-[10px]" title={d.method + '\n' + d.notes.join('\n')}>
      <span className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-zinc-400">
        <Scale className="h-3.5 w-3.5" /> One options read
      </span>
      <span className={`font-bold uppercase tracking-wider ${positive ? 'text-[var(--positive)]' : negative ? 'text-[var(--negative)]' : 'text-zinc-700 dark:text-zinc-200'}`}>
        {d.verdict}
      </span>
      <span className="text-zinc-500 dark:text-zinc-400">IV30 <span className="tabular-nums text-zinc-800 dark:text-zinc-100">{pct(d.iv30)}</span></span>
      <span className="text-zinc-500 dark:text-zinc-400">RV20 <span className="tabular-nums text-zinc-800 dark:text-zinc-100">{pct(d.rv20)}</span></span>
      <span className="text-zinc-500 dark:text-zinc-400">VRP <span className="tabular-nums text-zinc-800 dark:text-zinc-100">{d.vrpVariancePoints === null ? '—' : `${d.vrpVariancePoints >= 0 ? '+' : ''}${d.vrpVariancePoints.toFixed(0)} bp²`}</span></span>
      {d.curve && <span className="text-zinc-500 dark:text-zinc-400">Curve <span className="text-zinc-800 dark:text-zinc-100">{d.curve}</span></span>}
      {d.trap && <span className="text-zinc-500 dark:text-zinc-400">Trap <span className="text-zinc-800 dark:text-zinc-100">{d.trap.state.toUpperCase()} {d.trap.score}</span></span>}
      {d.positioning?.maxPainStrike != null && (
        <span className="text-zinc-500 dark:text-zinc-400">Max pain <span className="tabular-nums text-zinc-800 dark:text-zinc-100">{d.positioning.maxPainStrike.toFixed(0)}</span></span>
      )}
      {d.accruedSessions !== null && <span className="text-zinc-400">{d.accruedSessions} sessions</span>}
      <span className="text-zinc-400">IV²−RV² · 15–45d front · assumed GEX</span>
    </div>
  );
}
