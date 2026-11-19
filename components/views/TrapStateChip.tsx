'use client';

import React from 'react';
import { Magnet } from 'lucide-react';
import type { TrapState } from '@/components/views/volatility/types';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';

const STATE_STYLE: Record<TrapState, string> = {
  idle: 'text-zinc-500 dark:text-zinc-400',
  building: 'text-gold-700 dark:text-gold-300',
  armed: 'text-gold-800 dark:text-gold-200',
  triggered: 'text-rose-600 dark:text-rose-400',
  exhausted: 'text-zinc-500 dark:text-zinc-500',
};
const STATE_LABEL: Record<TrapState, string> = {
  idle: 'IDLE',
  building: 'BUILDING',
  armed: 'ARMED',
  triggered: 'TRIG',
  exhausted: 'EXHAUST',
};

/** Trap-score tile content — reads the shared snapshot, never fabricates a score. */
export function TrapStateChip() {
  const { chain, chainLoading } = useTickerSnapshot();
  const trap = chain?.trap;
  const available = chain?.available === true && trap && typeof trap.score === 'number' && trap.state;

  if (!available) {
    return <span className="font-mono text-sm font-semibold text-zinc-500">{chainLoading ? '···' : '—'}</span>;
  }

  return (
    <span
      title={`Dealer trap score ${trap!.score}/100 — ${trap!.state.toUpperCase()}. Assumed positioning, not observed; see T7.`}
      className="inline-flex cursor-help items-center gap-1.5 font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100"
    >
      <Magnet className="h-3.5 w-3.5" aria-hidden="true" />
      {trap!.score}
      <span className={`text-[9px] font-bold tracking-wider ${STATE_STYLE[trap!.state as TrapState]}`}>
        {STATE_LABEL[trap!.state as TrapState]}
      </span>
    </span>
  );
}
