'use client';

import React, { useEffect, useState } from 'react';
import { Magnet } from 'lucide-react';
import type { TrapState } from '@/components/views/volatility/types';

const STATE_STYLE: Record<TrapState, string> = {
  idle: 'border-zinc-300 bg-transparent text-zinc-500 dark:border-white/10 dark:text-zinc-500',
  building: 'border-gold-500/40 bg-gold-500/10 text-gold-800 dark:text-gold-300',
  armed: 'border-gold-500/60 bg-gold-500/15 text-gold-800 dark:text-gold-200',
  triggered: 'border-rose-500/50 bg-rose-500/10 text-rose-600 dark:text-rose-400',
  exhausted: 'border-zinc-400/40 text-zinc-500',
};
const STATE_LABEL: Record<TrapState, string> = {
  idle: 'IDLE',
  building: 'BUILDING',
  armed: 'ARMED',
  triggered: 'TRIG',
  exhausted: 'EXHAUST',
};

/** Trap-state chip for the Analysis header — score + state pill, hides on failure. */
export function TrapStateChip({ ticker }: { ticker: string }) {
  const [result, setResult] = useState<{ ticker: string; score: number; state: TrapState } | { ticker: string; error: true } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/options?mode=chain&symbol=${encodeURIComponent(ticker)}`)
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        if (d?.available && d.trap && typeof d.trap.score === 'number' && d.trap.state) {
          setResult({ ticker, score: d.trap.score, state: d.trap.state as TrapState });
        } else {
          setResult({ ticker, error: true });
        }
      })
      .catch(() => {
        if (!cancelled) setResult({ ticker, error: true });
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  const current = result && result.ticker === ticker ? result : null;
  if (!current) return null;
  if ('error' in current) return null;

  return (
    <span
      title={`Dealer trap score ${current.score}/100 — ${current.state.toUpperCase()}. Assumed positioning, not observed; see the Implied Vol tab, T7 panel.`}
      className={`hidden md:inline-flex cursor-help items-center gap-1.5 rounded-lg border px-2 py-1 font-mono text-[10px] tabular-nums ${STATE_STYLE[current.state]}`}
    >
      <Magnet className="h-3 w-3" />
      TRAP <strong>{current.score}</strong>
      <span className="text-[8px] font-bold tracking-wider opacity-80">{STATE_LABEL[current.state]}</span>
    </span>
  );
}
