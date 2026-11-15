'use client';

import React, { useEffect, useState } from 'react';
import { Magnet } from 'lucide-react';
import type { TrapState } from '@/components/views/volatility/types';

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

/**
 * Trap-score tile content for the Analysis KPI strip — score + state pill.
 * Hides the numeric slot on failure (never fabricates). The parent tile
 * supplies the label and chrome.
 */
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
  if (!current || 'error' in current) {
    return <span className="font-mono text-sm font-semibold text-zinc-500">—</span>;
  }

  return (
    <span
      title={`Dealer trap score ${current.score}/100 — ${current.state.toUpperCase()}. Assumed positioning, not observed; see the Implied Vol tab, T7 panel.`}
      className="inline-flex cursor-help items-center gap-1.5 font-mono text-sm font-semibold tabular-nums text-zinc-800 dark:text-zinc-100"
    >
      <Magnet className="h-3.5 w-3.5" aria-hidden="true" />
      {current.score}
      <span className={`text-[9px] font-bold tracking-wider ${STATE_STYLE[current.state]}`}>{STATE_LABEL[current.state]}</span>
    </span>
  );
}
