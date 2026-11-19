'use client';

import React, { useMemo } from 'react';
import { ShieldCheck } from 'lucide-react';
import { walkForwardForecastScores } from '@/lib/forecast-trust';

/** Per-forecast trust card: one walk-forward scoreboard for every offered model. */
export function ForecastTrustCard({ returns, compact = false }: { returns: number[]; compact?: boolean }) {
  const trust = useMemo(() => walkForwardForecastScores(returns), [returns]);

  return (
    <div className="panel-inset p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-2">
        <p className="kicker flex items-center gap-2"><ShieldCheck className="h-3.5 w-3.5" /> Forecast Trust · Walk-Forward</p>
        <span className="font-mono text-[9px] text-zinc-400">
          {trust ? `train ${trust.nTrain} · test ${trust.nTest}` : 'need ≥200 returns'}
        </span>
      </div>
      {trust ? (
        <>
          <table className="w-full border-collapse font-mono text-[10px]">
            <thead>
              <tr className="text-left text-[9px] uppercase tracking-wider text-zinc-400">
                <th className="py-1 font-medium">Model</th>
                <th className="py-1 text-right font-medium">QLIKE ↓</th>
                {!compact && <th className="py-1 text-right font-medium">MSE log σ²</th>}
                <th className="py-1 text-right font-medium">n</th>
              </tr>
            </thead>
            <tbody>
              {trust.scores.map((s, i) => (
                <tr key={s.model} className="border-t border-[var(--hairline)]">
                  <td className={`py-1 ${i === 0 ? 'font-semibold text-gold-700 dark:text-gold-300' : 'text-zinc-600 dark:text-zinc-300'}`}>
                    {i === 0 ? '★ ' : ''}{s.model}
                  </td>
                  <td className="py-1 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{s.qlike === null ? '—' : s.qlike.toFixed(4)}</td>
                  {!compact && <td className="py-1 text-right tabular-nums text-zinc-500">{s.mseLog === null ? '—' : s.mseLog.toFixed(4)}</td>}
                  <td className="py-1 text-right tabular-nums text-zinc-400">{s.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 font-mono text-[9px] leading-relaxed text-zinc-400">
            One-step-ahead one-step recursion on the held-out tail; parameters fit on the train slice only, never refit. QLIKE is proper — lower is better. This is walk-forward, not the in-sample horse race in T4.
          </p>
        </>
      ) : (
        <p className="font-mono text-[10px] text-zinc-400">
          Not enough aligned daily returns for an honest holdout. Forecast cones remain in-sample fits until this card reports a score.
        </p>
      )}
    </div>
  );
}
