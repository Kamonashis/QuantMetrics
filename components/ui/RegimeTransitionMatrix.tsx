'use client';

import React from 'react';
import { regimeNames } from '@/components/charts/RegimeTimeline';

export interface RegimeMatrixShape {
  k: number;
  transition: number[][];
  occupancy: number[];
  expectedDurations: number[];
}

interface RegimeTransitionMatrixProps {
  fit: RegimeMatrixShape | null;
  /** Override state labels (defaults to regimeNames(k): Calm/Normal/Crisis). */
  labels?: string[];
  className?: string;
  /** Show occupancy + expected duration caption below the grid */
  showStats?: boolean;
}

const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;

/**
 * Compact K×K HMM transition matrix, rows = from-state → cols = to-state.
 * Cells shaded gold with opacity proportional to probability.
 * Renders an honest "did not converge" line when fit is null.
 */
export function RegimeTransitionMatrix({ fit, labels, className, showStats = true }: RegimeTransitionMatrixProps) {
  if (!fit) {
    return (
      <div className={`rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-center font-mono text-[10px] text-zinc-500 ${className ?? ''}`}>
        HMM did not converge — transition matrix suppressed rather than fabricated.
      </div>
    );
  }
  const K = fit.k;
  const names = labels ?? regimeNames(K);
  const cellBg = (p: number) => {
    const a = 0.08 + 0.62 * Math.max(0, Math.min(1, p));
    return `rgba(222,192,132,${a.toFixed(3)})`;
  };
  const textTone = (p: number) =>
    p >= 0.5 ? 'text-[#3b2f10]' : p >= 0.2 ? 'text-[#5a4a1c]' : 'text-zinc-500 dark:text-zinc-400';

  return (
    <div className={`${className ?? ''}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
          Transition matrix · rows from → cols to
        </span>
        <span className="font-mono text-[8px] uppercase tracking-[0.14em] text-zinc-500">
          stay vs switch
        </span>
      </div>
      <table className="mt-1.5 w-full border-separate border-spacing-0.5 font-mono text-[9px]">
        <thead>
          <tr>
            <th aria-label="from \ to" className="px-1.5 py-1 text-left font-normal text-zinc-500">
              ↓ from \ to →
            </th>
            {names.map((n) => (
              <th key={n} className="px-1.5 py-1 text-center font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
                {n}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fit.transition.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="px-1.5 py-1 text-left font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
                {names[i]}
              </th>
              {row.map((p, j) => (
                <td
                  key={j}
                  className={`px-1.5 py-1 text-center tabular-nums ${textTone(p)} ${
                    i === j ? 'font-bold' : ''
                  }`}
                  style={{ backgroundColor: cellBg(p) }}
                  title={`${names[i]} → ${names[j]}: ${(p * 100).toFixed(1)}%`}
                >
                  {pct1(p)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {showStats && (
        <p className="mt-1 font-mono text-[8px] leading-snug text-zinc-500">
          occupancy {fit.occupancy.map((o, i) => `${names[i]} ${(o * 100).toFixed(0)}%`).join(' · ')}
          {'  '}·{'  '}E[τ] {fit.expectedDurations.map((d, i) => `${names[i]} ${d.toFixed(1)}d`).join(' · ')}
        </p>
      )}
    </div>
  );
}
