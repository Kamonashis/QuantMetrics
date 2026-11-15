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
    // Bulged curve: dim stay-ish probabilities recede, high probabilities go
    // near-solid gold — better stay-vs-switch separation and text contrast.
    const a = Math.min(1, 0.08 + 0.92 * Math.pow(Math.max(0, Math.min(1, p)), 0.8));
    return `rgba(222,192,132,${a.toFixed(3)})`;
  };
  // Text must contrast with the *blended* cell (gold over the panel base),
  // which differs by theme: gold-over-white stays light (dark text always OK),
  // gold-over-near-black only becomes dark-text-friendly at high alpha.
  const textTone = (p: number) =>
    p >= 0.25 ? 'text-[#33270e] dark:text-[#241c08]' : 'text-zinc-700 dark:text-zinc-100';

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
      <table className="mt-1.5 w-full border-separate border-spacing-0.5 font-mono text-[10px]">
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
                  className={`px-2 py-1.5 text-center font-medium tabular-nums ${textTone(p)} ${
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
