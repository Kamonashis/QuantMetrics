'use client';

import React from 'react';
import { Layers } from 'lucide-react';
import { HMM_REGIME_TOKENS, getHmmRegimeToken, type HmmStateIdx } from '@/lib/hmm-regime-overlay';

export interface HmmRegimeOverlayToggleProps {
  enabled: boolean;
  onToggle: () => void;
  /** Optional concise note shown after legend (e.g. "origin t−1") */
  caption?: string;
  className?: string;
}

/**
 * Compact single-line toggle + inline Calm · Watchful · Crisis legend for
 * causal HMM volatility regime overlays across TCN-Attention charts.
 */
export function HmmRegimeOverlayToggle({
  enabled,
  onToggle,
  caption,
  className = '',
}: HmmRegimeOverlayToggleProps) {
  return (
    <div className={`flex flex-wrap items-center gap-x-3 gap-y-1.5 ${className}`}>
      <button
        type="button"
        onClick={onToggle}
        aria-pressed={enabled}
        className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md border text-[11px] font-mono whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
          enabled
            ? 'border-gold-500/45 bg-gold-500/10 text-gold-700 dark:text-gold-300'
            : 'border-[var(--hairline)] bg-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 hover:border-[var(--hairline-strong)]'
        }`}
      >
        <Layers className="w-3.5 h-3.5 shrink-0" />
        <span>HMM Overlay: {enabled ? 'On' : 'Off'}</span>
      </button>

      <div
        className={`flex items-center gap-2 text-[11px] font-mono transition-opacity duration-150 ${
          enabled ? 'opacity-100' : 'opacity-45'
        }`}
      >
        {HMM_REGIME_TOKENS.map((tok, i) => (
          <React.Fragment key={tok.state}>
            {i > 0 && <span className="text-zinc-400 dark:text-zinc-600 select-none">·</span>}
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: tok.color }}
                aria-hidden="true"
              />
              <span className={tok.textClass}>{tok.label}</span>
            </span>
          </React.Fragment>
        ))}
        {caption && (
          <>
            <span className="text-zinc-400 dark:text-zinc-600 select-none">·</span>
            <span className="text-zinc-500 dark:text-zinc-400 whitespace-nowrap">{caption}</span>
          </>
        )}
      </div>
    </div>
  );
}

export interface HmmRegimeScorecardColumn {
  key: string;
  label: string;
  align?: 'left' | 'right';
}

export interface HmmRegimeScorecardRow {
  state: HmmStateIdx;
  count: number;
  cells: Record<string, React.ReactNode>;
}

export interface HmmRegimeScorecardProps {
  title?: string;
  subtitle?: string;
  columns: HmmRegimeScorecardColumn[];
  rows: HmmRegimeScorecardRow[];
  className?: string;
}

/**
 * Compact 3-row Calm / Watchful / Crisis OOS accuracy breakdown scorecard.
 * Always displays real sample count n and renders "—" for zero-occupancy regimes.
 */
export function HmmRegimeScorecard({
  title = 'OOS Performance by Causal HMM Regime',
  subtitle = 'Stratified by dominant forward-filtered state at prediction origin (t−1)',
  columns,
  rows,
  className = '',
}: HmmRegimeScorecardProps) {
  const totalN = rows.reduce((acc, r) => acc + r.count, 0);

  return (
    <div className={`panel-inset p-3 ${className}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
        <div className="flex items-baseline gap-2">
          <span className="text-[10px] font-mono uppercase tracking-[0.14em] text-gold-700 dark:text-gold-300">
            {title}
          </span>
          <span className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 tabular-nums">
            n = {totalN} OOS
          </span>
        </div>
        {subtitle && (
          <span className="text-[11px] text-zinc-500 dark:text-zinc-400">{subtitle}</span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[var(--hairline)] text-[10px] font-mono uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              <th className="py-1.5 pr-3 font-medium">Regime</th>
              <th className="py-1.5 px-2.5 text-right font-medium">n</th>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`py-1.5 px-2.5 font-medium whitespace-nowrap ${
                    col.align === 'left' ? 'text-left' : 'text-right'
                  }`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--hairline)] text-xs font-mono tabular-nums">
            {rows.map((row) => {
              const tok = getHmmRegimeToken(row.state);
              const empty = row.count === 0;
              return (
                <tr
                  key={row.state}
                  className={empty ? 'opacity-50' : 'hover:bg-zinc-500/[0.04] transition-colors'}
                >
                  <td className="py-1.5 pr-3 whitespace-nowrap">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: tok.color }}
                        aria-hidden="true"
                      />
                      <span className={`font-medium ${tok.textClass}`}>{tok.label}</span>
                    </span>
                  </td>
                  <td className="py-1.5 px-2.5 text-right text-zinc-600 dark:text-zinc-300">
                    {row.count}
                  </td>
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={`py-1.5 px-2.5 whitespace-nowrap ${
                        col.align === 'left' ? 'text-left' : 'text-right'
                      } text-zinc-800 dark:text-zinc-200`}
                    >
                      {empty ? '—' : (row.cells[col.key] ?? '—')}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
