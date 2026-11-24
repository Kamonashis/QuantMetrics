'use client';

import React from 'react';
import type { Evidence } from '@/lib/factor-regression';
import { EVIDENCE_LABEL } from '@/lib/factor-regression';

const EVIDENCE_STYLE: Record<Evidence, string> = {
  strong: 'border-[var(--positive)]/40 bg-[var(--positive)]/10 text-[var(--positive)]',
  moderate: 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300',
  weak: 'border-amber-500/30 bg-amber-500/5 text-amber-700 dark:text-amber-400',
  none: 'border-zinc-300 bg-zinc-100 text-zinc-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400',
};

export function EvidenceChip({ evidence, p }: { evidence: Evidence; p?: number }) {
  return (
    <span
      title={p !== undefined ? `p = ${p < 0.0001 ? '< 0.0001' : p.toFixed(4)} (Newey–West)` : undefined}
      className={`inline-flex shrink-0 items-center rounded-md border px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-wider ${EVIDENCE_STYLE[evidence]}`}
    >
      {EVIDENCE_LABEL[evidence]}
    </span>
  );
}

export function SectionHeader({
  kicker,
  title,
  description,
  right,
}: {
  kicker: string;
  title: string;
  description?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-col gap-3 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <p className="kicker mb-1">{kicker}</p>
        <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
        {description && <p className="mt-0.5 max-w-3xl text-xs text-zinc-500 dark:text-zinc-400">{description}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

export function StatTile({
  label,
  value,
  sub,
  tone = 'default',
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: 'default' | 'gold' | 'positive' | 'negative';
}) {
  const toneClass =
    tone === 'gold'
      ? 'text-gold-600 dark:text-gold-300'
      : tone === 'positive'
        ? 'text-[var(--positive)]'
        : tone === 'negative'
          ? 'text-[var(--negative)]'
          : 'text-zinc-900 dark:text-zinc-100';
  return (
    <div className="panel-inset p-3.5">
      <span className="block text-[11px] text-zinc-500 dark:text-zinc-400">{label}</span>
      <span className={`mt-0.5 block font-mono text-xl font-bold tabular-nums ${toneClass}`}>{value}</span>
      {sub && <span className="mt-0.5 block text-[11px] text-zinc-500 dark:text-zinc-400">{sub}</span>}
    </div>
  );
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'danger';
  title?: string;
  children: React.ReactNode;
}) {
  const cls =
    tone === 'danger'
      ? 'border-[var(--negative)]/35 bg-[var(--negative)]/5 text-zinc-700 dark:text-zinc-300'
      : tone === 'warn'
        ? 'border-amber-500/35 bg-amber-500/5 text-zinc-700 dark:text-zinc-300'
        : 'border-gold-500/25 bg-gold-500/5 text-zinc-700 dark:text-zinc-300';
  return (
    <div className={`rounded-xl border px-4 py-3 text-xs leading-relaxed ${cls}`}>
      {title && <p className="mb-0.5 font-semibold text-zinc-900 dark:text-zinc-100">{title}</p>}
      {children}
    </div>
  );
}

/** Coefficient + 95% interval on a shared axis, with a zero reference line. */
export function IntervalBar({
  estimate,
  low,
  high,
  domain,
}: {
  estimate: number;
  low: number;
  high: number;
  domain: [number, number];
}) {
  const [d0, d1] = domain;
  const span = d1 - d0 || 1;
  const pos = (v: number) => `${(((Math.min(d1, Math.max(d0, v)) - d0) / span) * 100).toFixed(2)}%`;
  const crossesZero = low <= 0 && high >= 0;
  return (
    <div className="relative h-5 w-full" aria-hidden="true">
      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-[var(--hairline)]" />
      {d0 < 0 && d1 > 0 && (
        <div className="absolute top-0 h-full w-px bg-zinc-400/60 dark:bg-zinc-500/60" style={{ left: pos(0) }} />
      )}
      <div
        className={`absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full ${crossesZero ? 'bg-zinc-400/50' : 'bg-gold-500/45'}`}
        style={{ left: pos(low), width: `calc(${pos(high)} - ${pos(low)})` }}
      />
      <div
        className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${
          crossesZero ? 'border-zinc-400 bg-[var(--bg-panel)]' : 'border-gold-500 bg-gold-400'
        }`}
        style={{ left: pos(estimate) }}
      />
    </div>
  );
}

export function SegmentedControl<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  ariaLabel: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={`cursor-pointer rounded-md px-2.5 py-1 font-mono text-[11px] font-semibold transition-colors ${
              active
                ? 'bg-gold-500/20 text-gold-800 dark:text-gold-200'
                : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-100'
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export const fmtNum = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
export const fmtP = (p: number) => (!Number.isFinite(p) ? '—' : p < 0.0001 ? '< 0.0001' : p.toFixed(4));
/** “p < 0.0001” or “p = 0.0123”. */
export const pText = (p: number) => (!Number.isFinite(p) ? 'p = —' : p < 0.0001 ? 'p < 0.0001' : `p = ${p.toFixed(4)}`);
