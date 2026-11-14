'use client';

import React from 'react';
import type { HmmFit } from '@/lib/hmm';
import { regimePalette, regimeNames } from '@/components/charts/RegimeTimeline';

/**
 * Regime chip — current-state badge for panel headers.
 * Shows dominant state + posterior + entropy + expected duration.
 * States are discovered, not labelled a priori — names are rank-based.
 */
export function RegimeChip({ fit, className, labels }: { fit: HmmFit | null; className?: string; labels?: string[] }) {
  if (!fit) {
    return (
      <span className={`inline-flex items-center gap-1.5 rounded-md border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/60 px-2 py-0.5 font-mono text-[10px] text-zinc-400 ${className ?? ''}`}>
        HMM: did not converge
      </span>
    );
  }
  const K = fit.k;
  const names = labels ?? regimeNames(K);
  const palette = regimePalette(K);
  const t = fit.posteriors.length - 1;
  const row = fit.posteriors[t] ?? [];
  let dominant = 0;
  let best = -Infinity;
  let entropy = 0;
  for (let k = 0; k < K; k++) {
    const p = row[k] ?? 0;
    if (p > best) { best = p; dominant = k; }
    if (p > 1e-12) entropy -= p * Math.log(p);
  }
  const expectedDur = fit.expectedDurations[dominant];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 font-mono text-[10px] leading-tight ${className ?? ''}`}
      style={{
        borderColor: palette[dominant].replace(/,\s*[\d.]+\)$/, ',0.55)'),
        background: palette[dominant].replace(/,\s*[\d.]+\)$/, ',0.12)'),
        color: palette[dominant].replace(/,\s*[\d.]+\)$/, ',1)'),
      }}
      title={`HMM-${K} states · logL ${fit.logLikelihood.toFixed(1)} · BIC ${fit.bic.toFixed(1)} · states ranked calm→crisis by mean vol feature`}
    >
      <span className="h-2 w-2 rounded-full" style={{ background: palette[dominant] }} />
      <span>{names[dominant]} {(best * 100).toFixed(0)}%</span>
      <span className="opacity-60">·</span>
      <span className="opacity-80">E[τ] {expectedDur.toFixed(1)}d · H={entropy.toFixed(2)}</span>
    </span>
  );
}
