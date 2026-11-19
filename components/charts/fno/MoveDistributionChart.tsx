'use client';

import React, { useMemo } from 'react';
import { ChartTooltip, useChartTooltip } from '@/components/ui/chart-tooltip';
import { EmptyState } from '@/components/ui/empty-state';

/**
 * Distribution of historical horizon |moves| with the option-market's implied
 * move overlaid as a gold rule. Answers "is the straddle rich or cheap right now?"
 */
export function MoveDistributionChart({
  samples,
  impliedPct,
  horizonLabel,
}: {
  samples: number[];
  impliedPct: number | null;
  horizonLabel: string;
}) {
  const { tooltipState, showTooltip, hideTooltip } = useChartTooltip();
  const width = 680;
  const height = 190;
  const pad = { l: 36, r: 14, t: 12, b: 24 };

  const hist = useMemo(() => {
    if (samples.length < 20) {
      return { bins: [] as { x0: number; x1: number; count: number }[], maxCount: 1, p25: 0, p50: 0, p75: 0, cap: 0 };
    }
    const sorted = [...samples].sort((a, b) => a - b);
    const p25 = sorted[Math.floor(sorted.length * 0.25)];
    const p50 = sorted[Math.floor(sorted.length * 0.5)];
    const p75 = sorted[Math.floor(sorted.length * 0.75)];
    const max = sorted[sorted.length - 1];
    const cap = Math.min(max, p75 * 2.5);
    const nBins = 30;
    const binWidth = cap / nBins;
    const bins = Array.from({ length: nBins }, (_, i) => ({ x0: i * binWidth, x1: (i + 1) * binWidth, count: 0 }));
    for (const s of sorted) {
      if (s >= cap) continue;
      const idx = Math.min(nBins - 1, Math.floor(s / binWidth));
      bins[idx].count++;
    }
    return { bins, maxCount: Math.max(1, ...bins.map((b) => b.count)), p25, p50, p75, cap };
  }, [samples]);

  if (hist.bins.length === 0) {
    return (
      <EmptyState title="Not enough history in this sample" detail="Need at least 20 realized-move windows" className="h-40" />
    );
  }

  const xOf = (v: number) => pad.l + (v / hist.cap) * (width - pad.l - pad.r);
  const yOf = (count: number) => pad.t + (1 - count / hist.maxCount) * (height - pad.t - pad.b);
  const barW = Math.max(4, ((hist.bins[0].x1 - hist.bins[0].x0) / hist.cap) * (width - pad.l - pad.r) - 1);

  return (
    <div className="relative w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[520px]" role="img" aria-label="Move distribution" onMouseLeave={hideTooltip}>
        {/* P75 gridline */}
        {[hist.p25, hist.p50, hist.p75].map((q, i) => (
          <g key={i}>
            <line x1={xOf(q)} x2={xOf(q)} y1={pad.t} y2={height - pad.b} stroke="rgba(236,236,241,0.13)" strokeDasharray="3 4" />
            <text x={xOf(q)} y={pad.t - 2} fontSize="7.5" fill="#8b8b93" fontFamily="var(--font-mono)" textAnchor="middle">
              P{[25, 50, 75][i]}
            </text>
          </g>
        ))}
        {/* bars */}
        {hist.bins.map((b, i) => {
          const x = xOf(b.x0);
          const y = yOf(b.count);
          return (
            <rect
              key={i} x={x} y={y} width={barW} height={height - pad.b - y}
              fill="rgba(222,192,132,0.32)" stroke="rgba(222,192,132,0.4)" strokeWidth="0.3"
              onMouseMove={(e) => showTooltip(e, [`|move| ${(b.x0 * 100).toFixed(1)}–${(b.x1 * 100).toFixed(1)}%`, `${b.count} sessions`])}
            />
          );
        })}
        {/* implied rule */}
        {impliedPct !== null && impliedPct > 0 && (
          <g>
            <line x1={xOf(Math.min(impliedPct / 100, hist.cap))} x2={xOf(Math.min(impliedPct / 100, hist.cap))} y1={pad.t} y2={height - pad.b} stroke="#DEC084" strokeWidth="1.3" />
            <text x={xOf(Math.min(impliedPct / 100, hist.cap))} y={height - 4} fontSize="8.5" fill="#DEC084" fontFamily="var(--font-mono)" textAnchor="middle">
              implied {impliedPct.toFixed(1)}%
            </text>
          </g>
        )}
        <text x={pad.l} y={height - 4} fontSize="8" fill="#8b8b93" fontFamily="var(--font-mono)">0%</text>
        <text x={width - pad.r} y={height - 4} fontSize="8" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">{(hist.cap * 100).toFixed(0)}%+</text>
      </svg>
      <ChartTooltip state={tooltipState} />
      <p className="mt-1 font-mono text-[9px] text-zinc-400">
        {horizonLabel} · n = {samples.length.toLocaleString()} windows · histogram capped at P75×2.5 · implied = 0.8×ATM straddle
      </p>
    </div>
  );
}
