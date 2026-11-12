'use client';

import React from 'react';
import { Crosshair, Info } from 'lucide-react';
import type { PanelPayload } from './types';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';

export function FnoMaxPainTab({ panel, symbol }: { panel: PanelPayload | null; symbol: string }) {
  const pain = panel?.maxPain ?? null;
  const spot = panel?.quote?.price ?? 0;
  const bars = pain?.curve ?? [];
  const maxPainStrike = pain?.maxPainStrike ?? null;
  const maxPainValue = pain?.pain ?? 0;
  const maxY = Math.max(1, ...bars.map((b) => b.pain));

  return (
    <div className="space-y-4">
      {/* Summary row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'Max Pain', value: maxPainStrike !== null ? String(maxPainStrike) : '—', tone: 'text-gold-700 dark:text-gold-300' },
          { label: 'Distance', value: pain?.distancePct !== null && pain?.distancePct !== undefined ? `${pain.distancePct >= 0 ? '+' : ''}${pain.distancePct.toFixed(1)}%` : '—', tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'Basin width', value: pain?.basinWidthPct !== null && pain?.basinWidthPct !== undefined ? `${pain.basinWidthPct.toFixed(1)}%` : '—', tone: 'text-zinc-900 dark:text-zinc-50' },
          { label: 'Spot', value: spot > 0 ? spot.toFixed(2) : '—', tone: 'text-zinc-900 dark:text-zinc-50' },
        ].map((s) => (
          <div key={s.label} className="panel-inset p-4">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{s.label}</div>
            <div className={`mt-1 font-mono text-2xl font-bold tabular-nums ${s.tone}`}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Pain curve */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2"><Crosshair className="h-3.5 w-3.5" /> Pain Curve</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Payout to option buyers at expiration, by strike — minimum marked
            </h3>
          </div>
          <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
            basin width = strikes ≤ 105% of min pain · distance = |spot − pain| / spot
          </span>
        </div>

        {bars.length === 0 ? (
          <div className="text-xs text-zinc-400">Chain too thin to build a pain curve — insufficient OI at nearby strikes.</div>
        ) : (
          <div className="w-full overflow-x-auto">
            <ChartFullscreen title="Max-Pain Curve">
            <svg viewBox={`0 0 720 220`} className="w-full min-w-[560px]" role="img" aria-label="Pain curve">
              {(() => {
                const midX = 40;
                const rightX = 700;
                const span = rightX - midX;
                const minK = Math.min(...bars.map((b) => b.strike));
                const maxK = Math.max(...bars.map((b) => b.strike));
                const xOf = (K: number) => midX + ((K - minK) / Math.max(1, maxK - minK)) * span;
                const yOf = (p: number) => 20 + (1 - p / maxY) * 160;
                const path = bars.map((b, i) => `${i === 0 ? 'M' : 'L'}${xOf(b.strike).toFixed(1)},${yOf(b.pain).toFixed(1)}`).join(' ');
                const fill = `${path} L${xOf(bars[bars.length - 1].strike).toFixed(1)},180 L${xOf(bars[0].strike).toFixed(1)},180 Z`;
                const spotX = xOf(Math.max(minK, Math.min(maxK, spot)));
                return (
                  <>
                    <path d={fill} fill="rgba(222,192,132,0.12)" stroke="none" />
                    <path d={path} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" />
                    {maxPainStrike !== null && (
                      <line x1={xOf(maxPainStrike)} x2={xOf(maxPainStrike)} y1={14} y2={180} stroke="#DEC084" strokeWidth="0.9" strokeDasharray="4 3" />
                    )}
                    <line x1={spotX} x2={spotX} y1={14} y2={180} stroke="rgba(236,236,241,0.3)" strokeWidth="0.9" />
                    <text x={xOf(maxPainStrike ?? bars[0].strike)} y={196} fontSize="8.5" fill="#DEC084" textAnchor="middle" fontFamily="var(--font-mono)">
                      pain {maxPainStrike}
                    </text>
                    <text x={spotX} y={208} fontSize="8.5" fill="#8b8b93" textAnchor="middle" fontFamily="var(--font-mono)">
                      spot {spot.toFixed(1)}
                    </text>
                  </>
                );
              })()}
            </svg>
            </ChartFullscreen>
          </div>
        )}
      </div>

      {/* Honesty footer */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Max pain is a <strong>heuristic, not a law</strong> — settlement is drawn toward pain only when open interest is concentrated and the basin is narrow. Treat the basin width, not the point estimate, as the signal: a flat basin (current width {pain?.basinWidthPct?.toFixed(1) ?? '—'}%) means the pin itself is arbitrary and you should not trade toward it. Historical hit rate will accrue with daily snapshots.
        </span>
      </div>
    </div>
  );
}
