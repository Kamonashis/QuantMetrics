'use client';

import React from 'react';
import { useChartSize } from '@/lib/use-chart-size';
import { ChartTooltip, useChartTooltip } from '@/components/ui/chart-tooltip';

export interface GexStrikePoint {
  strike: number;
  callOi: number;
  putOi: number;
  gex: number;
}

/**
 * GEX by strike: diverging horizontal bars (gold = re-hedge into strength,
 * rose = re-hedge into weakness) around a signed zero line, spot marked in.
 * Gamma flip shown as an amber rule when present.
 */
export function GammaProfileChart({
  points,
  spot,
  flip,
  maxBars = 34,
}: {
  points: GexStrikePoint[];
  spot: number;
  flip: number | null;
  maxBars?: number;
}) {
  const sorted = [...points].sort((a, b) => a.strike - b.strike);
  // Trim to the band around spot that carries signal (drop vanishing tails)
  const abs = points.map((p) => Math.abs(p.gex));
  const threshold = Math.max(...abs, 1) * 0.02;
  const meaningful = sorted.filter((p) => Math.abs(p.gex) >= threshold);
  const trimmed = meaningful.length > maxBars
    ? meaningful.slice(
        Math.max(0, Math.floor((meaningful.length - maxBars) / 2)),
        Math.max(0, Math.floor((meaningful.length - maxBars) / 2)) + maxBars
      )
    : meaningful;

  const { ref, size } = useChartSize<HTMLDivElement>();
  const { tooltipState, showTooltip, hideTooltip } = useChartTooltip();
  const maxGex = Math.max(...abs, 1);
  const width = size.width;
  const ready = width > 0;
  const rowH = 15;
  const padL = 44;
  const padR = 14;
  const mid = padL + Math.max(1, width - padL - padR) / 2;
  const halfW = Math.max(1, width - padL - padR) / 2;
  const height = trimmed.length * rowH + 30;

  const yOf = (i: number) => i * rowH + 10;

  return (
    <div>
      <div ref={ref} className="relative w-full max-h-[540px] overflow-y-auto contain-[layout_paint]">
      {ready && (
      <svg width={width} height={height} className="block h-auto w-full" role="img" aria-label="Gamma exposure by strike" onMouseLeave={hideTooltip}>
        {/* zero rule */}
        <line x1={mid} x2={mid} y1={4} y2={height - 18} stroke="rgba(236,236,241,0.25)" strokeDasharray="3 3" />
        {/* spot rule */}
        {(() => {
          // locate nearest strike row to spot
          const idx = trimmed.findIndex((p) => p.strike >= spot);
          const y = idx === -1 ? (trimmed.length - 0.5) * rowH + 10 : yOf(idx) + rowH / 2;
          return (
            <line x1={padL - 6} x2={width - padR} y1={y} y2={y} stroke="#DEC084" strokeWidth="0.9" strokeDasharray="5 3" opacity="0.8" />
          );
        })()}
        {/* flip rule */}
        {flip !== null && flip > 0 && (() => {
          const idx = trimmed.findIndex((p) => p.strike >= flip);
          const y = idx === -1 ? (trimmed.length - 0.5) * rowH + 10 : yOf(idx) + rowH / 2;
          return (
            <line x1={padL - 6} x2={width - padR} y1={y} y2={y} stroke="rgba(222,192,132,0.5)" strokeWidth="0.65" strokeDasharray="2 4" />
          );
        })()}

        {trimmed.map((p, i) => {
          const w = Math.max(1.5, (Math.abs(p.gex) / maxGex) * halfW);
          const up = p.gex >= 0;
          const y = yOf(i);
          return (
            <g key={p.strike}>
              <text x={padL - 6} y={y + rowH / 2 + 1} textAnchor="end" fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">
                {p.strike}
              </text>
              <rect
                x={up ? mid : mid - w}
                y={y + 2}
                width={w}
                height={rowH - 4}
                rx="1.5"
                fill={up ? 'rgba(222,192,132,0.75)' : 'rgba(224,101,95,0.7)'}
                onMouseMove={(e) => showTooltip(e, [`Strike ${p.strike}`, `GEX ${(p.gex / 1e6).toFixed(1)}M $ per 1%`, `call OI ${p.callOi.toLocaleString()} · put OI ${p.putOi.toLocaleString()}`])}
              />
            </g>
          );
        })}
        <text x={mid} y={height - 4} textAnchor="middle" fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">
          0 · dealers re-hedge → $ per 1% move
        </text>
      </svg>
      )}
      <ChartTooltip state={tooltipState} />
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-4 font-mono text-[9px] text-zinc-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[rgba(222,192,132,0.75)]" /> long gamma (re-hedge sells rallies)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[rgba(224,101,95,0.7)]" /> short gamma (re-hedge buys rallies)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-[#DEC084]" /> spot {spot.toFixed(2)} · {flip ? `flip ≈ ${flip.toFixed(2)}` : 'no flip in grid'}</span>
      </div>
    </div>
  );
}
