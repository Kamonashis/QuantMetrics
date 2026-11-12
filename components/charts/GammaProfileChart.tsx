'use client';

import React from 'react';

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

  const maxGex = Math.max(...abs, 1);
  const width = 720;
  const rowH = 15;
  const padL = 44;
  const padR = 14;
  const mid = padL + (width - padL - padR) / 2;
  const halfW = (width - padL - padR) / 2;
  const height = trimmed.length * rowH + 30;

  const yOf = (i: number) => i * rowH + 10;

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img" aria-label="Gamma exposure by strike">
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
              >
                <title>{`Strike ${p.strike}\nGEX ${(p.gex / 1e6).toFixed(1)}M $ per 1%\ncall OI ${p.callOi.toLocaleString()} · put OI ${p.putOi.toLocaleString()}`}</title>
              </rect>
            </g>
          );
        })}
        <text x={mid} y={height - 4} textAnchor="middle" fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">
          0 · dealers re-hedge → $ per 1% move
        </text>
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-4 font-mono text-[9px] text-zinc-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[rgba(222,192,132,0.75)]" /> long gamma (re-hedge sells rallies)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[rgba(224,101,95,0.7)]" /> short gamma (re-hedge buys rallies)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-[#DEC084]" /> spot {spot.toFixed(2)} · {flip ? `flip ≈ ${flip.toFixed(2)}` : 'no flip in grid'}</span>
      </div>
    </div>
  );
}
