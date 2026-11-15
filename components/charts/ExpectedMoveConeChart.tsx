'use client';

import React, { useMemo } from 'react';
import { useChartSize } from '@/lib/use-chart-size';
import type { PriceBar } from '@/lib/quant-engine';

const GOLD = '#DEC084';

/**
 * Price trajectory with a forward expected-move cone from spot.
 * Historical closes (left) flow into ±k·σ·√(t/252) bands fanning right from the last close.
 * Drift-free (risk-neutral), European BSM convention, matching lib/options.expectedMove:
 * half-width = k · spot · σ · √(f / 252), f in trading days.
 */
export function ExpectedMoveConeChart({ bars, iv30, horizonTd = 21 }: { bars: PriceBar[]; iv30: number | null; horizonTd?: number }) {
  const { ref, size } = useChartSize<HTMLDivElement>();

  const hist = useMemo(() => (bars ?? []).slice(-180), [bars]);
  const spot = hist.length > 0 ? hist[hist.length - 1].close : null;

  const chart = useMemo(() => {
    if (iv30 === null || !spot || hist.length < 2 || !isFinite(spot) || spot <= 0) return null;
    const nHist = hist.length;
    const total = nHist + horizonTd;
    const closes = hist.map((b) => b.close);
    const bandAt = (f: number, k: number) => k * spot * iv30 * Math.sqrt(f / 252);
    const upper1 = (f: number) => spot + bandAt(f, 1);
    const lower1 = (f: number) => spot - bandAt(f, 1);
    const upper2 = (f: number) => spot + bandAt(f, 2);
    const lower2 = (f: number) => Math.max(0, spot - bandAt(f, 2));

    let yMin = Infinity;
    let yMax = -Infinity;
    for (const c of closes) {
      if (c < yMin) yMin = c;
      if (c > yMax) yMax = c;
    }
    for (const b of [upper2(horizonTd), lower2(horizonTd), upper1(horizonTd), lower1(horizonTd)]) {
      if (b < yMin) yMin = b;
      if (b > yMax) yMax = b;
    }
    const padY = (yMax - yMin) * 0.06;
    yMin -= padY;
    yMax += padY;

    return { nHist, total, closes, yMin, yMax, upper1, lower1, upper2, lower2 };
  }, [hist, spot, iv30, horizonTd]);

  if (!chart || iv30 === null || spot === null) {
    return (
      <div className="flex h-44 flex-col items-center justify-center gap-1 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-6 text-center">
        <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400">
          Cone needs live ATM IV
        </p>
        <p className="max-w-md text-[11px] leading-snug text-zinc-400 dark:text-zinc-500">
          The forward fan is priced off spot × IV30 × √(t/252). The option chain is unavailable right now, so no cone is drawn — rather than one built on a stale or synthetic IV.
        </p>
      </div>
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const { nHist, total, closes, yMin, yMax, upper1, lower1, upper2, lower2 } = chart;

  const pad = { l: 46, r: 40, t: 12, b: 22 };
  const chartW = width - pad.l - pad.r;
  const chartH = height - pad.t - pad.b;
  const xOf = (i: number) => pad.l + (i / (total - 1)) * chartW;
  const yOf = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * chartH;

  // Historical path
  const histPath = closes.map((c, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yOf(c).toFixed(2)}`).join(' ');

  // Cone polygons (f = 0 coincides with today's point)
  const up2: string[] = [];
  const dn2: string[] = [];
  for (let f = 0; f <= horizonTd; f++) {
    up2.push(`${xOf(nHist - 1 + f).toFixed(2)},${yOf(upper2(f)).toFixed(2)}`);
    dn2.push(`${xOf(nHist - 1 + f).toFixed(2)},${yOf(lower2(f)).toFixed(2)}`);
  }
  const cone2 = `M ${up2.join(' L ')} L ${[...dn2].reverse().join(' L ')} Z`;
  const up1: string[] = [];
  const dn1: string[] = [];
  for (let f = 0; f <= horizonTd; f++) {
    up1.push(`${xOf(nHist - 1 + f).toFixed(2)},${yOf(upper1(f)).toFixed(2)}`);
    dn1.push(`${xOf(nHist - 1 + f).toFixed(2)},${yOf(lower1(f)).toFixed(2)}`);
  }
  const cone1 = `M ${up1.join(' L ')} L ${[...dn1].reverse().join(' L ')} Z`;
  const medianPath = `M ${xOf(nHist - 1).toFixed(2)},${yOf(spot).toFixed(2)} L ${xOf(total - 1).toFixed(2)},${yOf(spot).toFixed(2)}`;

  const oneSigma = (f: number) => spot * iv30 * Math.sqrt(f / 252);
  const marks = [10, 21].filter((f) => f <= horizonTd);

  return (
    <div>
      <div ref={ref} className="chart-frame">
        {ready && (
          <svg width={width} height={height} className="block h-full w-full" role="img" aria-label="Historical price with forward expected-move cone">
            {/* horizontal gridlines */}
            {[0.25, 0.5, 0.75].map((f, i) => {
              const v = yMin + f * (yMax - yMin);
              return (
                <g key={i}>
                  <line x1={pad.l} x2={width - pad.r} y1={yOf(v)} y2={yOf(v)} stroke="currentColor" strokeOpacity={0.28} strokeDasharray="3 4" className="text-zinc-400 dark:text-zinc-600" />
                  <text x={pad.l - 6} y={yOf(v) + 3} fontSize="9" textAnchor="end" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400 tabular-nums">{v.toFixed(0)}</text>
                </g>
              );
            })}

            {/* cone fills */}
            <path d={cone2} fill={GOLD} opacity="0.09" stroke="none" />
            <path d={cone1} fill={GOLD} opacity="0.16" stroke="none" />
            <path d={medianPath} fill="none" stroke={GOLD} strokeWidth="1" strokeDasharray="5 4" opacity="0.8" />

            {/* 1σ / 2σ rail lines */}
            {[
              { fn: upper1, k: 1, side: 'up' },
              { fn: lower1, k: 1, side: 'dn' },
              { fn: upper2, k: 2, side: 'up' },
              { fn: lower2, k: 2, side: 'dn' },
            ].map(({ fn, k, side }) => {
              const pts: string[] = [];
              for (let f = 0; f <= horizonTd; f++) {
                pts.push(`${xOf(nHist - 1 + f).toFixed(2)},${yOf(fn(f)).toFixed(2)}`);
              }
              return <path key={`${k}${side}`} d={`M ${pts.join(' L ')}`} fill="none" stroke={GOLD} strokeWidth={k === 1 ? 1.4 : 0.8} strokeDasharray={k === 1 ? '4 3' : '2 4'} opacity={k === 1 ? 1 : 0.6} />;
            })}

            {/* today divider */}
            <line x1={xOf(nHist - 1)} x2={xOf(nHist - 1)} y1={pad.t} y2={pad.t + chartH} stroke="currentColor" strokeOpacity={0.45} strokeDasharray="4 3" className="text-zinc-500 dark:text-zinc-400" />
            <text x={xOf(nHist - 1)} y={pad.t + 9} fontSize="8.5" stroke="none" textAnchor="middle" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">today</text>

            {/* forward-horizon markers */}
            {marks.map((f) => (
              <g key={f}>
                <line x1={xOf(nHist - 1 + f)} x2={xOf(nHist - 1 + f)} y1={pad.t} y2={pad.t + chartH} stroke={GOLD} strokeWidth="0.8" strokeDasharray="2 4" opacity="0.75" />
                <text x={xOf(nHist - 1 + f)} y={pad.t + chartH - 2} fontSize="8.5" textAnchor="middle" fontFamily="var(--font-mono)" className="fill-gold-700 dark:fill-gold-300">+{f}td</text>
              </g>
            ))}

            {/* historical price line */}
            <path d={histPath} fill="none" stroke={GOLD} strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" opacity="0.95" />

            {/* right-edge 1σ labels */}
            <text x={width - pad.r + 2} y={yOf(upper1(horizonTd)) + 3} fontSize="9" fontFamily="var(--font-mono)" className="fill-gold-700 dark:fill-gold-300">+1σ</text>
            <text x={width - pad.r + 2} y={yOf(lower1(horizonTd)) + 3} fontSize="9" fontFamily="var(--font-mono)" className="fill-gold-700 dark:fill-gold-300">−1σ</text>

            {/* x ticks */}
            <text x={pad.l} y={height - 6} fontSize="9" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">{hist[0].date.slice(0, 10)}</text>
          </svg>
        )}
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[9px] text-zinc-400">
        <span>spot <span className="text-zinc-200 dark:text-zinc-100">{spot.toFixed(2)}</span></span>
        <span>IV30 <span className="text-zinc-200 dark:text-zinc-100">{(iv30 * 100).toFixed(1)}%</span></span>
        <span>±1σ @10td <span className="text-zinc-200 dark:text-zinc-100">${oneSigma(10).toFixed(1)}</span></span>
        <span>±1σ @21td <span className="text-zinc-200 dark:text-zinc-100">${oneSigma(21).toFixed(1)}</span></span>
        <span className="ml-auto">±k·σ·√(t/252) from spot · no drift · European BSM · ~15-min delayed IV</span>
      </div>
    </div>
  );
}
