'use client';

import React, { useMemo, useState } from 'react';

export const REGIME_PALETTES: Record<number, string[]> = {
  2: ['rgba(63,178,127,0.9)', 'rgba(224,101,95,0.9)'],
  3: ['rgba(63,178,127,0.9)', 'rgba(222,192,132,0.9)', 'rgba(224,101,95,0.9)'],
  4: ['rgba(63,178,127,0.9)', 'rgba(163,196,90,0.85)', 'rgba(222,192,132,0.9)', 'rgba(224,101,95,0.9)'],
};

export const REGIME_NAMES: Record<number, string[]> = {
  2: ['Calm', 'Stress'],
  3: ['Calm', 'Normal', 'Crisis'],
  4: ['Calm', 'Watchful', 'Normal', 'Crisis'],
};

export function regimePalette(k: number): string[] {
  return REGIME_PALETTES[Math.max(2, Math.min(4, k))];
}

export function regimeNames(k: number): string[] {
  return REGIME_NAMES[Math.max(2, Math.min(4, k))];
}

interface RegimeTimelineProps {
  /** smoothed posteriors [T][K], ordered calm→crisis */
  posteriors: number[][];
  /** optional explicit labels; defaults to Calm/Normal… ordering */
  labels?: string[];
  className?: string;
  height?: number;
  /** Max bars to render (decimates uniformly) */
  maxBars?: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function RegimeTimeline({ posteriors, labels, className, height = 44, maxBars = 500 }: RegimeTimelineProps) {
  const [hover, setHover] = useState<{ idx: number; x: number; maxX: number } | null>(null);

  const K = posteriors[0]?.length ?? 0;
  const names = labels ?? regimeNames(K);
  const palette = regimePalette(K);

  const stride = Math.max(1, Math.ceil(posteriors.length / maxBars));
  const bars = useMemo(() => {
    const out: { start: number; dominant: number; color: string; label: string }[] = [];
    for (let i = 0; i < posteriors.length; i += stride) {
      const row = posteriors[i];
      let dominant = 0;
      let best = -Infinity;
      for (let k = 0; k < K; k++) {
        const p = row[k] ?? 0;
        if (p > best) { best = p; dominant = k; }
      }
      const base = palette[dominant].match(/[\d.]+/g)!.map(Number);
      const alpha = 0.25 + 0.75 * (row[dominant] ?? 0);
      out.push({ start: i, dominant, color: `rgba(${base[0]},${base[1]},${base[2]},${alpha.toFixed(2)})`, label: names[dominant] });
    }
    return out;
  }, [posteriors, K, stride, names, palette]);

  const width = 1000;
  const barW = Math.max(1, Math.floor(width / bars.length));

  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * width;
    const idx = clamp(Math.floor(x / (width / posteriors.length)), 0, posteriors.length - 1);
    setHover({ idx, x: e.clientX - rect.left, maxX: Math.max(90, rect.width - 90) });
  };

  return (
    <div className={`relative w-full ${className ?? ''}`}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full select-none"
        style={{ height }}
        role="img"
        aria-label="Volatility regime timeline"
        onMouseMove={onMouseMove}
        onMouseLeave={() => setHover(null)}
      >
        {bars.map((b, i) => (
          <rect
            key={i}
            x={i * barW - 0.5}
            y={0}
            width={barW + 1}
            height={height}
            fill={b.color}
          />
        ))}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 rounded-md border border-gold-500/40 bg-zinc-950/95 px-2 py-1.5 font-mono text-[9px] leading-snug text-zinc-200 shadow-lg"
          style={{ left: clamp(hover.x, 90, hover.maxX), bottom: '110%' }}
        >
          <div className="mb-0.5 text-zinc-400">t = {hover.idx}</div>
          {names.map((n, k) => (
            <div key={k} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: palette[k] }} />
                {n}
              </span>
              <span style={{ color: palette[k] }}>{((posteriors[hover.idx][k] ?? 0) * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-1 flex items-center justify-between font-mono text-[8px] text-zinc-500">
        <span>{posteriors.length} obs</span>
        <div className="flex items-center gap-2">
          {names.map((n, k) => (
            <span key={k} className="flex items-center gap-1">
              <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: palette[k] }} />
              {n}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
