'use client';

import React, { useMemo } from 'react';

/** Structural subset of an RND payload — SVI-fitted (`RndOut`) or market-price (`MarketRnd`). */
export interface DensityCurve {
  strikes: number[];
  density: number[];
}

/**
 * Risk-neutral density (gold, Breeden–Litzenberger) vs realized 21d return
 * histogram (zinc). Different axis maths, made comparable: the market's curve is
 * over terminal price K; we re-express it per unit of log-return k = ln(K/S₀).
 */
export function RndChart({
  rnd,
  fittedRnd,
  spot,
  realizedReturns,
  primaryLabel = 'RND (Breeden–Litzenberger)',
}: {
  rnd: DensityCurve;
  /** optional second curve, drawn dashed — e.g. the SVI-fitted density for comparison */
  fittedRnd?: DensityCurve | null;
  spot: number;
  realizedReturns: number[]; // horizon log returns, decimals
  primaryLabel?: string;
}) {
  const width = 680;
  const height = 240;
  const pad = { l: 38, r: 12, t: 12, b: 24 };

  // Re-express RND on the log-return axis: q_k = q(K) · K
  const kbd: { k: number; q: number }[] = useMemo(() => {
    return rnd.strikes.map((K, i) => ({
      k: Math.log(K / spot),
      q: rnd.density[i] * K,
    }));
  }, [rnd, spot]);

  const fittedKbd = useMemo(() => {
    if (!fittedRnd || fittedRnd.strikes.length < 2) return null;
    return fittedRnd.strikes.map((K, i) => ({
      k: Math.log(K / spot),
      q: fittedRnd.density[i] * K,
    }));
  }, [fittedRnd, spot]);

  // Histogram of realized horizon log returns
  const hist = useMemo(() => {
    if (realizedReturns.length < 20) return { bins: [] as { x0: number; x1: number; count: number }[], maxCount: 0, binWidth: 0.01 };
    const sorted = [...realizedReturns].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.01)];
    const hi = sorted[Math.floor(sorted.length * 0.99)];
    const nBins = 41;
    const binWidth = (hi - lo) / nBins;
    const bins = Array.from({ length: nBins }, (_, i) => ({ x0: lo + i * binWidth, x1: lo + (i + 1) * binWidth, count: 0 }));
    for (const r of realizedReturns) {
      const idx = Math.min(nBins - 1, Math.max(0, Math.floor((r - lo) / binWidth)));
      bins[idx].count++;
    }
    const maxCount = Math.max(...bins.map((b) => b.count));
    return { bins, maxCount, binWidth };
  }, [realizedReturns]);

  // y scale: common density scale (histogram density = count/(n*binWidth); each RND curve normalized by its own area ≈ 1)
  const rndArea = kbd.reduce((s, p, i) => {
    if (i === 0) return s;
    return s + ((kbd[i - 1].q + p.q) / 2) * (kbd[i].k - kbd[i - 1].k);
  }, 0);
  const fittedArea = fittedKbd?.reduce((s, p, i) => (i === 0 ? s : s + ((fittedKbd[i - 1].q + p.q) / 2) * (fittedKbd[i].k - fittedKbd[i - 1].k)), 0) ?? 0;
  const allK = fittedKbd ? [...kbd.map((p) => p.k), ...fittedKbd.map((p) => p.k)] : kbd.map((p) => p.k);
  const primaryMaxN = Math.max(...kbd.map((p) => p.q / Math.max(rndArea, 1e-9)));
  const fittedMaxN = fittedKbd ? Math.max(...fittedKbd.map((p) => p.q / Math.max(fittedArea, 1e-9))) : 0;
  const scaledMaxQ = Math.max(primaryMaxN, fittedMaxN);
  const kMinHist = hist.bins.length > 0 ? hist.bins[0].x0 : -0.3;
  const kMaxHist = hist.bins.length > 0 ? hist.bins[hist.bins.length - 1].x1 : 0.3;
  const kMin = Math.min(Math.min(...allK), kMinHist);
  const kMax = Math.max(Math.max(...allK), kMaxHist);
  const spanK = Math.max(1e-6, kMax - kMin);

  const histDensityMax = hist.bins.length > 0 ? hist.maxCount / (realizedReturns.length * hist.binWidth) : 0;
  const yMax = Math.max(scaledMaxQ, histDensityMax, 1e-6) * 1.15;

  const xPix = (k: number) => pad.l + ((k - kMin) / spanK) * (width - pad.l - pad.r);
  const yPix = (q: number) => pad.t + (1 - q / yMax) * (height - pad.t - pad.b);

  // RND paths normalized so their area within the visible window ≈ 1
  const rndPath = kbd.map((p, i) => `${i === 0 ? 'M' : 'L'}${xPix(p.k).toFixed(2)},${yPix(p.q / Math.max(rndArea, 1e-9)).toFixed(2)}`).join(' ');
  const fittedPath = fittedKbd
    ? fittedKbd.map((p, i) => `${i === 0 ? 'M' : 'L'}${xPix(p.k).toFixed(2)},${yPix(p.q / Math.max(fittedArea, 1e-9)).toFixed(2)}`).join(' ')
    : null;

  const barW = hist.binWidth > 0 ? (hist.binWidth / spanK) * (width - pad.l - pad.r) : 0;

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img" aria-label="Risk-neutral vs realized density">
        {/* zero return line */}
        <line x1={xPix(0)} x2={xPix(0)} y1={pad.t} y2={height - pad.b} stroke="rgba(236,236,241,0.14)" strokeDasharray="2 3" />

        {/* realized histogram as bars */}
        {hist.bins.map((b, i) => {
          const density = b.count / (Math.max(1, realizedReturns.length) * hist.binWidth);
          const x = xPix(b.x0);
          const y = yPix(density);
          return (
            <rect
              key={i}
              x={x + 0.5}
              y={y}
              width={Math.max(1, barW - 1)}
              height={height - pad.b - y}
              fill="rgba(163,163,173,0.32)"
              stroke="rgba(163,163,173,0.35)"
              strokeWidth="0.3"
            >
              <title>{`${(b.x0 * 100).toFixed(1)}% to ${(b.x1 * 100).toFixed(1)}%\n${b.count} occurrences · density ${density.toFixed(1)}`}</title>
            </rect>
          );
        })}

        {/* RND curve with soft fill */}
        <path
          d={`${rndPath} L${xPix(kbd[kbd.length - 1].k).toFixed(2)},${height - pad.b} L${xPix(kbd[0].k).toFixed(2)},${height - pad.b} Z`}
          fill="rgba(222,192,132,0.13)"
          stroke="none"
        />
        <path d={rndPath} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
        {fittedPath && (
          <path d={fittedPath} fill="none" stroke="rgba(228,228,231,0.75)" strokeWidth="1.2" strokeDasharray="4 3" strokeLinejoin="round" strokeLinecap="round" />
        )}

        {/* axis labels */}
        {[kMin, 0, kMax].map((k, i) => (
          <text
            key={i}
            x={i === 1 ? xPix(0) : i === 0 ? pad.l : width - pad.r}
            y={height - 6}
            fontSize="8.5"
            fill="#8b8b93"
            fontFamily="var(--font-mono)"
            textAnchor={i === 1 ? 'middle' : i === 0 ? 'start' : 'end'}
          >
            {(k * 100).toFixed(0)}%
          </text>
        ))}
        <text x={4} y={pad.t + 8} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">density</text>
      </svg>
      <div className="mt-1.5 flex flex-wrap items-center gap-4 font-mono text-[9px] text-zinc-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[#DEC084]" /> {primaryLabel}</span>
        {fittedPath && (
          <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-zinc-300" /> RND (SVI fit)</span>
        )}
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 bg-zinc-400/40" /> Realized 21d log-return distribution</span>
      </div>
    </div>
  );
}
