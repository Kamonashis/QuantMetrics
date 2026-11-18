'use client';

import React from 'react';
import { useChartSize } from '@/lib/use-chart-size';
import type { RealizedVolConeBand } from '@/lib/vol-analytics';

export interface TermStructurePoint {
  dte: number;
  iv: number | null;
}

/**
 * IV term structure. Can plot on the vol axis (σ) or the total-variance axis
 * (w = σ²T) — the quantity that is actually additive across maturities and a
 * variance swap prices. Reference band = 1y 10/90 percentile of ^VIX (or closest
 * available index series), labelled as reference only.
 *
 * With `showCone`, the historical realized-vol distribution per lookback tenor
 * is drawn behind the curve as nested percentile bands, with the current RV
 * line underneath the chain IV. The VIX reference band is suppressed in that
 * mode — two bands on one axis would be unreadable.
 */
export function VolTermStructureChart({
  points,
  axisMode,
  band,
  iv30Marker,
  cone = null,
  showCone = false,
}: {
  points: TermStructurePoint[];
  axisMode: 'vol' | 'totalVariance';
  band?: { low: number; high: number; label: string } | null;
  iv30Marker?: number | null;
  cone?: RealizedVolConeBand[] | null;
  showCone?: boolean;
}) {
  const { ref, size } = useChartSize<HTMLDivElement>();
  const valid = points.filter((p): p is { dte: number; iv: number } => p.iv !== null && isFinite(p.iv) && p.iv > 0);
  const coneSorted = showCone && cone
    ? [...cone].filter((b) => b.n >= 2).sort((a, b) => a.dte - b.dte)
    : [];
  const activeBand = coneSorted.length >= 2 ? null : band;
  const coneVisible = coneSorted.length >= 2;

  if (valid.length < 2 && !coneVisible) {
    return (
      <div className="flex h-44 items-center justify-center rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] text-xs text-zinc-400">
        Not enough expiries to draw the curve
      </div>
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const pad = { l: 40, r: 12, t: 14, b: 26 };

  const yOf = (p: { dte: number; iv: number }) =>
    axisMode === 'vol' ? p.iv * 100 : p.iv * p.iv * (p.dte / 365) * 1e4;
  const coneVal = (v: number, dte: number) =>
    axisMode === 'vol' ? v * 100 : v * v * (dte / 365) * 1e4;

  const allDtes = [...valid.map((p) => p.dte), ...coneSorted.map((b) => b.dte)];
  const xMin = Math.min(...allDtes);
  const xMax = Math.max(...allDtes);
  const spanX = Math.max(1, xMax - xMin);

  const ys = [
    ...valid.map(yOf),
    ...coneSorted.flatMap((b) => [
      coneVal(b.min, b.dte), coneVal(b.max, b.dte),
      coneVal(b.p10, b.dte), coneVal(b.p90, b.dte),
      coneVal(b.p25, b.dte), coneVal(b.p75, b.dte),
    ]),
  ];
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (activeBand && axisMode === 'vol') {
    yMin = Math.min(yMin, activeBand.low);
    yMax = Math.max(yMax, activeBand.high);
  }
  const spanY = Math.max(1e-6, yMax - yMin) * 1.15;
  yMin -= spanY * 0.08;
  yMax += spanY * 0.08;

  const xOf = (dte: number) => pad.l + ((dte - xMin) / spanX) * (width - pad.l - pad.r);
  const yPix = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);

  const linePath = valid
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.dte).toFixed(2)},${yPix(yOf(p)).toFixed(2)}`)
    .join(' ');

  const bandPix =
    activeBand && axisMode === 'vol'
      ? { y1: yPix(activeBand.high), y2: yPix(activeBand.low) }
      : null;

  const conePolygon = (hi: (b: RealizedVolConeBand) => number, lo: (b: RealizedVolConeBand) => number) => {
    const top = coneSorted.map((b) => `${xOf(b.dte).toFixed(2)},${yPix(coneVal(hi(b), b.dte)).toFixed(2)}`);
    const bottom = [...coneSorted].reverse().map((b) => `${xOf(b.dte).toFixed(2)},${yPix(coneVal(lo(b), b.dte)).toFixed(2)}`);
    return `M ${[...top, ...bottom].join(' L ')} Z`;
  };
  const coneLine = (pick: (b: RealizedVolConeBand) => number) =>
    coneSorted
      .map((b, i) => `${i === 0 ? 'M' : 'L'}${xOf(b.dte).toFixed(2)},${yPix(coneVal(pick(b), b.dte)).toFixed(2)}`)
      .join(' ');

  const xTicks = coneVisible
    ? coneSorted.map((b) => b.dte)
    : valid.filter((_, i) => i % 2 === 0).map((p) => p.dte);

  return (
    <div>
      <div ref={ref} className="chart-frame">
      {ready && (
      <svg
        width={width}
        height={height}
        className="block h-full w-full"
        role="img"
        aria-label={coneVisible ? 'Realized volatility cone with implied volatility term structure' : 'IV term structure'}
      >
        {/* realized-vol cone: nested historical percentile bands */}
        {coneVisible && (
          <g>
            <path d={conePolygon((b) => b.max, (b) => b.min)} fill="rgba(161,161,170,0.055)" stroke="none" />
            <path d={conePolygon((b) => b.p90, (b) => b.p10)} fill="rgba(161,161,170,0.10)" stroke="none" />
            <path d={conePolygon((b) => b.p75, (b) => b.p25)} fill="rgba(161,161,170,0.17)" stroke="none" />
            <path d={coneLine((b) => b.median)} fill="none" stroke="rgba(212,212,216,0.55)" strokeWidth="1" strokeDasharray="4 3" />
            <path d={coneLine((b) => b.current)} fill="none" stroke="rgba(228,228,232,0.9)" strokeWidth="1.3" />
            {coneSorted.map((b) => (
              <circle key={`cone-${b.dte}`} cx={xOf(b.dte)} cy={yPix(coneVal(b.current, b.dte))} r="2.6" fill="rgba(228,228,232,0.9)">
                <title>{`${b.dte}d RV · current ${(b.current * 100).toFixed(1)}% · median ${(b.median * 100).toFixed(1)}% · 10–90 ${(b.p10 * 100).toFixed(1)}–${(b.p90 * 100).toFixed(1)}% · n=${b.n} overlapping windows`}</title>
              </circle>
            ))}
          </g>
        )}

        {/* VIX reference band (hidden while the cone is shown) */}
        {bandPix && (
          <rect x={pad.l} y={bandPix.y1} width={width - pad.l - pad.r} height={Math.max(1, bandPix.y2 - bandPix.y1)} fill="rgba(222,192,132,0.07)" />
        )}
        {bandPix && (
          <>
            <line x1={pad.l} x2={width - pad.r} y1={bandPix.y1} y2={bandPix.y1} stroke="rgba(222,192,132,0.35)" strokeDasharray="3 4" />
            <line x1={pad.l} x2={width - pad.r} y1={bandPix.y2} y2={bandPix.y2} stroke="rgba(222,192,132,0.35)" strokeDasharray="3 4" />
          </>
        )}

        {/* IV30 vertical marker */}
        {iv30Marker && (
          <line
            x1={xOf(30)} x2={xOf(30)} y1={pad.t} y2={height - pad.b}
            stroke="rgba(236,236,241,0.18)" strokeDasharray="2 4"
          />
        )}

        {/* chain IV curve — gold, always on top */}
        <path d={linePath} fill="none" stroke="#DEC084" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
        {valid.map((p) => (
          <circle key={p.dte} cx={xOf(p.dte)} cy={yPix(yOf(p))} r="3.4" fill="#0B0B0D" stroke="#DEC084" strokeWidth="1.05">
            <title>{`${p.dte}d: ${(p.iv * 100).toFixed(2)}% IV${axisMode === 'totalVariance' ? ` · w=${(p.iv * p.iv * (p.dte / 365) * 1e4).toFixed(1)} bp` : ''}`}</title>
          </circle>
        ))}

        {/* axes */}
        <text x={pad.l - 6} y={pad.t + 4} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {axisMode === 'vol' ? (yMax).toFixed(0) + '%' : (yMax).toFixed(0) + 'bp'}
        </text>
        <text x={pad.l - 6} y={height - pad.b} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">
          {axisMode === 'vol' ? (yMin).toFixed(0) + '%' : (yMin).toFixed(0)}
        </text>
        {xTicks.map((dte) => (
          <text key={`tick-${dte}`} x={xOf(dte)} y={height - 8} fontSize="8.5" fill="#8b8b93" textAnchor="middle" fontFamily="var(--font-mono)">
            {dte}d
          </text>
        ))}
      </svg>
      )}
      </div>

      {activeBand && axisMode === 'vol' && (
        <p className="mt-1.5 font-mono text-[9px] text-zinc-400">
          dashed band = {activeBand.label} · full curve labelled; markers = chain ATM IV
        </p>
      )}
      {coneVisible && (
        <>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[9px] text-zinc-400">
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{ background: 'rgba(161,161,170,0.17)' }} /> RV 25–75</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{ background: 'rgba(161,161,170,0.10)' }} /> RV 10–90</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{ background: 'rgba(161,161,170,0.055)' }} /> RV min–max</span>
            <span>dashed = RV median · light solid = current RV · gold = chain IV</span>
          </div>
          <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-400">
            Cone = trailing realized-vol distribution over the available daily history (up to ~1y); overlapping windows, not independent draws. RV √252 and IV √365 are annualizations of the same variance, compared directly. Historical RV is not a forecast; chain IV is ~15-min-delayed European BSM.
          </p>
        </>
      )}
    </div>
  );
}
