'use client';

import React, { useMemo } from 'react';
import { useChartSize } from '@/lib/use-chart-size';
import { rollingZ } from '@/lib/vol-signals';

export interface IvRvSpreadPoint {
  date: string;
  iv30: number | null;
  rv: number | null;
}

const GOLD = '#DEC084';
const RV_COLOR = '#7dd3fc';

/**
 * IV30 (accrual) vs realized vol, dual panel:
 * top — level lines (IV30 gold, RV dashed cyan),
 * bottom — rolling z-score (20-session) of the IV−RV spread, with ±1σ guides.
 * Both series must be aligned by date by the caller. Spread < 0 (IV under realizing risk) is shaded gold.
 *
 * Honest-by-design: any day without an accrued IV print lowers the overlap;
 * below 2 overlapping sessions the chart renders a count-aware empty state.
 */
export function IvRvSpreadChart({ points, ivLabel = 'IV30 (accrued)', rvLabel = 'RV 20d' }: { points: IvRvSpreadPoint[]; ivLabel?: string; rvLabel?: string }) {
  const { ref, size } = useChartSize<HTMLDivElement>();

  const data = useMemo(
    () => points.filter((p) => p.iv30 !== null && isFinite(p.iv30 as number) && p.rv !== null && isFinite(p.rv as number)),
    [points],
  );

  const spread = useMemo(() => data.map((p) => (p.iv30 as number) - (p.rv as number)), [data]);
  const spreadZ = useMemo(() => rollingZ(spread, 20), [spread]);

  if (data.length < 2) {
    const accrued = points.filter((p) => p.iv30 !== null).length;
    return (
      <div className="flex h-44 flex-col items-center justify-center gap-1 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-6 text-center">
        <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400">
          {accrued} accrued IV session{accrued === 1 ? '' : 's'}
        </p>
        <p className="max-w-md text-[11px] leading-snug text-zinc-400 dark:text-zinc-500">
          The RV–IV spread needs overlapping IV + RV on the same sessions. The store accrues one snapshot per day you open the chain — z-score meaningful from ~15 sessions, stable by ~60. Nothing is interpolated in the meantime.
        </p>
      </div>
    );
  }

  const width = size.width;
  const height = size.height;
  const ready = width > 0;
  const s = height / 230;
  const pad = { l: 40, r: 12, t: 10 * s, b: 24 * s };
  const gap = 16 * s;
  const topH = 128 * s;
  const botH = height - topH - gap - pad.b;

  const ivMax = Math.max(...data.map((p) => p.iv30 as number));
  const rvMax = Math.max(...data.map((p) => p.rv as number));
  const yMax = Math.max(ivMax, rvMax, 0.1) * 1.12;

  const xOf = (i: number) => pad.l + (i / (data.length - 1)) * (width - pad.l - pad.r);
  const yTop = (v: number) => pad.t + (1 - v / yMax) * topH;
  const zOf = (z: number | null) => (z === null ? null : Math.max(-2.5, Math.min(2.5, z)));
  const yBot = (z: number) => pad.t + topH + gap + (1 - (z + 2.5) / 5) * botH;

  const ivPath = data.map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yTop(p.iv30 as number).toFixed(2)}`).join(' ');
  const rvPath = data.map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(2)},${yTop(p.rv as number).toFixed(2)}`).join(' ');
  const zPath = spreadZ
    .map((z, i) => (z === null ? null : `${i === 0 || spreadZ[i - 1] === null ? 'M' : 'L'}${xOf(i).toFixed(2)},${yBot(zOf(z) as number).toFixed(2)}`))
    .filter(Boolean)
    .join(' ');

  // Shade segments where IV < RV (spread < 0)
  const below: string[] = [];
  let bucket: number | null = null;
  spread.forEach((v, i) => {
    if (v < 0 && bucket === null) bucket = i;
    if (v >= 0 && bucket !== null) {
      below.push(`${xOf(bucket).toFixed(2)},${xOf(i - 1).toFixed(2)}`);
      bucket = null;
    }
  });
  if (bucket !== null) below.push(`${xOf(bucket).toFixed(2)},${xOf(spread.length - 1).toFixed(2)}`);

  const botTop = pad.t + topH + gap;
  const lastSpread = spread[spread.length - 1];
  const lastZ = spreadZ[spreadZ.length - 1];
  const zReady = spreadZ.some((z) => z !== null);

  return (
    <div>
      <div ref={ref} className="chart-frame">
        {ready && (
          <svg width={width} height={height} className="block h-full w-full" role="img" aria-label="Implied minus realized volatility spread and its rolling z-score">
            {/* top panel gridlines */}
            {[0.5, 1].map((f, i) => (
              <line key={i} x1={pad.l} x2={width - pad.r} y1={pad.t + topH * f} y2={pad.t + topH * f} stroke="currentColor" strokeOpacity={0.28} strokeDasharray="3 4" className="text-zinc-400 dark:text-zinc-600" />
            ))}
            {/* zero + ±1 guide lines on the z panel */}
            <line x1={pad.l} x2={width - pad.r} y1={yBot(0)} y2={yBot(0)} stroke="currentColor" strokeOpacity={0.4} strokeDasharray="4 3" className="text-zinc-400 dark:text-zinc-500" />
            {[-1, 1].map((g) => (
              <line key={g} x1={pad.l} x2={width - pad.r} y1={yBot(g)} y2={yBot(g)} stroke="currentColor" strokeOpacity={0.22} strokeDasharray="2 4" className="text-zinc-400 dark:text-zinc-600" />
            ))}

            {/* shade IV < RV segments */}
            {below.map((seg, i) => {
              const [x1, x2] = seg.split(',').map(Number);
              return <rect key={i} x={x1} y={pad.t} width={Math.max(1, x2 - x1)} height={topH} fill="rgba(222,192,132,0.10)" />;
            })}

            <path d={ivPath} fill="none" stroke={GOLD} strokeWidth="1.4" strokeLinejoin="round" />
            <path d={rvPath} fill="none" stroke={RV_COLOR} strokeWidth="1.1" strokeDasharray="4 2.5" strokeLinejoin="round" opacity="1" />
            {zReady && <path d={zPath} fill="none" stroke="currentColor" strokeWidth="0.9" className="text-zinc-500 dark:text-zinc-300" />}

            {/* labels */}
            <text x={pad.l - 6} y={pad.t + 8} fontSize="8.5" textAnchor="end" fontFamily="var(--font-mono)" className="fill-gold-700 dark:fill-gold-300">{(yMax * 100).toFixed(0)}%</text>
            <text x={pad.l - 6} y={pad.t + topH} fontSize="8.5" textAnchor="end" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">0</text>
            <text x={pad.l - 6} y={botTop + botH + 4} fontSize="8.5" textAnchor="end" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">-2.5σ</text>
            <text x={pad.l - 6} y={botTop + 8} fontSize="8.5" textAnchor="end" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">+2.5σ</text>
            <text x={pad.l} y={height - 6} fontSize="8.5" fontFamily="var(--font-mono)" className="fill-zinc-500 dark:fill-zinc-400">{data[0].date.slice(0, 10)}</text>
            <text x={width - pad.r} y={height - 6} fontSize="8.5" fontFamily="var(--font-mono)" textAnchor="end" className="fill-zinc-500 dark:fill-zinc-400">{data[data.length - 1].date.slice(0, 10)}</text>
          </svg>
        )}
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-4 font-mono text-[9px] text-zinc-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[#DEC084]" /> {ivLabel}</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0 w-4 border-t border-dashed border-[#7dd3fc]" /> {rvLabel}</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 bg-gold-500/20" /> IV &lt; RV (realized outrunning implied)</span>
        <span className="ml-auto text-zinc-500">
          spread {lastSpread >= 0 ? '+' : ''}{(lastSpread * 100).toFixed(1)} pts · z {lastZ !== null && lastZ !== undefined ? `${lastZ >= 0 ? '+' : ''}${lastZ.toFixed(2)}` : 'n/a (<20 sessions)'}
        </span>
      </div>
    </div>
  );
}
