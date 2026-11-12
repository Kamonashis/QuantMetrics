'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  Magnet,
  ShieldAlert,
  Activity,
  Download,
  ScanLine,
  Info,
  AlertTriangle,
} from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import type { ChainPayload, PanelRowOut, TrapPayload, TrapState } from './types';
import { GammaProfileChart } from '@/components/charts/GammaProfileChart';
import { TrapPriceChart } from '@/components/charts/TrapPriceChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { runTrapBacktest } from '@/lib/trap-backtest';
import { cn } from '@/lib/utils';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const pts = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 1).toFixed(d)}`;
const money = (v: number | null | undefined) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `$${(Math.abs(v) / 1e6).toFixed(1)}M`;
const sign = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;

const STATE_STYLE: Record<TrapState, string> = {
  idle: 'bg-zinc-200/60 text-zinc-500 dark:bg-white/[0.06] dark:text-zinc-400',
  building: 'bg-gold-500/15 text-gold-800 dark:text-gold-300',
  armed: 'bg-gold-500/25 text-gold-800 dark:text-gold-200',
  triggered: 'bg-rose-500/20 text-rose-600 dark:text-rose-400',
  exhausted: 'bg-zinc-500/20 text-zinc-500 dark:text-zinc-400',
};
const STATE_LABEL: Record<TrapState, string> = {
  idle: 'IDLE',
  building: 'BUILDING',
  armed: 'ARMED',
  triggered: 'TRIGGERED',
  exhausted: 'EXHAUSTED',
};

interface T7Props {
  chain: ChainPayload | null;
  panel: PanelRowOut[];
  bars: PriceBar[];
  ticker: string;
}

export function T7DealerTab({ chain, panel, bars, ticker }: T7Props) {
  const trap = chain?.trap ?? null;
  const closes = useMemo(() => (bars ?? []).map((b) => ({ date: b.date, close: b.close })), [bars]);
  const dateIndex = useMemo(() => {
    const m = new Map<string, number>();
    closes.forEach((b, i) => m.set(b.date, i));
    return m;
  }, [closes]);

  const [scanRows, setScanRows] = useState<Array<Record<string, unknown>> | null>(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [sortKey, setSortKey] = useState<'score' | 'expiryConcentration7d' | 'oiChange5d'>('score');
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const backtest = useMemo(
    () => runTrapBacktest(panel as never, closes, 20),
    [panel, closes]
  );

  const runScan = async () => {
    setScanLoading(true);
    try {
      const res = await fetch('/api/options?mode=trap-scan&symbols=AAPL,MSFT,GOOGL,NVDA,TSLA,SPY');
      const d = await res.json();
      setScanRows(Array.isArray(d?.rows) ? d.rows : []);
    } catch {
      setScanRows([]);
    } finally {
      setScanLoading(false);
    }
  };

  const sortedScan = useMemo(() => {
    if (!scanRows) return null;
    return [...scanRows].sort((a, b) => {
      const av = (a[sortKey] as number) ?? -Infinity;
      const bv = (b[sortKey] as number) ?? -Infinity;
      return (av - bv) * sortDir;
    });
  }, [scanRows, sortKey, sortDir]);

  // Forward returns for history table rows (from bars)
  const historyRows = useMemo(() => {
    return [...panel].slice(-25).reverse().map((row) => {
      const i = dateIndex.get(row.date);
      const fwd = (h: number): number | null => {
        if (i === undefined || i + h >= closes.length) return null;
        if (!(closes[i].close > 0)) return null;
        return Math.log(closes[i + h].close / closes[i].close);
      };
      return { row, fwd1d: fwd(1), fwd3d: fwd(3), fwd5d: fwd(5), fwd10d: fwd(10) };
    });
  }, [panel, closes, dateIndex]);

  const sparkline = useMemo(() => panel.slice(-60).map((p) => p.trapScore ?? 0), [panel]);

  const exportCsv = () => {
    const header = 'date,spot,call_wall,call_wall_oi,put_wall,put_wall_oi,expiry_7d,total_gex,gamma_flip,score,state,direction\n';
    const rows = panel.map((p) =>
      [p.date, p.spot, p.callWall?.strike ?? '', p.callWall?.oi ?? '', p.putWall?.strike ?? '', p.putWall?.oi ?? '',
       p.expiryConcentration7d ?? '', p.totalGex ?? '', p.gammaFlip ?? '', p.trapScore ?? '', p.trapState ?? '', p.trapDirection ?? ''].join(',')
    ).join('\n');
    const url = 'data:text/csv;charset=utf-8,' + encodeURIComponent(header + rows);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${ticker}_trap_panel.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!trap) {
    return (
      <div className="panel p-6 text-sm text-zinc-500 dark:text-zinc-400">
        {chain === null ? 'Fetching the chain…' : chain?.reason ?? 'Chain unavailable — trap state needs a live options chain.'}
      </div>
    );
  }

  const trappedWall = trap.nearWall === 'call' ? trap.walls.call : trap.nearWall === 'put' ? trap.walls.put : null;
  const runnerUp = trap.nearWall === 'call' ? trap.walls.runnerUpCall : trap.nearWall === 'put' ? trap.walls.runnerUpPut : null;
  const sparkMax = Math.max(100, ...sparkline);
  const sparkPath = sparkline.length > 1
    ? sparkline.map((v, i) => `${i === 0 ? 'M' : 'L'}${((i / (sparkline.length - 1)) * 110).toFixed(1)},${(34 - (v / sparkMax) * 30).toFixed(1)}`).join(' ')
    : '';

  const topStrikes = [...trap.byStrike]
    .map((s) => ({ strike: s.strike, oi: s.callOi + s.putOi, callOi: s.callOi, putOi: s.putOi }))
    .filter((s) => Math.abs(s.strike / trap.spot - 1) <= 0.15)
    .sort((a, b) => b.oi - a.oi)
    .slice(0, 8);
  const maxOi = Math.max(1, ...topStrikes.map((s) => s.oi));

  const componentDefs: { key: keyof TrapPayload['components']; label: string; max: number }[] = [
    { key: 'wall', label: 'Wall concentration', max: 25 },
    { key: 'proximity', label: 'Proximity to wall', max: 25 },
    { key: 'regime', label: 'Short-gamma regime', max: 25 },
    { key: 'velocity', label: 'OI velocity (5d)', max: 15 },
    { key: 'coil', label: 'Coil (grind, not spike)', max: 10 },
  ];

  return (
    <div className="space-y-4">
      {/* ══════ CARD — trap state ══════ */}
      <div className="panel p-5">
        <div className="flex flex-col gap-4 xl:flex-row">
          {/* Left gutter: OI histogram */}
          <div className="shrink-0 xl:w-44">
            <div className="mb-1.5 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              OI at strike (45d)
            </div>
            <div className="space-y-1">
              {topStrikes.map((s) => {
                const dominant = s.putOi >= s.callOi ? 'put' : 'call';
                return (
                  <div key={s.strike} className="flex items-center gap-1.5">
                    <span className="w-9 text-right font-mono text-[9px] tabular-nums text-zinc-500 dark:text-zinc-400">{s.strike}</span>
                    <div className="h-3.5 flex-1 rounded-sm bg-zinc-100 dark:bg-white/[0.04]">
                      <div
                        className={cn('h-full rounded-sm', dominant === 'put' ? 'bg-rose-400/70' : 'bg-gold-400/80')}
                        style={{ width: `${(s.oi / maxOi) * 100}%` }}
                        title={`${s.strike} · ${dominant} ${s[dominant === 'put' ? 'putOi' : 'callOi'].toLocaleString()} OI`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-2 font-mono text-[8px] leading-relaxed text-zinc-400">gold = calls · rose = puts · bars scaled to local OI</p>
          </div>

          {/* Price chart */}
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">{ticker} — price vs dealer levels</h3>
              <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
                OI as of last close · squeeze convention assumed
              </span>
            </div>
            <ChartFullscreen title="Price vs Dealer Levels" subtitle={`${ticker} · squeeze convention assumed`}>
              <TrapPriceChart
                closes={closes}
                spot={trap.spot}
                callWall={trap.walls.call?.strike ?? null}
                putWall={trap.walls.put?.strike ?? null}
                flip={trap.gammaFlip.flip}
              />
            </ChartFullscreen>
          </div>

          {/* Score + state + hedge flow */}
          <div className="flex w-full shrink-0 flex-col justify-between gap-3 xl:w-64">
            <div>
              <div className="flex items-center justify-between">
                <span className="kicker flex items-center gap-2"><Magnet className="h-3.5 w-3.5" /> Trap Score</span>
                <span className={cn('rounded-md px-2 py-0.5 font-mono text-[11px] font-bold tracking-widest', STATE_STYLE[trap.state])}>
                  {STATE_LABEL[trap.state]}
                </span>
              </div>
              <div className="mt-1.5 flex items-end gap-3">
                <span className="font-mono text-5xl font-bold tabular-nums leading-none text-zinc-900 dark:text-zinc-50">
                  {trap.score}
                </span>
                <span className="pb-1 font-mono text-[10px] text-zinc-400">/ 100</span>
                {sparkPath && (
                  <svg viewBox="0 0 110 36" className="mb-1 h-9 w-24" aria-label="score history">
                    <path d={sparkPath} fill="none" stroke="#DEC084" strokeWidth="1.05" strokeLinejoin="round" strokeLinecap="round" />
                  </svg>
                )}
              </div>
              <div className="mt-2 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-400">
                {trap.direction === 'none' ? 'no directional lean' : `direction: ${trap.direction === 'up' ? '↑ up — call wall' : '↓ down — put wall'}`}
                {' · '}regime: <span className={trap.regime === 'short' ? 'text-rose-500' : 'text-emerald-600 dark:text-emerald-400'}>{trap.regime} gamma</span>
              </div>
            </div>

            {/* Hedge flow sentence */}
            <div className="rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2.5">
              <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">Dealer hedging into ±2% band</div>
              <p className="mt-1 text-[11px] leading-snug text-zinc-600 dark:text-zinc-300">
                Dealers must buy <strong className="text-zinc-900 dark:text-zinc-50">{money(trap.flow.perPctUp)}</strong> per 1% up move; sell{' '}
                <strong className="text-zinc-900 dark:text-zinc-50">{money(trap.flow.perPctDown)}</strong> per 1% down —{' '}
                {trap.flow.net >= 0 ? 'net long-gamma re-hedge' : 'net short-gamma re-hedge'} ({money(trap.flow.net)} net).
              </p>
            </div>
          </div>
        </div>

        {/* Badge line — copy gated on state */}
        <div className="mt-3 border-t border-[var(--hairline)] pt-3">
          <p className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
            {trap.state === 'triggered' && trap.direction === 'up' && (
              <>⚡ Writers trapped at {trap.walls.call?.strike} — price is pressing through the wall.</>
            )}
            {trap.state === 'triggered' && trap.direction === 'down' && (
              <>⚡ Writers trapped at {trap.walls.put?.strike} — price is pressing through the wall.</>
            )}
            {trap.state === 'armed' && (
              <>Coiling {trap.nearWall === 'call' ? `${trap.walls.call?.strike} (call wall)` : `${trap.walls.put?.strike} (put wall)`} — {trappedWall ? `${trappedWall.distSigma.toFixed(2)}σ away` : '—'}. Trade direction unconfirmed; wait for the close through the wall.</>
            )}
            {trap.state === 'building' && trappedWall && (
              <>Writers piling up at {trappedWall.strike} — {pct(trappedWall.share, 0)} of local OI{trap.oiChange5d !== null ? `, ΔOI5d ${trap.oiChange5d >= 0 ? '+' : ''}${trap.oiChange5d.toLocaleString()}` : ''}.</>
            )}
            {trap.state === 'exhausted' && <>Wall absorbed — flow likely spent. Do not chase.</>}
            {trap.state === 'idle' && <>No meaningful dealer pressure building. Nothing to price yet.</>}
          </p>
          <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-400">
            Dealer positioning is assumed, not observed. squeeze convention: dealers long calls / short puts. Flip a dollar sign and the story changes — the toggle exposes the alternative.
          </p>
        </div>
      </div>

      {/* ══════ GEX PROFILE + COMPONENT BREAKDOWN ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <div className="panel p-5 xl:col-span-3">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2"><Activity className="h-3.5 w-3.5" /> Gamma Profile</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Dealer re-hedge intensity by strike (≤ 45 DTE)
              </h3>
            </div>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              total GEX {money(trap.totalGex)} / 1% · per-strike below
            </span>
          </div>
          <ChartFullscreen title="Dealer Re-hedge Intensity by Strike (≤ 45 DTE)" subtitle={`Total GEX ${money(trap.totalGex)} / 1%`}>
            <GammaProfileChart points={trap.byStrike} spot={trap.spot} flip={trap.gammaFlip.flip} />
          </ChartFullscreen>
        </div>

        {/* Component breakdown */}
        <div className="panel p-5 xl:col-span-2">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Why this score</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Component breakdown — auditable, not magic</h3>
          </div>
          <div className="space-y-2.5">
            {componentDefs.map(({ key, label, max }) => {
              const v = trap.components[key];
              const width = Math.min(100, (v / max) * 100);
              const degraded = key === 'velocity' && trap.velocityDegraded;
              return (
                <div key={key}>
                  <div className="flex items-baseline justify-between font-mono text-[10px]">
                    <span className="flex items-center gap-1.5 text-zinc-500 dark:text-zinc-400">
                      {label}
                      {degraded && <span className="rounded border border-gold-500/40 bg-gold-500/10 px-1 text-[8px] text-gold-800 dark:text-gold-300">ΔOI accruing</span>}
                    </span>
                    <span className="tabular-nums text-zinc-700 dark:text-zinc-200">
                      {v.toFixed(1)}<span className="text-zinc-400"> / {max}</span>
                    </span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-zinc-100 dark:bg-white/[0.05]">
                    <div
                      className={cn('h-full rounded-full', degraded ? 'bg-zinc-400/40' : 'bg-gradient-to-r from-gold-700 via-gold-500 to-gold-300')}
                      style={{ width: `${width}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-3 font-mono text-[9px] leading-relaxed text-zinc-400">
            velocity: ΔOI at wall vs trailing 5d baseline{trap.oiChange5dZ !== null ? `, z = ${trap.oiChange5dZ.toFixed(2)}` : ' — needs ≥ 20 accrued sessions (currently fewer)'} · coil: |5d| range vs RV20·√5, damped by RV−IV spread
          </div>
        </div>
      </div>

      {/* ══════ WALLS TABLE + BACKTEST ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Walls table */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Wall Ledger</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Where the open interest actually sits</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <th className="py-1.5 pr-2">Side</th>
                  <th className="py-1.5 pr-2">Strike</th>
                  <th className="py-1.5 pr-2 text-right">OI</th>
                  <th className="py-1.5 pr-2 text-right">ΔOI 1d</th>
                  <th className="py-1.5 pr-2 text-right">ΔOI 5d</th>
                  <th className="py-1.5 pr-2 text-right">Share</th>
                  <th className="py-1.5 text-right">σ away</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {[
                  { w: trap.walls.call, runnerUp: trap.walls.runnerUpCall, label: 'CALL', accent: 'text-gold-700 dark:text-gold-300' },
                  { w: trap.walls.put, runnerUp: trap.walls.runnerUpPut, label: 'PUT', accent: 'text-rose-500 dark:text-rose-400' },
                ].map(({ w, runnerUp, label, accent }) => (
                  <React.Fragment key={label}>
                    <tr className="border-t border-[var(--hairline)]">
                      <td className={cn('py-1.5 pr-2 font-bold', accent)}>{label}</td>
                      <td className="py-1.5 pr-2 tabular-nums text-zinc-900 dark:text-zinc-50">{w?.strike ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{w?.oi.toLocaleString() ?? '—'}</td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                        {w && trap.nearWall === (label === 'CALL' ? 'call' : 'put') && trap.oiChange1d !== null
                          ? `${trap.oiChange1d >= 0 ? '+' : ''}${trap.oiChange1d.toLocaleString()}`
                          : '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                        {w && trap.nearWall === (label === 'CALL' ? 'call' : 'put') && trap.oiChange5d !== null
                          ? `${trap.oiChange5d >= 0 ? '+' : ''}${trap.oiChange5d.toLocaleString()}`
                          : '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{w ? pct(w.share, 1) : '—'}</td>
                      <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{w?.distSigma.toFixed(2) ?? '—'}</td>
                    </tr>
                    {runnerUp && (
                      <tr className="border-t border-[var(--hairline)] opacity-55">
                        <td className="py-1.5 pr-2 font-mono text-[9px] uppercase text-zinc-400">{label} · 2nd</td>
                        <td className="py-1.5 pr-2 tabular-nums text-zinc-500">{runnerUp.strike}</td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">{runnerUp.oi.toLocaleString()}</td>
                        <td className="py-1.5 pr-2 text-right text-zinc-400">—</td>
                        <td className="py-1.5 pr-2 text-right text-zinc-400">—</td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">{pct(runnerUp.share, 1)}</td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-500">{runnerUp.distSigma.toFixed(2)}</td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 font-mono text-[9px] text-zinc-400">share = OI within ±10% of spot (≤ 45 DTE) · ΔOI only for the trapped side — history accrues from first snapshot</p>
        </div>

        {/* Backtest panel */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2"><ShieldAlert className="h-3.5 w-3.5" /> The Verdict</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Did the wall actually hold? — event study of past triggers</h3>
            </div>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">{backtest.nDedupedEvents} deduped events</span>
          </div>

          <div
            className={cn(
              'rounded-lg border px-3 py-2 text-[11px] font-semibold leading-snug',
              backtest.verdict === 'above_base_rate' && 'border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
              backtest.verdict === 'no_reliable_follow_through' && 'border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400',
              backtest.verdict === 'insufficient_history' && 'border-zinc-300 dark:border-white/[0.08] bg-[var(--bg-inset)] text-zinc-500 dark:text-zinc-400',
            )}
          >
            {backtest.verdictLine}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { label: 'Hit rate (5d)', value: backtest.hitRate5d !== null ? `${backtest.hitRate5d.toFixed(0)}%` : '—' },
              { label: 'Base rate', value: backtest.baseRate5d !== null ? `${backtest.baseRate5d.toFixed(0)}%` : '—' },
              { label: 'Mean 5d (dir)', value: sign(backtest.fwdMean.h5) },
              { label: 'Newey–West t', value: backtest.tStat5d !== null ? backtest.tStat5d.toFixed(2) : '—' },
            ].map((s) => (
              <div key={s.label} className="panel-inset p-2.5">
                <div className="font-mono text-[8.5px] uppercase tracking-wider text-zinc-400">{s.label}</div>
                <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">{s.value}</div>
              </div>
            ))}
          </div>

          {backtest.wilsonLo !== null && backtest.wilsonHi !== null && (
            <p className="mt-2 font-mono text-[9px] text-zinc-400">
              Wilson 95% CI for hit rate: [{backtest.wilsonLo.toFixed(0)}%, {backtest.wilsonHi.toFixed(0)}%] · raw triggered rows: {backtest.nRawTriggered} · MFE/MAE median: {backtest.medianMfeMae?.toFixed(2) ?? '—'}
            </p>
          )}

          {backtest.decileTable.length > 0 && (
            <div className="mt-3 overflow-x-auto">
              <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Score decile → mean 5d forward (direction-adjusted)</div>
              <div className="mt-1 flex gap-1">
                {backtest.decileTable.map((d) => (
                  <div key={d.decile} className="flex-1 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] p-1.5 text-center font-mono text-[9px]" title={`decile ${d.decile}: score≈${d.meanScore}, n=${d.n}`}>
                    <div className="text-zinc-400">D{d.decile}</div>
                    <div className={cn('font-bold tabular-nums', (d.mean5d ?? 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500')}>
                      {d.mean5d !== null ? `${d.mean5d >= 0 ? '+' : ''}${d.mean5d.toFixed(1)}%` : '—'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {backtest.decileTable.length === 0 && (
            <p className="mt-2 font-mono text-[9px] text-zinc-400">decile monotonicity unlocks at ≥ 10 triggered events ({backtest.nDedupedEvents} so far)</p>
          )}
        </div>
      </div>

      {/* ══════ HISTORY + WATCHLIST SCAN ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* History table */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1">State History</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Every daily verdict, newest first</h3>
            </div>
            <button
              onClick={exportCsv}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] px-2 py-1.5 font-mono text-[10px] font-semibold text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
            >
              <Download className="h-3.5 w-3.5" /> CSV
            </button>
          </div>
          {historyRows.length === 0 ? (
            <div className="text-xs text-zinc-400">History starts with the first snapshot — one row per day accrues here.</div>
          ) : (
            <div className="max-h-96 overflow-auto">
              <table className="w-full min-w-[520px] text-[10.5px]">
                <thead className="sticky top-0 bg-[var(--bg-panel)]">
                  <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    <th className="py-1.5 pr-2">Date</th>
                    <th className="py-1.5 pr-2">State</th>
                    <th className="py-1.5 pr-2 text-right">Spot</th>
                    <th className="py-1.5 pr-2 text-right">Score</th>
                    <th className="py-1.5 pr-2 text-right">+1d</th>
                    <th className="py-1.5 pr-2 text-right">+5d</th>
                    <th className="py-1.5 text-right">+10d</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {historyRows.map(({ row, fwd1d, fwd5d, fwd10d }) => (
                    <tr key={row.date} className="border-t border-[var(--hairline)]">
                      <td className="py-1.5 pr-2 text-zinc-500">{row.date.slice(5)}</td>
                      <td className="py-1.5 pr-2">
                        <span className={cn('rounded px-1.5 py-0.5 text-[8px] font-bold tracking-wider', STATE_STYLE[row.trapState ?? 'idle'])}>
                          {row.trapState ? STATE_LABEL[row.trapState] : '—'}
                        </span>
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{row.spot.toFixed(2)}</td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{row.trapScore ?? '—'}</td>
                      {[fwd1d, fwd5d, fwd10d].map((v, i) => (
                        <td
                          key={i}
                          className={cn('py-1.5 pr-2 text-right tabular-nums', v === null ? 'text-zinc-300 dark:text-zinc-700' : v >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500 dark:text-rose-400')}
                        >
                          {v === null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Watchlist scan */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2"><ScanLine className="h-3.5 w-3.5" /> Watchlist Trap Scan</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">Which of your names is coiling?</h3>
            </div>
            <button
              onClick={runScan}
              disabled={scanLoading}
              className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-gold-600/40 bg-gold-500/10 px-3 font-mono text-[10px] font-semibold text-gold-800 transition-colors hover:border-gold-500/70 dark:text-gold-200 disabled:opacity-60"
            >
              {scanLoading ? 'Scanning…' : sortedScan ? 'Rescan' : 'Scan watchlist'}
            </button>
          </div>
          {!sortedScan ? (
            <div className="text-[11px] text-zinc-400">Six names: AAPL · MSFT · GOOGL · NVDA · TSLA · SPY — one chain fetch each, cached 15 min server-side.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-[10.5px]">
                <thead>
                  <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    <th className="py-1.5">Symbol</th>
                    <th className="py-1.5">State</th>
                    <th className="cursor-pointer py-1.5 text-right hover:text-zinc-700 dark:hover:text-zinc-100" onClick={() => { if (sortKey === 'score') setSortDir((d) => (d === 1 ? -1 : 1)); else { setSortKey('score'); setSortDir(-1); } }}>
                      Score {sortKey === 'score' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                    <th className="py-1.5 text-right">Dir</th>
                    <th className="py-1.5 text-right">Regime</th>
                    <th className="py-1.5 text-right">Call wall</th>
                    <th className="py-1.5 text-right">Put wall</th>
                    <th className="cursor-pointer py-1.5 text-right hover:text-zinc-700 dark:hover:text-zinc-100" onClick={() => { if (sortKey === 'expiryConcentration7d') setSortDir((d) => (d === 1 ? -1 : 1)); else { setSortKey('expiryConcentration7d'); setSortDir(-1); } }}>
                      Exp 7d {sortKey === 'expiryConcentration7d' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {sortedScan.map((r) => {
                    if (r.available === false) {
                      return (
                        <tr key={String(r.symbol)} className="border-t border-[var(--hairline)] opacity-60">
                          <td className="py-1.5 font-bold text-zinc-500">{String(r.symbol)}</td>
                          <td colSpan={7} className="py-1.5 text-[9px] text-zinc-400">{String(r.reason ?? 'unavailable')}</td>
                        </tr>
                      );
                    }
                    const state = String(r.state ?? 'idle') as TrapState;
                    const cw = r.callWall as { strike: number; share: number } | null | undefined;
                    const pw = r.putWall as { strike: number; share: number } | null | undefined;
                    return (
                      <tr key={String(r.symbol)} className="border-t border-[var(--hairline)]">
                        <td className="py-1.5 font-bold text-zinc-800 dark:text-zinc-100">{String(r.symbol)}</td>
                        <td className="py-1.5"><span className={cn('rounded px-1 py-0.5 text-[8px] font-bold tracking-wider', STATE_STYLE[state])}>{STATE_LABEL[state]}</span></td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-800 dark:text-zinc-100">{Number(r.score ?? 0)}</td>
                        <td className="py-1.5 text-right text-zinc-500">{String(r.direction ?? '—')}</td>
                        <td className="py-1.5 text-right text-zinc-500">{String(r.regime ?? '—')}</td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{cw ? `${cw.strike} (${(cw.share * 100).toFixed(0)}%)` : '—'}</td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{pw ? `${pw.strike} (${(pw.share * 100).toFixed(0)}%)` : '—'}</td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">{pct(Number(r.expiryConcentration7d ?? 0), 0)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ══════ HONESTY FOOTNOTES ══════ */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Dealer positioning is assumed, not observed — squeeze convention: dealers long calls / short puts (flip sign and the story flips with it). OI updates once daily at close; the intraday signal is price moving toward a daily level. End-of-day OI ≠ net short — spreads, collars and multi-leg structures net out. 7d-expiration concentration is {pct(trap.expiryConcentration7d, 0)} of open interest; if it dominates, the wall decays same-day and the signal is not a multi-day story. Short-dated IVs are noisy — this panel uses mid-quote IV for gamma, provider IV only as a cross-check. The verdict panel is the contract: no directional arrow until follow-through clears base rate.
        </span>
      </div>
    </div>
  );
}
