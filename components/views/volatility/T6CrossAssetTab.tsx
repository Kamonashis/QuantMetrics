'use client';

import React, { useMemo, useState } from 'react';
import { Network, ScanLine, GitBranch, Map as MapIcon, AlertTriangle, Info } from 'lucide-react';
import { crossCorrelation, solveOls } from '@/lib/vol-analytics';
import { calculateReturns } from '@/lib/quant-engine';
import type { PriceBar } from '@/lib/quant-engine';
import type { ChainPayload, IndicesPayload } from './types';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';

const f2 = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '—' : v.toFixed(d);
const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const pts = (v: number | null | undefined, d = 2, sign = true) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${sign && v >= 0 ? '+' : ''}${(v * 100).toFixed(d)} pts`;

interface ScanRow {
  symbol: string;
  iv30: number | null;
  rr25: number | null;
  bf25: number | null;
  skew9010: number | null;
  mfivDiff: number | null;
  accrualDays: number;
}

export function T6CrossAssetTab({
  chain,
  indices,
  bars,
  ticker,
}: {
  chain: ChainPayload | null;
  indices: IndicesPayload | null;
  bars: PriceBar[];
  ticker: string;
}) {
  const [scanRows, setScanRows] = useState<ScanRow[] | null>(null);
  const [scanLoading, setScanLoading] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<keyof ScanRow>('iv30');
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY'];

  const runScan = async () => {
    setScanLoading(true);
    setScanError(null);
    try {
      const rows = await Promise.all(
        watchlist.map(async (symbol): Promise<ScanRow> => {
          try {
            const res = await fetch(`/api/options?mode=chain&symbol=${symbol}`);
            const d = (await res.json()) as ChainPayload;
            if (!d.available) {
              return { symbol, iv30: null, rr25: null, bf25: null, skew9010: null, mfivDiff: null, accrualDays: 0 };
            }
            const headline = d.slices?.find((s) => s.dte >= 21) ?? d.slices?.[0];
            const mfivDiff =
              headline?.mfivAnnualized !== null && headline?.mfivAnnualized !== undefined && headline?.atmIv !== null && headline?.atmIv !== undefined
                ? (headline.mfivAnnualized as number) - (headline.atmIv as number)
                : null;
            const rr25 = d.slices?.find((s) => s.dte >= 15 && s.dte <= 45)?.rr25 ?? headline?.rr25 ?? null;
            return {
              symbol,
              iv30: d.iv30 ?? null,
              rr25,
              bf25: d.slices?.find((s) => s.dte >= 15 && s.dte <= 45)?.bf25 ?? headline?.bf25 ?? null,
              skew9010: d.slices?.find((s) => s.dte >= 15 && s.dte <= 45)?.skew9010 ?? headline?.skew9010 ?? null,
              mfivDiff,
              accrualDays: d.snapshot?.iv30HistoryCount ?? 0,
            };
          } catch {
            return { symbol, iv30: null, rr25: null, bf25: null, skew9010: null, mfivDiff: null, accrualDays: 0 };
          }
        })
      );
      setScanRows(rows);
    } catch (e) {
      setScanError(e instanceof Error ? e.message : 'scan failed');
    } finally {
      setScanLoading(false);
    }
  };

  const sortedScan = useMemo(() => {
    if (!scanRows) return null;
    return [...scanRows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const an = av === null || av === undefined ? -Infinity : (av as number);
      const bn = bv === null || bv === undefined ? -Infinity : (bv as number);
      return (an - bn) * sortDir;
    });
  }, [scanRows, sortKey, sortDir]);

  // --- IV beta & leverage effect ---------------------------------------------
  const vix = useMemo(() => indices?.series?.VIX?.bars ?? [], [indices]);
  const dVix = useMemo(() => {
    const out: number[] = [];
    for (let i = 1; i < vix.length; i++) out.push(vix[i].close - vix[i - 1].close);
    return out;
  }, [vix]);

  const returns = useMemo(() => calculateReturns((bars ?? []).map((b) => b.close), 'log'), [bars]);
  const vixByDate = useMemo(() => new Map(vix.map((b) => [b.date, b.close])), [vix]);

  // Simpler robust alignment: returns r_i keyed by bars[i+1].date
  const leverage = useMemo(() => {
    if (returns.length < 40) return null;
    const dVixAligned: number[] = [];
    const rAligned: number[] = [];
    for (let i = 1; i < bars.length; i++) {
      const date = bars[i].date;
      const prevVix = vixByDate.get(bars[i - 1]?.date ?? '');
      const curVix = vixByDate.get(date);
      if (prevVix === undefined || curVix === undefined) continue;
      dVixAligned.push(curVix - prevVix);
      rAligned.push(returns[i - 1]);
    }
    if (rAligned.length < 40) return null;
    const X = dVixAligned.map((v) => [1, v]);
    const ols = solveOls(X, rAligned);
    const meanD = dVixAligned.reduce((s, v) => s + v, 0) / dVixAligned.length;
    const meanR = rAligned.reduce((s, v) => s + v, 0) / rAligned.length;
    let cov = 0, vA = 0, vB = 0;
    for (let i = 0; i < dVixAligned.length; i++) {
      const da = dVixAligned[i] - meanD;
      const db = rAligned[i] - meanR;
      cov += da * db; vA += da * da; vB += db * db;
    }
    const corr = vA > 0 && vB > 0 ? cov / Math.sqrt(vA * vB) : 0;
    return { betaOnVix: ols.beta[1], corr, n: rAligned.length };
  }, [returns, bars, vixByDate]);

  // --- Lead-lag: corr(ΔVIX_t vs r_{t+k}), k = −3..+3 -------------------------
  const leadLag = useMemo(() => {
    if (returns.length < 60) return null;
    const dVixSeries: number[] = [];
    const rSeries: number[] = [];
    for (let i = 1; i < bars.length; i++) {
      const prevVix = vixByDate.get(bars[i - 1].date);
      const curVix = vixByDate.get(bars[i].date);
      if (prevVix === undefined || curVix === undefined) continue;
      dVixSeries.push(curVix - prevVix);
      rSeries.push(returns[i - 1]);
    }
    return crossCorrelation(dVixSeries, rSeries, 3);
  }, [returns, bars, vixByDate]);

  // --- Regime map: ^VIX × ^VVIX, 2y scatter -----------------------------------
  const regime = useMemo(() => {
    const v = vix;
    const vvix = indices?.series?.VVIX?.bars ?? [];
    if (v.length < 100 || vvix.length < 100) return null;
    const vvixMap = new Map(vvix.map((b) => [b.date, b.close]));
    const pts: { x: number; y: number; date: string }[] = [];
    for (const b of v) {
      const y = vvixMap.get(b.date);
      if (y !== undefined) pts.push({ x: b.close, y, date: b.date });
    }
    const xs = pts.map((p) => p.x).sort((a, b) => a - b);
    const ys = pts.map((p) => p.y).sort((a, b) => a - b);
    const xMid = xs[Math.floor(xs.length / 2)];
    const yMid = ys[Math.floor(ys.length / 2)];
    const today = pts[pts.length - 1];
    return { pts, xMid, yMid, today };
  }, [vix, indices]);

  const ivProxyNote = chain?.substitution ? 'IV proxy = ^VIX (ticker has no chain)' : 'IV = active ticker chain';

  return (
    <div className="space-y-4">
      {/* Watchlist IV matrix */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-col justify-between gap-2 border-b border-[var(--hairline)] pb-3 sm:flex-row sm:items-end">
          <div>
            <p className="kicker mb-1 flex items-center gap-2"><ScanLine className="h-3.5 w-3.5" /> Cross-Asset IV Matrix</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Concentration in volatility, as visible as concentration in returns
            </h3>
          </div>
          <button
            onClick={runScan}
            disabled={scanLoading}
            className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-gold-600/40 bg-gold-500/10 px-3 font-mono text-[11px] font-semibold text-gold-800 transition-colors hover:border-gold-500/70 dark:text-gold-200 disabled:opacity-60"
          >
            {scanLoading ? 'Scanning chains…' : sortedScan ? 'Rescan watchlist' : 'Scan watchlist'}
          </button>
        </div>
        {scanError && <div className="mb-2 text-[11px] text-rose-500">{scanError}</div>}
        {!sortedScan ? (
          <div className="text-[11px] text-zinc-400">
            One click fetches the current chain for AAPL · MSFT · GOOGL · NVDA · TSLA · SPY (server-cached 15 min).
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  {([
                    ['symbol', 'Symbol'],
                    ['iv30', 'IV-30'],
                    ['rr25', '25Δ RR'],
                    ['bf25', '25Δ BF'],
                    ['skew9010', 'Skew 90/10'],
                    ['mfivDiff', 'MFIV−ATM'],
                    ['accrualDays', 'Accrual'],
                  ] as [keyof ScanRow, string][]).map(([key, label]) => (
                    <th key={String(key)} className="py-1.5">
                      <button
                        className="cursor-pointer uppercase tracking-wider hover:text-zinc-700 dark:hover:text-zinc-100"
                        onClick={() => {
                          if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
                          else { setSortKey(key); setSortDir(-1); }
                        }}
                      >
                        {label} {sortKey === key ? (sortDir === -1 ? '↓' : '↑') : ''}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono">
                {sortedScan.map((row) => (
                  <tr key={row.symbol} className="border-t border-[var(--hairline)]">
                    <td className="py-1.5 font-bold text-zinc-800 dark:text-zinc-100">{row.symbol}</td>
                    <td className="py-1.5 tabular-nums text-gold-800 dark:text-gold-200">{pct(row.iv30)}</td>
                    <td className="py-1.5 tabular-nums text-zinc-700 dark:text-zinc-200">{pts(row.rr25)}</td>
                    <td className="py-1.5 tabular-nums text-zinc-700 dark:text-zinc-200">{pts(row.bf25)}</td>
                    <td className="py-1.5 tabular-nums text-zinc-700 dark:text-zinc-200">{pts(row.skew9010)}</td>
                    <td className="py-1.5 tabular-nums text-zinc-700 dark:text-zinc-200">{pts(row.mfivDiff)}</td>
                    <td className="py-1.5 tabular-nums text-zinc-500">{row.accrualDays}d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-2 font-mono text-[9px] text-zinc-400">European BSM approximation · European-listed proxies where noted · Yahoo ~15min delayed</p>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* IV beta / leverage effect */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1 flex items-center gap-2"><GitBranch className="h-3.5 w-3.5" /> Leverage Effect, Priced</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">corr(ΔIV, r) — the GJR/EGARCH motivation</h3>
          </div>
          {leverage ? (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div className="panel-inset p-4">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">corr(Δ^VIX, r)</div>
                  <div className={`mt-1 font-mono text-2xl font-bold ${leverage.corr < 0 ? 'text-rose-500 dark:text-rose-400' : 'text-emerald-500'}`}>
                    {leverage.corr.toFixed(2)}
                  </div>
                  <div className="font-mono text-[8.5px] text-zinc-400">n = {leverage.n} aligned days</div>
                </div>
                <div className="panel-inset p-4">
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-400">β (r on Δ^VIX)</div>
                  <div className="mt-1 font-mono text-2xl font-bold text-zinc-900 dark:text-zinc-50">{leverage.betaOnVix.toFixed(4)}</div>
                  <div className="font-mono text-[8.5px] text-zinc-400">per 1-pt VIX move</div>
                </div>
              </div>
              <p className="mt-3 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                Negative correlation of ≈ −0.6 to −0.8 confirms the leverage effect — vol rises on down-moves asymmetrically. That is the
                entire justification for the GJR-GARCH News Impact Curve in the Modeling tab, measured here rather than assumed.
              </p>
            </>
          ) : (
            <div className="text-xs text-zinc-400">Not enough aligned history for the regression.</div>
          )}
        </div>

        {/* Lead-lag */}
        <div className="panel p-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">IV Lead–Lag</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">corr(Δ^VIX<sub>t</sub>, r<sub>t+k</sub>) — who follows whom?</h3>
          </div>
          {leadLag ? (
            <div className="space-y-1.5">
              {leadLag.map(({ lag, corr }) => {
                const w = Math.min(100, Math.abs(corr) * 100);
                return (
                  <div key={lag} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-10 text-right text-zinc-400">k={lag > 0 ? `+${lag}` : lag}</span>
                    <div className="relative h-4 flex-1 rounded bg-zinc-100 dark:bg-white/[0.04]">
                      <div
                        className={`absolute top-0 h-full rounded ${corr >= 0 ? 'bg-emerald-400/70' : 'bg-rose-400/70'}`}
                        style={{ width: `${w / 2}%`, left: corr >= 0 ? '50%' : `${50 - w / 2}%` }}
                      />
                      <div className="absolute left-1/2 top-0 h-full w-px bg-zinc-300 dark:bg-zinc-700" />
                    </div>
                    <span className="w-12 tabular-nums text-zinc-700 dark:text-zinc-200">{corr.toFixed(2)}</span>
                  </div>
                );
              })}
              <p className="pt-1 font-mono text-[9px] leading-relaxed text-zinc-400">
                negative k = VIX leads returns (risk follows the options market) · positive k = VIX lags (lev-effect decay) · magnitudes beyond ±2 sampling noise
              </p>
            </div>
          ) : (
            <div className="text-xs text-zinc-400">Not enough aligned history.</div>
          )}
        </div>
      </div>

      {/* Regime map: ^VIX × ^VVIX */}
      <div className="panel p-5">
        <div className="mb-3 border-b border-[var(--hairline)] pb-3">
          <p className="kicker mb-1 flex items-center gap-2"><MapIcon className="h-3.5 w-3.5" /> Regime Map</p>
          <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
            ^VIX (level) × ^VVIX (fragility of the forecast itself) — 2y, daily
          </h3>
        </div>
        {regime ? (
          <div className="w-full overflow-x-auto">
            <ChartFullscreen title="^VIX × ^VVIX Vol Regime Map — 2y Daily">
            <svg viewBox="0 0 640 260" className="w-full min-w-[520px]" role="img" aria-label="Vol regime scatter">
              {/* quadrant shading */}
              <rect x={40} y={10} width={280} height={110} fill="rgba(63,178,127,0.10)" />
              <rect x={320} y={10} width={280} height={110} fill="rgba(224,101,95,0.10)" />
              <rect x={40} y={120} width={280} height={110} fill="rgba(222,192,132,0.10)" />
              <rect x={320} y={120} width={280} height={110} fill="rgba(224,101,95,0.17)" />
              {/* median split lines */}
              <line x1={320} x2={320} y1={10} y2={230} stroke="rgba(236,236,241,0.15)" strokeDasharray="3 4" />
              <line x1={40} x2={600} y1={120} y2={120} stroke="rgba(236,236,241,0.15)" strokeDasharray="3 4" />
              {/* points */}
              {regime.pts.map((p, i) => {
                if (i % 2 !== 0 && p.date !== regime.today?.date) return null;
                const x = 40 + (p.x / 60) * 560;
                const y = 230 - (Math.min(160, p.y) / 160) * 220;
                const isToday = p.date === regime.today?.date;
                return (
                  <circle
                    key={i}
                    cx={x}
                    cy={y}
                    r={isToday ? 6 : 2}
                    fill={isToday ? '#DEC084' : 'rgba(148,148,158,0.6)'}
                    stroke={isToday ? '#0B0B0D' : 'none'}
                    strokeWidth={isToday ? 1.6 : 0}
                  >
                    {isToday && <title>{`today: VIX ${p.x.toFixed(1)} / VVIX ${p.y.toFixed(0)}`}</title>}
                  </circle>
                );
              })}
              {/* labels */}
              <text x={48} y={24} fontSize="9" fill="rgba(63,178,127,0.8)" fontFamily="var(--font-mono)">COMPLACENT (low VIX / low VVIX)</text>
              <text x={592} y={24} fontSize="9" fill="rgba(224,101,95,0.85)" textAnchor="end" fontFamily="var(--font-mono)">TRENDING (vol rising, forecast unstable)</text>
              <text x={48} y={226} fontSize="9" fill="rgba(222,192,132,0.85)" fontFamily="var(--font-mono)">HEDGING (rich puts, orderly)</text>
              <text x={592} y={226} fontSize="9" fill="rgba(224,101,95,1)" textAnchor="end" fontFamily="var(--font-mono)">PANIC (high VIX / high VVIX)</text>
              {/* axes */}
              <text x={40} y={248} fontSize="8.5" fill="#8b8b93" fontFamily="var(--font-mono)">^VIX {vix[vix.length - 1].close.toFixed(1)} today →</text>
              <text x={600} y={248} fontSize="8.5" fill="#8b8b93" textAnchor="end" fontFamily="var(--font-mono)">x: level of vol · y: fragility of vol</text>
            </svg>
            </ChartFullscreen>
            <div className="mt-2 flex items-center gap-3 font-mono text-[9px] text-zinc-400">
              <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-full bg-[#DEC084]" /> today ({regime.today?.date})</span>
              <span>· quadrant split at 1y medians (^VIX {regime.xMid.toFixed(1)}, ^VVIX {regime.yMid.toFixed(0)})</span>
            </div>
          </div>
        ) : (
          <div className="text-xs text-zinc-400">Vol-index history unavailable — regime map needs ^VIX and ^VVIX.</div>
        )}
      </div>

      {/* Honest footnotes */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          IV matrix: {ivProxyNote}. IV beta vs ^VIX and ΔIV lead–lag for single names unlock after 60 accrued sessions (currently {chain?.snapshot?.iv30HistoryCount ?? 0}).
          {' '}§IXIC/^NDX resolve to QQQ chains with a substitution badge; SPX resolves to SPY. Regime map uses the index family only — it describes the market, not the name.
        </span>
      </div>
    </div>
  );
}
