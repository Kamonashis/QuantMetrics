'use client';

import React, { useMemo, useState } from 'react';
import {
  Magnet,
  ShieldAlert,
  Activity,
  Download,
  ScanLine,
  Info,
  ChevronDown,
  Layers,
} from 'lucide-react';
import type { IntradayBar, PriceBar } from '@/lib/quant-engine';
import type {
  ChainPayload,
  GexStrikeOut,
  PanelRowOut,
  TrapPayload,
  TrapState,
} from './types';
import { useTickerSnapshot } from '@/components/providers/TickerSnapshotProvider';
import type { WatchRow } from '@/hooks/use-watchlist';
import { GammaProfileChart, type GammaProfileMetric } from '@/components/charts/GammaProfileChart';
import { TrapPriceChart } from '@/components/charts/TrapPriceChart';
import { GexFlipCurveChart } from '@/components/charts/GexFlipCurveChart';
import { GexExpiryHeatmap, type GexBucketFilter } from '@/components/charts/GexExpiryHeatmap';
import { GexHistoryChart } from '@/components/charts/GexHistoryChart';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { SegmentedControl } from '@/components/ui/lab-ui';
import {
  FINANCIAL_FORMULA_PRESETS,
  MathDisplay,
  type FinancialFormulaPreset,
} from '@/components/ui/MathDisplay';
import { runTrapBacktest } from '@/lib/trap-backtest';
import { RegimeChip } from '@/components/ui/RegimeChip';
import { RegimeTransitionMatrix } from '@/components/ui/RegimeTransitionMatrix';
import { fitGaussianHmm, type HmmFit } from '@/lib/hmm';
import { cn } from '@/lib/utils';

type WatchRowView = WatchRow & {
  totalGex?: number | null;
  gammaFlip?: number | null;
  flipDistPct?: number | null;
  regime?: string | null;
  callWall?: number | null;
  putWall?: number | null;
  gammaCallWall?: number | null;
  gammaPutWall?: number | null;
  expiryConcentration7d?: number | null;
  zeroDteShare?: number | null;
  gexPerAdv?: number | null;
};

type GexConvention = 'squeeze' | 'inverse';
type ScanSortKey = 'score' | 'totalGex' | 'flipDistPct' | 'expiryConcentration7d';

const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;
const signPct = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`;
const fmtSignedUsd = (v: number | null | undefined) => {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  const s = v > 0 ? '+' : v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${s}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${s}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${s}$${(abs / 1e3).toFixed(0)}K`;
  return `${s}$${abs.toFixed(0)}`;
};
const moneyAbs = (v: number | null | undefined) =>
  v === null || v === undefined || !isFinite(v) ? '—' : `$${(Math.abs(v) / 1e6).toFixed(1)}M`;

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

const METHODOLOGY_PRESETS: FinancialFormulaPreset[] = [
  'gamma-bsm',
  'gex',
  'gamma-flip',
  'wall-share',
  'trap-score',
  'hedge-flow',
];

function readWallStrike(val: unknown): number | null {
  if (typeof val === 'number' && isFinite(val)) return val;
  if (val && typeof val === 'object' && 'strike' in val) {
    const s = (val as { strike?: unknown }).strike;
    return typeof s === 'number' && isFinite(s) ? s : null;
  }
  return null;
}

export interface T7Props {
  ticker: string;
  bars: PriceBar[];
  intradayBars?: IntradayBar[];
  intradaySource?: string | null;
  chain?: ChainPayload | null;
  panel?: PanelRowOut[];
}

export function T7DealerTab({
  ticker,
  bars,
  intradayBars,
  intradaySource = null,
  chain: chainProp,
  panel: panelProp,
}: T7Props) {
  const snapshot = useTickerSnapshot();
  const chain = chainProp !== undefined ? chainProp : snapshot.chain;
  const panel = panelProp !== undefined ? panelProp : snapshot.panel;
  const trap = chain?.trap ?? null;

  const [convention, setConvention] = useState<GexConvention>('squeeze');
  const [include0Dte, setInclude0Dte] = useState<boolean>(true);
  const [bucketFilter, setBucketFilter] = useState<GexBucketFilter>('all');
  const [profileMetric, setProfileMetric] = useState<GammaProfileMetric>('gex');
  const [scanLoading, setScanLoading] = useState(false);
  const [sortKey, setSortKey] = useState<ScanSortKey>('score');
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const closes = useMemo(() => (bars ?? []).map((b) => ({ date: b.date, close: b.close })), [bars]);
  const dateIndex = useMemo(() => {
    const m = new Map<string, number>();
    closes.forEach((b, i) => m.set(b.date, i));
    return m;
  }, [closes]);

  const backtest = useMemo(
    () => runTrapBacktest(panel as never, closes, 20),
    [panel, closes]
  );

  // Active convention + 0-DTE view model (pure client-side transform, zero extra fetch)
  const activeView = useMemo(() => {
    if (!trap) return null;
    const serverConv: GexConvention = trap.convention ?? 'squeeze';
    const convMult = convention === serverConv ? 1 : -1;
    const convSummary = trap.byConvention?.[convention];

    const flipRegimeFallback =
      trap.regime === 'long' ? 'short' : trap.regime === 'short' ? 'long' : 'unknown';

    const baseTotalGex = convSummary?.totalGex ?? trap.totalGex * convMult;
    const baseCallGex = convSummary?.totalCallGex ?? (trap.totalCallGex ?? 0) * convMult;
    const basePutGex = convSummary?.totalPutGex ?? (trap.totalPutGex ?? 0) * convMult;
    const baseVolumeGex = convSummary?.totalVolumeGex ?? (trap.totalVolumeGex ?? 0) * convMult;
    const baseCharm = convSummary?.totalCharm ?? (trap.totalCharm ?? 0) * convMult;
    const baseVanna = convSummary?.totalVanna ?? (trap.totalVanna ?? 0) * convMult;
    const activeFlip =
      convSummary?.flip ??
      (convention === 'inverse' ? trap.gammaFlipInverse?.flip : trap.gammaFlip.flip) ??
      trap.gammaFlip.flip;
    const activeRegime =
      convSummary?.regime ??
      (convention === serverConv
        ? trap.regime
        : (trap.gammaFlipInverse?.regime ?? flipRegimeFallback));
    const activeFlow = convSummary?.flow ?? {
      perPctUp: trap.flow.perPctUp * convMult,
      perPctDown: trap.flow.perPctDown * convMult,
      net: trap.flow.net * convMult,
    };
    const activeScore = convSummary?.score ?? trap.score;
    const activeState = convSummary?.state ?? trap.state;
    const activeDirection = convSummary?.direction ?? trap.direction;
    const activeComponents = convSummary?.components ?? trap.components;
    const activeNearWall = convSummary?.nearWall ?? trap.nearWall;

    const zeroBucket = trap.expiryBuckets?.find((b) => b.bucket === '0dte') ?? null;
    const subtractZero = !include0Dte && zeroBucket !== null;

    const totalGex = subtractZero ? baseTotalGex - zeroBucket.totalGex * convMult : baseTotalGex;
    const totalCallGex = subtractZero ? baseCallGex - zeroBucket.callGex * convMult : baseCallGex;
    const totalPutGex = subtractZero ? basePutGex - zeroBucket.putGex * convMult : basePutGex;
    const totalVolumeGex = subtractZero
      ? baseVolumeGex - zeroBucket.totalVolumeGex * convMult
      : baseVolumeGex;
    const totalCharm = subtractZero ? baseCharm - zeroBucket.totalCharm * convMult : baseCharm;
    const totalVanna = subtractZero ? baseVanna - zeroBucket.totalVanna * convMult : baseVanna;

    const netRegime: 'long' | 'short' | 'unknown' = subtractZero
      ? Math.abs(totalGex) < 1e-8
        ? 'unknown'
        : totalGex > 0
          ? 'long'
          : 'short'
      : activeRegime;

    const gexPerAdv =
      trap.adv20Dollar && trap.adv20Dollar > 0
        ? totalGex / trap.adv20Dollar
        : !subtractZero && trap.gexPerAdv !== null && trap.gexPerAdv !== undefined
          ? trap.gexPerAdv * convMult
          : null;

    const flipDistPct =
      activeFlip !== null && trap.spot > 0 ? activeFlip / trap.spot - 1 : (trap.flipDistPct ?? null);
    const flipDistSigma = trap.flipDistSigma ?? null;

    // Filter / subtract byStrike in serverConvention basis so GammaProfileChart applies convMult cleanly
    let activeByStrike: GexStrikeOut[];
    if (bucketFilter === 'all') {
      if (subtractZero && zeroBucket.byStrike.length > 0) {
        const zMap = new Map<number, GexStrikeOut>();
        for (const z of zeroBucket.byStrike) zMap.set(z.strike, z);
        activeByStrike = trap.byStrike.map((row) => {
          const z = zMap.get(row.strike);
          if (!z) return row;
          return {
            ...row,
            callOi: Math.max(0, row.callOi - z.callOi),
            putOi: Math.max(0, row.putOi - z.putOi),
            callVolume: Math.max(0, (row.callVolume ?? 0) - (z.callVolume ?? 0)),
            putVolume: Math.max(0, (row.putVolume ?? 0) - (z.putVolume ?? 0)),
            callGex: (row.callGex ?? 0) - (z.callGex ?? 0),
            putGex: (row.putGex ?? 0) - (z.putGex ?? 0),
            gex: row.gex - z.gex,
            callVolumeGex: (row.callVolumeGex ?? 0) - (z.callVolumeGex ?? 0),
            putVolumeGex: (row.putVolumeGex ?? 0) - (z.putVolumeGex ?? 0),
            volumeGex: (row.volumeGex ?? 0) - (z.volumeGex ?? 0),
            callCharm: (row.callCharm ?? 0) - (z.callCharm ?? 0),
            putCharm: (row.putCharm ?? 0) - (z.putCharm ?? 0),
            charm: (row.charm ?? 0) - (z.charm ?? 0),
            callVanna: (row.callVanna ?? 0) - (z.callVanna ?? 0),
            putVanna: (row.putVanna ?? 0) - (z.putVanna ?? 0),
            vanna: (row.vanna ?? 0) - (z.vanna ?? 0),
          };
        });
      } else {
        activeByStrike = trap.byStrike;
      }
    } else if (bucketFilter === '0dte' && !include0Dte) {
      activeByStrike = [];
    } else {
      activeByStrike =
        trap.expiryBuckets?.find((b) => b.bucket === bucketFilter)?.byStrike ?? [];
    }

    const activeGammaWalls = trap.gammaWalls
      ? {
          ...trap.gammaWalls,
          call: trap.gammaWalls.call
            ? { ...trap.gammaWalls.call, gex: trap.gammaWalls.call.gex * convMult }
            : null,
          put: trap.gammaWalls.put
            ? { ...trap.gammaWalls.put, gex: trap.gammaWalls.put.gex * convMult }
            : null,
          runnerUpCall: trap.gammaWalls.runnerUpCall
            ? { ...trap.gammaWalls.runnerUpCall, gex: trap.gammaWalls.runnerUpCall.gex * convMult }
            : null,
          runnerUpPut: trap.gammaWalls.runnerUpPut
            ? { ...trap.gammaWalls.runnerUpPut, gex: trap.gammaWalls.runnerUpPut.gex * convMult }
            : null,
        }
      : null;

    return {
      serverConv,
      convMult,
      totalGex,
      totalCallGex,
      totalPutGex,
      totalVolumeGex,
      totalCharm,
      totalVanna,
      activeFlip,
      activeRegime: netRegime,
      activeFlow,
      activeScore,
      activeState,
      activeDirection,
      activeComponents,
      activeNearWall,
      gexPerAdv,
      flipDistPct,
      flipDistSigma,
      activeByStrike,
      activeGammaWalls,
      zeroBucketOi: zeroBucket?.oi ?? 0,
    };
  }, [trap, convention, include0Dte, bucketFilter]);

  // Unsupervised dealer-gamma regime — HMM on [r, |r|, ((spot-flip)/flip)/dailyVol]
  const activeFlipForHmm = activeView?.activeFlip ?? null;
  const hmmGamma = useMemo<HmmFit | null>(() => {
    if (!trap || !bars || bars.length < 120) return null;
    if (activeFlipForHmm === null || activeFlipForHmm <= 0) return null;
    const barCloses = bars.map((b) => b.close);
    const rets: number[] = [];
    for (let i = 1; i < barCloses.length; i++) {
      rets.push(Math.log(barCloses[i] / barCloses[i - 1]));
    }
    let m = 0;
    for (const r of rets) m += r;
    m /= rets.length;
    let v = 0;
    for (const r of rets) {
      const d = r - m;
      v += d * d;
    }
    const sd = Math.sqrt(v / rets.length) || 1e-4;
    const feats: number[][] = [];
    for (let i = 1; i < bars.length; i++) {
      const r = rets[i - 1];
      const z = (barCloses[i] - activeFlipForHmm) / activeFlipForHmm / sd;
      feats.push([r, Math.abs(r), z]);
    }
    return fitGaussianHmm(feats, { k: 2, rankFeature: 2 });
  }, [bars, trap, activeFlipForHmm]);

  const runScan = async () => {
    setScanLoading(true);
    try {
      await snapshot.scanWatchlist(true);
    } finally {
      setScanLoading(false);
    }
  };

  const watchRows = snapshot.watchRows as WatchRowView[];
  const sortedScan = useMemo(() => {
    if (!watchRows || watchRows.length === 0) return null;
    const convMult = convention === 'inverse' ? -1 : 1;
    return [...watchRows].sort((a, b) => {
      const getVal = (row: WatchRowView): number => {
        if (sortKey === 'score') return row.trapScore ?? -Infinity;
        if (sortKey === 'totalGex')
          return row.totalGex !== null && row.totalGex !== undefined
            ? row.totalGex * convMult
            : -Infinity;
        if (sortKey === 'flipDistPct') return row.flipDistPct ?? -Infinity;
        return row.expiryConcentration7d ?? -Infinity;
      };
      return (getVal(a) - getVal(b)) * sortDir;
    });
  }, [watchRows, sortKey, sortDir, convention]);

  // Forward returns for history table rows (from bars)
  const historyRows = useMemo(() => {
    const convMult = convention === 'inverse' ? -1 : 1;
    return [...panel]
      .slice(-25)
      .reverse()
      .map((row) => {
        const i = dateIndex.get(row.date);
        const fwd = (h: number): number | null => {
          if (i === undefined || i + h >= closes.length) return null;
          if (!(closes[i].close > 0)) return null;
          return Math.log(closes[i + h].close / closes[i].close);
        };
        const adjustedGex =
          row.totalGex !== null && Number.isFinite(row.totalGex) ? row.totalGex * convMult : null;
        return {
          row,
          adjustedGex,
          fwd1d: fwd(1),
          fwd3d: fwd(3),
          fwd5d: fwd(5),
          fwd10d: fwd(10),
        };
      });
  }, [panel, closes, dateIndex, convention]);

  const sparkline = useMemo(() => panel.slice(-60).map((p) => p.trapScore ?? 0), [panel]);

  const exportCsv = () => {
    const header =
      'date,spot,call_wall,call_wall_oi,put_wall,put_wall_oi,expiry_7d,total_gex,gamma_flip,score,state,direction\n';
    const rows = panel
      .map((p) =>
        [
          p.date,
          p.spot,
          p.callWall?.strike ?? '',
          p.callWall?.oi ?? '',
          p.putWall?.strike ?? '',
          p.putWall?.oi ?? '',
          p.expiryConcentration7d ?? '',
          p.totalGex ?? '',
          p.gammaFlip ?? '',
          p.trapScore ?? '',
          p.trapState ?? '',
          p.trapDirection ?? '',
        ].join(',')
      )
      .join('\n');
    const url = 'data:text/csv;charset=utf-8,' + encodeURIComponent(header + rows);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${ticker}_trap_panel.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!trap || !activeView) {
    return (
      <div className="panel p-6 text-sm text-zinc-500 dark:text-zinc-400">
        {chain === null
          ? 'Fetching the chain…'
          : chain?.reason ?? 'Chain unavailable — GEX & dealer positioning require a live options chain.'}
      </div>
    );
  }

  const trappedWall =
    activeView.activeNearWall === 'call'
      ? trap.walls.call
      : activeView.activeNearWall === 'put'
        ? trap.walls.put
        : null;

  const sparkMax = Math.max(100, ...sparkline);
  const sparkPath =
    sparkline.length > 1
      ? sparkline
          .map(
            (v, i) =>
              `${i === 0 ? 'M' : 'L'}${((i / (sparkline.length - 1)) * 110).toFixed(1)},${(34 - (v / sparkMax) * 30).toFixed(1)}`
          )
          .join(' ')
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

  const bucketFilterLabel =
    bucketFilter === 'all'
      ? include0Dte
        ? 'All ≤ 45 DTE'
        : 'All ≤ 45 DTE (ex 0-DTE)'
      : bucketFilter === '0dte'
        ? '0-DTE (0d)'
        : bucketFilter === 'weekly'
          ? 'Weekly (1–14d)'
          : bucketFilter === 'monthly'
            ? 'Monthly / OPEX'
            : 'Far (46–90d)';

  const handleSortClick = (key: ScanSortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
    else {
      setSortKey(key);
      setSortDir(-1);
    }
  };

  const gexDayDelta =
    snapshot.session?.totalGexChange !== null && snapshot.session?.totalGexChange !== undefined
      ? snapshot.session.totalGexChange * activeView.convMult
      : null;

  return (
    <div className="space-y-4">
      {/* ══════ WORKBENCH HEADER: CONTROLS + SECOND-ORDER KPI STRIP ══════ */}
      <div className="panel p-5">
        <div className="flex flex-col gap-3 border-b border-[var(--hairline)] pb-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="kicker mb-1 flex items-center gap-2">
              <Layers className="h-3.5 w-3.5" /> Gamma Exposure &amp; Dealer Positioning Workbench
            </p>
            <h2 className="font-display text-base font-semibold text-zinc-900 dark:text-zinc-50">
              {ticker} — Spot ${trap.spot.toFixed(2)} ·{' '}
              <span
                className={
                  activeView.activeRegime === 'short'
                    ? 'text-[var(--negative)]'
                    : activeView.activeRegime === 'long'
                      ? 'text-[var(--positive)]'
                      : 'text-zinc-500'
                }
              >
                {activeView.activeRegime === 'long'
                  ? 'Long-Gamma Stabilizing Regime'
                  : activeView.activeRegime === 'short'
                    ? 'Short-Gamma Amplifying Regime'
                    : 'Neutral / Unknown Regime'}
              </span>
            </h2>
            <p className="mt-0.5 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
              {convention === 'squeeze'
                ? 'Assumed Squeeze convention: dealers long calls / short puts (+GEX dampens moves above flip; −GEX amplifies below).'
                : 'Assumed Inverse convention: dealers short calls / long puts (sign inverted across GEX, Charm, Vanna, Flip regime & Trap score).'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              options={[
                { value: 'squeeze' as const, label: 'Squeeze (Long C / Short P)' },
                { value: 'inverse' as const, label: 'Inverse (Short C / Long P)' },
              ]}
              value={convention}
              onChange={setConvention}
              ariaLabel="Dealer positioning convention"
            />
            <SegmentedControl
              options={[
                { value: 'incl' as const, label: 'Incl. 0-DTE' },
                { value: 'excl' as const, label: 'Ex-0DTE' },
              ]}
              value={include0Dte ? 'incl' : 'excl'}
              onChange={(v) => setInclude0Dte(v === 'incl')}
              ariaLabel="Include or exclude 0-DTE contracts"
            />
          </div>
        </div>

        {/* 8-tile exposure & second-order matrix */}
        <div className="mt-4 grid grid-cols-2 gap-2.5 sm:grid-cols-4 xl:grid-cols-8">
          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Net GEX (≤45d)</div>
            <div
              className={cn(
                'mt-0.5 font-mono text-lg font-bold tabular-nums',
                activeView.totalGex >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
              )}
            >
              {fmtSignedUsd(activeView.totalGex)}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              {activeView.gexPerAdv !== null
                ? `${signPct(activeView.gexPerAdv, 2)} of 20d ADV$`
                : gexDayDelta !== null
                  ? `1d Δ ${fmtSignedUsd(gexDayDelta)}`
                  : 'per 1% spot move'}
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Call vs Put GEX</div>
            <div className="mt-0.5 font-mono text-xs font-bold tabular-nums text-gold-700 dark:text-gold-300">
              C {fmtSignedUsd(activeView.totalCallGex)}
            </div>
            <div className="font-mono text-xs font-bold tabular-nums text-rose-500 dark:text-rose-400">
              P {fmtSignedUsd(activeView.totalPutGex)}
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Gamma Flip</div>
            <div className="mt-0.5 font-mono text-lg font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
              {activeView.activeFlip !== null ? `$${activeView.activeFlip.toFixed(2)}` : '—'}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              {activeView.flipDistPct !== null
                ? `${signPct(activeView.flipDistPct, 2)}${activeView.flipDistSigma !== null ? ` · ${activeView.flipDistSigma.toFixed(2)}σ` : ''}`
                : 'outside ±15% grid'}
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Volume-GEX</div>
            <div
              className={cn(
                'mt-0.5 font-mono text-lg font-bold tabular-nums',
                activeView.totalVolumeGex >= 0 ? 'text-gold-700 dark:text-gold-300' : 'text-[var(--negative)]'
              )}
            >
              {fmtSignedUsd(activeView.totalVolumeGex)}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              intraday flow / 1%
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Charm (1d Drift)</div>
            <div
              className={cn(
                'mt-0.5 font-mono text-lg font-bold tabular-nums',
                activeView.totalCharm >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
              )}
            >
              {fmtSignedUsd(activeView.totalCharm)}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              Δ notional / day
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Vanna (+1 vol pt)</div>
            <div
              className={cn(
                'mt-0.5 font-mono text-lg font-bold tabular-nums',
                activeView.totalVanna >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
              )}
            >
              {fmtSignedUsd(activeView.totalVanna)}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              Δ notional / +1σv
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">0-DTE Share</div>
            <div className="mt-0.5 font-mono text-lg font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
              {include0Dte ? pct(trap.zeroDteShare ?? 0, 1) : 'Excluded'}
            </div>
            <div className="mt-0.5 font-mono text-[9.5px] text-zinc-500 dark:text-zinc-400">
              7d OI conc: {pct(trap.expiryConcentration7d, 0)}
            </div>
          </div>

          <div className="panel-inset p-3">
            <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">OI vs γ Walls</div>
            <div className="mt-0.5 font-mono text-[11px] font-bold tabular-nums text-zinc-800 dark:text-zinc-100">
              OI: {trap.walls.call?.strike ?? '—'} / {trap.walls.put?.strike ?? '—'}
            </div>
            <div className="font-mono text-[10.5px] tabular-nums text-gold-700 dark:text-gold-300">
              γ: {activeView.activeGammaWalls?.call?.strike ?? '—'} / {activeView.activeGammaWalls?.put?.strike ?? '—'}
            </div>
          </div>
        </div>
      </div>

      {/* ══════ CARD — PRICE VS DEALER LEVELS + TRAP STATE ══════ */}
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
                    <span className="w-9 text-right font-mono text-[9px] tabular-nums text-zinc-500 dark:text-zinc-400">
                      {s.strike}
                    </span>
                    <div className="h-3.5 flex-1 rounded-sm bg-zinc-100 dark:bg-white/[0.04]">
                      <div
                        className={cn(
                          'h-full rounded-sm',
                          dominant === 'put' ? 'bg-rose-400/70' : 'bg-gold-400/80'
                        )}
                        style={{ width: `${(s.oi / maxOi) * 100}%` }}
                        title={`${s.strike} · ${dominant} ${s[dominant === 'put' ? 'putOi' : 'callOi'].toLocaleString()} OI`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-2 font-mono text-[8px] leading-relaxed text-zinc-400">
              gold = calls · rose = puts · bars scaled to local OI
            </p>
          </div>

          {/* Price chart */}
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                {ticker} — price vs dealer levels
              </h3>
              <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
                OI as of last close · {convention} convention assumed
              </span>
            </div>
            <ChartFullscreen
              title="Price vs Dealer Levels"
              subtitle={`${ticker} · ${convention} convention assumed`}
            >
              <TrapPriceChart
                closes={closes}
                spot={trap.spot}
                callWall={trap.walls.call?.strike ?? null}
                putWall={trap.walls.put?.strike ?? null}
                gammaCallWall={activeView.activeGammaWalls?.call?.strike ?? null}
                gammaPutWall={activeView.activeGammaWalls?.put?.strike ?? null}
                flip={activeView.activeFlip}
                intradayBars={intradayBars}
                intradaySource={intradaySource}
              />
            </ChartFullscreen>
          </div>

          {/* Score + state + hedge flow */}
          <div className="flex w-full shrink-0 flex-col justify-between gap-3 xl:w-64">
            <div>
              <div className="flex items-center justify-between">
                <span className="kicker flex items-center gap-2">
                  <Magnet className="h-3.5 w-3.5" /> Trap Score
                </span>
                <span
                  className={cn(
                    'rounded-md px-2 py-0.5 font-mono text-[11px] font-bold tracking-widest',
                    STATE_STYLE[activeView.activeState]
                  )}
                >
                  {STATE_LABEL[activeView.activeState]}
                </span>
              </div>
              <div className="mt-1.5 flex items-end gap-3">
                <span className="font-mono text-5xl font-bold tabular-nums leading-none text-zinc-900 dark:text-zinc-50">
                  {activeView.activeScore}
                </span>
                <span className="pb-1 font-mono text-[10px] text-zinc-400">/ 100</span>
                {sparkPath && (
                  <svg viewBox="0 0 110 36" className="mb-1 h-9 w-24" aria-label="score history">
                    <path
                      d={sparkPath}
                      fill="none"
                      stroke="#DEC084"
                      strokeWidth="1.05"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  </svg>
                )}
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2 font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-400">
                <span>
                  {activeView.activeDirection === 'none'
                    ? 'no directional lean'
                    : `direction: ${activeView.activeDirection === 'up' ? '↑ up — call wall' : '↓ down — put wall'}`}
                  {' · '}regime:{' '}
                  <span
                    className={
                      activeView.activeRegime === 'short'
                        ? 'text-rose-500'
                        : 'text-emerald-600 dark:text-emerald-400'
                    }
                  >
                    {activeView.activeRegime} gamma
                  </span>
                </span>
                {hmmGamma && (
                  <RegimeChip fit={hmmGamma} labels={['Long γ', 'Short γ']} className="ml-auto" />
                )}
              </div>
              {hmmGamma && (
                <RegimeTransitionMatrix
                  fit={hmmGamma}
                  labels={['Long γ', 'Short γ']}
                  className="mt-3 border-t border-[var(--hairline)] pt-3"
                  showStats={false}
                />
              )}
            </div>

            {/* Hedge flow sentence */}
            <div className="rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2.5">
              <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
                Dealer hedging into ±2% band
              </div>
              <p className="mt-1 text-[11px] leading-snug text-zinc-600 dark:text-zinc-300">
                Dealers must {activeView.activeFlow.perPctUp >= 0 ? 'sell' : 'buy'}{' '}
                <strong className="text-zinc-900 dark:text-zinc-50">
                  {moneyAbs(activeView.activeFlow.perPctUp)}
                </strong>{' '}
                per 1% up move; {activeView.activeFlow.perPctDown >= 0 ? 'buy' : 'sell'}{' '}
                <strong className="text-zinc-900 dark:text-zinc-50">
                  {moneyAbs(activeView.activeFlow.perPctDown)}
                </strong>{' '}
                per 1% down —{' '}
                {activeView.activeFlow.net >= 0 ? 'net long-gamma re-hedge' : 'net short-gamma re-hedge'} (
                {fmtSignedUsd(activeView.activeFlow.net)} net).
              </p>
            </div>
          </div>
        </div>

        {/* Badge line — copy gated on active convention state */}
        <div className="mt-3 border-t border-[var(--hairline)] pt-3">
          <p className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
            {activeView.activeState === 'triggered' && activeView.activeDirection === 'up' && (
              <>⚡ Writers trapped at {trap.walls.call?.strike} — price is pressing through the wall.</>
            )}
            {activeView.activeState === 'triggered' && activeView.activeDirection === 'down' && (
              <>⚡ Writers trapped at {trap.walls.put?.strike} — price is pressing through the wall.</>
            )}
            {activeView.activeState === 'armed' && (
              <>
                Coiling{' '}
                {activeView.activeNearWall === 'call'
                  ? `${trap.walls.call?.strike} (call wall)`
                  : `${trap.walls.put?.strike} (put wall)`}{' '}
                — {trappedWall ? `${trappedWall.distSigma.toFixed(2)}σ away` : '—'}. Trade direction
                unconfirmed; wait for the close through the wall.
              </>
            )}
            {activeView.activeState === 'building' && trappedWall && (
              <>
                Writers piling up at {trappedWall.strike} — {pct(trappedWall.share, 0)} of local OI
                {trap.oiChange5d !== null
                  ? `, ΔOI5d ${trap.oiChange5d >= 0 ? '+' : ''}${trap.oiChange5d.toLocaleString()}`
                  : ''}
                .
              </>
            )}
            {activeView.activeState === 'exhausted' && (
              <>Wall absorbed — flow likely spent. Do not chase.</>
            )}
            {activeView.activeState === 'idle' && (
              <>No meaningful dealer pressure building under {convention} convention. Nothing to price yet.</>
            )}
          </p>
          <p className="mt-1 font-mono text-[9px] leading-relaxed text-zinc-400">
            Dealer positioning is assumed, not observed. Toggle Squeeze ↔ Inverse above to flip dealer call/put signs without an extra chain fetch.
            {!include0Dte && activeView.zeroBucketOi > 0 && ' Ex-0DTE subtracts 0-DTE contracts from net totals & strike profile; Gamma Flip & Trap Score reflect the full ≤45d surface.'}
          </p>
        </div>
      </div>

      {/* ══════ STRIKE PROFILE + SPOT-LADDER GEX(S) CONVEXITY CURVE ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="panel p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2">
                <Activity className="h-3.5 w-3.5" /> Strike Exposure Profile
              </p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Exposure by strike ({bucketFilterLabel})
              </h3>
            </div>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              {profileMetric === 'gex' && `Net OI GEX ${fmtSignedUsd(activeView.totalGex)}`}
              {profileMetric === 'volumeGex' && `Net Vol-GEX ${fmtSignedUsd(activeView.totalVolumeGex)}`}
              {profileMetric === 'charm' && `Net Charm ${fmtSignedUsd(activeView.totalCharm)}/d`}
              {profileMetric === 'vanna' && `Net Vanna ${fmtSignedUsd(activeView.totalVanna)}/+1σv`}
            </span>
          </div>
          <ChartFullscreen
            title={`Dealer Strike Profile — ${bucketFilterLabel}`}
            subtitle={`${ticker} · ${convention} convention`}
          >
            <GammaProfileChart
              points={activeView.activeByStrike}
              spot={trap.spot}
              flip={activeView.activeFlip}
              convention={convention}
              serverConvention={activeView.serverConv}
              metric={profileMetric}
              onMetricChange={setProfileMetric}
              bucketLabel={bucketFilterLabel}
              callWall={trap.walls.call?.strike ?? null}
              putWall={trap.walls.put?.strike ?? null}
              gammaCallWall={activeView.activeGammaWalls?.call?.strike ?? null}
              gammaPutWall={activeView.activeGammaWalls?.put?.strike ?? null}
            />
          </ChartFullscreen>
        </div>

        <div className="panel p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1">Spot-Ladder Convexity</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                GEX(S) across ±15% hypothetical spot
              </h3>
            </div>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              sticky-strike IV · F(S) scaled
            </span>
          </div>
          <ChartFullscreen
            title="Spot-Ladder GEX(S) Convexity Curve"
            subtitle={`${ticker} · ${convention} convention`}
          >
            <GexFlipCurveChart
              profile={trap.flipProfile}
              spot={trap.spot}
              flip={activeView.activeFlip}
              convention={convention}
              serverConvention={activeView.serverConv}
              expectedMove30={trap.expectedMove30}
              flipDistPct={activeView.flipDistPct}
              flipDistSigma={activeView.flipDistSigma}
            />
          </ChartFullscreen>
        </div>
      </div>

      {/* ══════ EXPIRY-BUCKETED GEX & STRIKE × EXPIRY MATRIX ══════ */}
      <div className="panel p-5">
        <GexExpiryHeatmap
          expiryBuckets={trap.expiryBuckets}
          matrix={trap.strikeExpiryMatrix}
          spot={trap.spot}
          convention={convention}
          serverConvention={activeView.serverConv}
          include0Dte={include0Dte}
          selectedBucket={bucketFilter}
          onSelectBucket={setBucketFilter}
        />
      </div>

      {/* ══════ WALL LEDGER (OI + GAMMA) + SCORE BREAKDOWN + EVENT STUDY ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        {/* Wall ledger: both OI Walls & Gamma Walls */}
        <div className="panel p-5 xl:col-span-5">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Wall Ledger</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Open Interest Walls &amp; Gamma-Weighted Walls
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[460px] text-xs">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <th className="py-1.5 pr-2">OI Wall</th>
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
                  {
                    w: trap.walls.call,
                    runnerUp: trap.walls.runnerUpCall,
                    label: 'CALL',
                    accent: 'text-gold-700 dark:text-gold-300',
                  },
                  {
                    w: trap.walls.put,
                    runnerUp: trap.walls.runnerUpPut,
                    label: 'PUT',
                    accent: 'text-rose-500 dark:text-rose-400',
                  },
                ].map(({ w, runnerUp, label, accent }) => (
                  <React.Fragment key={label}>
                    <tr className="border-t border-[var(--hairline)]">
                      <td className={cn('py-1.5 pr-2 font-bold', accent)}>{label}</td>
                      <td className="py-1.5 pr-2 tabular-nums text-zinc-900 dark:text-zinc-50">
                        {w?.strike ?? '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {w?.oi.toLocaleString() ?? '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                        {w &&
                        activeView.activeNearWall === (label === 'CALL' ? 'call' : 'put') &&
                        trap.oiChange1d !== null
                          ? `${trap.oiChange1d >= 0 ? '+' : ''}${trap.oiChange1d.toLocaleString()}`
                          : '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                        {w &&
                        activeView.activeNearWall === (label === 'CALL' ? 'call' : 'put') &&
                        trap.oiChange5d !== null
                          ? `${trap.oiChange5d >= 0 ? '+' : ''}${trap.oiChange5d.toLocaleString()}`
                          : '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {w ? pct(w.share, 1) : '—'}
                      </td>
                      <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {w?.distSigma.toFixed(2) ?? '—'}
                      </td>
                    </tr>
                    {runnerUp && (
                      <tr className="border-t border-[var(--hairline)] opacity-55">
                        <td className="py-1.5 pr-2 font-mono text-[9px] uppercase text-zinc-400">
                          {label} · 2nd
                        </td>
                        <td className="py-1.5 pr-2 tabular-nums text-zinc-500">{runnerUp.strike}</td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                          {runnerUp.oi.toLocaleString()}
                        </td>
                        <td className="py-1.5 pr-2 text-right text-zinc-400">—</td>
                        <td className="py-1.5 pr-2 text-right text-zinc-400">—</td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                          {pct(runnerUp.share, 1)}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-500">
                          {runnerUp.distSigma.toFixed(2)}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

          {activeView.activeGammaWalls && (
            <div className="mt-4 border-t border-[var(--hairline)] pt-3 overflow-x-auto">
              <table className="w-full min-w-[460px] text-xs">
                <thead>
                  <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    <th className="py-1.5 pr-2">γ Wall</th>
                    <th className="py-1.5 pr-2">Strike</th>
                    <th className="py-1.5 pr-2 text-right">Signed GEX</th>
                    <th className="py-1.5 pr-2 text-right">OI</th>
                    <th className="py-1.5 pr-2 text-right">Gross %</th>
                    <th className="py-1.5 text-right">σ away</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {[
                    {
                      w: activeView.activeGammaWalls.call,
                      runnerUp: activeView.activeGammaWalls.runnerUpCall,
                      label: 'γ CALL',
                      accent: 'text-[var(--positive)]',
                    },
                    {
                      w: activeView.activeGammaWalls.put,
                      runnerUp: activeView.activeGammaWalls.runnerUpPut,
                      label: 'γ PUT',
                      accent: 'text-[var(--negative)]',
                    },
                  ].map(({ w, runnerUp, label, accent }) => (
                    <React.Fragment key={label}>
                      <tr className="border-t border-[var(--hairline)]">
                        <td className={cn('py-1.5 pr-2 font-bold', accent)}>{label}</td>
                        <td className="py-1.5 pr-2 tabular-nums text-zinc-900 dark:text-zinc-50">
                          {w?.strike ?? '—'}
                        </td>
                        <td
                          className={cn(
                            'py-1.5 pr-2 text-right tabular-nums font-semibold',
                            (w?.gex ?? 0) >= 0 ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
                          )}
                        >
                          {w ? fmtSignedUsd(w.gex) : '—'}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {w?.oi.toLocaleString() ?? '—'}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {w ? pct(w.share, 1) : '—'}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {w?.distSigma.toFixed(2) ?? '—'}
                        </td>
                      </tr>
                      {runnerUp && (
                        <tr className="border-t border-[var(--hairline)] opacity-55">
                          <td className="py-1.5 pr-2 font-mono text-[9px] uppercase text-zinc-400">
                            {label} · 2nd
                          </td>
                          <td className="py-1.5 pr-2 tabular-nums text-zinc-500">{runnerUp.strike}</td>
                          <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                            {fmtSignedUsd(runnerUp.gex)}
                          </td>
                          <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                            {runnerUp.oi.toLocaleString()}
                          </td>
                          <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-500">
                            {pct(runnerUp.share, 1)}
                          </td>
                          <td className="py-1.5 text-right tabular-nums text-zinc-500">
                            {runnerUp.distSigma.toFixed(2)}
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="mt-2 font-mono text-[9px] text-zinc-400">
            OI share = local OI within ±10% of spot (≤ 45d) · Gamma walls rank highest |GEX| across full ≤ 45d chain
          </p>
        </div>

        {/* Component breakdown */}
        <div className="panel p-5 xl:col-span-3">
          <div className="mb-3 border-b border-[var(--hairline)] pb-3">
            <p className="kicker mb-1">Why this score</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Component breakdown
            </h3>
          </div>
          <div className="space-y-2.5">
            {componentDefs.map(({ key, label, max }) => {
              const v = activeView.activeComponents[key];
              const width = Math.min(100, (v / max) * 100);
              const degraded = key === 'velocity' && trap.velocityDegraded;
              return (
                <div key={key}>
                  <div className="flex items-baseline justify-between font-mono text-[10px]">
                    <span className="flex items-center gap-1.5 text-zinc-500 dark:text-zinc-400">
                      {label}
                      {degraded && (
                        <span className="rounded border border-gold-500/40 bg-gold-500/10 px-1 text-[8px] text-gold-800 dark:text-gold-300">
                          ΔOI accruing
                        </span>
                      )}
                    </span>
                    <span className="tabular-nums text-zinc-700 dark:text-zinc-200">
                      {v.toFixed(1)}
                      <span className="text-zinc-400"> / {max}</span>
                    </span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-zinc-100 dark:bg-white/[0.05]">
                    <div
                      className={cn(
                        'h-full rounded-full',
                        degraded
                          ? 'bg-zinc-400/40'
                          : 'bg-gradient-to-r from-gold-700 via-gold-500 to-gold-300'
                      )}
                      style={{ width: `${width}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-3 font-mono text-[9px] leading-relaxed text-zinc-400">
            velocity: ΔOI at wall vs trailing 5d baseline
            {trap.oiChange5dZ !== null
              ? `, z = ${trap.oiChange5dZ.toFixed(2)}`
              : ' — needs ≥ 20 accrued sessions'}
            {' · '}coil: |5d| range vs RV20·√5, damped by RV−IV spread
          </div>
        </div>

        {/* Backtest panel */}
        <div className="panel p-5 xl:col-span-4">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2">
                <ShieldAlert className="h-3.5 w-3.5" /> The Verdict
              </p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Event study of past triggers
              </h3>
            </div>
            <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-400">
              {backtest.nDedupedEvents} deduped events
            </span>
          </div>

          <div
            className={cn(
              'rounded-lg border px-3 py-2 text-[11px] font-semibold leading-snug',
              backtest.verdict === 'above_base_rate' &&
                'border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
              backtest.verdict === 'no_reliable_follow_through' &&
                'border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400',
              backtest.verdict === 'insufficient_history' &&
                'border-zinc-300 dark:border-white/[0.08] bg-[var(--bg-inset)] text-zinc-500 dark:text-zinc-400'
            )}
          >
            {backtest.verdictLine}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2">
            {[
              {
                label: 'Hit rate (5d)',
                value: backtest.hitRate5d !== null ? `${backtest.hitRate5d.toFixed(0)}%` : '—',
              },
              {
                label: 'Base rate',
                value: backtest.baseRate5d !== null ? `${backtest.baseRate5d.toFixed(0)}%` : '—',
              },
              { label: 'Mean 5d (dir)', value: signPct(backtest.fwdMean.h5) },
              {
                label: 'Newey–West t',
                value: backtest.tStat5d !== null ? backtest.tStat5d.toFixed(2) : '—',
              },
            ].map((s) => (
              <div key={s.label} className="panel-inset p-2.5">
                <div className="font-mono text-[8.5px] uppercase tracking-wider text-zinc-400">
                  {s.label}
                </div>
                <div className="mt-0.5 font-mono text-base font-bold tabular-nums text-zinc-900 dark:text-zinc-50">
                  {s.value}
                </div>
              </div>
            ))}
          </div>

          {backtest.wilsonLo !== null && backtest.wilsonHi !== null && (
            <p className="mt-2 font-mono text-[9px] text-zinc-400">
              Wilson 95% CI: [{backtest.wilsonLo.toFixed(0)}%, {backtest.wilsonHi.toFixed(0)}%] · raw{' '}
              {backtest.nRawTriggered} · MFE/MAE {backtest.medianMfeMae?.toFixed(2) ?? '—'}
            </p>
          )}

          {backtest.decileTable.length > 0 ? (
            <div className="mt-3 overflow-x-auto">
              <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                Score decile → mean 5d fwd
              </div>
              <div className="mt-1 flex gap-1">
                {backtest.decileTable.map((d) => (
                  <div
                    key={d.decile}
                    className="flex-1 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] p-1 text-center font-mono text-[8.5px]"
                    title={`decile ${d.decile}: score≈${d.meanScore}, n=${d.n}`}
                  >
                    <div className="text-zinc-400">D{d.decile}</div>
                    <div
                      className={cn(
                        'font-bold tabular-nums',
                        (d.mean5d ?? 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500'
                      )}
                    >
                      {d.mean5d !== null ? `${d.mean5d >= 0 ? '+' : ''}${d.mean5d.toFixed(1)}%` : '—'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="mt-2 font-mono text-[9px] text-zinc-400">
              decile monotonicity unlocks at ≥ 10 triggered events ({backtest.nDedupedEvents} so far)
            </p>
          )}
        </div>
      </div>

      {/* ══════ ACCRUED GEX & FLIP HISTORY CHART ══════ */}
      <div className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
          <div>
            <p className="kicker mb-1">Accrued Positioning Series</p>
            <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
              Historical Net GEX &amp; Gamma Flip vs. Spot ({panel.length} sessions stored)
            </h3>
          </div>
          {snapshot.session && (
            <span className="font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
              Last 1d Δ: GEX {fmtSignedUsd(gexDayDelta)} · Flip{' '}
              {snapshot.session.gammaFlipChange !== null
                ? `${snapshot.session.gammaFlipChange >= 0 ? '+' : ''}$${snapshot.session.gammaFlipChange.toFixed(2)}`
                : '—'}
            </span>
          )}
        </div>
        <ChartFullscreen
          title="Accrued Net GEX & Gamma Flip History"
          subtitle={`${ticker} · ${panel.length} accrued EOD sessions`}
        >
          <GexHistoryChart panel={panel} convention={convention} />
        </ChartFullscreen>
      </div>

      {/* ══════ STATE HISTORY + WATCHLIST GEX RADAR ══════ */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* State History table with Total GEX & Gamma Flip columns */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1">State History</p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Every daily verdict, newest first
              </h3>
            </div>
            <button
              type="button"
              onClick={exportCsv}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-[var(--hairline)] px-2 py-1.5 font-mono text-[10px] font-semibold text-zinc-500 transition-colors hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
            >
              <Download className="h-3.5 w-3.5" /> CSV
            </button>
          </div>
          {historyRows.length === 0 ? (
            <div className="text-xs text-zinc-400">
              History starts with the first snapshot — one row per day accrues here.
            </div>
          ) : (
            <div className="max-h-96 overflow-auto">
              <table className="w-full min-w-[620px] text-[10.5px]">
                <thead className="sticky top-0 bg-[var(--bg-panel)]">
                  <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    <th className="py-1.5 pr-2">Date</th>
                    <th className="py-1.5 pr-2">State</th>
                    <th className="py-1.5 pr-2 text-right">Spot</th>
                    <th className="py-1.5 pr-2 text-right">Total GEX</th>
                    <th className="py-1.5 pr-2 text-right">Gamma Flip</th>
                    <th className="py-1.5 pr-2 text-right">Score</th>
                    <th className="py-1.5 pr-2 text-right">+1d</th>
                    <th className="py-1.5 pr-2 text-right">+5d</th>
                    <th className="py-1.5 text-right">+10d</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {historyRows.map(({ row, adjustedGex, fwd1d, fwd5d, fwd10d }) => (
                    <tr key={row.date} className="border-t border-[var(--hairline)]">
                      <td className="py-1.5 pr-2 text-zinc-500">{row.date.slice(5)}</td>
                      <td className="py-1.5 pr-2">
                        <span
                          className={cn(
                            'rounded px-1.5 py-0.5 text-[8px] font-bold tracking-wider',
                            STATE_STYLE[row.trapState ?? 'idle']
                          )}
                        >
                          {row.trapState ? STATE_LABEL[row.trapState] : '—'}
                        </span>
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {row.spot.toFixed(2)}
                      </td>
                      <td
                        className={cn(
                          'py-1.5 pr-2 text-right tabular-nums',
                          adjustedGex === null
                            ? 'text-zinc-400'
                            : adjustedGex >= 0
                              ? 'text-[var(--positive)]'
                              : 'text-[var(--negative)]'
                        )}
                      >
                        {fmtSignedUsd(adjustedGex)}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {row.gammaFlip !== null && isFinite(row.gammaFlip)
                          ? `$${row.gammaFlip.toFixed(2)}`
                          : '—'}
                      </td>
                      <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                        {row.trapScore ?? '—'}
                      </td>
                      {[fwd1d, fwd5d, fwd10d].map((v, i) => (
                        <td
                          key={i}
                          className={cn(
                            'py-1.5 pr-2 text-right tabular-nums',
                            v === null
                              ? 'text-zinc-300 dark:text-zinc-700'
                              : v >= 0
                                ? 'text-emerald-600 dark:text-emerald-400'
                                : 'text-rose-500 dark:text-rose-400'
                          )}
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

        {/* Watchlist GEX Radar (shared via TickerSnapshotProvider) */}
        <div className="panel p-5">
          <div className="mb-3 flex items-center justify-between gap-2 border-b border-[var(--hairline)] pb-3">
            <div>
              <p className="kicker mb-1 flex items-center gap-2">
                <ScanLine className="h-3.5 w-3.5" /> Watchlist GEX Radar
              </p>
              <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Cross-ticker positioning &amp; trap pressure ({convention})
              </h3>
            </div>
            <button
              type="button"
              onClick={runScan}
              disabled={scanLoading}
              className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-gold-600/40 bg-gold-500/10 px-3 font-mono text-[10px] font-semibold text-gold-800 transition-colors hover:border-gold-500/70 dark:text-gold-200 disabled:opacity-60"
            >
              {scanLoading ? 'Scanning…' : sortedScan ? 'Rescan' : 'Scan watchlist'}
            </button>
          </div>
          {!sortedScan ? (
            <div className="text-[11px] text-zinc-400">
              Six names: AAPL · MSFT · GOOGL · NVDA · TSLA · SPY — shared snapshot scan, cached 5 min client / 15 min server.
            </div>
          ) : (
            <div className="max-h-96 overflow-auto">
              <table className="w-full min-w-[660px] text-[10.5px]">
                <thead className="sticky top-0 bg-[var(--bg-panel)]">
                  <tr className="text-left font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    <th className="py-1.5 pr-2">Symbol</th>
                    <th className="py-1.5 pr-2">State</th>
                    <th
                      className="cursor-pointer py-1.5 pr-2 text-right hover:text-zinc-700 dark:hover:text-zinc-100"
                      onClick={() => handleSortClick('score')}
                    >
                      Score {sortKey === 'score' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                    <th
                      className="cursor-pointer py-1.5 pr-2 text-right hover:text-zinc-700 dark:hover:text-zinc-100"
                      onClick={() => handleSortClick('totalGex')}
                    >
                      Net GEX {sortKey === 'totalGex' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                    <th className="py-1.5 pr-2 text-right">GEX/ADV</th>
                    <th className="py-1.5 pr-2 text-right">Regime</th>
                    <th
                      className="cursor-pointer py-1.5 pr-2 text-right hover:text-zinc-700 dark:hover:text-zinc-100"
                      onClick={() => handleSortClick('flipDistPct')}
                    >
                      Flip % {sortKey === 'flipDistPct' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                    <th className="py-1.5 pr-2 text-right">CW / PW</th>
                    <th
                      className="cursor-pointer py-1.5 pr-2 text-right hover:text-zinc-700 dark:hover:text-zinc-100"
                      onClick={() => handleSortClick('expiryConcentration7d')}
                    >
                      Exp 7d {sortKey === 'expiryConcentration7d' ? (sortDir === -1 ? '↓' : '↑') : ''}
                    </th>
                    <th className="py-1.5 text-right">0-DTE</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {sortedScan.map((r) => {
                    if (!r.available) {
                      return (
                        <tr key={r.symbol} className="border-t border-[var(--hairline)] opacity-60">
                          <td className="py-1.5 pr-2 font-bold text-zinc-500">{r.symbol}</td>
                          <td colSpan={9} className="py-1.5 text-[9px] text-zinc-400">
                            unavailable
                          </td>
                        </tr>
                      );
                    }
                    const state = (r.trapState ?? 'idle') as TrapState;
                    const mult = convention === 'inverse' ? -1 : 1;
                    const netGex =
                      r.totalGex !== null && r.totalGex !== undefined ? r.totalGex * mult : null;
                    const gexAdv =
                      r.gexPerAdv !== null && r.gexPerAdv !== undefined ? r.gexPerAdv * mult : null;
                    const regime =
                      r.regime === 'long'
                        ? convention === 'inverse'
                          ? 'short'
                          : 'long'
                        : r.regime === 'short'
                          ? convention === 'inverse'
                            ? 'long'
                            : 'short'
                          : r.regime ?? null;
                    const cw = readWallStrike(r.callWall);
                    const pw = readWallStrike(r.putWall);

                    return (
                      <tr key={r.symbol} className="border-t border-[var(--hairline)]">
                        <td className="py-1.5 pr-2 font-bold text-zinc-800 dark:text-zinc-100">
                          {r.symbol}
                        </td>
                        <td className="py-1.5 pr-2">
                          <span
                            className={cn(
                              'rounded px-1 py-0.5 text-[8px] font-bold tracking-wider',
                              STATE_STYLE[state] ?? STATE_STYLE.idle
                            )}
                          >
                            {STATE_LABEL[state] ?? 'IDLE'}
                          </span>
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-800 dark:text-zinc-100">
                          {r.trapScore ?? '—'}
                        </td>
                        <td
                          className={cn(
                            'py-1.5 pr-2 text-right tabular-nums',
                            netGex === null
                              ? 'text-zinc-400'
                              : netGex >= 0
                                ? 'text-[var(--positive)]'
                                : 'text-[var(--negative)]'
                          )}
                        >
                          {fmtSignedUsd(netGex)}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-600 dark:text-zinc-300">
                          {gexAdv !== null ? signPct(gexAdv, 1) : '—'}
                        </td>
                        <td
                          className={cn(
                            'py-1.5 pr-2 text-right font-semibold',
                            regime === 'long'
                              ? 'text-[var(--positive)]'
                              : regime === 'short'
                                ? 'text-[var(--negative)]'
                                : 'text-zinc-400'
                          )}
                        >
                          {regime === 'long' ? 'Long γ' : regime === 'short' ? 'Short γ' : '—'}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {signPct(r.flipDistPct, 1)}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {cw !== null || pw !== null ? `${cw ?? '—'} / ${pw ?? '—'}` : '—'}
                        </td>
                        <td className="py-1.5 pr-2 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {pct(r.expiryConcentration7d, 0)}
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-zinc-700 dark:text-zinc-200">
                          {pct(r.zeroDteShare, 0)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ══════ COLLAPSIBLE METHODOLOGY DRAWER ══════ */}
      <details className="panel group">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-3.5 select-none">
          <div>
            <span className="kicker">Methodology &amp; Formulas</span>
            <span className="ml-3 font-display text-xs font-semibold text-zinc-800 dark:text-zinc-200">
              Black–Scholes Gamma, GEX, Zero-Gamma Flip, Wall Concentration &amp; Hedge Flow
            </span>
          </div>
          <ChevronDown className="h-4 w-4 text-zinc-400 transition-transform group-open:rotate-180" />
        </summary>
        <div className="grid grid-cols-1 gap-3 border-t border-[var(--hairline)] p-5 md:grid-cols-2">
          {METHODOLOGY_PRESETS.map((id) => {
            const meta = FINANCIAL_FORMULA_PRESETS[id];
            return (
              <div key={id} className="panel-inset p-3.5">
                <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-gold-700 dark:text-gold-300">
                  {meta.label}
                </div>
                <MathDisplay preset={id} display="block" showBadge />
                <p className="text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                  {meta.description}
                </p>
              </div>
            );
          })}
        </div>
      </details>

      {/* ══════ HONESTY FOOTNOTES ══════ */}
      <div className="flex items-start gap-2 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-3 py-2 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Dealer positioning is assumed, not observed — active assumption: {convention} convention (flip sign and the story flips with it). OI updates once daily at close while Volume-GEX reflects cumulative intraday prints; multi-leg spreads and collars net inside aggregate OI. Quotes are ~15-min delayed European BSM approximations using sanitized mid-quote IVs (0-DTE T floored at 0.5 sessions). 7d-expiration OI concentration is {pct(trap.expiryConcentration7d, 0)} and 0-DTE gross GEX share is {pct(trap.zeroDteShare ?? 0, 1)}; if short-dated flow dominates, exposure decays same-day. The event-study verdict panel is the contract: directional arrows remain suppressed until follow-through clears base rate.
        </span>
      </div>
    </div>
  );
}
