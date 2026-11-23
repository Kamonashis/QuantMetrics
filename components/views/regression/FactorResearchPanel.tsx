'use client';

import React, { useMemo, useState } from 'react';
import type { Data, Layout } from 'plotly.js';
import { Activity, AlertTriangle, Info, RefreshCw } from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import { calculateReturns } from '@/lib/quant-engine';
import {
  ALL_FACTORS,
  FACTOR_FAMILIES,
  FACTOR_BY_ID,
  buildScoredFactorPanel,
  computeFactorIc,
  compositeEquityCurve,
  type FactorDefinition,
} from '@/lib/factors';
import { usePlotlyDarkTheme, PlotlyChart } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { TickerSuggest } from '@/components/ui/TickerSuggest';
import type { FactorDataState } from '@/hooks/use-factor-data';

interface FactorResearchPanelProps {
  symbols: string[];
  allStockBars: Record<string, PriceBar[]>;
  factorData: FactorDataState;
  onSymbolsChange: (symbols: string[]) => void;
  startDate: string;
  endDate: string;
  barsSource?: string | null;
}

const FAMILY_LABELS: Record<string, string> = {
  momentum: 'Momentum',
  reversal: 'Reversal',
  low_volatility: 'Low Volatility',
  low_beta: 'Low Beta',
  downside_risk: 'Downside Risk',
  liquidity: 'Liquidity',
  volatility_structure: 'Volatility Structure',
  value: 'Value',
  quality: 'Quality',
  investment: 'Investment',
  size: 'Size',
  analyst: 'Analyst / Earnings',
  options: 'Options-Implied',
};

function rv20FromBars(bars: PriceBar[]): number | null {
  const closes = bars.map((b) => (b as PriceBar & { adjClose?: number }).adjClose ?? b.close);
  const returns = calculateReturns(closes, 'log').slice(-20).filter((v) => Number.isFinite(v));
  if (returns.length < 10) return null;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (returns.length - 1);
  return Math.sqrt(Math.max(0, variance) * 252);
}

export function FactorResearchPanel({
  symbols,
  allStockBars,
  factorData,
  onSymbolsChange,
  startDate,
  endDate,
  barsSource,
}: FactorResearchPanelProps) {
  const [selectedIds, setSelectedIds] = useState<string[]>([
    'price:MOM_12_1',
    'price:RV60',
    'price:BAB_BETA',
    'price:AMIHUD',
    'price:GK_VOL',
  ]);
  const [horizon, setHorizon] = useState<1 | 5 | 22>(22);
  const [sectorNeutral, setSectorNeutral] = useState(true);
  const [winsorPct, setWinsorPct] = useState(0.01);
  const [weightMode, setWeightMode] = useState<'equal' | 'ic'>('equal');
  const [maxPoints, setMaxPoints] = useState(504);
  const [tickerInput, setTickerInput] = useState('');
  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  const definitions = useMemo<FactorDefinition[]>(
    () => selectedIds.map((id) => FACTOR_BY_ID[id]).filter((d): d is FactorDefinition => Boolean(d)),
    [selectedIds]
  );

  const { benchmarkBars, dates, enrichedOptions } = useMemo(() => {
    const bench = allStockBars['SPY'] ?? allStockBars[symbols[0]] ?? [];
    const filtered = bench.filter((b) => (!startDate || b.date >= startDate) && (!endDate || b.date <= endDate));
    const options: Record<string, (typeof factorData.optionsBySymbol)[string]> = {};
    for (const symbol of symbols) {
      const snaps = factorData.optionsBySymbol[symbol] ?? [];
      options[symbol] = snaps.map((snap) => ({
        ...snap,
        rv20: snap.rv20 ?? rv20FromBars(allStockBars[symbol] ?? []),
      }));
    }
    return { benchmarkBars: bench, dates: filtered.map((b) => b.date), enrichedOptions: options };
  }, [allStockBars, symbols, startDate, endDate, factorData]);

  const scored = useMemo(
    () =>
      buildScoredFactorPanel(
        {
          symbols,
          barsBySymbol: allStockBars,
          benchmarkBars,
          dates,
          horizon,
          maxPoints,
          sectors: factorData.sectors,
          fundamentalsBySymbol: factorData.fundamentalsBySymbol,
          analystBySymbol: factorData.analystBySymbol,
          optionsBySymbol: enrichedOptions,
        },
        definitions,
        { sectorNeutral, lower: winsorPct, upper: 1 - winsorPct }
      ),
    [
      symbols, allStockBars, benchmarkBars, dates, horizon, maxPoints,
      factorData.sectors, factorData.fundamentalsBySymbol, factorData.analystBySymbol,
      enrichedOptions, definitions, sectorNeutral, winsorPct,
    ]
  );

  const icResults = useMemo(() => {
    const base = computeFactorIc(scored, definitions);
    if (weightMode === 'equal') return base;
    const map = new Map(base.map((r) => [r.id, r]));
    const weights: Record<string, number> = {};
    let total = 0;
    for (const def of definitions) {
      const row = map.get(def.id);
      const ic = row?.meanIc;
      if (ic !== null && ic !== undefined && Number.isFinite(ic)) {
        weights[def.id] = Math.abs(ic);
        total += Math.abs(ic);
      }
    }
    if (total <= 0) return base;
    const composite = map.get('composite');
    if (composite) {
      composite.meanIc = definitions.reduce(
        (sum, def) => sum + ((map.get(def.id)?.meanIc ?? 0) * (weights[def.id] ?? 0)) / total,
        0
      );
    }
    return base;
  }, [scored, definitions, weightMode]);

  const equity = useMemo(() => compositeEquityCurve(scored), [scored]);
  const composite = icResults.find((r) => r.id === 'composite');
  const coverageIssues = symbols.filter((s) => factorData.coverage[s]?.degraded !== false);

  const equityTraces = useMemo<Data[]>(
    () => [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Top decile − Bottom decile',
        x: equity.map((p) => p.date),
        y: equity.map((p) => p.value),
        line: { color: QUANT_COLORS.gold, width: 1.8 },
        hovertemplate: '%{x} · %{y:.3f}<extra>L/S</extra>',
      },
    ],
    [equity]
  );
  const equityLayout = useMemo<Partial<Layout>>(
    () => ({
      yaxis: { title: { text: 'Growth of 1', font: { size: 10, color: tokens.textMuted } }, gridcolor: tokens.grid },
      margin: { l: 48, r: 12, t: 8, b: 32 },
    }),
    [tokens]
  );

  const toggleFactor = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const addSymbol = () => {
    const clean = tickerInput.trim().toUpperCase().split(',')[0];
    if (clean && !symbols.includes(clean) && symbols.length < 12) {
      onSymbolsChange([...symbols, clean]);
    }
    setTickerInput('');
  };

  const latestRanking = useMemo(() => {
    const lastDate = scored.length > 0 ? scored[scored.length - 1].date : null;
    return scored.filter((r) => r.date === lastDate).sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999));
  }, [scored]);

  return (
    <div className="panel p-5 space-y-5">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-3 pb-3 border-b border-[var(--hairline)]">
        <div>
          <p className="kicker mb-1 flex items-center gap-2"><Activity className="h-3.5 w-3.5" /> Cross-Sectional Factor Research</p>
          <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
            Rank, standardize and combine factors across the active universe
          </h3>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
            {symbols.length} symbols · {dates.length} sessions · {horizon}d forward returns · {definitions.length} factors
            {barsSource && barsSource !== 'live_market' ? ' · synthetic price inputs' : ''}
          </p>
        </div>
        <button
          type="button"
          onClick={factorData.refresh}
          disabled={factorData.isLoading}
          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-gold-500/40 bg-gold-500/10 text-[10px] font-mono text-gold-700 dark:text-gold-300 hover:bg-gold-500/20 disabled:opacity-50"
        >
          <RefreshCw className={`h-3 w-3 ${factorData.isLoading ? 'animate-spin' : ''}`} />
          Refresh factor inputs
        </button>
      </div>

      {/* Honesty banners */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] text-zinc-600 dark:text-zinc-400">
        <div className="panel-inset p-3 flex items-start gap-2">
          <AlertTriangle className="h-3.5 w-3.5 text-gold-500 shrink-0 mt-0.5" />
          <span>
            <strong className="text-zinc-800 dark:text-zinc-200">No point-in-time filings.</strong> Yahoo exposes fiscal period
            ends, not filing dates; fundamentals apply a 90d quarterly / 180d annual lag and are labeled by first observation.
            Analyst and option inputs are current snapshots, not history.
          </span>
        </div>
        <div className="panel-inset p-3 flex items-start gap-2">
          <Info className="h-3.5 w-3.5 text-gold-500 shrink-0 mt-0.5" />
          <span>
            <strong className="text-zinc-800 dark:text-zinc-200">Static universe.</strong> Delisted names are absent (survivorship
            bias), and adjusted-close histories drift when new dividends are paid. Treat IC and long-short spreads as
            exploratory, not investable backtest results.
          </span>
        </div>
      </div>

      {coverageIssues.length > 0 && (
        <div className="text-[11px] text-amber-700 dark:text-amber-400">
          {coverageIssues.length} of {symbols.length} symbols have &lt;20 accrued fundamental sessions:{' '}
          {coverageIssues.join(', ')}. Fundamental-family factors are unavailable for those names.
        </div>
      )}

      {/* Controls */}
      <div className="panel-inset p-3.5 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="kicker">Universe</span>
          {symbols.map((s) => (
            <span key={s} className="px-2 py-0.5 rounded font-mono text-[10px] bg-gold-500/10 text-gold-700 dark:text-gold-300 border border-gold-500/30">
              {s}
            </span>
          ))}
          <div className="flex items-center gap-1">
            <TickerSuggest
              ariaLabel="Add factor universe ticker"
              id="factor-universe-input"
              value={tickerInput}
              onChange={setTickerInput}
              placeholder="Add ticker"
              inputClassName="w-28 px-2 py-1 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono text-[10px]"
            />
            <button type="button" onClick={addSymbol} className="px-2 py-1 rounded-lg border border-gold-500/40 text-[10px] font-mono text-gold-700 dark:text-gold-300">
              Add
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-3">
          {FACTOR_FAMILIES.map((family) => {
            const inFamily = ALL_FACTORS.filter((f) => f.family === family);
            if (inFamily.length === 0) return null;
            return (
              <div key={family} className="min-w-[140px]">
                <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-1">
                  {FAMILY_LABELS[family]}
                </div>
                <div className="flex flex-wrap gap-1">
                  {inFamily.map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      title={`${f.formula} — ${f.rationale}`}
                      onClick={() => toggleFactor(f.id)}
                      className={`px-1.5 py-0.5 rounded font-mono text-[9px] border transition-colors ${
                        selectedIds.includes(f.id)
                          ? 'bg-gold-500/15 border-gold-500/50 text-gold-700 dark:text-gold-300'
                          : 'border-transparent bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-4 text-[11px] text-zinc-600 dark:text-zinc-400">
          <label className="flex items-center gap-1.5">
            <span>Horizon</span>
            <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value) as 1 | 5 | 22)} className="px-1.5 py-0.5 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono">
              <option value={1}>1d</option><option value={5}>5d</option><option value={22}>22d</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={sectorNeutral} onChange={(e) => setSectorNeutral(e.target.checked)} />
            <span>Sector-neutral z-scores</span>
          </label>
          <label className="flex items-center gap-1.5">
            <span>Winsorize</span>
            <select value={winsorPct} onChange={(e) => setWinsorPct(Number(e.target.value))} className="px-1.5 py-0.5 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono">
              <option value={0}>off</option><option value={0.01}>1/99%</option><option value={0.025}>2.5/97.5%</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            <span>Weights</span>
            <select value={weightMode} onChange={(e) => setWeightMode(e.target.value as 'equal' | 'ic')} className="px-1.5 py-0.5 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono">
              <option value="equal">Equal</option><option value="ic">IC-weighted</option>
            </select>
          </label>
          <label className="flex items-center gap-1.5">
            <span>Scoring dates</span>
            <select value={maxPoints} onChange={(e) => setMaxPoints(Number(e.target.value))} className="px-1.5 py-0.5 rounded border border-[var(--hairline)] bg-[var(--bg-inset)] font-mono">
              <option value={126}>126</option><option value={252}>252</option><option value={504}>504</option><option value={756}>756</option>
            </select>
          </label>
        </div>
      </div>

      {scored.length === 0 ? (
        <div className="panel-inset p-6 text-center text-xs text-zinc-500">
          Insufficient aligned history. Add tickers with at least 252 sessions of bars to score the cross-section.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="panel-inset p-3">
              <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">Scored rows</div>
              <div className="font-mono text-lg font-bold text-zinc-900 dark:text-zinc-100">{scored.length}</div>
            </div>
            <div className="panel-inset p-3">
              <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">Composite IC</div>
              <div className="font-mono text-lg font-bold text-gold-700 dark:text-gold-300">
                {composite?.meanIc !== null && composite?.meanIc !== undefined ? composite.meanIc.toFixed(3) : '—'}
              </div>
            </div>
            <div className="panel-inset p-3">
              <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">IC t-stat</div>
              <div className="font-mono text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {composite?.tStat !== null && composite?.tStat !== undefined ? composite.tStat.toFixed(2) : '—'}
              </div>
            </div>
            <div className="panel-inset p-3">
              <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">Top − Bottom</div>
              <div className="font-mono text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {composite?.topMinusBottom !== null && composite?.topMinusBottom !== undefined
                  ? `${(composite.topMinusBottom * 100).toFixed(2)}%`
                  : '—'}
              </div>
            </div>
          </div>

          <ChartFullscreen title="Factor Composite Long-Short" subtitle={`${horizon}d forward · top decile minus bottom decile`}>
            <PlotlyChart
              data={equityTraces}
              layout={equityLayout}
              plotType="2d"
              size="compact"
              title="Factor Composite Long-Short"
              hovermode="x unified"
              exportFilename="quantmetrics-factor-composite"
              uirevisionKey={`factor-${definitions.length}-${horizon}-${maxPoints}`}
            />
          </ChartFullscreen>

          <div className="panel-inset overflow-x-auto">
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-2">Information Coefficient by Factor</div>
            <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-800 text-[11px] font-mono">
              <thead>
                <tr className="text-zinc-500">
                  <th className="py-1.5 px-2 text-left">Factor</th>
                  <th className="py-1.5 px-2 text-right">Obs</th>
                  <th className="py-1.5 px-2 text-right">Mean IC</th>
                  <th className="py-1.5 px-2 text-right">IC std</th>
                  <th className="py-1.5 px-2 text-right">IR</th>
                  <th className="py-1.5 px-2 text-right">t-stat</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
                {icResults.map((row) => (
                  <tr key={row.id}>
                    <td className="py-1.5 px-2 text-zinc-800 dark:text-zinc-200">{row.label}</td>
                    <td className="py-1.5 px-2 text-right">{row.observations}</td>
                    <td className="py-1.5 px-2 text-right text-gold-700 dark:text-gold-300">
                      {row.meanIc !== null ? row.meanIc.toFixed(3) : '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right">{row.icStd !== null ? row.icStd.toFixed(3) : '—'}</td>
                    <td className="py-1.5 px-2 text-right">{row.icIr !== null ? row.icIr.toFixed(2) : '—'}</td>
                    <td className="py-1.5 px-2 text-right">{row.tStat !== null ? row.tStat.toFixed(2) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel-inset overflow-x-auto">
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mb-2">
              Latest Composite Ranking ({latestRanking[0]?.date ?? '—'})
            </div>
            <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-800 text-[11px] font-mono">
              <thead>
                <tr className="text-zinc-500">
                  <th className="py-1.5 px-2 text-left">Rank</th>
                  <th className="py-1.5 px-2 text-left">Symbol</th>
                  <th className="py-1.5 px-2 text-left">Sector</th>
                  <th className="py-1.5 px-2 text-right">Composite z</th>
                  <th className="py-1.5 px-2 text-right">Decile</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
                {latestRanking.map((row) => (
                  <tr key={row.symbol}>
                    <td className="py-1.5 px-2">{row.rank ?? '—'}</td>
                    <td className="py-1.5 px-2 font-bold text-zinc-900 dark:text-zinc-100">{row.symbol}</td>
                    <td className="py-1.5 px-2 text-zinc-500">{row.sector}</td>
                    <td className="py-1.5 px-2 text-right text-gold-700 dark:text-gold-300">
                      {row.composite !== null ? row.composite.toFixed(3) : '—'}
                    </td>
                    <td className="py-1.5 px-2 text-right">{row.decile ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
