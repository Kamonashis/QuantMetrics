'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, FlaskConical, Grid3x3, LineChart, RefreshCw, Sparkles, Waypoints } from 'lucide-react';
import type { PriceBar } from '@/lib/quant-engine';
import {
  CAUTION_OBSERVATIONS,
  MODEL_PRESETS,
  STYLE_FACTORS,
  attributeReturns,
  buildFactorSample,
  fitSample,
  requiredTickers,
  type StyleFactorId,
} from '@/lib/factor-regression';
import { useFactorData } from '@/hooks/use-factor-data';
import { FactorResearchPanel } from '@/components/views/regression/FactorResearchPanel';
import { ExplainTab } from '@/components/views/regression/ExplainTab';
import { StabilityTab } from '@/components/views/regression/StabilityTab';
import { PredictTab } from '@/components/views/regression/PredictTab';
import { MethodsTab } from '@/components/views/regression/MethodsTab';
import { Callout } from '@/components/ui/lab-ui';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { TickerSuggest } from '@/components/ui/TickerSuggest';
import { ProvenanceBadge } from '@/components/ui/provenance-badge';
import { AnimatedTabs } from '@/components/ui/animated-tabs';

const FETCH_CHUNK = 10;

type LabTab = 'explain' | 'stability' | 'predict' | 'screen' | 'methods';

interface RegressionViewProps {
  regressionTickers: string;
  onRegressionTickersChange: (val: string) => void;
  startDate: string;
  onStartDateChange: (val: string) => void;
  endDate: string;
  onEndDateChange: (val: string) => void;
  returnType: 'simple' | 'log';
  allStockBars: { [ticker: string]: PriceBar[] };
  isLoading: boolean;
  onRunRegression: (tickers: string[]) => void | Promise<void>;
  barsSource?: string | null;
  sourcesByTicker?: Record<string, string>;
}

function parseTickers(raw: string): string[] {
  return raw
    .split(',')
    .map((t) => t.trim().toUpperCase())
    .filter((t) => t.length > 0);
}

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

const inputClass =
  'w-full px-3 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-gold-400';

export function RegressionView({
  regressionTickers,
  onRegressionTickersChange,
  startDate,
  onStartDateChange,
  endDate,
  onEndDateChange,
  returnType,
  allStockBars,
  isLoading,
  onRunRegression,
  sourcesByTicker = {},
}: RegressionViewProps) {
  const parsed = useMemo(() => parseTickers(regressionTickers), [regressionTickers]);
  const asset = parsed[0] ?? 'AAPL';
  const benchmark = parsed[1] && parsed[1] !== asset ? parsed[1] : 'SPY';
  const customTickers = useMemo(() => parsed.slice(2).filter((t) => t !== asset && t !== benchmark), [parsed, asset, benchmark]);

  const [assetDraft, setAssetDraft] = useState(asset);
  const [benchmarkDraft, setBenchmarkDraft] = useState(benchmark);
  const [customDraft, setCustomDraft] = useState(customTickers.join(', '));
  const [factorIds, setFactorIds] = useState<StyleFactorId[]>(['SIZE', 'VALUE']);
  const [tab, setTab] = useState<LabTab>('explain');
  const [formError, setFormError] = useState<string | null>(null);

  const [universeSymbols, setUniverseSymbols] = useState<string[]>(() => Object.keys(allStockBars ?? {}).slice(0, 12));
  const factorData = useFactorData(universeSymbols, tab === 'screen');

  const config = useMemo(
    () => ({ asset, benchmark, factors: factorIds, customTickers, startDate, endDate, returnType }),
    [asset, benchmark, factorIds, customTickers, startDate, endDate, returnType]
  );
  const needed = useMemo(() => requiredTickers(config), [config]);
  const missingBars = useMemo(() => needed.filter((t) => !(allStockBars[t]?.length > 0)), [needed, allStockBars]);
  const syntheticTickers = useMemo(
    () => needed.filter((t) => (sourcesByTicker[t] ?? 'synthetic_gbm') !== 'live_market'),
    [needed, sourcesByTicker]
  );

  // Fetch any series the model needs but the cache has never seen.
  const requestedRef = useRef<Set<string>>(new Set());
  const onRunRef = useRef(onRunRegression);
  useEffect(() => {
    onRunRef.current = onRunRegression;
  }, [onRunRegression]);
  const missingKey = missingBars.join(',');
  useEffect(() => {
    const toFetch = missingKey ? missingKey.split(',').filter((t) => !requestedRef.current.has(t)) : [];
    if (toFetch.length === 0) return;
    // Mark as requested only once the timer fires, so a StrictMode cleanup can't swallow the fetch.
    const timer = setTimeout(() => {
      toFetch.forEach((t) => requestedRef.current.add(t));
      void (async () => {
        for (const group of chunk(toFetch, FETCH_CHUNK)) await onRunRef.current(group);
      })();
    }, 0);
    return () => clearTimeout(timer);
  }, [missingKey]);

  const refreshAll = () => {
    void (async () => {
      for (const group of chunk(needed, FETCH_CHUNK)) await onRunRegression(group);
    })();
  };

  const sampleResult = useMemo(() => buildFactorSample(allStockBars, config), [allStockBars, config]);
  const fit = useMemo(() => (sampleResult.ok ? fitSample(sampleResult.sample) : null), [sampleResult]);
  const goodFit = fit && fit.ok ? fit : null;
  const attribution = useMemo(
    () => (sampleResult.ok && goodFit ? attributeReturns(sampleResult.sample, goodFit) : null),
    [sampleResult, goodFit]
  );

  const presetMatch = MODEL_PRESETS.find(
    (p) => p.factors.length === factorIds.length && p.factors.every((f) => factorIds.includes(f))
  );
  const modelLabel = `${presetMatch ? presetMatch.label.toLowerCase() : `market + ${factorIds.length} style factor${factorIds.length === 1 ? '' : 's'}`}${
    customTickers.length > 0 ? ` + ${customTickers.join(', ')}` : ''
  }`;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const a = parseTickers(assetDraft)[0];
    const b = parseTickers(benchmarkDraft)[0];
    if (!a || !b) {
      setFormError('Enter both an asset and a benchmark.');
      return;
    }
    if (a === b) {
      setFormError('The asset and the benchmark must be different.');
      return;
    }
    const extras = parseTickers(customDraft).filter((t) => t !== a && t !== b).slice(0, 4);
    setFormError(null);
    setAssetDraft(a);
    setBenchmarkDraft(b);
    setCustomDraft(extras.join(', '));
    onRegressionTickersChange([a, b, ...extras].join(', '));
  };

  const toggleFactor = (id: StyleFactorId) =>
    setFactorIds((prev) => (prev.includes(id) ? prev.filter((f) => f !== id) : STYLE_FACTORS.map((f) => f.id).filter((f) => f === id || prev.includes(f))));

  const sample = sampleResult.ok ? sampleResult.sample : null;
  const audit = sampleResult.ok ? sampleResult.sample.audit : sampleResult.audit;
  // The data layer always lands bars (live or synthetic fallback), so "missing" means "in flight".
  const fetching = missingBars.length > 0;
  const allLive = syntheticTickers.length === 0;

  const tabs = [
    { id: 'explain', label: 'Explain returns', icon: FlaskConical },
    { id: 'stability', label: 'Test stability', icon: Waypoints },
    { id: 'predict', label: 'Predict', icon: Sparkles, meta: 'experimental' },
    { id: 'screen', label: 'Cross-section screen', icon: Grid3x3 },
    { id: 'methods', label: 'Methods & data', icon: BookOpen },
  ];

  return (
    <div className="space-y-6">
      {/* Setup */}
      <div className="panel p-5">
        <form onSubmit={handleSubmit}>
          <div className="flex flex-col justify-between gap-3 border-b border-[var(--hairline)] pb-4 lg:flex-row lg:items-end">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="kicker">Factor Regression Lab</span>
                <ProvenanceBadge daily={allLive ? 'live_market' : 'synthetic_gbm'} />
              </div>
              <h2 className="mt-1 font-display text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
                What drives {asset}&apos;s returns?
              </h2>
              <p className="mt-1 max-w-2xl text-xs text-zinc-500 dark:text-zinc-400">
                Pick an asset, a benchmark, and the factors to test. The lab explains past daily returns as exposures to each
                factor, then checks whether those exposures held up.
              </p>
            </div>
            <button
              type="button"
              onClick={refreshAll}
              disabled={isLoading}
              className="inline-flex cursor-pointer items-center gap-1.5 self-start rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] px-2.5 py-1.5 font-mono text-[11px] font-semibold text-zinc-700 hover:border-gold-500/50 disabled:opacity-50 dark:text-zinc-300 lg:self-auto"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh data
            </button>
          </div>

          <div className="mt-4 grid grid-cols-1 gap-4 text-xs sm:grid-cols-2 lg:grid-cols-12">
            <div className="lg:col-span-2">
              <label htmlFor="lab-asset" className="mb-1 block font-medium text-zinc-700 dark:text-zinc-300">Asset to explain</label>
              <TickerSuggest id="lab-asset" ariaLabel="Asset" value={assetDraft} onChange={setAssetDraft} placeholder="AAPL" inputClassName={inputClass} />
            </div>
            <div className="lg:col-span-2">
              <label htmlFor="lab-benchmark" className="mb-1 block font-medium text-zinc-700 dark:text-zinc-300">Market benchmark</label>
              <TickerSuggest id="lab-benchmark" ariaLabel="Benchmark" value={benchmarkDraft} onChange={setBenchmarkDraft} placeholder="SPY" inputClassName={inputClass} />
            </div>
            <div className="lg:col-span-3">
              <label htmlFor="lab-custom" className="mb-1 block font-medium text-zinc-700 dark:text-zinc-300">
                Extra regressors <span className="font-normal text-zinc-400">(optional, up to 4)</span>
              </label>
              <TickerSuggest
                id="lab-custom"
                multiple
                ariaLabel="Extra regressors"
                value={customDraft}
                onChange={setCustomDraft}
                placeholder="e.g. TLT, GLD, XLK"
                inputClassName={inputClass}
              />
            </div>
            <div className="lg:col-span-3">
              <span className="mb-1 block font-medium text-zinc-700 dark:text-zinc-300">Date range</span>
              <div className="grid grid-cols-2 gap-1.5">
                <input type="date" aria-label="Start date" value={startDate} onChange={(e) => onStartDateChange(e.target.value)} className={`${inputClass} font-normal text-[11px]`} />
                <input type="date" aria-label="End date" value={endDate} onChange={(e) => onEndDateChange(e.target.value)} className={`${inputClass} font-normal text-[11px]`} />
              </div>
            </div>
            <div className="flex items-end lg:col-span-2">
              <ShimmerButton type="submit" className="w-full justify-center px-4 py-2">
                <LineChart className="h-3.5 w-3.5 text-gold-500" />
                <span>Fit model</span>
              </ShimmerButton>
            </div>
          </div>
          {formError && <p className="mt-2 text-xs text-[var(--negative)]">{formError}</p>}
        </form>

        <div className="mt-4 border-t border-[var(--hairline)] pt-4">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="mr-1 font-medium text-zinc-700 dark:text-zinc-300">Model</span>
            {MODEL_PRESETS.map((p) => {
              const active = presetMatch?.id === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  title={p.hint}
                  aria-pressed={active}
                  onClick={() => setFactorIds([...p.factors])}
                  className={`cursor-pointer rounded-lg border px-2.5 py-1 font-medium transition-colors ${
                    active
                      ? 'border-gold-500/60 bg-gold-500/15 text-gold-800 dark:text-gold-200'
                      : 'border-[var(--hairline)] text-zinc-600 hover:border-gold-500/40 dark:text-zinc-300'
                  }`}
                >
                  {p.label}
                </button>
              );
            })}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] text-zinc-500 dark:text-zinc-400">Factors</span>
            <span className="rounded-md border border-gold-500/50 bg-gold-500/15 px-2 py-0.5 font-mono text-[11px] text-gold-800 dark:text-gold-200" title="Always included">
              Market · {benchmark}
            </span>
            {STYLE_FACTORS.map((f) => {
              const active = factorIds.includes(f.id);
              return (
                <button
                  key={f.id}
                  type="button"
                  aria-pressed={active}
                  title={`${f.description} Proxy: ${f.long} − ${f.shortLeg}.`}
                  onClick={() => toggleFactor(f.id)}
                  className={`cursor-pointer rounded-md border px-2 py-0.5 font-mono text-[11px] transition-colors ${
                    active
                      ? 'border-gold-500/50 bg-gold-500/15 text-gold-800 dark:text-gold-200'
                      : 'border-[var(--hairline)] text-zinc-500 hover:border-gold-500/40'
                  }`}
                >
                  {f.label} <span className="opacity-60">{f.long}−{f.shortLeg}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Sample status */}
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg bg-[var(--bg-inset)] px-3 py-2 font-mono text-[11px] text-zinc-600 dark:text-zinc-400">
          {fetching ? (
            <span className="inline-flex items-center gap-1.5">
              <RefreshCw className="h-3 w-3 animate-spin" /> Loading price history for {missingBars.join(', ')}…
            </span>
          ) : sample ? (
            <>
              <span>
                <strong className="text-zinc-900 dark:text-zinc-100">{audit.returnObs.toLocaleString()}</strong> daily returns
              </span>
              <span>
                {audit.start} → {audit.end}
              </span>
              <span>{returnType} returns · {audit.series.every((s) => s.basis === 'adjusted') ? 'dividend-adjusted' : 'some close-only prices'}</span>
              {audit.droppedAssetSessions > 0 && <span>{audit.droppedAssetSessions} sessions dropped to align calendars</span>}
              <button type="button" onClick={() => setTab('methods')} className="cursor-pointer text-gold-700 underline-offset-2 hover:underline dark:text-gold-300">
                full audit
              </button>
            </>
          ) : (
            <span>No usable sample yet.</span>
          )}
        </div>
      </div>

      {!allLive && !fetching && (
        <Callout tone="warn" title="Some inputs are synthetic">
          {syntheticTickers.join(', ')} {syntheticTickers.length === 1 ? 'is' : 'are'} using the offline simulated price generator
          because live data has not loaded. Results show how the lab works, not how these assets actually behaved. Use Refresh
          data to retry.
        </Callout>
      )}

      <AnimatedTabs tabs={tabs} value={tab} onChange={(id) => setTab(id as LabTab)} panelId="regression-lab-panel" />

      <div id="regression-lab-panel" role="tabpanel">
        {tab === 'screen' ? (
          <FactorResearchPanel
            symbols={universeSymbols}
            allStockBars={allStockBars}
            factorData={factorData}
            onSymbolsChange={setUniverseSymbols}
            startDate={startDate}
            endDate={endDate}
            barsSource={allLive ? 'live_market' : 'synthetic_gbm'}
          />
        ) : !sampleResult.ok ? (
          fetching ? (
            <div className="panel p-8 text-center text-xs text-zinc-500 dark:text-zinc-400">Fetching data…</div>
          ) : (
            <Callout tone="warn" title="Not enough data to fit">
              {sampleResult.reason} Widen the date range, remove a factor whose ETF has a short history, or use Refresh data.
            </Callout>
          )
        ) : fit && !fit.ok ? (
          <Callout tone="danger" title="This model cannot be estimated">
            {fit.reason}
          </Callout>
        ) : (
          sample &&
          goodFit &&
          attribution && (
            <div className="space-y-6">
              {goodFit.n < CAUTION_OBSERVATIONS && tab !== 'methods' && (
                <Callout tone="warn">
                  Only {goodFit.n} returns (under one year). Exposures are imprecise; read the ranges, not just the point estimates.
                </Callout>
              )}
              {tab === 'explain' && (
                <ExplainTab sample={sample} fit={goodFit} attribution={attribution} modelLabel={modelLabel} benchmark={benchmark} />
              )}
              {tab === 'stability' && <StabilityTab sample={sample} fit={goodFit} />}
              {tab === 'predict' && <PredictTab sample={sample} />}
              {tab === 'methods' && <MethodsTab sample={sample} fit={goodFit} sourcesByTicker={sourcesByTicker} />}
            </div>
          )
        )}
      </div>
    </div>
  );
}
