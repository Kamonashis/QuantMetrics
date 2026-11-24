'use client';

import { useEffect, useMemo, useState } from 'react';
import { completedResearchBars, MIN_RESEARCH_OUTCOMES, RESEARCH_HISTORY_RANGE, scoreOptionsResearch, usMarketDate,
  type ResearchBar, type ResearchObservation } from '@/lib/options-research';

const pct = (x: number | null) => x === null ? '—' : `${(100 * x).toFixed(1)}%`;
const note = 'mt-2 text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400';

function exportJournal(symbol: string, records: ResearchObservation[]) {
  const columns = ['symbol', 'session', 'quote_as_of', 'recorded_at', 'origin', 'spot', 'quality', 'iv30', 'rv20', 'variance_spread_vol_points_squared',
    'verdict', 'curve', 'expiry', 'digital_barrier', 'svi_probability', 'atm_probability', 'mfiv_by_expiry'];
  const rows = records.map((r) => [symbol, r.date, r.asOf, r.recordedAt, r.origin, r.spot, r.quality,
    r.vrp?.iv30, r.vrp?.rv20, r.vrp?.spread, r.vrp?.verdict, r.vrp?.curve, r.digital?.expiry, r.digital?.strike,
    r.digital?.svi, r.digital?.atm, JSON.stringify(r.mfiv)]);
  const csv = [columns, ...rows].map((row) => row.map((v) => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `${symbol}-options-research.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export function T5Calibration({ symbol, records, error }: { symbol: string; records: ResearchObservation[] | null; error: string | null }) {
  const [history, setHistory] = useState<{ symbol: string; bars: ResearchBar[] | null } | null>(null);
  const hasJournal = !!records?.length;
  useEffect(() => {
    if (!symbol || !hasJournal) return;
    const controller = new AbortController();
    const save = (bars: ResearchBar[] | null) => { if (!controller.signal.aborted) setHistory({ symbol, bars }); };
    fetch(`/api/market-data?tickers=${encodeURIComponent(symbol)}&range=${RESEARCH_HISTORY_RANGE}`, { signal: controller.signal })
      .then((r) => r.json()).then((r) => {
        const market = r.data?.[symbol];
        const today = usMarketDate(new Date().toISOString());
        save(market?.source === 'live_market' && Array.isArray(market.bars) && today ? completedResearchBars(market.bars, today) : null);
      }).catch(() => save(null));
    return () => controller.abort();
  }, [symbol, hasJournal]);
  const bars = history?.symbol === symbol ? history.bars : undefined;
  const score = useMemo(() => bars && records ? scoreOptionsResearch(records, bars) : null, [records, bars]);
  const digitalCount = records?.filter((r) => r.digital).length ?? 0;
  return (
    <section className="panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="kicker">Calibration &amp; research log{symbol ? ` · ${symbol}` : ''}</p>
        {hasJournal && <button type="button" className="panel-inset px-3 py-2 text-xs" onClick={() => exportJournal(symbol, records)}>Export observations CSV</button>}
      </div>
      <p className={note}>Automatically records the first eligible snapshot per session when the workstation is opened. {records?.length ?? 0} observations · {digitalCount} SVI digital forecasts. CSV logs the signal, not a simulated fill or a P&amp;L series.</p>
      {error && <p role="status" className={note}>{error}</p>}
      {bars === null && <p className={note}>Observed outcome bars unavailable. Synthetic data is not scored.</p>}
      {!symbol && <p className={note}>A valid option-chain symbol is required to load the journal.</p>}
      {symbol && records === null && !error && <p className={note}>Loading journal…</p>}
      {hasJournal && bars === undefined && <p className={note}>Loading observed closes…</p>}
      {score && <div className="mt-3 grid gap-3 xl:grid-cols-2">
        <div className="panel-inset p-3 text-xs">
          <strong>SVI +10% digital · exact expiry close</strong>
          <p className={note}>{score.digitalN} resolved · {score.digitalPending} pending or missing. Brier score: {score.brier?.toFixed(4) ?? 'insufficient history'}.</p>
          <p className={note}>Skill vs. flat ATM: {pct(score.atm.n >= MIN_RESEARCH_OUTCOMES ? score.atm.skill : null)} ({score.atm.n} paired) · vs. trailing historical frequency: {pct(score.historical.n >= MIN_RESEARCH_OUTCOMES ? score.historical.skill : null)} ({score.historical.n} paired).</p>
          {score.bins.length > 0 && <table className="mt-3 w-full text-right tabular-nums [&_td]:py-1">
            <thead><tr><th scope="col">Forecast band</th><th scope="col">n</th><th scope="col">Mean forecast</th><th scope="col">Observed</th></tr></thead>
            <tbody>{score.bins.map((b) => <tr key={b.lo}><td>{Math.round(b.lo * 100)}–{Math.round(b.hi * 100)}%</td><td>{b.n}</td><td>{pct(b.n >= 5 ? b.predicted : null)}</td><td>{pct(b.n >= 5 ? b.observed : null)}</td></tr>)}</tbody>
          </table>}
        </div>
        <div className="panel-inset p-3 text-xs">
          <strong>Trailing IV²−RV² label vs. next 20 sessions</strong>
          {score.groups.map((g) => <p key={g.origin} className={note}>{g.origin === 'prospective' ? 'Recorded · sample-stdev RV' : 'Reconstructed panel · RMS RV'}: {pct(g.hitRate)} hit rate ({g.hits}/{g.n} directional outcomes; {g.neutral} neutral/ties; {g.pending} pending or missing).</p>)}
          <p className={note}>A CHEAP label hits if subsequent RV exceeds the IV captured at entry, and RICH if it falls below. This is not a P&amp;L backtest.</p>
        </div>
      </div>}
      <p className={note}>Scores need {MIN_RESEARCH_OUTCOMES} resolved observations per comparison; reliability bins need 5 observations each. Overlapping sessions are dependent. Risk-neutral probabilities need not match realized frequencies; Brier is a diagnostic, not an edge estimate.</p>
      <p className={note}>For a quarter-long paper study, record bids/asks, fees, slippage, and exits separately. Spreads and condors can cap expiration losses under their stated assumptions; assignment, pin risk, and execution still matter. An inverted curve or IV below trailing RV is not a buy/sell instruction.</p>
    </section>
  );
}
