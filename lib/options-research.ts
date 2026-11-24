import { historicalHorizonReturns, type Quality } from './rnd-integrity';
import { canonicalVrp } from './ticker-snapshot';
import type { OptionsDecision } from './options-decision';
import { brierScore, brierSkillScore, reliabilityBins } from './probability-scoring';

export const MIN_RESEARCH_OUTCOMES = 30;
export const RESEARCH_HISTORY_RANGE = '2y';
export const MAX_RESEARCH_AGE_MS = 24 * 60 * 60 * 1000;
const DAY = 86_400_000;
const nyDate = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });

export function usMarketDate(iso: string) {
  if (!Number.isFinite(Date.parse(iso))) return null;
  const parts = Object.fromEntries(nyDate.formatToParts(new Date(iso)).map((p) => [p.type, p.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}

export type Verdict = Exclude<OptionsDecision['verdict'], 'insufficient-data'>;
type Curve = OptionsDecision['curve'];

export interface ResearchObservation {
  date: string;
  asOf: string;
  recordedAt: string;
  spot: number;
  origin: 'prospective' | 'panel-reconstruction';
  quality: Quality | null;
  vrp: { iv30: number; rv20: number; spread: number; verdict: Verdict; curve: Curve } | null;
  digital: { expiry: string; strike: number; svi: number; atm: number | null } | null;
  mfiv: { expiry: string; diffPts: number | null }[];
}

export function vrpObservation(iv30: number | null, rv20: number | null, curve: Curve = null): ResearchObservation['vrp'] {
  if (iv30 === null || rv20 === null || !(iv30 > 0) || rv20 < 0) return null;
  const spread = canonicalVrp(iv30, rv20).variancePoints;
  if (spread === null || !Number.isFinite(spread)) return null;
  return { iv30, rv20, spread, curve, verdict: spread > 5e-4 ? 'RICH' : spread < -5e-4 ? 'CHEAP' : 'NEUTRAL' };
}

export function reconstructPanelObservations(rows: { date: string; computedAt: string; spot: number; atmIv30: number | null; rv20: number | null }[], before: string, recordedAt: string) {
  return rows.flatMap((row): ResearchObservation[] => {
    const vrp = vrpObservation(row.atmIv30, row.rv20);
    if (!vrp || !Number.isFinite(row.spot) || row.spot <= 0 || row.date >= before || usMarketDate(row.computedAt) !== row.date) return [];
    const dow = new Date(`${row.date}T00:00:00Z`).getUTCDay();
    if (!dow || dow === 6) return [];
    return [{ date: row.date, asOf: row.computedAt, recordedAt, spot: row.spot, origin: 'panel-reconstruction', quality: null,
      vrp, digital: null, mfiv: [] }];
  });
}

export type ResearchBar = { date: string; close: number };

export function completedResearchBars(bars: ResearchBar[], before: string) {
  return [...new Map(bars.filter((b) => b.date < before && Number.isFinite(b.close) && b.close > 0)
    .map((b) => [b.date, b])).values()].sort((a, b) => a.date.localeCompare(b.date));
}

export function researchRv(closes: number[], estimator: ResearchObservation['origin']) {
  if (closes.length < 21 || closes.some((c) => !(c > 0) || !Number.isFinite(c))) return null;
  const series = closes.slice(-21);
  const returns = series.slice(1).map((c, i) => Math.log(c / series[i]));
  const sample = estimator === 'prospective';
  const mean = sample ? returns.reduce((s, x) => s + x, 0) / 20 : 0;
  return Math.sqrt(252 * returns.reduce((s, x) => s + (x - mean) ** 2, 0) / (sample ? 19 : 20));
}

export function trailingResearchRv(bars: ResearchBar[], quoteAsOf: string | null | undefined) {
  const date = usMarketDate(quoteAsOf ?? '');
  if (!date) return null;
  const complete = completedResearchBars(bars, date);
  const recent = complete.at(-1);
  if (!recent || Date.parse(date) - Date.parse(recent.date) > 4 * DAY) return null;
  return researchRv(complete.slice(-21).map((b) => b.close), 'prospective');
}

export function scoreOptionsResearch(records: ResearchObservation[], bars: ResearchBar[]) {
  const index = new Map(bars.map((b, i) => [b.date, i]));
  const forecasts: { probability: number; outcome: 0 | 1; atm: number | null; historical: number | null }[] = [];
  const groups = (['prospective', 'panel-reconstruction'] as const).map((origin) => ({ origin, resolved: 0, pending: 0, neutral: 0,
    n: 0, hits: 0 }));
  let digitalPending = 0;
  for (const rec of records) {
    const i = index.get(rec.date);
    if (rec.vrp) {
      const group = groups[rec.origin === 'prospective' ? 0 : 1];
      const forward = i !== undefined && i + 20 < bars.length ? researchRv(bars.slice(i, i + 21).map((b) => b.close), rec.origin) : null;
      if (forward === null) group.pending++;
      else {
        group.resolved++;
        if (rec.vrp.verdict === 'NEUTRAL' || forward === rec.vrp.iv30) group.neutral++;
        else {
          group.n++;
          if ((rec.vrp.verdict === 'CHEAP') === (forward > rec.vrp.iv30)) group.hits++;
        }
      }
    }
    if (rec.digital) {
      const expiry = index.get(rec.digital.expiry);
      // No nearest-date substitution for a contract's expiry-close outcome.
      if (i === undefined || expiry === undefined || expiry <= i) { digitalPending++; continue; }
      const days = Math.round((Date.parse(rec.digital.expiry) - Date.parse(rec.date)) / DAY);
      const prior = historicalHorizonReturns(bars.slice(0, i), days);
      const barrier = Math.log(rec.digital.strike / rec.spot);
      const historical = prior.length >= MIN_RESEARCH_OUTCOMES ? prior.filter((r) => r > barrier).length / prior.length : null;
      forecasts.push({ probability: rec.digital.svi, outcome: bars[expiry].close > rec.digital.strike ? 1 : 0, atm: rec.digital.atm, historical });
    }
  }
  const ready = forecasts.length >= MIN_RESEARCH_OUTCOMES;
  return { digitalPending, digitalN: forecasts.length, brier: ready ? brierScore(forecasts) : null,
    atm: brierSkillScore(forecasts.map((r) => ({ ...r, baseline: r.atm }))),
    historical: brierSkillScore(forecasts.map((r) => ({ ...r, baseline: r.historical }))),
    bins: ready ? reliabilityBins(forecasts) : [], groups: groups.map((g) => ({ ...g, hitRate: g.n >= MIN_RESEARCH_OUTCOMES ? g.hits / g.n : null })) };
}
