import fs from 'node:fs';
import path from 'node:path';
import { readPanel, snapshotDir } from './iv-store';
import { MAX_RESEARCH_AGE_MS, reconstructPanelObservations, usMarketDate, vrpObservation, type ResearchObservation } from './options-research';

const journalPath = (symbol: string) => path.join(snapshotDir(symbol), 'options-research.json');
const day = /^\d{4}-\d{2}-\d{2}$/;
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
const probability = (p: unknown): p is number => finite(p) && p >= 0 && p <= 1;
const object = (v: unknown): Record<string, unknown> | null => v != null && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : null;

export function readOptionsResearch(symbol: string): ResearchObservation[] {
  const file = journalPath(symbol);
  if (!fs.existsSync(file)) return [];
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as { version?: number; records?: ResearchObservation[] };
  if (parsed.version !== 1 || !Array.isArray(parsed.records)) throw new Error('Unrecognized options research journal.');
  return parsed.records;
}

export function writeOptionsResearch(symbol: string, input: unknown, now = new Date()) {
  const data = object(input);
  const asOf = typeof data?.asOf === 'string' ? data.asOf : '';
  const date = usMarketDate(asOf);
  const current = usMarketDate(now.toISOString());
  const age = now.getTime() - Date.parse(asOf);
  if (!data || !date || date !== current || age < -60_000 || age > MAX_RESEARCH_AGE_MS || !finite(data.spot) || data.spot <= 0) {
    throw new Error('A current-session vendor quote is required to record a forecast.');
  }
  const weekday = new Date(`${date}T00:00:00Z`).getUTCDay();
  if (weekday === 0 || weekday === 6) throw new Error('Research observations require a market session.');
  const quality = data.quality;
  if (quality !== 'READY' && quality !== 'DEGRADED' && quality !== 'NOT_READY') throw new Error('Invalid research quality.');
  const v = object(data.vrp);
  const vrp = v && finite(v.iv30) && finite(v.rv20) ? vrpObservation(v.iv30, v.rv20,
    v.curve === 'FLAT' || v.curve === 'CONTANGO' || v.curve === 'INVERTED' ? v.curve : null) : null;
  const d = object(data.digital);
  const expiry = typeof d?.expiry === 'string' && day.test(d.expiry) ? d.expiry : '';
  const days = (Date.parse(expiry) - Date.parse(date)) / 86_400_000;
  const digital = quality !== 'NOT_READY' && d && days > 0 && days <= 90 && probability(d.svi)
    ? { expiry, strike: data.spot * 1.1, svi: d.svi, atm: probability(d.atm) ? d.atm : null } : null;
  if (!vrp && !digital) throw new Error('No eligible VRP or digital observation.');

  const mfiv = Array.isArray(data.mfiv) ? data.mfiv.slice(0, 24).flatMap((item: unknown) => {
    const m = object(item);
    return typeof m?.expiry === 'string' && day.test(m.expiry) ? [{ expiry: m.expiry, diffPts: finite(m.diffPts) ? m.diffPts : null }] : [];
  }) : [];
  const recordedAt = now.toISOString();
  const records = readOptionsResearch(symbol);
  const dates = new Set(records.map((r) => r.date));
  const additions = reconstructPanelObservations(readPanel(symbol), date, recordedAt).filter((r) => !dates.has(r.date));
  const written = !dates.has(date);
  if (written) additions.push({ date, asOf, recordedAt, spot: data.spot, origin: 'prospective', quality, vrp, digital, mfiv });
  if (!additions.length) return { records, written };
  const merged = [...records, ...additions].sort((a, b) => a.date.localeCompare(b.date));
  const file = journalPath(symbol);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, JSON.stringify({ version: 1, records: merged }), { flag: 'w' });
  fs.renameSync(`${file}.tmp`, file);
  return { records: merged, written };
}
