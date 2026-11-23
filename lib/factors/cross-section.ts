// Cross-sectional factor standardization, composite construction and IC reads.
// Pure TS. All transformations are deterministic and null-aware.

import type {
  CrossSectionOptions,
  FactorDefinition,
  FactorIcResult,
  FactorPanelRow,
} from './types';

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function quantileSorted(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * Math.max(0, Math.min(1, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function winsorize(
  values: readonly (number | null)[],
  lower = 0.01,
  upper = 0.99
): (number | null)[] {
  const valid = values.filter(finite) as number[];
  if (valid.length === 0) return values.map(() => null);
  const sorted = [...valid].sort((a, b) => a - b);
  const lo = quantileSorted(sorted, lower);
  const hi = quantileSorted(sorted, upper);
  return values.map((v) => (finite(v) ? Math.max(lo, Math.min(hi, v)) : null));
}

/** Average-tie cross-sectional rank, 1 = smallest. */
export function rankify(values: readonly (number | null)[]): (number | null)[] {
  const indexed = values
    .map((v, i) => ({ v, i }))
    .filter((x): x is { v: number; i: number } => finite(x.v));
  const sorted = [...indexed].sort((a, b) => a.v - b.v);
  const out: (number | null)[] = values.map(() => null);
  let k = 0;
  while (k < sorted.length) {
    let j = k + 1;
    while (j < sorted.length && sorted[j].v === sorted[k].v) j++;
    const rank = (k + j - 1) / 2 + 1;
    for (let m = k; m < j; m++) out[sorted[m].i] = rank;
    k = j;
  }
  return out;
}

export function pearsonCorrelation(x: readonly number[], y: readonly number[]): number | null {
  const n = Math.min(x.length, y.length);
  if (n < 3) return null;
  const mx = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const my = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
  let cov = 0;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    cov += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  if (vx <= 1e-12 || vy <= 1e-12) return null;
  const r = cov / Math.sqrt(vx * vy);
  return finite(r) ? r : null;
}

export function rankIc(x: readonly (number | null)[], y: readonly (number | null)[]): number | null {
  const pairs: { x: number; y: number }[] = [];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    if (finite(x[i]) && finite(y[i])) pairs.push({ x: x[i] as number, y: y[i] as number });
  }
  if (pairs.length < 4) return null;
  return pearsonCorrelation(rankify(pairs.map((p) => p.x)) as number[], rankify(pairs.map((p) => p.y)) as number[]);
}

export interface FactorRawRow {
  date: string;
  symbol: string;
  sector: string;
  raw: Record<string, number | null>;
  forwardReturn: number | null;
}

function zScore(values: readonly (number | null)[]): (number | null)[] {
  const valid = values.filter(finite) as number[];
  if (valid.length < 3) return values.map(() => null);
  const m = valid.reduce((a, b) => a + b, 0) / valid.length;
  const variance = valid.reduce((a, b) => a + (b - m) ** 2, 0) / (valid.length - 1);
  const sd = Math.sqrt(variance);
  if (sd < 1e-12) return values.map(() => null);
  return values.map((v) => (finite(v) ? (v - m) / sd : null));
}

/** Standardize each factor within each date, optionally neutralizing by sector. */
export function scoreCrossSection(
  rows: readonly FactorRawRow[],
  definitions: readonly FactorDefinition[],
  options: CrossSectionOptions = {}
): FactorPanelRow[] {
  const byDate = new Map<string, FactorRawRow[]>();
  for (const row of rows) {
    const list = byDate.get(row.date) ?? [];
    list.push(row);
    byDate.set(row.date, list);
  }
  const factorById = new Map(definitions.map((d) => [d.id, d]));
  const out: FactorPanelRow[] = [];

  for (const [date, group] of byDate) {
    const zByFactor: Record<string, (number | null)[]> = {};
    for (const def of definitions) {
      let raw = winsorize(
        group.map((r) => r.raw[def.id] ?? null),
        options.lower ?? 0.01,
        options.upper ?? 0.99
      );
      // Direction: +1 means high raw is attractive; -1 flips the sign.
      raw = raw.map((v) => (v === null ? null : v * def.direction));
      if (options.sectorNeutral) {
        const bySector = new Map<string, number[]>();
        group.forEach((r, i) => {
          const v = raw[i];
          if (v === null) return;
          const list = bySector.get(r.sector) ?? [];
          list.push(v);
          bySector.set(r.sector, list);
        });
        const sectorMean = new Map<string, number>();
        for (const [sector, list] of bySector) {
          sectorMean.set(sector, list.reduce((a, b) => a + b, 0) / list.length);
        }
        raw = raw.map((v, i) => (v === null ? null : v - (sectorMean.get(group[i].sector) ?? 0)));
      }
      zByFactor[def.id] = zScore(raw).map((v) =>
        v === null ? null : Math.max(-(options.maxAbsoluteZ ?? 4), Math.min(options.maxAbsoluteZ ?? 4, v))
      );
    }

    group.forEach((row, i) => {
      const z: Record<string, number | null> = {};
      let weighted = 0;
      let weightTotal = 0;
      for (const def of definitions) {
        const value = zByFactor[def.id][i];
        z[def.id] = value;
        if (value === null) continue;
        const requested = options.weights?.[def.id];
        const weight = finite(requested) && (requested as number) !== 0 ? (requested as number) : 1;
        weighted += value * weight;
        weightTotal += Math.abs(weight);
      }
      out.push({
        date,
        symbol: row.symbol,
        sector: row.sector,
        raw: row.raw,
        z,
        composite: weightTotal > 0 ? weighted / weightTotal : null,
        rank: null,
        decile: null,
        forwardReturn: row.forwardReturn,
      });
    });

    const ranked = out
      .filter((r) => r.date === date)
      .filter((r) => r.composite !== null)
      .sort((a, b) => (b.composite as number) - (a.composite as number));
    ranked.forEach((r, idx) => {
      r.rank = idx + 1;
      r.decile = Math.min(10, Math.floor((idx / Math.max(1, ranked.length)) * 10) + 1);
    });
  }
  return out;
}

export function computeFactorIc(
  rows: readonly FactorPanelRow[],
  definitions: readonly FactorDefinition[],
  compositeId = 'composite'
): FactorIcResult[] {
  const byDate = new Map<string, FactorPanelRow[]>();
  for (const row of rows) {
    const list = byDate.get(row.date) ?? [];
    list.push(row);
    byDate.set(row.date, list);
  }
  const ids = [...definitions.map((d) => d.id), compositeId];
  const series: Record<string, number[]> = {};
  for (const id of ids) series[id] = [];
  const topBottoms: { top: number; bottom: number }[] = [];

  for (const [, group] of byDate) {
    for (const id of ids) {
      const ic = rankIc(
        group.map((r) => (id === compositeId ? r.composite : r.z[id] ?? null)),
        group.map((r) => r.forwardReturn)
      );
      if (ic !== null) series[id].push(ic);
    }
    const withReturn = group.filter(
      (r) => r.composite !== null && finite(r.forwardReturn)
    ) as (FactorPanelRow & { composite: number; forwardReturn: number })[];
    if (withReturn.length >= 4) {
      const top = withReturn.filter((r) => r.decile === 10);
      const bottom = withReturn.filter((r) => r.decile === 1);
      if (top.length > 0 && bottom.length > 0) {
        topBottoms.push({
          top: top.reduce((a, b) => a + b.forwardReturn, 0) / top.length,
          bottom: bottom.reduce((a, b) => a + b.forwardReturn, 0) / bottom.length,
        });
      }
    }
  }

  return ids.map((id) => {
    const values = series[id];
    const n = values.length;
    const meanIc = n > 0 ? values.reduce((a, b) => a + b, 0) / n : null;
    const sd = n > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - meanIc!) ** 2, 0) / (n - 1)) : null;
    const tStat = meanIc !== null && sd !== null && sd > 1e-12 && n > 0 ? (meanIc / sd) * Math.sqrt(n) : null;
    const def = definitions.find((d) => d.id === id);
    return {
      id,
      label: def?.label ?? 'Composite',
      observations: n,
      meanIc,
      rankIc: meanIc,
      icStd: sd,
      icIr: meanIc !== null && sd !== null && sd > 1e-12 ? meanIc / sd : null,
      tStat,
      topMinusBottom:
        id === compositeId && topBottoms.length > 0
          ? topBottoms.reduce((a, b) => a + (b.top - b.bottom), 0) / topBottoms.length
          : null,
    };
  });
}

/** Compound a daily long-short/decile spread series for the equity curve. */
export function compositeEquityCurve(
  rows: readonly FactorPanelRow[]
): { date: string; value: number }[] {
  const byDate = new Map<string, number[]>();
  for (const row of rows) {
    if (!finite(row.forwardReturn) || row.decile === null) continue;
    const list = byDate.get(row.date) ?? [];
    if (row.decile === 10) list.push(row.forwardReturn as number);
    else if (row.decile === 1) list.push(-(row.forwardReturn as number));
    byDate.set(row.date, list);
  }
  let equity = 1;
  return [...byDate.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([date, values]) => {
      if (values.length > 0) {
        const spread = values.reduce((a, b) => a + b, 0) / values.length;
        equity *= 1 + spread;
      }
      return { date, value: equity };
    });
}
