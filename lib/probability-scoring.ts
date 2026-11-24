export interface BinaryForecast { probability: number; outcome: 0 | 1 }

export function brierScore(rows: BinaryForecast[]) {
  return rows.length ? rows.reduce((sum, r) => sum + (r.probability - r.outcome) ** 2, 0) / rows.length : null;
}

export function brierSkillScore(rows: (BinaryForecast & { baseline: number | null })[]) {
  const paired = rows.filter((r) => r.baseline !== null);
  const reference = brierScore(paired.map((r) => ({ probability: r.baseline!, outcome: r.outcome })));
  const model = brierScore(paired);
  return { n: paired.length, skill: reference !== null && reference > 0 && model !== null ? 1 - model / reference : null };
}

export function reliabilityBins(rows: BinaryForecast[], count = 5) {
  return Array.from({ length: count }, (_, i) => {
    const lo = i / count, hi = (i + 1) / count;
    const bin = rows.filter((r) => r.probability >= lo && (i === count - 1 ? r.probability <= hi : r.probability < hi));
    return { lo, hi, n: bin.length, predicted: bin.length ? bin.reduce((s, r) => s + r.probability, 0) / bin.length : null,
      observed: bin.length ? bin.reduce((s, r) => s + r.outcome, 0) / bin.length : null };
  });
}
