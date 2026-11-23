// Volume-at-Price (Volume Profile) math — pure TS, no React.
//
// Daily bars carry only OHLCV, not tick-level prints, so the profile is an
// approximation: each bar's volume is spread uniformly across its [low, high]
// range. The UI must label it as OHLC-derived, never as actual traded volume
// at a specific price.

import type { PriceBar } from './quant-engine';

export interface VolumeProfileBin {
  lower: number;
  upper: number;
  mid: number;
  volume: number;
  /** Fraction of total profile volume in [0, 1]. */
  share: number;
}

export interface VolumeProfileValueArea {
  low: number;
  high: number;
  /** Achieved share of total volume, >= requested target when feasible. */
  volumeShare: number;
}

export interface VolumeProfileResult {
  bins: VolumeProfileBin[];
  /** Point of control — the bin with the most volume. */
  poc: VolumeProfileBin | null;
  valueArea: VolumeProfileValueArea | null;
  /** High-volume nodes (local maxima) sorted by descending volume. */
  hvn: VolumeProfileBin[];
  totalVolume: number;
  minPrice: number;
  maxPrice: number;
  binWidth: number;
  barCount: number;
  /** Fixed discriminator so callers can label the estimate honestly. */
  approximation: 'ohlc-range';
}

export interface VolumeProfileOptions {
  /** Number of price buckets. Clamped to [8, 120]. Default 48. */
  bins?: number;
  /** Target value-area volume share. Clamped to [0.1, 0.95]. Default 0.7. */
  valueAreaShare?: number;
  /** Minimum bars required. Default 10. */
  minBars?: number;
  /** Relative height for a bin to qualify as a high-volume node. Default 0.6. */
  hvnThreshold?: number;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function clampInt(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

/**
 * Build a Volume-at-Price profile from daily OHLCV bars.
 * Returns null when there is not enough usable data to make a meaningful read.
 */
export function computeVolumeProfile(
  bars: readonly PriceBar[] | null | undefined,
  options: VolumeProfileOptions = {}
): VolumeProfileResult | null {
  const list = Array.isArray(bars) ? bars : [];
  const minBars = clampInt(options.minBars ?? 10, 2, 1000);
  if (list.length < minBars) return null;

  const usable = list.filter(
    (b) =>
      b &&
      isFiniteNumber(b.high) &&
      isFiniteNumber(b.low) &&
      isFiniteNumber(b.close) &&
      isFiniteNumber(b.volume) &&
      b.volume > 0
  );
  if (usable.length < minBars) return null;

  let minPrice = Infinity;
  let maxPrice = -Infinity;
  let totalVolume = 0;
  for (const bar of usable) {
    const lo = Math.min(bar.low, bar.high);
    const hi = Math.max(bar.low, bar.high);
    if (lo < minPrice) minPrice = lo;
    if (hi > maxPrice) maxPrice = hi;
    totalVolume += bar.volume;
  }
  if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice) || totalVolume <= 0) return null;

  if (maxPrice <= minPrice) {
    // Degenerate flat series: build a narrow symmetric window around the price.
    const center = minPrice;
    const pad = Math.max(Math.abs(center) * 0.005, 0.01);
    minPrice = center - pad;
    maxPrice = center + pad;
  }

  const binCount = clampInt(options.bins ?? 48, 8, 120);
  const binWidth = (maxPrice - minPrice) / binCount;
  const volumes = new Array<number>(binCount).fill(0);

  for (const bar of usable) {
    const lo = Math.min(bar.low, bar.high);
    const hi = Math.max(bar.low, bar.high);
    const span = hi - lo;
    if (span <= 0) {
      const idx = clampInt(Math.floor((lo - minPrice) / binWidth), 0, binCount - 1);
      volumes[idx] += bar.volume;
      continue;
    }
    const start = clampInt(Math.floor((lo - minPrice) / binWidth), 0, binCount - 1);
    const end = clampInt(Math.floor((hi - minPrice) / binWidth), 0, binCount - 1);
    for (let i = start; i <= end; i++) {
      const overlap = Math.min(hi, minPrice + (i + 1) * binWidth) - Math.max(lo, minPrice + i * binWidth);
      if (overlap > 0) volumes[i] += bar.volume * (overlap / span);
    }
  }

  const bins: VolumeProfileBin[] = volumes.map((v, i) => {
    const lower = minPrice + i * binWidth;
    return {
      lower,
      upper: lower + binWidth,
      mid: lower + binWidth / 2,
      volume: v,
      share: totalVolume > 0 ? v / totalVolume : 0,
    };
  });

  let pocIdx = 0;
  for (let i = 1; i < bins.length; i++) {
    if (bins[i].volume > bins[pocIdx].volume) pocIdx = i;
  }
  const poc = bins[pocIdx] && bins[pocIdx].volume > 0 ? bins[pocIdx] : null;

  const targetShare = Math.max(0.1, Math.min(0.95, options.valueAreaShare ?? 0.7));
  let valueArea: VolumeProfileValueArea | null = null;
  if (poc) {
    let lowIdx = pocIdx;
    let highIdx = pocIdx;
    let cumulative = bins[pocIdx].volume;
    while (cumulative / totalVolume < targetShare && (lowIdx > 0 || highIdx < bins.length - 1)) {
      const lowerVol = lowIdx > 0 ? bins[lowIdx - 1].volume : -1;
      const upperVol = highIdx < bins.length - 1 ? bins[highIdx + 1].volume : -1;
      if (upperVol > lowerVol) {
        highIdx += 1;
        cumulative += bins[highIdx].volume;
      } else if (lowerVol >= 0) {
        lowIdx -= 1;
        cumulative += bins[lowIdx].volume;
      } else {
        break;
      }
    }
    valueArea = {
      low: bins[lowIdx].lower,
      high: bins[highIdx].upper,
      volumeShare: cumulative / totalVolume,
    };
  }

  const threshold = Math.max(0, Math.min(1, options.hvnThreshold ?? 0.6)) * (poc?.volume ?? 0);
  const hvn = bins
    .filter((bin, i) => {
      if (bin.volume <= 0 || bin.volume < threshold) return false;
      const prev = i > 0 ? bins[i - 1].volume : -1;
      const next = i < bins.length - 1 ? bins[i + 1].volume : -1;
      return bin.volume >= prev && bin.volume >= next;
    })
    .sort((a, b) => b.volume - a.volume)
    .slice(0, 6);

  return {
    bins,
    poc,
    valueArea,
    hvn,
    totalVolume,
    minPrice,
    maxPrice,
    binWidth,
    barCount: usable.length,
    approximation: 'ohlc-range',
  };
}
