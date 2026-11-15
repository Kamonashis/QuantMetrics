'use client';

import { useCallback, useEffect, useState } from 'react';

export const CHART_FRAME_HEIGHTS = { base: 220, sm: 280, lg: 340, xl: 380 } as const;

/**
 * Reserved chart heights by breakpoint — MIRRORED by `.chart-frame` in
 * globals.css (single source is the CSS; this map documents the values and
 * lets non-CSS consumers reason about them).
 */
export function chartHeightForWidth(width: number): number {
  if (width < 640) return CHART_FRAME_HEIGHTS.base;
  if (width < 1024) return CHART_FRAME_HEIGHTS.sm;
  if (width < 1600) return CHART_FRAME_HEIGHTS.lg;
  return CHART_FRAME_HEIGHTS.xl;
}

export interface ChartSize {
  width: number;
  height: number;
}

/**
 * ResizeObserver-driven container sizer for hand-rolled SVG/Canvas charts.
 * - Measures the content box (excludes padding) so charts can use measured
 *   pixel coordinates instead of a fixed viewBox.
 * - Height comes from the `.chart-frame` CSS map; width from the layout.
 * - Updates are trailing-debounced (~100ms) so drags don't remount charts;
 *   the first measurement is synchronous on mount.
 * - Callback-ref based: charts that mount their frame late (e.g. after an
 *   empty-state branch flips) still get measured — a bare mount-time
 *   useEffect would silently keep size at 0×0 forever in that case.
 */
export function useChartSize<T extends HTMLElement = HTMLDivElement>() {
  const [node, setNode] = useState<T | null>(null);
  const ref = useCallback((el: T | null) => setNode(el), []);
  const [size, setSize] = useState<ChartSize>({ width: 0, height: 0 });

  useEffect(() => {
    const el = node;
    if (!el) return;

    const apply = (width: number, height: number) => {
      const next = {
        width: Math.max(0, Math.floor(width)),
        height: Math.max(0, Math.floor(height)),
      };
      setSize((prev) =>
        prev.width === next.width && prev.height === next.height ? prev : next
      );
    };

    // Synchronous first measure (avoid skeleton flash on mount)
    const first = el.getBoundingClientRect();
    apply(first.width, first.height);

    let timer: ReturnType<typeof setTimeout> | null = null;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => apply(width, height), 100);
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      if (timer) clearTimeout(timer);
    };
  }, [node]);

  return { ref, size };
}
