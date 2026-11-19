'use client';

import React, { useCallback, useState } from 'react';

export interface ChartTooltipState {
  x: number;
  y: number;
  lines: string[];
}

/** One tooltip primitive for every chart — styled, multi-line, and touch-friendly. */
export function ChartTooltip({ state }: { state: ChartTooltipState | null }) {
  if (!state) return null;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 max-w-[240px] rounded-md border border-gold-500/30 bg-[var(--bg-panel)] px-2 py-1.5 font-mono text-[10px] leading-relaxed text-zinc-700 shadow-lg dark:text-zinc-200"
      style={{ left: Math.max(4, state.x + 10), top: Math.max(4, state.y - 8) }}
    >
      {state.lines.map((line, i) => <div key={i} className={i === 0 ? 'font-semibold text-gold-700 dark:text-gold-300' : ''}>{line}</div>)}
    </div>
  );
}

export function useChartTooltip() {
  const [tooltipState, setTooltipState] = useState<ChartTooltipState | null>(null);
  const showTooltip = useCallback((e: React.MouseEvent<Element>, lines: string[]) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltipState({ x: e.clientX - rect.left, y: e.clientY - rect.top, lines });
  }, []);
  const hideTooltip = useCallback(() => setTooltipState(null), []);
  return { tooltipState, showTooltip, hideTooltip };
}
