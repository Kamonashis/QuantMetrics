import React from 'react';
import { TrendingDown, TrendingUp } from 'lucide-react';

export interface TapeItem {
  symbol: string;
  price: number;
  changePct: number;
}

/**
 * Infinite ticker tape — duplicated list scrolled via CSS keyframes
 * (.animate-marquee), edge-faded with gradient masks.
 */
export function TickerTape({ items }: { items: TapeItem[] }) {
  const doubled = [...items, ...items];
  return (
    <div
      className="marquee-paused relative overflow-hidden border-y border-zinc-200/80 dark:border-white/[0.06] bg-white/60 dark:bg-white/[0.02] py-1.5 backdrop-blur-sm"
      aria-hidden="true"
    >
      {/* Edge fades */}
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-16 bg-gradient-to-r from-[var(--bg-base)] to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-16 bg-gradient-to-l from-[var(--bg-base)] to-transparent" />

      <div className="animate-marquee flex w-max items-center whitespace-nowrap">
        {doubled.map((item, i) => {
          const up = item.changePct >= 0;
          return (
            <span key={`${item.symbol}-${i}`} className="mx-5 flex items-center gap-2 font-mono text-[11px]">
              <span className="font-semibold tracking-wider text-zinc-700 dark:text-zinc-300">{item.symbol}</span>
              <span className="tabular-nums text-zinc-900 dark:text-zinc-100">
                {item.price.toFixed(2)}
              </span>
              <span
                className={`flex items-center gap-0.5 tabular-nums ${
                  up ? 'text-[var(--positive)]' : 'text-[var(--negative)]'
                }`}
              >
                {up ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {up ? '+' : ''}
                {item.changePct.toFixed(2)}%
              </span>
              <span className="ml-4 text-[8px] text-zinc-300 dark:text-zinc-700">◆</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}
