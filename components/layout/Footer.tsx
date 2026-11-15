import React from 'react';
import { CandlestickChart } from 'lucide-react';

/** Terminal footer — brand, module index, data-source honesty disclosure. */
export function Footer() {
  const modules = ['Stationarity Battery', 'GARCH / HAR-RV', 'Correlation & RMT', 'VaR Backtesting', 'OLS Diagnostics'];
  return (
    <footer className="border-t border-zinc-200/80 dark:border-white/[0.06] bg-[#f5f4f1] dark:bg-[#0a0a0c]">
      <div className="mx-auto flex w-full flex-col gap-3 px-[clamp(16px,2.2vw,48px)] py-5 sm:flex-row sm:items-center sm:justify-between sm:gap-6 sm:py-6 [@media(min-width:1920px)]:max-w-[2400px]">
        <div className="flex items-center gap-3 min-w-0">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-gold-500/30 bg-gold-400/10 text-gold-700 dark:text-gold-300">
            <CandlestickChart className="h-4 w-4" strokeWidth={1.8} />
          </span>
          <div className="flex min-w-0 flex-col">
            <span className="font-display text-[13px] font-semibold tracking-tight text-zinc-800 dark:text-zinc-100">
              QuantMetrics <span className="text-zinc-400 dark:text-zinc-500">· Institutional Risk</span>
            </span>
            <span className="truncate font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">
              {modules.join('  ·  ')}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-1 border-t border-zinc-100 pt-3 text-[10px] leading-relaxed text-zinc-400 dark:border-white/[0.05] dark:text-zinc-500 sm:border-t-0 sm:pt-0 sm:items-end sm:text-right">
          <span className="truncate font-mono uppercase tracking-[0.14em]">
            Market Data: Yahoo Finance · Deterministic Fallback
          </span>
          <span>
            Econometric analytics for research &amp; education. Not investment advice.
          </span>
        </div>
      </div>
    </footer>
  );
}
