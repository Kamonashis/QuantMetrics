import React from 'react';
import { CandlestickChart } from 'lucide-react';

/** Terminal footer — brand, module index, data-source honesty disclosure. */
export function Footer() {
  const modules = ['Stationarity Battery', 'GARCH / HAR-RV', 'Correlation & RMT', 'VaR Backtesting', 'OLS Diagnostics'];
  return (
    <footer className="border-t border-zinc-200/80 dark:border-white/[0.06] bg-[#f5f4f1] dark:bg-[#0a0a0c]">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 sm:px-6 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-gold-500/30 bg-gold-400/10 text-gold-700 dark:text-gold-300">
            <CandlestickChart className="h-4 w-4" strokeWidth={1.8} />
          </span>
          <div className="flex flex-col">
            <span className="font-display text-[13px] font-semibold tracking-tight text-zinc-800 dark:text-zinc-100">
              QuantMetrics <span className="text-zinc-400 dark:text-zinc-500">· Institutional Risk</span>
            </span>
            <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400 dark:text-zinc-500">
              {modules.join('  ·  ')}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-1 text-[10px] leading-relaxed text-zinc-400 dark:text-zinc-500 md:items-end">
          <span className="font-mono uppercase tracking-[0.14em]">
            Market Data: Yahoo Finance · Deterministic Fallback When Unavailable
          </span>
          <span>
            Econometric analytics for research &amp; education. Not investment advice.
          </span>
        </div>
      </div>
    </footer>
  );
}
