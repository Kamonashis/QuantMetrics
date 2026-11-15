'use client';

import React, { useEffect, useState } from 'react';
import { Sun, Moon, Home, Activity, Sigma, Waves, CandlestickChart, Network, TrendingUp } from 'lucide-react';
import { motion } from 'motion/react';

interface NavbarProps {
  activeView: 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression' | 'volatility' | 'fno';
  onViewChange: (view: 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression' | 'volatility' | 'fno') => void;
  isDark: boolean;
  onToggleTheme: () => void;
  activeTicker: string;
}

const NAV_ITEMS = [
  { id: 'home', label: 'Home', shortLabel: 'Home', icon: Home },
  { id: 'analysis', label: 'Analysis', shortLabel: 'Analysis', icon: Activity },
  { id: 'modeling', label: 'Modeling', shortLabel: 'Model', icon: Sigma },
  { id: 'volatility', label: 'Implied Vol', shortLabel: 'Vol', icon: Waves },
  { id: 'fno', label: 'F&O', shortLabel: 'F&O', icon: CandlestickChart },
  { id: 'correlation', label: 'Correlation', shortLabel: 'Corr.', icon: Network },
  { id: 'regression', label: 'Regression', shortLabel: 'Regr.', icon: TrendingUp },
] as const;

export function Navbar({ activeView, onViewChange, isDark, onToggleTheme, activeTicker }: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const handleNav = (view: NavbarProps['activeView']) => {
    onViewChange(view);
  };

  return (
    <header className="sticky top-0 z-40 w-full px-3 pt-3 sm:px-[clamp(16px,2.2vw,48px)] sm:pt-4 [@media(min-width:1920px)]:mx-auto [@media(min-width:1920px)]:max-w-[2400px]">
      <div
        className={`mx-auto flex h-12 sm:h-14 w-full min-w-0 items-center justify-between gap-3 border backdrop-blur-xl transition-all duration-500 ease-out ${
          scrolled
            ? 'rounded-2xl border-zinc-200/80 bg-[#f5f4f1]/85 px-3 shadow-[0_10px_36px_-18px_rgba(0,0,0,0.35)] dark:border-white/[0.07] dark:bg-[#0e0e11]/80'
            : 'rounded-2xl border-transparent bg-transparent px-1'
        }`}
      >
        {/* Brand */}
        <button
          type="button"
          onClick={() => handleNav('home')}
          className="flex min-h-[44px] shrink-0 cursor-pointer items-center gap-2.5 whitespace-nowrap text-left focus:outline-none"
          aria-label="QuantMetrics home"
        >
          <span className="relative flex h-8 w-8 items-center justify-center rounded-[10px] border border-gold-500/40 bg-gradient-to-b from-gold-400/25 to-gold-600/15 text-gold-700 dark:text-gold-300 shadow-[inset_0_1px_0_rgba(255,255,255,0.35)]">
            <CandlestickChart className="h-4.5 w-4.5 h-[18px] w-[18px]" strokeWidth={1.8} />
          </span>
          <span className="flex flex-col">
            <span className="font-display text-[15px] font-semibold leading-none tracking-tight text-zinc-900 dark:text-zinc-50">
              Quant<span className="text-gradient-gold">Metrics</span>
            </span>
            <span className="mt-1 hidden font-mono text-[9px] font-medium uppercase leading-none tracking-[0.18em] text-zinc-400 dark:text-zinc-500 sm:inline">
              Institutional Risk
            </span>
          </span>
        </button>

        {/* Desktop nav — scrollable chip strip below xl so it never overflows the header */}
        <nav className="hidden min-w-0 items-center gap-0 overflow-x-auto scrollbar-none md:flex xl:gap-0.5" aria-label="Primary">
          {NAV_ITEMS.map((item) => {
            const active = item.id === activeView;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNav(item.id)}
                aria-current={active ? 'page' : undefined}
                className={`relative inline-flex min-h-[44px] shrink-0 cursor-pointer items-center whitespace-nowrap rounded-lg px-2 py-2 text-[11px] font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 lg:px-2.5 xl:px-3.5 xl:text-[12.5px] ${
                  active
                    ? 'text-gold-800 dark:text-gold-200'
                    : 'text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="navbar-active-pill"
                    transition={{ type: 'spring', bounce: 0.16, duration: 0.55 }}
                    className="absolute inset-0 rounded-lg border border-gold-500/30 bg-gold-400/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.18)]"
                  />
                )}
                <span className="relative z-10">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right cluster */}
        <div className="flex shrink-0 items-center gap-2">
          {/* Live asset badge */}
          <div className="hidden items-center gap-2 rounded-lg border border-zinc-200/80 dark:border-white/[0.06] bg-zinc-100/60 dark:bg-white/[0.03] py-1.5 pl-2.5 pr-3 lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--positive)] animate-ping-dot" />
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-400 dark:text-zinc-500">
              Asset
            </span>
            <span className="font-mono text-[11px] font-semibold tabular-nums text-zinc-800 dark:text-zinc-100">
              {activeTicker}
            </span>
          </div>

          {/* Theme toggle */}
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label="Toggle dark and light theme"
            title={isDark ? 'Switch to Paper Light' : 'Switch to Obsidian Dark'}
            className="cursor-pointer rounded-lg border border-zinc-200/80 p-2.5 text-zinc-500 transition-colors hover:border-gold-500/40 hover:text-gold-700 dark:border-white/[0.07] dark:text-zinc-400 dark:hover:text-gold-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Mobile primary nav — <768px bottom tab bar (icons + short labels, safe-area aware) */}
      <nav
        className="fixed inset-x-0 bottom-0 z-40 border-t border-zinc-200/80 bg-[#f5f4f1]/92 backdrop-blur-xl dark:border-white/[0.07] dark:bg-[#0e0e11]/92 md:hidden pb-safe"
        aria-label="Primary"
      >
        <div className="grid grid-cols-7">
          {NAV_ITEMS.map((item) => {
            const active = item.id === activeView;
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleNav(item.id)}
                aria-current={active ? 'page' : undefined}
                className={`flex h-14 min-w-0 cursor-pointer flex-col items-center justify-center gap-0.5 px-0.5 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gold-400/60 ${
                  active
                    ? 'text-gold-800 dark:text-gold-200'
                    : 'text-zinc-500 dark:text-zinc-400'
                }`}
              >
                <Icon className="h-5 w-5" strokeWidth={active ? 2.2 : 1.8} aria-hidden="true" />
                <span className="w-full truncate text-center font-mono text-[8px] font-medium uppercase leading-none tracking-wide">
                  {item.shortLabel}
                </span>
                <span
                  aria-hidden="true"
                  className={`h-0.5 w-6 rounded-full transition-colors ${
                    active ? 'bg-gold-500 dark:bg-gold-300' : 'bg-transparent'
                  }`}
                />
              </button>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
