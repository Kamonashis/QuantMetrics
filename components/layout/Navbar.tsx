'use client';

import React from 'react';
import { Sun, Moon, TrendingUp } from 'lucide-react';

interface NavbarProps {
  activeView: 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression';
  onViewChange: (view: 'home' | 'analysis' | 'modeling' | 'correlation' | 'regression') => void;
  isDark: boolean;
  onToggleTheme: () => void;
  activeTicker: string;
}

export function Navbar({
  activeView,
  onViewChange,
  isDark,
  onToggleTheme,
  activeTicker,
}: NavbarProps) {
  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'analysis', label: 'Analysis' },
    { id: 'modeling', label: 'Modeling' },
    { id: 'correlation', label: 'Correlation' },
    { id: 'regression', label: 'Regression' },
  ] as const;

  return (
    <header className="sticky top-0 z-40 w-full bg-[#faf8f5]/95 dark:bg-[#121110]/95 backdrop-blur-md border-b border-stone-200 dark:border-stone-800/80 transition-colors">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4 px-4 sm:px-6 h-14">
        {/* Brand wordmark */}
        <button
          type="button"
          onClick={() => onViewChange('home')}
          className="flex items-center gap-2.5 text-left text-stone-900 dark:text-stone-100 whitespace-nowrap shrink-0 group focus:outline-hidden cursor-pointer"
        >
          <div className="w-7 h-7 rounded-md bg-amber-500/15 dark:bg-amber-500/20 text-amber-700 dark:text-amber-400 flex items-center justify-center font-bold text-sm border border-amber-600/30 group-hover:border-amber-500 transition-colors">
            <TrendingUp className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-sm tracking-tight text-stone-900 dark:text-stone-100">
              QuantTerminal
            </span>
            <span className="text-[10px] font-mono text-stone-500 dark:text-stone-400 -mt-0.5 hidden sm:inline">
              Institutional Risk
            </span>
          </div>
        </button>

        {/* Clean nav links */}
        <nav className="flex items-center gap-1 sm:gap-2 overflow-x-auto py-1">
          {navItems.map((item) => {
            const isActive = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onViewChange(item.id)}
                className={`text-xs font-semibold transition-all whitespace-nowrap shrink-0 px-3 py-1.5 rounded-md cursor-pointer ${
                  isActive
                    ? 'text-amber-800 dark:text-amber-300 bg-amber-100/60 dark:bg-amber-950/40 border border-amber-500/30 shadow-xs'
                    : 'text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-100 hover:bg-stone-200/50 dark:hover:bg-stone-800/40 border border-transparent'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Action Controls & Design System Trigger */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden lg:inline-block font-mono text-[11px] text-stone-500 dark:text-stone-400 tabular-nums">
            Asset: <strong className="text-amber-700 dark:text-amber-300">{activeTicker}</strong>
          </span>

          {/* Theme Switcher */}
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label="Toggle Dark and Light theme"
            className="p-1.5 rounded-md border border-stone-200 dark:border-stone-800 hover:bg-stone-200 dark:hover:bg-stone-800 text-stone-600 dark:text-stone-400 hover:text-stone-900 dark:hover:text-stone-100 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            title={isDark ? 'Switch to Parchment Light Theme' : 'Switch to Charcoal Dark Theme'}
          >
            {isDark ? (
              <Sun className="w-4 h-4 text-amber-400" />
            ) : (
              <Moon className="w-4 h-4 text-stone-700" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
