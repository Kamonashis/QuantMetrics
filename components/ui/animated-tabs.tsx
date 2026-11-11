'use client';

import React from 'react';
import { motion } from 'motion/react';
import type { LucideIcon } from 'lucide-react';

export interface TabItem {
  id: string;
  label: string;
  icon?: LucideIcon;
  meta?: string;
}

/**
 * Aceternity Animated Tabs — sliding gold "ink pill" under the active tab.
 * Used for the analysis battery tabs, correlation tiers, and modeling mode switch.
 */
export function AnimatedTabs({
  tabs,
  value,
  onChange,
  className = '',
  size = 'md',
}: {
  tabs: TabItem[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
  size?: 'sm' | 'md';
}) {
  const uid = React.useId();
  return (
    <div
      role="tablist"
      aria-label="Section tabs"
      className={`inline-flex items-center gap-1 rounded-xl border border-zinc-200 dark:border-white/[0.06] bg-zinc-100/70 dark:bg-white/[0.03] p-1 ${className}`}
    >
      {tabs.map((tab) => {
        const active = tab.id === value;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(tab.id)}
            className={`relative cursor-pointer whitespace-nowrap rounded-lg font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 ${
              size === 'sm' ? 'px-2.5 py-1.5 text-[11px]' : 'px-3.5 py-2 text-xs'
            } ${
              active
                ? 'text-gold-900 dark:text-[#141310]'
                : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-100'
            }`}
          >
            {active && (
              <motion.span
                layoutId={`${uid}-pill`}
                className="absolute inset-0 rounded-lg bg-gradient-to-b from-gold-200 to-gold-400 shadow-[0_2px_10px_-2px_rgba(222,192,132,0.55)]"
                transition={{ type: 'spring', bounce: 0.18, duration: 0.55 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-1.5">
              {Icon && <Icon className={size === 'sm' ? 'h-3 w-3' : 'h-3.5 w-3.5'} strokeWidth={2.2} />}
              {tab.label}
              {tab.meta && (
                <span
                  className={`rounded px-1 py-0.5 font-mono text-[9px] tracking-wide ${
                    active ? 'bg-black/15 text-[#141310]' : 'bg-zinc-200/70 dark:bg-white/[0.06] text-zinc-500 dark:text-zinc-500'
                  }`}
                >
                  {tab.meta}
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}
