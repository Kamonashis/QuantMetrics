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
  panelId,
}: {
  tabs: TabItem[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
  size?: 'sm' | 'md';
  /** Optional id of the tabpanel this list controls. */
  panelId?: string;
}) {
  const uid = React.useId();
  const listRef = React.useRef<HTMLDivElement>(null);
  const tabRefs = React.useRef<(HTMLButtonElement | null)[]>([]);

  // WAI-ARIA tab semantics: one tab stop, Left/Right/Home/End move and activate.
  const moveTo = (index: number) => {
    const i = (index + tabs.length) % tabs.length;
    onChange(tabs[i].id);
    tabRefs.current[i]?.focus();
  };
  const onTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (e.key === 'ArrowRight') { e.preventDefault(); moveTo(index + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); moveTo(index - 1); }
    else if (e.key === 'Home') { e.preventDefault(); moveTo(0); }
    else if (e.key === 'End') { e.preventDefault(); moveTo(tabs.length - 1); }
  };

  // Keep the active tab in view when it changes (esp. after deep-link restore)
  React.useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]');
    el?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
  }, [value]);

  return (
    <div
      ref={listRef}
      role="tablist"
      aria-label="Section tabs"
      className={`scrollbar-none edge-fade-x inline-flex max-w-full items-center gap-1 snap-x snap-mandatory overflow-x-auto rounded-xl border border-zinc-200 dark:border-white/[0.06] bg-zinc-100/70 dark:bg-white/[0.03] p-1 ${className}`}
    >
      {tabs.map((tab, index) => {
        const active = tab.id === value;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            ref={(el) => { tabRefs.current[index] = el; }}
            role="tab"
            aria-selected={active}
            aria-controls={panelId}
            tabIndex={active ? 0 : -1}
            id={`${uid}-tab-${tab.id}`}
            onKeyDown={(e) => onTabKeyDown(e, index)}
            onClick={() => onChange(tab.id)}
            className={`relative shrink-0 snap-start cursor-pointer whitespace-nowrap rounded-lg font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/60 ${
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
