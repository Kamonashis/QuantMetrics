'use client';

import React from 'react';
import { Check, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export type ActionState = 'idle' | 'loading' | 'success';

/**
 * Aceternity Stateful Button — shows progress and confirmation inline.
 * Views drive the state machine; the button renders the right affordance.
 */
export function StatefulButton({
  state,
  onClick,
  children,
  loadingLabel = 'Computing',
  successLabel = 'Complete',
  className = '',
  disabled,
}: {
  state: ActionState;
  onClick?: () => void;
  children: React.ReactNode;
  loadingLabel?: string;
  successLabel?: string;
  className?: string;
  disabled?: boolean;
}) {
  const busy = state === 'loading';
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      className={`relative inline-flex h-9 cursor-pointer items-center justify-center gap-2 rounded-lg border border-gold-600/40 bg-gold-500/10 px-4 text-xs font-semibold text-gold-800 dark:text-gold-200 transition-all duration-200 hover:border-gold-500/70 hover:bg-gold-500/15 hover:shadow-[0_0_24px_-8px_var(--gold-glow)] active:scale-[0.98] disabled:cursor-default disabled:opacity-80 ${className}`}
    >
      <AnimatePresence mode="wait" initial={false}>
        {state === 'loading' && (
          <motion.span
            key="loading"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="flex items-center gap-2"
          >
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{loadingLabel}…</span>
          </motion.span>
        )}
        {state === 'success' && (
          <motion.span
            key="success"
            initial={{ opacity: 0, scale: 0.6 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2"
          >
            <Check className="h-3.5 w-3.5" />
            <span>{successLabel}</span>
          </motion.span>
        )}
        {state === 'idle' && (
          <motion.span
            key="idle"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="flex items-center gap-2"
          >
            {children}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
