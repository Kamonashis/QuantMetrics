'use client';

import { ChevronDown } from 'lucide-react';

export interface AlgoOption {
  value: string;
  label: string;
  /** Optional muted trailing hint, e.g. "MLE · normal" */
  hint?: string;
}

interface AlgoSelectProps {
  /** Kicker-style label above the control, matching the form-field idiom. */
  label?: string;
  value: string;
  onChange: (value: string) => void;
  options: AlgoOption[];
  className?: string;
  disabled?: boolean;
  ariaLabel?: string;
}

/**
 * Shared algorithm picker — one look for every "which model fits this" decision in the app.
 * Native <select> for full keyboard/screen-reader support; gold focus ring like the rest of the form fields.
 */
export function AlgoSelect({ label, value, onChange, options, className, disabled, ariaLabel }: AlgoSelectProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-zinc-600 dark:text-zinc-400 font-medium mb-1 text-xs">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          aria-label={ariaLabel ?? label}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none pl-3 pr-8 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100 focus:outline-hidden focus:ring-1 focus:ring-gold-500 cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
              {o.hint ? ` — ${o.hint}` : ''}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-400" />
      </div>
    </div>
  );
}
