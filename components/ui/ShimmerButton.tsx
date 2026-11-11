'use client';

import React from 'react';

interface ShimmerButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  shimmerColor?: string;
  shimmerDuration?: string;
  borderRadius?: string;
  background?: string;
  children: React.ReactNode;
  className?: string;
}

export function ShimmerButton({
  shimmerColor = 'rgba(217, 119, 6, 0.45)', // Warm amber / bronze shimmer
  shimmerDuration = '2.5s',
  borderRadius = '0.5rem',
  background = 'var(--bg-card, #181715)',
  children,
  className = '',
  disabled,
  ...props
}: ShimmerButtonProps) {
  return (
    <button
      style={
        {
          borderRadius,
          background,
        } as React.CSSProperties
      }
      disabled={disabled}
      className={`relative inline-flex items-center justify-center overflow-hidden border border-amber-600/40 dark:border-amber-600/50 px-4 py-2 text-xs font-semibold text-stone-900 dark:text-stone-100 shadow-sm transition-all duration-200 hover:border-amber-500 hover:shadow-md active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none cursor-pointer group ${className}`}
      {...props}
    >
      {/* Moving Shimmer Sweep */}
      {!disabled && (
        <span
          className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          aria-hidden="true"
        >
          <span
            className="absolute -inset-full top-0 block w-1/2 h-full bg-gradient-to-r from-transparent via-amber-400/25 dark:via-amber-400/20 to-transparent skew-x-[-25deg] animate-shimmer-sweep"
            style={{ animationDuration: shimmerDuration }}
          />
        </span>
      )}

      {/* Subtle hairline inner glow */}
      <span
        className="absolute inset-0 rounded-[inherit] pointer-events-none ring-1 ring-inset ring-amber-500/10 dark:ring-amber-400/15"
        aria-hidden="true"
      />

      {/* Button content */}
      <span className="relative z-10 flex items-center gap-2">
        {children}
      </span>
    </button>
  );
}
