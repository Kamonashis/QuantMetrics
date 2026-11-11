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

/**
 * Primary action button — obsidian pill with a champagne shimmer sweep on hover.
 * Identical in both themes (like the hero CTA), so the primary action always reads as "the button".
 */
export function ShimmerButton({
  shimmerColor = 'rgba(222, 192, 132, 0.4)',
  shimmerDuration = '2.5s',
  borderRadius = '0.75rem',
  background = '#141310',
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
      className={`relative inline-flex items-center justify-center overflow-hidden border border-gold-600/50 px-4 py-2 text-xs font-semibold text-gold-200 shadow-sm transition-all duration-200 hover:border-gold-400/70 hover:shadow-[0_8px_28px_-10px_rgba(222,192,132,0.45)] active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none cursor-pointer group focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/70 ${className}`}
      {...props}
    >
      {/* Moving shimmer sweep */}
      {!disabled && (
        <span
          className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300"
          aria-hidden="true"
        >
          <span
            className="absolute -inset-full top-0 block w-1/2 h-full bg-gradient-to-r from-transparent via-gold-300/30 to-transparent skew-x-[-25deg] animate-shimmer-sweep"
            style={{ animationDuration: shimmerDuration }}
          />
        </span>
      )}

      {/* Subtle hairline inner glow */}
      <span
        className="absolute inset-0 rounded-[inherit] pointer-events-none ring-1 ring-inset ring-gold-400/15"
        aria-hidden="true"
      />

      {/* Content */}
      <span className="relative z-10 flex items-center gap-2">
        {children}
      </span>
    </button>
  );
}
