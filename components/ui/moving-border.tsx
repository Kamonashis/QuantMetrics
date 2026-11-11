'use client';

import React from 'react';

/**
 * Aceternity Moving Border — a comet of light travels the button's perimeter.
 * Implemented with a CSS @property-animated conic gradient that spans the full border box,
 * so it works for wide pills as well as square controls.
 */
export function MovingBorderButton({
  children,
  onClick,
  className = '',
  duration = 2.6,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  duration?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{ ['--border-spin-duration' as string]: `${duration}s` }}
      className={`group relative inline-flex h-11 cursor-pointer items-center justify-center rounded-xl p-[1.5px] focus:outline-none focus-visible:ring-2 focus-visible:ring-gold-400/70 ${className}`}
      {...rest}
    >
      {/* Rotating comet along the perimeter */}
      <span
        aria-hidden="true"
        className="moving-border-comet pointer-events-none absolute inset-0 rounded-xl"
      />
      {/* Opaque inner pill */}
      <span className="relative z-10 flex h-full w-full items-center justify-center gap-2 rounded-[10px] bg-[#141310] px-6 text-[13px] font-semibold tracking-wide text-gold-200 transition-colors duration-200 group-hover:text-gold-100">
        {children}
      </span>
    </button>
  );
}
