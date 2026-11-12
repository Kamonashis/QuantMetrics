'use client';

import { ReactNode, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Maximize2, X } from 'lucide-react';

interface ChartFullscreenProps {
  /** Shown in the full-screen header; also labels the expand button. */
  title?: string;
  /** Optional secondary line in the full-screen header. */
  subtitle?: string;
  /** Extra classes applied to the inline (non-full-screen) wrapper. */
  className?: string;
  children: ReactNode;
}

/**
 * Hover/focus-revealed expand control + in-app full-screen stage for charts.
 * The chart itself is left untouched; in full-screen mode it is re-mounted
 * inside a portal overlay (hover state resets — acceptable trade-off).
 * Close via the ✕ button, Escape, or backdrop click.
 * The expand trigger stays mounted while the overlay is open so focus can
 * return to it deterministically on close.
 */
export function ChartFullscreen({ title, subtitle, className, children }: ChartFullscreenProps) {
  const [open, setOpen] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const trigger = triggerRef.current;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = prevOverflow;
      trigger?.focus();
    };
  }, [open]);

  const heading = title ?? 'Chart';
  const expandLabel = `View ${heading} in full screen`;

  return (
    <>
      {/* Inline slot: children while closed, quiet placeholder while expanded */}
      <div className={`group/chartfs relative ${className ?? ''}`}>
        {open ? (
          <div
            aria-hidden
            className="flex h-48 items-center justify-center rounded-xl border border-dashed border-zinc-200 dark:border-zinc-800 text-[11px] font-mono uppercase tracking-wider text-zinc-400 dark:text-zinc-500"
          >
            Full screen open — Esc or ✕ to close
          </div>
        ) : (
          children
        )}
        <button
          ref={triggerRef}
          type="button"
          onClick={() => setOpen(true)}
          aria-label={expandLabel}
          title={expandLabel}
          className="absolute right-1.5 top-1/2 z-20 inline-flex h-7 w-7 -translate-y-1/2 cursor-pointer items-center justify-center rounded-md border border-zinc-200/80 dark:border-zinc-700/80 bg-white/85 dark:bg-zinc-900/85 text-zinc-500 dark:text-zinc-400 opacity-0 shadow-sm backdrop-blur-sm transition-opacity hover:border-gold-500/60 hover:text-gold-600 dark:hover:text-gold-400 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-gold-500 group-hover/chartfs:opacity-100 max-sm:opacity-100"
        >
          <Maximize2 className="h-3.5 w-3.5" />
        </button>
      </div>
      {open &&
        createPortal(
          <div role="dialog" aria-modal="true" aria-label={`${heading} — full screen view`} className="fixed inset-0 z-[70]">
            {/* Backdrop */}
            <button
              type="button"
              aria-label="Close full screen view"
              onClick={() => setOpen(false)}
              className="absolute inset-0 h-full w-full cursor-zoom-out bg-zinc-950/80 backdrop-blur-sm"
            />
            {/* Stage */}
            <div className="pointer-events-none absolute inset-2 flex sm:inset-4 lg:inset-6">
              <div className="pointer-events-auto m-auto flex h-full w-full max-w-[1700px] flex-col overflow-hidden rounded-2xl border border-gold-500/25 bg-white shadow-2xl dark:bg-[#101014]">
                <header className="flex items-center justify-between gap-4 border-b border-zinc-200/70 dark:border-zinc-800/80 px-5 py-3">
                  <div className="min-w-0">
                    <div className="kicker">Full screen</div>
                    <h2 className="mt-0.5 truncate font-semibold text-sm tracking-tight text-zinc-900 dark:text-zinc-100">
                      {heading}
                    </h2>
                    {subtitle && (
                      <p className="mt-0.5 truncate text-[11px] text-zinc-500 dark:text-zinc-400">{subtitle}</p>
                    )}
                  </div>
                  <button
                    ref={closeRef}
                    type="button"
                    onClick={() => setOpen(false)}
                    aria-label="Close full screen view"
                    title="Close full screen (Esc)"
                    className="inline-flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-500 dark:text-zinc-400 transition-colors hover:border-gold-500/60 hover:text-gold-600 dark:hover:text-gold-400 focus-visible:outline-2 focus-visible:outline-gold-500"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </header>
                <div className="flex min-h-0 flex-1 flex-col overflow-auto px-4 py-4 sm:px-6">
                  <div className="m-auto w-full">{children}</div>
                </div>
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
