'use client';

import React, { useEffect, useRef, useState } from 'react';

/**
 * Aceternity Glowing Effect — a "Cursor"-style border glow that wakes up near the pointer.
 * Attach to any relative-positioned card: renders a hairline conic-gradient border whose
 * bright arc tracks the mouse angle around the element, plus a soft inner spotlight.
 * Idle (no pointer inside), the card keeps a plain hairline border.
 */
export function GlowingEffect({
  borderWidth = 1,
  proximity = 48,
  className = '',
}: {
  borderWidth?: number;
  /** px distance from the element edge at which the glow becomes visible */
  proximity?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [angle, setAngle] = useState(0);
  const [active, setActive] = useState(false);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const parent = el.parentElement;
    if (!parent) return;

    const onMove = (e: PointerEvent) => {
      if (raf.current) return;
      raf.current = requestAnimationFrame(() => {
        raf.current = null;
        const rect = parent.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const deg = (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI + 90;
        setAngle(deg);
        const within =
          e.clientX > rect.left - proximity &&
          e.clientX < rect.right + proximity &&
          e.clientY > rect.top - proximity &&
          e.clientY < rect.bottom + proximity;
        setActive(within);
      });
    };
    const onLeave = () => setActive(false);

    window.addEventListener('pointermove', onMove, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);
    return () => {
      window.removeEventListener('pointermove', onMove);
      document.documentElement.removeEventListener('pointerleave', onLeave);
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [proximity]);

  return (
    <span
      ref={ref}
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 rounded-[inherit] transition-opacity duration-500 ${
        active ? 'opacity-100' : 'opacity-0'
      } ${className}`}
      style={{
        padding: borderWidth,
        background: `conic-gradient(from ${angle}deg, transparent 0deg, transparent 250deg, rgba(246,231,200,0.95) 285deg, rgba(222,192,132,0.75) 305deg, transparent 340deg)`,
        WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
        WebkitMaskComposite: 'xor',
        maskComposite: 'exclude',
      }}
    />
  );
}
