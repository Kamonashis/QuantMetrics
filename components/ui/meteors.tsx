'use client';

import React, { useMemo } from 'react';
import { motion } from 'motion/react';

/** Seeded PRNG so meteor layouts are stable across renders and SSR/CSR match. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Aceternity Meteors — sparse gold meteor trails for CTA strips. */
export function Meteors({ count = 14, className = '' }: { count?: number; className?: string }) {
  const meteors = useMemo(() => {
    const rand = mulberry32(1337);
    return Array.from({ length: count }, () => ({
      left: `${rand() * 100}%`,
      delay: rand() * 1.6 + 0.1,
      duration: rand() * 6 + 4,
      size: rand() * 1.4 + 0.9,
    }));
  }, [count]);

  return (
    <div aria-hidden="true" className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      {meteors.map((m, i) => (
        <motion.span
          key={i}
          className="absolute top-[-70px] h-px rounded-full"
          style={{
            left: m.left,
            width: `${m.size * 80}px`,
            background: 'linear-gradient(90deg, transparent, #DEC084, transparent)',
          }}
          initial={{ y: 0, x: 0, opacity: 0 }}
          animate={{ y: 480, x: -200, opacity: [0, 1, 1, 0] }}
          transition={{ duration: m.duration, delay: m.delay, repeat: Infinity, repeatDelay: 2.4, ease: 'easeIn' }}
        >
          <span
            className="absolute right-0 top-1/2 h-[3px] w-[3px] -translate-y-1/2 rounded-full"
            style={{ background: '#F3E2B8', boxShadow: '0 0 6px 1px rgba(222,192,132,0.9)' }}
          />
        </motion.span>
      ))}
    </div>
  );
}
