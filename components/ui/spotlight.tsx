'use client';

import React from 'react';
import { motion } from 'motion/react';

/**
 * Aceternity-style "Spotlight" — soft champagne beams sweeping across the hero.
 * Pure radial-gradient motion; no SVG filter cost.
 */
export function Spotlight({ className = '' }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      {/* Left sweep */}
      <motion.div
        initial={{ opacity: 0, x: -120, rotate: -18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1.6, ease: [0.16, 1, 0.3, 1] }}
        className="absolute -top-40 left-[8%] h-[560px] w-[560px] rounded-full"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(222,192,132,0.22) 0%, rgba(222,192,132,0.07) 38%, transparent 70%)',
          filter: 'blur(12px)',
        }}
      />
      {/* Right sweep */}
      <motion.div
        initial={{ opacity: 0, x: 120, rotate: 18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1.6, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="absolute -top-48 right-[4%] h-[640px] w-[640px] rounded-full"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(214,175,100,0.16) 0%, rgba(214,175,100,0.05) 40%, transparent 70%)',
          filter: 'blur(14px)',
        }}
      />
      {/* Center key light */}
      <div
        className="absolute -top-56 left-1/2 h-[520px] w-[900px] -translate-x-1/2"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(255,236,196,0.10) 0%, transparent 65%)',
        }}
      />
    </div>
  );
}
