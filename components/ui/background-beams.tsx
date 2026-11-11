'use client';

import React from 'react';
import { motion } from 'motion/react';

/**
 * Aceternity "Background Beams" — luminous SVG paths falling through the hero.
 * Two beam gradients, lightly staggered draw-in + infinite soft pulse.
 */
export function BackgroundBeams({ className = '' }: { className?: string }) {
  const paths = [
    'M-380 -189C-380 -189 -312 216 152 343C616 470 684 875 684 875',
    'M-373 -197C-373 -197 -305 208 159 335C623 462 691 867 691 867',
    'M-366 -205C-366 -205 -298 200 166 327C630 454 698 859 698 859',
    'M-359 -213C-359 -213 -291 192 173 319C637 446 705 851 705 851',
    'M-352 -221C-352 -221 -284 184 180 311C644 438 712 843 712 843',
    'M-345 -229C-345 -229 -277 176 187 303C651 430 719 835 719 835',
    'M-338 -237C-338 -237 -270 168 194 295C658 422 726 827 726 827',
  ];

  return (
    <div aria-hidden="true" className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`}>
      <svg
        className="absolute h-full w-full"
        width="100%"
        height="100%"
        viewBox="0 0 696 316"
        fill="none"
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <linearGradient id="beam-gold" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#DEC084" stopOpacity="0" />
            <stop offset="32%" stopColor="#DEC084" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#DEC084" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="beam-ice" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#E8EAF2" stopOpacity="0" />
            <stop offset="35%" stopColor="#C9CEDA" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#C9CEDA" stopOpacity="0" />
          </linearGradient>
        </defs>

        {paths.map((d, i) => (
          <motion.path
            key={`halo-${i}`}
            d={d}
            stroke={i % 3 === 2 ? 'url(#beam-ice)' : 'url(#beam-gold)'}
            strokeOpacity={0.35}
            strokeWidth={3.2}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: [0, 0.4, 0.28] }}
            transition={{ duration: 2.4 + i * 0.22, ease: 'easeInOut' }}
          />
        ))}
        {paths.map((d, i) => (
          <motion.path
            key={`core-${i}`}
            d={d}
            stroke={i % 3 === 2 ? 'url(#beam-ice)' : 'url(#beam-gold)'}
            strokeWidth={0.9}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: [0, 0.9, 0.6] }}
            transition={{ duration: 2.1 + i * 0.22, ease: 'easeInOut' }}
          />
        ))}
      </svg>
      {/* Bottom fade into page background */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#08080A] via-[#08080A]/25 to-transparent" />
    </div>
  );
}
