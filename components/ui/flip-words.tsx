'use client';

import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';

/** Aceternity Flip Words — vertical word rotation for hero emphasis. */
export function FlipWords({
  words,
  duration = 2600,
  className = '',
}: {
  words: string[];
  duration?: number;
  className?: string;
}) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setIndex((prev) => (prev + 1) % words.length);
    }, duration);
    return () => clearInterval(id);
  }, [words.length, duration]);

  return (
    <span className={`relative inline-flex overflow-hidden align-baseline ${className}`}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={index}
          initial={{ y: '110%', opacity: 0 }}
          animate={{ y: '0%', opacity: 1 }}
          exit={{ y: '-110%', opacity: 0 }}
          transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
          className="inline-block text-gradient-gold font-semibold"
        >
          {words[index]}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}
