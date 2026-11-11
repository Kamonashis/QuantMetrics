'use client';

import React from 'react';
import { motion } from 'motion/react';

/** Aceternity Text Generate Effect — words fade/blur in sequentially. */
export function TextGenerateEffect({
  words,
  className = '',
  as: Tag = 'span',
}: {
  words: string;
  className?: string;
  as?: 'span' | 'h1' | 'h2';
}) {
  const tokens = words.split(' ');
  return (
    <Tag className={className}>
      {tokens.map((word, i) => (
        <motion.span
          key={`${word}-${i}`}
          initial={{ opacity: 0, filter: 'blur(8px)', y: 8 }}
          animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 + i * 0.055, ease: 'easeOut' }}
          className="inline-block whitespace-pre"
        >
          {word}{' '}
        </motion.span>
      ))}
    </Tag>
  );
}
