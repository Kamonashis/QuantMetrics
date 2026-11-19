'use client';

import React from 'react';
import { MotionConfig } from 'motion/react';

/** Honour the OS reduced-motion preference for every motion/react animation. */
export function MotionReducer({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
