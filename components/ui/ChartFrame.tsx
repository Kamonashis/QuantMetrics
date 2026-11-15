'use client';

import React from 'react';

/**
 * Reserved-height chart container. `.chart-frame` in globals.css owns the
 * height map (220/280/340/380px at <640/<1024/<1600/≥1600) so the space is
 * reserved BEFORE the chart mounts — no CLS, and no fixed pixel boxes.
 * Pair with `useChartSize()` inside the chart for measured coordinates.
 */
export function ChartFrame({
  className = '',
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  return <div className={`chart-frame w-full ${className}`}>{children}</div>;
}
