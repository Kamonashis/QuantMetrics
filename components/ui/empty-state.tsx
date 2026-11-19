'use client';

import React from 'react';

/** One empty-state shape, with the "need N, have M" line the vol suite made standard. */
export function EmptyState({
  title,
  detail,
  className = '',
}: {
  title: string;
  detail?: string;
  className?: string;
}) {
  return (
    <div className={`flex min-h-32 flex-col items-center justify-center gap-1.5 rounded-xl border border-[var(--hairline)] bg-[var(--bg-inset)] px-4 py-6 text-center ${className}`}>
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{title}</span>
      {detail && <span className="font-mono text-[10px] text-zinc-400">{detail}</span>}
    </div>
  );
}
