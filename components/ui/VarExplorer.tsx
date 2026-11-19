'use client';

import React, { useState } from 'react';
import { ShieldAlert } from 'lucide-react';
import type { VaRBacktestModelResult } from '@/lib/analysis-engine';

/** One VaR explorer: one model list, one confidence toggle, explicit scope. */
export function VaRExplorer({ models }: { models: VaRBacktestModelResult[] }) {
  const [confidence, setConfidence] = useState<95 | 99>(99);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="kicker flex items-center gap-2"><ShieldAlert className="h-3.5 w-3.5" /> VaR Explorer · Single-Name · 1-Day</p>
        <div className="inline-flex items-center gap-1 rounded-lg border border-[var(--hairline)] bg-[var(--bg-inset)] p-1">
          {([95, 99] as const).map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setConfidence(c)}
              className={`cursor-pointer rounded-md px-2.5 py-1 font-mono text-[10px] font-semibold ${confidence === c ? 'bg-gold-500 text-zinc-950' : 'text-zinc-500'}`}
            >
              {c}%
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="table-sticky-first w-full text-xs text-left">
          <thead>
            <tr className="border-b border-zinc-200 dark:border-zinc-800 text-zinc-400 font-mono">
              <th className="py-2.5 px-3">Model Specification</th>
              <th className="py-2.5 px-3">{confidence}% 1d VaR</th>
              {confidence === 99 && <th className="py-2.5 px-3">99% 1d ES</th>}
              <th className="py-2.5 px-3">Violations (Act/Exp)</th>
              <th className="py-2.5 px-3">Kupiec POF</th>
              <th className="py-2.5 px-3">Independence</th>
              <th className="py-2.5 px-3">Basel Zone</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/60 font-mono">
            {models.map((m) => (
              <tr key={m.key} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/30">
                <td className="py-3 px-3 font-semibold text-zinc-900 dark:text-zinc-100">
                  {m.modelName}
                  <span className="ml-2 font-mono text-[9px] font-normal uppercase tracking-wider text-zinc-400">single-name</span>
                </td>
                <td className="py-3 px-3 font-semibold text-rose-600 dark:text-rose-400">
                  {((confidence === 99 ? m.var99Pct : m.var95Pct) / 100).toFixed(2)}%
                </td>
                {confidence === 99 && <td className="py-3 px-3 font-semibold text-rose-700 dark:text-rose-300">{(m.es99Pct / 100).toFixed(2)}%</td>}
                <td className="py-3 px-3">
                  <span className="font-semibold">{confidence === 99 ? m.exceedances99 : m.exceedances95}</span>
                  {' / '}{(confidence === 99 ? m.expectedExceedances99 : m.expectedExceedances95).toFixed(1)}
                </td>
                <td className="py-3 px-3">
                  LR = {(confidence === 99 ? m.kupiecLr99 : m.kupiecLr95).toFixed(2)}
                  <div className="text-[11px] text-zinc-400">
                    p = {(confidence === 99 ? m.kupiecPVal99 : m.kupiecPVal95).toFixed(3)}{' '}
                    <span className={(confidence === 99 ? m.kupiecVerdict99 : m.kupiecVerdict95) === 'Pass' ? 'text-gold-500 font-medium' : 'text-rose-500 font-semibold'}>
                      [{(confidence === 99 ? m.kupiecVerdict99 : m.kupiecVerdict95) === 'Pass' ? 'Pass' : 'Fail'}]
                    </span>
                  </div>
                </td>
                <td className="py-3 px-3">
                  LR = {m.christoffersenLr99.toFixed(2)}
                  <div className="text-[11px] text-zinc-400">[{m.christoffersenVerdict99.startsWith('Pass') ? 'Independent' : 'Clustered'}]</div>
                </td>
                <td className="py-3 px-3">
                  <span className={`rounded px-2 py-0.5 font-semibold ${m.baselZone === 'Red' ? 'bg-rose-100 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300' : 'bg-gold-100 dark:bg-gold-950/40 text-gold-700 dark:text-gold-300'}`}>
                    {m.baselZone}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 font-mono text-[9px] leading-relaxed text-zinc-400">
        One model list, one confidence toggle, one scope (single-name, 1-day loss magnitude). The EVT row is the same POT-GPD estimator shown in Stationarity — it is not a second, independent model. Portfolio VaR lives in Correlation → Risk with its own scope.
      </p>
    </div>
  );
}
