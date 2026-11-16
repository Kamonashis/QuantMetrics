'use client';

import React from 'react';
import { FlaskConical } from 'lucide-react';

type Verdict = 'arithmetic' | 'estimate' | 'hypothesis';

interface Reconciliation {
  id: string;
  claim: string;
  evidence: string;
  verdict: Verdict;
}

const ROWS: Reconciliation[] = [
  {
    id: 'ann-vol-vs-iv30',
    claim: '“IV < ANN VOL, options cheap” — headline trap',
    evidence: '250-day blended realized vs 30cd implied are not comparable. In this app: RV windows vs IV30 only, plus the RV²/IV² ratio tile. The 250-day figure appears nowhere on this tab.',
    verdict: 'arithmetic',
  },
  {
    id: 'basis-rescale',
    claim: 'The “RV on calendar basis / IV on trading basis” rescale',
    evidence: '√(252/365)·RV and √(365/252)·IV are inconsistent with each other (gap would be 9.7 pts one way, 11.8 the other) — neither is a normalization. RV and IV annualize the same total variance; compared directly by convention. Nothing is ever rescaled.',
    verdict: 'arithmetic',
  },
  {
    id: 'rv-vs-iv-sign',
    claim: 'Direction of the RV–IV gap',
    evidence: 'RV richer than IV: RV30 30.3% vs IV30 28.6% (1.7 pts), span-consistent variance ratio 1.4×. The naive screen had the sign right; the basis was the error, not the direction.',
    verdict: 'arithmetic',
  },
  {
    id: 'signature-noise-floor',
    claim: '“≈31% of 1-minute variance is bid-ask bounce”',
    evidence: '40.50² vs plateau 33.6² → ≥31% of 1-min variance is microstructure noise. Lower bound by construction — the plateau is not noise-free. Do not size off 1-min RV.',
    verdict: 'estimate',
  },
  {
    id: 'jumps-vs-kurtosis',
    claim: '“Tails are vol-of-vol, not jumps”',
    evidence: 'Jump share ≈ 9.1% of RV cannot generate κ = 10.2 alone; BNS found 0/7 significant days — but n=7 days is power-starved. Direction defensible, precise % is not an estimate anyone can defend.',
    verdict: 'estimate',
  },
  {
    id: 'hurst-artifact-test',
    claim: '“Hurst 0.652 ⇒ trending” — possibly a clustering artifact',
    evidence: 'Persistent GARCH with leverage produces H ≈ 0.6 in pure simulation. Decided only by H on σ̂-standardized innovations (GARCH filter): Ĥ ≈ 0.5 ⇒ artifact, drop momentum tilts; Ĥ ≳ 0.6 ⇒ dependence survives. See the toggle on Analysis → Overview and the σ̂ readout above the checklist.',
    verdict: 'hypothesis',
  },
  {
    id: 'overnight-vs-gamma',
    claim: '“Asymmetry is flow, not news”',
    evidence: 'Only 18.5% of variance arrives overnight and 0/7 days jumped — but that is inference, not measurement. The open→close GJR refit (Dynamics tab) settles it: γ survives gap removal ⇒ intraday flow; γ collapses ⇒ event/gap phenomenon. Watch Δγ.',
    verdict: 'hypothesis',
  },
  {
    id: 'multiple-comparisons',
    claim: 'Some “findings” on these panels are noise',
    evidence: 'n ≈ 250 daily returns: skew SE ≈ √(6/n) ≈ 0.155, kurtosis SE ≈ √(24/n) ≈ 0.31, Hurst SE ≈ 0.063. Across a multi-panel terminal with a dozen tests, uncorrected p-values flatter everything. The arithmetic-backed rows stand; everything else is a hypothesis until refit on a fresh sample.',
    verdict: 'arithmetic',
  },
];

const VERDICT_STYLE: Record<Verdict, string> = {
  arithmetic: 'border-gold-500/50 bg-gold-500/10 text-gold-800 dark:text-gold-200',
  estimate: 'border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300',
  hypothesis: 'border-zinc-300 dark:border-zinc-700 bg-[var(--bg-inset)] text-zinc-500 dark:text-zinc-400',
};

/**
 * Reconciliation log — which claim rests on which test, and how much that test knows.
 * Not a backtest and not a ledger of truth: a triage of what this terminal can and cannot certify.
 */
export function ReconciliationLog({ n }: { n: number }) {
  const seSkew = Math.sqrt(6 / Math.max(40, n));
  const seKurt = Math.sqrt(24 / Math.max(40, n));
  const seHurst = 0.063 * Math.sqrt(250 / Math.max(40, n));

  return (
    <details className="panel p-5">
      <summary className="cursor-pointer select-none">
        <span className="kicker flex items-center gap-2">
          <FlaskConical className="h-3.5 w-3.5" />
          Reconciliation log — which conclusion rests on which test
          <span className="font-mono text-[9px] normal-case tracking-normal text-zinc-400">
            n = {n} · skew SE {seSkew.toFixed(3)} · kurt SE {seKurt.toFixed(3)} · H SE {seHurst.toFixed(3)}
          </span>
        </span>
      </summary>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[820px] border-collapse text-left font-mono text-[10px]">
          <thead>
            <tr className="border-b border-[var(--hairline)] text-[8.5px] uppercase tracking-[0.18em] text-zinc-400">
              <th className="py-1.5 pr-3 font-semibold">Claim (as asserted)</th>
              <th className="py-1.5 pr-3 font-semibold">What the data actually supports</th>
              <th className="py-1.5 font-semibold">Standing</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--hairline)]">
            {ROWS.map((r) => (
              <tr key={r.id} className="align-top">
                <td className="py-2 pr-3 text-zinc-700 dark:text-zinc-300">{r.claim}</td>
                <td className="py-2 pr-3 leading-snug text-zinc-500 dark:text-zinc-400">{r.evidence}</td>
                <td className="py-2">
                  <span className={`inline-flex rounded border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider ${VERDICT_STYLE[r.verdict]}`}>
                    {r.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
