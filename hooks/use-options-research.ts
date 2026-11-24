'use client';

import { useEffect, useMemo, useState } from 'react';
import type { ChainPayload } from '@/components/views/volatility/types';
import type { buildRndResearch } from '@/lib/rnd-integrity';
import { usMarketDate, vrpObservation, type ResearchObservation } from '@/lib/options-research';
import { curveVerdict } from '@/lib/ticker-snapshot';

export function useOptionsResearch({ chain, research, rv20, loading, showLedger }: {
  chain: ChainPayload | null;
  research: ReturnType<typeof buildRndResearch>;
  rv20: number | null;
  loading: boolean;
  showLedger: boolean;
}) {
  const symbol = chain?.symbol ?? '';
  const key = /^[A-Z0-9][A-Z0-9.-]{0,31}$/.test(symbol) ? symbol : '';
  const [result, setResult] = useState<{ key: string; records: ResearchObservation[]; error: string | null } | null>(null);
  const body = useMemo(() => {
    const asOf = chain?.quote?.asOf;
    if (loading || !asOf || !research || !key) return null;
    const vrp = vrpObservation(chain.iv30 ?? null, rv20, curveVerdict(chain.termStructure ?? []));
    const up = research.probabilities[0];
    const digital = research.checks[0].pass && up.fitted != null ? { expiry: research.slice.expirationDate, svi: up.fitted, atm: up.bs } : null;
    if (!vrp && !digital) return null;
    const mfiv = (chain.slices ?? []).map((s) => ({ expiry: s.expirationDate,
      diffPts: s.atmIv != null && s.mfivAnnualized != null ? 100 * (s.mfivAnnualized - s.atmIv) : null }));
    return JSON.stringify({ asOf, spot: research.spot, quality: research.quality, vrp, digital, mfiv });
  }, [rv20, chain, key, loading, research]);

  useEffect(() => {
    if (!key || (!body && !showLedger)) return;
    const controller = new AbortController();
    const url = `/api/signal-journal?kind=options-research&symbol=${encodeURIComponent(key)}`;
    const read = () => fetch(url, { signal: controller.signal }).then((r) => r.json());
    const run = async () => {
      let error: string | null = null;
      try {
        let response;
        const post = body && usMarketDate(JSON.parse(body).asOf) === usMarketDate(new Date().toISOString());
        if (post) {
          const saved = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: controller.signal }).then((r) => r.json());
          if (saved.available) response = saved;
          else error = saved.reason ?? 'Snapshot was not recorded.';
        }
        if (!response) response = await read();
        if (!controller.signal.aborted) setResult({ key, records: response.available && Array.isArray(response.records) ? response.records : [],
          error: error ?? (!response.available ? response.reason ?? 'Journal unavailable.' : null) });
      } catch {
        if (!controller.signal.aborted) setResult({ key, records: [], error: 'Unable to load the research journal.' });
      }
    };
    const timer = setTimeout(() => { void run(); }, 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [body, key, showLedger]);
  return { symbol: key, records: key && result?.key === key ? result.records : null, error: key && result?.key === key ? result.error : null };
}
