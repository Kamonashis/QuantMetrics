'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { HOME_UNIVERSE_SYMBOLS } from '@/lib/home-universe';
import type { FundamentalSnapshot } from '@/lib/factors/fundamental-factors';
import type { AnalystSnapshot, OptionsFactorSnapshot } from '@/lib/factors/sentiment-options-factors';

export interface FactorCoverage {
  available: boolean;
  reason?: string;
  sessions: number;
  firstDate: string | null;
  lastDate: string | null;
  degraded: boolean;
  sector: string | null;
}

export interface FactorDataState {
  fundamentalsBySymbol: Record<string, FundamentalSnapshot[]>;
  analystBySymbol: Record<string, AnalystSnapshot | null>;
  optionsBySymbol: Record<string, OptionsFactorSnapshot[]>;
  sectors: Record<string, string>;
  coverage: Record<string, FactorCoverage>;
  isLoading: boolean;
  refresh: () => void;
}

const FALLBACK_SECTORS: Record<string, string> = Object.fromEntries(
  HOME_UNIVERSE_SYMBOLS.map((s) => [s.ticker, s.sector])
);

const EMPTY_STATE: FactorDataState = {
  fundamentalsBySymbol: {},
  analystBySymbol: {},
  optionsBySymbol: {},
  sectors: { ...FALLBACK_SECTORS },
  coverage: {},
  isLoading: false,
  refresh: () => undefined,
};

interface FundamentalsResponse {
  success?: boolean;
  data?: Record<
    string,
    {
      available?: boolean;
      reason?: string;
      sessions?: number;
      firstDate?: string | null;
      lastDate?: string | null;
      degraded?: boolean;
      sector?: string | null;
      snapshot?: FundamentalSnapshot | null;
      analyst?: AnalystSnapshot | null;
    }
  >;
}

interface OptionsScanResponse {
  data?: Record<string, Partial<OptionsFactorSnapshot> & { available?: boolean; reason?: string }>;
}

/** Loads fundamentals / analyst / options factor inputs for a symbol set. */
export function useFactorData(symbols: readonly string[], enabled = true): FactorDataState {
  const symbolsKey = useMemo(
    () => [...new Set(symbols.map((s) => s.trim().toUpperCase()).filter(Boolean))].slice(0, 10).join(','),
    [symbols]
  );
  const [state, setState] = useState<FactorDataState>(EMPTY_STATE);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!enabled || !symbolsKey) return;
    let cancelled = false;
    const controller = new AbortController();

    void (async () => {
      setState((prev) => ({ ...prev, isLoading: true }));
      const list = symbolsKey.split(',');
      const [fundRes, optRes] = await Promise.all([
        fetch(`/api/fundamentals?symbols=${encodeURIComponent(symbolsKey)}`, {
          cache: 'no-store',
          signal: controller.signal,
        }).then((r) => (r.ok ? (r.json() as Promise<FundamentalsResponse>) : null)).catch(() => null),
        fetch(`/api/options?mode=factor-scan&symbols=${encodeURIComponent(symbolsKey.split(',').slice(0, 5).join(','))}`, {
          cache: 'no-store',
          signal: controller.signal,
        }).then((r) => (r.ok ? (r.json() as Promise<OptionsScanResponse>) : null)).catch(() => null),
      ]);
      if (cancelled) return;

      const fundamentalsBySymbol: Record<string, FundamentalSnapshot[]> = {};
      const analystBySymbol: Record<string, AnalystSnapshot | null> = {};
      const optionsBySymbol: Record<string, OptionsFactorSnapshot[]> = {};
      const sectors: Record<string, string> = { ...FALLBACK_SECTORS };
      const coverage: Record<string, FactorCoverage> = {};

      for (const symbol of list) {
        const entry = fundRes?.data?.[symbol];
        if (entry?.snapshot) fundamentalsBySymbol[symbol] = [entry.snapshot];
        analystBySymbol[symbol] = entry?.analyst ?? null;
        if (entry?.sector) sectors[symbol] = entry.sector;
        coverage[symbol] = {
          available: entry?.available ?? false,
          reason: entry?.reason,
          sessions: entry?.sessions ?? 0,
          firstDate: entry?.firstDate ?? null,
          lastDate: entry?.lastDate ?? null,
          degraded: entry?.degraded ?? true,
          sector: entry?.sector ?? FALLBACK_SECTORS[symbol] ?? null,
        };
      }

      const optData = optRes?.data ?? {};
      for (const [key, value] of Object.entries(optData)) {
        if (!value || value.available === false) continue;
        const snap: OptionsFactorSnapshot = {
          symbol: key.toUpperCase(),
          date: value.date ?? new Date().toISOString().slice(0, 10),
          iv30: typeof value.iv30 === 'number' ? value.iv30 : null,
          rv20: typeof value.rv20 === 'number' ? value.rv20 : null,
          atmIv: typeof value.atmIv === 'number' ? value.atmIv : null,
          put25Iv: typeof value.put25Iv === 'number' ? value.put25Iv : null,
          call25Iv: typeof value.call25Iv === 'number' ? value.call25Iv : null,
          putCallOiRatio: typeof value.putCallOiRatio === 'number' ? value.putCallOiRatio : null,
        };
        optionsBySymbol[key.toUpperCase()] = [snap];
      }

      setState({
        fundamentalsBySymbol,
        analystBySymbol,
        optionsBySymbol,
        sectors,
        coverage,
        isLoading: false,
        refresh,
      });
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [symbolsKey, enabled, nonce, refresh]);

  return enabled && symbolsKey ? state : EMPTY_STATE;
}
