'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Loader2, Search, X } from 'lucide-react';
import { KNOWN_TICKERS } from '@/lib/market-service';

export interface TickerSuggestion {
  symbol: string;
  name: string;
  exchange: string;
  quoteType: string;
}

interface TickerSuggestProps {
  /** Raw input value. In `multiple` mode this may be a comma-separated list. */
  value: string;
  onChange: (value: string) => void;
  /** Suggest the last comma-separated segment instead of the whole value. */
  multiple?: boolean;
  placeholder?: string;
  className?: string;
  inputClassName?: string;
  ariaLabel?: string;
  id?: string;
  /** Fired when a suggestion is accepted or an exact symbol is entered. */
  onCommit?: (symbol: string) => void;
}

interface FetchedResults {
  q: string;
  quotes: TickerSuggestion[];
  source: 'yahoo' | 'fallback';
}

const CLIENT_FALLBACK: TickerSuggestion[] = Object.entries(KNOWN_TICKERS).map(
  ([symbol, info]) => ({ symbol, name: info.name, exchange: '', quoteType: 'EQUITY' })
);

function fallbackMatches(segment: string): TickerSuggestion[] {
  const needle = segment.trim().toLowerCase();
  if (!needle) return [];
  return CLIENT_FALLBACK.filter(
    (t) => t.symbol.toLowerCase().includes(needle) || t.name.toLowerCase().includes(needle)
  ).slice(0, 8);
}

/** Replace the last comma-separated segment with `symbol`, leaving a trailing ", ". */
function applyMultiple(value: string, symbol: string): string {
  const segments = value.split(',');
  segments[segments.length - 1] = ` ${symbol}`;
  const head = segments.slice(0, -1).map((s) => s.trim()).filter(Boolean);
  return [...head, symbol].join(', ') + ', ';
}

export function TickerSuggest({
  value,
  onChange,
  multiple = false,
  placeholder,
  className,
  inputClassName,
  ariaLabel,
  id,
  onCommit,
}: TickerSuggestProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();
  const requestIdRef = useRef(0);

  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState<FetchedResults | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  // Dropdown stays closed until the user explicitly types or asks — page load never pops it open.
  const interactedRef = useRef(false);
  // Escape dismisses until the next keystroke/click.
  const dismissedRef = useRef(false);
  // Bump to (re)trigger the effect when the user asks for suggestions before any typing happened.
  const [interactionTick, setInteractionTick] = useState(0);

  // Which part of the value the suggestions describe
  const segment = useMemo(() => {
    const raw = multiple ? (value.split(',').pop() ?? '') : value;
    return raw.trim();
  }, [value, multiple]);

  // Debounced Yahoo search — local matches render instantly, Yahoo results replace them.
  useEffect(() => {
    const q = segment;
    if (!q || !interactedRef.current || dismissedRef.current) return; // fetch only after explicit user input

    const requestId = ++requestIdRef.current;
    const timer = window.setTimeout(() => {
      setLoading(true);
      fetch(`/api/ticker-search?q=${encodeURIComponent(q)}`)
        .then((r) => r.json() as Promise<{ quotes?: TickerSuggestion[] }>)
        .then((data) => {
          if (requestId !== requestIdRef.current) return;
          const yahoo = Array.isArray(data?.quotes) ? data.quotes : [];
          setFetched(
            yahoo.length > 0
              ? { q, quotes: yahoo, source: 'yahoo' }
              : { q, quotes: fallbackMatches(q), source: 'fallback' }
          );
          setLoading(false);
          if (!dismissedRef.current) setOpen(true);
          setActiveIndex(0);
        })
        .catch(() => {
          if (requestId !== requestIdRef.current) return;
          setFetched({ q, quotes: fallbackMatches(q), source: 'fallback' });
          setLoading(false);
          if (!dismissedRef.current) setOpen(true);
          setActiveIndex(0);
        });
    }, 250);

    return () => window.clearTimeout(timer);
  }, [segment, interactionTick]);

  // Close on outside pointer down
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [open]);

  const accept = useCallback(
    (symbol: string) => {
      dismissedRef.current = true; // accept = done with suggestions until the next keystroke
      onChange(multiple ? applyMultiple(value, symbol) : symbol);
      onCommit?.(symbol);
      setOpen(false);
    },
    [multiple, onChange, onCommit, value]
  );

  // What to render for the current segment (Yahoo > local cache > local fallback)
  const localMatches = useMemo(() => fallbackMatches(segment), [segment]);
  const fetchedMatches = fetched && fetched.q === segment ? fetched : null;
  const results = fetchedMatches?.quotes ?? localMatches;
  const source: 'yahoo' | 'fallback' | null = fetchedMatches?.source ?? (localMatches.length > 0 ? 'fallback' : null);

  const showUseTyped = segment.length > 0;
  const rowCount = results.length + (showUseTyped ? 1 : 0);
  const active = rowCount > 0 ? Math.min(Math.max(activeIndex, 0), rowCount - 1) : -1;

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown' && !open) {
      e.preventDefault();
      interactedRef.current = true;
      dismissedRef.current = false;
      if (results.length > 0) {
        setOpen(true);
        setActiveIndex(0);
      } else {
        // Nothing fetched yet — ask the effect to run now
        setInteractionTick((c) => c + 1);
      }
      return;
    }
    if (!open) return;
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setActiveIndex((i) => (rowCount > 0 ? (i + 1) % rowCount : 0));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex((i) => (rowCount > 0 ? (i - 1 + rowCount) % rowCount : 0));
        break;
      case 'Enter': {
        if (active < 0) return; // nothing highlighted — let the form submit proceed
        const target = active < results.length ? results[active] : null;
        const targetSymbol = target ? target.symbol : segment.toUpperCase();
        if (targetSymbol.toUpperCase() === segment.toUpperCase()) {
          // Exact symbol already typed — commit it and let the form submit proceed.
          onCommit?.(targetSymbol.toUpperCase());
          setOpen(false);
          return;
        }
        e.preventDefault();
        accept(targetSymbol);
        break;
      }
      case 'Escape':
        e.preventDefault();
        dismissedRef.current = true;
        setOpen(false);
        break;
      case 'Tab':
        setOpen(false);
        break;
      default:
        break;
    }
  };

  const highlight = (text: string, needle: string) => {
    if (!needle) return text;
    const lower = text.toLowerCase();
    const i = lower.indexOf(needle.toLowerCase());
    if (i === -1) return text;
    return (
      <>
        {text.slice(0, i)}
        <mark className="bg-gold-500/30 text-inherit">{text.slice(i, i + needle.length)}</mark>
        {text.slice(i + needle.length)}
      </>
    );
  };

  return (
    <div ref={containerRef} className={`relative ${className ?? ''}`}>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-400" />
        <input
          ref={inputRef}
          id={id}
          type="text"
          autoComplete="off"
          spellCheck={false}
          role="combobox"
          aria-expanded={open}
          aria-controls={listboxId}
          aria-activedescendant={active >= 0 ? `${listboxId}-option-${active}` : undefined}
          aria-autocomplete="list"
          aria-label={ariaLabel}
          value={value}
          placeholder={placeholder}
          onChange={(e) => {
            interactedRef.current = true;
            dismissedRef.current = false;
            setOpen(true);
            onChange(e.target.value.toUpperCase());
          }}
          onFocus={() => {
            if (interactedRef.current && !dismissedRef.current && results.length > 0) {
              setOpen(true);
            }
          }}
          onKeyDown={onKeyDown}
          className={`${inputClassName ?? ''} pl-9 ${loading ? 'pr-9' : ''}`}
        />
        {loading && (
          <Loader2 className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-gold-500" />
        )}
      </div>

      {open && segment.length > 0 && (
        <div className="absolute z-40 mt-1 w-full min-w-[16rem] overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#141419] shadow-xl">
          {source === 'yahoo' && (
            <div className="border-b border-zinc-100 dark:border-zinc-800/70 px-3 py-1 font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400">
              Yahoo Finance
            </div>
          )}
          {source === 'fallback' && results.length > 0 && (
            <div className="border-b border-zinc-100 dark:border-zinc-800/70 px-3 py-1 font-mono text-[9px] uppercase tracking-[0.18em] text-zinc-400">
              Known tickers
            </div>
          )}
          <ul id={listboxId} role="listbox" className="max-h-64 overflow-y-auto py-1">
            {results.map((quote, i) => {
              const isActive = i === active;
              return (
                <li
                  key={`${quote.symbol}-${i}`}
                  role="option"
                  id={`${listboxId}-option-${i}`}
                  aria-selected={isActive}
                  onMouseEnter={() => setActiveIndex(i)}
                  onMouseDown={(e) => {
                    e.preventDefault(); // keep input focus
                    accept(quote.symbol);
                  }}
                  className={`flex cursor-pointer items-center justify-between gap-3 px-3 py-1.5 text-left ${
                    isActive ? 'bg-gold-500/10' : ''
                  }`}
                >
                  <span className="min-w-0">
                    <span className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100">
                      {highlight(quote.symbol, segment)}
                    </span>
                    <span className="ml-2 truncate text-[11px] text-zinc-500 dark:text-zinc-400">
                      {highlight(quote.name, segment)}
                    </span>
                  </span>
                  {quote.exchange && (
                    <span className="shrink-0 font-mono text-[9px] uppercase tracking-wide text-zinc-400">
                      {quote.exchange}
                    </span>
                  )}
                </li>
              );
            })}
            {results.length === 0 && (
              <li className="px-3 py-2 text-[11px] text-zinc-400">
                No matches for “{segment}”
              </li>
            )}
            {showUseTyped && (
              <li
                role="option"
                id={`${listboxId}-option-${results.length}`}
                aria-selected={active === results.length}
                onMouseEnter={() => setActiveIndex(results.length)}
                onMouseDown={(e) => {
                  e.preventDefault();
                  accept(segment.toUpperCase());
                }}
                className={`flex cursor-pointer items-center gap-2 border-t border-zinc-100 dark:border-zinc-800/70 px-3 py-1.5 text-[11px] text-zinc-500 dark:text-zinc-400 ${
                  active === results.length ? 'bg-gold-500/10' : ''
                }`}
              >
                <X className="h-3 w-3 shrink-0" />
                <span>
                  Use <span className="font-mono font-semibold">{segment.toUpperCase()}</span> as typed
                </span>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
