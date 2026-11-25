/**
 * Pure-TS lifecycle & single-flight coordinator for all TCN-Attention models.
 *
 * Policy:
 *  - Train ONCE automatically as soon as the app starts/loads for the first time
 *    (once initial market bars settle) or whenever the active ticker/pair changes.
 *  - After that one-time initial fit completes for a given key, training becomes
 *    strictly MANUAL — live quote ticks, view switches, parameter tweaks, or cache
 *    clears on the active ticker never trigger automatic retraining; only explicit
 *    user Retrain button clicks (`runManualTcnJob`) do.
 *  - Concurrent views requesting the same key deduplicate onto the single in-flight
 *    promise and subscribe to live training/completion state changes.
 */

export type TcnJobDomain =
  | 'volatility'
  | 'risk_var'
  | 'regression'
  | 'correlation'
  | 'decomposition';

interface InFlightJob {
  promise: Promise<unknown>;
  label: string;
  isManual: boolean;
}

const autoTrainedKeys = new Set<string>();
const inFlightJobs = new Map<string, InFlightJob>();
const listeners = new Set<() => void>();

let coordinatorRevision = 0;
let marketDataReady = false;

function notify(): void {
  coordinatorRevision += 1;
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      // Never let one subscriber break notification fan-out.
    }
  }
}

export function tcnJobKey(
  domain: TcnJobDomain,
  primaryTicker: string,
  extra?: string | string[]
): string {
  const sym = primaryTicker.trim().toUpperCase();
  if (extra === undefined) return `${domain}:${sym}`;
  const suffix = Array.isArray(extra)
    ? extra
        .slice(0, 6)
        .map((s) => s.trim().toUpperCase())
        .join(',')
    : extra.trim().toUpperCase();
  return `${domain}:${sym}|${suffix}`;
}

export function subscribeTcnCoordinator(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getTcnCoordinatorRevision(): number {
  return coordinatorRevision;
}

export function setTcnMarketDataReady(ready: boolean): void {
  if (marketDataReady === ready) return;
  marketDataReady = ready;
  notify();
}

export function isTcnMarketDataReady(): boolean {
  return marketDataReady;
}

export function hasAutoTrainedKey(key: string): boolean {
  return autoTrainedKeys.has(key);
}

export function isTcnJobRunning(key: string): boolean {
  return inFlightJobs.has(key);
}

export function getActiveTcnJobLabel(keys: string[]): string | null {
  for (const key of keys) {
    const job = inFlightJobs.get(key);
    if (job) return job.label;
  }
  return null;
}

/**
 * Runs a ONE-TIME automatic fit for `key` if it has not been auto-trained yet.
 * - If `key` is already in-flight, joins that promise (no duplicate fit).
 * - If `key` was already auto-trained (or manually trained), returns `null`
 *   immediately without training.
 * - If cancelled mid-flight before finishing (`isCancelled()` returns true and
 *   result is null), releases the slot so returning to that ticker later will
 *   still perform its initial one-time fit.
 */
export async function runAutoTcnJob<T>(
  key: string,
  label: string,
  runner: () => Promise<T | null>,
  isCancelled?: () => boolean
): Promise<T | null> {
  const existing = inFlightJobs.get(key);
  if (existing) {
    return (await existing.promise) as T | null;
  }
  if (autoTrainedKeys.has(key)) {
    return null;
  }

  autoTrainedKeys.add(key);

  // Declared before assignment: the IIFE's `finally` compares against it (TS2454).
  let jobPromise: Promise<T | null> | undefined = undefined;
  jobPromise = (async (): Promise<T | null> => {
    try {
      const result = await runner();
      if (result === null && isCancelled?.()) {
        autoTrainedKeys.delete(key);
      }
      return result;
    } catch (err) {
      autoTrainedKeys.delete(key);
      throw err;
    } finally {
      const current = inFlightJobs.get(key);
      if (current && current.promise === jobPromise) {
        inFlightJobs.delete(key);
      }
      notify();
    }
  })();

  inFlightJobs.set(key, { promise: jobPromise, label, isManual: false });
  notify();
  return jobPromise;
}

/**
 * Runs an EXPLICIT manual retrain triggered by a user clicking a Retrain button.
 * Always executes and marks `key` as satisfied for auto-training so subsequent
 * live ticks remain manual-only.
 */
export async function runManualTcnJob<T>(
  key: string,
  label: string,
  runner: () => Promise<T | null>
): Promise<T | null> {
  autoTrainedKeys.add(key);

  // Declared before assignment: the IIFE's `finally` compares against it (TS2454).
  let jobPromise: Promise<T | null> | undefined = undefined;
  jobPromise = (async (): Promise<T | null> => {
    try {
      return await runner();
    } finally {
      const current = inFlightJobs.get(key);
      if (current && current.promise === jobPromise) {
        inFlightJobs.delete(key);
      }
      notify();
    }
  })();

  inFlightJobs.set(key, { promise: jobPromise, label, isManual: true });
  notify();
  return jobPromise;
}

/**
 * Called when the user explicitly clears TCN caches.
 * Keeps keys belonging to `currentTicker` locked in manual-only mode so clearing
 * cache does NOT immediately auto-retrain the active symbol, while releasing
 * other tickers so changing to a new ticker later will train once.
 */
export function markTcnCachesCleared(currentTicker?: string): void {
  const keepSym = currentTicker?.trim().toUpperCase() ?? '';
  for (const key of Array.from(autoTrainedKeys)) {
    if (!keepSym) {
      autoTrainedKeys.delete(key);
      continue;
    }
    const colonIdx = key.indexOf(':');
    const rest = colonIdx >= 0 ? key.slice(colonIdx + 1) : key;
    const primary = rest.split('|')[0];
    if (primary !== keepSym) {
      autoTrainedKeys.delete(key);
    }
  }
  notify();
}

/**
 * Notifies subscribers that underlying TCN caches changed outside a coordinated
 * job (e.g. a sensitivity sweep in DecompositionChart).
 */
export function notifyTcnCacheUpdated(): void {
  notify();
}

/**
 * Resets all coordinator state. Intended for unit tests.
 */
export function resetTcnCoordinatorForTests(): void {
  autoTrainedKeys.clear();
  inFlightJobs.clear();
  listeners.clear();
  coordinatorRevision = 0;
  marketDataReady = false;
}
