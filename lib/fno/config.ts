// F&O (US) universe + market conventions. Config as data, not code —
// the strike ladders and expiry rules change; hardcoded constants rot here.
// US options on equities/ETFs: weekly (Friday) + monthly (3rd Friday), PM-settled,
// physically settled. American style (SPY/QQQ/AAPL…), labelled as such in the UI.

export const US_UNIVERSE = [
  'AAPL',
  'MSFT',
  'NVDA',
  'TSLA',
  'AMZN',
  'GOOGL',
  'META',
  'JPM',
  'SPY',
  'QQQ',
] as const;
export type UniverseSymbol = (typeof US_UNIVERSE)[number];

export const STRIKE_STEP: Record<UniverseSymbol, number> = {
  AAPL: 2.5,
  MSFT: 2.5,
  NVDA: 2.5,
  TSLA: 5.0,
  AMZN: 2.5,
  GOOGL: 2.5,
  META: 2.5,
  JPM: 2.5,
  SPY: 1.0,
  QQQ: 1.0,
} as Record<string, number>;

// US weekly options expire the Friday before the regular/monthly settle.
// Monthly = 3rd Friday of the month.
export const WEEKLY_EXPIRY_WEEKDAY = 5; // Friday
export const MONTHLY_RULE = '3rdFriday' as const;

// Days to include in the chain analysis
export const MAX_DTE = 90;
export const TOP_EXPIRIES = 4;

// True when the symbol trades on a US exchange (all currently are) — used for labels only.
export function isUsUniverse(symbol: string): boolean {
  return (US_UNIVERSE as readonly string[]).includes(symbol.toUpperCase());
}
