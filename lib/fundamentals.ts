// Pure parsers for Yahoo v10 quoteSummary fundamentals / analyst modules.
// Mirrors the defensive, null-degrading style of lib/earnings.ts.

import type {
  AnalystSnapshot,
  EarningsSurprise,
} from './factors/sentiment-options-factors';
import type {
  FundamentalSnapshot,
  FundamentalStatement,
} from './factors/fundamental-factors';

type RawObject = Record<string, unknown>;

function resultNode(json: unknown): RawObject | null {
  const root = (json as { quoteSummary?: { result?: unknown[] | null } } | null)?.quoteSummary;
  if (!root || !Array.isArray(root.result) || root.result.length === 0) return null;
  const node = root.result[0];
  return node && typeof node === 'object' ? (node as RawObject) : null;
}

function rawNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value && typeof value === 'object') {
    const raw = (value as RawObject).raw;
    if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
    if (typeof raw === 'string' && raw.trim() !== '' && Number.isFinite(Number(raw))) return Number(raw);
  }
  return null;
}

function rawString(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object') {
    const raw = (value as RawObject).raw;
    if (typeof raw === 'string') return raw;
  }
  return null;
}

function toYmd(value: unknown): string | null {
  const epoch = rawNumber(value);
  if (epoch !== null && epoch > 0) {
    const d = new Date(epoch * 1000);
    if (!isNaN(d.getTime())) return d.toISOString().slice(0, 10);
  }
  const text = rawString(value);
  if (text && /^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
  return null;
}

function asObject(value: unknown): RawObject | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as RawObject) : null;
}

function statementsFrom(node: unknown, dateFields = ['endDate', 'date']): FundamentalStatement[] {
  if (!Array.isArray(node)) return [];
  const out: FundamentalStatement[] = [];
  for (const entry of node) {
    const obj = asObject(entry);
    if (!obj) continue;
    let periodEnd: string | null = null;
    for (const field of dateFields) {
      periodEnd = toYmd(obj[field]);
      if (periodEnd) break;
    }
    if (!periodEnd) continue;
    const values: Record<string, number | null> = {};
    for (const [key, value] of Object.entries(obj)) {
      const n = rawNumber(value);
      if (n !== null) values[key] = n;
    }
    out.push({ periodEnd, values });
  }
  return out.sort((a, b) => b.periodEnd.localeCompare(a.periodEnd));
}

/** Parse the fundamental modules of a quoteSummary response. */
export function parseFundamentalSnapshot(
  symbol: string,
  observedAt: string,
  json: unknown
): FundamentalSnapshot | null {
  const result = resultNode(json);
  if (!result) return null;

  const profile = asObject(result.assetProfile);
  const price = asObject(result.price);
  const stats = asObject(result.defaultKeyStatistics);
  const statements: FundamentalSnapshot['statements'] = {
    annualIncome: statementsFrom(asObject(result.incomeStatementHistory)?.incomeStatementHistory),
    quarterlyIncome: statementsFrom(
      asObject(result.incomeStatementHistoryQuarterly)?.incomeStatementHistoryQuarterly
    ),
    annualBalance: statementsFrom(asObject(result.balanceSheetHistory)?.balanceSheetStatements),
    quarterlyBalance: statementsFrom(
      asObject(result.balanceSheetHistoryQuarterly)?.balanceSheetStatements
    ),
    annualCashflow: statementsFrom(asObject(result.cashflowStatementHistory)?.cashflowStatements),
    quarterlyCashflow: statementsFrom(
      asObject(result.cashflowStatementHistoryQuarterly)?.cashflowStatements
    ),
  };
  const hasStatements = Object.values(statements).some((list) => list.length > 0);
  const marketCap = rawNumber(price?.marketCap) ?? rawNumber(stats?.marketCap);
  const shares = rawNumber(stats?.sharesOutstanding);
  const sector = rawString(profile?.sector);

  if (!hasStatements && marketCap === null && shares === null && sector === null) return null;

  return {
    symbol: symbol.toUpperCase(),
    observedAt,
    sector,
    industry: rawString(profile?.industry),
    marketCap,
    sharesOutstanding: shares,
    price: rawNumber(price?.regularMarketPrice),
    statements,
  };
}

function parseSurprise(entry: unknown): EarningsSurprise | null {
  const obj = asObject(entry);
  if (!obj) return null;
  const period = rawString(obj.period) ?? rawString(obj.periodEnding);
  if (!period) return null;
  return {
    periodEnd: period.slice(0, 10),
    surprisePercent: rawNumber(obj.surprisePercent),
    epsDifference: rawNumber(obj.difference),
  };
}

/** Parse analyst/estimate/recommendation modules of a quoteSummary response. */
export function parseAnalystSnapshot(
  symbol: string,
  observedAt: string,
  json: unknown
): AnalystSnapshot | null {
  const result = resultNode(json);
  if (!result) return null;

  const earningsHistoryNode = asObject(result.earningsHistory)?.history;
  const earningsHistory = Array.isArray(earningsHistoryNode)
    ? earningsHistoryNode.map(parseSurprise).filter((e): e is EarningsSurprise => e !== null)
    : [];

  const trendNode = asObject(result.earningsTrend)?.trend;
  const trends = Array.isArray(trendNode)
    ? trendNode.map(asObject).filter((t): t is RawObject => t !== null)
    : [];
  const current = trends.find((t) => rawString(t.period) === '+0y' || rawString(t.period) === '0y') ?? trends[0] ?? null;
  const revisions = asObject(current?.epsRevisions);
  const epsTrendNode = asObject(result.epsTrend)?.trend;
  const epsTrends = Array.isArray(epsTrendNode)
    ? epsTrendNode
        .map(asObject)
        .filter((t): t is RawObject => t !== null)
        .filter((t) => rawString(t.period) === '+0y' || rawString(t.period) === '0y')
    : [];
  const epsTrend = epsTrends[0] ?? null;
  const estimateNode = asObject(current?.earningsEstimate);
  const revenueNode = asObject(current?.revenueEstimate);
  const estimate = estimateNode ?? revenueNode;

  const recommendationNode = asObject(result.recommendationTrend)?.trend;
  const recommendations = Array.isArray(recommendationNode)
    ? recommendationNode
        .map(asObject)
        .filter((r): r is RawObject => r !== null)
        .map((r) => ({
          date: rawString(r.period) ?? observedAt,
          strongBuy: rawNumber(r.strongBuy),
          buy: rawNumber(r.buy),
          hold: rawNumber(r.hold),
          sell: rawNumber(r.sell),
          strongSell: rawNumber(r.strongSell),
        }))
    : [];

  const hasData =
    earningsHistory.length > 0 ||
    revisions !== null ||
    epsTrend !== null ||
    estimate !== null ||
    recommendations.length > 0;
  if (!hasData) return null;

  return {
    symbol: symbol.toUpperCase(),
    observedAt,
    earningsHistory,
    revisions: {
      upLast30days: rawNumber(revisions?.upLast30days),
      downLast30days: rawNumber(revisions?.downLast30days),
      currentMean: rawNumber(epsTrend?.current) ?? rawNumber(estimate?.avg),
      mean90daysAgo: rawNumber(epsTrend?.['90daysAgo']),
    },
    estimates: [
      {
        high: rawNumber(estimate?.high),
        low: rawNumber(estimate?.low),
        average: rawNumber(estimate?.avg),
      },
    ],
    recommendations,
  };
}

export type { AnalystSnapshot, FundamentalSnapshot };
