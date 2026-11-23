const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { parseFundamentalSnapshot, parseAnalystSnapshot } = require(path.join(buildDir, 'fundamentals.js'));

const wrap = (modules) => ({ quoteSummary: { result: [modules] } });
const raw = (v) => ({ raw: v, fmt: String(v) });

test('fundamental parser reads sector, market cap and statement histories', () => {
  const json = wrap({
    assetProfile: { sector: 'Technology', industry: 'Consumer Electronics' },
    price: { marketCap: raw(3_000_000_000), regularMarketPrice: raw(200) },
    defaultKeyStatistics: { sharesOutstanding: raw(15_000_000) },
    incomeStatementHistory: {
      incomeStatementHistory: [
        { endDate: raw(1759200000), totalRevenue: raw(1000), netIncome: raw(150) },
        { endDate: raw(1727654400), totalRevenue: raw(900), netIncome: raw(120) },
      ],
    },
    balanceSheetHistory: {
      balanceSheetStatements: [{ endDate: raw(1759200000), totalAssets: raw(2000), stockholdersEquity: raw(900) }],
    },
    cashflowStatementHistory: {
      cashflowStatements: [{ endDate: raw(1759200000), operatingCashFlow: raw(180), capitalExpenditure: raw(50) }],
    },
    incomeStatementHistoryQuarterly: { incomeStatementHistoryQuarterly: [] },
    balanceSheetHistoryQuarterly: { balanceSheetStatements: [] },
    cashflowStatementHistoryQuarterly: { cashflowStatements: [] },
  });
  const snap = parseFundamentalSnapshot('AAPL', '2026-01-01', json);
  assert.ok(snap);
  assert.equal(snap.symbol, 'AAPL');
  assert.equal(snap.sector, 'Technology');
  assert.equal(snap.marketCap, 3_000_000_000);
  assert.equal(snap.sharesOutstanding, 15_000_000);
  assert.equal(snap.statements.annualIncome.length, 2);
  assert.equal(snap.statements.annualIncome[0].periodEnd, '2025-09-30');
  assert.equal(snap.statements.annualIncome[0].values.netIncome, 150);
  assert.equal(snap.statements.annualBalance[0].values.totalAssets, 2000);
});

test('analyst parser reads surprises, revisions and recommendations', () => {
  const json = wrap({
    earningsHistory: {
      history: [
        { period: raw('2025-12-31'), surprisePercent: raw(4), difference: raw(0.05) },
        { period: raw('2025-09-30'), surprisePercent: raw(2), difference: raw(0.02) },
      ],
    },
    earningsTrend: {
      trend: [
        {
          period: raw('+0y'),
          epsRevisions: { upLast30days: raw(8), downLast30days: raw(2) },
          earningsEstimate: { high: raw(8), low: raw(4), avg: raw(6) },
        },
      ],
    },
    epsTrend: { trend: [{ period: raw('+0y'), current: raw(6), '90daysAgo': raw(5) }] },
    recommendationTrend: {
      trend: [{ period: raw('0m'), strongBuy: raw(4), buy: raw(3), hold: raw(1), sell: raw(0), strongSell: raw(0) }],
    },
  });
  const snap = parseAnalystSnapshot('MSFT', '2026-01-01', json);
  assert.ok(snap);
  assert.equal(snap.earningsHistory.length, 2);
  assert.equal(snap.revisions.upLast30days, 8);
  assert.equal(snap.revisions.currentMean, 6);
  assert.equal(snap.revisions.mean90daysAgo, 5);
  assert.equal(snap.estimates[0].average, 6);
  assert.equal(snap.recommendations[0].strongBuy, 4);
});

test('parsers null-degrade on garbage payloads', () => {
  assert.equal(parseFundamentalSnapshot('AAPL', '2026-01-01', {}), null);
  assert.equal(parseAnalystSnapshot('AAPL', '2026-01-01', null), null);
  assert.equal(parseFundamentalSnapshot('AAPL', '2026-01-01', { quoteSummary: { result: [] } }), null);
});
