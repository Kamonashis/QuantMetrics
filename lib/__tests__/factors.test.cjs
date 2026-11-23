const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { computePriceFactorSeries, PRICE_FACTORS } = require(path.join(buildDir, 'factors/price-factors.js'));
const { computeFundamentalFactors, FUNDAMENTAL_FACTORS } = require(path.join(buildDir, 'factors/fundamental-factors.js'));
const { computeAnalystFactors, computeOptionsFactors } = require(path.join(buildDir, 'factors/sentiment-options-factors.js'));
const { winsorize, rankify, scoreCrossSection, computeFactorIc, rankIc } = require(path.join(buildDir, 'factors/cross-section.js'));
const { ALL_FACTORS, buildFactorPanel } = require(path.join(buildDir, 'factors/index.js'));

function makeBars(n = 300, start = 100) {
  const bars = [];
  let price = start;
  for (let i = 0; i < n; i++) {
    const open = price;
    price = price * (1 + 0.0008 * Math.sin(i / 7) + 0.0002 * (i % 3));
    const high = Math.max(open, price) * 1.01;
    const low = Math.min(open, price) * 0.99;
    bars.push({
      date: new Date(Date.UTC(2024, 0, 1) + i * 86400_000).toISOString().slice(0, 10),
      open, high, low, close: price,
      volume: 1_000_000 + (i % 11) * 100_000,
    });
  }
  return bars;
}

test('factor registry is complete and ids are unique', () => {
  const ids = ALL_FACTORS.map((f) => f.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.ok(ids.length >= 40, `expected a broad factor library, got ${ids.length}`);
  for (const id of ['price:MOM_12_1', 'price:AMIHUD', 'fund:PIOTROSKI_F', 'fund:SIZE_SMB', 'sent:PEAD_SUE', 'opt:VRP']) {
    assert.ok(ids.includes(id), `missing ${id}`);
  }
});

test('winsorize clips only the tails and rankify averages ties', () => {
  const values = [1, 2, 3, 4, 1000];
  const w = winsorize(values, 0.2, 0.8);
  assert.ok(w[4] < 1000);
  assert.ok(w[0] > 1 || Math.abs(w[0] - 1) < 1e-9);
  assert.deepEqual(rankify([10, 20, 20, 30]), [1, 2.5, 2.5, 4]);
  assert.equal(rankIc([1, 2, 3, 4], [1, 2, 3, 4]), 1);
});

test('price factors respect lookback and compute MOM / reversal / range vol', () => {
  const bars = makeBars(300);
  const bench = makeBars(300, 90);
  const series = computePriceFactorSeries(bars, bench, { maxPoints: 2 });
  const full = computePriceFactorSeries(bars, bench);
  for (const def of PRICE_FACTORS) assert.ok(Array.isArray(series[def.id]), `missing ${def.id}`);
  const t = bars.length - 1;
  assert.equal(full['price:MOM_12_1'][251], null);
  assert.ok(Number.isFinite(full['price:MOM_12_1'][252]));
  assert.ok(Number.isFinite(series['price:MOM_12_1'][t]));
  assert.ok(Number.isFinite(series['price:REV_1M'][t]));
  assert.ok(Number.isFinite(series['price:AMIHUD'][t]));
  assert.ok(series['price:AMIHUD'][t] > 0);
  assert.ok(Number.isFinite(series['price:GK_VOL'][t]));
  assert.ok(Number.isFinite(series['price:PARKINSON_VOL'][t]));
  assert.ok(Number.isFinite(series['price:BAB_BETA'][t]));
});

test('a pure uptrend gives positive classic momentum and negative 1-month reversal', () => {
  const bars = makeBars(300, 100).map((b, i) => ({
    ...b,
    close: 100 * Math.pow(1.001, i),
    high: 100 * Math.pow(1.001, i) * 1.01,
    low: 100 * Math.pow(1.001, i) * 0.99,
    open: 100 * Math.pow(1.001, Math.max(0, i - 1)),
  }));
  const series = computePriceFactorSeries(bars, null, { maxPoints: 1 });
  const t = bars.length - 1;
  assert.ok(series['price:MOM_12_1'][t] > 0);
  assert.ok(series['price:REV_1M'][t] < 0);
  assert.ok(series['price:HIGH_52W'][t] > 0.99);
});

test('fundamental factors compute yields and enforce reporting lag', () => {
  const values = (extra = {}) => ({
    totalRevenue: 1000, costOfRevenue: 600, grossProfit: 400, operatingIncome: 200,
    ebit: 210, ebitda: 260, netIncome: 150, totalAssets: 2000, stockholdersEquity: 900,
    totalDebt: 300, cash: 100, operatingCashFlow: 180, capitalExpenditure: 50,
    dividendsPaid: 20, paymentsForRepurchaseOfCommonStock: 30, proceedsFromIssuanceOfCommonStock: 10,
    currentAssets: 500, currentLiabilities: 250, interestExpense: 10,
    dilutedAverageShares: 100, incomeTaxExpenseBenefit: 40,
    incomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest: 190,
    ...extra,
  });
  const stmt = (periodEnd, v) => ({ periodEnd, values: v });
  const snapshot = {
    symbol: 'AAPL',
    observedAt: '2026-01-01',
    sector: 'Tech',
    industry: 'Hardware',
    marketCap: 5000,
    sharesOutstanding: 100,
    price: 50,
    statements: {
      annualIncome: [stmt('2025-09-30', values()), stmt('2024-09-30', values({ netIncome: 120 }))],
      quarterlyIncome: [stmt('2025-06-30', values())],
      annualBalance: [stmt('2025-09-30', values()), stmt('2024-09-30', values())],
      quarterlyBalance: [stmt('2025-06-30', values())],
      annualCashflow: [stmt('2025-09-30', values()), stmt('2024-09-30', values())],
      quarterlyCashflow: [stmt('2025-06-30', values())],
    },
  };
  // 2025-09-15: the 2025 statements have not cleared their lag, so the most
  // recent *usable* annual filing is still FY2024 (net income 120).
  const early = computeFundamentalFactors(snapshot, '2025-09-15');
  assert.ok(Math.abs(early['fund:EARNINGS_YIELD'] - 120 / 5000) < 1e-9);
  assert.equal(early['fund:SIZE_SMB'], -Math.log(5000));
  // After the lag, the FY2025 statements become usable.
  const ready = computeFundamentalFactors(snapshot, '2026-04-01');
  assert.ok(Math.abs(ready['fund:EARNINGS_YIELD'] - 150 / 5000) < 1e-9);
  assert.ok(Math.abs(ready['fund:BOOK_TO_MARKET'] - 900 / 5000) < 1e-9);
  assert.ok(Math.abs(ready['fund:GROSS_PROFITABILITY'] - 400 / 2000) < 1e-9);
  assert.ok(Math.abs(ready['fund:FCF_YIELD'] - 130 / 5000) < 1e-9);
  assert.ok(ready['fund:PIOTROSKI_F'] >= 0 && ready['fund:PIOTROSKI_F'] <= 9);
  for (const def of FUNDAMENTAL_FACTORS) assert.ok(def.id in ready);
});

test('analyst and options factors compute their core formulas', () => {
  const analyst = {
    symbol: 'AAPL', observedAt: '2026-01-01',
    earningsHistory: [{ periodEnd: '2025-12-31', surprisePercent: 4, epsDifference: 0.05 }],
    revisions: { upLast30days: 8, downLast30days: 2, currentMean: 6, mean90daysAgo: 5 },
    estimates: [{ high: 8, low: 4, average: 6 }],
    recommendations: [
      { date: '2025-12-01', strongBuy: 1, buy: 2, hold: 3, sell: 1, strongSell: 0 },
      { date: '2025-12-20', strongBuy: 4, buy: 3, hold: 1, sell: 0, strongSell: 0 },
    ],
  };
  const a = computeAnalystFactors(analyst);
  assert.equal(a['sent:PEAD_SUE'], 4);
  assert.equal(a['sent:REVISION_BREADTH'], 0.6);
  assert.equal(a['sent:ESTIMATE_DRIFT'], 0.2);
  assert.ok(Math.abs(a['sent:DISPERSION'] - 4 / 6) < 1e-9);
  assert.ok(a['sent:RATING_CHANGE'] > 0);

  const o = computeOptionsFactors({
    symbol: 'AAPL', date: '2026-01-01',
    iv30: 0.30, rv20: 0.22, atmIv: 0.28, put25Iv: 0.34, call25Iv: 0.25, putCallOiRatio: 1.2,
  });
  assert.ok(Math.abs(o['opt:VRP'] - 0.08) < 1e-9);
  assert.ok(Math.abs(o['opt:SKEW'] - 0.06) < 1e-9);
  assert.equal(o['opt:PCR'], 1.2);
});

test('cross-sectional scoring standardizes, sector-neutralizes and ranks', () => {
  const defs = [{ id: 'price:RV60', family: 'low_volatility', label: 'RV', formula: '', rationale: '', direction: -1, requires: 'price', minObservations: 1 }];
  const rows = [
    { date: '2026-01-02', symbol: 'A', sector: 'Tech', raw: { 'price:RV60': 0.2 }, forwardReturn: 0.03 },
    { date: '2026-01-02', symbol: 'B', sector: 'Tech', raw: { 'price:RV60': 0.4 }, forwardReturn: 0.01 },
    { date: '2026-01-02', symbol: 'C', sector: 'Banks', raw: { 'price:RV60': 0.3 }, forwardReturn: 0.02 },
    { date: '2026-01-02', symbol: 'D', sector: 'Banks', raw: { 'price:RV60': 0.6 }, forwardReturn: -0.01 },
  ];
  const scored = scoreCrossSection(rows, defs, { sectorNeutral: true });
  assert.equal(scored.length, 4);
  // Direction -1 flips vol; sector neutralization then makes C (Banks, 0.3)
  // outrank A (Tech, 0.2) because it is the lowest-vol name within its own sector.
  assert.equal(scored.find((r) => r.symbol === 'C').rank, 1);
  assert.equal(scored.find((r) => r.symbol === 'A').rank, 2);
  assert.ok(scored.every((r) => r.composite !== null));
  const ic = computeFactorIc(scored, defs);
  assert.equal(ic.length, 2);
  assert.ok(Number.isFinite(ic[0].meanIc));
});

test('buildFactorPanel aligns scores with forward returns', () => {
  const barsA = makeBars(300);
  const barsB = makeBars(300, 80);
  const dates = barsA.map((b) => b.date);
  const rows = buildFactorPanel({
    symbols: ['A', 'B'],
    barsBySymbol: { A: barsA, B: barsB },
    benchmarkBars: barsA,
    dates,
    horizon: 5,
    maxPoints: 30,
    sectors: { A: 'Tech', B: 'Banks' },
  });
  assert.ok(rows.length > 0);
  assert.ok(rows.every((r) => r.raw['price:MOM_12_1'] !== null));
  assert.ok(rows.some((r) => r.forwardReturn !== null));
});
