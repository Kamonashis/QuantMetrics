// Unit tests for lib/trap-server.ts and lib/options/providers/yahoo.ts.
// Compiles those modules into an isolated .cache directory so `npm run test:unit`
// runs cleanly with zero network calls or shared-build races.
const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { rmSync, mkdirSync } = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');
const BUILD = path.join(ROOT, '.cache', 'test-build-trap-server');

rmSync(BUILD, { recursive: true, force: true });
mkdirSync(BUILD, { recursive: true });
const tscBin = require.resolve('typescript/bin/tsc');
execFileSync(
  process.execPath,
  [
    tscBin,
    'lib/trap-server.ts',
    'lib/options/providers/yahoo.ts',
    '--outDir',
    BUILD,
    '--target',
    'es2020',
    '--module',
    'commonjs',
    '--skipLibCheck',
    '--esModuleInterop',
    '--lib',
    'es2022,dom',
  ],
  { cwd: ROOT, stdio: 'pipe' }
);

const TS = require(path.join(BUILD, 'trap-server.js'));
const Y = require(path.join(BUILD, 'options', 'providers', 'yahoo.js'));

function makeRow(strike, type, oi, dte, expiry, iv = 0.22, volume = 200) {
  return {
    expiry,
    dte,
    strike,
    type,
    oi,
    volume,
    bid: Math.max(0.05, strike * 0.01),
    ask: Math.max(0.08, strike * 0.011),
    last: strike * 0.0105,
    iv,
    contractSymbol: `${type[0].toUpperCase()}${strike}_${dte}`,
  };
}

function makeChain(rows, spot = 500) {
  return {
    symbol: 'SPY',
    substitution: null,
    asOf: '2026-11-27T15:00:00Z',
    spot,
    prevClose: spot - 1,
    quoteAsOf: '2026-11-27T14:45:00Z',
    currency: 'USD',
    name: 'SPDR S&P 500 ETF Trust',
    changePct: 0.2,
    marketCap: 500e9,
    quoteType: 'ETF',
    riskFreeRate: 0.04,
    dividendYield: 0.012,
    delayedMinutes: 15,
    provider: 'yahoo',
    expirationDates: [],
    expiries: [],
    rows,
  };
}

function makeDailyBars(n = 30, close = 500, volume = 50_000_000) {
  const bars = [];
  const start = Date.parse('2026-10-10T00:00:00Z');
  for (let i = 0; i < n; i++) {
    const c = close * (1 + 0.005 * Math.sin(i * 0.7));
    bars.push({
      date: new Date(start + i * 86400000).toISOString().slice(0, 10),
      close: Number(c.toFixed(2)),
      volume,
    });
  }
  return bars;
}

// ---------------------------------------------------------------------------
// 1) Fix 275.9σ distSigma bug on 0-DTE chains
// ---------------------------------------------------------------------------

test('0-DTE only chain floors frontT at 0.5/365 so distSigma stays realistic (never 275σ)', async () => {
  const rows = [
    makeRow(480, 'put', 8000, 0, '2026-11-27', 0.22),
    makeRow(495, 'put', 2000, 0, '2026-11-27', 0.20),
    makeRow(500, 'call', 1500, 0, '2026-11-27', 0.19),
    makeRow(505, 'call', 2000, 0, '2026-11-27', 0.18),
    makeRow(520, 'call', 9000, 0, '2026-11-27', 0.18),
  ];
  const bundle = await TS.computeTrapBundle(makeChain(rows, 500), {
    dailyBars: makeDailyBars(30, 500, 60_000_000),
    priorSnapshots: [],
    panelHistory: [],
  });
  assert.ok(bundle);
  assert.equal(bundle.headlineDte, 0);
  assert.ok(bundle.frontT >= 0.5 / 365 - 1e-12, `expected frontT >= 0.5/365, got ${bundle.frontT}`);
  assert.ok(bundle.walls.call && bundle.walls.call.distSigma < 20, `distSigma blew up: ${bundle.walls.call?.distSigma}`);
  assert.ok(bundle.walls.put && bundle.walls.put.distSigma < 20, `distSigma blew up: ${bundle.walls.put?.distSigma}`);
});

test('headlineDte prefers >=21, then >=7, then >=1 over 0-DTE when mixed expiries exist', async () => {
  const rows = [
    makeRow(490, 'put', 4000, 0, '2026-11-27', 0.25),
    makeRow(510, 'call', 4000, 0, '2026-11-27', 0.25),
    makeRow(480, 'put', 5000, 10, '2026-12-07', 0.20),
    makeRow(520, 'call', 5000, 10, '2026-12-07', 0.20),
  ];
  const bundle = await TS.computeTrapBundle(makeChain(rows, 500), {
    dailyBars: makeDailyBars(30, 500, 60_000_000),
    priorSnapshots: [],
    panelHistory: [],
  });
  assert.ok(bundle);
  assert.equal(bundle.headlineDte, 10);
  assert.ok(Math.abs(bundle.frontT - 10 / 365) < 1e-12);
});

// ---------------------------------------------------------------------------
// 2) ADV20, gexPerAdv, zeroDteShare, byConvention, and toTrapPayload
// ---------------------------------------------------------------------------

test('computeTrapBundle exposes adv20Dollar, gexPerAdv, zeroDteShare, expiryBuckets, matrix and both conventions', async () => {
  const rows = [
    makeRow(500, 'call', 5000, 0, '2026-11-27', 0.22, 12000),
    makeRow(490, 'put', 4000, 0, '2026-11-27', 0.24, 9000),
    makeRow(515, 'call', 8000, 21, '2026-12-18', 0.19, 3000),
    makeRow(485, 'put', 7000, 21, '2026-12-18', 0.21, 2500),
    makeRow(530, 'call', 6000, 60, '2027-01-26', 0.18, 1000),
  ];
  const bars = makeDailyBars(30, 500, 40_000_000);
  const bundle = await TS.computeTrapBundle(makeChain(rows, 500), {
    iv30: 0.20,
    dailyBars: bars,
    priorSnapshots: [],
    panelHistory: [],
  });
  assert.ok(bundle);
  assert.ok(bundle.adv20Dollar !== null && bundle.adv20Dollar > 1e10);
  assert.ok(bundle.gexPerAdv !== null && Math.abs(bundle.gexPerAdv - bundle.totalGex / bundle.adv20Dollar) < 1e-12);
  assert.ok(bundle.zeroDteShare > 0 && bundle.zeroDteShare < 1);
  assert.equal(bundle.expiryBuckets.length, 4);
  assert.ok(bundle.strikeExpiryMatrix.expiries.length === 3);
  assert.ok(bundle.gammaWalls.call && bundle.gammaWalls.put);

  // Convention mirroring: squeeze vs inverse
  const sq = bundle.byConvention.squeeze;
  const inv = bundle.byConvention.inverse;
  assert.ok(Math.abs(sq.totalGex + inv.totalGex) < 1e-6);
  assert.ok(Math.abs(sq.flow.net + inv.flow.net) < 1e-6);
  if (sq.regime === 'long') assert.equal(inv.regime, 'short');
  if (sq.regime === 'short') assert.equal(inv.regime, 'long');

  const payload = TS.toTrapPayload(bundle);
  assert.equal(payload.available, true);
  assert.equal(payload.totalGex, bundle.totalGex);
  assert.equal(payload.gexPerAdv, bundle.gexPerAdv);
  assert.equal(payload.zeroDteShare, bundle.zeroDteShare);
  assert.ok(payload.byConvention && payload.gammaWalls && payload.expiryBuckets && payload.strikeExpiryMatrix);
});

test('include0Dte=false excludes 0-DTE rows and sets zeroDteShare to 0', async () => {
  const rows = [
    makeRow(500, 'call', 5000, 0, '2026-11-27', 0.22),
    makeRow(515, 'call', 8000, 21, '2026-12-18', 0.19),
    makeRow(485, 'put', 7000, 21, '2026-12-18', 0.21),
  ];
  const bundle = await TS.computeTrapBundle(makeChain(rows, 500), {
    include0Dte: false,
    dailyBars: null,
    priorSnapshots: [],
    panelHistory: [],
  });
  assert.ok(bundle);
  assert.equal(bundle.include0Dte, false);
  assert.equal(bundle.zeroDteShare, 0);
  assert.equal(bundle.adv20Dollar, null);
  assert.equal(bundle.gexPerAdv, null);
});

// ---------------------------------------------------------------------------
// 3) Stratified expiry selection & bounded chain cache in yahoo.ts
// ---------------------------------------------------------------------------

test('selectStratifiedExpiries covers 0-DTE, near weeklies, 3rd-Friday OPEX, and 30/60/90d anchors', () => {
  // Base Wednesday 2026-11-18T15:00:00Z -> daily & weekly expiries up to 90d
  const nowSec = Math.floor(Date.parse('2026-11-18T15:00:00Z') / 1000);
  const epochs = [];
  for (let d = 1; d <= 90; d++) {
    // 00:00Z of future days (including 2026-11-20 3rd Fri, 2026-12-18 3rd Fri, 2027-01-15 3rd Fri)
    const dt = new Date((nowSec + d * 86400) * 1000);
    dt.setUTCHours(0, 0, 0, 0);
    const ep = Math.floor(dt.getTime() / 1000);
    if (ep > nowSec) epochs.push(ep);
  }
  // Also add an out-of-window > 90d expiry and duplicate epochs
  epochs.push(nowSec + 120 * 86400, epochs[0]);

  const picked = Y.selectStratifiedExpiries(epochs, nowSec, 8, 90);
  assert.equal(picked.length, 8);
  // Strictly ascending & deduped
  for (let i = 1; i < picked.length; i++) {
    assert.ok(picked[i].epoch > picked[i - 1].epoch);
    assert.ok(picked[i].dte <= 90);
  }
  // Includes 0/1 DTE front
  assert.ok(picked.some((p) => p.dte <= 1), 'missing 0-DTE / 1-DTE');
  // Includes >=7 DTE weekly for SVI/IV30
  assert.ok(picked.some((p) => p.dte >= 7 && p.dte <= 10), 'missing 7..10 DTE weekly');
  // Includes 3rd-Friday monthly OPEX dates beyond DTE=12
  assert.ok(picked.some((p) => p.isMonthlyOpex && p.dte >= 20), 'missing 3rd-Friday monthly OPEX >= 20 DTE');
  // Spans far out past old DTE=12 truncation
  assert.ok(picked[picked.length - 1].dte >= 55, `expected far expiry >= 55 DTE, got ${picked[picked.length - 1].dte}`);
});

test('fetchYahooChain caches chains in memory and coalesces concurrent requests', async () => {
  Y.clearYahooChainCache();
  const origFetch = globalThis.fetch;
  let coreCalls = 0;
  let expiryCalls = 0;
  const nowSec = Math.floor(Date.now() / 1000);
  const exp1 = nowSec + 7 * 86400;
  const exp2 = nowSec + 30 * 86400;

  globalThis.fetch = async (url) => {
    const s = String(url);
    if (s.includes('fc.yahoo.com')) {
      return new Response('', { status: 200, headers: { 'set-cookie': 'A1=test;' } });
    }
    if (s.includes('getcrumb')) {
      return new Response('TESTCRUMB', { status: 200 });
    }
    if (s.includes('chart/%5EIRX')) {
      return new Response(
        JSON.stringify({ chart: { result: [{ indicators: { quote: [{ close: [4.2] }] } }] } }),
        { status: 200 }
      );
    }
    if (s.includes('v7/finance/options/')) {
      if (s.includes('date=')) expiryCalls++;
      else coreCalls++;
      // Small delay so concurrent callers join in-flight promise
      await new Promise((r) => setTimeout(r, 15));
      const dateMatch = /date=(\d+)/.exec(s);
      const exp = dateMatch ? Number(dateMatch[1]) : exp1;
      return new Response(
        JSON.stringify({
          optionChain: {
            result: [
              {
                quote: { regularMarketPrice: 500, quoteType: 'ETF', shortName: 'SPY' },
                expirationDates: [exp1, exp2],
                options: [
                  {
                    expirationDate: exp,
                    calls: [{ strike: 505, bid: 2.0, ask: 2.1, lastPrice: 2.05, openInterest: 1000, volume: 100, impliedVolatility: 0.2, inTheMoney: false }],
                    puts: [{ strike: 495, bid: 2.0, ask: 2.1, lastPrice: 2.05, openInterest: 1000, volume: 100, impliedVolatility: 0.2, inTheMoney: false }],
                  },
                ],
              },
            ],
          },
        }),
        { status: 200 }
      );
    }
    return new Response('', { status: 404 });
  };

  try {
    // Concurrent calls for ^SPX and SPY share the SPY cache/in-flight key while preserving substitution
    const [r1, r2] = await Promise.all([
      Y.fetchYahooChain('SPY', { maxExpiries: 8, maxDte: 90 }),
      Y.fetchYahooChain('^SPX', { maxExpiries: 8, maxDte: 90 }),
    ]);
    assert.ok(r1 && r2);
    assert.equal(r1.substitution, null);
    assert.ok(r2.substitution && r2.substitution.includes('SPY'));
    assert.equal(coreCalls, 1, 'concurrent callers should trigger only 1 core chain fetch');
    assert.equal(expiryCalls, 2, 'concurrent callers should fetch each expiry once');

    // Repeat call hits in-memory cache (zero new HTTP fetches)
    const r3 = await Y.fetchYahooChain('SPY', { maxExpiries: 8, maxDte: 90 });
    assert.ok(r3);
    assert.equal(coreCalls, 1);
    assert.equal(expiryCalls, 2);
    assert.equal(Y.getYahooChainCacheStats().size, 1);
  } finally {
    globalThis.fetch = origFetch;
    Y.clearYahooChainCache();
  }
});
