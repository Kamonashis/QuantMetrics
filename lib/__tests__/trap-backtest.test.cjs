// Unit tests for lib/trap-backtest.ts.
// Compiles lib/trap-backtest.ts into an isolated .cache directory so
// `npm run test:unit` exercises event-study deduplication, base-rate gating,
// Newey-West t-stats, Wilson CIs, and decile tables cleanly.
const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { rmSync, mkdirSync } = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');
const BUILD = path.join(ROOT, '.cache', 'test-build-trap-backtest');

rmSync(BUILD, { recursive: true, force: true });
mkdirSync(BUILD, { recursive: true });
const tscBin = require.resolve('typescript/bin/tsc');
execFileSync(
  process.execPath,
  [
    tscBin,
    'lib/trap-backtest.ts',
    '--outDir',
    BUILD,
    '--target',
    'es2020',
    '--module',
    'commonjs',
    '--skipLibCheck',
    '--esModuleInterop',
  ],
  { cwd: ROOT, stdio: 'pipe' }
);

const B = require(path.join(BUILD, 'trap-backtest.js'));

function makeBars(n, fnPrice) {
  const bars = [];
  const start = Date.parse('2025-01-02T00:00:00Z');
  for (let i = 0; i < n; i++) {
    bars.push({
      date: new Date(start + i * 86400000).toISOString().slice(0, 10),
      close: fnPrice(i),
    });
  }
  return bars;
}

function makePanelRow(bar, state = 'triggered', direction = 'up', score = 80) {
  return {
    date: bar.date,
    spot: bar.close,
    atmIv30: 0.25,
    expectedMove30: bar.close * 0.05,
    totalGex: -1e7,
    gammaFlip: bar.close + 2,
    regime: 'short',
    callWall: null,
    putWall: null,
    expiryConcentration7d: 0.4,
    oiWallChange1d: 100,
    oiWallChange5d: 500,
    oiWallChangeZ: 2.0,
    trapScore: score,
    trapState: state,
    trapDirection: direction,
    rv20: 0.22,
    ret5d: 0.02,
    computedAt: '2025-01-02T21:00:00Z',
  };
}

test('empty or short bars returns insufficient_history with zero counts', () => {
  const shortBars = makeBars(20, (i) => 100 + i);
  const res = B.runTrapBacktest([makePanelRow(shortBars[0])], shortBars, 20);
  assert.equal(res.verdict, 'insufficient_history');
  assert.equal(res.nDedupedEvents, 0);
  assert.equal(res.nRawTriggered, 0);
});

test('dedupes consecutive triggers within 10 sessions and gates below minEvents', () => {
  const bars = makeBars(120, (i) => 100 + 0.2 * i);
  // 5 triggers on consecutive days -> 1 deduped event, plus 1 trigger 15 days later -> 2 deduped events
  const panel = [
    makePanelRow(bars[10]),
    makePanelRow(bars[11]),
    makePanelRow(bars[12]),
    makePanelRow(bars[25]),
  ];
  const res = B.runTrapBacktest(panel, bars, 20);
  assert.equal(res.verdict, 'insufficient_history');
  assert.equal(res.nRawTriggered, 4);
  assert.equal(res.nDedupedEvents, 2);
  assert.ok(res.verdictLine.includes('n=2'));
});

test('strong positive follow-through clears base rate -> above_base_rate with deciles and Wilson CI', () => {
  // Alternating quiet drift + sharp positive jump ONLY after each triggered event every 12 bars
  const nBars = 320;
  const triggerIdx = new Set();
  for (let i = 10; i < 290; i += 12) triggerIdx.add(i);

  const closes = [100];
  for (let i = 1; i < nBars; i++) {
    // Most days drift down slightly (-0.3%), except 1..5 days after a trigger jump up (+2.5%)
    let r = -0.003;
    for (let lag = 1; lag <= 5; lag++) {
      if (triggerIdx.has(i - lag)) {
        r = 0.025;
        break;
      }
    }
    closes.push(closes[i - 1] * Math.exp(r));
  }
  const bars = makeBars(nBars, (i) => closes[i]);
  const panel = [...triggerIdx].map((idx, k) => makePanelRow(bars[idx], 'triggered', 'up', 70 + (k % 30)));

  const res = B.runTrapBacktest(panel, bars, 20);
  assert.equal(res.verdict, 'above_base_rate');
  assert.ok(res.nDedupedEvents >= 20);
  assert.ok(res.hitRate5d !== null && res.baseRate5d !== null && res.hitRate5d > res.baseRate5d);
  assert.ok(res.tStat5d !== null && res.tStat5d > 1.96);
  assert.ok(
    res.wilsonLo !== null &&
      res.wilsonHi !== null &&
      res.wilsonLo < res.hitRate5d &&
      res.wilsonHi >= res.hitRate5d - 1e-9
  );
  assert.ok(res.fwdMean.h5 !== null && res.fwdMean.h5 > 0);
  assert.equal(res.decileTable.length, 10);
});

test('negative/no follow-through fails significance gate -> no_reliable_follow_through', () => {
  const nBars = 300;
  const bars = makeBars(nBars, (i) => 100 * Math.exp(-0.002 * i)); // steady downtrend
  const panel = [];
  for (let i = 10; i < 270; i += 12) {
    panel.push(makePanelRow(bars[i], 'triggered', 'up', 75)); // triggered 'up' into a downtrend
  }
  const res = B.runTrapBacktest(panel, bars, 20);
  assert.equal(res.verdict, 'no_reliable_follow_through');
  assert.equal(res.hitRate5d, 0);
});
