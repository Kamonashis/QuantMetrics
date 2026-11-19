// Unit tests for lib/ticker-snapshot.ts — one definition per shared quantity.
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const S = require(path.join(BUILD, 'ticker-snapshot.js'));

const closes = (n) => Array.from({ length: n }, (_, i) => 100 + 3 * Math.sin(i / 9) + i * 0.05);

test('canonicalRvLadder: one sample-stdev × √252 convention, latest value', () => {
  const ladder = S.canonicalRvLadder(closes(300), [5, 20]);
  assert.deepEqual(ladder.map((r) => r.window), [5, 20]);
  for (const r of ladder) {
    assert.ok(r.n > 0);
    assert.ok(r.vol !== null && r.vol > 0 && Number.isFinite(r.vol));
  }
  assert.deepEqual(S.canonicalRvLadder([100], [20]).map((r) => r.vol), [null]);
});

test('selectFrontSlice: prefers 15–45 DTE and ignores null ATM IV', () => {
  const slices = [
    { dte: 7, atmIv: 0.2 },
    { dte: 21, atmIv: 0.22 },
    { dte: 40, atmIv: 0.24 },
    { dte: 120, atmIv: 0.26 },
    { dte: 25, atmIv: null },
  ];
  assert.equal(S.selectFrontSlice(slices).dte, 21);
  assert.equal(S.selectFrontSlice([{ dte: 7, atmIv: 0.2 }, { dte: 90, atmIv: 0.25 }]).dte, 7);
  assert.equal(S.selectFrontSlice([]), null);
});

test('canonicalVrp: IV² − RV² and IV−RV, null-degrading', () => {
  const r = S.canonicalVrp(0.3, 0.2);
  assert.ok(Math.abs(r.variancePoints - (0.09 - 0.04) * 1e4) < 1e-6);
  assert.ok(Math.abs(r.premiumPoints - 10) < 1e-9);
  assert.deepEqual(S.canonicalVrp(null, 0.2), { variancePoints: null, premiumPoints: null });
});

test('sessionDiff: compares the last two dated panel rows', () => {
  const rows = [
    { date: '2026-11-06', spot: 100, atmIv30: 0.2, trapScore: 40, rv20: 0.18 },
    { date: '2026-11-07', spot: 102, atmIv30: 0.19, trapScore: 55, rv20: 0.17 },
  ];
  const d = S.sessionDiff(rows);
  assert.equal(d.date, '2026-11-07');
  assert.equal(d.dSpot, 2);
  assert.ok(Math.abs(d.dIv30 + 0.01) < 1e-9);
  assert.equal(d.dTrapScore, 15);
  assert.equal(S.sessionDiff(rows.slice(0, 1)), null);
});

test('vixContext and curveVerdict: single percentile and slope rule', () => {
  const bars = Array.from({ length: 100 }, (_, i) => ({ date: String(i), close: 10 + (i % 7) }));
  const ctx = S.vixContext(bars, 16);
  assert.equal(ctx.level, 16);
  assert.ok(ctx.percentile !== null && ctx.percentile >= 0 && ctx.percentile <= 100);
  assert.equal(S.vixContext(bars, null).percentile, null);
  assert.equal(S.curveVerdict([{ dte: 10, iv: 0.3 }, { dte: 60, iv: 0.2 }]), 'INVERTED');
  assert.equal(S.curveVerdict([{ dte: 10, iv: 0.15 }, { dte: 60, iv: 0.25 }]), 'CONTANGO');
  assert.equal(S.curveVerdict([{ dte: 10, iv: 0.2 }]), null);
});
