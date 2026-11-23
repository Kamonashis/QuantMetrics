const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const buildDir = path.resolve(__dirname, '../../.cache/test-build');
const { computeVolumeProfile } = require(path.join(buildDir, 'volume-profile.js'));

function bar(close, low, high, volume, i) {
  return {
    date: `2026-01-${String(i + 1).padStart(2, '0')}`,
    open: close,
    high: high ?? close,
    low: low ?? close,
    close,
    volume,
  };
}

function makeBars(n = 40) {
  const bars = [];
  for (let i = 0; i < n; i++) {
    const price = 100 + Math.sin(i / 5) * 5;
    bars.push(bar(price, price - 1, price + 1, 1_000_000 + (i % 7) * 100_000, i));
  }
  return bars;
}

test('volume profile returns null when there are too few usable bars', () => {
  assert.equal(computeVolumeProfile([]), null);
  assert.equal(computeVolumeProfile(makeBars(5)), null);
  assert.equal(computeVolumeProfile(makeBars(40).map((b) => ({ ...b, volume: 0 }))), null);
});

test('volume conservation: distributed bin volume equals total bar volume', () => {
  const profile = computeVolumeProfile(makeBars(40), { bins: 32 });
  assert.ok(profile, 'expected a profile');
  const distributed = profile.bins.reduce((sum, bin) => sum + bin.volume, 0);
  assert.ok(
    Math.abs(distributed - profile.totalVolume) / profile.totalVolume < 1e-9,
    `distributed ${distributed} vs total ${profile.totalVolume}`
  );
  const shareSum = profile.bins.reduce((sum, bin) => sum + bin.share, 0);
  assert.ok(Math.abs(shareSum - 1) < 1e-9);
});

test('a narrow high-volume range becomes the point of control', () => {
  const bars = makeBars(40).map((b, i) => {
    if (i < 30) return bar(100, 99, 101, 100_000, i);
    return bar(130, 129.5, 130.5, 20_000_000, i);
  });
  const profile = computeVolumeProfile(bars, { bins: 40 });
  assert.ok(profile && profile.poc);
  assert.ok(profile.poc.mid > 128, `expected POC near 130, got ${profile.poc.mid}`);
});

test('value area contains the POC and meets its target volume share', () => {
  const profile = computeVolumeProfile(makeBars(60), { bins: 48, valueAreaShare: 0.7 });
  assert.ok(profile && profile.poc && profile.valueArea);
  const { low, high, volumeShare } = profile.valueArea;
  assert.ok(low <= profile.poc.mid && high >= profile.poc.mid);
  assert.ok(volumeShare >= 0.7 - 1e-9, `share ${volumeShare}`);
  assert.ok(volumeShare <= 1 + 1e-9);
});

test('high-volume nodes are local maxima and include the POC', () => {
  const profile = computeVolumeProfile(makeBars(50), { bins: 24 });
  assert.ok(profile && profile.poc);
  assert.ok(profile.hvn.length > 0);
  const pocVolume = profile.poc.volume;
  for (const node of profile.hvn) {
    assert.ok(node.volume <= pocVolume + 1e-9);
  }
  assert.ok(profile.hvn.some((node) => node.mid === profile.poc.mid));
});

test('degenerate flat prices produce a valid narrow profile', () => {
  const bars = makeBars(20).map((b) => bar(50, 50, 50, 500_000, 0));
  const profile = computeVolumeProfile(bars);
  assert.ok(profile && profile.poc && profile.valueArea);
  assert.ok(profile.maxPrice > profile.minPrice);
  assert.equal(profile.barCount, 20);
  assert.equal(profile.approximation, 'ohlc-range');
});
