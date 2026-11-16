// Unit tests for lib/earnings.ts (compiled output in .cache/test-build).
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const BUILD = path.join(__dirname, '..', '..', '.cache', 'test-build');
const E = require(path.join(BUILD, 'earnings.js'));

const payload = (earnings) => ({
  quoteSummary: { result: [{ calendarEvents: { earnings } }], error: null },
});

test('parseQuoteSummaryEarnings: parses earningsDate array, sorts ascending, takes nearest upcoming', () => {
  const json = payload({
    earningsDate: [
      { raw: 1808870400, fmt: '2027-04-28' },   // later
      { raw: 1806278400, fmt: '2027-03-29' },   // earlier
    ],
    isEarningsDateEstimate: true,
  });
  const out = E.parseQuoteSummaryEarnings(json, '2026-11-16');
  assert.deepEqual(out.allDates, [...out.allDates].sort());
  assert.deepEqual(out.allDates, ['2027-03-29', '2027-04-28']);
  assert.equal(out.nextEarningsDate, '2027-03-29');
  assert.equal(out.isEstimate, true);
});

test('parseQuoteSummaryEarnings: nowYmd filters to on/after that date', () => {
  const json = payload({
    earningsDate: [
      { raw: 1769558400, fmt: '2026-01-28' },   // past relative to nowYmd
      { raw: 1808870400, fmt: '2027-04-28' },   // future
    ],
  });
  const out = E.parseQuoteSummaryEarnings(json, '2026-11-16');
  assert.equal(out.allDates.length, 2);
  assert.equal(out.nextEarningsDate, '2027-04-28');
});

test('parseQuoteSummaryEarnings: earningsDateNext (older nesting) is also collected', () => {
  const json = payload({ earningsDateNext: { raw: 1808870400, fmt: '2027-04-28' } });
  const out = E.parseQuoteSummaryEarnings(json);
  assert.equal(out.allDates.length, 1);
});

test('parseQuoteSummaryEarnings: garbage degrades to null, never throws', () => {
  assert.equal(E.parseQuoteSummaryEarnings(null), null);
  assert.equal(E.parseQuoteSummaryEarnings({}), null);
  assert.equal(E.parseQuoteSummaryEarnings({ quoteSummary: { result: [] } }), null);
  assert.equal(E.parseQuoteSummaryEarnings({ quoteSummary: { result: [{ calendarEvents: {} }] } }), null);
  assert.equal(E.parseQuoteSummaryEarnings({ quoteSummary: { result: [{ calendarEvents: { earnings: { earningsDate: [{ raw: 'not-a-number' }] } } }] } }), null);
  assert.equal(E.parseQuoteSummaryEarnings('a string'), null);
});

test('sessionsUntil: rough trading-day estimate, never negative, parses ymd', () => {
  const s = E.sessionsUntil('2027-01-28', '2026-11-16');
  assert.ok(s >= 48 && s <= 58, `expected ~53, got ${s}`);
  assert.equal(E.sessionsUntil('2026-11-16', '2027-01-28'), 0); // inverted → 0
  assert.equal(E.sessionsUntil('garbage', '2026-11-16'), 0);
});
