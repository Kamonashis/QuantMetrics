const test = require('node:test');
const assert = require('node:assert/strict');
const {
  computeWheelXAxisRange,
  buildHmmRegimeShapes,
  extractTracesXDomain,
  computeVisibleYBounds,
  computeInitialYBounds,
  resolveResponsiveChartHeight,
  formatChartDisplayTitle,
  parsePlotlyXCoord,
  formatPlotlyXCoord,
} = require('../../.cache/test-build/plotly-theme.js');

test('computeWheelXAxisRange: first plain wheel scroll from full span engages 50% scrollable viewport', () => {
  const resRight = computeWheelXAxisRange({
    currentRange: [0, 1000],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: 120,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(resRight.changed, true);
  assert.deepEqual(resRight.nextRange, [500, 1000]);

  const resLeft = computeWheelXAxisRange({
    currentRange: [0, 1000],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: -120,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(resLeft.changed, true);
  assert.equal(resLeft.nextRange[1] - resLeft.nextRange[0], 500);
  assert.ok(resLeft.nextRange[1] < 1000);
});

test('computeWheelXAxisRange: plain wheel pans horizontally when zoomed and clamps at domain edges', () => {
  const panRight = computeWheelXAxisRange({
    currentRange: [200, 600],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: 100,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(panRight.changed, true);
  assert.equal(panRight.nextRange[0], 240);
  assert.equal(panRight.nextRange[1], 640);

  const clampLeft = computeWheelXAxisRange({
    currentRange: [10, 410],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: -200,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(clampLeft.changed, true);
  assert.deepEqual(clampLeft.nextRange, [0, 400]);

  const atEdgeNoOp = computeWheelXAxisRange({
    currentRange: [0, 400],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: -100,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(atEdgeNoOp.changed, false);
  assert.deepEqual(atEdgeNoOp.nextRange, [0, 400]);
});

test('computeWheelXAxisRange: Ctrl+Wheel zooms in and out anchored at cursorFrac', () => {
  const zoomIn = computeWheelXAxisRange({
    currentRange: [0, 1000],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: -100,
    isZoomGesture: true,
    cursorFrac: 0.25,
  });
  assert.equal(zoomIn.changed, true);
  const newSpan = zoomIn.nextRange[1] - zoomIn.nextRange[0];
  assert.ok(newSpan < 1000);
  // Anchor at 250 must remain at cursorFrac = 0.25 of the new range
  const anchorAfter = zoomIn.nextRange[0] + 0.25 * newSpan;
  assert.ok(Math.abs(anchorAfter - 250) < 1e-6);

  const zoomOutClamped = computeWheelXAxisRange({
    currentRange: [0, 1000],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: 100,
    isZoomGesture: true,
    cursorFrac: 0.5,
  });
  assert.equal(zoomOutClamped.changed, false);
  assert.deepEqual(zoomOutClamped.nextRange, [0, 1000]);
});

test('computeWheelXAxisRange: zero deltas are no-op', () => {
  const res = computeWheelXAxisRange({
    currentRange: [100, 500],
    fullDomain: [0, 1000],
    deltaX: 0,
    deltaY: 0,
    isZoomGesture: false,
    cursorFrac: 0.5,
  });
  assert.equal(res.changed, false);
  assert.deepEqual(res.nextRange, [100, 500]);
});

test('buildHmmRegimeShapes merges contiguous spans into native Plotly rect shapes', () => {
  const dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05'];
  const states = [0, 0, 2, 2, 1];
  const shapes = buildHmmRegimeShapes(dates, states, true);
  assert.equal(shapes.length, 3);
  assert.equal(shapes[0].type, 'rect');
  assert.equal(shapes[0].x0, '2026-01-01');
  assert.equal(shapes[0].x1, '2026-01-03');
  assert.equal(shapes[1].x0, '2026-01-03');
  assert.equal(shapes[1].x1, '2026-01-05');
  assert.equal(shapes[2].x0, '2026-01-05');

  const disabled = buildHmmRegimeShapes(dates, states, false);
  assert.deepEqual(disabled, []);
});

test('extractTracesXDomain & computeVisibleYBounds handle dates, numeric axes, and hidden traces', () => {
  const traces = [
    {
      type: 'scatter',
      x: ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
      y: [100, 120, 110, 200],
    },
    {
      type: 'scatter',
      x: ['2026-01-01', '2026-01-04'],
      y: [999, 999],
      visible: 'legendonly',
    },
    {
      type: 'bar',
      x: ['2026-01-01', '2026-01-04'],
      y: [5000000, 8000000],
      yaxis: 'y2',
    },
  ];
  const domain = extractTracesXDomain(traces, true);
  assert.ok(domain !== null);
  assert.equal(domain[0], parsePlotlyXCoord('2026-01-01', true));
  assert.equal(domain[1], parsePlotlyXCoord('2026-01-04', true));

  // Slice to first 3 dates -> y in [100, 120] ignoring hidden trace & y2 volume trace
  const xMin = parsePlotlyXCoord('2026-01-01', true);
  const xMax = parsePlotlyXCoord('2026-01-03', true);
  const yBounds = computeVisibleYBounds(traces, xMin, xMax, true, 0.1);
  assert.ok(yBounds !== null);
  assert.ok(Math.abs(yBounds[0] - 98) < 1e-6);
  assert.ok(Math.abs(yBounds[1] - 122) < 1e-6);

  // Round-trip date formatting
  const formatted = formatPlotlyXCoord(domain[0], true);
  assert.equal(formatted, '2026-01-01 00:00:00');
});

test('computeInitialYBounds tightly frames tozeroy price traces without forced zero baseline', () => {
  const priceAndVol = [
    {
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      x: ['2026-01-01', '2026-01-02', '2026-01-03'],
      y: [200, 250, 300],
      yaxis: 'y',
    },
    {
      type: 'bar',
      x: ['2026-01-01', '2026-01-02', '2026-01-03'],
      y: [40000000, 60000000, 50000000],
      yaxis: 'y2',
    },
  ];
  const bounds = computeInitialYBounds(priceAndVol, 0.06);
  assert.ok(bounds !== null);
  assert.equal(bounds[0], 194);
  assert.equal(bounds[1], 306);

  // Series starting cleanly at 0 (e.g. density curve) preserves 0 floor with upper headroom
  const density = [{ type: 'scatter', y: [0, 0.4, 1.0, 0.2, 0] }];
  const densityBounds = computeInitialYBounds(density, 0.06);
  assert.deepEqual(densityBounds, [0, 1.06]);

  // Pure bar charts & histograms return null so Plotly preserves standard zero baseline
  assert.equal(computeInitialYBounds([{ type: 'bar', y: [10, 20, 30] }]), null);
  assert.equal(computeInitialYBounds([{ type: 'histogram', x: [1, 2, 3] }]), null);
});

test('resolveResponsiveChartHeight maps tiers and legacy numeric heights to viewport-scaled CSS clamps', () => {
  assert.equal(resolveResponsiveChartHeight(320, '2d'), 'clamp(440px, 50vh, 520px)');
  assert.equal(resolveResponsiveChartHeight(255, '2d'), 'clamp(340px, 38vh, 420px)');
  assert.equal(resolveResponsiveChartHeight(170, '2d'), 'clamp(220px, 26vh, 280px)');
  assert.equal(resolveResponsiveChartHeight(460, '3d'), 'clamp(500px, 60vh, 640px)');
  assert.equal(resolveResponsiveChartHeight(undefined, '2d', 'primary'), 'clamp(440px, 50vh, 520px)');
  assert.equal(resolveResponsiveChartHeight('100%', '2d'), '100%');
});

test('formatChartDisplayTitle prefers explicit title and title-cases acronyms from exportFilename', () => {
  assert.equal(formatChartDisplayTitle('Historical Price', 'quantmetrics-price-history'), 'Historical Price');
  assert.equal(formatChartDisplayTitle(undefined, 'quantmetrics-iv-vs-rv'), 'IV Vs RV');
  assert.equal(formatChartDisplayTitle(undefined, 'volatility-surface-3d-svi'), 'Volatility Surface 3D SVI');
  assert.equal(formatChartDisplayTitle(undefined, 'quantmetrics-tcn-var-es'), 'TCN VaR ES');
});
