// Pure TypeScript theme, layout, HMM overlay, and 2D wheel pan/zoom math
// for the unified QuantMetrics Plotly.js charting system.
// Zero React/DOM dependencies — relative imports only for node --test compatibility.

import type { Config, Data, Layout, Shape } from 'plotly.js';
import {
  buildRegimeBandSpans,
  getHmmRegimeToken,
  type HmmStateIdx,
} from './hmm-regime-overlay';

export const QUANT_COLORS = {
  gold: '#C9A45C',
  goldLight: '#DEC084',
  goldDark: '#A6833F',
  positive: '#3FB27F',
  positiveDark: '#2F8F66',
  negative: '#E0655F',
  negativeDark: '#CF4B45',
  sky: '#5AA9E6',
  violet: '#A78BFA',
  amber: '#E8A849',
  cyan: '#38BDF8',
  slate: '#8C8C9A',
  mutedDark: '#6E6E7C',
  mutedLight: '#71717A',
} as const;

export interface PlotlyThemeTokens {
  isDark: boolean;
  text: string;
  textMuted: string;
  grid: string;
  zeroLine: string;
  spike: string;
  hoverBg: string;
  hoverBorder: string;
  hoverText: string;
  sliderBg: string;
  sliderBorder: string;
  sceneBg: string;
  sceneWall: string;
  modebarBg: string;
  modebarIcon: string;
  modebarActive: string;
}

export function getPlotlyThemeTokens(isDark: boolean): PlotlyThemeTokens {
  if (isDark) {
    return {
      isDark: true,
      text: '#D4D4DC',
      textMuted: '#8E8E9E',
      grid: 'rgba(255, 255, 255, 0.065)',
      zeroLine: 'rgba(255, 255, 255, 0.16)',
      spike: 'rgba(222, 192, 132, 0.48)',
      hoverBg: '#16161C',
      hoverBorder: 'rgba(222, 192, 132, 0.42)',
      hoverText: '#F4F4F7',
      sliderBg: 'rgba(23, 23, 28, 0.72)',
      sliderBorder: 'rgba(255, 255, 255, 0.10)',
      sceneBg: 'rgba(16, 16, 20, 0.0)',
      sceneWall: 'rgba(255, 255, 255, 0.03)',
      modebarBg: 'rgba(16, 16, 20, 0.86)',
      modebarIcon: '#9A9AA8',
      modebarActive: '#DEC084',
    };
  }
  return {
    isDark: false,
    text: '#27272A',
    textMuted: '#646470',
    grid: 'rgba(24, 24, 27, 0.075)',
    zeroLine: 'rgba(24, 24, 27, 0.18)',
    spike: 'rgba(166, 131, 63, 0.45)',
    hoverBg: '#FFFFFF',
    hoverBorder: 'rgba(166, 131, 63, 0.45)',
    hoverText: '#18181B',
    sliderBg: 'rgba(236, 234, 230, 0.72)',
    sliderBorder: 'rgba(24, 24, 27, 0.12)',
    sceneBg: 'rgba(255, 255, 255, 0.0)',
    sceneWall: 'rgba(24, 24, 27, 0.025)',
    modebarBg: 'rgba(255, 255, 255, 0.90)',
    modebarIcon: '#52525B',
    modebarActive: '#9A7632',
  };
}

export interface BaseLayoutOptions {
  plotType?: '2d' | '3d' | 'heatmap' | 'categorical';
  rangeSlider?: boolean;
  hovermode?: Layout['hovermode'];
  showLegend?: boolean;
  uirevision?: string;
  margin?: Partial<Layout['margin']>;
  xAxisTitle?: string;
  yAxisTitle?: string;
  yTickPrefix?: string;
  yTickSuffix?: string;
  yTickFormat?: string;
  xTickSuffix?: string;
  xTickFormat?: string;
}

export function buildBasePlotlyLayout(
  isDark: boolean,
  opts: BaseLayoutOptions = {}
): Partial<Layout> {
  const t = getPlotlyThemeTokens(isDark);
  const plotType = opts.plotType ?? '2d';
  const is3d = plotType === '3d';
  const isHeatmap = plotType === 'heatmap';

  const defaultMargin = is3d
    ? { l: 0, r: 0, t: 4, b: 0, pad: 0 }
    : {
        l: 42,
        r: 8,
        t: opts.showLegend ? 26 : 8,
        b: opts.rangeSlider ? 12 : 26,
        pad: 2,
      };

  const base: Partial<Layout> = {
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'var(--font-mono), "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace',
      size: 11,
      color: t.text,
    },
    margin: { ...defaultMargin, ...(opts.margin ?? {}) },
    hovermode:
      opts.hovermode ??
      (is3d || isHeatmap ? 'closest' : 'x unified'),
    hoverlabel: {
      bgcolor: t.hoverBg,
      bordercolor: t.hoverBorder,
      font: {
        family: 'var(--font-mono), "IBM Plex Mono", ui-monospace, SFMono-Regular, monospace',
        size: 11,
        color: t.hoverText,
      },
      namelength: 28,
    },
    showlegend: opts.showLegend ?? false,
    legend: {
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: 1.03,
      yanchor: 'bottom',
      bgcolor: 'rgba(0,0,0,0)',
      borderwidth: 0,
      font: { size: 10.5, color: t.textMuted },
      itemsizing: 'constant',
    },
    uirevision: opts.uirevision ?? 'qm-default',
    modebar: {
      bgcolor: t.modebarBg,
      color: t.modebarIcon,
      activecolor: t.modebarActive,
      orientation: 'h',
    },
  };

  if (!is3d) {
    base.dragmode = 'zoom';
    base.xaxis = {
      gridcolor: t.grid,
      zerolinecolor: t.zeroLine,
      zeroline: false,
      linecolor: t.grid,
      tickfont: { size: 10.5, color: t.textMuted },
      title: opts.xAxisTitle
        ? { text: opts.xAxisTitle, font: { size: 10.5, color: t.textMuted }, standoff: 8 }
        : undefined,
      showspikes: !isHeatmap,
      spikemode: 'across',
      spikesnap: 'cursor',
      spikedash: 'dot',
      spikethickness: 1,
      spikecolor: t.spike,
      ticksuffix: opts.xTickSuffix,
      tickformat: opts.xTickFormat,
      automargin: true,
      rangeslider: opts.rangeSlider
        ? {
            visible: true,
            thickness: 0.075,
            bgcolor: t.sliderBg,
            bordercolor: t.sliderBorder,
            borderwidth: 1,
          }
        : { visible: false },
    };

    base.yaxis = {
      gridcolor: t.grid,
      zerolinecolor: t.zeroLine,
      zeroline: false,
      linecolor: t.grid,
      tickfont: { size: 10.5, color: t.textMuted },
      title: opts.yAxisTitle
        ? { text: opts.yAxisTitle, font: { size: 10.5, color: t.textMuted }, standoff: 8 }
        : undefined,
      showspikes: false,
      tickprefix: opts.yTickPrefix,
      ticksuffix: opts.yTickSuffix,
      tickformat: opts.yTickFormat,
      automargin: true,
      fixedrange: false,
    };
  }

  return base;
}

export function buildPlotlyConfig(
  plotType: '2d' | '3d' | 'heatmap' | 'categorical' = '2d',
  exportFilename = 'quantmetrics-chart'
): Partial<Config> {
  const is3d = plotType === '3d';
  return {
    responsive: true,
    displaylogo: false,
    // Always mount modebar; CSS reveals it smoothly on hover/focus-within
    displayModeBar: true,
    // 3D keeps native scroll zoom; 2D uses our custom wheel horizontal-pan + Ctrl-wheel zoom
    scrollZoom: is3d,
    doubleClick: 'reset+autosize',
    toImageButtonOptions: {
      format: 'png',
      filename: exportFilename,
      scale: 2,
    },
    modeBarButtonsToRemove: is3d
      ? ['sendDataToCloud']
      : ['lasso2d', 'select2d', 'sendDataToCloud'],
  };
}

/**
 * Translates contiguous causal 3-state HMM volatility regime spans into native
 * Plotly background `vrect` shapes that pan, zoom, and export in lockstep.
 */
export function buildHmmRegimeShapes(
  xValues: readonly (string | number)[],
  states: readonly (HmmStateIdx | null | undefined)[],
  enabled = true,
  opacity = 0.13
): Partial<Shape>[] {
  if (!enabled || xValues.length < 2 || states.length === 0) return [];
  const len = Math.min(xValues.length, states.length);
  const spans = buildRegimeBandSpans(states.slice(0, len));
  const shapes: Partial<Shape>[] = [];

  for (const span of spans) {
    const x0 = xValues[span.startIdx];
    const x1 = xValues[Math.min(len - 1, span.endIdx + 1)] ?? xValues[span.endIdx];
    if (x0 === undefined || x1 === undefined) continue;
    const token = getHmmRegimeToken(span.state);
    shapes.push({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0,
      x1,
      y0: 0,
      y1: 1,
      fillcolor: `rgba(${token.rgb}, ${opacity})`,
      line: { width: 0 },
      layer: 'below',
    });
  }
  return shapes;
}

// ---------------------------------------------------------------------------
// PURE 2D WHEEL HORIZONTAL PAN + CTRL-WHEEL ZOOM MATH (UNIT-TESTABLE)
// ---------------------------------------------------------------------------

export interface WheelAxisInput {
  /** Current visible [min, max] in numeric coordinates (ms for date axes, number for linear) */
  currentRange: [number, number];
  /** Full data domain [min, max] in the same numeric coordinates */
  fullDomain: [number, number];
  /** Wheel deltaX (horizontal trackpad/tilt) */
  deltaX: number;
  /** Wheel deltaY (vertical scroll wheel) */
  deltaY: number;
  /** True when Ctrl or Cmd is held -> cursor-anchored zoom */
  isZoomGesture: boolean;
  /** Cursor horizontal position within the plot area in [0, 1] */
  cursorFrac: number;
  /** Minimum allowable zoomed span as fraction of full span (default 0.025) */
  minSpanFrac?: number;
}

export interface WheelAxisResult {
  nextRange: [number, number];
  changed: boolean;
}

/**
 * Computes the next x-axis range for a 2D mouse wheel gesture:
 * - Ctrl/Cmd + Wheel: smooth zoom in/out anchored at `cursorFrac`.
 * - Plain Wheel: horizontal pan along x-axis (prefers deltaX if horizontal swipe,
 *   else maps vertical wheel deltaY to horizontal scroll). If currently viewing
 *   >= 98% of the full domain, automatically engages a 50% scrollable viewport
 *   anchored to the edge matching scroll direction so panning works on the very
 *   first wheel tick without requiring a prior manual zoom.
 */
export function computeWheelXAxisRange(input: WheelAxisInput): WheelAxisResult {
  const {
    currentRange,
    fullDomain,
    deltaX,
    deltaY,
    isZoomGesture,
    cursorFrac,
    minSpanFrac = 0.025,
  } = input;

  const fullMin = Math.min(fullDomain[0], fullDomain[1]);
  const fullMax = Math.max(fullDomain[0], fullDomain[1]);
  const fullSpan = fullMax - fullMin;
  if (!Number.isFinite(fullSpan) || fullSpan <= 0) {
    return { nextRange: currentRange, changed: false };
  }

  let curMin = Math.min(currentRange[0], currentRange[1]);
  let curMax = Math.max(currentRange[0], currentRange[1]);
  if (!Number.isFinite(curMin) || !Number.isFinite(curMax) || curMax <= curMin) {
    curMin = fullMin;
    curMax = fullMax;
  }

  const curSpan = curMax - curMin;
  const minSpan = Math.max(fullSpan * minSpanFrac, 1e-9);
  const clampedFrac = Math.max(0, Math.min(1, Number.isFinite(cursorFrac) ? cursorFrac : 0.5));

  if (isZoomGesture) {
    const rawDelta = Math.abs(deltaY) >= Math.abs(deltaX) ? deltaY : deltaX;
    if (rawDelta === 0) return { nextRange: [curMin, curMax], changed: false };
    // Positive deltaY = scroll down = zoom out; Negative deltaY = scroll up = zoom in
    const zoomFactor = rawDelta > 0 ? 1.18 : 1 / 1.18;
    const targetSpan = Math.max(minSpan, Math.min(fullSpan, curSpan * zoomFactor));
    const anchor = curMin + clampedFrac * curSpan;
    let nextMin = anchor - clampedFrac * targetSpan;
    let nextMax = nextMin + targetSpan;

    if (nextMin < fullMin) {
      nextMin = fullMin;
      nextMax = Math.min(fullMax, fullMin + targetSpan);
    }
    if (nextMax > fullMax) {
      nextMax = fullMax;
      nextMin = Math.max(fullMin, fullMax - targetSpan);
    }

    const changed = Math.abs(nextMin - curMin) > fullSpan * 1e-6 || Math.abs(nextMax - curMax) > fullSpan * 1e-6;
    return { nextRange: [nextMin, nextMax], changed };
  }

  // Plain wheel -> horizontal pan
  const primaryDelta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;
  if (primaryDelta === 0) return { nextRange: [curMin, curMax], changed: false };
  const dir = primaryDelta > 0 ? 1 : -1;

  // If currently at full span (>= 98%), step smoothly into a 50% scrollable viewport
  // so horizontal wheel scrolling responds immediately on the first flick.
  if (curSpan >= fullSpan * 0.98) {
    const windowSpan = fullSpan * 0.5;
    if (dir < 0) {
      // Scrolling left/up -> inspect left-to-mid historical segment first, shifting left from latest
      const nextMax = fullMax - fullSpan * 0.08;
      const nextMin = Math.max(fullMin, nextMax - windowSpan);
      return { nextRange: [nextMin, nextMin + windowSpan], changed: true };
    }
    // Scrolling right/down -> focus on most recent 50% window
    const nextMax = fullMax;
    const nextMin = fullMax - windowSpan;
    return { nextRange: [nextMin, nextMax], changed: true };
  }

  // Scale pan step by wheel magnitude (normalized ~100px per notch => 10% of visible span)
  const notches = Math.max(0.25, Math.min(3, Math.abs(primaryDelta) / 100));
  const shift = dir * curSpan * 0.1 * notches;

  let nextMin = curMin + shift;
  let nextMax = curMax + shift;

  if (nextMin < fullMin) {
    nextMin = fullMin;
    nextMax = Math.min(fullMax, fullMin + curSpan);
  }
  if (nextMax > fullMax) {
    nextMax = fullMax;
    nextMin = Math.max(fullMin, fullMax - curSpan);
  }

  const changed = Math.abs(nextMin - curMin) > fullSpan * 1e-6 || Math.abs(nextMax - curMax) > fullSpan * 1e-6;
  return { nextRange: [nextMin, nextMax], changed };
}

/**
 * Parses a Plotly axis coordinate (ISO date string, timestamp string, or number)
 * into a numeric value (ms epoch for date axes, raw number for linear axes).
 */
export function parsePlotlyXCoord(val: unknown, isDateAxis: boolean): number | null {
  if (typeof val === 'number' && Number.isFinite(val)) return val;
  if (typeof val === 'string') {
    if (isDateAxis) {
      const normalized = val.includes('T') || val.includes(' ') ? val : `${val}T00:00:00Z`;
      const ms = Date.parse(normalized);
      return Number.isFinite(ms) ? ms : null;
    }
    const num = Number(val);
    return Number.isFinite(num) ? num : null;
  }
  return null;
}

/**
 * Formats a numeric coordinate back to a Plotly axis range value.
 */
export function formatPlotlyXCoord(val: number, isDateAxis: boolean): string | number {
  if (!isDateAxis) return val;
  return new Date(val).toISOString().slice(0, 19).replace('T', ' ');
}

/**
 * Extracts overall [min, max] x-domain from an array of Plotly 2D traces.
 */
export function extractTracesXDomain(
  traces: readonly Data[],
  isDateAxis: boolean
): [number, number] | null {
  let min = Infinity;
  let max = -Infinity;

  for (const trace of traces) {
    const t = trace as { x?: unknown[]; visible?: boolean | 'legendonly' };
    if (t.visible === false || t.visible === 'legendonly') continue;
    const xs = t.x;
    if (!Array.isArray(xs) || xs.length === 0) continue;
    for (let i = 0; i < xs.length; i++) {
      const n = parsePlotlyXCoord(xs[i], isDateAxis);
      if (n !== null) {
        if (n < min) min = n;
        if (n > max) max = n;
      }
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  return [min, max];
}

/**
 * Computes tight visible y-bounds [yMin, yMax] with padding for primary y-axis traces
 * within the given numeric x-window [xMin, xMax].
 */
export function computeVisibleYBounds(
  traces: readonly Data[],
  xMin: number,
  xMax: number,
  isDateAxis: boolean,
  padFrac = 0.07
): [number, number] | null {
  let minY = Infinity;
  let maxY = -Infinity;

  for (const trace of traces) {
    const t = trace as {
      x?: unknown[];
      y?: unknown[];
      yaxis?: string;
      visible?: boolean | 'legendonly';
      type?: string;
    };
    if (t.visible === false || t.visible === 'legendonly') continue;
    // Only auto-scale primary y-axis traces
    if (t.yaxis && t.yaxis !== 'y' && t.yaxis !== 'y1') continue;
    if (t.type === 'histogram') continue;
    const xs = t.x;
    const ys = t.y;
    if (!Array.isArray(xs) || !Array.isArray(ys)) continue;
    const len = Math.min(xs.length, ys.length);
    for (let i = 0; i < len; i++) {
      const xv = parsePlotlyXCoord(xs[i], isDateAxis);
      if (xv === null || xv < xMin || xv > xMax) continue;
      const yv = typeof ys[i] === 'number' ? (ys[i] as number) : Number(ys[i]);
      if (Number.isFinite(yv)) {
        if (yv < minY) minY = yv;
        if (yv > maxY) maxY = yv;
      }
    }
  }

  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return null;
  const span = Math.max(maxY - minY, Math.abs(maxY) * 0.02 || 1e-4);
  return [minY - span * padFrac, maxY + span * padFrac];
}

const SKIP_AUTOFIT_TYPES = new Set([
  'histogram',
  'heatmap',
  'surface',
  'pie',
  'box',
  'violin',
  'scatter3d',
  'mesh3d',
  'contour',
]);

/**
 * Computes tight initial y-axis bounds across all visible primary y-axis traces
 * so area-filled series (`fill: 'tozeroy'`) and line/scatter charts immediately fill
 * vertical real estate on first paint instead of forcing a dead-space zero baseline.
 *
 * Returns `null` when traces are purely categorical/bar/histogram/heatmap/3D so Plotly's
 * native domain framing is preserved.
 */
export function computeInitialYBounds(
  traces: readonly Data[],
  padFrac = 0.06
): [number, number] | null {
  let minY = Infinity;
  let maxY = -Infinity;
  let hasContinuousTrace = false;

  for (const trace of traces) {
    const t = trace as {
      type?: string;
      orientation?: string;
      yaxis?: string;
      visible?: boolean | 'legendonly';
      y?: unknown[];
      high?: unknown[];
      low?: unknown[];
    };
    if (t.visible === false || t.visible === 'legendonly') continue;
    if (t.yaxis && t.yaxis !== 'y' && t.yaxis !== 'y1') continue;

    const kind = (t.type ?? 'scatter').toLowerCase();
    if (SKIP_AUTOFIT_TYPES.has(kind)) return null;
    if (kind === 'bar' && t.orientation === 'h') return null;

    if (kind !== 'bar') {
      hasContinuousTrace = true;
    }

    const arrays: Array<unknown[] | undefined> =
      kind === 'candlestick' || kind === 'ohlc'
        ? [t.low, t.high, t.y]
        : [t.y];

    for (const arr of arrays) {
      if (!Array.isArray(arr)) continue;
      for (let i = 0; i < arr.length; i++) {
        const raw = arr[i];
        if (raw === null || raw === undefined || raw === '') continue;
        const yv = typeof raw === 'number' ? raw : Number(raw);
        if (Number.isFinite(yv)) {
          if (yv < minY) minY = yv;
          if (yv > maxY) maxY = yv;
        }
      }
    }
  }

  // Pure bar charts keep standard zero-baseline proportionality
  if (!hasContinuousTrace) return null;
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return null;

  const span = Math.max(maxY - minY, Math.abs(maxY) * 0.02 || 1e-4);
  const pad = span * padFrac;
  // Non-negative series that legitimately start at 0 (e.g. density curves) keep a clean 0 floor
  const lower = minY === 0 ? 0 : minY - pad;
  const upper = maxY + pad;
  return [lower, upper];
}

export type ChartHeightTier = 'primary' | 'secondary' | 'compact' | '3d';

/**
 * Resolves embedded chart height into generous viewport-scaled CSS heights
 * (440-520px for primary charts, 500-640px for 3D WebGL surfaces, proportional
 * scaling for secondary/compact strips).
 */
export function resolveResponsiveChartHeight(
  height: number | string | undefined,
  plotType: '2d' | '3d' | 'heatmap' | 'categorical' = '2d',
  size?: ChartHeightTier
): string {
  if (size === '3d') return 'clamp(500px, 60vh, 640px)';
  if (size === 'primary') return 'clamp(440px, 50vh, 520px)';
  if (size === 'secondary') return 'clamp(340px, 38vh, 420px)';
  if (size === 'compact') return 'clamp(220px, 26vh, 280px)';

  if (typeof height === 'string') return height;
  if (plotType === '3d') return 'clamp(500px, 60vh, 640px)';

  if (typeof height === 'number' && Number.isFinite(height)) {
    if (height >= 380) {
      const minPx = Math.max(Math.round(height), 460);
      const maxPx = Math.max(Math.round(height + 90), 560);
      return `clamp(${minPx}px, 54vh, ${maxPx}px)`;
    }
    if (height >= 280) return 'clamp(440px, 50vh, 520px)';
    if (height >= 215) return 'clamp(340px, 38vh, 420px)';
    if (height >= 120) return 'clamp(220px, 26vh, 280px)';
    return `${height}px`;
  }

  return 'clamp(440px, 50vh, 520px)';
}

const CHART_TITLE_ACRONYMS = new Set([
  'IV',
  'RV',
  'GEX',
  'TCN',
  'HMM',
  'ACF',
  'PACF',
  'CDF',
  'PDF',
  'MLE',
  'HAR',
  'SVI',
  'SSVI',
  'OI',
  '3D',
  '2D',
  'CUSUM',
  'VAR',
  'ES',
  'GARCH',
  'EGARCH',
  'EWMA',
  'MST',
  'RMT',
  'PCA',
  'VRP',
  'DTE',
  'FNO',
  'OLS',
  'QQ',
]);

/**
 * Formats a clean human-readable display title for the fullscreen modal header,
 * preferring an explicit `title` prop and falling back cleanly to `exportFilename`.
 */
export function formatChartDisplayTitle(
  title: string | undefined,
  exportFilename: string | undefined
): string {
  if (title && title.trim().length > 0) return title.trim();
  const raw = (exportFilename ?? 'Interactive Chart')
    .replace(/^quantmetrics[-_]/i, '')
    .replace(/[-_]+/g, ' ')
    .trim();
  if (!raw) return 'Interactive Chart';
  return raw
    .split(/\s+/)
    .map((word) => {
      const upper = word.toUpperCase();
      if (CHART_TITLE_ACRONYMS.has(upper)) return upper === 'VAR' ? 'VaR' : upper;
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(' ');
}
