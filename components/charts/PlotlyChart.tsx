'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Maximize2, Minimize2 } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { Config, Data, Layout, ModeBarButtonAny, PlotlyHTMLElement } from 'plotly.js';
import {
  buildBasePlotlyLayout,
  buildPlotlyConfig,
  computeInitialYBounds,
  computeVisibleYBounds,
  computeWheelXAxisRange,
  extractTracesXDomain,
  formatChartDisplayTitle,
  formatPlotlyXCoord,
  parsePlotlyXCoord,
  resolveResponsiveChartHeight,
  type BaseLayoutOptions,
  type ChartHeightTier,
} from '@/lib/plotly-theme';

type PlotlyModule = typeof import('plotly.js');

let plotlySingletonPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  if (!plotlySingletonPromise) {
    plotlySingletonPromise = import('plotly.js-dist-min').then((mod) => {
      return ((mod as unknown as { default?: PlotlyModule }).default ?? mod) as unknown as PlotlyModule;
    });
  }
  return plotlySingletonPromise;
}

/**
 * Observes `<html class="dark">` reactively with deterministic dark-first SSR parity.
 */
export function usePlotlyDarkTheme(): boolean {
  const [isDark, setIsDark] = useState<boolean>(true);

  useEffect(() => {
    const sync = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    const timer = setTimeout(sync, 0);
    const observer = new MutationObserver(sync);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, []);

  return isDark;
}

interface InternalPlotlyGraphDiv extends PlotlyHTMLElement {
  _fullLayout?: {
    xaxis?: {
      type?: string;
      range?: [unknown, unknown];
      _offset?: number;
      _length?: number;
      fixedrange?: boolean;
    };
    yaxis?: {
      range?: [unknown, unknown];
      _offset?: number;
      _length?: number;
    };
    scene?: {
      camera?: Record<string, unknown>;
    };
  };
  removeListener?: (event: string, listener: (...args: unknown[]) => void) => void;
}

interface ChartViewSnapshot {
  xRange?: [unknown, unknown];
  yRange?: [unknown, unknown];
  camera?: Record<string, unknown>;
}

function captureGraphView(gd: InternalPlotlyGraphDiv | null): ChartViewSnapshot | null {
  const fl = gd?._fullLayout;
  if (!fl) return null;
  const snap: ChartViewSnapshot = {};
  if (Array.isArray(fl.xaxis?.range) && fl.xaxis.range.length === 2) {
    snap.xRange = [fl.xaxis.range[0], fl.xaxis.range[1]];
  }
  if (Array.isArray(fl.yaxis?.range) && fl.yaxis.range.length === 2) {
    snap.yRange = [fl.yaxis.range[0], fl.yaxis.range[1]];
  }
  if (fl.scene?.camera) {
    try {
      snap.camera = JSON.parse(JSON.stringify(fl.scene.camera)) as Record<string, unknown>;
    } catch {
      // Ignore non-serializable camera edge case
    }
  }
  return snap.xRange || snap.yRange || snap.camera ? snap : null;
}

// Modebar icon definitions consumed by Plotly's internal modebar button API
const MODEBAR_EXPAND_ICON = {
  width: 24,
  height: 24,
  path: 'M15 3h6v6h-2V6.41l-4.29 4.3-1.42-1.42L17.59 5H15V3zM3 15h2v2.59l4.29-4.3 1.42 1.42L6.41 19H9v2H3v-6z',
};

const MODEBAR_COMPRESS_ICON = {
  width: 24,
  height: 24,
  path: 'M4 14h6v6H8v-2.59l-4.29 4.3-1.42-1.42L6.59 16H4v-2zm16-4h-6V4h2v2.59l4.29-4.3 1.42 1.42L17.41 8H20v2z',
};

export interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  plotType?: '2d' | '3d' | 'heatmap' | 'categorical';
  rangeSlider?: boolean;
  hovermode?: Layout['hovermode'];
  showLegend?: boolean;
  uirevisionKey?: string;
  exportFilename?: string;
  enableWheelPan?: boolean;
  autoScaleYOnWheel?: boolean;
  /** Automatically fit primary y-axis tightly to data on load & reset (default true) */
  autoFitYOnLoad?: boolean;
  /** Enable fullscreen button & overlay modal (default true) */
  allowFullscreen?: boolean;
  /** Semantic height tier overriding numeric height scaling */
  size?: ChartHeightTier;
  height?: number | string;
  className?: string;
  /** Optional display title shown in the fullscreen overlay header */
  title?: string;
  /** Optional subtitle shown beside the fullscreen overlay title */
  subtitle?: string;
  xAxisTitle?: string;
  yAxisTitle?: string;
  yTickPrefix?: string;
  yTickSuffix?: string;
  yTickFormat?: string;
  xTickSuffix?: string;
  xTickFormat?: string;
  margin?: Partial<Layout['margin']>;
  onHover?: (event: unknown) => void;
  onUnhover?: () => void;
  onClick?: (event: unknown) => void;
  onRelayout?: (event: Record<string, unknown>) => void;
  /** Optional subtle interaction hint rendered below chart when idle */
  showWheelHint?: boolean;
}

function deepMergeAxis(
  baseAxis: Record<string, unknown> | undefined,
  overrideAxis: Record<string, unknown> | undefined
): Record<string, unknown> | undefined {
  if (!baseAxis && !overrideAxis) return undefined;
  if (!overrideAxis) return baseAxis;
  if (!baseAxis) return overrideAxis;
  return {
    ...baseAxis,
    ...overrideAxis,
    title:
      typeof overrideAxis.title === 'object' && typeof baseAxis.title === 'object'
        ? { ...(baseAxis.title as object), ...(overrideAxis.title as object) }
        : overrideAxis.title ?? baseAxis.title,
    rangeslider:
      typeof overrideAxis.rangeslider === 'object' && typeof baseAxis.rangeslider === 'object'
        ? { ...(baseAxis.rangeslider as object), ...(overrideAxis.rangeslider as object) }
        : overrideAxis.rangeslider ?? baseAxis.rangeslider,
  };
}

interface PlotlyCanvasProps {
  data: Data[];
  layout: Partial<Layout>;
  config: Partial<Config>;
  plotType: '2d' | '3d' | 'heatmap' | 'categorical';
  enableWheelPan: boolean;
  autoScaleYOnWheel: boolean;
  initialYBounds: [number, number] | null;
  viewSnapshot: ChartViewSnapshot | null;
  viewSnapshotNonce: number;
  isActive: boolean;
  graphRef: React.MutableRefObject<InternalPlotlyGraphDiv | null>;
  plotlyModRef: React.MutableRefObject<PlotlyModule | null>;
  onHover?: (event: unknown) => void;
  onUnhover?: () => void;
  onClick?: (event: unknown) => void;
  onRelayout?: (event: Record<string, unknown>) => void;
}

function PlotlyCanvas({
  data,
  layout,
  config,
  plotType,
  enableWheelPan,
  autoScaleYOnWheel,
  initialYBounds,
  viewSnapshot,
  viewSnapshotNonce,
  isActive,
  graphRef,
  plotlyModRef,
  onHover,
  onUnhover,
  onClick,
  onRelayout,
}: PlotlyCanvasProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  const dataRef = useRef(data);
  const onHoverRef = useRef(onHover);
  const onUnhoverRef = useRef(onUnhover);
  const onClickRef = useRef(onClick);
  const onRelayoutRef = useRef(onRelayout);
  const autoScaleYRef = useRef(autoScaleYOnWheel);
  const initialYBoundsRef = useRef(initialYBounds);
  const isActiveRef = useRef(isActive);
  const internalRelayoutRef = useRef(false);
  const appliedSnapshotNonceRef = useRef<number>(-1);

  useEffect(() => {
    dataRef.current = data;
    onHoverRef.current = onHover;
    onUnhoverRef.current = onUnhover;
    onClickRef.current = onClick;
    onRelayoutRef.current = onRelayout;
    autoScaleYRef.current = autoScaleYOnWheel;
    initialYBoundsRef.current = initialYBounds;
    isActiveRef.current = isActive;
  });

  // Mount / reactively update Plotly figure and force immediate post-paint resize settlement
  useEffect(() => {
    let cancelled = false;
    let rafId: number | null = null;
    let settleTimer: ReturnType<typeof setTimeout> | null = null;
    const el = plotRef.current;
    if (!el) return;

    void loadPlotly().then((Plotly) => {
      if (cancelled || !plotRef.current) return;
      plotlyModRef.current = Plotly;
      const gd = plotRef.current as unknown as InternalPlotlyGraphDiv;
      graphRef.current = gd;

      void Plotly.react(gd, data, layout, config).then(() => {
        if (cancelled || !plotRef.current) return;
        setReady(true);

        // Ensure Plotly measures final post-settlement container width & height immediately
        const resizeNow = () => {
          const target = plotRef.current;
          if (!cancelled && target && target.offsetWidth > 0 && target.offsetHeight > 0) {
            try {
              Plotly.Plots.resize(target);
            } catch {
              // Ignore resize during unmount
            }
          }
        };

        rafId = requestAnimationFrame(resizeNow);
        settleTimer = setTimeout(resizeNow, 150);
      });
    });

    return () => {
      cancelled = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (settleTimer !== null) clearTimeout(settleTimer);
    };
  }, [data, layout, config, graphRef, plotlyModRef]);

  // Apply transferred zoom/pan/3D-camera snapshot when entering or exiting fullscreen
  useEffect(() => {
    if (!ready || !viewSnapshot || viewSnapshotNonce === appliedSnapshotNonceRef.current) return;
    const gd = graphRef.current;
    const Plotly = plotlyModRef.current;
    if (!gd || !Plotly) return;

    appliedSnapshotNonceRef.current = viewSnapshotNonce;
    const update: Record<string, unknown> = {};
    if (plotType === '3d') {
      if (viewSnapshot.camera) {
        update['scene.camera'] = viewSnapshot.camera;
      }
    } else {
      if (viewSnapshot.xRange) {
        update['xaxis.range'] = viewSnapshot.xRange;
        update['xaxis.autorange'] = false;
      }
      if (viewSnapshot.yRange) {
        update['yaxis.range'] = viewSnapshot.yRange;
        update['yaxis.autorange'] = false;
      }
    }

    if (Object.keys(update).length > 0) {
      internalRelayoutRef.current = true;
      void Promise.resolve(Plotly.relayout(gd, update as Partial<Layout>)).finally(() => {
        internalRelayoutRef.current = false;
      });
    }
  }, [ready, viewSnapshot, viewSnapshotNonce, plotType, graphRef, plotlyModRef]);

  // Bind Plotly events: hover, click, relayout (including double-click tight y-reset & slider y-fit)
  useEffect(() => {
    const el = plotRef.current as unknown as InternalPlotlyGraphDiv | null;
    if (!el || !ready) return;

    const handleHover = (ev: unknown) => onHoverRef.current?.(ev);
    const handleUnhover = () => onUnhoverRef.current?.();
    const handleClick = (ev: unknown) => onClickRef.current?.(ev);
    const handleRelayout = (ev: Record<string, unknown>) => {
      onRelayoutRef.current?.(ev);

      if (internalRelayoutRef.current || plotType !== '2d') return;
      const Plotly = plotlyModRef.current;
      if (!Plotly || !el._fullLayout?.xaxis) return;

      // Double-click / Reset Axes -> re-apply tight initial y-bounds instead of reverting to forced-zero
      if (ev['yaxis.autorange'] === true && initialYBoundsRef.current) {
        internalRelayoutRef.current = true;
        void Promise.resolve(
          Plotly.relayout(el, {
            'yaxis.range': initialYBoundsRef.current,
            'yaxis.autorange': false,
          } as Partial<Layout>)
        ).finally(() => {
          internalRelayoutRef.current = false;
        });
        return;
      }

      // Range-slider scrub or horizontal-only zoom -> auto-fit visible y-bounds smoothly
      const xRangeChanged =
        ev['xaxis.range[0]'] !== undefined ||
        ev['xaxis.range[1]'] !== undefined ||
        Array.isArray(ev['xaxis.range']);
      const yRangeExplicit =
        ev['yaxis.range[0]'] !== undefined ||
        ev['yaxis.range[1]'] !== undefined ||
        Array.isArray(ev['yaxis.range']);

      if (xRangeChanged && !yRangeExplicit && autoScaleYRef.current) {
        const xa = el._fullLayout.xaxis;
        const isDateAxis = xa.type === 'date';
        const rawRange = Array.isArray(ev['xaxis.range'])
          ? (ev['xaxis.range'] as [unknown, unknown])
          : ev['xaxis.range[0]'] !== undefined && ev['xaxis.range[1]'] !== undefined
          ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']]
          : xa.range;
        if (rawRange && rawRange.length === 2) {
          const r0 = parsePlotlyXCoord(rawRange[0], isDateAxis);
          const r1 = parsePlotlyXCoord(rawRange[1], isDateAxis);
          if (r0 !== null && r1 !== null && r1 > r0) {
            const traces = el.data ?? dataRef.current;
            const yBounds = computeVisibleYBounds(traces, r0, r1, isDateAxis);
            if (yBounds) {
              internalRelayoutRef.current = true;
              void Promise.resolve(
                Plotly.relayout(el, {
                  'yaxis.range': yBounds,
                  'yaxis.autorange': false,
                } as Partial<Layout>)
              ).finally(() => {
                internalRelayoutRef.current = false;
              });
            }
          }
        }
      }
    };

    if (typeof el.on === 'function') {
      el.on('plotly_hover', handleHover as never);
      el.on('plotly_unhover', handleUnhover as never);
      el.on('plotly_click', handleClick as never);
      el.on('plotly_relayout', handleRelayout as never);
    }

    return () => {
      if (typeof el.removeListener === 'function') {
        el.removeListener('plotly_hover', handleHover as never);
        el.removeListener('plotly_unhover', handleUnhover as never);
        el.removeListener('plotly_click', handleClick as never);
        el.removeListener('plotly_relayout', handleRelayout as never);
      }
    };
  }, [ready, plotType, plotlyModRef]);

  // Clean up WebGL/DOM resources on unmount
  useEffect(() => {
    const el = plotRef.current;
    return () => {
      if (graphRef.current === (el as unknown as InternalPlotlyGraphDiv)) {
        graphRef.current = null;
      }
      if (el && plotlyModRef.current) {
        try {
          plotlyModRef.current.purge(el);
        } catch {
          // Ignore cleanup errors if element already detached
        }
      }
    };
  }, [graphRef, plotlyModRef]);

  // ResizeObserver + window resize listener (re-bound once `ready` so initial settlement never drops)
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !ready) return;

    let timer: ReturnType<typeof setTimeout> | null = null;
    const scheduleResize = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        const el = plotRef.current;
        const Plotly = plotlyModRef.current;
        if (el && Plotly && el.offsetWidth > 0 && el.offsetHeight > 0) {
          try {
            Plotly.Plots.resize(el);
          } catch {
            // Safe ignore during unmount transition
          }
        }
      }, 30);
    };

    const ro = new ResizeObserver(scheduleResize);
    ro.observe(host);
    window.addEventListener('resize', scheduleResize);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', scheduleResize);
      if (timer) clearTimeout(timer);
    };
  }, [ready, plotlyModRef]);

  // Non-passive wheel handler for 2D charts:
  // - Plain mouse wheel -> horizontal pan along x-axis
  // - Ctrl/Cmd + Wheel -> cursor-anchored x-axis zoom + tight y auto-fit
  useEffect(() => {
    if (plotType !== '2d' || !enableWheelPan) return;
    const host = hostRef.current;
    if (!host) return;

    const onWheel = (e: WheelEvent) => {
      if (!isActiveRef.current) return;
      const el = plotRef.current as unknown as InternalPlotlyGraphDiv | null;
      const Plotly = plotlyModRef.current;
      if (!el || !Plotly || !el._fullLayout?.xaxis) return;

      const xa = el._fullLayout.xaxis;
      const ya = el._fullLayout.yaxis;
      if (xa.fixedrange || xa.type === 'category' || xa.type === 'multicategory') return;

      const rect = el.getBoundingClientRect();
      const relX = e.clientX - rect.left;
      const relY = e.clientY - rect.top;
      const xOffset = xa._offset ?? 0;
      const xLen = xa._length ?? rect.width;
      const yOffset = ya?._offset ?? 0;
      const yLen = ya?._length ?? rect.height;

      // Only intercept wheel events inside the plot canvas so outer page scrolls naturally
      if (
        relX < xOffset ||
        relX > xOffset + xLen ||
        relY < yOffset ||
        relY > yOffset + yLen
      ) {
        return;
      }

      const isDateAxis = xa.type === 'date';
      const traces = el.data ?? dataRef.current;
      const fullDomain = extractTracesXDomain(traces, isDateAxis);
      if (!fullDomain) return;

      const rawRange = xa.range;
      const r0 = rawRange ? parsePlotlyXCoord(rawRange[0], isDateAxis) : fullDomain[0];
      const r1 = rawRange ? parsePlotlyXCoord(rawRange[1], isDateAxis) : fullDomain[1];
      if (r0 === null || r1 === null) return;

      const cursorFrac = xLen > 0 ? (relX - xOffset) / xLen : 0.5;
      const isZoomGesture = e.ctrlKey || e.metaKey;

      const result = computeWheelXAxisRange({
        currentRange: [r0, r1],
        fullDomain,
        deltaX: e.deltaX,
        deltaY: e.deltaY,
        isZoomGesture,
        cursorFrac,
      });

      e.preventDefault();
      e.stopPropagation();

      if (!result.changed) return;

      const nextX0 = formatPlotlyXCoord(result.nextRange[0], isDateAxis);
      const nextX1 = formatPlotlyXCoord(result.nextRange[1], isDateAxis);

      const update: Record<string, unknown> = {
        'xaxis.range': [nextX0, nextX1],
        'xaxis.autorange': false,
      };

      if (autoScaleYRef.current) {
        const yBounds = computeVisibleYBounds(
          traces,
          result.nextRange[0],
          result.nextRange[1],
          isDateAxis
        );
        if (yBounds) {
          update['yaxis.range'] = yBounds;
          update['yaxis.autorange'] = false;
        }
      }

      internalRelayoutRef.current = true;
      void Promise.resolve(Plotly.relayout(el, update as Partial<Layout>)).finally(() => {
        internalRelayoutRef.current = false;
      });
    };

    host.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      host.removeEventListener('wheel', onWheel);
    };
  }, [plotType, enableWheelPan, plotlyModRef]);

  return (
    <div ref={hostRef} className="relative w-full h-full">
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg border border-dashed border-[var(--hairline)] bg-[var(--bg-inset)]/30">
          <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
            Initializing interactive chart…
          </span>
        </div>
      )}
      <div ref={plotRef} className="w-full h-full" />
    </div>
  );
}

export function PlotlyChart({
  data,
  layout,
  config,
  plotType = '2d',
  rangeSlider = false,
  hovermode,
  showLegend = false,
  uirevisionKey,
  exportFilename = 'quantmetrics-chart',
  enableWheelPan = true,
  autoScaleYOnWheel = true,
  autoFitYOnLoad = true,
  allowFullscreen = true,
  size,
  height = 320,
  className = '',
  title,
  subtitle,
  xAxisTitle,
  yAxisTitle,
  yTickPrefix,
  yTickSuffix,
  yTickFormat,
  xTickSuffix,
  xTickFormat,
  margin,
  onHover,
  onUnhover,
  onClick,
  onRelayout,
  showWheelHint = false,
}: PlotlyChartProps) {
  const isDark = usePlotlyDarkTheme();
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [fsSnapshot, setFsSnapshot] = useState<ChartViewSnapshot | null>(null);
  const [fsSnapshotNonce, setFsSnapshotNonce] = useState(0);
  const [embeddedSnapshot, setEmbeddedSnapshot] = useState<ChartViewSnapshot | null>(null);
  const [embeddedSnapshotNonce, setEmbeddedSnapshotNonce] = useState(0);

  const embeddedGraphRef = useRef<InternalPlotlyGraphDiv | null>(null);
  const fullscreenGraphRef = useRef<InternalPlotlyGraphDiv | null>(null);
  const embeddedPlotlyRef = useRef<PlotlyModule | null>(null);
  const fullscreenPlotlyRef = useRef<PlotlyModule | null>(null);

  const openFullscreen = useCallback(() => {
    const snap = captureGraphView(embeddedGraphRef.current);
    setFsSnapshot(snap);
    setFsSnapshotNonce((n) => n + 1);
    setIsFullscreen(true);
  }, []);

  const closeFullscreen = useCallback(() => {
    const snap = captureGraphView(fullscreenGraphRef.current);
    setIsFullscreen(false);
    if (snap) {
      setEmbeddedSnapshot(snap);
      setEmbeddedSnapshotNonce((n) => n + 1);
    }
  }, []);

  const openFullscreenRef = useRef(openFullscreen);
  const closeFullscreenRef = useRef(closeFullscreen);
  useEffect(() => {
    openFullscreenRef.current = openFullscreen;
    closeFullscreenRef.current = closeFullscreen;
  });

  // Lock page scroll & bind Escape dismissal while fullscreen overlay is active
  useEffect(() => {
    if (!isFullscreen) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeFullscreenRef.current();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [isFullscreen]);

  // Compute tight initial y-bounds when caller has not specified an explicit fixed y-range
  const initialYBounds = useMemo<[number, number] | null>(() => {
    if (plotType !== '2d' || !autoFitYOnLoad) return null;
    const overY = layout?.yaxis as Record<string, unknown> | undefined;
    if (overY?.range && Array.isArray(overY.range)) return null;
    if (overY?.type === 'category' || overY?.type === 'multicategory' || overY?.type === 'log') {
      return null;
    }
    if (layout?.barmode === 'stack' || layout?.barmode === 'relative') return null;
    return computeInitialYBounds(data);
  }, [plotType, autoFitYOnLoad, layout, data]);

  const buildLayoutForSurface = useCallback(
    (forFullscreen: boolean): Partial<Layout> => {
      const baseOpts: BaseLayoutOptions = {
        plotType,
        rangeSlider,
        hovermode,
        showLegend,
        uirevision: uirevisionKey,
        margin,
        xAxisTitle,
        yAxisTitle,
        yTickPrefix,
        yTickSuffix,
        yTickFormat,
        xTickSuffix,
        xTickFormat,
      };
      const base = buildBasePlotlyLayout(isDark, baseOpts);
      const over = layout ?? {};

      const mergedX = deepMergeAxis(
        base.xaxis as Record<string, unknown> | undefined,
        over.xaxis as Record<string, unknown> | undefined
      );

      // Slightly slimmer range slider in fullscreen so the main canvas claims even more height
      if (forFullscreen && rangeSlider && mergedX && typeof mergedX.rangeslider === 'object') {
        mergedX.rangeslider = {
          ...(mergedX.rangeslider as object),
          thickness: 0.06,
        };
      }

      const mergedY = deepMergeAxis(
        base.yaxis as Record<string, unknown> | undefined,
        over.yaxis as Record<string, unknown> | undefined
      );

      if (mergedY && initialYBounds && !Array.isArray(mergedY.range)) {
        mergedY.range = initialYBounds;
        mergedY.autorange = false;
      }

      const surfaceMargin = { ...(base.margin ?? {}), ...(over.margin ?? {}) };
      if (forFullscreen && plotType !== '3d') {
        surfaceMargin.r = Math.max(surfaceMargin.r ?? 8, 12);
      }

      return {
        ...base,
        ...over,
        margin: surfaceMargin,
        font: { ...(base.font ?? {}), ...(over.font ?? {}) },
        legend: { ...(base.legend ?? {}), ...(over.legend ?? {}) },
        hoverlabel: { ...(base.hoverlabel ?? {}), ...(over.hoverlabel ?? {}) },
        xaxis: mergedX as Layout['xaxis'],
        yaxis: mergedY as Layout['yaxis'],
      };
    },
    [
      isDark,
      plotType,
      rangeSlider,
      hovermode,
      showLegend,
      uirevisionKey,
      margin,
      xAxisTitle,
      yAxisTitle,
      yTickPrefix,
      yTickSuffix,
      yTickFormat,
      xTickSuffix,
      xTickFormat,
      layout,
      initialYBounds,
    ]
  );

  const embeddedLayout = useMemo(() => buildLayoutForSurface(false), [buildLayoutForSurface]);
  const fullscreenLayout = useMemo(() => buildLayoutForSurface(true), [buildLayoutForSurface]);

  const embeddedConfig = useMemo<Partial<Config>>(() => {
    const baseCfg = buildPlotlyConfig(plotType, exportFilename);
    const callerButtons = (config?.modeBarButtonsToAdd ?? []) as ModeBarButtonAny[];
    const fsBtn: ModeBarButtonAny[] = allowFullscreen
      ? [
          {
            name: 'Fullscreen',
            title: 'Expand chart to fullscreen',
            icon: MODEBAR_EXPAND_ICON,
            click: () => openFullscreenRef.current(),
          },
        ]
      : [];
    return {
      ...baseCfg,
      ...(config ?? {}),
      modeBarButtonsToAdd: [...callerButtons, ...fsBtn],
    };
  }, [plotType, exportFilename, config, allowFullscreen]);

  const fullscreenConfig = useMemo<Partial<Config>>(() => {
    const baseCfg = buildPlotlyConfig(plotType, `${exportFilename}-fullscreen`);
    const callerButtons = (config?.modeBarButtonsToAdd ?? []) as ModeBarButtonAny[];
    const exitBtn: ModeBarButtonAny[] = [
      {
        name: 'Exit Fullscreen',
        title: 'Exit fullscreen (Esc)',
        icon: MODEBAR_COMPRESS_ICON,
        click: () => closeFullscreenRef.current(),
      },
    ];
    return {
      ...baseCfg,
      ...(config ?? {}),
      modeBarButtonsToAdd: [...callerButtons, ...exitBtn],
    };
  }, [plotType, exportFilename, config]);

  const resolvedHeight = useMemo(
    () => resolveResponsiveChartHeight(height, plotType, size),
    [height, plotType, size]
  );

  const resolvedTitle = useMemo(
    () => formatChartDisplayTitle(title, exportFilename),
    [title, exportFilename]
  );

  const modalPortal =
    isFullscreen && typeof document !== 'undefined'
      ? createPortal(
          <AnimatePresence>
            <motion.div
              key="qm-chart-fullscreen-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
              onClick={closeFullscreen}
              className="fixed inset-0 z-[90] flex flex-col p-2 sm:p-4 bg-[#F5F4F1]/92 dark:bg-[#08080A]/92 backdrop-blur-md"
            >
              <motion.div
                role="dialog"
                aria-modal="true"
                aria-label={resolvedTitle}
                initial={{ opacity: 0, scale: 0.985 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.985 }}
                transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
                onClick={(e) => e.stopPropagation()}
                className="w-full h-full flex flex-col rounded-2xl border border-[var(--hairline-strong)] bg-[var(--bg-panel)] shadow-2xl overflow-hidden"
              >
                {/* Minimal single-row fullscreen header */}
                <div className="h-11 sm:h-12 px-4 border-b border-[var(--hairline)] flex items-center justify-between gap-4 shrink-0">
                  <div className="flex items-baseline gap-2.5 min-w-0">
                    <h2 className="font-display text-sm sm:text-[15px] font-semibold text-zinc-900 dark:text-zinc-100 truncate">
                      {resolvedTitle}
                    </h2>
                    {subtitle && (
                      <span className="hidden md:inline text-xs font-mono text-zinc-500 dark:text-zinc-400 truncate">
                        · {subtitle}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-4 shrink-0">
                    <div className="hidden lg:flex items-center gap-2 text-[10.5px] font-mono text-zinc-500 dark:text-zinc-400 select-none">
                      {plotType === '3d' ? (
                        <>
                          <span>Left-drag: Orbit</span>
                          <span aria-hidden="true">·</span>
                          <span>Right-drag: Pan</span>
                          <span aria-hidden="true">·</span>
                          <span>Wheel: Zoom</span>
                        </>
                      ) : (
                        <>
                          <span>Wheel: Pan X</span>
                          <span aria-hidden="true">·</span>
                          <span>Ctrl+Wheel: Zoom</span>
                          <span aria-hidden="true">·</span>
                          <span>Drag: Box Zoom</span>
                          <span aria-hidden="true">·</span>
                          <span>Double-click: Reset</span>
                        </>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={closeFullscreen}
                      aria-label="Exit fullscreen"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-[var(--hairline-strong)] bg-[var(--bg-inset)]/70 text-xs font-mono text-zinc-700 dark:text-zinc-200 hover:border-gold-500/50 hover:text-gold-700 dark:hover:text-gold-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gold-500/50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <Minimize2 className="w-3.5 h-3.5 shrink-0" />
                      <span>Exit Fullscreen</span>
                      <span className="hidden sm:inline text-[10px] text-zinc-400 dark:text-zinc-500 ml-0.5">
                        Esc
                      </span>
                    </button>
                  </div>
                </div>

                {/* Fullscreen Plotly canvas (~92% of viewport) */}
                <div className="qm-plotly-host qm-plotly-fullscreen relative flex-1 min-h-0 w-full p-2 sm:p-3">
                  <PlotlyCanvas
                    data={data}
                    layout={fullscreenLayout}
                    config={fullscreenConfig}
                    plotType={plotType}
                    enableWheelPan={enableWheelPan}
                    autoScaleYOnWheel={autoScaleYOnWheel}
                    initialYBounds={initialYBounds}
                    viewSnapshot={fsSnapshot}
                    viewSnapshotNonce={fsSnapshotNonce}
                    isActive={isFullscreen}
                    graphRef={fullscreenGraphRef}
                    plotlyModRef={fullscreenPlotlyRef}
                    onHover={onHover}
                    onUnhover={onUnhover}
                    onClick={onClick}
                    onRelayout={onRelayout}
                  />
                </div>
              </motion.div>
            </motion.div>
          </AnimatePresence>,
          document.body
        )
      : null;

  return (
    <div
      className={`qm-plotly-host ${allowFullscreen ? 'qm-has-fs-btn' : ''} relative w-full ${className}`}
    >
      {allowFullscreen && (
        <button
          type="button"
          onClick={openFullscreen}
          aria-label={`Expand ${resolvedTitle} to fullscreen`}
          title="Expand fullscreen"
          className="absolute top-1.5 right-1.5 z-20 inline-flex items-center justify-center w-7 h-7 rounded-lg border border-[var(--hairline-strong)] bg-[var(--bg-panel)]/85 backdrop-blur-sm text-zinc-500 dark:text-zinc-400 hover:text-gold-600 dark:hover:text-gold-300 hover:border-gold-500/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gold-500/50 transition-colors cursor-pointer"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      )}

      <div className="relative w-full" style={{ height: resolvedHeight }}>
        <PlotlyCanvas
          data={data}
          layout={embeddedLayout}
          config={embeddedConfig}
          plotType={plotType}
          enableWheelPan={enableWheelPan}
          autoScaleYOnWheel={autoScaleYOnWheel}
          initialYBounds={initialYBounds}
          viewSnapshot={embeddedSnapshot}
          viewSnapshotNonce={embeddedSnapshotNonce}
          isActive={!isFullscreen}
          graphRef={embeddedGraphRef}
          plotlyModRef={embeddedPlotlyRef}
          onHover={onHover}
          onUnhover={onUnhover}
          onClick={onClick}
          onRelayout={onRelayout}
        />
      </div>

      {showWheelHint && plotType === '2d' && (
        <div className="mt-1 flex items-center justify-end gap-3 text-[10px] font-mono text-zinc-400 dark:text-zinc-500 select-none">
          <span>Wheel: Pan X</span>
          <span aria-hidden="true">·</span>
          <span>Ctrl+Wheel: Zoom</span>
          <span aria-hidden="true">·</span>
          <span>Drag: Box Zoom</span>
          <span aria-hidden="true">·</span>
          <span>Double-click: Reset</span>
        </div>
      )}

      {modalPortal}
    </div>
  );
}
