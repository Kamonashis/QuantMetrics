'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { Config, Data, Layout, PlotlyHTMLElement } from 'plotly.js';
import {
  buildBasePlotlyLayout,
  buildPlotlyConfig,
  computeVisibleYBounds,
  computeWheelXAxisRange,
  extractTracesXDomain,
  formatPlotlyXCoord,
  parsePlotlyXCoord,
  type BaseLayoutOptions,
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
      _offset?: number;
      _length?: number;
    };
  };
  removeListener?: (event: string, listener: (...args: unknown[]) => void) => void;
}

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
  height?: number | string;
  className?: string;
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
  height = 320,
  className = '',
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
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const plotlyModRef = useRef<PlotlyModule | null>(null);
  const [ready, setReady] = useState(false);

  // Latest props in refs for event handlers without re-binding DOM listeners
  const dataRef = useRef(data);
  const onHoverRef = useRef(onHover);
  const onUnhoverRef = useRef(onUnhover);
  const onClickRef = useRef(onClick);
  const onRelayoutRef = useRef(onRelayout);
  const autoScaleYRef = useRef(autoScaleYOnWheel);

  useEffect(() => {
    dataRef.current = data;
    onHoverRef.current = onHover;
    onUnhoverRef.current = onUnhover;
    onClickRef.current = onClick;
    onRelayoutRef.current = onRelayout;
    autoScaleYRef.current = autoScaleYOnWheel;
  });

  const mergedLayout = useMemo<Partial<Layout>>(() => {
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
    return {
      ...base,
      ...over,
      margin: { ...(base.margin ?? {}), ...(over.margin ?? {}) },
      font: { ...(base.font ?? {}), ...(over.font ?? {}) },
      legend: { ...(base.legend ?? {}), ...(over.legend ?? {}) },
      hoverlabel: { ...(base.hoverlabel ?? {}), ...(over.hoverlabel ?? {}) },
      xaxis: deepMergeAxis(
        base.xaxis as Record<string, unknown> | undefined,
        over.xaxis as Record<string, unknown> | undefined
      ) as Layout['xaxis'],
      yaxis: deepMergeAxis(
        base.yaxis as Record<string, unknown> | undefined,
        over.yaxis as Record<string, unknown> | undefined
      ) as Layout['yaxis'],
    };
  }, [
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
  ]);

  const mergedConfig = useMemo<Partial<Config>>(() => {
    return {
      ...buildPlotlyConfig(plotType, exportFilename),
      ...(config ?? {}),
    };
  }, [plotType, exportFilename, config]);

  // Mount / reactively update Plotly figure
  useEffect(() => {
    let cancelled = false;
    const el = plotRef.current;
    if (!el) return;

    void loadPlotly().then((Plotly) => {
      if (cancelled || !plotRef.current) return;
      plotlyModRef.current = Plotly;
      const gd = plotRef.current as unknown as InternalPlotlyGraphDiv;
      void Plotly.react(gd, data, mergedLayout, mergedConfig).then(() => {
        if (!cancelled) setReady(true);
      });
    });

    return () => {
      cancelled = true;
    };
  }, [data, mergedLayout, mergedConfig]);

  // Bind Plotly events & clean up WebGL/DOM resources on unmount
  useEffect(() => {
    const el = plotRef.current as unknown as InternalPlotlyGraphDiv | null;
    if (!el || !ready) return;

    const handleHover = (ev: unknown) => onHoverRef.current?.(ev);
    const handleUnhover = () => onUnhoverRef.current?.();
    const handleClick = (ev: unknown) => onClickRef.current?.(ev);
    const handleRelayout = (ev: Record<string, unknown>) => onRelayoutRef.current?.(ev);

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
  }, [ready]);

  useEffect(() => {
    const el = plotRef.current;
    return () => {
      if (el && plotlyModRef.current) {
        try {
          plotlyModRef.current.purge(el);
        } catch {
          // Ignore cleanup errors if element already detached
        }
      }
    };
  }, []);

  // ResizeObserver for smooth responsive + fullscreen modal resizing
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let timer: ReturnType<typeof setTimeout> | null = null;
    const ro = new ResizeObserver(() => {
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
      }, 60);
    });

    ro.observe(host);
    return () => {
      ro.disconnect();
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Non-passive wheel handler for 2D charts:
  // - Plain mouse wheel -> horizontal pan along x-axis
  // - Ctrl/Cmd + Wheel -> cursor-anchored x-axis zoom + tight y auto-fit
  useEffect(() => {
    if (plotType !== '2d' || !enableWheelPan) return;
    const host = hostRef.current;
    if (!host) return;

    const onWheel = (e: WheelEvent) => {
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

      // Prevent default page scroll while actively panning or zooming inside the plot
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

      void Plotly.relayout(el, update as Partial<Layout>);
    };

    host.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      host.removeEventListener('wheel', onWheel);
    };
  }, [plotType, enableWheelPan]);

  const heightStyle = typeof height === 'number' ? `${height}px` : height;

  return (
    <div className={`qm-plotly-host relative w-full ${className}`}>
      <div
        ref={hostRef}
        className="relative w-full"
        style={{ height: heightStyle }}
      >
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center rounded-lg border border-dashed border-[var(--hairline)] bg-[var(--bg-inset)]/30">
            <span className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
              Initializing interactive chart…
            </span>
          </div>
        )}
        <div ref={plotRef} className="w-full h-full" />
      </div>
      {showWheelHint && plotType === '2d' && (
        <div className="mt-1 flex items-center justify-end gap-3 text-[10px] font-mono text-zinc-400 dark:text-zinc-500 select-none">
          <span>Wheel: Pan X</span>
          <span>·</span>
          <span>Ctrl+Wheel: Zoom</span>
          <span>·</span>
          <span>Drag: Box Zoom</span>
          <span>·</span>
          <span>Double-click: Reset</span>
        </div>
      )}
    </div>
  );
}
