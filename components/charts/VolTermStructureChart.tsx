'use client';

import React, { useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import type { RealizedVolConeBand } from '@/lib/vol-analytics';
import { EmptyState } from '@/components/ui/empty-state';
import { PlotlyChart } from '@/components/charts/PlotlyChart';
import { QUANT_COLORS } from '@/lib/plotly-theme';

export interface TermStructurePoint {
  dte: number;
  iv: number | null;
}

/**
 * Interactive Plotly IV Term Structure (σ or total variance w = σ²T),
 * with optional nested realized-volatility cone bands or 1Y VIX reference band.
 */
export function VolTermStructureChart({
  points,
  axisMode,
  band,
  iv30Marker,
  cone = null,
  showCone = false,
}: {
  points: TermStructurePoint[];
  axisMode: 'vol' | 'totalVariance';
  band?: { low: number; high: number; label: string } | null;
  iv30Marker?: number | null;
  cone?: RealizedVolConeBand[] | null;
  showCone?: boolean;
}) {
  const valid = useMemo(
    () =>
      points
        .filter((p): p is { dte: number; iv: number } => p.iv !== null && Number.isFinite(p.iv) && p.iv > 0)
        .sort((a, b) => a.dte - b.dte),
    [points]
  );

  const coneSorted = useMemo(
    () =>
      showCone && cone
        ? [...cone].filter((b) => b.n >= 2).sort((a, b) => a.dte - b.dte)
        : [],
    [showCone, cone]
  );

  const coneVisible = coneSorted.length >= 2;
  const activeBand = coneVisible ? null : band;

  const { traces, layout } = useMemo(() => {
    const valOf = (iv: number, dte: number) =>
      axisMode === 'vol' ? iv * 100 : iv * iv * (dte / 365) * 1e4;
    const unit = axisMode === 'vol' ? '%' : ' bp²';

    const list: Data[] = [];

    if (coneVisible) {
      const xs = coneSorted.map((b) => b.dte);
      // Outer Min–Max band
      list.push(
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV Min',
          x: xs,
          y: coneSorted.map((b) => valOf(b.min, b.dte)),
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV Min–Max',
          x: xs,
          y: coneSorted.map((b) => valOf(b.max, b.dte)),
          fill: 'tonexty',
          fillcolor: 'rgba(140, 140, 154, 0.11)',
          line: { width: 0 },
          hovertemplate: `Max RV: %{y:.1f}${unit}<extra></extra>`,
        }
      );

      // P10–P90 band
      list.push(
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV P10',
          x: xs,
          y: coneSorted.map((b) => valOf(b.p10, b.dte)),
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV P10–P90',
          x: xs,
          y: coneSorted.map((b) => valOf(b.p90, b.dte)),
          fill: 'tonexty',
          fillcolor: 'rgba(140, 140, 154, 0.16)',
          line: { width: 0 },
          hovertemplate: `P90 RV: %{y:.1f}${unit}<extra></extra>`,
        }
      );

      // P25–P75 IQR band
      list.push(
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV P25',
          x: xs,
          y: coneSorted.map((b) => valOf(b.p25, b.dte)),
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
        },
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV P25–P75',
          x: xs,
          y: coneSorted.map((b) => valOf(b.p75, b.dte)),
          fill: 'tonexty',
          fillcolor: 'rgba(201, 164, 92, 0.16)',
          line: { width: 0 },
          hovertemplate: `P75 RV: %{y:.1f}${unit}<extra></extra>`,
        }
      );

      // Median & Current RV
      list.push(
        {
          type: 'scatter',
          mode: 'lines',
          name: 'RV Median',
          x: xs,
          y: coneSorted.map((b) => valOf(b.median, b.dte)),
          line: { color: QUANT_COLORS.slate, width: 1.4, dash: 'dash' },
          hovertemplate: `Median RV: %{y:.1f}${unit}<extra></extra>`,
        },
        {
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Current RV',
          x: xs,
          y: coneSorted.map((b) => valOf(b.current, b.dte)),
          line: { color: QUANT_COLORS.positive, width: 1.9 },
          marker: { size: 5, color: QUANT_COLORS.positive },
          hovertemplate: `Current RV: %{y:.1f}${unit}<extra></extra>`,
        }
      );
    }

    if (valid.length >= 2) {
      list.push({
        type: 'scatter',
        mode: 'lines+markers',
        name: axisMode === 'vol' ? 'ATM IV (%)' : 'Total Variance (bp²)',
        x: valid.map((p) => p.dte),
        y: valid.map((p) => valOf(p.iv, p.dte)),
        line: { color: QUANT_COLORS.goldLight, width: 2.4 },
        marker: { size: 6.5, color: QUANT_COLORS.goldLight },
        hovertemplate: `<b>%{x}d</b> · %{y:.2f}${unit}<extra></extra>`,
      });
    }

    if (iv30Marker != null && Number.isFinite(iv30Marker) && iv30Marker > 0) {
      list.push({
        type: 'scatter',
        mode: 'markers',
        name: 'IV30 Constant Maturity',
        x: [30],
        y: [valOf(iv30Marker, 30)],
        marker: {
          size: 9,
          symbol: 'diamond',
          color: QUANT_COLORS.gold,
          line: { color: '#101014', width: 1.5 },
        },
        hovertemplate: `<b>IV30</b> · %{y:.2f}${unit}<extra></extra>`,
      });
    }

    const shapes: NonNullable<Layout['shapes']> = [];
    const annotations: NonNullable<Layout['annotations']> = [];

    if (activeBand && axisMode === 'vol') {
      shapes.push({
        type: 'rect',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: activeBand.low,
        y1: activeBand.high,
        fillcolor: 'rgba(201, 164, 92, 0.08)',
        line: { color: 'rgba(201, 164, 92, 0.25)', width: 1, dash: 'dot' },
        layer: 'below',
      });
      annotations.push({
        x: 0.99,
        y: activeBand.high,
        xref: 'paper',
        yref: 'y',
        text: activeBand.label,
        showarrow: false,
        xanchor: 'right',
        yanchor: 'bottom',
        font: { size: 9.5, color: QUANT_COLORS.goldLight },
      });
    }

    return {
      traces: list,
      layout: { shapes, annotations } as Partial<Layout>,
    };
  }, [valid, coneSorted, coneVisible, activeBand, axisMode, iv30Marker]);

  if (valid.length < 2 && !coneVisible) {
    return (
      <EmptyState
        title="Not enough expiries to draw the curve"
        detail="Need at least two usable ATM IV points"
        className="h-44"
      />
    );
  }

  return (
    <PlotlyChart
      data={traces}
      layout={layout}
      plotType="2d"
      hovermode="x unified"
      showLegend
      xAxisTitle="Days to Expiration (DTE)"
      xTickSuffix="d"
      yAxisTitle={axisMode === 'vol' ? 'Implied Vol (%)' : 'Total Variance (bp²)'}
      yTickSuffix={axisMode === 'vol' ? '%' : ''}
      height={290}
      uirevisionKey={`term-${axisMode}-${showCone}`}
      exportFilename={`vol-term-structure-${axisMode}`}
      showWheelHint
    />
  );
}
