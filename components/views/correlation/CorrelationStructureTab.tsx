'use client';

import React, { useState, useMemo } from 'react';
import type { Data, Layout } from 'plotly.js';
import {
  computeMantegnaNetwork,
  computeLeadLagCorrelation,
  computeMatrixStability,
} from '@/lib/correlation-engine';
import { MantegnaDistanceEquation, FrobeniusDivergenceEquation } from '@/components/math/MathEquations';
import { ChartFullscreen } from '@/components/ui/ChartFullscreen';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens, QUANT_COLORS } from '@/lib/plotly-theme';
import {
  GitCommit,
  Network,
  Activity,
  ArrowRightLeft,
  ShieldAlert,
  Info,
  Layers,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Zap,
} from 'lucide-react';

interface CorrelationStructureTabProps {
  tickers: string[];
  matrix: number[][];
  returnsByTicker: { [ticker: string]: number[] };
  dates: string[];
}

export function CorrelationStructureTab({
  tickers,
  matrix,
  returnsByTicker,
  dates,
}: CorrelationStructureTabProps) {
  const n = tickers.length;

  // Selected pair for Lead-Lag analysis
  const [leadLagPairA, setLeadLagPairA] = useState<string>(tickers[0] || 'AAPL');
  const [leadLagPairB, setLeadLagPairB] = useState<string>(
    tickers[1] || (tickers.length > 1 ? tickers[1] : 'SPY')
  );
  const [maxLag, setMaxLag] = useState<number>(5);

  // Selected node in MST for detailed inspection
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showMathReference, setShowMathReference] = useState<boolean>(false);

  // 1. Mantegna Minimum Spanning Tree Computation
  const mstResult = useMemo(() => {
    if (n <= 1 || matrix.length !== n) {
      return { edges: [], centrality: [], hubTicker: tickers[0] || '' };
    }
    return computeMantegnaNetwork(matrix, tickers);
  }, [matrix, tickers, n]);

  const isDark = usePlotlyDarkTheme();
  const tokens = useMemo(() => getPlotlyThemeTokens(isDark), [isDark]);

  // Node coordinates for deterministic circular MST layout
  const nodePositions = useMemo(() => {
    const coords: { [sym: string]: { x: number; y: number; angle: number } } = {};
    const radius = 1;

    tickers.forEach((sym, idx) => {
      const angle = (idx / Math.max(1, n)) * 2 * Math.PI - Math.PI / 2;
      const x = radius * Math.cos(angle);
      const y = -radius * Math.sin(angle);
      coords[sym] = { x, y, angle };
    });

    return { coords, radius };
  }, [tickers, n]);

  const { mstTraces, mstLayout } = useMemo<{
    mstTraces: Data[];
    mstLayout: Partial<Layout>;
  }>(() => {
    const traces: Data[] = [];

    // Subtle reference ring
    const ringX: number[] = [];
    const ringY: number[] = [];
    for (let i = 0; i <= 72; i++) {
      const theta = (i / 72) * 2 * Math.PI;
      ringX.push(Math.cos(theta));
      ringY.push(Math.sin(theta));
    }
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: ringX,
      y: ringY,
      line: {
        color: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(24,24,27,0.1)',
        width: 1,
        dash: 'dot',
      },
      hoverinfo: 'skip',
      showlegend: false,
    });

    // MST edges + midpoint correlation labels
    const midX: number[] = [];
    const midY: number[] = [];
    const midLabels: string[] = [];
    const midHover: string[] = [];

    for (const edge of mstResult.edges) {
      const p1 = nodePositions.coords[edge.source];
      const p2 = nodePositions.coords[edge.target];
      if (!p1 || !p2) continue;
      const isConn =
        selectedNode !== null &&
        (edge.source === selectedNode || edge.target === selectedNode);
      const width = Math.max(1.2, (2 - edge.distance) * 2.2) + (isConn ? 1.4 : 0);
      const opacity = selectedNode ? (isConn ? 0.95 : 0.16) : 0.7;
      const color = isConn
        ? '#10b981'
        : edge.correlation >= 0
          ? QUANT_COLORS.gold
          : '#E0655F';

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [p1.x, p2.x],
        y: [p1.y, p2.y],
        line: { color, width },
        opacity,
        hoverinfo: 'skip',
        showlegend: false,
      });

      if (!selectedNode || isConn) {
        midX.push((p1.x + p2.x) / 2);
        midY.push((p1.y + p2.y) / 2);
        midLabels.push(edge.correlation.toFixed(2));
        midHover.push(
          `${edge.source} ↔ ${edge.target}<br>ρ = ${edge.correlation.toFixed(3)} · d = ${edge.distance.toFixed(3)}`
        );
      }
    }

    if (midX.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'text',
        x: midX,
        y: midY,
        text: midLabels,
        textfont: {
          size: 9.5,
          color: tokens.textMuted,
          family: 'IBM Plex Mono, monospace',
        },
        hovertext: midHover,
        hoverinfo: 'text',
        showlegend: false,
      });
    }

    // Hub outer ring highlight
    const hubPos = nodePositions.coords[mstResult.hubTicker];
    if (hubPos) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: [hubPos.x],
        y: [hubPos.y],
        marker: {
          size: 46,
          color: 'rgba(16,185,129,0.08)',
          line: { color: '#10b981', width: 1.5 },
        },
        hoverinfo: 'skip',
        showlegend: false,
      });
    }

    // MST Nodes
    const nodeX: number[] = [];
    const nodeY: number[] = [];
    const nodeText: string[] = [];
    const nodeHover: string[] = [];
    const nodeCustom: string[] = [];
    const nodeSize: number[] = [];
    const nodeFill: string[] = [];
    const nodeLine: string[] = [];

    for (const sym of tickers) {
      const pos = nodePositions.coords[sym];
      if (!pos) continue;
      const cent = mstResult.centrality.find((c) => c.ticker === sym);
      const degree = cent?.degree ?? 0;
      const isHub = cent?.isHub ?? false;
      const isSel = selectedNode === sym;
      nodeX.push(pos.x);
      nodeY.push(pos.y);
      nodeText.push(sym);
      nodeCustom.push(sym);
      nodeHover.push(
        `<b>${sym}</b>${isHub ? ' (Hub)' : ''}<br>Degree: ${degree}<br>Centrality: ${(cent?.centralityScore ?? 0).toFixed(3)}<br><i>Click to inspect neighborhood</i>`
      );
      nodeSize.push(isHub ? 36 : Math.max(26, 22 + degree * 3.5));
      nodeFill.push(isSel ? '#10b981' : isHub ? '#065f46' : isDark ? '#1e293b' : '#27272a');
      nodeLine.push(isSel ? '#ffffff' : isHub ? '#34d399' : '#64748b');
    }

    traces.push({
      type: 'scatter',
      mode: 'text+markers',
      name: 'Assets',
      x: nodeX,
      y: nodeY,
      text: nodeText,
      customdata: nodeCustom,
      textposition: 'middle center',
      textfont: {
        size: 10,
        color: '#ffffff',
        family: 'IBM Plex Mono, monospace',
      },
      marker: {
        size: nodeSize,
        color: nodeFill,
        line: { color: nodeLine, width: 1.6 },
      },
      hovertext: nodeHover,
      hoverinfo: 'text',
      showlegend: false,
    });

    return {
      mstTraces: traces,
      mstLayout: {
        margin: { l: 16, r: 16, t: 16, b: 16 },
        xaxis: {
          range: [-1.35, 1.35],
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          fixedrange: false,
        },
        yaxis: {
          range: [-1.35, 1.35],
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          scaleanchor: 'x',
          scaleratio: 1,
          fixedrange: false,
        },
      },
    };
  }, [tickers, mstResult, nodePositions, selectedNode, isDark, tokens]);

  // 2. Lead-Lag Cross-Correlation Computation
  const leadLagResult = useMemo(() => {
    const retA = returnsByTicker[leadLagPairA] || [];
    const retB = returnsByTicker[leadLagPairB] || [];
    return computeLeadLagCorrelation(retA, retB, maxLag);
  }, [returnsByTicker, leadLagPairA, leadLagPairB, maxLag]);

  // 3. Matrix Stability & Frobenius Norm Divergence Computation
  const stabilityResult = useMemo(() => {
    return computeMatrixStability(returnsByTicker, dates, 60);
  }, [returnsByTicker, dates]);

  // Selected node details
  const selectedNodeInfo = useMemo(() => {
    if (!selectedNode) return null;
    const cent = mstResult.centrality.find((c) => c.ticker === selectedNode);
    const connectedEdges = mstResult.edges.filter(
      (e) => e.source === selectedNode || e.target === selectedNode
    );
    const neighbors = connectedEdges.map((e) => ({
      neighbor: e.source === selectedNode ? e.target : e.source,
      distance: e.distance,
      correlation: e.correlation,
    }));
    return {
      ticker: selectedNode,
      degree: cent?.degree || 0,
      isHub: cent?.isHub || false,
      centralityScore: cent?.centralityScore || 0,
      neighbors,
    };
  }, [selectedNode, mstResult]);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="panel p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-zinc-100 dark:border-zinc-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="kicker">
                Tier 4 Network &amp; Stability
              </span>
              <span className="text-zinc-300 dark:text-zinc-700" aria-hidden="true">·</span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                Mantegna MST Topology, Lead-Lag &amp; Matrix Drift
              </span>
            </div>
            <h3 className="text-lg font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mt-0.5">
              Correlation Structure, Network Geometry &amp; Dynamic Stability
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1 max-w-2xl">
              Extract the topological backbone of cross-asset co-movement using Mantegna distance, identify central contagion hubs, assess information spillover delays (lead-lag), and track matrix stability over time via Frobenius distance.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-zinc-50 dark:bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-100 dark:border-zinc-800">
            <Network className="w-5 h-5 text-gold-500 shrink-0" />
            <div className="text-xs">
              <div className="text-zinc-500 dark:text-zinc-400">Network Hub Asset:</div>
              <div className="font-mono font-bold text-gold-600 dark:text-gold-400 text-sm">
                {mstResult.hubTicker || 'N/A'} (Degree {mstResult.centrality.find((c) => c.isHub)?.degree || 0})
              </div>
            </div>
          </div>
        </div>

        {/* Quick diagnostic row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] text-zinc-500 dark:text-zinc-400">MST Edges</div>
            <div className="text-lg font-mono font-bold text-zinc-800 dark:text-zinc-200 mt-0.5">
              {mstResult.edges.length} <span className="text-xs text-zinc-400">({n - 1} links)</span>
            </div>
            <div className="text-[10px] text-zinc-400">Spanning Tree without cycles</div>
          </div>

          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] text-zinc-500 dark:text-zinc-400">Frobenius Drift (||ΔΣ||_F)</div>
            <div className="text-lg font-mono font-bold text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300 mt-0.5">
              {stabilityResult.currentFrobeniusDivergence.toFixed(3)}
            </div>
            <div className="text-[10px] text-zinc-400">1-month structural drift</div>
          </div>

          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] text-zinc-500 dark:text-zinc-400">High-Corr Pairs (&gt;0.80)</div>
            <div className="text-lg font-mono font-bold text-gold-600 dark:text-gold-400 mt-0.5">
              {stabilityResult.highCorrPairCount}
            </div>
            <div className="text-[10px] text-zinc-400">Concentrated systemic co-movement</div>
          </div>

          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded-lg border border-zinc-100 dark:border-zinc-800/80">
            <div className="text-[11px] text-zinc-500 dark:text-zinc-400">Lead-Lag Dynamics</div>
            <div className="text-lg font-mono font-bold text-gold-600 dark:text-gold-400 mt-0.5">
              {leadLagResult.peakLag === 0 ? 'Synchronous' : `${Math.abs(leadLagResult.peakLag)}d Lag`}
            </div>
            <div className="text-[10px] text-zinc-400 truncate">{leadLagResult.leadingTicker}</div>
          </div>
        </div>
      </div>

      {/* Row 1: Mantegna MST Diagram & Centrality Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): Interactive SVG Minimum Spanning Tree */}
        <div className="lg:col-span-7 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <Network className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  Mantegna Minimum Spanning Tree (MST)
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-400">
                d_ij = √(2(1 - ρ_ij))
              </span>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
              Kruskal&apos;s algorithm isolates the most statistically significant correlation pathways. Thicker links represent higher correlation (shorter metric distance). Click any node to inspect its local neighborhood.
            </p>

            {/* Interactive Plotly MST Network Canvas */}
            <ChartFullscreen title="Mantegna Minimum Spanning Tree" subtitle="Correlation-metric network — click nodes to inspect neighborhoods">
              <div className="w-full rounded-xl border border-zinc-100 dark:border-zinc-800/80 bg-zinc-50 dark:bg-zinc-900/40 p-1.5">
                <PlotlyChart
                  data={mstTraces}
                  layout={mstLayout}
                  plotType="2d"
                  enableWheelPan={false}
                  hovermode="closest"
                  showLegend={false}
                  height={370}
                  exportFilename="mantegna-mst-network"
                  uirevisionKey={`mst-${tickers.join('-')}`}
                  onClick={(ev) => {
                    const pt = (ev as { points?: Array<{ customdata?: unknown }> })?.points?.[0];
                    if (pt && typeof pt.customdata === 'string') {
                      const clicked = pt.customdata;
                      setSelectedNode((prev) => (prev === clicked ? null : clicked));
                    }
                  }}
                />
              </div>
            </ChartFullscreen>
          </div>

          <div className="mt-3 flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 border-t border-zinc-100 dark:border-zinc-800 pt-3">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-gold-700 border border-gold-400 inline-block" /> Hub Asset (Central)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-zinc-800 border border-zinc-500 inline-block" /> Peripheral Leaf
              </span>
            </div>
            {selectedNode && (
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="text-xs text-gold-600 dark:text-gold-400 hover:underline font-mono"
              >
                Clear Selection
              </button>
            )}
          </div>
        </div>

        {/* Right Column (5 cols): Node Inspection & Degree Centrality Ranking */}
        <div className="lg:col-span-5 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  {selectedNodeInfo ? `Inspector: ${selectedNodeInfo.ticker}` : 'Degree Centrality & Hub Ranking'}
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-400">
                {selectedNodeInfo ? 'Selected Node' : 'All Tickers'}
              </span>
            </div>

            {selectedNodeInfo ? (
              /* Selected Node Detail View */
              <div className="mt-3 space-y-3">
                <div className="p-3 bg-gold-50/60 dark:bg-gold-950/20 border border-gold-200 dark:border-gold-800/40 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold font-mono text-zinc-900 dark:text-zinc-100">
                      {selectedNodeInfo.ticker} {selectedNodeInfo.isHub && '★ (Network Hub)'}
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-gold-100 dark:bg-gold-900/40 text-gold-700 dark:text-gold-300">
                      Degree {selectedNodeInfo.degree}
                    </span>
                  </div>
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                    Centrality Score: {(selectedNodeInfo.centralityScore * 100).toFixed(1)}% of total possible spanning links.
                  </div>
                </div>

                <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mt-2">
                  Connected Spanning Tree Neighbors ({selectedNodeInfo.neighbors.length}):
                </div>

                <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
                  {selectedNodeInfo.neighbors.map((nb) => (
                    <div
                      key={nb.neighbor}
                      className="flex items-center justify-between p-2 rounded bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-100 dark:border-zinc-800/80 text-xs font-mono"
                    >
                      <span className="font-bold text-zinc-800 dark:text-zinc-200">
                        {nb.neighbor}
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-zinc-500">
                          d = {nb.distance.toFixed(3)}
                        </span>
                        <span
                          className={`font-semibold ${
                            nb.correlation >= 0 ? 'text-gold-600 dark:text-gold-300' : 'text-rose-400'
                          }`}
                        >
                          ρ = {nb.correlation.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              /* Default Centrality Ranking List */
              <div className="mt-3">
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-3">
                  The hub asset serves as the primary conduit for systemic correlation shocks across the portfolio.
                </p>

                <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
                  {mstResult.centrality.map((item, rank) => {
                    return (
                      <div
                        key={item.ticker}
                        onClick={() => setSelectedNode(item.ticker)}
                        className={`flex items-center justify-between p-2.5 rounded-lg border cursor-pointer transition-all ${
                          item.isHub
                            ? 'bg-gold-50/70 dark:bg-gold-950/30 border-gold-300 dark:border-gold-800/60 shadow-2xs'
                            : 'bg-zinc-50/80 dark:bg-zinc-900/40 border-zinc-100 dark:border-zinc-800/60 hover:border-zinc-300 dark:hover:border-zinc-700'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="w-5 text-center font-mono text-xs font-bold text-zinc-400">
                            #{rank + 1}
                          </span>
                          <div>
                            <span className="font-mono font-bold text-xs text-zinc-900 dark:text-zinc-100">
                              {item.ticker}
                            </span>
                            {item.isHub && (
                              <span className="ml-1.5 text-[10px] font-mono px-1.5 py-0.2 rounded bg-gold-200 dark:bg-gold-900/60 text-gold-800 dark:text-gold-300 font-semibold">
                                HUB
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center gap-3">
                          <div className="w-20 h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${
                                item.isHub ? 'bg-gold-500' : 'bg-gold-500'
                              }`}
                              style={{ width: `${item.centralityScore * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono font-bold text-zinc-700 dark:text-zinc-300 min-w-[50px] text-right">
                            {item.degree} links
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-400">
            <strong>Topological Insight:</strong> Mantegna showed that equity markets exhibit hierarchical taxonomy: a small number of core nodes (often broad index surrogates) anchor the entire covariance network.
          </div>
        </div>
      </div>

      {/* Row 2: Lead-Lag Cross-Correlation & Matrix Frobenius Drift */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (6 cols): Lead-Lag Cross Correlation */}
        <div className="lg:col-span-6 panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
            <div className="flex items-center gap-2">
              <ArrowRightLeft className="w-4 h-4 text-gold-500" />
              <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                Lead-Lag Cross-Correlation Analysis
              </h4>
            </div>

            <div className="flex items-center gap-2 text-xs font-mono">
              <select
                value={leadLagPairA}
                onChange={(e) => setLeadLagPairA(e.target.value)}
                className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded px-1.5 py-0.5 text-xs text-zinc-800 dark:text-zinc-200 focus:outline-hidden"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <span className="text-zinc-400">vs</span>
              <select
                value={leadLagPairB}
                onChange={(e) => setLeadLagPairB(e.target.value)}
                className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded px-1.5 py-0.5 text-xs text-zinc-800 dark:text-zinc-200 focus:outline-hidden"
              >
                {tickers.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
            Cross-correlation ρ(r_A,t, r_B,t-k) for lag k ∈ [-5, +5] days. Tests whether price discovery or liquidity shocks originate in one asset before spilling into another.
          </p>

          {/* Lead-Lag Bar Chart */}
          <div className="space-y-1.5 my-3">
            {leadLagResult.lags.map((item) => {
              const isPeak = item.lag === leadLagResult.peakLag;
              const isZero = item.lag === 0;
              const barPct = Math.min(100, Math.max(0, Math.abs(item.correlation) * 100));

              return (
                <div
                  key={item.lag}
                  className={`flex items-center gap-3 p-1.5 rounded text-xs font-mono transition-colors ${
                    isPeak
                      ? 'bg-gold-50 dark:bg-gold-950/40 border border-gold-300 dark:border-gold-800/60'
                      : 'hover:bg-zinc-50 dark:hover:bg-zinc-900/40'
                  }`}
                >
                  <span
                    className={`w-14 text-right font-bold ${
                      isZero
                        ? 'text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300'
                        : 'text-zinc-500 dark:text-zinc-400'
                    }`}
                  >
                    {item.lag > 0 ? `+${item.lag}d` : item.lag < 0 ? `${item.lag}d` : '0d (Sync)'}
                  </span>

                  <div className="flex-1 h-3.5 bg-zinc-100 dark:bg-zinc-800 rounded overflow-hidden flex items-center">
                    <div
                      className={`h-full rounded ${
                        isPeak
                          ? 'bg-gold-500'
                          : isZero
                          ? 'bg-gold-500'
                          : 'bg-zinc-400 dark:bg-zinc-600'
                      }`}
                      style={{ width: `${barPct}%` }}
                    />
                  </div>

                  <span
                    className={`w-16 text-right font-bold ${
                      isPeak
                        ? 'text-gold-600 dark:text-gold-400'
                        : 'text-zinc-700 dark:text-zinc-300'
                    }`}
                  >
                    {item.correlation.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="p-3 bg-zinc-50 dark:bg-zinc-900/40 rounded-lg border border-zinc-100 dark:border-zinc-800 text-xs mt-3">
            <div className="font-bold text-zinc-800 dark:text-zinc-200">
              {leadLagResult.leadingTicker}
            </div>
            <div className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-0.5">
              {leadLagResult.lagInterpretation}
            </div>
          </div>
        </div>

        {/* Right Column (6 cols): Matrix Stability & Frobenius Drift */}
        <div className="lg:col-span-6 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <GitCommit className="w-4 h-4 text-gold-500" />
                <h4 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  Matrix Stability &amp; Frobenius Drift (||ΔΣ||_F)
                </h4>
              </div>
              <span className="text-xs font-mono text-zinc-400">Lag: 22 trading days</span>
            </div>

            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-2 mb-3">
              Measures the geometric rate of change in the multi-asset correlation structure: <code className="text-gold-600 dark:text-gold-400">|| C_t - C_&#123;t-22&#125; ||_F</code>. Spikes indicate abrupt market regime breakdowns.
            </p>

            {/* Historical Divergence Mini Bar/Timeline */}
            <div className="space-y-2 my-3 p-3 bg-zinc-50 dark:bg-zinc-900/40 rounded-lg border border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-zinc-500">Latest Divergence Distance:</span>
                <strong className="text-zinc-800 dark:text-zinc-200">
                  {stabilityResult.currentFrobeniusDivergence.toFixed(3)}
                </strong>
              </div>

              {/* Sparkline-like representation */}
              <div className="pt-2">
                <div className="text-[10px] font-mono text-zinc-400 mb-1 flex justify-between">
                  <span>Rolling Frobenius Divergence History</span>
                  <span>T-0</span>
                </div>
                <div className="h-16 flex items-end gap-1.5 w-full bg-white dark:bg-zinc-950 p-1.5 rounded border border-zinc-200 dark:border-zinc-800">
                  {stabilityResult.frobeniusDistance.slice(-20).map((pt, idx) => {
                    const maxDist = 0.5;
                    const heightPct = Math.min(100, Math.max(10, (pt.distance / maxDist) * 100));
                    const isHigh = pt.distance > 0.25;

                    return (
                      <div
                        key={idx}
                        className="flex-1 flex flex-col items-center justify-end h-full group relative"
                      >
                        <div
                          className={`w-full rounded-xs transition-all ${
                            isHigh ? 'bg-gold-500' : 'bg-gold-500'
                          }`}
                          style={{ height: `${heightPct}%` }}
                        />
                        <div className="hidden group-hover:block absolute bottom-full mb-1 z-10 bg-zinc-800 text-white text-[9px] font-mono p-1 rounded whitespace-nowrap">
                          {pt.date}: {pt.distance.toFixed(3)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="p-3 bg-zinc-50 dark:bg-zinc-900/40 rounded-lg border border-zinc-100 dark:border-zinc-800 text-xs">
              <div className="flex items-center justify-between font-mono">
                <span className="text-zinc-500">Systemic Co-Movement Cluster:</span>
                <span className="font-bold text-gold-600 dark:text-gold-400">
                  {stabilityResult.highCorrPairCount} pairs &gt; 0.80
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1">
                When high-correlation count expands, the portfolio loses degrees of freedom, transitioning from idiosyncratic diversification to single-factor market beta.
              </p>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-400">
            <strong>Rebalancing Implication:</strong> High Frobenius divergence warns portfolio managers that static mean-variance weights are experiencing severe alpha-decay and turnover drag.
          </div>
        </div>
      </div>

      {/* Expandable Mathematical Reference Panel */}
      <div className="panel p-5">
        <button
          type="button"
          onClick={() => setShowMathReference((prev) => !prev)}
          className="w-full flex items-center justify-between text-left text-sm font-bold text-zinc-900 dark:text-zinc-100"
        >
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-gold-500" />
            <span>Mathematical Formulations (Mantegna Ultrametric &amp; Frobenius Distance)</span>
          </div>
          {showMathReference ? (
            <ChevronUp className="w-4 h-4 text-zinc-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-zinc-400" />
          )}
        </button>

        {showMathReference && (
          <div className="mt-4 pt-4 border-t border-zinc-100 dark:border-zinc-800 grid-cards">
            <div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mb-2">
                Mantegna Correlation Metric Distance
              </div>
              <MantegnaDistanceEquation display="block" />
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                Converts correlation coefficients r_ij into an exact Euclidean metric space satisfying non-negativity d_ij ≥ 0, symmetry d_ij = d_ji, and the triangle inequality d_ij ≤ d_ik + d_kj.
              </p>
            </div>

            <div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 mb-2">
                Matrix Frobenius Norm Divergence
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-zinc-900/50 rounded border border-zinc-100 dark:border-zinc-800 text-center">
                <FrobeniusDivergenceEquation display="block" />
              </div>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-2">
                The matrix norm quantifies overall correlation drift over a 1-month trading period (22 days), serving as an early-warning signal for macro regime transitions.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
