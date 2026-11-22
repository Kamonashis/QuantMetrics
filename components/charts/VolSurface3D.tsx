'use client';

import React, { useMemo, useState } from 'react';
import { AlertTriangle, Eye, Layers } from 'lucide-react';
import type { Data, Layout } from 'plotly.js';
import {
  normCdf,
  sviTotalVariance,
  d1d2,
  fitLocalSmile,
  evalLocalSmileIv,
} from '@/lib/options';
import type { SliceOut } from '@/components/views/volatility/types';
import { PlotlyChart, usePlotlyDarkTheme } from '@/components/charts/PlotlyChart';
import { getPlotlyThemeTokens } from '@/lib/plotly-theme';

export type SmileFitMethod = 'svi' | 'ssvi' | 'spline';

/** A per-slice fit resolved once per (slices, method) render — never per rotation frame. */
export interface ResolvedSliceFit {
  slice: SliceOut;
  ivAt: (k: number) => number;
  source: 'spline' | 'ssvi' | 'svi';
  nUsed: number;
}

/** Resolve every fitted slice for a given method, with SVI as per-slice fallback. */
export function resolveSliceFits(
  slices: SliceOut[],
  method: SmileFitMethod
): { resolved: ResolvedSliceFit[]; fellBack: number } {
  const usable = slices
    .filter(
      (s) =>
        Number.isFinite(s.dte) &&
        s.dte > 0 &&
        s.t > 0 &&
        Number.isFinite(s.forward) &&
        s.forward > 0
    )
    .slice()
    .sort((a, b) => a.dte - b.dte);

  let fellBack = 0;
  const resolved: ResolvedSliceFit[] = [];
  for (const s of usable) {
    if (method === 'svi') {
      if (!s.svi) continue;
      const svi = s.svi;
      resolved.push({
        slice: s,
        ivAt: (k) =>
          Math.sqrt(
            Math.max(
              1e-12,
              sviTotalVariance(
                svi as { a: number; b: number; rho: number; m: number; sigma: number },
                k
              ) / s.t
            )
          ),
        source: 'svi',
        nUsed: svi.nUsed,
      });
      continue;
    }
    const local = fitLocalSmile(
      s.points.map((p) => ({ k: p.k, iv: p.iv, weight: p.weight })),
      s.t,
      method
    );
    if (local) {
      resolved.push({
        slice: s,
        ivAt: (k) => evalLocalSmileIv(local, k, s.t),
        source: local.model,
        nUsed: local.nUsed,
      });
      continue;
    }
    if (s.svi) {
      fellBack += 1;
      const svi = s.svi;
      resolved.push({
        slice: s,
        ivAt: (k) =>
          Math.sqrt(
            Math.max(
              1e-12,
              sviTotalVariance(
                svi as { a: number; b: number; rho: number; m: number; sigma: number },
                k
              ) / s.t
            )
          ),
        source: 'svi',
        nUsed: svi.nUsed,
      });
    }
  }
  return { resolved, fellBack };
}

const K_MIN = -0.45;
const K_MAX = 0.45;
const K_STEPS = 29;

type CameraPresetKey = 'iso' | 'smile' | 'term' | 'top';
type ColorScaleKey = 'gold' | 'viridis' | 'cividis';

const CAMERA_PRESETS: Record<
  CameraPresetKey,
  { label: string; eye: { x: number; y: number; z: number } }
> = {
  iso: { label: 'Isometric', eye: { x: 1.55, y: -1.65, z: 1.05 } },
  smile: { label: 'Smile', eye: { x: 0.02, y: -2.35, z: 0.28 } },
  term: { label: 'Term', eye: { x: 2.35, y: 0.02, z: 0.28 } },
  top: { label: 'Top', eye: { x: 0.01, y: -0.01, z: 2.45 } },
};

const GOLD_OBSIDIAN_SCALE: [number, string][] = [
  [0.0, '#182436'],
  [0.22, '#244A54'],
  [0.45, '#3FB27F'],
  [0.68, '#C9A45C'],
  [0.86, '#DEC084'],
  [1.0, '#E0655F'],
];

interface VolSurface3DProps {
  slices: SliceOut[];
  /** spot price, for K and Δ readouts */
  spot: number | null;
  rate?: number | null;
  dividendYield?: number | null;
  /** smile fitting method for the mesh — 'svi' = server fit, others refit client-side */
  fitMethod?: SmileFitMethod;
  className?: string;
}

function clampIv(iv: number): number {
  if (!Number.isFinite(iv)) return 0.2;
  return Math.max(0.01, Math.min(3.5, iv));
}

export function VolSurface3D({
  slices,
  spot,
  rate,
  dividendYield,
  className,
  fitMethod = 'svi',
}: VolSurface3DProps) {
  const isDark = usePlotlyDarkTheme();
  const [cameraPreset, setCameraPreset] = useState<CameraPresetKey>('iso');
  const [cameraBump, setCameraBump] = useState(0);
  const [showQuotes, setShowQuotes] = useState(true);
  const [showContours, setShowContours] = useState(true);
  const [colorScale, setColorScale] = useState<ColorScaleKey>('gold');

  const r = rate ?? 0;
  const q = dividendYield ?? 0;
  const spot0 = spot ?? 0;

  const { resolved, fellBack } = useMemo(
    () => resolveSliceFits(slices, fitMethod),
    [slices, fitMethod]
  );

  const unfittedWithPoints = useMemo(
    () => slices.filter((s) => s.svi === null && s.points.length > 0).length,
    [slices]
  );

  const surfaceData = useMemo(() => {
    if (resolved.length === 0) return null;

    const kGrid = Array.from(
      { length: K_STEPS },
      (_, c) => K_MIN + (c / (K_STEPS - 1)) * (K_MAX - K_MIN)
    );
    const dteGrid = resolved.map((f) => f.slice.dte);

    // Z matrix [row=dte][col=k] in IV %
    const zGrid: number[][] = [];
    const customGrid: string[][] = [];
    let minIv = Infinity;
    let maxIv = -Infinity;

    for (let row = 0; row < resolved.length; row++) {
      const fit = resolved[row];
      const s = fit.slice;
      const zRow: number[] = [];
      const customRow: string[] = [];
      for (let c = 0; c < K_STEPS; c++) {
        const k = kGrid[c];
        const iv = clampIv(fit.ivAt(k));
        const ivPct = iv * 100;
        if (iv < minIv) minIv = iv;
        if (iv > maxIv) maxIv = iv;
        zRow.push(ivPct);

        const strike = s.forward * Math.exp(k);
        let deltaStr = '—';
        if (spot0 > 0 && strike > 0 && s.t > 0) {
          const { d1 } = d1d2(spot0, strike, s.t, r, q, iv);
          const cDelta = Math.exp(-q * s.t) * normCdf(d1);
          deltaStr = `${cDelta >= 0 ? '+' : ''}${cDelta.toFixed(2)}`;
        }
        const arbNote =
          s.svi?.arbitrageFlags && s.svi.arbitrageFlags.length > 0
            ? `<br><span style="color:#E8A849">⚠ SVI arb: ${s.svi.arbitrageFlags.join(', ')}</span>`
            : '';
        customRow.push(
          `K ≈ $${strike.toFixed(2)} · Δ ${deltaStr}<br>Expiry ${s.expirationDate} · ${fit.nUsed} quotes${arbNote}`
        );
      }
      zGrid.push(zRow);
      customGrid.push(customRow);
    }

    // If only 1 fitted slice exists, duplicate slightly in DTE so Plotly WebGL renders a crisp 3D ribbon
    const renderDte =
      dteGrid.length === 1 ? [Math.max(1, dteGrid[0] - 1), dteGrid[0] + 1] : dteGrid;
    const renderZ = zGrid.length === 1 ? [zGrid[0], zGrid[0]] : zGrid;
    const renderCustom = customGrid.length === 1 ? [customGrid[0], customGrid[0]] : customGrid;

    // Observed raw quotes across all usable slices
    const quoteK: number[] = [];
    const quoteDte: number[] = [];
    const quoteIvPct: number[] = [];
    const quoteText: string[] = [];

    for (const s of slices) {
      if (!Number.isFinite(s.dte) || s.dte <= 0 || !Number.isFinite(s.forward) || s.forward <= 0) {
        continue;
      }
      for (const p of s.points) {
        if (p.k < K_MIN || p.k > K_MAX) continue;
        const iv = clampIv(p.iv);
        quoteK.push(p.k);
        quoteDte.push(s.dte);
        quoteIvPct.push(iv * 100);
        const strike = s.forward * Math.exp(p.k);
        quoteText.push(
          `Observed ${p.type.toUpperCase()} Quote<br>K $${strike.toFixed(2)} (k ${p.k >= 0 ? '+' : ''}${p.k.toFixed(2)})<br>${s.dte}d (${s.expirationDate}) · IV ${(iv * 100).toFixed(1)}%`
        );
      }
    }

    // ATM ridge (k = 0)
    const atmDte = resolved.map((f) => f.slice.dte);
    const atmK = resolved.map(() => 0);
    const atmIvPct = resolved.map((f) => clampIv(f.ivAt(0)) * 100 + 0.15);

    // Arbitrage-flagged slices
    const arbK: number[] = [];
    const arbDte: number[] = [];
    const arbIvPct: number[] = [];
    const arbText: string[] = [];
    for (const fit of resolved) {
      const flags = fit.slice.svi?.arbitrageFlags ?? [];
      if (flags.length > 0) {
        arbK.push(K_MIN);
        arbDte.push(fit.slice.dte);
        arbIvPct.push(clampIv(fit.ivAt(K_MIN)) * 100 + 0.4);
        arbText.push(`SVI Arbitrage (${fit.slice.dte}d): ${flags.join(', ')}`);
      }
    }

    return {
      kGrid,
      renderDte,
      renderZ,
      renderCustom,
      quoteK,
      quoteDte,
      quoteIvPct,
      quoteText,
      atmDte,
      atmK,
      atmIvPct,
      arbK,
      arbDte,
      arbIvPct,
      arbText,
      minIv,
      maxIv,
      fittedCount: resolved.length,
    };
  }, [resolved, slices, spot0, r, q]);

  const traces = useMemo<Data[]>(() => {
    if (!surfaceData) return [];

    const resolvedScale =
      colorScale === 'gold'
        ? GOLD_OBSIDIAN_SCALE
        : colorScale === 'viridis'
          ? 'Viridis'
          : 'Cividis';

    const list: Data[] = [
      {
        type: 'surface',
        name: 'Fitted IV Surface',
        x: surfaceData.kGrid,
        y: surfaceData.renderDte,
        z: surfaceData.renderZ,
        customdata: surfaceData.renderCustom,
        colorscale: resolvedScale,
        opacity: 0.94,
        showscale: true,
        colorbar: {
          title: { text: 'IV %', font: { size: 10 } },
          thickness: 12,
          len: 0.72,
          x: 1.0,
          ticksuffix: '%',
          tickfont: { size: 10 },
          outlinewidth: 0,
        },
        contours: {
          z: {
            show: showContours,
            usecolormap: true,
            highlightcolor: '#DEC084',
            project: { z: showContours },
          },
          x: {
            show: true,
            color: isDark ? 'rgba(255,255,255,0.12)' : 'rgba(24,24,27,0.12)',
            width: 1,
          },
          y: {
            show: true,
            color: isDark ? 'rgba(255,255,255,0.14)' : 'rgba(24,24,27,0.14)',
            width: 1,
          },
        },
        lighting: {
          ambient: 0.65,
          diffuse: 0.75,
          specular: 0.28,
          roughness: 0.6,
        },
        hovertemplate:
          '<b>k = %{x:+.2f} · %{y:.0f}d · IV %{z:.1f}%</b><br>%{customdata}<extra></extra>',
      } as Data,
    ];

    // ATM ridge along k = 0
    if (surfaceData.atmDte.length >= 2) {
      list.push({
        type: 'scatter3d',
        mode: 'lines+markers',
        name: 'ATM Ridge (k=0)',
        x: surfaceData.atmK,
        y: surfaceData.atmDte,
        z: surfaceData.atmIvPct,
        line: { color: '#DEC084', width: 5 },
        marker: { size: 3.5, color: '#DEC084' },
        hovertemplate: '<b>ATM Ridge (k=0)</b><br>%{y:.0f}d · IV %{z:.1f}%<extra></extra>',
      } as Data);
    }

    // Raw observed market quotes
    if (showQuotes && surfaceData.quoteK.length > 0) {
      list.push({
        type: 'scatter3d',
        mode: 'markers',
        name: 'Observed Quotes',
        x: surfaceData.quoteK,
        y: surfaceData.quoteDte,
        z: surfaceData.quoteIvPct,
        text: surfaceData.quoteText,
        marker: {
          size: 3,
          color: isDark ? 'rgba(236,236,241,0.85)' : 'rgba(24,24,27,0.72)',
          line: {
            color: isDark ? '#08080A' : '#FFFFFF',
            width: 0.6,
          },
        },
        hovertemplate: '%{text}<extra></extra>',
      } as Data);
    }

    // Arbitrage warning markers
    if (surfaceData.arbK.length > 0) {
      list.push({
        type: 'scatter3d',
        mode: 'markers',
        name: 'SVI Arbitrage Flag',
        x: surfaceData.arbK,
        y: surfaceData.arbDte,
        z: surfaceData.arbIvPct,
        text: surfaceData.arbText,
        marker: {
          size: 6,
          symbol: 'diamond',
          color: '#E8A849',
          line: { color: '#101014', width: 1 },
        },
        hovertemplate: '%{text}<extra></extra>',
      } as Data);
    }

    return list;
  }, [surfaceData, colorScale, showContours, showQuotes, isDark]);

  const layout = useMemo<Partial<Layout>>(() => {
    const t = getPlotlyThemeTokens(isDark);
    const axisCommon = {
      gridcolor: t.grid,
      zerolinecolor: t.zeroLine,
      backgroundcolor: t.sceneWall,
      showbackground: true,
      showspikes: true,
      spikesides: true,
      spikecolor: '#DEC084',
      spikethickness: 1.5,
      tickfont: { size: 10, color: t.textMuted },
      titlefont: { size: 11, color: t.text },
    };

    return {
      margin: { l: 0, r: 0, t: 8, b: 0 },
      scene: {
        bgcolor: t.sceneBg,
        dragmode: 'turntable',
        aspectmode: 'manual',
        aspectratio: { x: 1.25, y: 1.15, z: 0.72 },
        camera: {
          eye: CAMERA_PRESETS[cameraPreset].eye,
          center: { x: 0, y: 0, z: -0.06 },
          up: { x: 0, y: 0, z: 1 },
        },
        xaxis: {
          ...axisCommon,
          title: { text: 'Log-Moneyness k = ln(K/F)' },
          tickformat: '+.2f',
        },
        yaxis: {
          ...axisCommon,
          title: { text: 'Expiry (DTE)' },
          ticksuffix: 'd',
        },
        zaxis: {
          ...axisCommon,
          title: { text: 'Implied Vol (%)' },
          ticksuffix: '%',
        },
      },
    };
  }, [isDark, cameraPreset]);

  const fitLabel = fitMethod === 'svi' ? 'SVI' : fitMethod === 'ssvi' ? 'SSVI' : 'Spline';

  if (!surfaceData) {
    return (
      <div
        className={`panel p-6 text-sm text-zinc-500 dark:text-zinc-400 ${className ?? ''}`}
      >
        3D surface needs at least one slice with a fit (≥ 5 quotes for SVI/SSVI, ≥ 3 for Spline).
        Available slices: {slices.length}, unfitted slices: {unfittedWithPoints}. Option chain too
        thin today — check back when OI accrues.
      </div>
    );
  }

  return (
    <div className={`panel overflow-hidden ${className ?? ''}`}>
      {/* Interactive 3D Controls Header */}
      <div className="px-4 py-2.5 border-b border-[var(--hairline)] flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Camera Presets */}
          <div
            role="group"
            aria-label="3D Camera View"
            className="inline-flex items-center rounded-md border border-[var(--hairline)] bg-[var(--bg-inset)] p-0.5"
          >
            {(Object.keys(CAMERA_PRESETS) as CameraPresetKey[]).map((key) => {
              const active = cameraPreset === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => {
                    setCameraPreset(key);
                    setCameraBump((b) => b + 1);
                  }}
                  aria-pressed={active}
                  className={`px-2 py-0.5 text-[10.5px] font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-white dark:bg-[#1E1E24] text-zinc-900 dark:text-gold-300 shadow-xs'
                      : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  {CAMERA_PRESETS[key].label}
                </button>
              );
            })}
          </div>

          {/* Quotes Toggle */}
          <button
            type="button"
            onClick={() => setShowQuotes((v) => !v)}
            aria-pressed={showQuotes}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10.5px] font-mono border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showQuotes
                ? 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300'
                : 'border-[var(--hairline)] text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
            }`}
          >
            <Eye className="w-3 h-3 shrink-0" />
            <span>Quotes ({surfaceData.quoteK.length})</span>
          </button>

          {/* Contours Toggle */}
          <button
            type="button"
            onClick={() => setShowContours((v) => !v)}
            aria-pressed={showContours}
            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10.5px] font-mono border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              showContours
                ? 'border-gold-500/40 bg-gold-500/10 text-gold-700 dark:text-gold-300'
                : 'border-[var(--hairline)] text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
            }`}
          >
            <Layers className="w-3 h-3 shrink-0" />
            <span>Contours</span>
          </button>
        </div>

        {/* Colorscale Switcher */}
        <div className="flex items-center gap-2">
          <div
            role="group"
            aria-label="Colorscale"
            className="inline-flex items-center rounded-md border border-[var(--hairline)] bg-[var(--bg-inset)] p-0.5"
          >
            {(
              [
                { id: 'gold', label: 'Gold' },
                { id: 'viridis', label: 'Viridis' },
                { id: 'cividis', label: 'Cividis' },
              ] as const
            ).map((sc) => (
              <button
                key={sc.id}
                type="button"
                onClick={() => setColorScale(sc.id)}
                aria-pressed={colorScale === sc.id}
                className={`px-2 py-0.5 text-[10.5px] font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  colorScale === sc.id
                    ? 'bg-white dark:bg-[#1E1E24] text-zinc-900 dark:text-gold-300 shadow-xs'
                    : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                }`}
              >
                {sc.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* True WebGL 3D Plotly Viewport */}
      <div className="p-2">
        <PlotlyChart
          data={traces}
          layout={layout}
          plotType="3d"
          height={460}
          uirevisionKey={`vol-surface-3d-${cameraPreset}-${cameraBump}`}
          exportFilename={`volatility-surface-3d-${fitMethod}`}
        />
      </div>

      {/* Honesty Footer & 3D Interaction Guide */}
      <div className="px-4 py-2 border-t border-[var(--hairline)] flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">
        <div className="flex flex-wrap items-center gap-2">
          <span>Left-drag: Orbit</span>
          <span>·</span>
          <span>Right-drag: Pan</span>
          <span>·</span>
          <span>Wheel: Zoom</span>
          {surfaceData.arbK.length > 0 && (
            <>
              <span>·</span>
              <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400">
                <AlertTriangle className="w-3 h-3 shrink-0" />
                {surfaceData.arbK.length} SVI arb flag{surfaceData.arbK.length > 1 ? 's' : ''}
              </span>
            </>
          )}
        </div>
        <div>
          {surfaceData.fittedCount} fitted ({fitLabel}
          {fellBack > 0 ? ` +${fellBack} SVI fallback` : ''}) · IV{' '}
          {(surfaceData.minIv * 100).toFixed(0)}%–{(surfaceData.maxIv * 100).toFixed(0)}% ·
          European BSM · ~15-min delayed
        </div>
      </div>
    </div>
  );
}
