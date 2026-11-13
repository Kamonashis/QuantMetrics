'use client';

import { useMemo, useRef, useState } from 'react';
import { RotateCw, AlertTriangle } from 'lucide-react';
import { normCdf, sviTotalVariance, d1d2, fitLocalSmile, evalLocalSmileIv, type LocalSmileFit, type SsviSmileFit, type NaturalSplineSmileFit } from '@/lib/options';
import type { SliceOut } from '@/components/views/volatility/types';

export type SmileFitMethod = 'svi' | 'ssvi' | 'spline';

/** A per-slice fit resolved once per (slices, method) render — never per rotation frame. */
export interface ResolvedSliceFit {
  slice: SliceOut;
  ivAt: (k: number) => number;
  source: 'spline' | 'ssvi' | 'svi';
  nUsed: number;
}

/** Resolve every fitted slice for a given method, with SVI as per-slice fallback. */
export function resolveSliceFits(slices: SliceOut[], method: SmileFitMethod): { resolved: ResolvedSliceFit[]; fellBack: number } {
  const usable = slices
    .filter((s) => Number.isFinite(s.dte) && s.dte > 0 && s.t > 0 && Number.isFinite(s.forward) && s.forward > 0)
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
        ivAt: (k) => Math.sqrt(Math.max(1e-12, sviTotalVariance(svi as { a: number; b: number; rho: number; m: number; sigma: number }, k) / s.t)),
        source: 'svi',
        nUsed: svi.nUsed,
      });
      continue;
    }
    const local = fitLocalSmile(s.points.map((p) => ({ k: p.k, iv: p.iv, weight: p.weight })), s.t, method);
    if (local) {
      resolved.push({ slice: s, ivAt: (k) => evalLocalSmileIv(local, k, s.t), source: local.model, nUsed: local.nUsed });
      continue;
    }
    if (s.svi) {
      fellBack += 1;
      const svi = s.svi;
      resolved.push({
        slice: s,
        ivAt: (k) => Math.sqrt(Math.max(1e-12, sviTotalVariance(svi as { a: number; b: number; rho: number; m: number; sigma: number }, k) / s.t)),
        source: 'svi',
        nUsed: svi.nUsed,
      });
    }
  }
  return { resolved, fellBack };
}

/* ------------------------------------------------------------------ */
/*  Pure projection helpers — module scope                              */
/* ------------------------------------------------------------------ */

const K_MIN = -0.45;
const K_MAX = 0.45;
const K_STEPS = 19; // 0.05 grid



interface FloorCorner {
  key: string;
  sx: number;
  sy: number;
}

interface MeshVertex {
  k: number;
  sx: number;
  sy: number;
  depth: number;
  iv: number;
  row: number;
}

interface ProjectedQuad {
  key: string;
  pts: string;
  depth: number;
  ivAvg: number;
}

interface SurfaceModel {
  vertices: MeshVertex[][];
  quads: ProjectedQuad[];
  dots: { sx: number; sy: number; depth: number; iv: number; k: number; dte: number; strike: number }[];
  atmRidge: string;
  minIv: number;
  maxIv: number;
  dteLabels: { dte: number; sx: number; sy: number }[];
  flags: { row: number; dte: number; flags: string[] }[];
  /** floor plane at fixed depth, for the base parallelogram */
  floor: FloorCorner[];
  /** per-slice screen-space floor edge — curtain bottoms */
  floorVerts: { sx: number; sy: number }[][];
}

function buildSurface(
  resolvedIn: ResolvedSliceFit[],
  allSlices: SliceOut[],
  yawDeg: number,
  pitchDeg: number
): SurfaceModel | null {
  // dteMin/Max from ALL usable slices (honest depth axis); fits come pre-resolved (never per frame)
  const usable = allSlices
    .filter((s) => Number.isFinite(s.dte) && s.dte > 0 && s.t > 0 && Number.isFinite(s.forward) && s.forward > 0)
    .slice()
    .sort((a, b) => a.dte - b.dte);
  if (usable.length === 0) return null;
  const dteMin = usable[0].dte;
  const dteMax = usable[usable.length - 1].dte;
  const dteSpan = Math.max(1, dteMax - dteMin);

  const fitted = resolvedIn;
  if (fitted.length === 0) return null;

  // World spans
  const xSpan = K_MAX - K_MIN;
  const depthSpan = 1.55;
  const zScale = 1.15;

  const Y = (dte: number) => ((dte - dteMin) / dteSpan - 0.5) * depthSpan;
  const X = (k: number) => ((k - (K_MIN + K_MAX) / 2) / xSpan) * xSpan;
  // Pass-through z given the IV normalization band determined below
  const clampIv = (iv: number) => Math.min(2.5, Math.max(0.03, iv));

  // First pass: IV grid of fitted slices → z normalization band
  const grid: number[][] = fitted.map((r) =>
    Array.from({ length: K_STEPS }, (_, c) => {
      const k = K_MIN + (c / (K_STEPS - 1)) * (K_MAX - K_MIN);
      const iv = r.ivAt(k);
      return clampIv(iv);
    })
  );
  let minIv = Infinity;
  let maxIv = -Infinity;
  for (const row of grid) {
    for (const iv of row) {
      if (iv < minIv) minIv = iv;
      if (iv > maxIv) maxIv = iv;
    }
  }
  // Keep the band a bit wider than observed so the surface doesn't hug the floor
  const ivSpan = Math.max(0.02, maxIv - minIv);
  const Z = (iv: number) => ((clampIv(iv) - minIv) / ivSpan) * zScale;

  const cosY = Math.cos((yawDeg * Math.PI) / 180);
  const sinY = Math.sin((yawDeg * Math.PI) / 180);
  const cosP = Math.cos((pitchDeg * Math.PI) / 180);
  const sinP = Math.sin((pitchDeg * Math.PI) / 180);

  const project = (x: number, y: number, z: number) => {
    const x1 = x * cosY - y * sinY;
    const y1 = x * sinY + y * cosY;
    const y2 = y1 * cosP - z * sinP;
    const z2 = y1 * sinP + z * cosP;
    return { x: x1, y: -z2, depth: y2 };
  };

  // Project everything first, then fit the point cloud into the viewBox with padding.
  const projected: { x: number; y: number; depth: number }[] = [];
  const worldVert = (r: number, c: number) => ({ x: X(K_MIN + (c / (K_STEPS - 1)) * (K_MAX - K_MIN)), y: Y(fitted[r].slice.dte), z: Z(grid[r][c]) });
  for (let r = 0; r < fitted.length; r++) {
    for (let c = 0; c < K_STEPS; c++) projected.push(project(worldVert(r, c).x, worldVert(r, c).y, worldVert(r, c).z));
  }
  // Floor lives just under the surface minimum
  const floorZ = -0.09;
  const floorWorld = [
    { x: -xSpan / 2, y: -depthSpan / 2 },
    { x: xSpan / 2, y: -depthSpan / 2 },
    { x: xSpan / 2, y: depthSpan / 2 },
    { x: -xSpan / 2, y: depthSpan / 2 },
  ];
  for (const f of floorWorld) projected.push(project(f.x, f.y, floorZ));

  let minPX = Infinity, maxPX = -Infinity, minPY = Infinity, maxPY = -Infinity;
  for (const p of projected) {
    if (p.x < minPX) minPX = p.x;
    if (p.x > maxPX) maxPX = p.x;
    if (p.y < minPY) minPY = p.y;
    if (p.y > maxPY) maxPY = p.y;
  }
  const spanPX = Math.max(1e-6, maxPX - minPX);
  const spanPY = Math.max(1e-6, maxPY - minPY);
  const VB_W = 900;
  const VB_H = 430;
  const PAD_X = 56;
  const PAD_TOP = 8;
  const PAD_BOT = 30;
  const scale = Math.min((VB_W - PAD_X * 2) / spanPX, (VB_H - PAD_TOP - PAD_BOT) / spanPY);
  const offsetX = PAD_X + ((VB_W - PAD_X * 2) - spanPX * scale) / 2 - minPX * scale;
  const offsetY = PAD_TOP + ((VB_H - PAD_TOP - PAD_BOT) - spanPY * scale) / 2 - minPY * scale;
  const toScreen = (p: { x: number; y: number; depth: number }) => ({
    sx: p.x * scale + offsetX,
    sy: p.y * scale + offsetY,
    depth: p.depth,
  });

  const vertices: MeshVertex[][] = fitted.map((fit, r) => {
    const s = fit.slice;
    return Array.from({ length: K_STEPS }, (_, c) => {
      const k = K_MIN + (c / (K_STEPS - 1)) * (K_MAX - K_MIN);
      const iv = grid[r][c];
      const p = project(X(k), Y(s.dte), Z(iv));
      const sc = toScreen(p);
      return { k, sx: sc.sx, sy: sc.sy, depth: sc.depth, iv, row: r };
    });
  });

  const floorVerts: { sx: number; sy: number }[][] = fitted.map((fit) => {
    const s = fit.slice;
    return Array.from({ length: K_STEPS }, (_, c) => {
      const k = K_MIN + (c / (K_STEPS - 1)) * (K_MAX - K_MIN);
      return toScreen(project(X(k), Y(s.dte), floorZ));
    });
  });

  const floor: FloorCorner[] = floorWorld.map((f, i) => {
    const sc = toScreen(project(f.x, f.y, floorZ));
    return { key: `f-${i}`, sx: sc.sx, sy: sc.sy };
  });

  const quads: ProjectedQuad[] = [];
  for (let r = 0; r < fitted.length - 1; r++) {
    for (let c = 0; c < K_STEPS - 1; c++) {
      const v00 = vertices[r][c];
      const v01 = vertices[r][c + 1];
      const v10 = vertices[r + 1][c + 1];
      const v11 = vertices[r + 1][c];
      quads.push({
        key: `q-${r}-${c}`,
        pts: `${v00.sx.toFixed(2)},${v00.sy.toFixed(2)} ${v01.sx.toFixed(2)},${v01.sy.toFixed(2)} ${v10.sx.toFixed(2)},${v10.sy.toFixed(2)} ${v11.sx.toFixed(2)},${v11.sy.toFixed(2)}`,
        depth: (v00.depth + v01.depth + v10.depth + v11.depth) / 4,
        ivAvg: (v00.iv + v01.iv + v10.iv + v11.iv) / 4,
      });
    }
  }
  quads.sort((a, b) => b.depth - a.depth);

  // Observed quotes as hollow dots — every slice, fitted or not
  const dots: SurfaceModel['dots'] = [];
  for (const s of usable) {
    for (const p of s.points) {
      const iv = clampIv(p.iv);
      const k = p.k;
      if (k < K_MIN || k > K_MAX) continue;
      const strike = s.forward * Math.exp(k);
      const sc = toScreen(project(X(k), Y(s.dte), Z(iv)));
      dots.push({ sx: sc.sx, sy: sc.sy, depth: sc.depth, iv, k, dte: s.dte, strike });
    }
  }
  dots.sort((a, b) => b.depth - a.depth);

  const atmCol = Math.round(((0 - K_MIN) / (K_MAX - K_MIN)) * (K_STEPS - 1));
  const atmRidge = vertices.map((row) => `${row[atmCol].sx.toFixed(2)},${row[atmCol].sy.toFixed(2)}`).join(' ');

  const dteLabels = fitted.map((fit, r) => ({
    dte: fit.slice.dte,
    sx: vertices[r][0].sx - 10,
    sy: vertices[r][0].sy + 4,
  }));

  const flags = fitted
    .map((fit, r) => ({ row: r, dte: fit.slice.dte, flags: fit.slice.svi?.arbitrageFlags ?? [] }))
    .filter((f) => f.flags.length > 0);

  return { vertices, quads, dots, atmRidge, minIv, maxIv, dteLabels, flags, floor, floorVerts };
}

function ivColor(iv: number, minIv: number, maxIv: number, alpha = 1): string {
  const t = (iv - minIv) / Math.max(1e-6, maxIv - minIv);
  const a = (0.12 + t * 0.88) * alpha;
  return `rgba(222, 192, 132, ${a.toFixed(3)})`;
}

/* ------------------------------------------------------------------ */
/*  Component                                                           */
/* ------------------------------------------------------------------ */

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

interface HoverInfo {
  sx: number;
  sy: number;
  k: number;
  dte: number;
  iv: number;
  strike: number;
  callDelta: number;
  usedQuotes: number;
  arbitrageFlags: string[];
  isObserved: boolean;
}

export function VolSurface3D({ slices, spot, rate, dividendYield, className, fitMethod = 'svi' }: VolSurface3DProps) {
  const [yaw, setYaw] = useState(-26);
  const [pitch, setPitch] = useState(24);
  const [dragging, setDragging] = useState(false);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const dragRef = useRef({ startX: 0, startY: 0, yaw0: -26, pitch0: 24, active: false });
  const svgRef = useRef<SVGSVGElement>(null);

  const r = rate ?? 0;
  const q = dividendYield ?? 0;
  const spot0 = spot ?? 0;

  const { resolved, fellBack } = useMemo(() => resolveSliceFits(slices, fitMethod), [slices, fitMethod]);
  const model = useMemo(() => buildSurface(resolved, slices, yaw, pitch), [resolved, slices, yaw, pitch]);
  const fittedCount = model ? model.vertices.length : 0;
  const fitLabel = fitMethod === 'svi' ? 'SVI' : fitMethod === 'ssvi' ? 'SSVI' : 'Spline';
  const unfittedWithPoints = useMemo(
    () => slices.filter((s) => s.svi === null && s.points.length > 0).length,
    [slices]
  );

  if (!model) {
    return (
      <div className={`rounded-xl border border-white/[0.07] bg-[#0F0F12] p-6 text-sm text-zinc-500 ${className ?? ''}`}>
        3D surface needs at least one slice with a fit (≥ 5 quotes for SVI/SSVI, ≥ 3 for Spline). Available slices: {slices.length}, unfitted slices: {unfittedWithPoints}. Option chain too thin today — check back when OI accrues.
      </div>
    );
  }

  const toSvgCoords = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 900,
      y: ((e.clientY - rect.top) / rect.height) * 430,
    };
  };

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType !== 'mouse' && e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, yaw0: yaw, pitch0: pitch, active: true };
    setDragging(true);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* synthetic/invalid pointer — drag still works via pointermove */
    }
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (drag.active) {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      setYaw(Math.max(-42, Math.min(42, drag.yaw0 + dx * 0.35)));
      setPitch(Math.max(8, Math.min(48, drag.pitch0 - dy * 0.3)));
      return;
    }
    const p = toSvgCoords(e);
    if (!p) return;
    let best: HoverInfo | null = null;
    let bestD = Infinity;
    const fittedSorted = slices
      .filter((s) => s.svi !== null)
      .slice()
      .sort((a, b) => a.dte - b.dte);
    const strikeFor = (k: number, dte: number) => {
      const sl = slices.find((s) => s.dte === dte);
      return sl ? sl.forward * Math.exp(k) : 0;
    };
    for (const row of model.vertices) {
      const slice = fittedSorted[row[0].row];
      for (const v of row) {
        const d2 = (v.sx - p.x) ** 2 + (v.sy - p.y) ** 2;
        if (d2 < bestD) {
          bestD = d2;
          const strike = strikeFor(v.k, slice?.dte ?? 0);
          const { d1 } = d1d2(spot0, strike, slice?.t ?? 0, r, q, v.iv);
          best = {
            sx: v.sx, sy: v.sy, k: v.k, dte: slice?.dte ?? 0, iv: v.iv, strike,
            callDelta: Math.exp(-q * (slice?.t ?? 0)) * normCdf(d1),
            usedQuotes: slice?.svi?.nUsed ?? 0,
            arbitrageFlags: slice?.svi?.arbitrageFlags ?? [],
            isObserved: false,
          };
        }
      }
    }
    for (const d of model.dots) {
      const d2 = (d.sx - p.x) ** 2 + (d.sy - p.y) ** 2;
      if (d2 < bestD) {
        bestD = d2;
        const strike = d.strike;
        const t = (slices.find((s) => s.dte === d.dte)?.t) ?? 0;
        const { d1 } = d1d2(spot0, strike, t, r, q, d.iv);
        best = {
          sx: d.sx, sy: d.sy, k: d.k, dte: d.dte, iv: d.iv, strike,
          callDelta: Math.exp(-q * t) * normCdf(d1),
          usedQuotes: -1,
          arbitrageFlags: [],
          isObserved: true,
        };
      }
    }
    setHover(bestD < 1200 ? best : null);
  };

  const endDrag = (e: React.PointerEvent<SVGSVGElement>) => {
    dragRef.current.active = false;
    setDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  };

  const readout = hover && (
    <div
      className="pointer-events-none absolute z-10 rounded-lg border border-gold-500/40 bg-zinc-950/95 px-2.5 py-1.5 font-mono text-[10px] leading-snug text-zinc-200 shadow-lg"
      style={{ left: `${(hover.sx / 900) * 100}%`, top: `${(hover.sy / 430) * 100}%`, transform: 'translate(-50%, -118%)' }}
    >
      <div className="whitespace-nowrap">
        <span className="text-gold-400">k</span> {hover.k >= 0 ? '+' : ''}{hover.k.toFixed(2)} · <span className="text-gold-400">{hover.dte}d</span> · IV <span className="text-gold-400">{(hover.iv * 100).toFixed(1)}%</span>
        {hover.isObserved && <span className="ml-2 rounded bg-white/10 px-1 text-[8px] uppercase text-zinc-400">quote</span>}
      </div>
      <div className="whitespace-nowrap text-zinc-400">
        K ≈ ${hover.strike.toFixed(2)} · Δ ≈ {hover.callDelta >= 0 ? '+' : ''}{hover.callDelta.toFixed(2)}{!hover.isObserved && ` · quotes ${hover.usedQuotes}`}
      </div>
      {hover.arbitrageFlags.length > 0 && (
        <div className="mt-0.5 flex items-center gap-1 text-amber-400">
          <AlertTriangle className="h-3 w-3" />
          <span className="text-[9px]">SVI arbitrage: {hover.arbitrageFlags.join(', ')}</span>
        </div>
      )}
    </div>
  );

  return (
    <div className={`relative select-none rounded-xl border border-white/[0.07] bg-[#0F0F12] p-3 ${className ?? ''}`}>
      {/* Rotate hint */}
      <div className="pointer-events-none absolute right-4 top-3 z-10 flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.16em] text-zinc-500">
        <RotateCw className="h-3 w-3" /> drag to rotate
      </div>

      <svg
        ref={svgRef}
        viewBox="0 0 900 430"
        role="img"
        aria-label="3D implied volatility surface"
        className={`w-full touch-none ${dragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => setHover(null)}
      >
        {/* Floor grid — the projected base plane */}
        <path
          d={`M ${model.floor.map((f) => `${f.sx.toFixed(1)} ${f.sy.toFixed(1)}`).join(' L ')} Z`}
          fill="rgba(222,192,132,0.03)"
          stroke="rgba(236,236,241,0.07)"
          strokeWidth="0.65"
        />

        {/* Slice curtains — back-to-front, honest for thin chains (1 slice still reads as a ribbon) */}
        {[...model.vertices].reverse().map((row) => {
          const rowId = row[0].row;
          const top = row.map((v) => `${v.sx.toFixed(2)},${v.sy.toFixed(2)}`).join(' L ');
          const bottom = [...model.floorVerts[rowId]].reverse().map((v) => `${v.sx.toFixed(2)},${v.sy.toFixed(2)}`).join(' L ');
          return (
            <path
              key={`curtain-${rowId}`}
              d={`M ${top} L ${bottom} Z`}
              fill="url(#surfaceCurtain)"
              opacity="0.9"
            />
          );
        })}

        {/* Shared vertical fade for curtains */}
        <defs>
          <linearGradient id="surfaceCurtain" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#DEC084" stopOpacity="0.38" />
            <stop offset="45%" stopColor="#DEC084" stopOpacity="0.10" />
            <stop offset="100%" stopColor="#DEC084" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Surface quads — painter's algorithm */}
        {model.quads.map((q) => (
          <polygon
            key={q.key}
            points={q.pts}
            fill={ivColor(q.ivAvg, model.minIv, model.maxIv, 1)}
            stroke="rgba(8,8,10,0.55)"
            strokeWidth="0.4"
            strokeLinejoin="round"
          />
        ))}

        {/* Wireframe along tenor (k columns) */}
        {Array.from({ length: K_STEPS }, (_, c) => (
          <polyline
            key={`wl-${c}`}
            points={model.vertices.map((row) => `${row[c].sx.toFixed(2)},${row[c].sy.toFixed(2)}`).join(' ')}
            fill="none"
            stroke="rgba(226,205,160,0.35)"
            strokeWidth={c % 4 === 0 ? 0.9 : 0.35}
          />
        ))}
        {/* Wireframe along k (slice rows) */}
        {model.vertices.map((row, r) => (
          <polyline
            key={`wr-${r}`}
            points={row.map((v) => `${v.sx.toFixed(2)},${v.sy.toFixed(2)}`).join(' ')}
            fill="none"
            stroke="rgba(226,205,160,0.4)"
            strokeWidth="0.65"
          />
        ))}

        {/* Observed quote dots — hollow white, unfitted slices included */}
        {model.dots.map((d, i) => (
          <circle
            key={`d-${i}`}
            cx={d.sx.toFixed(2)}
            cy={d.sy.toFixed(2)}
            r="2.1"
            fill="rgba(236,236,241,0.10)"
            stroke="rgba(236,236,241,0.65)"
            strokeWidth="0.9"
          />
        ))}

        {/* ATM ridge (k = 0) — the money spine */}
        <polyline
          points={model.atmRidge}
          fill="none"
          stroke="#DEC084"
          strokeWidth="2.2"
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity="0.95"
        />

        {/* DTE labels on the left flank */}
        {model.dteLabels.map(({ dte, sx, sy }) => (
          <text key={`dte-${dte}`} x={sx} y={sy} textAnchor="end" className="fill-zinc-500 font-mono" fontSize="10">
            {dte}d
          </text>
        ))}

        {/* k axis labels on the front edge */}
        {[-0.4, -0.2, 0, 0.2, 0.4].map((k) => {
          const c = Math.round(((k - K_MIN) / (K_MAX - K_MIN)) * (K_STEPS - 1));
          const v = model.vertices[model.vertices.length - 1][c];
          return (
            <text key={`k-${k}`} x={v.sx} y={v.sy + 16} textAnchor="middle" className="fill-zinc-500 font-mono" fontSize="9">
              {k > 0 ? `+${k.toFixed(1)}` : k.toFixed(1)}
            </text>
          );
        })}

        {/* IV scale caption */}
        <text x="886" y="24" textAnchor="end" className="fill-zinc-400 font-mono" fontSize="10">
          IV {(model.minIv * 100).toFixed(0)}%
        </text>
        <text x="886" y="40" textAnchor="end" className="fill-gold-400 font-mono" fontSize="10">
          peak {(model.maxIv * 100).toFixed(0)}%
        </text>

        {/* Arbitrage-flag ticks on the left flank */}
        {model.flags.map((f) => {
          const v = model.vertices[f.row][0];
          return (
            <g key={`arb-${f.row}`} transform={`translate(${v.sx + 5}, ${v.sy - 8})`}>
              <circle r="3.2" fill="#f59e0b" opacity="0.9" />
            </g>
          );
        })}

        {/* Hover marker */}
        {hover && (
          <g>
            <circle cx={hover.sx} cy={hover.sy} r="4.5" fill="#DEC084" stroke="#0B0B0D" strokeWidth="1.4" />
            <circle cx={hover.sx} cy={hover.sy} r="9" fill="none" stroke="#DEC084" strokeWidth="0.8" opacity="0.5" />
          </g>
        )}
      </svg>

      {readout}

      {/* Legend + honesty footer */}
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-2 font-mono text-[9px] text-zinc-500">
        <div className="flex items-center gap-2">
          <span>{(model.minIv * 100).toFixed(0)}%</span>
          <div className="h-2 w-32 rounded-full" style={{ background: 'linear-gradient(90deg, rgba(222,192,132,0.12), rgba(222,192,132,0.55), rgba(222,192,132,1))' }} />
          <span>{(model.maxIv * 100).toFixed(0)}%</span>
          <span className="text-zinc-600">·</span>
          <svg width="26" height="6" className="inline-block align-middle"><rect width="26" height="6" rx="1" fill="rgba(222,192,132,0.35)" stroke="#DEC084" strokeWidth="1.4" /></svg>
          <span>ATM ridge</span>
          <span className="text-zinc-600">·</span>
          <span className="inline-block h-2 w-2 rounded-full border border-zinc-400/70 align-middle" />
          <span>observed quotes</span>
        </div>
        <div>{fittedCount} fitted ({fitLabel}{fellBack > 0 ? ` +${fellBack} SVI fallback` : ''}) · {model.dots.length} quotes · k = ln(K/F) · European BSM · ~15-min delayed · not executable</div>
      </div>
    </div>
  );
}
