'use client';

import React, { useState } from 'react';
import {
  NewsImpactPoint,
  VolatilitySignaturePoint,
  OvernightIntradaySplitDay,
} from '@/lib/analysis-engine';
import { DailyRealizedVol, DiurnalPoint } from '@/lib/quant-engine';
import { IntradayDiurnalChart } from '@/components/charts/IntradayDiurnalChart';
import { JumpDecompositionChart } from '@/components/charts/JumpDecompositionChart';
import {
  NewsImpactEquation,
  SignaturePlotEquation,
} from '@/components/math/MathEquations';
import {
  Zap,
  Clock,
  Info,
} from 'lucide-react';

interface AnalysisDynamicsTabProps {
  newsImpactPoints: NewsImpactPoint[];
  volSignaturePoints: VolatilitySignaturePoint[];
  overnightIntradaySplit: OvernightIntradaySplitDay[];
  dailyRealizedVol: DailyRealizedVol[];
  diurnalProfile: DiurnalPoint[];
  ticker: string;
}

export function AnalysisDynamicsTab({
  newsImpactPoints,
  volSignaturePoints,
  overnightIntradaySplit,
  dailyRealizedVol,
  diurnalProfile,
  ticker,
}: AnalysisDynamicsTabProps) {
  const [newsHoverIdx, setNewsHoverIdx] = useState<number | null>(null);
  const [sigHoverIdx, setSigHoverIdx] = useState<number | null>(null);
  const [splitHoverIdx, setSplitHoverIdx] = useState<number | null>(null);

  // Aggregate overnight share
  const avgOvernightPct = React.useMemo(() => {
    if (overnightIntradaySplit.length === 0) return 0;
    const sum = overnightIntradaySplit.reduce((acc, d) => acc + d.overnightRatioPct, 0);
    return sum / overnightIntradaySplit.length;
  }, [overnightIntradaySplit]);

  // Jump statistics
  const jumpStats = React.useMemo(() => {
    if (dailyRealizedVol.length === 0) return { jumpDays: 0, totalDays: 0, maxZ: 0 };
    const significantDays = dailyRealizedVol.filter((d) => d.hasSignificantJump).length;
    const maxZ = Math.max(...dailyRealizedVol.map((d) => d.jumpZStat));
    return {
      jumpDays: significantDays,
      totalDays: dailyRealizedVol.length,
      maxZ: isFinite(maxZ) ? maxZ : 0,
    };
  }, [dailyRealizedVol]);

  // News Impact pure SVG scaling
  const nNews = newsImpactPoints.length;
  let maxNewsVar = 0.0001;
  for (const p of newsImpactPoints) {
    if (p.asymmetricVarianceGjr > maxNewsVar) maxNewsVar = p.asymmetricVarianceGjr;
    if (p.symmetricVariance > maxNewsVar) maxNewsVar = p.symmetricVariance;
  }
  maxNewsVar *= 1.1;

  const newsW = 700;
  const newsH = 240;
  const nPadL = 50;
  const nPadR = 25;
  const nPadT = 20;
  const nPadB = 35;
  const ncW = newsW - nPadL - nPadR;
  const ncH = newsH - nPadT - nPadB;

  const getNewsX = (i: number) => nPadL + (i / Math.max(1, nNews - 1)) * ncW;
  const getNewsY = (val: number) => nPadT + ncH - (val / maxNewsVar) * ncH;

  const activeNewsIdx = newsHoverIdx !== null ? newsHoverIdx : Math.floor(nNews / 2);
  const activeNewsPoint = newsImpactPoints[activeNewsIdx] || { shockStd: 0, symmetricVariance: 0, asymmetricVarianceGjr: 0 };

  // Volatility Signature pure SVG scaling
  const nSig = volSignaturePoints.length;
  let minSigRv = Infinity;
  let maxSigRv = -Infinity;
  for (const p of volSignaturePoints) {
    if (p.annualizedRv < minSigRv) minSigRv = p.annualizedRv;
    if (p.annualizedRv > maxSigRv) maxSigRv = p.annualizedRv;
  }
  const sigPad = (maxSigRv - minSigRv) * 0.15 || 0.05;
  const sigMin = Math.max(0, minSigRv - sigPad);
  const sigMax = maxSigRv + sigPad;
  const sigRange = sigMax - sigMin || 1;

  const sigW = 550;
  const sigH = 210;
  const sPadL = 50;
  const sPadR = 25;
  const sPadT = 20;
  const sPadB = 35;
  const scW = sigW - sPadL - sPadR;
  const scH = sigH - sPadT - sPadB;

  const getSigX = (i: number) => sPadL + (i / Math.max(1, nSig - 1)) * scW;
  const getSigY = (val: number) => sPadT + scH - ((val - sigMin) / sigRange) * scH;

  const activeSigIdx = sigHoverIdx !== null ? sigHoverIdx : 0;
  const activeSigPoint = volSignaturePoints[activeSigIdx] || { samplingMinutes: 1, annualizedRv: 0 };

  // Overnight vs Intraday Split scaling
  const nSplit = overnightIntradaySplit.length;
  let maxSplitVar = 0.0001;
  for (const p of overnightIntradaySplit) {
    const tot = p.intradayRv + p.overnightVar;
    if (tot > maxSplitVar) maxSplitVar = tot;
  }
  maxSplitVar *= 1.15;

  const spW = 550;
  const spH = 210;
  const spPadL = 45;
  const spPadR = 20;
  const spPadT = 20;
  const spPadB = 30;
  const spcW = spW - spPadL - spPadR;
  const spcH = spH - spPadT - spPadB;

  const getSpX = (i: number) => spPadL + (i / Math.max(1, nSplit - 1)) * spcW;
  const getSpY = (val: number) => spPadT + spcH - (val / maxSplitVar) * spcH;

  const activeSplitIdx = splitHoverIdx !== null ? splitHoverIdx : nSplit - 1;
  const activeSplitPoint = overnightIntradaySplit[activeSplitIdx] || { date: '', intradayRv: 0, overnightVar: 0, overnightRatioPct: 0 };

  return (
    <div className="space-y-6">
      {/* GJR-GARCH News Impact Curve */}
      <div className="panel p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-zinc-100 dark:border-zinc-800/60 gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-gold-500" />
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                News Impact Curve (Symmetric GARCH vs. Asymmetric GJR-GARCH)
              </h3>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
              Empirical asymmetric volatility response: negative return shocks trigger larger conditional variance spikes
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-zinc-400">
            <span>
              Shock: <strong>{activeNewsPoint.shockStd > 0 ? `+${activeNewsPoint.shockStd}` : activeNewsPoint.shockStd}σ</strong>
            </span>
            <span className="text-zinc-300 dark:text-zinc-700">·</span>
            <span>
              GJR Var: <strong className="text-gold-500">{activeNewsPoint.asymmetricVarianceGjr.toFixed(2)}</strong>
            </span>
            <NewsImpactEquation className="hidden lg:inline" />
          </div>
        </div>

        <div className="mt-4">
          <svg
            viewBox={`0 0 ${newsW} ${newsH}`}
            className="w-full h-64 select-none"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const mouseX = e.clientX - rect.left;
              const ratio = Math.max(0, Math.min(1, (mouseX - nPadL) / ncW));
              const idx = Math.round(ratio * (nNews - 1));
              setNewsHoverIdx(idx);
            }}
            onMouseLeave={() => setNewsHoverIdx(null)}
          >
            {/* Gridlines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
              const yVal = pct * maxNewsVar;
              const yPos = getNewsY(yVal);
              return (
                <g key={pct}>
                  <line
                    x1={nPadL}
                    y1={yPos}
                    x2={newsW - nPadR}
                    y2={yPos}
                    stroke="#334155"
                    strokeDasharray="2 2"
                    strokeWidth={0.8}
                    opacity={0.3}
                  />
                  <text x={nPadL - 8} y={yPos + 3} textAnchor="end" fontSize="10" fill="#64748b" className="font-mono">
                    {yVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Zero shock vertical line */}
            <line
              x1={getNewsX(Math.floor(nNews / 2))}
              y1={nPadT}
              x2={getNewsX(Math.floor(nNews / 2))}
              y2={newsH - nPadB}
              stroke="#64748b"
              strokeDasharray="3 3"
              strokeWidth={1}
              opacity={0.4}
            />

            {/* Symmetric GARCH Curve */}
            {nNews > 1 && (
              <path
                d={newsImpactPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getNewsX(i)} ${getNewsY(p.symmetricVariance)}`).join(' ')}
                fill="none"
                stroke="#64748b"
                strokeDasharray="4 4"
                strokeWidth={1.8}
              />
            )}

            {/* Asymmetric GJR-GARCH Curve */}
            {nNews > 1 && (
              <path
                d={newsImpactPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getNewsX(i)} ${getNewsY(p.asymmetricVarianceGjr)}`).join(' ')}
                fill="none"
                stroke="#10b981"
                strokeWidth={2.5}
              />
            )}

            {/* Hover Indicator */}
            {activeNewsIdx >= 0 && activeNewsIdx < nNews && (
              <circle
                cx={getNewsX(activeNewsIdx)}
                cy={getNewsY(activeNewsPoint.asymmetricVarianceGjr)}
                r="4"
                fill="#10b981"
              />
            )}
          </svg>
        </div>

        <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mt-1">
          <span>-5σ (Catastrophic Selloff)</span>
          <div className="flex items-center gap-4">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-zinc-400 border-b border-dashed inline-block" /> Symmetric GARCH(1,1)
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-gold-500 inline-block" /> GJR-GARCH (Leverage γ &gt; 0)
            </span>
          </div>
          <span>+5σ (Extreme Rally)</span>
        </div>

        <div className="mt-3 pt-3 border-t border-zinc-100 dark:border-zinc-800/60 text-xs text-zinc-500 dark:text-zinc-400 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>
              The asymmetric tilt confirms the Black (1976) financial leverage hypothesis: bad news lowers firm equity,
              increasing debt-to-equity ratio and magnifying downstream asset risk.
            </span>
          </div>
        </div>
      </div>

      {/* Volatility Signature Plot & Overnight vs Intraday Split */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Volatility Signature Plot */}
        <div className="panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Volatility Signature Plot (Andersen et al., 2000)
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Sampling frequency vs. Realized Volatility: isolates microstructure noise bias
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              {activeSigPoint.samplingMinutes}m: <strong className="text-gold-600 dark:text-gold-300">{(activeSigPoint.annualizedRv * 100).toFixed(2)}%</strong>
            </div>
          </div>

          <div className="mt-4">
            <svg
              viewBox={`0 0 ${sigW} ${sigH}`}
              className="w-full h-52 select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, (mouseX - sPadL) / scW));
                const idx = Math.round(ratio * (nSig - 1));
                setSigHoverIdx(idx);
              }}
              onMouseLeave={() => setSigHoverIdx(null)}
            >
              {/* Horizontal Gridlines */}
              {[0, 0.33, 0.66, 1].map((pct) => {
                const val = sigMin + pct * sigRange;
                const yPos = getSigY(val);
                return (
                  <g key={pct}>
                    <line x1={sPadL} y1={yPos} x2={sigW - sPadR} y2={yPos} stroke="#334155" strokeDasharray="2 2" strokeWidth={0.8} opacity={0.3} />
                    <text x={sPadL - 8} y={yPos + 3} textAnchor="end" fontSize="10" fill="#64748b" className="font-mono">
                      {(val * 100).toFixed(1)}%
                    </text>
                  </g>
                );
              })}

              {/* Signature Curve */}
              {nSig > 1 && (
                <path
                  d={volSignaturePoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getSigX(i)} ${getSigY(p.annualizedRv)}`).join(' ')}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={2.2}
                />
              )}

              {/* Data points */}
              {volSignaturePoints.map((p, i) => (
                <circle
                  key={p.samplingMinutes}
                  cx={getSigX(i)}
                  cy={getSigY(p.annualizedRv)}
                  r={i === activeSigIdx ? '5' : '3'}
                  fill={i === activeSigIdx ? '#60a5fa' : '#3b82f6'}
                />
              ))}
            </svg>
          </div>

          <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mt-1">
            <span>1m (High Noise)</span>
            <span>Sampling Horizon</span>
            <span>60m (Diffusion Plateau)</span>
          </div>

          <div className="mt-3 text-[11px] text-zinc-400">
            Microstructure noise inflates 1-minute RV; the curve stabilizes around the 5–15 min frequency, which provides
            the optimal bias-variance tradeoff for intraday pricing.
          </div>
        </div>

        {/* Overnight vs Intraday Split */}
        <div className="panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Overnight vs. Intraday Variance Decomposition
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Earnings releases &amp; macro news break outside market hours
              </p>
            </div>
            <div className="font-mono text-xs text-gold-600 dark:text-gold-400">
              Avg Overnight Share: <strong>{avgOvernightPct.toFixed(1)}%</strong>
            </div>
          </div>

          <div className="mt-4">
            <svg
              viewBox={`0 0 ${spW} ${spH}`}
              className="w-full h-52 select-none"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, (mouseX - spPadL) / spcW));
                const idx = Math.round(ratio * (nSplit - 1));
                setSplitHoverIdx(idx);
              }}
              onMouseLeave={() => setSplitHoverIdx(null)}
            >
              {/* Stacked Bars / Areas */}
              {overnightIntradaySplit.map((d, i) => {
                const x = getSpX(i);
                const barW = Math.max(8, (spcW / nSplit) * 0.65);
                const total = d.intradayRv + d.overnightVar;
                const yTot = getSpY(total);
                const yIntra = getSpY(d.intradayRv);
                const baseY = spPadT + spcH;

                return (
                  <g key={d.date} className="cursor-pointer">
                    {/* Intraday portion (bottom) */}
                    <rect
                      x={x - barW / 2}
                      y={yIntra}
                      width={barW}
                      height={baseY - yIntra}
                      fill="#10b981"
                      fillOpacity={0.8}
                      rx={2}
                    />
                    {/* Overnight portion (top) */}
                    <rect
                      x={x - barW / 2}
                      y={yTot}
                      width={barW}
                      height={yIntra - yTot}
                      fill="#DEC084"
                      fillOpacity={0.8}
                      rx={2}
                    />
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="flex items-center justify-between text-[11px] text-zinc-400 font-mono mt-1">
            <span>{overnightIntradaySplit[0]?.date}</span>
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-gold-500 rounded-xs inline-block" /> Intraday RV
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 bg-gold-500 rounded-xs inline-block" /> Overnight Gap Var
              </span>
            </div>
            <span>{overnightIntradaySplit[nSplit - 1]?.date}</span>
          </div>

          <div className="mt-3 text-[11px] text-zinc-400">
            Overnight variance represents gap risk unhedged by standard intraday stops. High overnight ratios favor option
            straddles over delta-neutral spot market-making.
          </div>
        </div>
      </div>

      {/* Intraday 390-min Diurnal Profile & Jump Decomposition */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Diurnal U-Shape Profile */}
        <div className="panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                390-Minute Intraday Diurnal Volatility Profile
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                U-shaped intraday volatility smile: open rush → midday lull → market-on-close (MOC)
              </p>
            </div>
            <Clock className="w-4 h-4 text-gold-500" />
          </div>

          <div className="mt-4">
            <IntradayDiurnalChart points={diurnalProfile} ticker={ticker} />
          </div>
        </div>

        {/* Jump Decomposition */}
        <div className="panel p-5">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-100 dark:border-zinc-800/60">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Bipower Variation &amp; Jump Shock Decomposition
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Barndorff-Nielsen &amp; Shephard: RV = Continuous BV + Discrete Jump J_t
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-400">
              Significant Jumps: <strong className="text-gold-500">{jumpStats.jumpDays}</strong> / {jumpStats.totalDays} days
            </div>
          </div>

          <div className="mt-4">
            <JumpDecompositionChart dailyRV={dailyRealizedVol} ticker={ticker} />
          </div>
        </div>
      </div>
    </div>
  );
}
