'use client';

import React, { useState } from 'react';
import {
  AdvancedCorrelationMatrixResult,
  generateLatexCorrelationTable,
} from '@/lib/correlation-engine';
import {
  MathDisplay,
  ShrinkageEquation,
  FisherZEquation,
  MarchenkoPasturEquation,
  PartialCorrelationEquation,
} from '@/components/math/MathEquations';
import {
  Download,
  Copy,
  Check,
  Info,
  Filter,
  BarChart2,
  TrendingUp,
  Layers,
  Activity,
  ArrowUpRight,
  ShieldAlert,
} from 'lucide-react';

interface CorrelationOverviewTabProps {
  result: AdvancedCorrelationMatrixResult;
  returnsByTicker: { [ticker: string]: number[] };
  dates: string[];
}

export function CorrelationOverviewTab({
  result,
  returnsByTicker,
  dates,
}: CorrelationOverviewTabProps) {
  const [matrixMode, setMatrixMode] = useState<
    'pearson' | 'spearman' | 'kendall' | 'partial' | 'shrunk'
  >('pearson');
  const [useClustered, setUseClustered] = useState<boolean>(false);
  const [filterInsignificant, setFilterInsignificant] = useState<boolean>(false);
  const [selectedPair, setSelectedPair] = useState<{ tA: string; tB: string } | null>(null);
  const [copiedLatex, setCopiedLatex] = useState<boolean>(false);
  const [showLatexModal, setShowLatexModal] = useState<boolean>(false);

  const {
    tickers,
    matrix,
    pValues,
    fdrSignificantMask,
    spearmanMatrix,
    kendallMatrix,
    partialCorrelationMatrix,
    shrunkCorrelation,
    shrinkageIntensity,
    conditionNumberRaw,
    conditionNumberShrunk,
    clusteredTickers,
    clusteredMatrix,
    termStructure,
    rmtSpectrum,
  } = result;

  // Active display tickers and matrix based on toggles
  const activeTickers = useClustered ? clusteredTickers : tickers;

  const getActiveMatrix = () => {
    if (useClustered && matrixMode === 'pearson') return clusteredMatrix;
    switch (matrixMode) {
      case 'spearman':
        return spearmanMatrix;
      case 'kendall':
        return kendallMatrix;
      case 'partial':
        return partialCorrelationMatrix;
      case 'shrunk':
        return shrunkCorrelation;
      case 'pearson':
      default:
        return matrix;
    }
  };

  const displayMatrix = getActiveMatrix();

  // Active pair selection
  const activePair =
    selectedPair ||
    (activeTickers.length >= 2 ? { tA: activeTickers[0], tB: activeTickers[1] } : null);

  const idxA = activePair ? tickers.indexOf(activePair.tA) : 0;
  const idxB = activePair ? tickers.indexOf(activePair.tB) : 1;

  const pairCorr = matrix[idxA]?.[idxB] ?? 0;
  const pairSpearman = spearmanMatrix[idxA]?.[idxB] ?? pairCorr;
  const pairPVal = pValues[idxA]?.[idxB] ?? 0.05;
  const isFdrSig = fdrSignificantMask[idxA]?.[idxB] ?? true;

  const retA = activePair ? returnsByTicker[activePair.tA] || [] : [];
  const retB = activePair ? returnsByTicker[activePair.tB] || [] : [];
  const scatterLen = Math.min(retA.length, retB.length);

  // Scatter points
  const scatterPoints: { x: number; y: number }[] = [];
  for (let i = 0; i < scatterLen; i++) {
    scatterPoints.push({ x: retA[i], y: retB[i] });
  }

  // Linear regression slope for scatter plot
  let sXY = 0;
  let sXX = 0;
  let mA = 0;
  let mB = 0;
  for (let i = 0; i < scatterLen; i++) {
    mA += retA[i];
    mB += retB[i];
  }
  mA /= Math.max(1, scatterLen);
  mB /= Math.max(1, scatterLen);

  for (let i = 0; i < scatterLen; i++) {
    const dx = retA[i] - mA;
    const dy = retB[i] - mB;
    sXY += dx * dy;
    sXX += dx * dx;
  }
  const slope = sXX > 1e-12 ? sXY / sXX : 1.0;
  const intercept = mB - slope * mA;

  // Color interpolator for correlation:
  const getCellBg = (corr: number, isMasked: boolean) => {
    if (isMasked) return 'rgba(100, 116, 139, 0.12)';
    if (corr >= 0) {
      const alpha = Math.min(0.9, Math.max(0.08, corr * 0.85));
      return `rgba(14, 165, 233, ${alpha})`;
    } else {
      const alpha = Math.min(0.9, Math.max(0.08, Math.abs(corr) * 0.85));
      return `rgba(244, 63, 94, ${alpha})`;
    }
  };

  const getTextColor = (corr: number, isMasked: boolean) => {
    if (isMasked) return '#94a3b8';
    if (Math.abs(corr) > 0.45) return '#ffffff';
    return undefined;
  };

  // CSV Export
  const handleExportCsv = () => {
    const headers = ['Asset', ...activeTickers].join(',');
    const rows = activeTickers.map((rowTicker, i) => {
      const vals = activeTickers.map((_, j) => displayMatrix[i]?.[j]?.toFixed(4) ?? '1.0000');
      return [rowTicker, ...vals].join(',');
    });
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers, ...rows].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `correlation_matrix_${matrixMode}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // LaTeX Copy
  const latexCode = generateLatexCorrelationTable(tickers, matrix, pValues);
  const handleCopyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    setCopiedLatex(true);
    setTimeout(() => setCopiedLatex(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Matrix Controls & View Mode Toolbar */}
      <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 rounded-xl p-4 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Mode Switcher */}
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-zinc-400 font-mono text-[11px] mr-1">Matrix Metric:</span>
            {[
              { id: 'pearson', label: 'Pearson Linear (ρ)' },
              { id: 'spearman', label: 'Spearman Rank' },
              { id: 'kendall', label: "Kendall's Tau (τ)" },
              { id: 'partial', label: 'Partial Corr (Ω)' },
              { id: 'shrunk', label: 'Ledoit-Wolf Shrunk' },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => setMatrixMode(m.id as any)}
                className={`px-2.5 py-1.5 rounded-lg font-medium transition-all ${
                  matrixMode === m.id
                    ? 'bg-gold-500 text-zinc-950 font-bold shadow-xs'
                    : 'bg-zinc-100 dark:bg-zinc-800/80 text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Filtering & Layout Toggles */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Clustered Ordering Toggle */}
            <button
              onClick={() => setUseClustered(!useClustered)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                useClustered
                  ? 'bg-gold-500/10 border-gold-500/40 text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300'
                  : 'border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>{useClustered ? 'Clustered (Seriated)' : 'Original Order'}</span>
            </button>

            {/* Significance Mask Toggle */}
            <button
              onClick={() => setFilterInsignificant(!filterInsignificant)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                filterInsignificant
                  ? 'bg-gold-500/10 border-gold-500/40 text-gold-600 dark:text-gold-400'
                  : 'border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400'
              }`}
            >
              <Filter className="w-3.5 h-3.5" />
              <span>{filterInsignificant ? 'Masking Insignificant (FDR)' : 'Show All Cells'}</span>
            </button>

            {/* Export Menu */}
            <div className="flex items-center gap-1">
              <button
                onClick={handleExportCsv}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:border-gold-500 text-xs transition-colors"
                title="Download Matrix as CSV"
              >
                <Download className="w-3.5 h-3.5" />
                <span>CSV</span>
              </button>
              <button
                onClick={() => setShowLatexModal(true)}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:border-gold-500 text-xs transition-colors"
                title="View and copy LaTeX formatted table"
              >
                <Copy className="w-3.5 h-3.5" />
                <span>LaTeX</span>
              </button>
            </div>
          </div>
        </div>

        {/* Dynamic Mode Explanatory Pill */}
        <div className="mt-3 pt-2 border-t border-zinc-100 dark:border-zinc-800 text-[11px] text-zinc-500 dark:text-zinc-400 flex flex-wrap items-center justify-between gap-2">
          <div>
            {matrixMode === 'pearson' && 'Standard linear co-movement normalized by standard deviations.'}
            {matrixMode === 'spearman' && 'Non-parametric monotonic rank correlation robust to non-linear co-jumps.'}
            {matrixMode === 'kendall' && 'Concordance probability quantifying joint ordinal pair consistency.'}
            {matrixMode === 'partial' && 'Direct asset co-movement controlling for market & third-party common drivers (via precision matrix Ω).'}
            {matrixMode === 'shrunk' && 'Ledoit-Wolf analytical shrinkage mitigating sample noise and conditioning inversion instability.'}
          </div>
          <div className="font-mono text-zinc-400">
            {useClustered ? 'Hierarchical quasi-diagonalized order active' : 'Natural user entry order'}
          </div>
        </div>
      </div>

      {/* Main Heatmap & Cell Drill-Down Drawer Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap Matrix Table */}
        <div className="lg:col-span-7 panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                {matrixMode === 'pearson' && 'Cross-Asset Pearson Correlation Heatmap'}
                {matrixMode === 'spearman' && 'Spearman Non-Parametric Rank Correlation Heatmap'}
                {matrixMode === 'kendall' && "Kendall's Tau Concordance Correlation Heatmap"}
                {matrixMode === 'partial' && 'Partial Correlation Heatmap (Direct Dependencies)'}
                {matrixMode === 'shrunk' && 'Ledoit-Wolf Shrunk Correlation Heatmap'}
              </h3>
              <span className="text-[11px] text-zinc-500">
                Click any cell to drill into pairwise co-movement, Fisher z-test, and regression slope
              </span>
            </div>

            <span className="text-xs font-mono text-zinc-400">
              {activeTickers.length} × {activeTickers.length} Matrix
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-800 border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden text-xs font-mono">
              <thead className="bg-zinc-50 dark:bg-zinc-900/80">
                <tr>
                  <th className="py-2.5 px-3 text-left font-semibold text-zinc-500">Asset</th>
                  {activeTickers.map((t) => (
                    <th key={t} className="py-2.5 px-3 text-center font-bold text-zinc-700 dark:text-zinc-300">
                      {t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800/80">
                {activeTickers.map((rowTicker, i) => (
                  <tr key={rowTicker} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/40">
                    <td className="py-2.5 px-3 font-bold text-zinc-900 dark:text-zinc-100 bg-zinc-50/50 dark:bg-zinc-900/60 whitespace-nowrap">
                      {rowTicker}
                    </td>
                    {activeTickers.map((colTicker, j) => {
                      const corr = displayMatrix[i]?.[j] ?? (i === j ? 1 : 0);
                      const isSelf = i === j;
                      const origIdxI = tickers.indexOf(rowTicker);
                      const origIdxJ = tickers.indexOf(colTicker);
                      const isSignificant = fdrSignificantMask[origIdxI]?.[origIdxJ] ?? true;
                      const isMasked = filterInsignificant && !isSignificant && !isSelf;

                      const isSelected =
                        activePair &&
                        ((activePair.tA === rowTicker && activePair.tB === colTicker) ||
                          (activePair.tA === colTicker && activePair.tB === rowTicker));

                      return (
                        <td
                          key={colTicker}
                          onClick={() => !isSelf && setSelectedPair({ tA: rowTicker, tB: colTicker })}
                          style={{
                            backgroundColor: getCellBg(corr, isMasked),
                            color: getTextColor(corr, isMasked),
                          }}
                          className={`py-2 px-2.5 text-center font-bold tabular-nums cursor-pointer transition-all ${
                            isSelected
                              ? 'ring-2 ring-gold-400 font-extrabold z-10 scale-[1.03]'
                              : 'hover:opacity-90'
                          } ${isSelf ? 'cursor-default opacity-60 font-medium' : ''} ${
                            isMasked ? 'line-through text-zinc-400' : ''
                          }`}
                        >
                          {corr.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Color Ramp Legend */}
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-500 dark:text-zinc-400 mt-4 pt-3 border-t border-zinc-100 dark:border-zinc-800">
            <div className="flex items-center gap-2">
              <span className="text-[11px]">Negative (-1.0)</span>
              <span className="w-8 h-2 rounded bg-rose-500 inline-block" />
              <span className="w-8 h-2 rounded bg-zinc-400 inline-block" />
              <span className="w-8 h-2 rounded bg-gold-500 inline-block" />
              <span className="text-[11px]">Positive (+1.0)</span>
            </div>
            {filterInsignificant && (
              <span className="text-[11px] text-gold-500 font-mono">
                * Struck-through cells fail Benjamini-Hochberg FDR at q = 0.05
              </span>
            )}
          </div>
        </div>

        {/* Pairwise Drill-Down & Return Co-Movement Drawer */}
        <div className="lg:col-span-5 panel p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
              <div className="flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-gold-500" />
                <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                  Pairwise Drill-Down Inspection
                </h4>
              </div>
              {activePair && (
                <span className="text-xs font-mono font-bold text-gold-600 dark:text-gold-400">
                  {activePair.tA} / {activePair.tB}
                </span>
              )}
            </div>

            {activePair ? (
              <div className="space-y-4">
                {/* Metrics Readout Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2 bg-zinc-50 dark:bg-zinc-900/60 rounded border border-zinc-200 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block">Pearson ρ</span>
                    <span className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">
                      {pairCorr.toFixed(3)}
                    </span>
                  </div>
                  <div className="p-2 bg-zinc-50 dark:bg-zinc-900/60 rounded border border-zinc-200 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block">Spearman</span>
                    <span className="font-bold text-gold-600 dark:text-gold-300 text-sm">
                      {pairSpearman.toFixed(3)}
                    </span>
                  </div>
                  <div className="p-2 bg-zinc-50 dark:bg-zinc-900/60 rounded border border-zinc-200 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block">Fisher p-val</span>
                    <span
                      className={`font-bold text-sm ${
                        pairPVal < 0.05 ? 'text-gold-500' : 'text-gold-500'
                      }`}
                    >
                      {pairPVal < 0.0001 ? '<0.0001' : pairPVal.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-2 bg-zinc-50 dark:bg-zinc-900/60 rounded border border-zinc-200 dark:border-zinc-800">
                    <span className="text-[10px] text-zinc-400 block">FDR Status</span>
                    <span
                      className={`font-bold text-xs ${
                        isFdrSig ? 'text-gold-500' : 'text-rose-500'
                      }`}
                    >
                      {isFdrSig ? 'Significant' : 'Noise'}
                    </span>
                  </div>
                </div>

                {/* Return Scatter Chart with OLS Line */}
                <div>
                  <div className="flex items-center justify-between text-xs text-zinc-500 mb-1">
                    <span>Synchronized Return Scatter &amp; Regression Slope</span>
                    <span className="font-mono">β = {slope.toFixed(3)}</span>
                  </div>
                  <div className="w-full aspect-[4/3] bg-zinc-50 dark:bg-zinc-900/80 rounded-lg border border-zinc-200 dark:border-zinc-800 relative p-2">
                    <svg viewBox="0 0 300 200" className="w-full h-full overflow-visible">
                      {/* Zero Crosshairs */}
                      <line x1="20" y1="100" x2="280" y2="100" stroke="#64748b" strokeWidth="0.8" strokeDasharray="2 2" />
                      <line x1="150" y1="20" x2="150" y2="180" stroke="#64748b" strokeWidth="0.8" strokeDasharray="2 2" />

                      {/* Regression Line */}
                      <line
                        x1="30"
                        y1={100 - (slope * (-0.05) + intercept) * 1200}
                        x2="270"
                        y2={100 - (slope * 0.05 + intercept) * 1200}
                        stroke="#10b981"
                        strokeWidth="1.5"
                      />

                      {/* Scatter points */}
                      {scatterPoints.slice(0, 180).map((pt, idx) => {
                        const cx = 150 + pt.x * 1200;
                        const cy = 100 - pt.y * 1200;
                        return (
                          <circle
                            key={idx}
                            cx={Math.max(25, Math.min(275, cx))}
                            cy={Math.max(25, Math.min(175, cy))}
                            r="2.5"
                            fill="#38bdf8"
                            opacity="0.65"
                          />
                        );
                      })}

                      {/* Axis Labels */}
                      <text x="275" y="95" textAnchor="end" className="text-[9px] font-mono fill-zinc-500">
                        {activePair.tA} Return
                      </text>
                      <text x="155" y="25" textAnchor="start" className="text-[9px] font-mono fill-zinc-500">
                        {activePair.tB} Return
                      </text>
                    </svg>
                  </div>
                </div>

                <div className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  <strong>Statistical Interpretation: </strong>
                  Fisher z-transform establishes a test statistic confirming{' '}
                  <span className="font-mono text-zinc-800 dark:text-zinc-200">
                    r = {pairCorr.toFixed(3)}
                  </span>{' '}
                  is {isFdrSig ? 'statistically distinct from zero' : 'likely sampling noise'}{' '}
                  after multiple testing adjustments (N = {scatterLen}).
                </div>
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-xs text-zinc-400">
                Select an asset pair from the heatmap matrix to view drill-down details.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Multi-Horizon Term Structure & Random Matrix Theory (RMT) Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Correlation Term Structure */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Correlation Term Structure Across Horizons
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Evaluation across daily, weekly, monthly, and quarterly horizons
              </p>
            </div>
            <span className="text-xs font-mono text-gold-600 dark:text-gold-300 dark:text-gold-600 dark:text-gold-300 font-semibold">
              Hedge Horizon Diagnostic
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {termStructure.horizons.map((h) => {
              const hMat = termStructure.matrices[h.days] || matrix;
              // Average pairwise correlation at this horizon
              let s = 0;
              let c = 0;
              for (let i = 0; i < tickers.length; i++) {
                for (let j = i + 1; j < tickers.length; j++) {
                  s += hMat[i]?.[j] ?? matrix[i]?.[j] ?? 0;
                  c++;
                }
              }
              const avgH = c > 0 ? s / c : 0.5;

              return (
                <div
                  key={h.days}
                  className="p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-xl border border-zinc-200 dark:border-zinc-800 text-center"
                >
                  <span className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider block">
                    {h.label}
                  </span>
                  <span className="font-mono text-xl font-bold text-zinc-900 dark:text-zinc-100 my-1 block">
                    {avgH.toFixed(2)}
                  </span>
                  <span className="text-[10px] text-zinc-500 block">
                    {h.days === 1
                      ? 'Microstructure noise'
                      : h.days === 5
                      ? 'Portfolio cascade'
                      : 'Macro regime'}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="mt-4 p-3 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-gold-600 dark:text-gold-300 flex-shrink-0 mt-0.5" />
              <div>
                <strong>Horizon Effect Rationale: </strong>
                Short-horizon correlation (1-day) is commonly muted (~0.35) due to non-synchronous intraday order flow, whereas long-horizon correlation (22d, 66d) converges higher (~0.65–0.75) as common macroeconomic fundamentals dominate. Calibrating equity hedges at the wrong horizon creates substantial tracking error.
              </div>
            </div>
          </div>
        </div>

        {/* Random Matrix Theory (RMT) & Ledoit-Wolf Shrinkage */}
        <div className="panel p-5">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
            <div>
              <h4 className="font-semibold text-zinc-900 dark:text-zinc-100 text-sm">
                Random Matrix Theory (RMT) &amp; Covariance Shrinkage
              </h4>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                Marchenko-Pastur spectral noise filtering and Ledoit-Wolf regularization
              </p>
            </div>
            <span className="text-xs font-mono text-gold-600 dark:text-gold-400 font-semibold">
              Signal vs Noise
            </span>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs mb-3">
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Upper Noise Band (λ₊)</span>
              <span className="font-bold text-gold-500 text-sm">
                {rmtSpectrum.lambdaPlus.toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Market Mode (λ₁)</span>
              <span className="font-bold text-gold-500 text-sm">
                {(rmtSpectrum.eigenvalues[0] || 1).toFixed(2)}
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Shrinkage (δ*)</span>
              <span className="font-bold text-gold-600 dark:text-gold-300 text-sm">
                {(shrinkageIntensity * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-2.5 bg-zinc-50 dark:bg-zinc-900/60 rounded-lg border border-zinc-200 dark:border-zinc-800">
              <span className="text-[10px] text-zinc-400 block">Cond Num (Raw → Shrunk)</span>
              <span className="font-bold text-zinc-800 dark:text-zinc-200 text-xs">
                {conditionNumberRaw.toFixed(0)} → {conditionNumberShrunk.toFixed(0)}
              </span>
            </div>
          </div>

          {/* Eigenvalue Spectrum Bar */}
          <div className="space-y-1.5 mt-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-500">Eigenvalue Spectrum vs Marchenko-Pastur Boundary</span>
              <span className="font-mono text-gold-500 font-semibold">
                {rmtSpectrum.signalEigenvaluesCount} Signal Modes (λ &gt; λ₊)
              </span>
            </div>

            <div className="flex items-center gap-1.5 h-6 bg-zinc-100 dark:bg-zinc-900 rounded p-1">
              {rmtSpectrum.eigenvalues.slice(0, 8).map((val, idx) => {
                const isSignal = val > rmtSpectrum.lambdaPlus;
                const widthPct = Math.min(
                  40,
                  Math.max(6, (val / (rmtSpectrum.eigenvalues[0] || 1)) * 35)
                );
                return (
                  <div
                    key={idx}
                    style={{ width: `${widthPct}%` }}
                    className={`h-full rounded text-[10px] font-mono flex items-center justify-center font-bold transition-all ${
                      idx === 0
                        ? 'bg-gold-500 text-zinc-950'
                        : isSignal
                        ? 'bg-gold-500 text-zinc-950'
                        : 'bg-zinc-400 text-zinc-950 opacity-40'
                    }`}
                    title={`Eigenvalue λ_${idx + 1} = ${val.toFixed(2)} (${
                      isSignal ? 'Signal Mode' : 'Noise Bulk'
                    })`}
                  >
                    λ{idx + 1}
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono mt-1">
              <span>λ₁: Market Mode ({(rmtSpectrum.marketModeRatio * 100).toFixed(0)}% var)</span>
              <span>λ &lt; {rmtSpectrum.lambdaPlus.toFixed(2)}: Filtered Random Matrix Noise</span>
            </div>
          </div>

          <div className="mt-3 pt-2.5 border-t border-zinc-100 dark:border-zinc-800 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <ShrinkageEquation display="inline" className="text-xs" />
            </div>
            <span className="text-[11px] text-zinc-400">
              Guarantees positive definite covariance for portfolio optimization
            </span>
          </div>
        </div>
      </div>

      {/* LaTeX Table Export Modal */}
      {showLatexModal && (
        <div className="fixed inset-0 z-50 bg-zinc-950/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-[#101014] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 max-w-2xl w-full shadow-xl">
            <div className="flex items-center justify-between mb-3 pb-2 border-b border-zinc-100 dark:border-zinc-800">
              <h3 className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">
                LaTeX Correlation Table (Report Ready)
              </h3>
              <button
                onClick={() => setShowLatexModal(false)}
                className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 text-sm"
              >
                ✕
              </button>
            </div>
            <textarea
              readOnly
              value={latexCode}
              rows={12}
              className="w-full font-mono text-xs p-3 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 select-all"
            />
            <div className="flex items-center justify-between mt-4">
              <span className="text-xs text-zinc-400">
                Includes significance stars (* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001)
              </span>
              <button
                onClick={handleCopyLatex}
                className="inline-flex items-center gap-1.5 px-4 py-2 bg-gold-500 hover:bg-gold-400 text-zinc-950 font-semibold text-xs rounded-lg transition-all"
              >
                {copiedLatex ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                {copiedLatex ? 'Copied LaTeX Code!' : 'Copy to Clipboard'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
