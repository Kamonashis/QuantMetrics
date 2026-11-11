'use client';

import React from 'react';
import {
  MathDisplay,
  MathDisplayProps,
  FinancialFormulaPreset,
  MathSymbolName,
  FINANCIAL_FORMULA_PRESETS,
  MATH_SYMBOLS,
} from '@/components/ui/MathDisplay';

export {
  MathDisplay,
  FINANCIAL_FORMULA_PRESETS,
  MATH_SYMBOLS,
};
export type {
  MathDisplayProps,
  FinancialFormulaPreset,
  MathSymbolName,
};

interface MathComponentProps {
  className?: string;
  display?: 'inline' | 'block';
  outputMode?: 'htmlAndMathml' | 'mathml';
  showBadge?: boolean;
}

/**
 * GARCH(1,1) Conditional Variance Equation
 * \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
 */
export function GarchEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
  showBadge = false,
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="garch"
      display={display}
      outputMode={outputMode}
      className={className}
      showBadge={showBadge}
    />
  );
}

/**
 * EWMA (RiskMetrics) Variance Equation
 * \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2
 */
export function EwmaEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
  showBadge = false,
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="ewma"
      display={display}
      outputMode={outputMode}
      className={className}
      showBadge={showBadge}
    />
  );
}

/**
 * Corsi (2009) HAR-RV Equation (Standard In-Sample Form)
 * RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{w,t-1} + \beta_m RV_{m,t-1} + \varepsilon_t
 */
export function HarRvEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
  showBadge = false,
}: MathComponentProps) {
  return (
    <MathDisplay
      formula={String.raw`RV_t = \beta_0 + \beta_d RV_{t-1} + \beta_w RV_{w,t-1} + \beta_m RV_{m,t-1} + \varepsilon_t`}
      display={display}
      outputMode={outputMode}
      className={className}
      showBadge={showBadge}
    />
  );
}

/**
 * Corsi (2009) Forward HAR-RV Forecast Equation
 * RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w RV_{w,t} + \beta_m RV_{m,t} + \varepsilon_{t+1}
 */
export function HarRvForecastEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
  showBadge = false,
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="har-rv-forecast"
      display={display}
      outputMode={outputMode}
      className={className}
      showBadge={showBadge}
    />
  );
}

/**
 * Daily Realized Volatility Component Equation (RV_t)
 * RV_t = \sum_{j=1}^M r_{t,j}^2
 */
export function HarRvDailyEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="har-rv-daily"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Weekly Realized Volatility Component Equation (RV_{w,t})
 * RV_{w,t} = \frac{1}{5} \sum_{i=0}^4 RV_{t-i}
 */
export function HarRvWeeklyEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="har-rv-weekly"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Monthly Realized Volatility Component Equation (RV_{m,t})
 * RV_{m,t} = \frac{1}{22} \sum_{i=0}^{21} RV_{t-i}
 */
export function HarRvMonthlyEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="har-rv-monthly"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Realized Volatility Definition
 * RV_t = \sum_{j=1}^M r_{t,j}^2
 */
export function RealizedVolEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="realized-vol"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Bipower Variation (Barndorff-Nielsen & Shephard)
 * BV_t = \frac{\pi}{2} \sum_{j=2}^M |r_{t,j}| |r_{t,j-1}|
 */
export function BipowerVariationEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="bipower-variation"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Jump Shock Component
 * J_t = \max(0, RV_t - BV_t)
 */
export function JumpComponentEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="jump-component"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Simple Return Equation
 * R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
 */
export function SimpleReturnEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="simple-return"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Logarithmic (Continuously Compounded) Return Equation
 * r_t = \ln(P_t / P_{t-1})
 */
export function LogReturnEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="log-return"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Rolling Annualized Volatility Equation
 * \sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}
 */
export function AnnualizedVolEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="annualized-vol"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Sample Skewness Equation
 */
export function SkewnessEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="skewness"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Excess Kurtosis Equation
 */
export function KurtosisEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="kurtosis"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Augmented Dickey-Fuller (ADF) Regression Equation
 */
export function AdfEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="adf"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Additive Time Series Decomposition Equation
 */
export function DecompositionAdditiveEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="decomposition-additive"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Multiplicative Time Series Decomposition Equation
 */
export function DecompositionMultiplicativeEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="decomposition-multiplicative"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Multiple Linear Regression Model Equation
 */
export function RegressionModelEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="ols-model"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * OLS Normal Equation (Matrix Estimator)
 */
export function OlsMatrixEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="ols-matrix"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Robust IRLS Matrix Estimator
 */
export function RobustIrlsEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="robust-irls"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Variance Inflation Factor (VIF)
 */
export function VifEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="vif"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Expected Next-Day Return Forecast
 */
export function ExpectedReturnEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="expected-return"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Pearson Correlation Coefficient Equation
 */
export function CorrelationEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="correlation"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Autocorrelation of Squared Returns (Volatility Clustering)
 */
export function AcfSquaredReturnsEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="acf-squared"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Long-run Unconditional GARCH Volatility
 */
export function LongRunVolEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="long-run-vol"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Quick mathematical symbol helper using MathDisplay
 */
export function MathSymbol({
  name,
  className = '',
  outputMode = 'htmlAndMathml',
}: {
  name: MathSymbolName;
  className?: string;
  outputMode?: 'htmlAndMathml' | 'mathml';
}) {
  return (
    <MathDisplay
      symbol={name}
      display="inline"
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Engle DCC-GARCH Covariance Recursion Q_t
 */
export function DccQEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="dcc-q"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * DCC Correlation Matrix Normalization R_t
 */
export function DccCorrelationEquation({
  className = '',
  display = 'block',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="dcc-correlation"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Ledoit-Wolf Analytical Shrinkage Covariance
 */
export function ShrinkageEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="shrinkage"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Fisher z-Transformation & Significance
 */
export function FisherZEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="fisher-z"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Marchenko-Pastur Law (RMT Noise Bound)
 */
export function MarchenkoPasturEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="marchenko-pastur"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Markowitz Portfolio Variance
 */
export function PortfolioVarianceEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="portfolio-variance"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Euler Component Contribution to Risk (CCTR)
 */
export function RiskContributionEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="risk-contribution"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Choueifaty Diversification Ratio (DR)
 */
export function DiversificationRatioEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="diversification-ratio"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Lower Tail Dependence Coefficient
 */
export function TailDependenceEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="tail-dependence"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Mantegna Ultrametric Distance
 */
export function MantegnaDistanceEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="mantegna-distance"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Partial Correlation Precision Matrix
 */
export function PartialCorrelationEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="partial-correlation"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Standardized Student-t Distribution Equation
 */
export function StudentTEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="student-t-pdf"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Hansen (1994) Skewed Student-t Distribution Equation
 */
export function HansenSkewTEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="hansen-skew-t"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Generalized Error Distribution (GED) Equation
 */
export function GedEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="ged-pdf"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Akaike Information Criterion Equation
 */
export function AicEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="aic"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Engle (1982) ARCH-LM Test Equation
 */
export function ArchLmEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="arch-lm"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Ljung-Box Portmanteau Test Equation
 */
export function LjungBoxEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="ljung-box"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Hill (1975) Tail Index Estimator Equation
 */
export function HillEstimatorEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="hill-estimator"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Peaks-Over-Threshold (POT) Generalized Pareto VaR Equation
 */
export function GpdVarEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="gpd-var"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Cornish-Fisher Polynomial Quantile Expansion Equation
 */
export function CornishFisherEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="cornish-fisher"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Kupiec (1995) Proportion of Failures Equation
 */
export function KupiecPofEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="kupiec-pof"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Christoffersen (1998) Independence Test Equation
 */
export function ChristoffersenEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="christoffersen"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Peak-to-Trough Drawdown & MDD Equation
 */
export function DrawdownEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="drawdown"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Bailey-López de Prado Probabilistic Sharpe Ratio Equation
 */
export function PsrEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="psr"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Roll Effective Bid-Ask Spread Equation
 */
export function RollSpreadEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="roll-spread"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Amihud (2002) Illiquidity Ratio Equation
 */
export function AmihudEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="amihud"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Realized Semivariance Decomposition Equation
 */
export function RealizedSemivarianceEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="realized-semivariance"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Volatility Signature Plot Equation
 */
export function SignaturePlotEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="signature-plot"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * Lo-MacKinlay Variance Ratio Test Equation
 */
export function VarianceRatioEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="variance-ratio"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

/**
 * GJR-GARCH News Impact Curve Equation
 */
export function NewsImpactEquation({
  className = '',
  display = 'inline',
  outputMode = 'htmlAndMathml',
}: MathComponentProps) {
  return (
    <MathDisplay
      preset="news-impact"
      display={display}
      outputMode={outputMode}
      className={className}
    />
  );
}

