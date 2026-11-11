'use client';

import React, { useMemo } from 'react';
import katex from 'katex';

export type FinancialFormulaPreset =
  | 'garch'
  | 'ewma'
  | 'har-rv'
  | 'realized-vol'
  | 'bipower-variation'
  | 'jump-component'
  | 'jump-z-test'
  | 'simple-return'
  | 'log-return'
  | 'annualized-vol'
  | 'skewness'
  | 'kurtosis'
  | 'adf'
  | 'decomposition-additive'
  | 'decomposition-multiplicative'
  | 'ols-model'
  | 'ols-matrix'
  | 'robust-irls'
  | 'vif'
  | 'expected-return'
  | 'correlation'
  | 'acf-squared'
  | 'long-run-vol'
  | 'har-rv-forecast'
  | 'har-rv-daily'
  | 'har-rv-weekly'
  | 'har-rv-monthly'
  | 'mincer-zarnowitz'
  | 'dcc-q'
  | 'dcc-correlation'
  | 'shrinkage'
  | 'fisher-z'
  | 'marchenko-pastur'
  | 'portfolio-variance'
  | 'risk-contribution'
  | 'diversification-ratio'
  | 'tail-dependence'
  | 'mantegna-distance'
  | 'partial-correlation'
  | 'student-t-pdf'
  | 'hansen-skew-t'
  | 'ged-pdf'
  | 'aic'
  | 'arch-lm'
  | 'ljung-box'
  | 'hill-estimator'
  | 'gpd-var'
  | 'cornish-fisher'
  | 'kupiec-pof'
  | 'christoffersen'
  | 'drawdown'
  | 'psr'
  | 'roll-spread'
  | 'amihud'
  | 'realized-semivariance'
  | 'signature-plot'
  | 'variance-ratio'
  | 'news-impact';

export type MathSymbolName =
  | 'sigma'
  | 'sigma_sq'
  | 'alpha'
  | 'beta'
  | 'omega'
  | 'lambda'
  | 'beta_0'
  | 'beta_d'
  | 'beta_w'
  | 'beta_m'
  | 'r_squared'
  | 'adj_r_squared'
  | 'rv'
  | 'rv_t'
  | 'rv_lag'
  | 'rv_w_t'
  | 'rv_w_lag'
  | 'rv_m_t'
  | 'rv_m_lag'
  | 'bv'
  | 'j_t'
  | 'alpha_plus_beta'
  | 'tau'
  | 'mu'
  | 'epsilon'
  | 'epsilon_t'
  | 'f_stat';

export const FINANCIAL_FORMULA_PRESETS: Record<FinancialFormulaPreset, { tex: string; label: string; description: string }> = {
  garch: {
    tex: String.raw`\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2`,
    label: 'GARCH(1,1) Variance Equation',
    description: 'Bollerslev (1986) Conditional Heteroskedasticity specification with autoregressive variance memory.',
  },
  ewma: {
    tex: String.raw`\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2`,
    label: 'RiskMetrics EWMA Equation',
    description: 'Exponentially Weighted Moving Average variance with constant decay factor lambda.',
  },
  'har-rv': {
    tex: String.raw`RV_{t+1} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \varepsilon_{t+1}`,
    label: 'Corsi (2009) HAR-RV Equation',
    description: 'Heterogeneous Autoregressive Realized Volatility cascade across daily, weekly, and monthly horizons.',
  },
  'realized-vol': {
    tex: String.raw`RV_t = \sum_{j=1}^M r_{t,j}^2`,
    label: 'Realized Volatility Summation',
    description: 'Sum of M intraday 1-minute squared returns estimating integrated variance without noise.',
  },
  'bipower-variation': {
    tex: String.raw`BV_t = \frac{\pi}{2} \left(\frac{M}{M-1}\right) \sum_{j=2}^M |r_{t,j}| |r_{t,j-1}|`,
    label: 'Barndorff-Nielsen Bipower Variation',
    description: 'Asymptotically isolates continuous Brownian diffusion variance from discrete jump shocks.',
  },
  'jump-component': {
    tex: String.raw`J_t = \max(0, RV_t - BV_t)`,
    label: 'Discrete Jump Variance Component',
    description: 'Separates abnormal event-driven discrete jumps from diffusive continuous market turbulence.',
  },
  'jump-z-test': {
    tex: String.raw`Z = \frac{\frac{RV_t - BV_t}{RV_t}}{\sqrt{\left(\left(\frac{\pi}{2}\right)^2 + \pi - 5\right) \frac{1}{M} \max\left(1, \frac{TP_t}{BV_t^2}\right)}}`,
    label: 'Barndorff-Nielsen Jump Z-Test',
    description: 'Asymptotic test statistic evaluating whether intraday price jumps are statistically significant (Z > 1.96).',
  },
  'simple-return': {
    tex: String.raw`R_t = \frac{P_t - P_{t-1}}{P_{t-1}}`,
    label: 'Simple Percentage Return',
    description: 'Discrete arithmetic percentage price change between consecutive trading bars.',
  },
  'log-return': {
    tex: String.raw`r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})`,
    label: 'Logarithmic (Continuous) Return',
    description: 'Continuously compounded log price increment, ensuring time-additive multi-period returns.',
  },
  'annualized-vol': {
    tex: String.raw`\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}`,
    label: 'Annualized Volatility Scaling',
    description: 'Standard square-root-of-time scaling mapping daily standard deviation to 252 annual trading days.',
  },
  skewness: {
    tex: String.raw`S = \frac{\frac{1}{N}\sum_{t=1}^N (r_t - \bar{r})^3}{\left[\frac{1}{N}\sum_{t=1}^N (r_t - \bar{r})^2\right]^{3/2}}`,
    label: 'Fisher-Pearson Sample Skewness',
    description: 'Measures third standardized return moment asymmetry (crash risk / negative skew).',
  },
  kurtosis: {
    tex: String.raw`K = \frac{\frac{1}{N}\sum_{t=1}^N (r_t - \bar{r})^4}{\left[\frac{1}{N}\sum_{t=1}^N (r_t - \bar{r})^2\right]^2} - 3`,
    label: 'Fisher Excess Kurtosis',
    description: 'Measures fourth standardized moment tail heaviness relative to normal distribution (K > 0 implies fat tails).',
  },
  adf: {
    tex: String.raw`\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^p \delta_j \Delta y_{t-j} + \varepsilon_t`,
    label: 'Augmented Dickey-Fuller (ADF) Regression',
    description: 'Evaluates unit-root non-stationarity under null hypothesis H0: gamma = 0 vs alternative H1: gamma < 0.',
  },
  'decomposition-additive': {
    tex: String.raw`Y_t = T_t + S_t + I_t`,
    label: 'Additive Classical Decomposition',
    description: 'Decomposes price series linearly into Trend (T), Seasonal factor (S), and Residual noise (I).',
  },
  'decomposition-multiplicative': {
    tex: String.raw`Y_t = T_t \times S_t \times I_t`,
    label: 'Multiplicative Classical Decomposition',
    description: 'Decomposes price series proportionally when seasonal swings scale with the trend level.',
  },
  'ols-model': {
    tex: String.raw`Y_t = \beta_0 + \sum_{k=1}^K \beta_k X_{k,t} + \varepsilon_t`,
    label: 'Multiple Factor Regression Model',
    description: 'Specifies linear factor dependencies between dependent asset return and market predictors.',
  },
  'ols-matrix': {
    tex: String.raw`\widehat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}`,
    label: 'Ordinary Least Squares Matrix Estimator',
    description: 'Normal equation analytical solution minimizing sum of squared residual errors.',
  },
  'robust-irls': {
    tex: String.raw`\widehat{\boldsymbol{\beta}}_{\text{robust}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{Y}`,
    label: 'Iteratively Reweighted Least Squares (IRLS)',
    description: 'Robust M-estimator applying Huber / Bisquare weighting to mitigate outlier return distortion.',
  },
  vif: {
    tex: String.raw`\text{VIF}_j = \frac{1}{1 - R_j^2}`,
    label: 'Variance Inflation Factor',
    description: 'Quantifies multicollinearity inflation on beta variance (VIF > 5 signals severe colinearity).',
  },
  'expected-return': {
    tex: String.raw`\mathbb{E}[R_{t+1}] = \widehat{\beta}_0 + \sum_{k=1}^K \widehat{\beta}_k X_{k,t}`,
    label: 'Conditional Next-Day Return Forecast',
    description: 'Point forecast conditioned on factor loadings and latest observation vector.',
  },
  correlation: {
    tex: String.raw`\rho_{X,Y} = \frac{\sum_{t=1}^N (X_t - \bar{X})(Y_t - \bar{Y})}{\sqrt{\sum_{t=1}^N (X_t - \bar{X})^2 \sum_{t=1}^N (Y_t - \bar{Y})^2}}`,
    label: 'Pearson Correlation Coefficient',
    description: 'Standardized co-movement measure bounded between -1.0 (perfect hedge) and +1.0 (perfect correlation).',
  },
  'acf-squared': {
    tex: String.raw`\rho_k(r^2) = \frac{\sum_{t=k+1}^N (r_t^2 - \bar{r^2})(r_{t-k}^2 - \bar{r^2})}{\sum_{t=1}^N (r_t^2 - \bar{r^2})^2}`,
    label: 'Autocorrelation of Squared Returns',
    description: 'Diagnoses second-moment memory and volatility clustering across lag horizons k.',
  },
  'long-run-vol': {
    tex: String.raw`V_L = \sqrt{\frac{\omega}{1 - (\alpha + \beta)} \times 252}`,
    label: 'Unconditional Long-Run Volatility',
    description: 'Ergodic mean-reverting annualized variance floor implied by stationary GARCH parameters.',
  },
  'mincer-zarnowitz': {
    tex: String.raw`RV_{t+1} = a + b \, \widehat{\sigma}_{t+1}^2 + \eta_{t+1}`,
    label: 'Mincer-Zarnowitz Out-of-Sample Regression',
    description: 'Empirical regression testing unbiasedness (a = 0, b = 1) and forecast explanatory power.',
  },
  'har-rv-forecast': {
    tex: String.raw`RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w RV_{w,t} + \beta_m RV_{m,t} + \varepsilon_{t+1}`,
    label: 'Corsi Forward 1-Step-Ahead Forecast',
    description: 'One-step-ahead forward volatility forecast conditioned on contemporaneous daily, weekly, and monthly components.',
  },
  'har-rv-daily': {
    tex: String.raw`RV_t = \sum_{j=1}^M r_{t,j}^2`,
    label: 'Daily Realized Volatility Component',
    description: 'Sum of M intraday 1-minute squared log returns for trading day t.',
  },
  'har-rv-weekly': {
    tex: String.raw`RV_{w,t} = \frac{1}{5} \sum_{i=0}^4 RV_{t-i}`,
    label: 'Weekly Realized Volatility Component',
    description: '5-day normalized average of daily realized volatilities representing intermediate institutional rebalancing.',
  },
  'har-rv-monthly': {
    tex: String.raw`RV_{m,t} = \frac{1}{22} \sum_{i=0}^{21} RV_{t-i}`,
    label: 'Monthly Realized Volatility Component',
    description: '22-day normalized average of daily realized volatilities representing longer-term asset allocation horizon.',
  },
  'dcc-q': {
    tex: String.raw`Q_t = (1 - a - b)\bar{Q} + a \, \boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}^T + b \, Q_{t-1}`,
    label: 'Engle (2002) DCC-GARCH Covariance Recursion',
    description: 'Dynamic quasi-correlation matrix equation driven by standardized residuals and unconditional correlation matrix Q_bar.',
  },
  'dcc-correlation': {
    tex: String.raw`R_t = \text{diag}(Q_t)^{-1/2} \, Q_t \, \text{diag}(Q_t)^{-1/2}`,
    label: 'DCC Correlation Matrix Normalization',
    description: 'Rescaling step transforming pseudo-covariance matrix Q_t into a proper correlation matrix R_t with unit diagonals.',
  },
  shrinkage: {
    tex: String.raw`\widehat{\boldsymbol{\Sigma}} = (1 - \delta)\mathbf{S} + \delta \mathbf{F}`,
    label: 'Ledoit-Wolf (2004) Analytical Shrinkage',
    description: 'Optimal convex linear combination of sample covariance S and structured constant-correlation target F.',
  },
  'fisher-z': {
    tex: String.raw`z = \frac{1}{2}\ln\left(\frac{1 + r}{1 - r}\right) \times \sqrt{\frac{n - 3}{2}}`,
    label: 'Fisher z-Transformation & Significance',
    description: 'Variance-stabilizing transformation converting sample correlation r into asymptotically standard normal z-scores.',
  },
  'marchenko-pastur': {
    tex: String.raw`\lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{N}{T}}\right)^2`,
    label: 'Random Matrix Theory (Marchenko-Pastur Law)',
    description: 'Defines the upper and lower spectral boundaries of eigenvalues expected from pure random noise matrices.',
  },
  'portfolio-variance': {
    tex: String.raw`\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}`,
    label: 'Markowitz Portfolio Variance',
    description: 'Quadratic form combining portfolio asset weights and annualized asset covariance matrix.',
  },
  'risk-contribution': {
    tex: String.raw`\text{CCTR}_i = w_i \frac{(\boldsymbol{\Sigma}\mathbf{w})_i}{\sigma_p}`,
    label: 'Euler Component Contribution to Risk (CCTR)',
    description: 'Euler decomposition attributing total portfolio volatility to individual asset positions, where sum of CCTR equals sigma_p.',
  },
  'diversification-ratio': {
    tex: String.raw`DR = \frac{\sum_{i=1}^N w_i \sigma_i}{\sigma_p}`,
    label: 'Choueifaty Diversification Ratio (DR)',
    description: 'Ratio of weighted asset volatilities to total portfolio volatility, measuring volatility reduction via correlation offsets.',
  },
  'tail-dependence': {
    tex: String.raw`\lambda_L = \lim_{u \to 0^+} P\left(U < u \;\middle|\; V < u\right)`,
    label: 'Lower Tail Dependence Coefficient',
    description: 'Quantifies joint probability of extreme simultaneous market crashes (strictly positive for fat-tailed empirical copulas).',
  },
  'mantegna-distance': {
    tex: String.raw`d_{ij} = \sqrt{2(1 - \rho_{ij})}`,
    label: 'Mantegna Ultrametric Correlation Distance',
    description: 'Maps bounded Pearson correlations into Euclidean metric distances for Minimum Spanning Tree (MST) network topology.',
  },
  'partial-correlation': {
    tex: String.raw`\rho_{ij|k} = -\frac{\Omega_{ij}}{\sqrt{\Omega_{ii} \Omega_{jj}}}`,
    label: 'Partial Correlation via Precision Matrix',
    description: 'Isolates direct pairwise co-movement by conditioning out common market factors via inverted covariance matrix Omega = Sigma^-1.',
  },
  'student-t-pdf': {
    tex: String.raw`f(x; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sigma \sqrt{\pi(\nu-2)}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{(x-\mu)^2}{(\nu-2)\sigma^2}\right)^{-\frac{\nu+1}{2}}`,
    label: 'Standardized Student-t Distribution',
    description: 'Fat-tailed leptokurtic distribution with degrees of freedom nu > 2 controlling heavy tail decay.',
  },
  'hansen-skew-t': {
    tex: String.raw`f(z; \nu, \lambda) = b c \left(1 + \frac{1}{\nu-2}\left(\frac{b z + a}{1 \pm \lambda}\right)^2\right)^{-\frac{\nu+1}{2}}`,
    label: 'Hansen (1994) Skewed Student-t Distribution',
    description: 'Asymmetric fat-tailed density allowing simultaneous modeling of tail heaviness nu and crash skewness lambda.',
  },
  'ged-pdf': {
    tex: String.raw`f(x; \kappa) = \frac{\kappa}{2 \lambda_\kappa \sigma\,\Gamma(1/\kappa)} \exp\left(-\left|\frac{x-\mu}{\lambda_\kappa \sigma}\right|^\kappa\right)`,
    label: 'Generalized Error Distribution (GED / Subbotin)',
    description: 'Exponential power family with shape parameter kappa (kappa = 2 Normal, kappa = 1 Laplace, kappa < 2 Leptokurtic).',
  },
  'aic': {
    tex: String.raw`\text{AIC} = 2k - 2\ln L \quad (\Delta\text{AIC} = \text{AIC}_m - \min \text{AIC})`,
    label: 'Akaike Information Criterion (Model Selection)',
    description: 'Penalized log-likelihood criterion ranking model goodness-of-fit against parsimony.',
  },
  'arch-lm': {
    tex: String.raw`\text{LM} = T \cdot R^2 \sim \chi^2(q)`,
    label: 'Engle (1982) ARCH-LM Test for Conditional Heteroskedasticity',
    description: 'Lagrange Multiplier test testing null of no ARCH effects in squared residuals against q lagged squared shocks.',
  },
  'ljung-box': {
    tex: String.raw`Q(h) = T(T+2) \sum_{k=1}^h \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(h)`,
    label: 'Ljung-Box Portmanteau Test',
    description: 'Tests whether a group of autocorrelations is significantly different from zero for returns or squared returns.',
  },
  'hill-estimator': {
    tex: String.raw`\hat{\xi} = \frac{1}{k}\sum_{i=1}^k \ln\left(\frac{x_{(n-i+1)}}{u}\right)`,
    label: 'Hill (1975) Tail Index Estimator',
    description: 'Extreme Value Theory semi-parametric estimator for the tail index xi of heavy-tailed Fréchet-type Pareto distributions.',
  },
  'gpd-var': {
    tex: String.raw`\text{VaR}_\alpha = u + \frac{\beta}{\xi}\left[\left(\frac{N}{N_u}(1-\alpha)\right)^{-\xi} - 1\right]`,
    label: 'Peaks-Over-Threshold (POT) Generalized Pareto VaR',
    description: 'Pickands-Balkema-de Haan asymptotic extreme quantile estimator exceeding threshold u.',
  },
  'cornish-fisher': {
    tex: String.raw`z_\alpha^{\text{CF}} = z_\alpha + \frac{S}{6}(z_\alpha^2-1) + \frac{K}{24}(z_\alpha^3-3z_\alpha) - \frac{S^2}{36}(2z_\alpha^3-5z_\alpha)`,
    label: 'Cornish-Fisher Polynomial Quantile Expansion',
    description: 'Adjusts Gaussian critical values for higher-order skewness S and excess kurtosis K.',
  },
  'kupiec-pof': {
    tex: String.raw`LR_{\text{POF}} = -2 \ln \left[ \frac{(1-p)^{T-x} p^x}{(1 - \hat{p})^{T-x} \hat{p}^x} \right] \sim \chi^2(1)`,
    label: 'Kupiec (1995) Proportion of Failures (POF) Likelihood Ratio',
    description: 'Unconditional coverage test evaluating whether VaR failure frequency x/T matches expected failure probability p.',
  },
  'christoffersen': {
    tex: String.raw`LR_{\text{ind}} = -2 \ln \left[ \frac{L(\Pi_2)}{L(\boldsymbol{\Pi}_1)} \right] \sim \chi^2(1), \quad LR_{\text{cc}} = LR_{\text{POF}} + LR_{\text{ind}} \sim \chi^2(2)`,
    label: 'Christoffersen (1998) Independence & Conditional Coverage',
    description: 'Markov-chain test testing whether VaR exceedances cluster or occur independently over consecutive trading days.',
  },
  'drawdown': {
    tex: String.raw`DD_t = \frac{P_t - \max_{s \le t} P_s}{\max_{s \le t} P_s}, \quad \text{MDD} = \max_t (-DD_t)`,
    label: 'Peak-to-Trough Drawdown & Maximum Drawdown (MDD)',
    description: 'Measures continuous percentage decline from all-time running peak and worst historical peak-to-trough drop.',
  },
  'psr': {
    tex: String.raw`\text{PSR}(\text{SR}^*) = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{\text{SR}}^2}} \right)`,
    label: 'Bailey-López de Prado (2012) Probabilistic Sharpe Ratio',
    description: 'Adjusts Sharpe Ratio for non-normality (skewness gamma_3, kurtosis gamma_4) and sample length T.',
  },
  'roll-spread': {
    tex: String.raw`s_{\text{Roll}} = 2\sqrt{\max\left(0, -\text{Cov}(\Delta p_t, \Delta p_{t-1})\right)}`,
    label: 'Roll (1984) Effective Bid-Ask Spread Estimator',
    description: 'Infers effective trading spread from serial negative covariance in tick/bar price increments caused by order-flow bounce.',
  },
  'amihud': {
    tex: String.raw`\text{ILLIQ}_t = \frac{1}{D_t}\sum_{d=1}^{D_t} \frac{|R_{t,d}|}{\text{PVol}_{t,d}} \times 10^6`,
    label: 'Amihud (2002) Illiquidity Ratio',
    description: 'Average daily price response per dollar of trading volume, measuring market price impact elasticity.',
  },
  'realized-semivariance': {
    tex: String.raw`RS_t^- = \sum_{j=1}^M r_{t,j}^2 \mathbb{I}_{\{r_{t,j} < 0\}}, \quad RV_t = RS_t^- + RS_t^+`,
    label: 'Realized Semivariance Decomposition (Barndorff-Nielsen et al., 2010)',
    description: 'Separates continuous realized volatility into downside variance RS^- and upside variance RS^+.',
  },
  'signature-plot': {
    tex: String.raw`RV^{(\Delta\tau)}_t = \sum_{k=1}^{\lfloor M / \Delta\tau \rfloor} \left( \ln P_{k\Delta\tau} - \ln P_{(k-1)\Delta\tau} \right)^2`,
    label: 'Andersen et al. (2000) Volatility Signature Plot',
    description: 'Plots realized variance as a function of sampling frequency Delta tau to isolate microstructure noise bias from true diffusion.',
  },
  'variance-ratio': {
    tex: String.raw`VR(q) = \frac{\sigma^2(q)}{q \cdot \sigma^2(1)} = 1 + 2\sum_{k=1}^{q-1}\left(1 - \frac{k}{q}\right)\rho_k`,
    label: 'Lo-MacKinlay (1988) Variance Ratio Test',
    description: 'Tests random walk hypothesis: VR = 1 indicates pure Brownian motion, VR < 1 mean reversion, VR > 1 trending momentum.',
  },
  'news-impact': {
    tex: String.raw`\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{I}_{\{\varepsilon_{t-1} < 0\}})\varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2`,
    label: 'Glosten-Jagannathan-Runkle (GJR-GARCH) News Impact',
    description: 'Asymmetric volatility response where negative returns (bad news) generate gamma extra variance over positive shocks.',
  },
};

export const MATH_SYMBOLS: Record<MathSymbolName, string> = {
  sigma: String.raw`\sigma`,
  sigma_sq: String.raw`\sigma_t^2`,
  alpha: String.raw`\alpha`,
  beta: String.raw`\beta`,
  omega: String.raw`\omega`,
  lambda: String.raw`\lambda`,
  beta_0: String.raw`\beta_0`,
  beta_d: String.raw`\beta_d`,
  beta_w: String.raw`\beta_w`,
  beta_m: String.raw`\beta_m`,
  r_squared: String.raw`R^2`,
  adj_r_squared: String.raw`R_{\text{adj}}^2`,
  rv: String.raw`RV_t`,
  rv_t: String.raw`RV_t`,
  rv_lag: String.raw`RV_{t-1}`,
  rv_w_t: String.raw`RV_{w,t}`,
  rv_w_lag: String.raw`RV_{w,t-1}`,
  rv_m_t: String.raw`RV_{m,t}`,
  rv_m_lag: String.raw`RV_{m,t-1}`,
  bv: String.raw`BV_t`,
  j_t: String.raw`J_t`,
  alpha_plus_beta: String.raw`\alpha + \beta`,
  tau: String.raw`\tau`,
  mu: String.raw`\mu`,
  epsilon: String.raw`\varepsilon`,
  epsilon_t: String.raw`\varepsilon_t`,
  f_stat: String.raw`F`,
};

export interface MathDisplayProps {
  /** LaTeX formula string or MathML snippet */
  formula?: string;
  /** Pre-configured quantitative finance formula preset */
  preset?: FinancialFormulaPreset;
  /** Convenience shortcut for mathematical single symbols */
  symbol?: MathSymbolName;
  /** Display mode: 'inline' for running text or 'block' for standalone equation card */
  display?: 'inline' | 'block';
  /** KaTeX output mode: 'htmlAndMathml' (default) or pure 'mathml' */
  outputMode?: 'htmlAndMathml' | 'mathml';
  /** Optional custom CSS classes for typography and coloring */
  className?: string;
  /** Whether to show a subtle MathML badge indicator */
  showBadge?: boolean;
}

/**
 * Reusable MathDisplay Component
 * Uses KaTeX to render financial formulas and econometric equations
 * with standard W3C MathML syntax, formatted to match the terminal's professional UI.
 */
export function MathDisplay({
  formula,
  preset,
  symbol,
  display = 'inline',
  outputMode = 'htmlAndMathml',
  className = '',
  showBadge = false,
}: MathDisplayProps) {
  // Determine LaTeX source string
  const rawTex = useMemo(() => {
    if (formula) return formula;
    if (preset && FINANCIAL_FORMULA_PRESETS[preset]) {
      return FINANCIAL_FORMULA_PRESETS[preset].tex;
    }
    if (symbol && MATH_SYMBOLS[symbol]) {
      return MATH_SYMBOLS[symbol];
    }
    return '';
  }, [formula, preset, symbol]);

  // Generate rendered HTML containing MathML syntax via KaTeX
  const htmlContent = useMemo(() => {
    if (!rawTex) return '';
    try {
      const renderFn =
        typeof katex?.renderToString === 'function'
          ? katex.renderToString
          : (katex as unknown as { default?: typeof katex })?.default?.renderToString;

      if (typeof renderFn !== 'function') {
        return `<code class="font-mono text-xs text-gold-500">${rawTex}</code>`;
      }

      return renderFn(rawTex, {
        displayMode: display === 'block',
        output: outputMode,
        throwOnError: false,
        strict: false,
      });
    } catch (err) {
      console.warn('MathDisplay KaTeX render warning:', err);
      return `<code class="font-mono text-xs text-gold-500">${rawTex}</code>`;
    }
  }, [rawTex, display, outputMode]);

  if (!rawTex) return null;

  if (display === 'block') {
    return (
      <div className={`my-2 overflow-x-auto py-1 ${className}`}>
        <div className="flex items-center justify-between gap-2">
          <div
            className="text-zinc-800 dark:text-zinc-200 text-sm leading-relaxed"
            dangerouslySetInnerHTML={{ __html: htmlContent }}
          />
          {showBadge && (
            <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400 dark:text-zinc-500 select-none">
              MathML
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <span
      className={`inline-flex items-center text-zinc-800 dark:text-zinc-200 align-baseline ${className}`}
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
}
