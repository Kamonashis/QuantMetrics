# QuantMetrics: Stock Analysis & Volatility Modeling Terminal

QuantMetrics is an advanced quantitative stock analysis, econometric diagnostics, and volatility modeling terminal built for quantitative researchers, risk managers, and financial analysts. It combines institutional-grade statistical models with interactive visualizations, real-time market data ingestion, and rigorous risk backtesting.

---

## 🚀 Key Modules & Capabilities

### 1. Advanced Econometric & Stationarity Battery
- **Stationarity Suite**: Augmented Dickey-Fuller (ADF), Phillips-Perron (PP), and KPSS tests evaluated across price levels $P_t$, log returns $r_t$, and squared returns $r_t^2$.
- **Heteroskedasticity & Autocorrelation**: Engle's ARCH-LM test and Ljung-Box Q-statistic across multiple lag horizons.
- **Structural Break Detection**: Cumulative Sum of Squares (CUSUM-SQ) test with empirical 95% confidence bands to identify parameter instability.
- **Long Memory & Efficiency**: Hurst Exponent calculation ($H > 0.5$ trending, $H < 0.5$ mean-reverting) and Lo's overlapping Variance Ratio test.
- **Market Microstructure**: Roll (1984) effective spread, Corwin-Schultz (2012) high-low bid-ask spread estimator, Amihud illiquidity ratio, and Realized Semivariance / Quarticity.

### 2. Volatility & High-Frequency Modeling
- **GARCH(1,1)**: Maximum Likelihood Estimation with analytical multistep forecasts and forecast confidence envelopes:
  $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
- **Asymmetric Volatility**: News Impact Curve and GJR-GARCH leverage diagnostics ($\epsilon_{t-1} < 0$).
- **RiskMetrics EWMA**: Exponentially Weighted Moving Average ($\lambda = 0.94$) volatility tracking.
- **HAR-RV (Heterogeneous Autoregressive Realized Volatility)**: Cascade model using daily, weekly (5-day), and monthly (22-day) realized volatility components:
  $$RV_{t+1d} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \epsilon_{t+1}$$
- **Jump Decomposition**: Barndorff-Nielsen & Shephard Bipower Variation (BV) separating continuous diffusion from discrete jumps ($J_t$).
- **Intraday Diurnal U-Shape**: Minute-level diurnal seasonality tracking and Two-Scale Realized Volatility (TSRV) noise-robust estimation.

### 3. Return Distribution Fitting & Tail Risk
- **Parametric Maximum Likelihood Fitting**:
  - **Normal Distribution**: Gaussian baseline.
  - **Student-t Distribution**: Symmetrical heavy-tail modeling with degrees of freedom parameter $\nu$.
  - **Hansen (1994) Skewed Student-t**: Joint estimation of tail fatness ($\nu$) and directional skewness ($\lambda$).
  - **Generalized Error Distribution (GED / Subbotin)**: Tail flexibility with shape parameter $\kappa$.
- **Model Selection**: Comparative scoring via Log-Likelihood, Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), $\Delta\text{AIC}$, and Kolmogorov-Smirnov (KS) goodness-of-fit statistics.

### 4. Multivariate Correlation & Risk Suite
- **Pairwise Correlation Metrics**: Pearson linear, Spearman rank, and Kendall's tau correlation matrices.
- **Multiple Hypothesis Control**: Fisher $z$-transform $p$-values with Benjamini-Hochberg False Discovery Rate (FDR) adjustment.
- **Ledoit-Wolf Shrinkage**: Optimal linear shrinkage towards constant correlation target to condition ill-behaved covariance matrices.
- **Random Matrix Theory (RMT)**: Marchenko-Pastur spectral density clipping to denoise eigenvalues from sample noise:
  $$\lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{N/T}\right)^2$$
- **Network Topology**: Minimum Spanning Tree (MST) asset clustering using distance metric $d_{ij} = \sqrt{2(1 - \rho_{ij})}$.
- **Portfolio Construction**: Global Minimum Variance (GMV) optimization and Euler marginal risk contribution decomposition.

### 5. Risk Diagnostics & Regulatory Backtesting
- **5-Model VaR & Expected Shortfall (ES / CVaR)** at 95% and 99% confidence:
  1. Historical Simulation
  2. Parametric Gaussian
  3. Cornish-Fisher Expansion (adjusting for skewness and excess kurtosis)
  4. Student-t Distribution
  5. Extreme Value Theory (EVT: Hill Estimator & Peaks-Over-Threshold GPD)
- **Regulatory Backtesting**:
  - **Kupiec POF Test**: Binomial likelihood ratio test for unconditional coverage frequency.
  - **Christoffersen Independence Test**: First-order Markov test evaluating violation clustering and serial independence.
- **Performance Evaluation**: Underwater drawdown profiles, Pain Index, and Bailey & López de Prado's Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

### 6. Regression & Factor Diagnostics
- Ordinary Least Squares (OLS) single-index CAPM model against benchmark index (SPY).
- Parameter estimates: Jensen's Alpha ($\alpha$), Beta ($\beta$), $t$-statistics, $p$-values, $R^2$, and Adjusted $R^2$.
- Durbin-Watson residual autocorrelation statistic and residual Q-Q diagnostic plots.

---

## 🛠️ Tech Stack

- **Framework**: [Next.js 15](https://nextjs.org/) (App Router, Standalone build)
- **UI Library**: [React 19](https://react.dev/)
- **Language**: [TypeScript 5.9](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Animations**: [Motion](https://motion.dev/) (Framer Motion v12)
- **Mathematics & Typography**: [KaTeX](https://katex.org/) for LaTeX rendering
- **Icons**: [Lucide React](https://lucide.dev/)

---

## 📁 Project Structure

```plaintext
QuantMetrics/
├── app/
│   ├── api/
│   │   └── market-data/route.ts    # Market data fetching & historical caching API
│   ├── globals.css                 # Global styling & Tailwind directives
│   ├── layout.tsx                  # Root layout & theme wrapper
│   └── page.tsx                    # Main workstation view coordinator
├── components/
│   ├── charts/                     # Custom Canvas & SVG analytical charts
│   │   ├── AcfPacfChart.tsx
│   │   ├── CorrelationHeatmap.tsx
│   │   ├── DecompositionChart.tsx
│   │   ├── DistributionFittingPanel.tsx
│   │   ├── HarRvForecastChart.tsx
│   │   ├── IntradayDiurnalChart.tsx
│   │   ├── JumpDecompositionChart.tsx
│   │   ├── ReturnDistributionChart.tsx
│   │   ├── ReturnsChart.tsx
│   │   ├── RollingVolatilityChart.tsx
│   │   ├── TimeSeriesChart.tsx
│   │   └── VolatilityForecastChart.tsx
│   ├── layout/
│   │   └── Navbar.tsx              # Top navigation & ticker search
│   ├── math/
│   │   └── MathEquations.tsx       # Formula reference sheets (KaTeX)
│   ├── ui/                         # Reusable UI primitives
│   └── views/                      # Workstation dashboards
│       ├── HomeView.tsx            # Terminal landing & market pulse
│       ├── AnalysisView.tsx        # Econometrics & microstructure tabs
│       ├── ModelingView.tsx        # GARCH, EWMA & HAR-RV volatility studio
│       ├── CorrelationView.tsx     # Matrix, RMT, MST & portfolio risk tabs
│       └── RegressionView.tsx      # CAPM & factor diagnostics
├── lib/
│   ├── analysis-engine.ts          # Stationarity, EVT, VaR/ES & microstructure math
│   ├── correlation-engine.ts       # RMT, Ledoit-Wolf, MST & GMV algorithms
│   ├── distribution-engine.ts      # MLE for Normal, Student-t, Skewed-t, GED
│   ├── market-service.ts           # Synthetic generator & live data adapter
│   ├── quant-engine.ts             # GARCH, HAR-RV, returns & base statistics
│   └── utils.ts                    # Class name utilities
├── next.config.ts                  # Next.js build configuration
├── package.json                    # Project dependencies & scripts
└── tsconfig.json                   # TypeScript compiler options
```

---

## ⚡ Getting Started

### Prerequisites
- **Node.js**: `v20.0.0` or later (tested on `v22.22.1`)
- **Package Manager**: `npm`, `pnpm`, or `bun`

### Installation

1. Clone or navigate to the repository directory:
   ```bash
   cd QuantMetrics
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. (Optional) Configure environment variables:
   ```bash
   cp .env.example .env.local
   ```
   Add any optional API keys (such as `GEMINI_API_KEY` for AI features) in `.env.local`.

---

## 🖥️ Running the Application

### Development Server
Start the development server with Hot Module Replacement (HMR):
```bash
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** (or port `3001` if `3000` is occupied) in your browser.

### Production Build
Create an optimized production bundle:
```bash
npm run build
npm run start
```

### Linting & Type Checking
Verify codebase consistency and type safety:
```bash
npm run lint
npx tsc --noEmit
```

---

## 📜 Available Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Runs the Next.js development server |
| `npm run build` | Compiles and optimizes the app for production |
| `npm run start` | Serves the production build |
| `npm run lint` | Runs ESLint to check for code issues |
| `npm run clean` | Cleans Next.js build cache |

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
