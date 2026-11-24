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
- **Asymmetric Volatility**: News Impact Curve and GJR-GARCH leverage diagnostics ($\epsilon_{t-1} < 0$) — now fitted twice on the Analysis → Dynamics tab: close→close (overnight included, green) and open→close (gap removed, dashed gold), so the leverage effect is *tested* rather than assumed: γ surviving the gap means continuous intraday flow asymmetry, γ collapsing means an event/gap phenomenon; Δγ is printed on the legend. The diurnal U-shape panel carries a diurnal÷RV reconciliation badge (synthetic-data or noise-dominated panels say so instead of sizing silently).
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

### 6. Factor Regression Lab
- **Explain returns**: regress any asset's daily returns on a benchmark plus optional style-factor proxies (Size IWM−SPY, Value IWD−IWF, Momentum MTUM−SPY, Quality QUAL−SPY, Low-vol USMV−SPY, Nasdaq tilt QQQ−SPY) and up to four extra tickers. Results lead with plain-language readouts, exposure intervals, variance shares (sum to R²), and an exact return attribution over time.
- **Test stability**: rolling 3M/6M/1Y exposures with 95% bands, a same-sample market-only comparison (HAC Wald + BIC), factor correlation and VIF overlap, Cook's-distance influential dates, a Huber robustness check, and Ljung–Box / Jarque–Bera residual checks.
- **Predict (experimental)**: the TCN-Attention sequence model on lagged factor returns, scored out of sample against lag-matched OLS. It is kept apart from the descriptive fits.
- **Cross-section screen**: factor rankings, IC, and composite long–short (the existing cross-sectional panel).
- **Methods & data**: exact specification, per-series audit (bars, price basis, live vs. synthetic), and limits.
- Engine: `lib/factor-regression.ts`. Returns are computed on the intersection of price dates, never by array position. OLS uses QR with collinearity refusal, Newey–West HAC inference, and exact t/F/χ² p-values. Returns are raw (not excess), so alpha is not Jensen's alpha.

---

### 7. Implied Volatility & Risk-Neutral Distribution Terminal
- **T1 — State Dashboard**: IV-30 (interpolated on total variance), IV rank with honest accrual state, IV percentile vs 1y, ^VIX/^VXN/^VVIX/^SKEW context, 30d/90d slope, ^VIX9D/3M inversion verdict (`CONTANGO / FLAT / INVERTED`), 25Δ risk reversal & butterfly, 90/10 skew, surface quality (strikes used / spread), and the IV−RV gap in plain language.
- **T2 — Surface & Smile**: IV heatmap (moneyness × expiry), SVI-fitted smiles with per-slice RMSE and residuals visible, 25Δ RR term structure, expected-move ladder (±σ√T by tenor).
- **T3 — Term Structure & VRP**: total-variance (w = σ²T) interpolation, ^VIX 1y percentile bands, IV vs RV dual-axis with IV<RV regime shading, VRP by tenor vs RV20 and GARCH path, and an honestly-labelled short-vol carry backtest (mean, hit-rate, worst window, drawdown).
- **T4 — Forecast Validation**: forecast horse race (GARCH(1,1) vs EWMA vs HAR-RV vs IV) scored by QLIKE and MSE on log-RV, Mincer–Zarnowitz (a, b, R², joint (a,b)=(0,1) F-test), Diebold–Mariano with Newey–West LRV, encompassing blend (IV + GARCH), and GARCH-X with γ·IV²<sub>t-1</sub>.
- **T5 — Risk-Neutral Density**: Breeden–Litzenberger density from the fitted smile (vs realized 21d horizon histogram), implied vs realized moments, MFIV vs interpolation consistency audit, digital/touch probabilities (N(d2), 2·N(−|d|)), and a Merton (1974) distance-to-default credit bridge with PD = N(−DD).
- **T6 — Cross-Asset & Regime**: sortable IV matrix across the watchlist (IV-30, 25Δ RR, BF, skew, MFIV−ATM, accrual days), leverage-effect pricing (corr(Δ^VIX, r)), IV lead–lag cross-correlations, and a ^VIX × ^VVIX regime map (complacent / trending / hedging / panic).
- **T7 — Dealer Positioning & Trap**: a dealer gamma-hedging state machine, not a prediction arrow. GEX by strike ($ re-hedge per 1% move), gamma-flip point via 121-spot sticky-IV grid, call/put wall detection (excluding the 2 strikes nearest spot), 5-component deterministic trap score (wall / proximity / short-gamma regime / OI velocity / coil), USD hedge-flow per 1% band, and an event-study backtest of every past *triggered* state with a Wilson-interval verdict against the unconditional base rate. The directional arrow renders only when follow-through clears base rate — otherwise the card shows state only. Every GEX figure is captioned "dealer positioning is assumed, not observed" (squeeze convention: dealers long calls / short puts). Daily OI accrual is forward-only via the snapshot store. See §8 for the sibling US Options F&O suite.
- **T8 — Decision (Vol Edge)**: the decision layer, not another chart pile. An apples-to-apples strip (RV10/20/30, IV30, RV²/IV² variance ratio, pre-spike variance baseline, HAR next-day when intraday bars exist — the 250-day blend is deliberately absent), a variance-budget panel (continuous diffusion via bipower vs jumps vs signature-plateau microstructure noise, a lower bound; overnight close→open share; "harvestable intraday" share — gated and badge-tagged when the intraday cache is synthetic), a breakeven chart with IV30 and a configurable 1.25× IV30 reference line (`BREAKEVEN_MULT` in `lib/vol-signals.ts`, labelled uncalibrated), the IV30−RV20 spread with its 20-session rolling z-score, a forward expected-move cone (±1σ/±2σ from spot at IV30), an RV5/10/20/30 triangle classifier (backwardation/contango/humped/mixed), a realized-vs-implied forward-volatility decomposition (variance decomposition vs calendar w-difference, never σ interpolation), an HMM k=3 regime strip (posterior-shaded), a σ̂-standardized innovation readout above the checklist (Ĥ/min-VR-z/ADF-p on GARCH-filtered residuals — the decisive test against vol-clustering artifacts), a next-earnings line (per Yahoo quoteSummary calendarEvents, estimate-flagged, null-degrading), and a pre-registered signal checklist encoded as data in `lib/vol-signals.ts::evaluateVolSignals` — every rule degrades to `insufficient-data` when its inputs are missing; it never defaults to "met". A collapsible **reconciliation log** closes the tab: eight claim-by-claim verdicts (`arithmetic` / `estimate` / `hypothesis`) with the n≈250 asymptotic SEs and the multiple-comparison caveat. Signals are one operator's pre-registered rules with inputs visible — not orders, not investment advice, no broker connectivity. RV (√252) and IV (√365) are compared directly as annualizations of the same total variance — nothing is ever rescaled.

**Data & honesty**: chains are fetched server-side from Yahoo Finance's options endpoint (crumb session held in memory, ~15-min delayed). Index volatility history (`^VIX`, `^VIX9D`, `^VIX3M`, `^VIX6M`, `^VXN`, `^VVIX`, `^SKEW`, `^IRX`) comes from the chart endpoint. Next earnings dates (flagged as estimates) come from quoteSummary `calendarEvents` via `mode=earnings` — null-degrading to "no earnings data" whenever Yahoo's calendar surface is unreachable; failure is the designed answer, not an error. Single-name IV history cannot be backfilled — snapshots accrete per symbol/day in `.cache/iv/{SYMBOL}/{YYYY-MM-DD}.json` (gitignored); until ~60 sessions accrue, IV-rank panels show the real count rather than a fabricated percentile. Indices map to ETF proxies with a visible badge (`^IXIC`→QQQ, `^SPX`→SPY, `^NDX`→QQQ, `^DJI`→DIA, `^RUT`→IWM). Every IV number is a European BSM approximation; the surface reports its own filtered-strike count and median spread. When the live fetch fails the terminal falls back to deterministic synthetic GBM data labelled `synthetic_gbm` — the intraday budget panels tag that case as illustrative instead of sizing off it.

### 8. US Options F&O Suite (T1–T6)
- **T1 — Option Chain**: dual-sided table (CE/PE split), ATM row pinned, per-strike mid IV via Newton from mid-quote (never Yahoo's IV field), 25Δ/50Δ approximations, vega & delta per strike, OI/volume bars, strikes-used badge. 0-DTE expiries are displayed correctly (T floored at half a session).
- **T2 — OI Positioning**: dual-sided OI ladder (±15% of spot), per-expiry OI distribution with expiry-decay visualization, and a ≤7-DTE concentration metric that flags when the positioning is structurally short-dated.
- **T3 — PCR**: aggregate OI PCR, volume PCR, and ATM ±5-strike PCR side by side, plus a divergence warning when positioning and flow disagree by more than 0.4.
- **T4 — Max Pain**: full payout curve with the minimum marked, distance-to-pain, basin width (strikes ≤ 105% of minimum pain), and an explicit "heuristic, not a law" honesty footer.
- **T5 — Term Structure & Carry**: ATM IV across all twelve expiries, verdict anchored to ≥ 10 DTE (front week is event-noise), forward carry between expiries extracted via put–call parity, and 25Δ risk reversal per expiry where the smile fits.
- **T6 — Expected vs Realized Move**: ATM straddle and σ-implied move per expiry, compared against the realized |log-move| histogram over the matching horizon — with a CHEAP / FAIR / RICH verdict and percentile rank per row. Historical IV is not fetchable; this is today's pricing vs base rates, not an implied-vol backtest.

**Universe**: AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, JPM, SPY, QQQ. American-style equity options; prices are ~15-min delayed from Yahoo Finance; OI is end-of-day; IV history accrues at most one snapshot per day on page load and is never backfilled.

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
│   │   ├── market-data/route.ts   # Market data fetching & historical caching API
│   │   └── options/route.ts       # Option chains (Yahoo v7 + crumb), vol-index family, IV snapshot store
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
│   ├── fno/                        # US F&O suite: expiry ladder, positioning, term structure, expected move, config
│   ├── analysis-engine.ts          # Stationarity, EVT, VaR/ES & microstructure math
│   ├── correlation-engine.ts       # RMT, Ledoit-Wolf, MST & GMV algorithms
│   ├── distribution-engine.ts      # MLE for Normal, Student-t, Skewed-t, GED
│   ├── market-service.ts           # Synthetic generator & live data adapter
│   ├── options.ts                  # BSM, Newton IV, chain cleaning, SVI, MFIV, Breeden–Litzenberger
│   ├── gex.ts                      # Dealer GEX, gamma flip grid, walls, trap score & state machine
│   ├── trap-server.ts              # Server-side trap bundle (closes fetch, ΔOI tracking, panel accrual)
│   ├── trap-backtest.ts            # Event study of triggered traps (verdict strings, deciles)
│   ├── vol-analytics.ts            # Horse race (QLIKE/MSE), MZ/DM/encompassing, GARCH-X, moments
│   ├── options/providers/          # Provider interface + Yahoo adapter (crumb session)
│   ├── vol-indices.ts              # ^VIX family fetching (timeout + deterministic fallback)
│   ├── iv-store.ts                 # Accruing OI/IV snapshot store (.cache/iv, server-side fs) + panel.json
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
Open **[http://localhost:3528](http://localhost:3528)** in your browser.

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

### Implied Volatility Data
No configuration is required — chains and the vol-index family are fetched from Yahoo's public endpoints at request time (server-side, cached ~15 minutes) and IV snapshots accrete locally in `.cache/iv/` on each successful fetch. If Yahoo is unreachable, the app degrades gracefully: vol indices fall back to clearly-labelled synthetic levels, chain-dependent panels show an "unavailable" banner, and everything else keeps working.

---

## 📜 Available Scripts

| Command | Description |
| :--- | :--- |
| `npm run dev` | Runs the Next.js development server |
| `npm run build` | Compiles and optimizes the app for production |
| `npm run start` | Serves the production build |
| `npm run lint` | Runs ESLint to check for code issues |
| `npm run clean` | Cleans Next.js build cache |
| `npm run test:unit` | Pure-math unit tests (`node --test`, zero dependencies) |

---

## 🛡️ License

This project is open-source and available under the [MIT License](LICENSE).
