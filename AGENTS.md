# AGENTS.md

QuantMetrics: a quantitative stock-analysis / volatility-modeling terminal. Next.js 15 (App Router, `output: 'standalone'`) + React 19 + TypeScript 5.9 + Tailwind 4. All quant math lives in pure-TS modules under `lib/`; UI is client-side (`'use client'`) and talks to one API route.

## Commands

- `npm run dev` — dev server on :3000 (falls back to :3001).
- `npm run lint` — ESLint 9 flat config (`eslint.config.mjs`). Currently passes clean.
- `npx tsc --noEmit` — typecheck. Currently passes clean. This is the real type gate.
- `npm run build` / `npm run start` — production build / serve.
- There is **no test suite and no CI**. Verify changes with `npm run lint` + `npx tsc --noEmit` (+ `npm run build` for risky changes).
- `npm run clean` — removes `.next/` (`rm -rf`). Note: `.next/` is shared, so never run any `next dev`/`start` for this repo twice concurrently — doing so corrupts the dev server (API routes 404 with "missing required error components"). Restart one server only.
- Use **npm**, not bun/pnpm. `package-lock.json` is canonical (node_modules was installed with npm per git history); `bun.lock` is a stale leftover.

## Architecture / wiring

- Path alias: `@/*` → repo root (e.g. `@/lib/quant-engine`).
- `lib/` holds the engines (no React): `quant-engine.ts` (GARCH/HAR-RV/returns/base stats), `analysis-engine.ts` (stationarity, EVT, VaR/ES), `correlation-engine.ts` (RMT, Ledoit-Wolf, MST, GMV), `distribution-engine.ts` (MLE fits), `market-service.ts` (data layer). Import these directly — don't re-implement math in components.
- `app/page.tsx` is the single-page coordinator (all client state, view switching). Views live in `components/views/*View.tsx`, composed from `components/views/<domain>/*Tab.tsx`. Charts in `components/charts/`, math/KaTeX rendering in `components/math/` + `components/ui/MathDisplay.tsx`.
- Data flow: browser → `app/api/market-data/route.ts` (`GET /api/market-data?tickers=...&range=...`, optional `interval=1m`, `startDate`/`endDate`; max 10 tickers) → `lib/market-service.ts` → Yahoo Finance chart API.
- `market-service.ts` **silently falls back to deterministic synthetic data** when Yahoo fails/rate-limits/times out (6s daily / 4.5s intraday). Responses carry `source: 'live_market' | 'cached_institutional' | 'synthetic_gbm'`. If numbers look "too clean", check `source` before assuming a bug. Synthetic data is seeded per ticker and reproducible.
- Client also imports `generateRealisticStockHistory` / `generateRealisticIntradayHistory` directly from `lib/market-service.ts` as an offline fallback — keep that module free of server-only APIs.

## Gotchas (hard-won)

- **Hydration/determinism**: `DETERMINISTIC_BASE_DATE` in `lib/market-service.ts` (`2026-11-08`) and hardcoded `todayStr = '2026-11-08'` / `oneYearAgoStr` in `app/page.tsx` exist for SSR/client parity. Do not replace with `new Date()` — it will cause hydration mismatches.
- **Theme**: server and client both default to dark; `layout.tsx` sets `<html className="dark">` + `suppressHydrationWarning`, and client syncs `localStorage['qt_theme']` in a `useEffect`. Keep this pattern when touching theme code.
- **`next build` does NOT lint**: `next.config.ts` sets `eslint.ignoreDuringBuilds: true` (but `typescript.ignoreBuildErrors: false`, so the build does typecheck). Run `npm run lint` separately.
- `eslint.config.mjs` (flat) is the config ESLint 9 actually uses; `.eslintrc.json` is legacy and ignored. `eslint-config-next` is v16 while `next` is v15 — works, but don't "fix" the mismatch casually.
- `DISABLE_HMR=true` env var makes the dev server ignore all file watching (AI Studio anti-flicker). If the dev server seems to not reload, check that var.
- `allowedDevOrigins` in `next.config.ts` includes AI Studio `*.run.app` origins — intentional, leave alone.
- `.env.local` is gitignored and holds AI Studio-injected vars (`GEMINI_API_KEY`, `APP_URL`). **No code uses `@google/genai`** — it's unused scaffolding; don't assume AI features exist. `metadata.json`/`.aistudio/` are AI Studio remnants too.
- `output: 'standalone'` + `transpilePackages: ['motion']` in `next.config.ts` — motion must stay transpiled or the build breaks.
- `.next/` and `.env*` (except `.env.example`) are gitignored. `tsconfig.tsbuildinfo` is tracked and changes on every `tsc` run — leave it alone.
- `KNOWN_TICKERS` in `lib/market-service.ts` defines the synthetic-data parameters for AAPL/MSFT/GOOGL/NVDA/TSLA/AMZN/SPY/QQQ/JPM/META; unknown tickers get generic fallback params.
- Charts use Canvas/SVG directly (no chart library); KaTeX CSS is imported once in `app/layout.tsx`.

## Design system (post-redesign)

- Tailwind 4 only; tokens live in `app/globals.css` (`@theme`: `--color-gold-*` champagne scale incl. `gold-950`; `--font-display` → Space Grotesk, `--font-sans` → Inter, `--font-mono` → IBM Plex Mono, wired via `next/font` in `layout.tsx`).
- `.panel` / `.panel-inset` are the card primitives (hairline border + inset highlight + soft shadow). `.kicker` is the gold mono uppercase section label. Use them instead of hand-rolled `bg-white … dark:bg-[#…] rounded-xl border` strings.
- Dark-first "obsidian" (`#08080A` base) + retuned light "paper" (`#F5F4F1`). Light panels are white; both share the gold accent and CSS vars (`--bg-*`, `--hairline`, `--positive`, `--negative`).
- Primary CTAs: `<ShimmerButton>` / `<MovingBorderButton>` (obsidian pill, gold text — identical in both themes). Inline pending actions: `components/ui/stateful-button.tsx`. Tab strips: `components/ui/animated-tabs.tsx` (shared gold pill).
- Aceternity-adapted components live in `components/ui/` (`spotlight`, `background-beams`, `meteors`, `moving-border`, `glowing-effect`, `bento-grid`, `text-generate-effect`, `flip-words`, `ticker-tape`). Import `motion` from `motion/react`, not `framer-motion`.
- Chart palette: gold = primary series, desaturated green/red = semantic up/down, sky/indigo largely migrated to gold/zinc — keep new charts on the gold/zinc/semantic set.

## Layout of the repo

`app/` (routes, layout, globals.css) · `app/api/market-data/route.ts` (the only API route) · `components/{charts,math,ui,layout,views}` · `lib/` (engines + utils) · `hooks/` (`use-mobile.ts` only) · `types/mathml.d.ts`. No `src/` directory.

## README

`README.md` is the product/feature doc (model descriptions). Trust code + config over it when they conflict.
