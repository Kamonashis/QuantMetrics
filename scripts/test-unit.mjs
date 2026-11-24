// Compile-free local test runner for the pure math modules.
// Bundles tsc emit into .cache/test-build (gitignored) and runs node --test.
// Usage: npm run test:unit
import { execSync } from 'node:child_process';
import { rmSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, '.cache', 'test-build');

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const cmd = [
  'npx tsc',
  'lib/fundamentals.ts lib/factors/types.ts lib/factors/price-factors.ts lib/factors/fundamental-factors.ts lib/factors/sentiment-options-factors.ts lib/factors/cross-section.ts lib/factors/index.ts lib/volume-profile.ts lib/options.ts lib/gex.ts lib/vol-analytics.ts lib/quant-engine.ts lib/factor-regression.ts lib/correlation-engine.ts lib/market-service.ts lib/market-cache.ts lib/home-universe.ts lib/plotly-theme.ts lib/tcn-attention.ts lib/tcn-multi-domain.ts lib/tcn-diagnostics.ts lib/hmm.ts lib/hmm-regime-overlay.ts lib/vol-signals.ts lib/analysis-engine.ts lib/earnings.ts lib/rnd.ts lib/ticker-snapshot.ts lib/options-decision.ts lib/forecast-trust.ts lib/opportunity-engine.ts lib/signal-outcomes.ts lib/fno/config.ts lib/fno/types.ts lib/fno/expiry-ladder.ts lib/fno/positioning.ts lib/fno/term-structure.ts lib/fno/expected-move.ts',
  `--outDir ${JSON.stringify(outDir)}`,
  '--target es2020 --module commonjs --skipLibCheck --esModuleInterop',
  '--lib es2022,dom lib/options-research-store.ts lib/credit-bridge.ts',
].join(' ');

execSync(cmd, { stdio: 'inherit', cwd: root });
console.log('\n--- running node --test ---\n');
execSync('node --test lib/__tests__/*.test.cjs', { stdio: 'inherit', cwd: root });
