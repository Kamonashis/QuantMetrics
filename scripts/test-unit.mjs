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
  'lib/options.ts lib/gex.ts lib/vol-analytics.ts lib/quant-engine.ts lib/hmm.ts lib/vol-signals.ts lib/fno/config.ts lib/fno/types.ts lib/fno/expiry-ladder.ts lib/fno/positioning.ts lib/fno/term-structure.ts lib/fno/expected-move.ts',
  `--outDir ${JSON.stringify(outDir)}`,
  '--target es2020 --module commonjs --skipLibCheck --esModuleInterop',
].join(' ');

execSync(cmd, { stdio: 'inherit', cwd: root });
console.log('\n--- running node --test ---\n');
execSync('node --test lib/__tests__/*.test.cjs', { stdio: 'inherit', cwd: root });
