// Production runner for `output: 'standalone'`.
// Syncs .next/static (and public/, if present) into the standalone tree —
// the documented requirement for serving standalone builds — then execs server.js.
// Usage: npm run start   (PORT env var overrides the default 3528)
import { cpSync, existsSync, rmSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const standaloneDir = path.join(root, '.next', 'standalone');
const serverPath = path.join(standaloneDir, 'server.js');

if (!existsSync(serverPath)) {
  console.error('No standalone build at .next/standalone/server.js — run `npm run build` first.');
  process.exit(1);
}

// 1. Sync static assets into the standalone tree (idempotent)
rmSync(path.join(standaloneDir, '.next', 'static'), { recursive: true, force: true });
cpSync(path.join(root, '.next', 'static'), path.join(standaloneDir, '.next', 'static'), {
  recursive: true,
});

// 2. Sync public/ when it exists
const publicDir = path.join(root, 'public');
if (existsSync(publicDir)) {
  rmSync(path.join(standaloneDir, 'public'), { recursive: true, force: true });
  cpSync(publicDir, path.join(standaloneDir, 'public'), { recursive: true });
}

// 3. Launch the standalone server
const port = process.env.PORT ?? '3000';
const hostname = process.env.HOSTNAME ?? '0.0.0.0';
const child = spawn(process.execPath, [serverPath], {
  stdio: 'inherit',
  env: { ...process.env, PORT: port, HOSTNAME: hostname },
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
child.on('exit', (code) => process.exit(code ?? 0));
