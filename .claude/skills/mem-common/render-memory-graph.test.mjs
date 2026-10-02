import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const script = path.join(path.dirname(fileURLToPath(import.meta.url)), 'render-memory-graph.mjs');
assert.ok(fs.existsSync(script), 'main memory graph renderer exists');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'main-memory-graph-'));
let checks = 0;
function check(value, message) { assert.ok(value, message); checks++; }
function run(repo, runtime, ...flags) { return spawnSync(process.execPath, [script, repo, '--runtime', runtime, ...flags], { encoding: 'utf8' }); }
try {
  for (const runtime of ['claude', 'codex']) {
    const repo = fs.mkdtempSync(path.join(root, 'repo-'));
    const memory = path.join(repo, `.${runtime}/agents/memory`);
    fs.mkdirSync(memory, { recursive: true });
    const index = path.join(memory, 'MEMORY.md');
    const preserved = '# Memory\n\n## Start by task\nKeep my manual route.\n\n## Topics\n\n### 01-payments.md — Payments\n[Payments](01-payments.md)\n**Keywords:** money\n**Latest:** verified\n\n### 27-filters.md — Filters\n[Filters](27-filters.md)\n**Keywords:** filters\n**Latest:** verified\n';
    fs.writeFileSync(index, preserved);
    fs.writeFileSync(path.join(memory, '01-payments.md'), '# Payments\n\n## Related memory\n[Filters](27-filters.md) — payment classification.\n');
    fs.writeFileSync(path.join(memory, '27-filters.md'), '# Filters\n\n## Related memory\n[Payments](01-payments.md) — actual money state.\n');
    fs.writeFileSync(path.join(memory, 'memory-graph.json'), JSON.stringify({ version: 1, navigation: { generated: true }, topics: [{ id: 'filters', canonical: '27-filters.md', entrypoints: ['01-payments.md'], facts: [] }] }));
    check(run(repo, runtime, '--check').status === 1, `${runtime}: missing diagram fails`);
    check(run(repo, runtime).status === 0, `${runtime}: diagram renders`);
    const initial = fs.readFileSync(index, 'utf8');
    check(initial.includes('```mermaid') && initial.includes('flowchart LR'), `${runtime}: main file contains visual graph`);
    check(initial.includes('27-filters.md') && initial.includes('click'), `${runtime}: nodes link to actual notes`);
    check(initial.includes('<-->|"managed relationship"|'), `${runtime}: managed relationship drawn`);
    check(initial.includes('Keep my manual route.'), `${runtime}: manual index content preserved`);
    check(run(repo, runtime, '--check').status === 0, `${runtime}: diagram matches sources`);
    check(run(repo, runtime).status === 0 && fs.readFileSync(index, 'utf8') === initial, `${runtime}: repeated generation is deterministic`);
    fs.appendFileSync(path.join(memory, '27-filters.md'), '\nA new verified fact.\n');
    check(run(repo, runtime, '--check').status === 1, `${runtime}: fact-only ingest must refresh graph source stamp`);
    check(run(repo, runtime).status === 0, `${runtime}: fact-only refresh succeeds`);
    fs.writeFileSync(path.join(memory, '28-new.md'), '# New\n[Payments](01-payments.md)\n');
    fs.appendFileSync(index, '\n### 28-new.md — New\n[New](28-new.md)\n**Keywords:** new\n**Latest:** verified\n');
    check(run(repo, runtime, '--check').status === 1, `${runtime}: new indexed file makes diagram stale`);
    check(run(repo, runtime).status === 0 && fs.readFileSync(index, 'utf8').includes('n28_new'), `${runtime}: newly indexed file appears`);
    fs.writeFileSync(path.join(memory, '28-new.md'), '# New\n[Filters](27-filters.md)\n');
    check(run(repo, runtime, '--check').status === 1, `${runtime}: changed connection makes diagram stale`);
    check(run(repo, runtime).status === 0 && run(repo, runtime, '--check').status === 0, `${runtime}: changed connection regenerated and checked`);
    fs.appendFileSync(index, '\n<!-- memory-visual-graph:start -->\nextra\n<!-- memory-visual-graph:end -->\n');
    check(run(repo, runtime).status === 1, `${runtime}: duplicate block requires reconciliation`);
  }
} finally { fs.rmSync(root, { recursive: true, force: true }); }
console.log(`PASS ${checks} main-graph rendering and freshness checks`);
