import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const checker = path.resolve(process.argv[2] || path.join(path.dirname(fileURLToPath(import.meta.url)), 'check-memory.mjs'));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'memory-structural-controls-'));
function fixture(runtime) {
  const repo = fs.mkdtempSync(path.join(root, 'repo-'));
  const memory = path.join(repo, `.${runtime}/agents/memory`), projects = path.join(repo, 'external');
  fs.mkdirSync(memory, { recursive: true }); fs.mkdirSync(projects);
  fs.writeFileSync(path.join(repo, 'CLAUDE.md'), 'Every file lives in-repo except 06-working-style.md. Numbering is a single sequence.\n');
  fs.writeFileSync(path.join(repo, 'AGENTS.md'), `Project memory: .${runtime}/agents/memory/\n`);
  fs.writeFileSync(path.join(memory, '01-payments.md'), '# Payments\n');
  fs.writeFileSync(path.join(memory, 'MEMORY.md'), '# Memory\n\n### 01-payments.md — payments\n');
  fs.writeFileSync(path.join(projects, '06-working-style.md'), '# Style\n');
  fs.writeFileSync(path.join(projects, 'MEMORY.md'), '# External\n\n### 06-working-style.md — style\n');
  return { repo, memory, projects, runtime };
}
const cases = [
  ['valid legacy layout', () => {}, null],
  ['frontmatter becomes agent', f => fs.writeFileSync(path.join(f.memory, '01-payments.md'), '---\nname: bad\n---\n'), 'G1'],
  ['two homes', f => fs.copyFileSync(path.join(f.memory, '01-payments.md'), path.join(f.projects, '01-payments.md')), 'G2'],
  ['unindexed note', f => fs.writeFileSync(path.join(f.memory, '02-extra.md'), '# Extra\n'), 'G3'],
  ['missing indexed note', f => fs.unlinkSync(path.join(f.memory, '01-payments.md')), 'G4'],
  ['misrouted project fact', f => { fs.writeFileSync(path.join(f.projects, '17-shop.md'), '# Shop\n'); fs.appendFileSync(path.join(f.projects, 'MEMORY.md'), '\n### 17-shop.md — shop\n'); }, 'G5'],
  ['reused number', f => fs.writeFileSync(path.join(f.memory, '06-other.md'), '# Other\n'), 'G6'],
  ['missing selected store', f => fs.renameSync(f.memory, f.memory + '-moved'), 'G0'],
];
let failures = 0;
try {
  for (const runtime of ['claude', 'codex']) for (const [name, mutate, gate] of cases) {
    const f = fixture(runtime); mutate(f);
    const result = spawnSync(process.execPath, [checker, f.repo, f.projects, '--runtime', runtime], { encoding: 'utf8' });
    const output = (result.stdout || '') + (result.stderr || '');
    const passed = gate ? result.status === 1 && output.includes(`${gate}  `) : result.status === 0 && output.includes('structural gates only');
    if (!passed) failures++;
    console.log(`${passed ? 'ok' : 'BAD'} ${runtime}: ${name}`);
  }
  for (const runtime of ['claude', 'codex']) {
    const f = fixture(runtime);
    const installed = path.join(f.repo, `.${runtime}/skills/mem-common/check-memory.mjs`);
    fs.mkdirSync(path.dirname(installed), { recursive: true }); fs.copyFileSync(checker, installed);
    fs.copyFileSync(path.join(path.dirname(checker), 'render-memory-graph.mjs'), path.join(path.dirname(installed), 'render-memory-graph.mjs'));
    for (const [name, flags] of [['installation default', []], ['explicit project auto', ['--runtime', 'auto']]]) {
      const result = spawnSync(process.execPath, [installed, f.repo, f.projects, ...flags], { encoding: 'utf8' });
      const passed = result.status === 0 && result.stdout.includes(`checked ${runtime}:`);
      if (!passed) failures++;
      console.log(`${passed ? 'ok' : 'BAD'} ${runtime}: ${name}`);
    }
  }
} finally { fs.rmSync(root, { recursive: true, force: true }); }
assert.equal(failures, 0, `${failures} structural control(s) failed`);
console.log('PASS all structural controls; fixtures removed');
