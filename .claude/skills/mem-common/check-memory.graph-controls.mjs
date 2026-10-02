import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const checker = path.resolve(process.argv[2] || path.join(path.dirname(fileURLToPath(import.meta.url)), 'check-memory.mjs'));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'memory-graph-controls-'));
const owner = '27-admin-order-filters.md';
const peers = ['09-admin-design-system.md', '17-storefront-playbook.md', '23-order-editing-balance.md', '04-infra-deploy.md'];
function fixture(runtime) {
  const repo = fs.mkdtempSync(path.join(root, 'repo-'));
  const memory = path.join(repo, `.${runtime}/agents/memory`);
  const projects = path.join(repo, 'external-memory');
  fs.mkdirSync(memory, { recursive: true }); fs.mkdirSync(projects);
  fs.writeFileSync(path.join(repo, 'CLAUDE.md'), 'Every file lives in-repo except 06-working-style.md. Numbering is a single sequence.\n');
  fs.writeFileSync(path.join(repo, 'AGENTS.md'), `Project memory: .${runtime}/agents/memory/\n`);
  fs.writeFileSync(path.join(memory, 'MEMORY.md'), '# Memory map\n\n' + [owner, ...peers].map(f => `### ${f} â€” topic\n[Read topic](${f})\n**Keywords:** filters\n**Latest:** 2026-09-30 verified\n`).join('\n'));
  fs.writeFileSync(path.join(memory, owner), '# 27 â€” Filters\n\n**Fact: admin-order-filters.integration:** merged_local\n\n## Related\n' + peers.map(f => `- [Related](${f}) â€” owns a relevant dependency.\n`).join(''));
  for (const f of peers) fs.writeFileSync(path.join(memory, f), `# ${f}\n\n## Related\n- [Filters](${owner}) â€” canonical feature contract.\n`);
  fs.writeFileSync(path.join(memory, 'memory-graph.json'), JSON.stringify({ version: 1, topics: [{ id: 'admin-order-filters', canonical: owner, entrypoints: peers, facts: [{ id: 'admin-order-filters.integration', value: 'merged_local', observed: '2026-09-30', evidence: ['git ancestor check'] }] }] }, null, 2));
  return { repo, memory, projects, runtime };
}
function edit(f, name, fn) { const p = path.join(f.memory, name); fs.writeFileSync(p, fn(fs.readFileSync(p, 'utf8'))); }
function mirrored(f) {
  const other = path.join(f.repo, `.${f.runtime === 'claude' ? 'codex' : 'claude'}/agents/memory`);
  const graphPath = path.join(f.memory, 'memory-graph.json');
  const graph = JSON.parse(fs.readFileSync(graphPath, 'utf8'));
  graph.mirrors = { runtime_views: ['.codex/agents/memory', '.claude/agents/memory'], files: [owner, 'memory-graph.json'] };
  fs.writeFileSync(graphPath, JSON.stringify(graph));
  fs.cpSync(f.memory, other, { recursive: true });
  return other;
}
function visual(f, render = false) {
  edit(f, 'memory-graph.json', text => { const graph = JSON.parse(text); graph.navigation = { generated: true }; return JSON.stringify(graph); });
  if (render) {
    const result = spawnSync(process.execPath, [path.join(path.dirname(checker), 'render-memory-graph.mjs'), f.repo, '--runtime', f.runtime], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
  }
}
const cases = [
  ['missing main graph', f => visual(f), 'G13'],
  ['fresh main graph', f => visual(f, true), null],
  ['fact-only stale main graph', f => { visual(f, true); edit(f, owner, t => t + '\nNew fact without new edges.\n'); }, 'G13'],
  ['valid connected graph', () => {}, null],
  ['dangling link', f => edit(f, owner, t => t + '\n[Missing](99-missing.md)\n'), 'G7'],
  ['missing incoming route', f => edit(f, peers[0], t => t.replace(`[Filters](${owner})`, 'Filters')), 'G8'],
  ['missing outgoing relationship', f => edit(f, owner, t => t.replace(`[Related](${peers[0]})`, 'Related')), 'G8'],
  ['contradictory current state', f => edit(f, peers[2], t => t + '\n**Fact: admin-order-filters.integration:** awaiting_merge\n'), 'G9'],
  ['duplicate fact owner', f => edit(f, peers[2], t => t + '\n**Fact: admin-order-filters.integration:** merged_local\n'), 'G9'],
  ['missing owner fact', f => edit(f, owner, t => t.replace('**Fact: admin-order-filters.integration:** merged_local', '')), 'G9'],
  ['oversized index', f => edit(f, 'MEMORY.md', t => t + ('noise '.repeat(2500))), 'G10'],
  ['oversized canonical topic', f => edit(f, owner, t => t + ('noise '.repeat(2500))), 'G10'],
  ['duplicate latest in index', f => edit(f, 'MEMORY.md', t => t.replace('**Latest:** 2026-09-30 verified', '**Latest:** current\n**Latest:** outdated')), 'G11'],
  ['historical snapshot is not current truth', f => { fs.mkdirSync(path.join(f.memory, 'archive')); fs.writeFileSync(path.join(f.memory, 'archive/old.md'), '**Fact: admin-order-filters.integration:** awaiting_merge\n'); }, null],
  ['missing evidence', f => edit(f, 'memory-graph.json', t => { const g = JSON.parse(t); g.topics[0].facts[0].evidence = []; return JSON.stringify(g); }), 'G9'],
  ['frontmatter', f => edit(f, owner, t => '---\nname: bad\n---\n' + t), 'G1'],
  ['unindexed node', f => fs.writeFileSync(path.join(f.memory, '28-orphan.md'), '# Orphan\n'), 'G3'],
  ['managed views agree', f => mirrored(f), null],
  ['managed view drift', f => { const other = mirrored(f); fs.appendFileSync(path.join(other, owner), '\nStale copy\n'); }, 'G12'],
  ['missing managed view', f => { const other = mirrored(f); fs.unlinkSync(path.join(other, owner)); }, 'G12'],
  ['broken heading anchor', f => edit(f, peers[0], t => t.replace(`](${owner})`, `](${owner}#missing-section)`)), 'G7'],
  ['broken index anchor', f => edit(f, 'MEMORY.md', t => t.replace(`](${owner})`, `](${owner}#missing-section)`)), 'G7'],
];
let failures = 0;
try {
  for (const runtime of ['claude', 'codex']) for (const [name, mutate, gate] of cases) {
    const f = fixture(runtime); mutate(f);
    const result = spawnSync(process.execPath, [checker, f.repo, f.projects, '--runtime', runtime], { encoding: 'utf8' });
    const output = (result.stdout || '') + (result.stderr || '');
    const passed = gate ? result.status === 1 && output.includes(`${gate}  `) : result.status === 0;
    if (!passed) failures++;
    console.log(`${passed ? 'ok' : 'BAD'} ${runtime}: ${name}${gate ? ` â†’ ${gate}` : ''}`);
    if (!passed) console.log(output);
  }
} finally { fs.rmSync(root, { recursive: true, force: true }); }
assert.equal(failures, 0, `${failures} graph control(s) failed`);
console.log('PASS all graph controls; fixtures removed');

