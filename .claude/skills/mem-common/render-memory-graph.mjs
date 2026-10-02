#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const START = '<!-- memory-visual-graph:start -->';
const END = '<!-- memory-visual-graph:end -->';
const read = p => fs.readFileSync(p, 'utf8');
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
export function stripVisualGraph(index) {
  const starts = index.split(START).length - 1, ends = index.split(END).length - 1;
  if (starts !== ends || starts > 1) throw new Error('duplicate or incomplete main graph markers; reconcile before writing');
  const match = index.match(/<!-- memory-visual-graph:start -->[\s\S]*?<!-- memory-visual-graph:end -->\r?\n(?:\r?\n)?/);
  if (starts && !match) throw new Error('main graph markers are malformed');
  return { base: match ? index.replace(match[0], '') : index, existing: match?.[0] || '' };
}
export function buildVisualGraph(memory) {
  const index = read(path.join(memory, 'MEMORY.md'));
  const { base } = stripVisualGraph(index);
  const graphText = read(path.join(memory, 'memory-graph.json'));
  const graph = JSON.parse(graphText);
  const files = [...new Set([...base.matchAll(/^###\s+(\d{2}-[^\s]+\.md)\b/gm)].map(m => m[1]))].sort();
  if (!files.length) throw new Error('no indexed memory topics');
  const documents = files.map(file => ({ file, text: read(path.join(memory, file)) }));
  const byStem = new Map(files.map(file => [file.replace(/\.md$/, ''), file]));
  const ids = new Map(files.map(file => [file, 'n' + file.replace(/\.md$/, '').replace(/[^a-zA-Z0-9]/g, '_')]));
  const edges = new Map();
  const add = (from, to, managed = false) => {
    if (!ids.has(from) || !ids.has(to) || from === to) return;
    const pair = [from, to].sort(), key = pair.join('\0');
    if (!edges.has(key) || managed) edges.set(key, { from: pair[0], to: pair[1], managed });
  };
  for (const topic of graph.topics || []) for (const peer of topic.entrypoints || []) add(topic.canonical, peer, true);
  for (const { file, text } of documents) {
    for (const match of text.matchAll(/\[[^\]\n]*\]\(([^)\n]+)\)|\[\[([^\]\n]+)\]\]/g)) {
      const target = (match[1] || match[2]).split(/[|#]/)[0].trim();
      if (/^[a-z][a-z0-9+.-]*:/i.test(target)) continue;
      const normalized = target.replace(/^\.\//, '');
      const destination = ids.has(normalized) ? normalized : byStem.get(normalized);
      if (destination) add(file, destination);
    }
  }
  const signature = hash(JSON.stringify([base.replace(/\r\n/g, '\n'), graphText.replace(/\r\n/g, '\n'), documents.map(d => [d.file, d.text.replace(/\r\n/g, '\n')])]));
  const lines = [START, '## Memory graph', '', '<details>', '<summary>Show all indexed topics and their connections</summary>', '', 'Solid links are managed canonical relationships; dotted links are existing cross-references, not proof of current state. Click a node to read its note. Historical archives stay outside this current-topic view.', '', `<!-- memory-graph-source: ${signature} -->`, '```mermaid', 'flowchart LR'];
  for (const file of files) lines.push(`  ${ids.get(file)}["${file.replace(/\.md$/, '').replace('-', ' · ')}"]`);
  for (const edge of [...edges.values()].sort((a, b) => (a.from + a.to).localeCompare(b.from + b.to))) lines.push(`  ${ids.get(edge.from)} ${edge.managed ? '<-->|"managed relationship"|' : '-.-'} ${ids.get(edge.to)}`);
  for (const file of files) lines.push(`  click ${ids.get(file)} "${file}" "Read ${file}"`);
  lines.push('```', '', '</details>', END, '', '');
  return { block: lines.join('\n'), base, nodes: files.length, edges: edges.size };
}
export function checkVisualGraph(memory) {
  const current = stripVisualGraph(read(path.join(memory, 'MEMORY.md'))).existing;
  return current === buildVisualGraph(memory).block;
}
function main() {
  const args = process.argv.slice(2), runtimeAt = args.indexOf('--runtime');
  const installed = fileURLToPath(import.meta.url).replace(/\\/g, '/').match(/\/\.(claude|codex)\/skills\//)?.[1];
  const runtime = runtimeAt >= 0 ? args[runtimeAt + 1] : installed || 'codex';
  if (runtimeAt >= 0) args.splice(runtimeAt, 2);
  const check = args.includes('--check');
  const repo = path.resolve(args.find(arg => !arg.startsWith('--')) || process.cwd());
  if (!['claude', 'codex'].includes(runtime)) throw new Error('runtime must be claude or codex');
  const memory = path.join(repo, `.${runtime}/agents/memory`), indexPath = path.join(memory, 'MEMORY.md');
  const before = read(indexPath), { block, base, nodes, edges } = buildVisualGraph(memory);
  const oldBlock = stripVisualGraph(before).existing;
  if (check) {
    if (oldBlock !== block) { console.error('G13  main MEMORY.md graph is missing or stale; regenerate after every ingestion'); process.exitCode = 1; return; }
    console.log(`PASS main graph freshness (${nodes} topics, ${edges} connections)`); return;
  }
  if (oldBlock === block) { console.log('PASS main graph already current'); return; }
  const insertion = base.search(/^## Start by task/m);
  const position = insertion >= 0 ? insertion : base.search(/^## Topics/m);
  const after = position >= 0 ? base.slice(0, position) + block + base.slice(position) : base + block;
  const backup = fs.mkdtempSync(path.join(os.tmpdir(), 'memory-graph-render-'));
  fs.writeFileSync(path.join(backup, 'MEMORY.md'), before);
  if (read(indexPath) !== before) throw new Error('concurrent index edit; re-read before regeneration');
  fs.writeFileSync(indexPath, after);
  console.log(`PASS refreshed main graph (${nodes} topics, ${edges} connections); manual routes preserved`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(`G13  ${error.message}`); process.exitCode = 1; }
}
