#!/usr/bin/env node
// Structural gates plus opt-in graph/current-state gates; no network or writes.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { stripVisualGraph, checkVisualGraph } from './render-memory-graph.mjs';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const runtimeAt = args.indexOf('--runtime');
const installedRuntime = fileURLToPath(import.meta.url).replace(/\\/g, '/').match(/\/\.(claude|codex)\/skills\//)?.[1];
const requested = runtimeAt < 0 ? (installedRuntime || 'auto') : args[runtimeAt + 1];
if (runtimeAt >= 0) args.splice(runtimeAt, 2);
const repo = path.resolve(args[0] || process.cwd());
const exists = p => fs.existsSync(p);
const read = p => exists(p) ? fs.readFileSync(p, 'utf8') : '';
const agents = read(path.join(repo, 'AGENTS.md'));
const claude = read(path.join(repo, 'CLAUDE.md'));
const runtime = requested === 'auto'
  ? (/\.codex[\\/]agents[\\/]memory/.test(agents) || !exists(path.join(repo, '.claude/agents/memory')) ? 'codex' : 'claude')
  : requested;
if (!['codex', 'claude'].includes(runtime)) { console.error('G0  --runtime must be codex, claude or auto'); process.exit(1); }
const slug = repo.replace(/^([A-Za-z]):/, (_, d) => d.toUpperCase() + '-').replace(/[\\/:]/g, '-').replace(/^([A-Z])-+/, '$1--');
const ownExternal = path.join(os.homedir(), `.${runtime}`, 'projects', slug, 'memory');
const legacyExternal = path.join(os.homedir(), '.claude', 'projects', slug, 'memory');
const storeA = path.resolve(args[1] || (exists(ownExternal) ? ownExternal : legacyExternal));
const storeB = path.join(repo, `.${runtime}`, 'agents', 'memory');
const fails = [];
const fail = (gate, message) => fails.push(`${gate}  ${message}`);
const numbered = dir => exists(dir) ? fs.readdirSync(dir).filter(f => /^\d{2}-.+\.md$/.test(f)) : [];
const listed = text => new Set([...text.matchAll(/^###\s+(\d{2}-[^\s]+\.md)\b(.*)$/gm)].filter(m => !/MOVED/i.test(m[2])).map(m => m[1]));

if (!exists(storeB)) { console.error(`G0  selected memory store does not exist: ${storeB}`); process.exit(1); }
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p);
    else if (entry.name.endsWith('.md') && read(p).replace(/^\uFEFF/, '').startsWith('---')) fail('G1', `frontmatter in memory: ${path.relative(storeB, p)}`);
  }
}
walk(storeB);
const A = numbered(storeA), B = numbered(storeB);
for (const f of A.filter(f => B.includes(f))) fail('G2', `same numbered file in project and external stores: ${f}`);
const idx = read(path.join(storeB, 'MEMORY.md'));
for (const [name, dir, files] of [['projects-store', storeA, A], ['in-repo', storeB, B]]) {
  const index = read(path.join(dir, 'MEMORY.md'));
  if (!index && files.length) { fail('G3', `${name} has notes but no MEMORY.md`); continue; }
  const indexed = listed(index);
  for (const f of files) if (!indexed.has(f)) fail('G3', `${name}: ${f} is not indexed`);
  for (const f of indexed) if (!files.includes(f)) fail('G4', `${name}: index target missing: ${f}`);
}
const routing = agents + '\n' + claude;
const allowed = routing.match(/every file lives in-repo except\s+`?(\d{2}-[^`\s]+\.md)`?/i);
if (allowed) for (const f of A) if (f !== allowed[1] && !/working-style/.test(f)) fail('G5', `external ${f} conflicts with project routing (only ${allowed[1]} is allowed)`);
if (/single sequence/i.test(routing)) {
  const numbers = new Map();
  for (const f of [...A, ...B]) { const n = f.slice(0, 2); if (!numbers.has(n)) numbers.set(n, new Set()); numbers.get(n).add(f); }
  for (const [n, files] of numbers) if (files.size > 1) fail('G6', `number ${n} reused: ${[...files].join(', ')}`);
}

const graphPath = path.join(storeB, 'memory-graph.json');
const wordCount = text => text.trim().split(/\s+/u).filter(Boolean).length;
let graph;
if (exists(graphPath)) {
  try { graph = JSON.parse(read(graphPath)); } catch { fail('G8', 'memory-graph.json is not valid JSON'); }
  if (graph && (graph.version !== 1 || !Array.isArray(graph.topics))) { fail('G8', 'graph must have version1 and a topics array'); graph = undefined; }
  let manualIndex = idx;
  try { manualIndex = stripVisualGraph(idx).base; } catch (error) { fail('G13', error.message); }
  if (graph?.navigation?.generated) {
    try { if (!checkVisualGraph(storeB)) fail('G13', 'main MEMORY.md graph is missing or stale; regenerate after every ingestion'); } catch (error) { fail('G13', error.message); }
  }
  if (wordCount(manualIndex) > 2000 || manualIndex.split('\n').length > 220) fail('G10', 'MEMORY.md exceeds 2000 words or 220 lines; split evidence out of the navigation map');
  const blocks = [...idx.matchAll(/^###\s+(\d{2}-[^\s]+\.md)\b[^\n]*\n([\s\S]*?)(?=^###\s|$(?![\s\S]))/gm)];
  const names = new Set();
  for (const block of blocks) {
    if (names.has(block[1])) fail('G11', `duplicate index block: ${block[1]}`);
    names.add(block[1]);
    for (const field of ['Keywords', 'Latest']) if ([...block[2].matchAll(new RegExp(`^\\*\\*${field}:\\*\\*`, 'gm'))].length !== 1) fail('G11', `${block[1]} needs exactly one ${field} field`);
  }
}
const mdLinks = text => [...text.matchAll(/\[[^\]\n]*\]\(([^)\n]+)\)/g)].map(m => m[1].trim().replace(/^<|>$/g, ''));
const slugify = heading => heading.toLowerCase().replace(/[`*]/g, '').replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
const anchors = text => {
  const counts = new Map();
  return new Set([...text.matchAll(/^#{1,6}\s+(.+?)\s*#*$/gm)].map(m => { const slug = slugify(m[1]); const n = counts.get(slug) || 0; counts.set(slug, n + 1); return n ? `${slug}-${n}` : slug; }));
};
function checkLinks(file, text) {
  for (const target of mdLinks(text)) {
    if (/^[a-z][a-z0-9+.-]*:/i.test(target)) continue;
    let decoded;
    try { decoded = decodeURIComponent(target); } catch { fail('G7', `${file}: malformed link encoding`); continue; }
    const [name, anchor] = decoded.split('#');
    const destination = name ? path.resolve(storeB, path.dirname(file), name) : path.join(storeB, file);
    if (!exists(destination) || !fs.statSync(destination).isFile()) { fail('G7', `${file}: missing link target ${target}`); continue; }
    if (anchor && !anchors(read(destination)).has(anchor)) fail('G7', `${file}: missing heading anchor ${target}`);
  }
}
const linksTo = (text, file) => mdLinks(text).some(link => link.split('#')[0] === file);
const relatedBlock = text => text.match(/^## Related memory\s*\n([\s\S]*?)(?=^## |$(?![\s\S]))/m)?.[1] || '';
const declarations = new Map();
for (const file of B) for (const declaration of read(path.join(storeB, file)).matchAll(/^\*\*Fact:\s*([a-z0-9.-]+):\*\*\s*([^\r\n]+)$/gm)) {
  if (!declarations.has(declaration[1])) declarations.set(declaration[1], []);
  declarations.get(declaration[1]).push({ file, value: declaration[2].trim() });
}
const topicIds = new Set(), factIds = new Set(), owners = new Set();
if (graph) checkLinks('MEMORY.md', idx);
for (const topic of graph?.topics || []) {
  if (!topic || typeof topic.id !== 'string' || typeof topic.canonical !== 'string' || !B.includes(topic.canonical) || !Array.isArray(topic.entrypoints) || !Array.isArray(topic.facts)) { fail('G8', 'invalid managed topic owner/entrypoints/facts'); continue; }
  if (topicIds.has(topic.id) || owners.has(topic.canonical)) fail('G8', `duplicate topic ID or canonical owner: ${topic.id}`);
  topicIds.add(topic.id); owners.add(topic.canonical);
  const text = read(path.join(storeB, topic.canonical));
  if (wordCount(text) > 1800) fail('G10', `${topic.canonical} exceeds 1800 words; split a coherent topic`);
  checkLinks(topic.canonical, text);
  if (!linksTo(idx, topic.canonical)) fail('G8', `MEMORY.md has no real route to ${topic.canonical}`);
  if (!topic.entrypoints.length) fail('G8', `${topic.id}: no entrypoints`);
  for (const peer of topic.entrypoints) {
    if (typeof peer !== 'string' || !B.includes(peer) || peer === topic.canonical) { fail('G8', `${topic.id}: invalid entrypoint ${peer}`); continue; }
    const peerText = read(path.join(storeB, peer));
    // Only the maintained related block (or the small leading summary in fixtures)
    // is strict. Untouched legacy prose is not falsely certified as graph-clean.
    const routeText = relatedBlock(peerText) || peerText.slice(0, 6000);
    checkLinks(peer, routeText);
    if (!linksTo(routeText, topic.canonical)) fail('G8', `${peer}: missing incoming route to ${topic.canonical}`);
    if (!linksTo(text, peer)) fail('G8', `${topic.canonical}: missing reason-linked dependency/backlink to ${peer}`);
  }
  for (const fact of topic.facts) {
    if (!fact || typeof fact.id !== 'string' || typeof fact.value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(fact.observed || '') || !Array.isArray(fact.evidence) || !fact.evidence.length || fact.evidence.some(item => typeof item !== 'string' || !item.trim())) { fail('G9', `${topic.id}: invalid or unevidenced fact`); continue; }
    if (factIds.has(fact.id)) fail('G9', `fact ${fact.id} is registered twice`);
    factIds.add(fact.id);
    const claims = declarations.get(fact.id) || [];
    if (claims.length !== 1 || claims[0]?.file !== topic.canonical || claims[0]?.value !== fact.value) fail('G9', `${fact.id}: expected one ${topic.canonical} owner with value ${fact.value}; found ${claims.map(c => `${c.file}=${c.value}`).join(', ') || 'none'}`);
  }
}
if (graph) for (const id of declarations.keys()) if (!factIds.has(id)) fail('G9', `current fact ${id} has no registered owner/evidence`);

if (graph?.mirrors) {
  const { runtime_views: views, files } = graph.mirrors;
  if (!Array.isArray(views) || views.length < 2 || new Set(views).size !== views.length || views.some(view => !/^\.(codex|claude)\/agents\/memory$/.test(view)) || !Array.isArray(files) || !files.length || files.some(file => typeof file !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.-]+$/.test(file))) {
    fail('G12', 'managed mirror declaration needs distinct authorized runtime views and simple filenames');
  } else {
    for (const file of files) {
      const paths = views.map(view => path.join(repo, view, file));
      if (paths.some(p => !exists(p) || !fs.statSync(p).isFile())) { fail('G12', `managed mirror missing: ${file}`); continue; }
      const canonicalBytes = fs.readFileSync(paths[0]);
      if (paths.slice(1).some(p => !canonicalBytes.equals(fs.readFileSync(p)))) fail('G12', `managed mirror drift: ${file}`);
    }
  }
}

console.log(`checked ${runtime}: ${storeB} (${B.length} notes); external ${A.length} notes`);
if (fails.length) { console.log(`FAIL (${fails.length})\n${fails.join('\n')}`); process.exit(1); }
console.log(graph ? 'PASS all structural and managed graph gates' : 'PASS structural gates only; graph not configured');
