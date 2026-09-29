'use strict';

// Q38FNInference control panel. Plain JS, no build step.
// Structure: helpers -> feedback (toasts, dialog) -> navigation -> settings ->
// hardware -> models -> run (status, plan, on-demand, speed test, log) -> init.

const state = {
  cfg: null,
  info: {},
  model: null,        // { path, label } chosen for single mode
  localGroups: [],
  plan: null,         // last /api/tune response
  gpus: [],
  status: {},
  mode: 'single',
  wasReady: false,
  lastLog: '',
};

const $ = (id) => document.getElementById(id);

// ---- helpers --------------------------------------------------------------

async function api(method, path, body) {
  const resp = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const e = data && data.error;
    throw new Error((e && (e.message || e)) || resp.statusText || `HTTP ${resp.status}`);
  }
  return data;
}

function fmtBytes(n) {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (n >= 100 || i === 0 ? n.toFixed(0) : n.toFixed(1)) + ' ' + u[i];
}

function fmtDuration(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return s + ' s';
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${s % 60} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

// "27.8 GB" -> ["27.8", "GB"], for number + unit at different sizes
function splitBytes(n) {
  const [v, u] = fmtBytes(n).split(' ');
  return [v, u];
}

// Infinite marquee: two identical sets scrolled by one set width (CSS keyframes).
// Rebuilt only when the text changes so the motion never restarts; speed in px/s.
function marquee(host, items, speed) {
  if (!host) return;
  host._args = [items, speed];
  // a strip on a hidden page measures 0 px wide: build it when the page is shown
  if (!host.offsetParent) { host.dataset.key = ''; return; }
  const key = JSON.stringify(items);
  if (host.dataset.key === key) return;
  host.dataset.key = key;
  const set = () => {
    const s = el('div', 'marquee-set');
    for (const it of items) {
      const span = el('span', 'marquee-item');
      if (Array.isArray(it)) { span.appendChild(el('b', '', it[0])); span.appendChild(document.createTextNode(it[1])); }
      else span.textContent = it;
      s.appendChild(span);
    }
    return s;
  };
  const track = el('div', 'marquee-track');
  track.append(set());
  host.replaceChildren(track);
  // repeat until one set is wider than the strip, so no gap ever shows
  const first = track.firstChild;
  while (first.scrollWidth < host.clientWidth && first.childNodes.length < 64) {
    for (const c of [...set().childNodes]) first.appendChild(c);
  }
  track.appendChild(first.cloneNode(true));
  host.style.setProperty('--marquee-dur', Math.max(8, first.scrollWidth / speed) + 's');
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const ICONS = {
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  warn: '<path d="M12 3l9 16H3l9-16z"/><path d="M12 10v4M12 17h.01"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>',
  ok: '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
};
function icon(name) {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = ICONS[name];
  return s;
}

function button(text, onClick, cls) {
  const b = el('button', 'btn ' + (cls || 'btn-secondary btn-sm'), text);
  b.type = 'button';
  b.addEventListener('click', (e) => { e.stopPropagation(); onClick(e); });
  return b;
}

function badge(text, kind) {
  return el('span', 'badge' + (kind ? ' badge-' + kind : ''), text);
}

// list row: title (+badges), meta line, actions on the right
function item({ title, badges = [], meta, actions = [], lead }) {
  const it = el('div', 'item');
  if (lead) it.appendChild(lead);
  const main = el('div', 'item-main');
  const t = el('div', 'item-title');
  t.appendChild(el('span', '', title));
  badges.forEach((b) => b && t.appendChild(b));
  main.appendChild(t);
  if (meta) main.appendChild(el('div', 'item-meta', meta));
  it.appendChild(main);
  if (actions.length) {
    const a = el('div', 'item-actions');
    actions.forEach((x) => x && a.appendChild(x));
    it.appendChild(a);
  }
  return it;
}

function empty(title, text, action) {
  const d = el('div', 'empty');
  d.appendChild(el('b', '', title));
  if (text) d.appendChild(el('span', '', text));
  if (action) d.appendChild(action);
  return d;
}

function callout(text, kind) {
  const c = el('div', 'callout callout-' + kind);
  c.appendChild(icon(kind === 'error' ? 'error' : kind === 'warn' ? 'warn' : 'info'));
  c.appendChild(el('div', '', text));
  return c;
}

// quantization from a GGUF file name, e.g. "...-IQ4_XS-00001-of-00008.gguf"
function quantOf(name) {
  const m = name.match(/(?:^|[-_.])((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|F32|MXFP4)(?=[-_.]|$)/i);
  return m ? m[1].toUpperCase() : '';
}

// does a model of `size` bytes fit the selected hardware? (weights only: a rough guide)
function fitOf(size) {
  const vram = selectedGPUs().reduce((a, g) => a + g.TotalBytes, 0);
  const ram = state.info.ram_total || 0;
  if (!size) return null;
  if (vram && size * 1.1 <= vram) return { kind: 'ok', label: 'fits in GPU' };
  if (ram && size <= vram + ram * 0.85) return { kind: 'warn', label: vram ? 'GPU + RAM' : 'runs on CPU' };
  if (ram) return { kind: 'bad', label: 'too large' };
  return null;
}

// shell-ish split so edited args with quoted paths survive
function splitArgs(s) {
  const out = [];
  const re = /"([^"]*)"|(\S+)/g;
  let m;
  while ((m = re.exec(s))) out.push(m[1] !== undefined ? m[1] : m[2]);
  return out;
}
const joinArgs = (a) => a.map((x) => (/\s/.test(x) || x === '' ? `"${x}"` : x)).join(' ');

// ---- feedback: toasts and confirm dialog ---------------------------------------
// Toasts for system messages that need no decision; the dialog only where an
// action is destructive or replaces the user's work.

function toast(text, kind = 'info', ms = 4500) {
  const t = el('div', 'toast ' + kind);
  t.appendChild(icon(kind === 'error' ? 'error' : kind === 'ok' ? 'ok' : kind === 'warn' ? 'warn' : 'info'));
  t.appendChild(el('div', 'toast-body', text));
  const x = el('button', 'toast-close', '×');
  x.setAttribute('aria-label', 'Dismiss');
  x.addEventListener('click', () => t.remove());
  t.appendChild(x);
  $('toasts').appendChild(t);
  if (kind !== 'error') setTimeout(() => t.remove(), ms);
}

function confirmDialog(title, text, okLabel = 'OK', danger = false) {
  const d = $('confirm');
  $('confirm-title').textContent = title;
  $('confirm-text').textContent = text;
  const ok = $('confirm-ok');
  ok.textContent = okLabel;
  ok.className = 'btn ' + (danger ? 'btn-danger' : 'btn-primary');
  d.returnValue = '';
  d.showModal();
  return new Promise((resolve) => d.addEventListener('close', () => resolve(d.returnValue === 'ok'), { once: true }));
}

// ---- navigation -----------------------------------------------------------

function showPage(name) {
  if (!$('page-' + name)) name = 'home';
  document.querySelectorAll('.nav-btn').forEach((b) => {
    const on = b.dataset.page === name;
    b.classList.toggle('active', on);
    if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  document.querySelectorAll('.page').forEach((p) => p.classList.toggle('active', p.id === 'page-' + name));
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  $('main').scrollTop = 0;
  window.scrollTo(0, 0);
  document.querySelectorAll('.page-head h1').forEach((h) => { h.style.transform = ''; h.style.opacity = ''; });
  remeasureMarquees();
}
document.querySelectorAll('.nav-btn').forEach((b) => b.addEventListener('click', () => showPage(b.dataset.page)));
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-goto]');
  if (t) showPage(t.dataset.goto);
});
window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));

// ---- settings -------------------------------------------------------------

const form = $('settings-form');
const numberFields = new Set(['port', 'model.ctx_size', 'model.parallel', 'model.threads', 'model.batch_size',
  'model.ubatch_size', 'model.temperature', 'model.top_p', 'model.top_k', 'model.min_p', 'model.repeat_penalty',
  'model.presence_penalty', 'model.max_tokens', 'model.seed', 'model.offload_min_batch']);

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setPath(obj, path, v) {
  const keys = path.split('.');
  const last = keys.pop();
  const target = keys.reduce((o, k) => (o[k] = o[k] || {}), obj);
  if (v === undefined) delete target[last]; else target[last] = v;
}

function fillForm(cfg) {
  for (const f of form.elements) {
    if (!f.name) continue;
    const v = getPath(cfg, f.name);
    if (f.type === 'checkbox') f.checked = !!v;
    else if (f.type === 'radio') f.checked = f.value === (v || 'off');
    else if (f.name === 'extra_model_dirs') f.value = (v || []).join('\n');
    else if (f.tagName === 'SELECT') {
      // unset or legacy values (e.g. reasoning "on") fall back to the marked default
      const ok = [...f.options].some((o) => o.value === String(v));
      const def = f.querySelector('option[selected]') || f.options[0];
      f.value = v != null && v !== '' && ok ? v : def.value;
    } else f.value = v == null ? '' : v;
  }
  updateSavebar();
}

function readForm() {
  const cfg = JSON.parse(JSON.stringify(state.cfg || {}));
  cfg.model = cfg.model || {};
  for (const f of form.elements) {
    if (!f.name) continue;
    let v;
    if (f.type === 'radio') { if (!f.checked) continue; v = f.value; }
    else if (f.type === 'checkbox') v = f.checked;
    else if (f.name === 'extra_model_dirs') v = f.value.split('\n').map((s) => s.trim()).filter(Boolean);
    else if (numberFields.has(f.name)) v = f.value === '' ? undefined : Number(f.value);
    else v = f.value.trim() === '' ? undefined : f.value.trim();
    if (f.name === 'system_prompt' && v !== undefined) v = f.value; // keep formatting
    setPath(cfg, f.name, v);
  }
  return cfg;
}

// compare only what the form edits, so values changed elsewhere (GPUs, mode) don't count
function formDirty() {
  if (!state.cfg) return false;
  const a = readForm();
  for (const f of form.elements) {
    if (!f.name || (f.type === 'radio' && !f.checked)) continue;
    const x = getPath(a, f.name), y = getPath(state.cfg, f.name);
    const norm = (v) => (v === undefined || v === null || v === '' || v === false || (Array.isArray(v) && !v.length) ? '' : JSON.stringify(v));
    if (f.type === 'radio' && (y || 'off') === x) continue;
    if (norm(x) !== norm(y)) return true;
  }
  return false;
}
function updateSavebar() { $('savebar').hidden = !formDirty(); }
form.addEventListener('input', updateSavebar);
form.addEventListener('change', updateSavebar);
$('settings-reset').addEventListener('click', () => fillForm(state.cfg));

async function saveConfig(cfg) {
  state.cfg = await api('POST', '/api/config', cfg);
  return state.cfg;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const invalid = [...form.elements].find((f) => f.willValidate && !f.checkValidity());
  if (invalid) {
    invalid.focus();
    toast(`${invalid.closest('.field')?.querySelector('.field-label')?.textContent || 'A field'}: ${invalid.validationMessage}`, 'error');
    return;
  }
  try {
    await saveConfig(readForm());
    fillForm(state.cfg);
    toast(state.status.Running ? 'Settings saved. Restart the model to apply model settings.' : 'Settings saved.', 'ok');
    computeTune();
  } catch (err) {
    toast('Could not save settings: ' + err.message, 'error');
  }
});

window.addEventListener('beforeunload', (e) => { if (formDirty()) { e.preventDefault(); e.returnValue = ''; } });

// presets only fill the text box; nothing is applied until Save
const promptPresets = {
  assistant: 'You are a helpful, knowledgeable assistant. Answer accurately and clearly, and say so when you are unsure.',
  concise: 'You are a helpful assistant. Keep answers short and to the point: no preamble, no restating the question, no filler.',
  direct: 'Answer the user directly and completely. Treat the user as a capable adult: do not moralize, lecture, add disclaimers or safety warnings they did not ask for, or water down your answer. If you genuinely cannot help with something, say so in one sentence without a lecture.',
  coding: [
    'You are an expert software engineer. Write correct, idiomatic, production-quality code.',
    '- Understand the goal before writing code; if the request is ambiguous in a way that changes the solution, ask one precise question instead of guessing.',
    '- Follow the language, framework and style already used in the code you are shown. Prefer the standard library and existing dependencies over new ones.',
    '- Give complete, runnable code for what you change: include imports and signatures, never elide parts with "..." or "rest unchanged" inside code the user must paste.',
    '- Handle realistic edge cases and errors at system boundaries, but do not add speculative abstractions, options or layers nobody asked for.',
    '- When fixing a bug, find the root cause and explain it in one or two sentences before the fix.',
    '- Keep explanations short and put them after the code. Mention anything you could not verify, such as an API you are unsure exists.',
  ].join('\n'),
  agentic: [
    'You are an autonomous coding agent working in the user\'s repository through the tools you are given. You carry tasks through to a verified result, not just a plan.',
    '',
    'How you work:',
    '1. Explore before you edit: list and read the relevant files, search for existing helpers and conventions, and check how the project builds and tests.',
    '2. Plan briefly, then act in small steps. After each change, verify it by building, running the tests or executing the code, and read the output.',
    '3. When something fails, read the error, find the root cause and fix that. Do not paper over it, disable tests or delete code to make errors disappear.',
    '4. Only use the tools you were given, exactly as their schemas describe. Never invent tool results, file contents, command output or test outcomes; if you did not run it, you do not know the result.',
    '5. Keep changes minimal and focused on the task, match the existing style, and do not touch unrelated code.',
    '6. Ask the user before destructive or irreversible actions (deleting files or data, force-pushing, installing system packages, network calls with side effects) and when a decision is genuinely theirs to make.',
    '7. Stop when the task is done and verified. Finish with a short report: what you changed, how you verified it, and anything left open.',
  ].join('\n'),
  research: [
    'You are a careful research assistant. Your job is to gather accurate information and report it clearly.',
    '- Start by making sure you understand what the user needs to know and why; ask one clarifying question if the request is ambiguous.',
    '- If you have search, browsing or retrieval tools, use them: look at several independent, authoritative sources rather than stopping at the first hit, and prefer primary sources (official documentation, papers, original data) over summaries.',
    '- Cite the source of every non-obvious claim (title and URL, or document name). Never invent citations, quotes, numbers or URLs.',
    '- Separate what the sources say from your own inference, note when sources disagree, and state how current the information is.',
    '- Say plainly when you do not know or could not find something, instead of filling the gap.',
    '- Structure the result: a short direct answer first, then supporting details, then open questions or suggested next steps.',
  ].join('\n'),
};

$('prompt-presets').addEventListener('click', async (e) => {
  const chip = e.target.closest('[data-preset]');
  if (!chip) return;
  const preset = chip.dataset.preset;
  const box = form.elements['system_prompt'];
  if (box.value.trim() && box.value !== promptPresets[preset] &&
      !(await confirmDialog('Replace the system prompt?', `The current text will be replaced by the "${chip.textContent}" preset.`, 'Replace'))) return;
  box.value = promptPresets[preset];
  const radios = form.querySelectorAll('input[name=system_prompt_mode]');
  const mode = [...radios].find((r) => r.checked);
  if (preset === 'agentic') {
    // agent clients send their own system prompt with the tool instructions
    radios.forEach((r) => { r.checked = r.value === 'combine'; });
    toast('Mode set to "Put in front" so agents keep their own instructions. Save to apply.');
  } else if (!mode || mode.value === 'off') {
    radios.forEach((r) => { r.checked = r.value === 'default'; });
    toast('Preset inserted and mode set to "As a default". Save to apply.');
  }
  updateSavebar();
  box.focus();
});

// ---- hardware -------------------------------------------------------------

function selectedGPUs() {
  const sel = (state.cfg && state.cfg.gpus) || [];
  if (sel.length === 1 && sel[0] === -1) return [];
  return state.gpus.filter((g) => sel.length === 0 || sel.includes(g.Index));
}

async function loadGPUs() {
  try {
    state.gpus = await api('GET', '/api/gpus');
  } catch (e) {
    state.gpus = [];
    $('gpu-list').replaceChildren(callout('Could not detect GPUs: ' + e.message, 'error'));
    return;
  }
  renderGPUs();
}

function renderGPUs() {
  const box = $('gpu-list');
  box.textContent = '';
  const ram = state.info.ram_total;
  $('hw-summary').textContent = ram ? `${fmtBytes(ram)} RAM (${fmtBytes(state.info.ram_available)} free)` : '';
  if (state.gpus.length === 0) {
    box.appendChild(callout('No NVIDIA GPU found. Models will run on the CPU, which is much slower. Install a current NVIDIA driver if you have a GPU.', 'warn'));
    return;
  }
  const chosen = new Set(selectedGPUs().map((g) => g.Index));
  const planBytes = {};
  const p = state.plan && state.plan.plan;
  if (p && p.Devices) p.Devices.forEach((d, i) => { planBytes[d] = p.DeviceBytes[i]; });

  for (const g of state.gpus) {
    const card = el('label', 'gpu' + (chosen.has(g.Index) ? '' : ' off'));
    const top = el('div', 'gpu-top');
    const cb = el('input');
    cb.type = 'checkbox';
    cb.checked = chosen.has(g.Index);
    cb.dataset.index = g.Index;
    cb.addEventListener('change', onGPUToggle);
    top.append(cb, el('span', '', g.Name));
    card.appendChild(top);
    const num = el('div', 'gpu-num', splitBytes(g.TotalBytes)[0]);
    num.appendChild(el('small', '', splitBytes(g.TotalBytes)[1] + ' VRAM'));
    card.appendChild(num);

    const used = g.TotalBytes - g.FreeBytes;
    const model = state.mode === 'single' && chosen.has(g.Index) ? (planBytes[g.Index] || 0) : 0;
    const meter = el('div', 'meter');
    meter.setAttribute('role', 'img');
    const usedBar = el('span', 'm-used');
    usedBar.style.width = (used / g.TotalBytes * 100) + '%';
    const modelBar = el('span', 'm-model');
    modelBar.style.width = Math.min(100, model / g.TotalBytes * 100) + '%';
    meter.append(usedBar, modelBar);
    meter.setAttribute('aria-label', `${fmtBytes(used)} in use by other programs, ${fmtBytes(model)} planned for the model, of ${fmtBytes(g.TotalBytes)}`);
    card.appendChild(meter);
    const leg = el('div', 'meter-legend');
    leg.append(el('span', '', model ? `model ${fmtBytes(model)}` : chosen.has(g.Index) ? `${fmtBytes(g.FreeBytes)} free` : 'not used'),
      el('span', '', `${fmtBytes(g.TotalBytes)} total`));
    card.appendChild(leg);
    box.appendChild(card);
  }
}

async function onGPUToggle() {
  const boxes = [...$('gpu-list').querySelectorAll('input[type=checkbox]')];
  const chosen = boxes.filter((b) => b.checked).map((b) => Number(b.dataset.index));
  const cfg = JSON.parse(JSON.stringify(state.cfg));
  // all selected == "use every GPU", which also covers GPUs added later
  cfg.gpus = chosen.length === boxes.length ? [] : chosen;
  if (chosen.length === 0) cfg.gpus = [-1];
  try { await saveConfig(cfg); } catch (e) { toast(e.message, 'error'); }
  if (chosen.length === 0) toast('No GPU selected: the model will run on the CPU.', 'warn');
  renderGPUs();
  computeTune();
}

// ---- models ---------------------------------------------------------------

async function loadLocalModels() {
  try {
    state.localGroups = await api('GET', '/api/models/local');
  } catch (e) {
    $('local-models').replaceChildren(callout('Could not scan for models: ' + e.message, 'error'));
    return;
  }
  if (!state.model && state.cfg && state.cfg.last_model_path) {
    const g = state.localGroups.find((x) => x.Files.some((f) => f.Path === state.cfg.last_model_path));
    if (g) selectModel(state.cfg.last_model_path, g.Name, true);
  }
  if (!state.model && state.localGroups.length === 1) {
    selectModel(state.localGroups[0].Files[0].Path, state.localGroups[0].Name, true);
  }
  renderLocalModels();
  renderPicker();
}
$('refresh-local').addEventListener('click', async () => { await loadLocalModels(); toast('Model folders rescanned.'); });

function modelBadges(g) {
  const q = quantOf(g.Files[0].Path.split(/[\\/]/).pop());
  const fit = fitOf(g.TotalSize);
  return [q && badge(q, 'accent'), g.Files.length > 1 && badge(`${g.Files.length} parts`), fit && badge(fit.label, fit.kind)];
}

function renderLocalModels() {
  marquee($('models-marquee'), state.localGroups.length
    ? state.localGroups.map((g) => [g.Name, fmtBytes(g.TotalSize)])
    : [['No models yet', 'download one below']], 25);
  const box = $('local-models');
  box.textContent = '';
  $('local-count').textContent = state.localGroups.length ? `${state.localGroups.length} model${state.localGroups.length > 1 ? 's' : ''}` : '';
  if (state.localGroups.length === 0) {
    const go = button('Search Hugging Face', () => $('hf-query').focus(), 'btn-primary btn-sm');
    box.appendChild(empty('No models on this PC yet', 'Download a GGUF model below, or add a folder that has some in Settings.', go));
    return;
  }
  for (const g of state.localGroups) {
    const path = g.Files[0].Path;
    const selected = state.model && state.model.path === path;
    const use = selected
      ? badge('selected', 'ok')
      : button('Use this model', () => { selectModel(path, g.Name); showPage('home'); }, 'btn-primary btn-sm');
    box.appendChild(item({ title: g.Name, badges: modelBadges(g), meta: `${fmtBytes(g.TotalSize)} · ${path}`, actions: [use] }));
  }
}

// Run page picker: the same list as radio-style rows (recognition rather than recall)
function renderPicker() {
  const box = $('model-picker');
  box.textContent = '';
  if (state.localGroups.length === 0) {
    box.appendChild(empty('No model yet', 'Download one from Hugging Face to get started.',
      button('Find a model', () => showPage('models'), 'btn-primary btn-sm')));
    return;
  }
  for (const g of state.localGroups) {
    const path = g.Files[0].Path;
    const row = item({ title: g.Name, badges: modelBadges(g), meta: `${fmtBytes(g.TotalSize)} · ${path}`, lead: el('span', 'radio-dot') });
    row.classList.add('selectable');
    row.tabIndex = 0;
    row.setAttribute('role', 'radio');
    const on = state.model && state.model.path === path;
    row.setAttribute('aria-checked', on ? 'true' : 'false');
    if (on) row.classList.add('selected');
    const pick = () => { if (!on) selectModel(path, g.Name); };
    row.addEventListener('click', pick);
    row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    box.appendChild(row);
  }
  box.setAttribute('role', 'radiogroup');
  box.setAttribute('aria-label', 'Model');
}

function selectModel(path, label, quiet) {
  state.model = { path, label };
  renderPicker();
  renderLocalModels();
  renderHero();
  if (!quiet) toast(`${label} selected.`);
  computeTune();
}

// Hugging Face
$('hf-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = $('hf-query').value.trim();
  const box = $('hf-results');
  box.replaceChildren(el('div', 'skeleton'));
  try {
    const res = await api('GET', '/api/hf/search?q=' + encodeURIComponent(q));
    box.textContent = '';
    if (res.length === 0) {
      box.appendChild(empty('No GGUF models found', 'Try a shorter name, or the repository of the model you want.'));
      return;
    }
    for (const r of res) {
      box.appendChild(item({
        title: r.id,
        badges: [badge(`♥ ${r.likes || 0}`)],
        actions: [button('Show files', () => listFiles(r.id))],
      }));
    }
  } catch (err) {
    box.replaceChildren(callout('Search failed: ' + err.message + ' (is the internet reachable?)', 'error'));
  }
});

$('hf-repo-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const repo = $('hf-repo').value.trim();
  if (repo) listFiles(repo);
});

async function listFiles(repo) {
  $('hf-repo').value = repo;
  const wrap = $('hf-files-wrap');
  const box = $('hf-files');
  wrap.hidden = false;
  $('hf-files-title').textContent = repo;
  box.replaceChildren(el('div', 'skeleton'));
  try {
    const files = await api('GET', '/api/hf/files?repo=' + encodeURIComponent(repo));
    box.textContent = '';
    if (files.length === 0) {
      box.appendChild(empty('No GGUF files here', 'This repository has no .gguf files. Search for a "GGUF" version of the model.'));
      return;
    }
    // group split shards so one click fetches the whole model
    const groups = new Map();
    for (const f of files) {
      const key = f.rfilename.replace(/-\d{5}-of-\d{5}\.gguf$/i, '');
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(f);
    }
    const have = new Set(state.localGroups.flatMap((g) => g.Files.map((f) => f.Path.split(/[\\/]/).pop())));
    for (const [key, fs] of [...groups].sort((a, b) => sizeOf(a[1]) - sizeOf(b[1]))) {
      const size = sizeOf(fs);
      const fit = fitOf(size);
      const name = fs.length > 1 ? key.split('/').pop() : fs[0].rfilename.split('/').pop();
      const downloaded = fs.every((f) => have.has(f.rfilename.split('/').pop()));
      const action = downloaded ? badge('on this PC', 'ok') : button('Download', (e) => download(repo, fs, name, e.currentTarget), 'btn-primary btn-sm');
      box.appendChild(item({
        title: name,
        badges: [quantOf(name) && badge(quantOf(name), 'accent'), fs.length > 1 && badge(`${fs.length} parts`), fit && badge(fit.label, fit.kind)],
        meta: size ? fmtBytes(size) : '',
        actions: [action],
      }));
    }
    wrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    box.replaceChildren(callout(`Could not list files of ${repo}: ${err.message}`, 'error'));
  }
}
const sizeOf = (fs) => fs.reduce((a, f) => a + (f.size || 0), 0);

async function download(repo, fs, name, btn) {
  btn.disabled = true;
  for (const f of fs) {
    try {
      await api('POST', '/api/hf/download', { repo, filename: f.rfilename });
    } catch (e) {
      toast(`Download of ${name} failed to start: ${e.message}`, 'error');
      btn.disabled = false;
      return;
    }
  }
  btn.textContent = 'Downloading…';
  toast(`Downloading ${name}. Progress is shown under Downloads.`);
  refreshDownloads();
}

const dlSpeed = {}; // key -> {bytes, t, rate}
let hadActiveDownloads = false;
async function refreshDownloads() {
  let list;
  try { list = await api('GET', '/api/hf/downloads'); } catch (_) { return; }
  const card = $('downloads-card');
  card.hidden = list.length === 0;
  const box = $('downloads');
  box.textContent = '';
  list.sort((a, b) => (a.started_at < b.started_at ? 1 : -1));
  const now = Date.now();
  for (const d of list) {
    const key = d.repo + '/' + d.filename;
    const pct = d.total_bytes ? (d.done_bytes / d.total_bytes) * 100 : (d.state === 'done' ? 100 : 0);
    const prev = dlSpeed[key];
    if (prev && d.state === 'active' && now > prev.t) {
      const r = (d.done_bytes - prev.bytes) / ((now - prev.t) / 1000);
      prev.rate = prev.rate ? prev.rate * 0.7 + r * 0.3 : r;
    }
    dlSpeed[key] = { bytes: d.done_bytes, t: now, rate: prev && prev.rate };
    const rate = dlSpeed[key].rate;
    let meta = `${fmtBytes(d.done_bytes)}${d.total_bytes ? ' of ' + fmtBytes(d.total_bytes) : ''}`;
    if (d.state === 'active' && rate > 0) {
      meta += ` · ${fmtBytes(rate)}/s`;
      if (d.total_bytes) meta += ` · ${fmtDuration((d.total_bytes - d.done_bytes) / rate)} left`;
    }
    if (d.error) meta += ' · ' + d.error;
    const kinds = { done: 'ok', active: 'accent', pending: '', error: 'bad', failed: 'bad' };
    const bar = el('div', 'progress dl-progress');
    const fill = el('div');
    fill.style.width = pct.toFixed(1) + '%';
    bar.appendChild(fill);
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuenow', pct.toFixed(0));
    bar.setAttribute('aria-label', d.filename);
    box.appendChild(item({
      title: d.filename.split('/').pop(),
      badges: [badge(d.state === 'active' ? `${pct.toFixed(0)}%` : d.state, kinds[d.state])],
      meta: `${d.repo} · ${meta}`,
      actions: [d.state === 'active' || d.state === 'pending' ? bar : null],
    }));
  }
  const active = list.some((d) => d.state === 'active' || d.state === 'pending');
  if (hadActiveDownloads && !active) {
    const failed = list.filter((d) => d.error);
    toast(failed.length ? `${failed.length} download(s) failed; see Models.` : 'Download finished. The model is ready to use.', failed.length ? 'error' : 'ok');
    loadLocalModels();
  }
  hadActiveDownloads = active;
}
setInterval(refreshDownloads, 1500);

// ---- run: mode ------------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  document.body.classList.toggle('mode-single', mode !== 'ondemand');
  document.body.classList.toggle('mode-ondemand', mode === 'ondemand');
  document.querySelectorAll('.seg-btn').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  renderHero();
  renderGPUs();
}

document.querySelectorAll('.seg-btn').forEach((b) => b.addEventListener('click', async () => {
  const mode = b.dataset.mode;
  if (mode === state.mode) return;
  const cfg = JSON.parse(JSON.stringify(state.cfg));
  cfg.on_demand = mode === 'ondemand';
  try { await saveConfig(cfg); } catch (e) { toast(e.message, 'error'); return; }
  setMode(mode);
  if (state.status.Running && state.status.OnDemand !== cfg.on_demand) {
    toast('The new mode applies after Stop and Start.', 'warn');
  }
}));
// arrow keys move between the two options, like native radio buttons
document.querySelector('.seg').addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const btns = [...document.querySelectorAll('.seg-btn')];
  const next = btns.find((b) => b.dataset.mode !== state.mode);
  next.click();
  next.focus();
});

// ---- run: plan ------------------------------------------------------------

let tuneSeq = 0;
async function computeTune() {
  const out = $('plan');
  if (!state.model) return;
  const seq = ++tuneSeq;
  out.replaceChildren(el('div', 'skeleton'));
  try {
    const resp = await api('POST', '/api/tune', { model_path: state.model.path });
    if (seq !== tuneSeq) return;
    state.plan = resp;
    $('tune-args').value = joinArgs(resp.args);
    renderPlan();
    renderGPUs();
  } catch (e) {
    if (seq !== tuneSeq) return;
    state.plan = null;
    out.replaceChildren(callout('Could not read this model: ' + e.message, 'error'));
  }
}
$('retune').addEventListener('click', computeTune);

function renderPlan() {
  const out = $('plan');
  const p = state.plan.plan;
  out.textContent = '';

  // one line in plain words first, details after (progressive disclosure)
  let summary;
  if (p.FullyOnGPU) summary = 'Everything fits on the GPU: fastest setup.';
  else if (!p.Devices || !p.Devices.length || p.NGpuLayers === 0) summary = 'Runs on the CPU only.';
  else if (p.NCPUMoE > 0 && p.NGpuLayers >= p.NCPUMoE) summary = `Most of the model runs on the GPU; the experts of ${p.NCPUMoE} layer${p.NCPUMoE > 1 ? 's' : ''} stay in RAM.`;
  else summary = 'Split between GPU and CPU: expect slower answers.';
  out.appendChild(callout(summary, p.NGpuLayers === 0 && state.gpus.length ? 'warn' : 'info'));

  if (p.Devices && p.Devices.length) {
    const devs = el('div', 'plan-devices');
    p.Devices.forEach((d, i) => {
      const g = state.gpus.find((x) => x.Index === d);
      const total = g ? g.TotalBytes : 0;
      const box = el('div', 'plan-dev');
      const name = el('div', 'plan-dev-name');
      name.append(el('span', '', p.DeviceNames[i] || `GPU ${d}`), el('span', 'muted', fmtBytes(p.DeviceBytes[i])));
      box.appendChild(name);
      if (total) {
        const m = el('div', 'meter');
        const bar = el('span', 'm-model');
        bar.style.width = Math.min(100, p.DeviceBytes[i] / total * 100) + '%';
        m.appendChild(bar);
        box.appendChild(m);
      }
      box.appendChild(el('div', 'muted small', p.TensorSplit ? `${p.TensorSplit[i]} layer slot(s)` : `${p.NGpuLayers} layer slot(s)`));
      devs.appendChild(box);
    });
    out.appendChild(devs);
  }

  // the numbers that matter, at poster scale
  const stats = el('div', 'stats');
  const stat = (label, value, unit, text) => {
    const t = el('div', 'stat');
    const v = el('div', 'stat-value' + (text ? ' stat-text' : ''), value);
    if (unit) v.appendChild(el('small', '', unit));
    t.append(v, el('div', 'stat-label', label));
    stats.appendChild(t);
  };
  const [ctxNum, ctxUnit] = p.CtxSize >= 1024 ? [Math.round(p.CtxSize / 1024), 'K'] : [p.CtxSize, ''];
  stat('Tokens of context', String(ctxNum), ctxUnit);
  stat('On GPU', ...splitBytes(p.GPUFitBytes));
  stat('In RAM', ...splitBytes(p.CPUFitBytes));
  stat('KV cache', p.CacheTypeKV);
  if (p.Parallel > 1) stat('Parallel slots', String(p.Parallel));
  stat('Thinking control', state.plan.reasoning || 'none', '', true);
  out.appendChild(stats);

  const notes = (p.Notes || []).filter(Boolean);
  if (notes.length) {
    const box = el('div', 'notes');
    for (const n of notes) {
      const bad = /^warning|too small|too large|very slow|can't hold/i.test(n);
      const warn = /slow|close other|missing|unavailable/i.test(n);
      box.appendChild(callout(n, bad ? 'error' : warn ? 'warn' : 'info'));
    }
    out.appendChild(box);
  }
}

// ---- run: start / stop / status ---------------------------------------------

$('start-server').addEventListener('click', async () => {
  const btn = $('start-server');
  if (state.mode !== 'ondemand' && !state.model) {
    toast('Choose a model first.', 'warn');
    return;
  }
  btn.disabled = true;
  try {
    if (state.mode === 'ondemand') {
      await api('POST', '/api/server/start', { on_demand: true });
    } else {
      await api('POST', '/api/server/start', {
        model_path: state.model.path,
        args: splitArgs($('tune-args').value),
        devices: state.plan ? state.plan.plan.Devices : [],
      });
    }
    $('server-log').textContent = '';
    state.wasReady = false;
    await refreshServerStatus();
  } catch (e) {
    toast('Could not start: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

$('stop-server').addEventListener('click', async () => {
  try { await api('POST', '/api/server/stop'); toast('Model stopped. Its memory is free again.'); }
  catch (e) { toast('Could not stop: ' + e.message, 'error'); }
  refreshServerStatus();
});

const openChat = () => api('POST', '/api/open').catch((e) => toast(e.message, 'error'));
$('open-webui').addEventListener('click', openChat);
$('side-chat').addEventListener('click', openChat);

$('copy-api').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('api-url').textContent); toast('API address copied.', 'ok', 2000); }
  catch (_) { toast('Copy failed; select the address and copy it manually.', 'warn'); }
});

$('quit').addEventListener('click', async () => {
  const running = state.status.Running;
  if (!(await confirmDialog('Quit Q38FNInference?', running ? 'The running model will be stopped and API clients disconnected.' : 'The app and its API will close.', 'Quit', true))) return;
  try { await api('POST', '/api/quit'); } catch (_) {}
  document.body.innerHTML = '<div class="exited"><div><h1>Q38FNInference has closed</h1><p>You can close this window. Start the app again to reopen it.</p></div></div>';
});

async function refreshServerStatus() {
  let st;
  try { st = await api('GET', '/api/server/status'); } catch (_) { return; }
  state.status = st;
  if (st.Ready && !state.wasReady && st.Running) {
    toast(st.OnDemand ? 'Server ready. Models load on first request.' : 'Model loaded and ready.', 'ok');
  }
  state.wasReady = st.Running && st.Ready;
  if (st.OnDemand && st.Ready) refreshRouterModels();
  renderHero();
}
setInterval(refreshServerStatus, 1500);

function runningModelName(st) {
  if (st.OnDemand) return 'All local models (on demand)';
  const i = (st.Args || []).indexOf('--model');
  const path = i >= 0 ? st.Args[i + 1] : '';
  const g = state.localGroups.find((x) => x.Files.some((f) => f.Path === path));
  return g ? g.Name : path.split(/[\\/]/).pop() || 'model';
}

// one place decides what the status looks like, used by the hero and the sidebar
function renderHero() {
  const st = state.status || {};
  let dot = '', title, detail, side;
  if (st.Running && st.Ready) {
    dot = 'ok'; title = st.OnDemand ? 'On demand' : 'Ready';
    detail = st.OnDemand ? 'Requests load the model they name.' : `${runningModelName(st)} is answering requests.`;
    side = 'Ready';
  } else if (st.Running) {
    dot = 'busy'; title = st.OnDemand ? 'Starting' : 'Loading';
    const secs = st.StartedAt ? (Date.now() - Date.parse(st.StartedAt)) / 1000 : 0;
    detail = `${runningModelName(st)} · ${fmtDuration(secs)}${state.lastLog ? ' · ' + state.lastLog.slice(0, 90) : ''}`;
    side = 'Loading…';
  } else if (st.ExitErr) {
    dot = 'bad'; title = 'Crashed';
    detail = `The model stopped unexpectedly (${st.ExitErr}). The server log below has the details.`;
    side = 'Stopped (error)';
  } else {
    title = 'Stopped';
    detail = state.mode === 'ondemand' ? 'Start to offer every local model through the API.'
      : state.model ? state.model.label : 'Choose a model below.';
    side = 'Stopped';
  }
  $('hero').dataset.state = { ok: 'ready', busy: 'busy', bad: 'error' }[dot] || 'idle';
  $('hero-dot').className = 'dot dot-lg ' + dot;
  $('hero-title').textContent = title;
  $('hero-detail').textContent = detail;
  $('hero-progress').hidden = !(st.Running && !st.Ready);
  $('start-server').hidden = !!st.Running;
  $('stop-server').hidden = !st.Running;
  $('open-webui').disabled = !st.Ready;
  $('bench-run').disabled = !st.Ready;

  $('side-dot').className = 'dot ' + dot;
  $('side-state').textContent = side;
  $('side-model').textContent = st.Running ? runningModelName(st) : 'No model running';
  $('side-chat').disabled = !st.Ready;
  if (st.ExitErr && !st.Running) $('log-card').open = true;
  document.title = (st.Ready ? '● ' : st.Running ? '○ ' : '') + 'Q38FNInference';
  renderStatusMarquee();
}

// the status ticker restates the essentials in poster type; it is aria-hidden,
// everything in it is also on the page as plain text
function renderStatusMarquee() {
  const st = state.status || {};
  const items = [st.Running ? (st.Ready ? 'Ready' : 'Loading') : st.ExitErr ? 'Crashed' : 'Stopped'];
  if (st.Running) items.push(runningModelName(st));
  else if (state.mode === 'ondemand') items.push('All models on demand');
  else if (state.model) items.push(state.model.label);
  const p = state.mode === 'single' && state.plan && state.plan.plan;
  if (p) {
    items.push(`${Number(p.CtxSize).toLocaleString()} tokens context`, `${fmtBytes(p.GPUFitBytes)} on GPU`, `${fmtBytes(p.CPUFitBytes)} in RAM`);
  }
  const gpus = selectedGPUs();
  items.push(gpus.length ? gpus.map((g) => g.Name).join(' + ') : 'CPU only');
  if (state.info.api_url) items.push('API ' + state.info.api_url.replace(/^https?:\/\//, ''));
  marquee($('status-marquee'), items, 60);
}

// ---- run: on-demand models ------------------------------------------------

$('idle-unload').addEventListener('change', async () => {
  const cfg = JSON.parse(JSON.stringify(state.cfg));
  cfg.idle_unload_minutes = Math.max(0, parseInt($('idle-unload').value, 10) || 0);
  try { await saveConfig(cfg); toast('Saved. Applies on the next start.', 'ok', 2500); } catch (e) { toast(e.message, 'error'); }
});

async function refreshRouterModels() {
  const box = $('router-models');
  let list;
  try { list = (await api('GET', '/models')).data || []; } catch (_) { return; }
  box.textContent = '';
  if (list.length === 0) {
    box.appendChild(empty('No models to offer', 'Download a model and it will appear here.', button('Find a model', () => showPage('models'), 'btn-primary btn-sm')));
  }
  for (const m of list) {
    const st = m.status || {};
    const status = st.value || 'unknown';
    const failed = st.failed && status === 'unloaded';
    const kinds = { loaded: 'ok', loading: 'warn', sleeping: '', unloaded: '', downloading: 'warn' };
    const b = failed ? badge(`failed${st.exit_code != null ? ' (exit ' + st.exit_code + ')' : ''}`, 'bad') : badge(status, kinds[status]);
    const busy = status === 'loading' || status === 'downloading';
    const loaded = status === 'loaded' || status === 'sleeping';
    const btn = loaded
      ? button('Unload', () => routerAction('unload', m.id))
      : button('Load', () => routerAction('load', m.id), 'btn-primary btn-sm');
    btn.disabled = busy;
    box.appendChild(item({ title: m.id, badges: [b], meta: failed ? 'Could not load; see the server log.' : '', actions: [btn] }));
  }
  const sel = $('bench-model');
  const current = sel.value;
  sel.replaceChildren(...list.map((m) => { const o = el('option', '', m.id); o.value = m.id; return o; }));
  const loadedOne = list.find((m) => m.status && m.status.value === 'loaded');
  sel.value = list.some((m) => m.id === current) ? current : (loadedOne ? loadedOne.id : (list[0] || {}).id || '');
  try {
    const n = (await api('GET', '/api/router/notes')).notes;
    const nb = $('router-notes');
    nb.hidden = !n.length;
    if (n.length) nb.replaceChildren(icon('info'), el('div', '', 'Not offered:\n' + n.join('\n')));
  } catch (_) { /* optional */ }
}

async function routerAction(action, model) {
  try {
    await api('POST', '/models/' + action, { model });
    toast(action === 'load' ? `Loading ${model}…` : `${model} unloaded.`);
  } catch (e) {
    toast(`Could not ${action} ${model}: ${e.message}`, 'error');
  }
  refreshRouterModels();
}

$('router-rescan').addEventListener('click', async () => {
  try {
    await api('POST', '/api/router/rescan');
    toast('Rescanning model folders…');
    setTimeout(refreshRouterModels, 1500);
  } catch (e) {
    toast(e.message, 'error');
  }
});

// ---- speed test -------------------------------------------------------------

// the launch settings that matter for speed, pulled from the running args
function speedSettings(args) {
  const keep = ['--n-gpu-layers', '--n-cpu-moe', '--tensor-split', '--ubatch-size', '--ctx-size', '--cache-type-k', '--threads', '--no-op-offload'];
  const out = [];
  for (let i = 0; i < args.length; i++) {
    if (!keep.includes(args[i])) continue;
    const v = args[i + 1] && !args[i + 1].startsWith('--') ? ' ' + args[i + 1] : '';
    out.push(args[i].replace(/^--/, '') + v);
  }
  return out.join(', ');
}

$('bench-run').addEventListener('click', async () => {
  const btn = $('bench-run');
  btn.disabled = true;
  $('bench-status').textContent = 'Running… a large model can take a minute.';
  try {
    const st = state.status;
    const body = st.OnDemand ? { model: $('bench-model').value } : {};
    const r = await api('POST', '/api/benchmark', body);
    const tr = document.createElement('tr');
    for (const [v, cls] of [
      [new Date().toLocaleTimeString(), ''],
      [r.model || runningModelName(st), ''],
      [`${r.prompt_per_second.toFixed(0)} tok/s`, 'num'],
      [`${r.generated_per_second.toFixed(1)} tok/s`, 'num'],
      [st.OnDemand ? 'on demand' : speedSettings(st.Args || []), 'args'],
    ]) tr.appendChild(el('td', cls, v));
    $('bench-body').prepend(tr);
    $('bench-wrap').hidden = false;
    $('bench-status').textContent = `Done in ${r.wall_seconds.toFixed(1)} s.`;
  } catch (e) {
    $('bench-status').textContent = '';
    toast('Speed test failed: ' + e.message, 'error');
  } finally {
    btn.disabled = !state.status.Ready;
  }
});

// ---- log --------------------------------------------------------------------

function streamLogs() {
  const box = $('server-log');
  const es = new EventSource('/api/server/logs/stream');
  es.onmessage = (ev) => {
    const line = JSON.parse(ev.data);
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
    box.textContent += line + '\n';
    if (box.textContent.length > 400000) box.textContent = box.textContent.slice(-300000);
    if (atBottom) box.scrollTop = box.scrollHeight;
    const clean = line.replace(/^\S+:\s*/, '').trim();
    if (clean) { state.lastLog = clean; $('log-last').textContent = clean; }
  };
  es.onerror = () => { es.close(); setTimeout(streamLogs, 2000); };
}
$('log-copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('server-log').textContent); toast('Log copied.', 'ok', 2000); }
  catch (_) { toast('Copy failed.', 'warn'); }
});
$('log-clear').addEventListener('click', () => { $('server-log').textContent = ''; });

function remeasureMarquees() {
  document.querySelectorAll('.page.active .marquee').forEach((m) => {
    if (m._args) { m.dataset.key = ''; marquee(m, ...m._args); }
  });
}

// ---- scroll motion -------------------------------------------------------------

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let scrollTick = false;
function onScroll() {
  if (scrollTick) return;
  scrollTick = true;
  requestAnimationFrame(() => {
    scrollTick = false;
    const h1 = document.querySelector('.page.active .page-head h1');
    if (!h1 || reducedMotion.matches) return;
    const p = Math.min(window.scrollY / 360, 1);
    h1.style.transform = `scale(${1 + p * 0.2})`;
    h1.style.opacity = String(1 - p * 0.75);
  });
}
window.addEventListener('scroll', onScroll, { passive: true });
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  // re-measure marquees for the new width
  resizeTimer = setTimeout(remeasureMarquees, 200);
});

// ---- init -------------------------------------------------------------------

(async function init() {
  showPage(location.hash.slice(1) || 'home');
  try {
    state.cfg = await api('GET', '/api/config');
    fillForm(state.cfg);
    setMode(state.cfg.on_demand ? 'ondemand' : 'single');
    $('idle-unload').value = state.cfg.idle_unload_minutes || 0;
  } catch (e) {
    toast('Could not load settings: ' + e.message, 'error');
  }
  try {
    state.info = await api('GET', '/api/info');
    $('api-url').textContent = state.info.api_url + '/v1';
    if (state.info.api_error) {
      $('api-error').hidden = false;
      $('api-error').replaceChildren(icon('error'), el('div', '', 'The API port could not be opened: ' + state.info.api_error + '. Change the port in Settings and restart the app.'));
    }
  } catch (_) { /* shown by other calls */ }
  await loadGPUs();
  await loadLocalModels();
  refreshDownloads();
  refreshServerStatus();
  streamLogs();
})();
