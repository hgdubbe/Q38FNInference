'use strict';

const state = {
  cfg: null,
  model: null,   // { path, label }
  plan: null,    // last /api/tune response
  gpus: [],
  chat: [],      // [{role, content}]
  abort: null,
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
  if (!resp.ok) throw new Error((data && data.error) || resp.statusText);
  return data;
}

function fmtBytes(n) {
  if (!n) return '0 B';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + ' ' + u[i];
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function item(name, meta, ...right) {
  const it = el('div', 'item');
  const left = el('div');
  left.appendChild(el('div', 'name', name));
  if (meta) left.appendChild(el('div', 'meta', meta));
  it.appendChild(left);
  right.forEach((r) => it.appendChild(r));
  return it;
}

function button(text, onClick, cls) {
  const b = el('button', cls || '', text);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
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

// ---- tabs -----------------------------------------------------------------

function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
}
document.querySelectorAll('.tab-btn').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ---- settings -------------------------------------------------------------

const form = $('settings-form');
const numberFields = new Set(['port', 'model.ctx_size', 'model.parallel', 'model.threads', 'model.batch_size',
  'model.ubatch_size', 'model.temperature', 'model.top_p', 'model.top_k', 'model.min_p', 'model.repeat_penalty',
  'model.presence_penalty', 'model.max_tokens', 'model.seed']);

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
    else if (f.name === 'extra_model_dirs') f.value = (v || []).join('\n');
    else if (f.tagName === 'SELECT') {
      // unset or legacy values (e.g. reasoning "on") fall back to the marked default
      const ok = [...f.options].some((o) => o.value === String(v));
      const def = f.querySelector('option[selected]') || f.options[0];
      f.value = v != null && v !== '' && ok ? v : def.value;
    } else f.value = v == null ? '' : v;
  }
}

function readForm() {
  const cfg = JSON.parse(JSON.stringify(state.cfg || {}));
  cfg.model = cfg.model || {};
  for (const f of form.elements) {
    if (!f.name) continue;
    let v;
    if (f.type === 'checkbox') v = f.checked;
    else if (f.name === 'extra_model_dirs') v = f.value.split('\n').map((s) => s.trim()).filter(Boolean);
    else if (numberFields.has(f.name)) v = f.value === '' ? undefined : Number(f.value);
    else v = f.value.trim() === '' ? undefined : f.value.trim();
    if (f.name === 'system_prompt' && v !== undefined) v = f.value; // keep formatting
    setPath(cfg, f.name, v);
  }
  return cfg;
}

async function saveConfig(cfg) {
  state.cfg = await api('POST', '/api/config', cfg);
  renderSystemHint();
  return state.cfg;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await saveConfig(readForm());
    $('settings-status').textContent = 'Saved. Model settings apply on the next Start.';
    computeTune();
  } catch (err) {
    $('settings-status').textContent = 'Error: ' + err.message;
  }
});

function renderSystemHint() {
  const c = state.cfg || {};
  $('chat-sys').textContent = c.system_prompt && c.system_prompt_mode !== 'off'
    ? `System prompt (${c.system_prompt_mode}): ${c.system_prompt.slice(0, 120)}${c.system_prompt.length > 120 ? '…' : ''}`
    : '';
}

// ---- GPUs -----------------------------------------------------------------

async function loadGPUs() {
  const box = $('gpu-list');
  try {
    state.gpus = await api('GET', '/api/gpus');
  } catch (e) {
    box.textContent = 'Error detecting GPUs: ' + e.message;
    return;
  }
  box.textContent = '';
  if (state.gpus.length === 0) {
    box.appendChild(el('div', 'hint', 'No NVIDIA GPU detected (nvidia-smi not found) — models will run on CPU.'));
    return;
  }
  const sel = (state.cfg && state.cfg.gpus) || [];
  for (const g of state.gpus) {
    const cb = el('input');
    cb.type = 'checkbox';
    cb.checked = sel.length === 0 || sel.includes(g.Index);
    cb.addEventListener('change', onGPUToggle);
    cb.dataset.index = g.Index;
    const lbl = el('label', 'inline name');
    lbl.append(cb, `GPU ${g.Index}: ${g.Name}`);
    const row = el('div', 'item');
    row.append(lbl, el('div', 'meta', `${fmtBytes(g.FreeBytes)} free of ${fmtBytes(g.TotalBytes)}`));
    box.appendChild(row);
  }
}

async function onGPUToggle() {
  const boxes = [...$('gpu-list').querySelectorAll('input[type=checkbox]')];
  const chosen = boxes.filter((b) => b.checked).map((b) => Number(b.dataset.index));
  const cfg = JSON.parse(JSON.stringify(state.cfg));
  // all selected == "use every GPU", which also covers GPUs added later
  cfg.gpus = chosen.length === boxes.length ? [] : chosen;
  if (chosen.length === 0) cfg.gpus = [-1];
  await saveConfig(cfg);
  computeTune();
}

// ---- models ---------------------------------------------------------------

async function loadLocalModels() {
  const box = $('local-models');
  box.textContent = 'Scanning…';
  try {
    const groups = await api('GET', '/api/models/local');
    box.textContent = '';
    if (groups.length === 0) box.appendChild(el('div', 'hint', 'No GGUF models found yet — download one below.'));
    for (const g of groups) {
      const shards = g.Files.length > 1 ? `${g.Files.length} shards, ` : '';
      const file = g.Files[0].Path;
      box.appendChild(item(g.Name, `${shards}${fmtBytes(g.TotalSize)} — ${file}`,
        button('Use', () => selectModel(file, g.Name))));
    }
    if (!state.model && state.cfg && state.cfg.last_model_path) {
      const last = groups.flatMap((g) => g.Files).find((f) => f.Path === state.cfg.last_model_path);
      if (last) selectModel(last.Path, last.Repo || last.Path, true);
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
}
$('refresh-local').addEventListener('click', loadLocalModels);

$('hf-search').addEventListener('click', async () => {
  const q = $('hf-query').value.trim();
  const box = $('hf-results');
  box.textContent = 'Searching…';
  try {
    const res = await api('GET', '/api/hf/search?q=' + encodeURIComponent(q));
    box.textContent = '';
    if (res.length === 0) box.appendChild(el('div', 'hint', 'No results.'));
    for (const r of res) {
      box.appendChild(item(r.id, `♥ ${r.likes || 0}`, button('Files', () => {
        $('hf-repo').value = r.id;
        $('hf-list-files').click();
      }, 'secondary')));
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
});
$('hf-query').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('hf-search').click(); });

$('hf-list-files').addEventListener('click', async () => {
  const repo = $('hf-repo').value.trim();
  const box = $('hf-files');
  if (!repo) return;
  box.textContent = 'Loading…';
  try {
    const files = await api('GET', '/api/hf/files?repo=' + encodeURIComponent(repo));
    box.textContent = '';
    if (files.length === 0) box.appendChild(el('div', 'hint', 'No .gguf files in this repo.'));
    // group split shards so one click fetches the whole model
    const groups = new Map();
    for (const f of files) {
      const key = f.rfilename.replace(/-\d{5}-of-\d{5}\.gguf$/i, '');
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(f);
    }
    for (const [key, fs] of groups) {
      const size = fs.reduce((a, f) => a + (f.size || 0), 0);
      const label = fs.length > 1 ? `${key} (${fs.length} shards)` : fs[0].rfilename;
      box.appendChild(item(label, size ? fmtBytes(size) : '', button('Download', async () => {
        for (const f of fs) {
          try { await api('POST', '/api/hf/download', { repo, filename: f.rfilename }); }
          catch (e) { alert('Download failed to start: ' + e.message); return; }
        }
        refreshDownloads();
      })));
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
});

let hadActiveDownloads = false;
async function refreshDownloads() {
  const box = $('downloads');
  try {
    const list = await api('GET', '/api/hf/downloads');
    box.textContent = '';
    if (list.length === 0) box.appendChild(el('div', 'hint', 'No downloads yet.'));
    list.sort((a, b) => (a.started_at < b.started_at ? 1 : -1));
    for (const d of list) {
      const pct = d.total_bytes ? (d.done_bytes / d.total_bytes) * 100 : (d.state === 'done' ? 100 : 0);
      const bar = el('div', 'progress');
      const fill = el('div');
      fill.style.width = pct.toFixed(1) + '%';
      bar.appendChild(fill);
      const size = `${fmtBytes(d.done_bytes)}${d.total_bytes ? ' / ' + fmtBytes(d.total_bytes) : ''}`;
      box.appendChild(item(`${d.repo}/${d.filename}`, `${d.state} — ${size}${d.error ? ' — ' + d.error : ''}`, bar));
    }
    const active = list.some((d) => d.state === 'active' || d.state === 'pending');
    if (hadActiveDownloads && !active) loadLocalModels();
    hadActiveDownloads = active;
  } catch (_) { /* transient */ }
}
setInterval(refreshDownloads, 1500);

// ---- run ------------------------------------------------------------------

function selectModel(path, label, quiet) {
  state.model = { path, label };
  $('selected-model').textContent = `${label}\n${path}`;
  if (!quiet) showTab('run');
  computeTune();
}

async function computeTune() {
  const out = $('tune-plan');
  if (!state.model) return;
  out.textContent = 'Reading model and computing plan…';
  try {
    const resp = await api('POST', '/api/tune', { model_path: state.model.path });
    state.plan = resp;
    const p = resp.plan;
    const lines = [];
    if (p.Devices && p.Devices.length) {
      p.Devices.forEach((d, i) => {
        const layers = p.TensorSplit ? ` — ${p.TensorSplit[i]} slot(s)` : '';
        lines.push(`GPU ${d} ${p.DeviceNames[i]}: ${fmtBytes(p.DeviceBytes[i])}${layers}`);
      });
    }
    lines.push(`GPU layers: ${p.NGpuLayers}` + (p.NCPUMoE ? `, experts of the first ${p.NCPUMoE} blocks in CPU RAM` : ''));
    lines.push(`context: ${p.CtxSize} tokens, KV cache ${p.CacheTypeKV}, ${p.Parallel} slot(s)`);
    lines.push(`reasoning controls: ${resp.reasoning}`);
    lines.push(`on GPU ${fmtBytes(p.GPUFitBytes)} · in RAM ${fmtBytes(p.CPUFitBytes)} · total ${fmtBytes(p.TotalBytes)}`);
    (p.Notes || []).forEach((n) => lines.push('• ' + n));
    out.textContent = lines.join('\n');
    $('tune-args').value = joinArgs(resp.args);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}
$('retune').addEventListener('click', computeTune);

$('start-server').addEventListener('click', async () => {
  if (!state.model) { alert('Pick a model in the Models tab first.'); return; }
  const args = splitArgs($('tune-args').value);
  try {
    await api('POST', '/api/server/start', {
      model_path: state.model.path,
      args,
      devices: state.plan ? state.plan.plan.Devices : [],
    });
    $('server-log').textContent = '';
    refreshServerStatus();
  } catch (e) {
    alert('Failed to start: ' + e.message);
  }
});

$('stop-server').addEventListener('click', async () => {
  try { await api('POST', '/api/server/stop'); } catch (e) { alert('Failed to stop: ' + e.message); }
  refreshServerStatus();
});

$('open-webui').addEventListener('click', () => api('POST', '/api/open').catch(() => {}));

$('quit').addEventListener('click', async () => {
  if (!confirm('Stop the model and exit Q38FNInference?')) return;
  try { await api('POST', '/api/quit'); } catch (_) {}
  document.body.innerHTML = '<main><div class="panel"><h2>Q38FNInference has exited.</h2><p class="hint">You can close this tab.</p></div></main>';
});

async function refreshServerStatus() {
  try {
    const st = await api('GET', '/api/server/status');
    const pill = $('hdr-status');
    let label = 'stopped';
    if (st.Running) label = st.Ready ? 'ready' : 'loading';
    pill.textContent = label;
    pill.className = 'pill ' + (st.Running ? label : '');
    $('server-status').textContent = st.Running ? `${label} (pid ${st.PID})` : (st.ExitErr ? 'exited: ' + st.ExitErr : '');
    $('start-server').disabled = st.Running;
    $('stop-server').disabled = !st.Running;
    $('open-webui').disabled = !st.Ready;
    $('chat-send').disabled = !st.Ready;
  } catch (_) { /* transient */ }
}
setInterval(refreshServerStatus, 1500);

function streamLogs() {
  const box = $('server-log');
  const es = new EventSource('/api/server/logs/stream');
  es.onmessage = (ev) => {
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
    box.textContent += JSON.parse(ev.data) + '\n';
    if (box.textContent.length > 400000) box.textContent = box.textContent.slice(-300000);
    if (atBottom) box.scrollTop = box.scrollHeight;
  };
  es.onerror = () => { es.close(); setTimeout(streamLogs, 2000); };
}

// ---- chat -----------------------------------------------------------------

function renderChat() {
  const log = $('chat-log');
  log.textContent = '';
  for (const m of state.chat) {
    const div = el('div', 'msg ' + m.role);
    if (m.reasoning) div.appendChild(el('div', 'think', m.reasoning));
    div.appendChild(document.createTextNode(m.content || (m.role === 'assistant' && !m.done ? '…' : '')));
    log.appendChild(div);
  }
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const text = $('chat-input').value.trim();
  if (!text || state.abort) return;
  $('chat-input').value = '';
  state.chat.push({ role: 'user', content: text });
  const reply = { role: 'assistant', content: '', reasoning: '' };
  state.chat.push(reply);
  renderChat();

  const stream = $('chat-stream').checked;
  const messages = state.chat.filter((m) => m !== reply && m.role !== 'error').map((m) => ({ role: m.role, content: m.content }));
  state.abort = new AbortController();
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, stream }),
      signal: state.abort.signal,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error((err.error && (err.error.message || err.error)) || resp.statusText);
    }
    if (!stream) {
      const data = await resp.json();
      const msg = data.choices[0].message;
      reply.content = msg.content || '';
      reply.reasoning = msg.reasoning_content || '';
    } else {
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim();
          buf = buf.slice(i + 1);
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          const delta = (JSON.parse(payload).choices[0] || {}).delta || {};
          if (delta.content) reply.content += delta.content;
          if (delta.reasoning_content) reply.reasoning += delta.reasoning_content;
          renderChat();
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') state.chat.push({ role: 'error', content: 'Error: ' + e.message });
  } finally {
    reply.done = true;
    state.abort = null;
    renderChat();
  }
}

$('chat-send').addEventListener('click', sendChat);
$('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
$('chat-stop').addEventListener('click', () => state.abort && state.abort.abort());
$('chat-clear').addEventListener('click', () => { state.chat = []; renderChat(); });

// ---- init -----------------------------------------------------------------

(async function init() {
  try {
    state.cfg = await api('GET', '/api/config');
    fillForm(state.cfg);
    renderSystemHint();
    const info = await api('GET', '/api/info');
    $('api-url').textContent = info.api_url + '/v1';
    $('api-error').textContent = info.api_error || '';
  } catch (e) {
    $('settings-status').textContent = 'Error loading settings: ' + e.message;
  }
  loadGPUs();
  loadLocalModels();
  refreshDownloads();
  refreshServerStatus();
  streamLogs();
})();
