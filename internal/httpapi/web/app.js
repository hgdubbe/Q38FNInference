'use strict';

const state = {
  selectedModel: null, // absolute path on disk
  port: 8080,
};

// ---- tabs -------------------------------------------------------------

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ---- helpers ------------------------------------------------------------

async function api(method, path, body) {
  const resp = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    throw new Error((data && data.error) || resp.statusText);
  }
  return data;
}

function fmtBytes(n) {
  if (!n) return '0 B';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + ' ' + units[i];
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---- local models ---------------------------------------------------------

async function loadLocalModels() {
  const box = document.getElementById('local-models');
  box.textContent = 'Loading…';
  try {
    const groups = await api('GET', '/api/models/local');
    box.textContent = '';
    if (!groups || groups.length === 0) {
      box.appendChild(el('div', 'hint', 'No local GGUF models found yet.'));
      return;
    }
    for (const g of groups) {
      const item = el('div', 'item');
      const left = el('div');
      left.appendChild(el('div', 'name', g.Name));
      left.appendChild(el('div', 'meta', `${g.Files.length} file(s), ${fmtBytes(g.TotalSize)}`));
      item.appendChild(left);
      const btn = el('button', '', 'Select');
      btn.addEventListener('click', () => selectModel(g.Files[0].Path, g.Name));
      item.appendChild(btn);
      box.appendChild(item);
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
}

document.getElementById('refresh-local').addEventListener('click', loadLocalModels);

// ---- hugging face download ------------------------------------------------

document.getElementById('hf-list-files').addEventListener('click', async () => {
  const repo = document.getElementById('hf-repo').value.trim();
  const box = document.getElementById('hf-files');
  if (!repo) return;
  box.textContent = 'Loading…';
  try {
    const files = await api('GET', '/api/hf/files?repo=' + encodeURIComponent(repo));
    box.textContent = '';
    if (!files || files.length === 0) {
      box.appendChild(el('div', 'hint', 'No .gguf files found in this repo.'));
      return;
    }
    for (const f of files) {
      const item = el('div', 'item');
      const left = el('div');
      left.appendChild(el('div', 'name', f.rfilename));
      left.appendChild(el('div', 'meta', fmtBytes(f.size)));
      item.appendChild(left);
      const btn = el('button', '', 'Download');
      btn.addEventListener('click', () => startDownload(repo, f.rfilename));
      item.appendChild(btn);
      box.appendChild(item);
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
});

async function startDownload(repo, filename) {
  try {
    await api('POST', '/api/hf/download', { repo, filename });
    refreshDownloads();
  } catch (e) {
    alert('Download failed to start: ' + e.message);
  }
}

async function refreshDownloads() {
  const box = document.getElementById('downloads');
  try {
    const list = await api('GET', '/api/hf/downloads');
    box.textContent = '';
    if (!list || list.length === 0) {
      box.appendChild(el('div', 'hint', 'No downloads yet.'));
      return;
    }
    for (const d of list.sort((a, b) => (a.started_at < b.started_at ? 1 : -1))) {
      const item = el('div', 'item');
      const left = el('div');
      left.appendChild(el('div', 'name', `${d.repo}/${d.filename}`));
      const pct = d.total_bytes ? Math.round((d.done_bytes / d.total_bytes) * 100) : 0;
      left.appendChild(el('div', 'meta', `${d.state} — ${fmtBytes(d.done_bytes)}${d.total_bytes ? ' / ' + fmtBytes(d.total_bytes) : ''}${d.error ? ' — ' + d.error : ''}`));
      item.appendChild(left);
      const bar = el('div', 'progress');
      const fill = el('div');
      fill.style.width = pct + '%';
      bar.appendChild(fill);
      item.appendChild(bar);
      box.appendChild(item);
    }
  } catch (e) {
    box.textContent = 'Error: ' + e.message;
  }
}

setInterval(refreshDownloads, 1500);

// ---- run tab: selection + tuning -------------------------------------------

function selectModel(path, label) {
  state.selectedModel = path;
  document.getElementById('selected-model').textContent = (label || path) + '\n' + path;
  document.querySelectorAll('.tab-btn')[1].click();
  computeTune();
}

async function loadGPUs() {
  const box = document.getElementById('gpu-info');
  try {
    const gpus = await api('GET', '/api/gpus');
    if (!gpus || gpus.length === 0) {
      box.textContent = 'No NVIDIA GPU detected (nvidia-smi not found, or no CUDA device) — CPU-only.';
      return;
    }
    box.textContent = gpus.map(g => `${g.Name}: ${fmtBytes(g.FreeBytes)} free`).join('; ');
  } catch (e) {
    box.textContent = 'Error detecting GPU: ' + e.message;
  }
}

async function computeTune() {
  const out = document.getElementById('tune-plan');
  const argsBox = document.getElementById('tune-args');
  if (!state.selectedModel) {
    out.textContent = 'Select a model to compute an offload plan.';
    return;
  }
  out.textContent = 'Computing…';
  argsBox.textContent = '';
  try {
    const ctxOverride = document.getElementById('ctx-override').value;
    const body = { model_path: state.selectedModel };
    if (ctxOverride) body.requested_ctx = parseInt(ctxOverride, 10);

    const resp = await api('POST', '/api/tune', body);
    const p = resp.plan;
    const lines = [
      `n_gpu_layers = ${p.NGpuLayers}`,
      p.NCPUMoE ? `n_cpu_moe = ${p.NCPUMoE} (experts of the first ${p.NCPUMoE} layers stay on CPU RAM)` : null,
      `ctx_size = ${p.CtxSize}`,
      `fits on GPU: ${fmtBytes(p.GPUFitBytes)} / total ${fmtBytes(p.TotalBytes)}`,
      ...(p.Notes || []).map(n => '• ' + n),
    ].filter(Boolean);
    out.textContent = lines.join('\n');
    argsBox.textContent = resp.args.join(' ');
    state.lastArgs = resp.args;
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

document.getElementById('retune').addEventListener('click', computeTune);

// ---- server lifecycle -------------------------------------------------------

async function refreshServerStatus() {
  try {
    const st = await api('GET', '/api/server/status');
    const label = document.getElementById('server-status');
    const links = document.getElementById('server-links');
    if (st.Running) {
      label.textContent = `running (pid ${st.PID})`;
      links.style.display = '';
      document.getElementById('link-webui').href = `http://127.0.0.1:${state.port}/`;
      document.getElementById('link-api').href = `http://127.0.0.1:${state.port}/v1/models`;
    } else {
      label.textContent = st.ExitErr ? 'stopped (' + st.ExitErr + ')' : 'stopped';
      links.style.display = 'none';
    }
  } catch (e) {
    // ignore transient errors while polling
  }
}

document.getElementById('start-server').addEventListener('click', async () => {
  if (!state.lastArgs) {
    alert('Compute a launch plan first (select a model in the Models tab).');
    return;
  }
  try {
    await api('POST', '/api/server/start', { args: state.lastArgs });
    refreshServerStatus();
  } catch (e) {
    alert('Failed to start: ' + e.message);
  }
});

document.getElementById('stop-server').addEventListener('click', async () => {
  try {
    await api('POST', '/api/server/stop');
    refreshServerStatus();
  } catch (e) {
    alert('Failed to stop: ' + e.message);
  }
});

setInterval(refreshServerStatus, 2000);

function streamLogs() {
  const box = document.getElementById('server-log');
  const es = new EventSource('/api/server/logs/stream');
  es.onmessage = (ev) => {
    try {
      const line = JSON.parse(ev.data);
      box.textContent += line + '\n';
      box.scrollTop = box.scrollHeight;
    } catch (_) {}
  };
  es.onerror = () => {
    es.close();
    setTimeout(streamLogs, 2000);
  };
}

// ---- settings ---------------------------------------------------------------

async function loadSettings() {
  try {
    const cfg = await api('GET', '/api/config');
    document.getElementById('cfg-bin').value = cfg.llama_server_path || '';
    document.getElementById('cfg-port').value = cfg.port || 8080;
    document.getElementById('cfg-token').value = cfg.hf_token || '';
    document.getElementById('cfg-dirs').value = (cfg.extra_model_dirs || []).join('\n');
    state.port = cfg.port || 8080;
  } catch (e) {
    document.getElementById('settings-status').textContent = 'Error: ' + e.message;
  }
}

document.getElementById('save-settings').addEventListener('click', async () => {
  const cfg = {
    llama_server_path: document.getElementById('cfg-bin').value.trim(),
    port: parseInt(document.getElementById('cfg-port').value, 10) || 8080,
    hf_token: document.getElementById('cfg-token').value.trim(),
    extra_model_dirs: document.getElementById('cfg-dirs').value.split('\n').map(s => s.trim()).filter(Boolean),
  };
  try {
    await api('POST', '/api/config', cfg);
    state.port = cfg.port;
    document.getElementById('settings-status').textContent = 'Saved.';
  } catch (e) {
    document.getElementById('settings-status').textContent = 'Error: ' + e.message;
  }
});

// ---- init ---------------------------------------------------------------

loadSettings().then(() => {
  loadLocalModels();
  loadGPUs();
  refreshDownloads();
  refreshServerStatus();
  streamLogs();
});
