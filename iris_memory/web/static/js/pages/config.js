/**
 * 配置管理页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { toast } from '../components/toast.js';
import { showConfirm } from '../components/modal.js';

let configItems = [];
let currentFilter = '';

export async function loadConfig() {
  const container = el('config-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const res = await api.get('/config');
  if (!res || res.status !== 'ok') { container.innerHTML = '<div class="card">加载失败</div>'; return; }

  configItems = res.data || [];
  renderConfig();
}

function renderConfig() {
  const container = el('config-container');
  const filtered = currentFilter
    ? configItems.filter(c => c.key?.includes(currentFilter) || c.label?.includes(currentFilter) || c.section?.includes(currentFilter))
    : configItems;

  // 按 section 分组
  const sections = {};
  for (const item of filtered) {
    const sec = item.section || '其他';
    if (!sections[sec]) sections[sec] = [];
    sections[sec].push(item);
  }

  if (!Object.keys(sections).length) {
    container.innerHTML = '<div style="text-align:center;color:var(--text2);padding:40px">无匹配的配置项</div>';
    return;
  }

  container.innerHTML = Object.entries(sections).map(([sec, items]) => `
    <div class="card">
      <div class="card-title">${esc(sec)}</div>
      <div class="config-items">${items.map(renderConfigItem).join('')}</div>
    </div>`
  ).join('');

  // 绑定事件
  container.querySelectorAll('.config-input').forEach(input => {
    input.addEventListener('change', () => saveConfigItem(input));
  });
}

function renderConfigItem(item) {
  const key = item.key || '';
  const value = item.value;
  const readonly = item.access_level === 'READONLY' || item.access === 'readonly';
  const desc = item.description || item.label || key;

  let inputHtml;
  const type = typeof value;

  if (type === 'boolean') {
    inputHtml = `<label class="toggle-switch">
      <input type="checkbox" class="config-input" data-key="${esc(key)}" data-type="bool" ${value ? 'checked' : ''} ${readonly ? 'disabled' : ''}>
      <span class="toggle-label">${value ? '开启' : '关闭'}</span>
    </label>`;
  } else if (type === 'number') {
    inputHtml = `<input type="number" class="config-input" data-key="${esc(key)}" data-type="number" value="${value}" ${readonly ? 'disabled' : ''} style="width:120px">`;
  } else if (Array.isArray(value)) {
    inputHtml = `<input type="text" class="config-input" data-key="${esc(key)}" data-type="json" value="${esc(JSON.stringify(value))}" ${readonly ? 'disabled' : ''}>`;
  } else {
    inputHtml = `<input type="text" class="config-input" data-key="${esc(key)}" data-type="string" value="${esc(String(value ?? ''))}" ${readonly ? 'disabled' : ''}>`;
  }

  const readonlyClass = readonly ? ' config-item-readonly' : '';
  return `<div class="config-item${readonlyClass}">
    <div class="config-meta">
      <span class="config-key">${esc(key)}</span>
      <span class="config-desc">${esc(desc)}</span>
    </div>
    <div class="config-control">${inputHtml}</div>
  </div>`;
}

async function saveConfigItem(input) {
  const key = input.dataset.key;
  const type = input.dataset.type;
  let value;

  if (type === 'bool') value = input.checked;
  else if (type === 'number') value = Number(input.value);
  else if (type === 'json') {
    try { value = JSON.parse(input.value); }
    catch { toast.err('JSON 格式无效'); return; }
  } else value = input.value;

  const res = await api.put(`/config/${key}`, { value });
  if (res?.status === 'ok') toast.ok(`已保存: ${key}`);
  else toast.err(res?.message || '保存失败');
}

export function filterConfig() {
  currentFilter = val('config-filter');
  renderConfig();
}

export async function showDiff() {
  const res = await api.get('/config/diff');
  if (!res || res.status !== 'ok') { toast.err('加载对比失败'); return; }

  const diff = res.data || {};
  const entries = Object.entries(diff);
  if (!entries.length) { toast.info('所有配置均为默认值'); return; }

  const body = document.querySelector('#config-diff-modal .modal-body');
  body.innerHTML = `
    <h3>与默认值不同的配置</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>配置项</th><th>当前值</th><th>默认值</th></tr></thead>
      <tbody>${entries.map(([k, v]) => `<tr>
        <td>${esc(k)}</td>
        <td>${esc(JSON.stringify(v.current))}</td>
        <td style="color:var(--text2)">${esc(JSON.stringify(v.default))}</td>
      </tr>`).join('')}</tbody>
    </table></div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="document.getElementById('config-diff-modal').classList.remove('show')">关闭</button>
    </div>`;
  el('config-diff-modal').classList.add('show');
}

export async function exportSnapshot() {
  const res = await api.get('/config/snapshot');
  if (!res || res.status !== 'ok') { toast.err('导出失败'); return; }
  const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `iris_config_${Date.now()}.json`;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
  toast.ok('导出完成');
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }
