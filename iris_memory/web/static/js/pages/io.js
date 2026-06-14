/**
 * 数据导入导出页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { downloadBlob, typeLabels, layerLabels } from '../utils/format.js';
import { toast } from '../components/toast.js';

let pendingImport = { memories: null, kg: null, personas: null };

// ── Tab 切换 ──
export function switchIoTab(tab) {
  el('io-tab-export').style.display = tab === 'export' ? '' : 'none';
  el('io-tab-import').style.display = tab === 'import' ? '' : 'none';
  document.querySelectorAll('#sec-io .tab').forEach((t, i) => {
    t.classList.toggle('active', ['export', 'import'][i] === tab);
  });
}

// ── 导出 ──
export async function exportMemories() {
  const uid = val('exp-mem-user'), gid = val('exp-mem-group');
  const layer = val('exp-mem-layer'), fmt = val('exp-mem-format');

  toast.info('正在导出...');
  try {
    const resp = await api.download('/io/export/memories', {
      format: fmt, user_id: uid, group_id: gid, storage_layer: layer,
    });
    if (!resp.ok) { toast.err('导出失败'); return; }
    const blob = await resp.blob();
    const ext = fmt === 'csv' ? 'csv' : 'json';
    downloadBlob(blob, `iris_memories_${Date.now()}.${ext}`);
    toast.ok('导出完成');
  } catch (e) {
    toast.err(`导出失败: ${e.message}`);
  }
}

export async function exportKg() {
  const uid = val('exp-kg-user'), gid = val('exp-kg-group'), fmt = val('exp-kg-format');

  toast.info('正在导出...');
  try {
    const resp = await api.download('/io/export/kg', {
      format: fmt, user_id: uid, group_id: gid,
    });
    if (!resp.ok) { toast.err('导出失败'); return; }
    const blob = await resp.blob();
    const ext = fmt === 'csv' ? 'csv' : 'json';
    downloadBlob(blob, `iris_kg_${Date.now()}.${ext}`);
    toast.ok('导出完成');
  } catch (e) {
    toast.err(`导出失败: ${e.message}`);
  }
}

export async function exportPersonas() {
  const uid = val('exp-persona-user'), fmt = val('exp-persona-format');

  toast.info('正在导出...');
  try {
    const resp = await api.download('/io/export/personas', {
      format: fmt, user_id: uid,
    });
    if (!resp.ok) { toast.err('导出失败'); return; }
    const blob = await resp.blob();
    const ext = fmt === 'csv' ? 'csv' : 'json';
    downloadBlob(blob, `iris_personas_${Date.now()}.${ext}`);
    toast.ok('导出完成');
  } catch (e) {
    toast.err(`导出失败: ${e.message}`);
  }
}

// ── 迁移到 Iris Chat Memory（新版）──
export async function exportToIrisChatMemory() {
  const uid = val('exp-icm-user'), gid = val('exp-icm-group');
  const layer = val('exp-icm-layer');

  toast.info('正在导出迁移文件...');
  try {
    const resp = await api.download('/io/export/iris_chat_memory', {
      user_id: uid, group_id: gid, storage_layer: layer,
    });
    if (!resp.ok) { toast.err('导出失败'); return; }
    const blob = await resp.blob();
    downloadBlob(blob, `iris_chat_memory_l2_${Date.now()}.json`);
    toast.ok('导出完成，可在新版 Iris Chat Memory 中导入');
  } catch (e) {
    toast.err(`导出失败: ${e.message}`);
  }
}

// ── 导入 ──
export function handleFileDrop(event, type) {
  event.preventDefault();
  event.currentTarget.classList.remove('dragover');
  const file = event.dataTransfer?.files?.[0];
  if (file) previewFile(file, type);
}

export function handleFileSelect(input, type) {
  const file = input.files?.[0];
  if (file) previewFile(file, type);
}

async function previewFile(file, type) {
  const fmt = file.name.endsWith('.csv') ? 'csv' : 'json';
  const previewEl = el(`import-${type}-preview`);
  const resultEl = el(`import-${type}-result`);
  resultEl.innerHTML = '';

  previewEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  try {
    const text = await file.text();
    // 发送预览请求
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch(`/api/v1/io/preview?format=${fmt}&type=${type}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${localStorage.getItem('iris_token') || ''}` },
      body: text,
    });
    const res = await resp.json();

    if (res.status !== 'ok') {
      previewEl.innerHTML = `<div style="color:var(--danger)">预览失败: ${esc(res.message)}</div>`;
      return;
    }

    pendingImport[type] = { text, fmt };
    const preview = res.data;

    previewEl.innerHTML = `
      <div class="preview-summary">
        <strong>文件: </strong>${esc(file.name)} (${(file.size / 1024).toFixed(1)} KB)<br>
        <strong>格式: </strong>${fmt.toUpperCase()}<br>
        <strong>条目数: </strong>${preview.count || preview.total || 0}
      </div>
      <div style="margin-top:10px;display:flex;gap:8px">
        <button class="btn btn-primary" onclick="window.__io.confirmImport('${type}')">确认导入</button>
        <button class="btn btn-outline" onclick="window.__io.cancelImport('${type}')">取消</button>
      </div>`;
  } catch (e) {
    previewEl.innerHTML = `<div style="color:var(--danger)">读取文件失败: ${esc(e.message)}</div>`;
  }
}

export async function confirmImport(type) {
  const data = pendingImport[type];
  if (!data) return;

  const resultEl = el(`import-${type}-result`);
  resultEl.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  try {
    const endpoints = {
      memories: '/io/import/memories',
      kg: '/io/import/kg',
      personas: '/io/import/personas',
    };
    const endpoint = endpoints[type] || '/io/import/memories';
    const ct = data.fmt === 'csv' ? 'text/csv' : 'application/json';
    const res = await api.postRaw(`${endpoint}?format=${data.fmt}`, data.text, ct);

    if (res?.status === 'ok') {
      const d = res.data || {};
      resultEl.innerHTML = `<div class="preview-summary" style="border-left:3px solid var(--success)">
        <strong>导入完成!</strong><br>
        成功: ${d.success_count ?? 0} 条 | 失败: ${d.fail_count ?? 0} 条${d.skipped ? ` | 跳过: ${d.skipped} 条` : ''}
        ${d.errors?.length ? `<br><span style="color:var(--danger)">错误: ${d.errors.slice(0, 3).map(e => esc(e)).join('; ')}</span>` : ''}
      </div>`;
      toast.ok('导入完成');
    } else {
      resultEl.innerHTML = `<div style="color:var(--danger)">导入失败: ${esc(res?.message || '未知错误')}</div>`;
    }
  } catch (e) {
    resultEl.innerHTML = `<div style="color:var(--danger)">导入失败: ${esc(e.message)}</div>`;
  }

  pendingImport[type] = null;
  el(`import-${type}-preview`).innerHTML = '';
}

export function cancelImport(type) {
  pendingImport[type] = null;
  el(`import-${type}-preview`).innerHTML = '';
  el(`import-${type}-result`).innerHTML = '';
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }

window.__io = { confirmImport, cancelImport };
