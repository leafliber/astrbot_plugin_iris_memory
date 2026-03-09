/**
 * 记忆管理页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { typeLabels, layerLabels, highlightText, fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';
import { showConfirm, closeModal, showDetailModal } from '../components/modal.js';
import { renderPagination } from '../components/pagination.js';

const state = { page: 1, pageSize: 20, total: 0, loaded: false, selected: new Set() };

export function getState() { return state; }

export async function searchMemories() {
  state.loaded = true;
  const q = val('mem-query'), uid = val('mem-user'), gid = val('mem-group');
  const layer = val('mem-layer'), type = val('mem-type');

  showLoading(true);
  const res = await api.get('/memories', {
    query: q, user_id: uid, group_id: gid,
    storage_layer: layer, memory_type: type, page: state.page, page_size: state.pageSize,
  });
  showLoading(false);

  if (!res || res.status !== 'ok') { setTbody('<tr><td colspan="8">加载失败</td></tr>'); return; }

  const d = res.data;
  state.total = d.total || 0;
  updateInfo();

  const items = d.items || [];
  if (!items.length) { setTbody('<tr><td colspan="8" style="text-align:center;color:var(--text2)">暂无数据</td></tr>'); renderPag(); return; }

  setTbody(items.map(m => {
    const content = q ? highlightText(esc(truncate(m.content, 80)), q) : esc(truncate(m.content, 80));
    return `<tr>
      <td class="checkbox-col"><input type="checkbox" class="row-cb" data-id="${esc(m.id)}" onchange="window.__mem.toggleSelect()"></td>
      <td class="clickable-row" onclick="window.__mem.showDetail('${esc(m.id)}')" title="${esc(m.content)}">${content}</td>
      <td>${esc(m.sender_name || m.user_id || '-')}</td>
      <td><span class="badge badge-${esc(m.type)}">${esc(typeLabels[m.type] || m.type)}</span></td>
      <td><span class="badge badge-${esc(m.storage_layer)}">${esc(layerLabels[m.storage_layer] || m.storage_layer)}</span></td>
      <td>${m.confidence != null ? (m.confidence * 100).toFixed(0) + '%' : '-'}</td>
      <td>${esc(fmtTime(m.created_time))}</td>
      <td>
        <button class="btn btn-outline btn-sm" onclick="window.__mem.openEdit('${esc(m.id)}')">编辑</button>
        <button class="btn btn-danger btn-sm" onclick="window.__mem.deleteSingle('${esc(m.id)}')">删除</button>
      </td>
    </tr>`;
  }).join(''));

  state.selected.clear();
  updateSelectedCount();
  el('select-all').checked = false;
  renderPag();
}

export function memPage(p) { state.page = p; searchMemories(); }

export function toggleSelectAll() {
  const checked = el('select-all').checked;
  document.querySelectorAll('#mem-tbody .row-cb').forEach(cb => {
    cb.checked = checked;
    if (checked) state.selected.add(cb.dataset.id); else state.selected.delete(cb.dataset.id);
  });
  updateSelectedCount();
}

export function toggleSelect() {
  document.querySelectorAll('#mem-tbody .row-cb').forEach(cb => {
    if (cb.checked) state.selected.add(cb.dataset.id); else state.selected.delete(cb.dataset.id);
  });
  updateSelectedCount();
}

function updateSelectedCount() {
  const n = state.selected.size;
  el('selected-count').textContent = n;
  el('batch-del-btn').disabled = n === 0;
  el('batch-export-btn').disabled = n === 0;
}

export async function showDetail(id) {
  const res = await api.get(`/memories/${encodeURIComponent(id)}`);
  if (!res || res.status !== 'ok') { toast.err('无法加载记忆详情'); return; }
  const m = res.data;
  const html = `
    <h3>◈ 记忆详情</h3>
    <div class="detail-content">${esc(m.content || '')}</div>
    ${m.summary ? `<div style="margin-bottom:12px"><strong>摘要: </strong>${esc(m.summary)}</div>` : ''}
    <div class="detail-grid">
      ${dItem('ID', m.id)}${dItem('用户', m.sender_name || m.user_id)}
      ${dItem('群组', m.group_id || '-')}${dItem('类型', typeLabels[m.type] || m.type)}
      ${dItem('层级', layerLabels[m.storage_layer] || m.storage_layer)}${dItem('作用域', m.scope || '-')}
      ${dItem('置信度', m.confidence != null ? (m.confidence * 100).toFixed(1) + '%' : '-')}
      ${dItem('重要性', m.importance_score != null ? (m.importance_score * 100).toFixed(1) + '%' : '-')}
      ${dItem('RIF 评分', m.rif_score ?? '-')}${dItem('访问次数', m.access_count ?? '-')}
      ${dItem('质量等级', m.quality_level ?? '-')}${dItem('创建时间', fmtTime(m.created_time))}
    </div>
    ${m.keywords?.length ? `<div><strong>关键词: </strong>${m.keywords.map(k => `<span class="badge" style="margin:2px">${esc(k)}</span>`).join('')}</div>` : ''}
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="window.__mem.closeDetail()">关闭</button>
      <button class="btn btn-primary" onclick="window.__mem.openEdit('${esc(m.id)}')">编辑</button>
      <button class="btn btn-danger" onclick="window.__mem.deleteSingle('${esc(m.id)}')">删除</button>
    </div>`;
  showDetailModal('detail-modal', html);
}

export function closeDetail() { closeModal('detail-modal'); }

export async function openEdit(id) {
  const res = await api.get(`/memories/${encodeURIComponent(id)}`);
  if (!res || res.status !== 'ok') { toast.err('无法加载'); return; }
  const m = res.data;
  closeModal('detail-modal');

  const body = document.querySelector('#edit-modal .modal-body');
  body.innerHTML = `
    <h3>编辑记忆</h3>
    <input type="hidden" id="edit-id" value="${esc(m.id)}">
    <div class="form-group"><label>内容</label><textarea id="edit-content" rows="4">${esc(m.content)}</textarea></div>
    <div class="form-row">
      <div class="form-group"><label>类型</label>
        <select id="edit-type">${Object.entries(typeLabels).map(([k, v]) => `<option value="${k}" ${m.type === k ? 'selected' : ''}>${v}</option>`).join('')}</select>
      </div>
      <div class="form-group"><label>层级</label>
        <select id="edit-layer">${Object.entries(layerLabels).map(([k, v]) => `<option value="${k}" ${m.storage_layer === k ? 'selected' : ''}>${v}</option>`).join('')}</select>
      </div>
    </div>
    <div class="form-row">
      <div class="form-group"><label>置信度 (0-1)</label><input id="edit-confidence" type="number" step="0.01" min="0" max="1" value="${m.confidence ?? ''}"></div>
      <div class="form-group"><label>重要性 (0-1)</label><input id="edit-importance" type="number" step="0.01" min="0" max="1" value="${m.importance_score ?? ''}"></div>
    </div>
    <div class="form-group"><label>摘要</label><input id="edit-summary" value="${esc(m.summary || '')}"></div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="window.__mem.closeEdit()">取消</button>
      <button class="btn btn-primary" onclick="window.__mem.saveEdit()">保存</button>
    </div>`;
  document.getElementById('edit-modal').classList.add('show');
}

export function closeEdit() { closeModal('edit-modal'); }

export async function saveEdit() {
  const id = val('edit-id');
  const updates = {};
  const content = val('edit-content'); if (content) updates.content = content;
  updates.type = val('edit-type');
  updates.storage_layer = val('edit-layer');
  const conf = val('edit-confidence'); if (conf !== '') updates.confidence = Number(conf);
  const imp = val('edit-importance'); if (imp !== '') updates.importance_score = Number(imp);
  const summary = val('edit-summary'); if (summary) updates.summary = summary;

  const res = await api.put(`/memories/${encodeURIComponent(id)}`, updates);
  if (res?.status === 'ok') { toast.ok('已保存'); closeEdit(); searchMemories(); }
  else toast.err(res?.message || '保存失败');
}

export function deleteSingle(id) {
  showConfirm('删除记忆', '确定要删除此记忆吗？此操作不可撤销。', async () => {
    const res = await api.del(`/memories/${encodeURIComponent(id)}`);
    if (res?.status === 'ok') { toast.ok('已删除'); closeModal('detail-modal'); searchMemories(); }
    else toast.err(res?.message || '删除失败');
  });
}

export function batchDelete() {
  const ids = [...state.selected];
  if (!ids.length) return;
  showConfirm('批量删除', `确定要删除选中的 ${ids.length} 条记忆吗？`, async () => {
    const res = await api.post('/memories/batch-delete', { ids });
    if (res?.status === 'ok') {
      toast.ok(`成功删除 ${res.data?.success_count ?? ids.length} 条`);
      searchMemories();
    } else toast.err(res?.message || '批量删除失败');
  });
}

export function resetFilters() {
  ['mem-query', 'mem-user', 'mem-group'].forEach(id => { el(id).value = ''; });
  ['mem-layer', 'mem-type'].forEach(id => { el(id).selectedIndex = 0; });
  state.page = 1;
  searchMemories();
}

export function changePageSize(v) { state.pageSize = Number(v); state.page = 1; searchMemories(); }

// ── 内部辅助 ──

function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }
function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : s || ''; }
function setTbody(html) { el('mem-tbody').innerHTML = html; }
function showLoading(v) { const o = el('mem-loading'); if (o) o.style.display = v ? 'flex' : 'none'; }
function updateInfo() { el('mem-total-info').textContent = `共 ${state.total} 条`; }
function dItem(label, value) {
  return `<div class="detail-item"><div class="detail-label">${esc(label)}</div><div class="detail-value">${esc(value ?? '-')}</div></div>`;
}

function renderPag() {
  renderPagination({
    page: state.page, pageSize: state.pageSize, total: state.total,
    onChange: memPage,
    container: el('mem-pagination'),
  });
}

// 挂载到 window 供 inline onclick 调用
window.__mem = {
  showDetail: showDetail, openEdit: openEdit, deleteSingle: deleteSingle,
  closeDetail: closeDetail, closeEdit: closeEdit, saveEdit: saveEdit,
  toggleSelect: toggleSelect,
};
