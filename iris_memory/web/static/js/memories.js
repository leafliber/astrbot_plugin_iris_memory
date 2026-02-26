// ========== Memories Section ==========

const memState = { page:1, pageSize:20, loaded:false, selected:new Set(), currentQuery:'' };

async function searchMemories() {
  const query = document.getElementById('mem-query').value;
  const userId = document.getElementById('mem-user').value;
  const groupId = document.getElementById('mem-group').value;
  const layer = document.getElementById('mem-layer').value;
  const type = document.getElementById('mem-type').value;

  memState.currentQuery = query;

  let path;
  if (query) {
    path = `/memories/search?q=${encodeURIComponent(query)}&page=${memState.page}&page_size=${memState.pageSize}`;
  } else {
    path = `/memories?page=${memState.page}&page_size=${memState.pageSize}`;
  }
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (groupId) path += `&group_id=${encodeURIComponent(groupId)}`;
  if (layer) path += `&layer=${layer}`;
  if (type) path += `&type=${type}`;

  document.getElementById('mem-loading').style.display = 'flex';
  document.getElementById('mem-tbody').innerHTML = '';

  const resp = await api(path);
  document.getElementById('mem-loading').style.display = 'none';
  memState.loaded = true;

  if (!resp || resp.status !== 'ok') {
    document.getElementById('mem-tbody').innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text2)">查询失败</td></tr>';
    return;
  }

  const data = resp.data;
  const items = data.items || [];

  document.getElementById('mem-total-info').textContent = `共 ${data.total} 条记忆`;

  if (items.length === 0) {
    document.getElementById('mem-tbody').innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text2)">暂无记忆数据</td></tr>';
    document.getElementById('mem-pagination').innerHTML = '';
    return;
  }

  memState.selected.clear();
  updateSelectedCount();
  document.getElementById('select-all').checked = false;

  const tbody = items.map(m => {
    const layerBadge = m.storage_layer ? `<span class="badge badge-${m.storage_layer}">${layerLabels[m.storage_layer] || m.storage_layer}</span>` : '';
    const typeBadge = m.type ? `<span class="badge badge-${m.type}">${typeLabels[m.type] || m.type}</span>` : '';
    const time = m.created_time ? new Date(m.created_time).toLocaleString('zh-CN') : '';
    const contentRaw = (m.summary || m.content || '').substring(0, 100);
    const contentHtml = memState.currentQuery ? highlightText(contentRaw, memState.currentQuery) : escHtml(contentRaw);
    return `<tr>
      <td class="checkbox-col"><input type="checkbox" value="${m.id}" onchange="toggleSelect(this)"></td>
      <td class="clickable-row" onclick="showMemoryDetail('${m.id}')" title="点击查看详情">${contentHtml}</td>
      <td>${escHtml(m.sender_name || m.user_id || '')}</td>
      <td>${typeBadge}</td>
      <td>${layerBadge}</td>
      <td>${(m.confidence || 0).toFixed(2)}</td>
      <td style="font-size:12px;">${time}</td>
      <td style="white-space:nowrap;">
        <button class="btn btn-outline btn-sm" onclick="openEditModal('${m.id}')" style="margin-right:4px;">编辑</button>
        <button class="btn btn-danger btn-sm" onclick="deleteSingleMemory('${m.id}')">删除</button>
      </td>
    </tr>`;
  }).join('');

  document.getElementById('mem-tbody').innerHTML = tbody;

  // Pagination
  const totalPages = Math.ceil(data.total / memState.pageSize);
  let pagHtml = '';
  if (totalPages > 1) {
    pagHtml += `<button class="btn btn-outline btn-sm" ${memState.page<=1?'disabled':''} onclick="memPage(1)">首页</button>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${memState.page<=1?'disabled':''} onclick="memPage(${memState.page-1})">上一页</button>`;
    pagHtml += `<span class="page-info">${memState.page} / ${totalPages}</span>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${memState.page>=totalPages?'disabled':''} onclick="memPage(${memState.page+1})">下一页</button>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${memState.page>=totalPages?'disabled':''} onclick="memPage(${totalPages})">末页</button>`;
  }
  document.getElementById('mem-pagination').innerHTML = pagHtml;
}

function highlightText(text, query) {
  if (!query) return escHtml(text);
  const escaped = escHtml(text);
  const queryEsc = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  try {
    return escaped.replace(new RegExp(`(${queryEsc})`, 'gi'), '<span class="highlight">$1</span>');
  } catch(e) { return escaped; }
}

function memPage(p) { memState.page = p; searchMemories(); }

function resetMemoryFilters() {
  document.getElementById('mem-query').value = '';
  document.getElementById('mem-user').value = '';
  document.getElementById('mem-group').value = '';
  document.getElementById('mem-layer').value = '';
  document.getElementById('mem-type').value = '';
  memState.page = 1;
  searchMemories();
}

function toggleSelect(cb) {
  if (cb.checked) memState.selected.add(cb.value);
  else memState.selected.delete(cb.value);
  updateSelectedCount();
}

function toggleSelectAll() {
  const checked = document.getElementById('select-all').checked;
  document.querySelectorAll('#mem-tbody input[type="checkbox"]').forEach(cb => {
    cb.checked = checked;
    if (checked) memState.selected.add(cb.value);
    else memState.selected.delete(cb.value);
  });
  updateSelectedCount();
}

function updateSelectedCount() {
  const count = memState.selected.size;
  document.getElementById('selected-count').textContent = count;
  document.getElementById('batch-del-btn').disabled = count === 0;
  document.getElementById('batch-export-btn').disabled = count === 0;
}

// Memory Detail Modal
async function showMemoryDetail(id) {
  const resp = await api(`/memories/detail?id=${encodeURIComponent(id)}`);
  if (!resp || resp.status !== 'ok') { toast('获取详情失败', 'error'); return; }

  const m = resp.data;
  const layerBadge = m.storage_layer ? `<span class="badge badge-${m.storage_layer}">${layerLabels[m.storage_layer] || m.storage_layer}</span>` : '';
  const typeBadge = m.type ? `<span class="badge badge-${m.type}">${typeLabels[m.type] || m.type}</span>` : '';

  document.getElementById('detail-content').innerHTML = `
    <div class="detail-content">${escHtml(m.content || '')}</div>
    ${m.summary ? `<div style="margin-bottom:12px;"><strong>摘要:</strong> ${escHtml(m.summary)}</div>` : ''}
    <div class="detail-grid">
      <div class="detail-item"><div class="detail-label">ID</div><div class="detail-value" style="font-size:12px;font-family:monospace;">${m.id}</div></div>
      <div class="detail-item"><div class="detail-label">用户</div><div class="detail-value">${escHtml(m.sender_name || m.user_id || '')}</div></div>
      <div class="detail-item"><div class="detail-label">类型</div><div class="detail-value">${typeBadge}</div></div>
      <div class="detail-item"><div class="detail-label">层级</div><div class="detail-value">${layerBadge}</div></div>
      <div class="detail-item"><div class="detail-label">置信度</div><div class="detail-value">${(m.confidence || 0).toFixed(3)}</div></div>
      <div class="detail-item"><div class="detail-label">重要性</div><div class="detail-value">${(m.importance_score || 0).toFixed(3)}</div></div>
      <div class="detail-item"><div class="detail-label">RIF 评分</div><div class="detail-value">${(m.rif_score || 0).toFixed(3)}</div></div>
      <div class="detail-item"><div class="detail-label">访问次数</div><div class="detail-value">${m.access_count || 0}</div></div>
      <div class="detail-item"><div class="detail-label">群组 ID</div><div class="detail-value">${escHtml(m.group_id || '无')}</div></div>
      <div class="detail-item"><div class="detail-label">作用域</div><div class="detail-value">${m.scope || '—'}</div></div>
      <div class="detail-item"><div class="detail-label">质量等级</div><div class="detail-value">${m.quality_level || '—'}</div></div>
      <div class="detail-item"><div class="detail-label">创建时间</div><div class="detail-value">${m.created_time ? new Date(m.created_time).toLocaleString('zh-CN') : ''}</div></div>
    </div>
  `;

  document.getElementById('detail-edit-btn').onclick = () => { closeModal('detail-modal'); openEditModal(id); };
  document.getElementById('detail-modal').classList.add('show');
}

// Memory Edit Modal
async function openEditModal(id) {
  const resp = await api(`/memories/detail?id=${encodeURIComponent(id)}`);
  if (!resp || resp.status !== 'ok') { toast('获取记忆失败', 'error'); return; }

  const m = resp.data;
  document.getElementById('edit-id').value = m.id;
  document.getElementById('edit-content').value = m.content || '';
  document.getElementById('edit-type').value = m.type || 'fact';
  document.getElementById('edit-layer').value = m.storage_layer || 'episodic';
  document.getElementById('edit-confidence').value = m.confidence || 0.5;
  document.getElementById('edit-importance').value = m.importance_score || 0.5;
  document.getElementById('edit-summary').value = m.summary || '';
  document.getElementById('edit-modal').classList.add('show');
}

async function saveMemoryEdit() {
  const id = document.getElementById('edit-id').value;
  const updates = {
    content: document.getElementById('edit-content').value,
    type: document.getElementById('edit-type').value,
    storage_layer: document.getElementById('edit-layer').value,
    confidence: parseFloat(document.getElementById('edit-confidence').value),
    importance_score: parseFloat(document.getElementById('edit-importance').value),
    summary: document.getElementById('edit-summary').value,
  };

  if (!updates.content.trim()) { toast('内容不能为空', 'error'); return; }

  const resp = await api('/memories/update', { method:'POST', body:JSON.stringify({ id, updates }) });
  if (resp && resp.status === 'ok') {
    toast('更新成功', 'success');
    closeModal('edit-modal');
    searchMemories();
  } else {
    toast(resp?.message || '更新失败', 'error');
  }
}

function deleteSingleMemory(id) {
  showConfirm('确认删除', '确定要删除这条记忆吗？此操作不可撤销。', async () => {
    const resp = await api('/memories/delete', { method:'POST', body:JSON.stringify({id}) });
    if (resp && resp.status === 'ok') {
      toast('删除成功', 'success');
      searchMemories();
    } else {
      toast(resp?.message || '删除失败', 'error');
    }
  });
}

function batchDeleteMemories() {
  const ids = [...memState.selected];
  showConfirm('批量删除', `确定要删除选中的 ${ids.length} 条记忆吗？此操作不可撤销。`, async () => {
    const resp = await api('/memories/batch-delete', { method:'POST', body:JSON.stringify({ids}) });
    if (resp && resp.status === 'ok') {
      const d = resp.data;
      toast(`成功删除 ${d.success_count} 条，失败 ${d.fail_count} 条`, d.fail_count > 0 ? 'error' : 'success');
      memState.selected.clear();
      searchMemories();
    } else {
      toast(resp?.message || '批量删除失败', 'error');
    }
  });
}

async function exportSelectedMemories() {
  const ids = [...memState.selected];
  if (ids.length === 0) return;
  toast('正在导出选中记忆...', 'info');

  const items = [];
  for (const id of ids) {
    const resp = await api(`/memories/detail?id=${encodeURIComponent(id)}`);
    if (resp && resp.status === 'ok') items.push(resp.data);
  }

  const blob = new Blob([JSON.stringify({ memories: items, exported_at: new Date().toISOString() }, null, 2)], { type: 'application/json' });
  downloadBlob(blob, `selected_memories_${items.length}.json`);
  toast(`已导出 ${items.length} 条记忆`, 'success');
}
