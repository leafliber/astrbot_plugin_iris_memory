// ========== Import / Export Section ==========

function switchIoTab(tab) {
  document.getElementById('io-tab-export').style.display = tab === 'export' ? 'block' : 'none';
  document.getElementById('io-tab-import').style.display = tab === 'import' ? 'block' : 'none';
  document.querySelectorAll('#sec-io .tab').forEach((t, i) => t.classList.toggle('active', (i===0&&tab==='export')||(i===1&&tab==='import')));
}

async function exportMemories() {
  const userId = document.getElementById('exp-mem-user').value;
  const groupId = document.getElementById('exp-mem-group').value;
  const layer = document.getElementById('exp-mem-layer').value;
  const format = document.getElementById('exp-mem-format').value;

  let path = `/export/memories?format=${format}`;
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (groupId) path += `&group_id=${encodeURIComponent(groupId)}`;
  if (layer) path += `&layer=${layer}`;

  toast('正在导出...', 'info');

  try {
    const headers = {};
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(API_BASE + path, { headers });
    if (!resp.ok) { toast('导出失败', 'error'); return; }

    const blob = await resp.blob();
    const cd = resp.headers.get('content-disposition') || '';
    const match = cd.match(/filename="?(.+?)"?$/);
    const filename = match ? match[1] : `memories_export.${format}`;

    downloadBlob(blob, filename);
    toast('导出成功', 'success');
  } catch(e) {
    toast(`导出失败: ${e.message}`, 'error');
  }
}

async function exportKg() {
  const userId = document.getElementById('exp-kg-user').value;
  const groupId = document.getElementById('exp-kg-group').value;
  const format = document.getElementById('exp-kg-format').value;

  let path = `/export/kg?format=${format}`;
  if (userId) path += `&user_id=${encodeURIComponent(userId)}`;
  if (groupId) path += `&group_id=${encodeURIComponent(groupId)}`;

  toast('正在导出...', 'info');

  try {
    const headers = {};
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch(API_BASE + path, { headers });
    if (!resp.ok) { toast('导出失败', 'error'); return; }

    const blob = await resp.blob();
    const cd = resp.headers.get('content-disposition') || '';
    const match = cd.match(/filename="?(.+?)"?$/);
    const filename = match ? match[1] : `kg_export.${format}`;

    downloadBlob(blob, filename);
    toast('导出成功', 'success');
  } catch(e) {
    toast(`导出失败: ${e.message}`, 'error');
  }
}

function handleFileDrop(ev, type) {
  ev.preventDefault();
  ev.currentTarget.classList.remove('dragover');
  const file = ev.dataTransfer.files[0];
  if (file) previewImportFile(file, type);
}

function handleFileSelect(input, type) {
  const file = input.files[0];
  if (file) previewImportFile(file, type);
  input.value = '';
}

// Import preview before actual import
let pendingImport = { file:null, type:'', content:'', format:'' };

async function previewImportFile(file, type) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['json', 'csv'].includes(ext)) {
    toast('仅支持 JSON 和 CSV 格式', 'error');
    return;
  }

  if (file.size > 50 * 1024 * 1024) {
    toast('文件过大，最大支持 50MB', 'error');
    return;
  }

  const previewDiv = document.getElementById(type === 'memories' ? 'import-mem-preview' : 'import-kg-preview');
  const resultDiv = document.getElementById(type === 'memories' ? 'import-mem-result' : 'import-kg-result');
  resultDiv.innerHTML = '';
  previewDiv.innerHTML = '<div class="loading"><div class="spinner"></div> 正在解析预览...</div>';

  try {
    const content = await file.text();
    pendingImport = { file, type, content, format: ext };

    const resp = await api('/import/preview', {
      method: 'POST',
      body: JSON.stringify({ data: content, format: ext, type })
    });

    if (resp && resp.status === 'ok') {
      const d = resp.data;
      let html = `<div class="preview-summary">
        <strong>⬡ 预览结果:</strong> 共 ${d.total} 条数据，有效 <span style="color:var(--success);">${d.valid}</span> 条，无效 <span style="color:var(--danger);">${d.invalid}</span> 条
        ${d.errors.length > 0 ? `<br><span style="color:var(--danger);font-size:12px;">错误: ${d.errors.slice(0,3).map(e => escHtml(e)).join('; ')}</span>` : ''}
      </div>`;

      if (d.preview_items.length > 0) {
        html += '<div class="preview-table"><table>';
        if (type === 'memories') {
          html += '<thead><tr><th>内容</th><th>用户</th><th>类型</th><th>层级</th></tr></thead><tbody>';
          html += d.preview_items.map(it => `<tr>
            <td>${escHtml((it.content||'').substring(0,80))}</td>
            <td>${escHtml(it.user_id||'')}</td>
            <td>${escHtml(it.type||'')}</td>
            <td>${escHtml(it.storage_layer||'')}</td>
          </tr>`).join('');
        } else {
          html += '<thead><tr><th>类型</th><th>名称/源</th><th>详情</th></tr></thead><tbody>';
          html += d.preview_items.map(it => {
            if (it.item_type === 'node') {
              return `<tr><td>节点</td><td>${escHtml(it.name||'')}</td><td>类型: ${escHtml(it.node_type||'')}</td></tr>`;
            } else {
              return `<tr><td>边</td><td>${escHtml(it.source_id||'')} → ${escHtml(it.target_id||'')}</td><td>${escHtml(it.relation_type||'')}</td></tr>`;
            }
          }).join('');
        }
        html += '</tbody></table></div>';
      }

      html += `<div style="display:flex;gap:10px;margin-top:12px;">
        <button class="btn btn-primary" onclick="confirmImport('${type}')">确认导入 (${d.valid} 条有效)</button>
        <button class="btn btn-outline" onclick="cancelImport('${type}')">取消</button>
      </div>`;

      previewDiv.innerHTML = html;
    } else {
      previewDiv.innerHTML = `<div style="color:var(--danger);">预览失败: ${resp?.message || '未知错误'}</div>`;
    }
  } catch(e) {
    previewDiv.innerHTML = `<div style="color:var(--danger);">预览失败: ${e.message}</div>`;
  }
}

async function confirmImport(type) {
  const previewDiv = document.getElementById(type === 'memories' ? 'import-mem-preview' : 'import-kg-preview');
  const resultDiv = document.getElementById(type === 'memories' ? 'import-mem-result' : 'import-kg-result');
  previewDiv.innerHTML = '';
  resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div> 正在导入...</div>';

  try {
    const endpoint = type === 'memories' ? '/import/memories' : '/import/kg';
    const resp = await api(endpoint, {
      method: 'POST',
      body: JSON.stringify({ data: pendingImport.content, format: pendingImport.format })
    });

    if (resp && resp.status === 'ok') {
      const d = resp.data;
      if (type === 'memories') {
        resultDiv.innerHTML = `<div style="padding:12px;background:var(--bg3);border-radius:var(--radius);">
          ✅ 导入完成：成功 <strong>${d.success_count}</strong> 条，失败 <strong>${d.fail_count}</strong> 条
          ${d.errors?.length ? '<br>错误: ' + d.errors.join('<br>') : ''}
        </div>`;
      } else {
        resultDiv.innerHTML = `<div style="padding:12px;background:var(--bg3);border-radius:var(--radius);">
          ✅ 导入完成：节点 <strong>${d.nodes_imported}</strong>，边 <strong>${d.edges_imported}</strong>
          ${d.errors?.length ? '<br>错误: ' + d.errors.join('<br>') : ''}
        </div>`;
      }
      toast('导入完成', 'success');
    } else {
      resultDiv.innerHTML = `<div style="padding:12px;color:var(--danger);">导入失败: ${resp?.message || '未知错误'}</div>`;
    }
  } catch(e) {
    resultDiv.innerHTML = `<div style="padding:12px;color:var(--danger);">导入失败: ${e.message}</div>`;
  }

  pendingImport = { file:null, type:'', content:'', format:'' };
}

function cancelImport(type) {
  const previewDiv = document.getElementById(type === 'memories' ? 'import-mem-preview' : 'import-kg-preview');
  previewDiv.innerHTML = '';
  pendingImport = { file:null, type:'', content:'', format:'' };
}
