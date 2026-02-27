// ========== Proactive Reply Section ==========

const proactiveState = { loaded: false, status: null };

async function loadProactiveStatus() {
  const resp = await api('/proactive/status');
  if (!resp || resp.status !== 'ok') {
    document.getElementById('proactive-status-container').innerHTML = 
      '<div style="color:var(--danger);text-align:center;padding:20px;">获取状态失败</div>';
    return;
  }
  proactiveState.status = resp.data;
  proactiveState.loaded = true;
  renderProactiveStatus(resp.data);
}

function renderProactiveStatus(data) {
  const container = document.getElementById('proactive-status-container');
  
  if (!data.enabled) {
    container.innerHTML = `
      <div class="card" style="text-align:center;padding:40px;">
        <div style="font-size:48px;margin-bottom:16px;">⏸</div>
        <div style="font-size:16px;color:var(--text2);">主动回复功能未启用</div>
        <div style="font-size:13px;color:var(--text2);margin-top:8px;">
          请在配置中开启 <code style="background:var(--bg3);padding:2px 6px;border-radius:4px;">proactive_reply.enable</code>
        </div>
      </div>
    `;
    return;
  }

  const stats = data.stats || {};
  const config = data.config || {};
  
  let html = `
    <div class="stats-grid" style="margin-bottom:16px;">
      <div class="stat-card">
        <div class="stat-value">${stats.replies_sent || 0}</div>
        <div class="stat-label">已发送回复</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.replies_skipped || 0}</div>
        <div class="stat-label">已跳过</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.replies_failed || 0}</div>
        <div class="stat-label">发送失败</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${stats.pending_tasks || 0}</div>
        <div class="stat-label">待处理任务</div>
      </div>
    </div>
    
    <div class="card" style="margin-bottom:16px;">
      <div class="card-title">⚙️ 配置信息</div>
      <div class="detail-grid">
        <div class="detail-item">
          <div class="detail-label">冷却时间</div>
          <div class="detail-value">${config.cooldown_seconds || 60} 秒</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">每日最大回复数</div>
          <div class="detail-value">${config.max_daily_replies || 20}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">白名单模式</div>
          <div class="detail-value">${data.whitelist_mode ? 
            '<span style="color:var(--success);">✓ 已开启</span>' : 
            '<span style="color:var(--text2);">✗ 未开启</span>'}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">白名单群数</div>
          <div class="detail-value">${(data.whitelist || []).length}</div>
        </div>
      </div>
    </div>
  `;

  if (!data.whitelist_mode) {
    html += `
      <div class="card" style="background:var(--bg3);border-color:var(--border);">
        <div style="display:flex;align-items:center;gap:12px;">
          <span style="font-size:24px;">💡</span>
          <div>
            <div style="font-size:14px;color:var(--text);">白名单模式未开启</div>
            <div style="font-size:12px;color:var(--text2);margin-top:4px;">
              请在配置中开启 <code style="background:var(--bg2);padding:2px 6px;border-radius:4px;">proactive_reply.group_whitelist_mode</code> 后，才能通过此页面管理群聊白名单
            </div>
          </div>
        </div>
      </div>
    `;
  }

  container.innerHTML = html;
  
  if (data.whitelist_mode) {
    loadProactiveWhitelist();
  }
}

async function loadProactiveWhitelist() {
  const resp = await api('/proactive/whitelist');
  if (!resp || resp.status !== 'ok') {
    document.getElementById('proactive-whitelist-container').innerHTML = 
      '<div style="color:var(--danger);text-align:center;padding:20px;">获取白名单失败</div>';
    return;
  }
  renderProactiveWhitelist(resp.data.items || []);
}

function renderProactiveWhitelist(items) {
  const container = document.getElementById('proactive-whitelist-container');
  
  if (items.length === 0) {
    container.innerHTML = `
      <div style="color:var(--text2);text-align:center;padding:20px;">
        白名单为空，添加群聊 ID 以启用主动回复
      </div>
    `;
    return;
  }

  const html = items.map(groupId => `
    <div class="whitelist-item">
      <span class="whitelist-id">${escHtml(groupId)}</span>
      <button class="btn btn-danger btn-sm" onclick="removeProactiveWhitelist('${escHtml(groupId)}')">移除</button>
    </div>
  `).join('');

  container.innerHTML = `<div class="whitelist-list">${html}</div>`;
}

async function addProactiveWhitelist() {
  const input = document.getElementById('proactive-group-input');
  const groupId = input.value.trim();
  
  if (!groupId) {
    toast('请输入群聊 ID', 'error');
    return;
  }

  const resp = await api('/proactive/whitelist', {
    method: 'POST',
    body: JSON.stringify({ group_id: groupId })
  });

  if (resp && resp.status === 'ok') {
    toast(resp.data.message || '添加成功', 'success');
    input.value = '';
    loadProactiveWhitelist();
    loadProactiveStatus();
  } else {
    toast(resp?.message || '添加失败', 'error');
  }
}

async function removeProactiveWhitelist(groupId) {
  if (!confirm(`确定要从白名单移除群聊 ${groupId} 吗？`)) {
    return;
  }

  const resp = await api(`/proactive/whitelist/${encodeURIComponent(groupId)}`, {
    method: 'DELETE'
  });

  if (resp && resp.status === 'ok') {
    toast(resp.data.message || '移除成功', 'success');
    loadProactiveWhitelist();
    loadProactiveStatus();
  } else {
    toast(resp?.message || '移除失败', 'error');
  }
}

async function checkProactiveWhitelist() {
  const input = document.getElementById('proactive-check-input');
  const groupId = input.value.trim();
  const resultEl = document.getElementById('proactive-check-result');
  
  if (!groupId) {
    resultEl.innerHTML = '';
    return;
  }

  const resp = await api(`/proactive/whitelist/check?group_id=${encodeURIComponent(groupId)}`);
  
  if (resp && resp.status === 'ok') {
    const inList = resp.data.in_whitelist;
    resultEl.innerHTML = inList 
      ? '<span style="color:var(--success);">✓ 在白名单中</span>'
      : '<span style="color:var(--text2);">✗ 不在白名单中</span>';
  } else {
    resultEl.innerHTML = '<span style="color:var(--danger);">查询失败</span>';
  }
}

function refreshProactiveTab() {
  loadProactiveStatus();
}
