// ========== Dashboard Section ==========

async function loadDashboard() {
  const resp = await api('/dashboard');
  if (!resp || resp.status !== 'ok') return;
  const d = resp.data;

  const sys = d.system || {};
  const mem = d.memories || {};
  const kg = d.knowledge_graph || {};
  const health = d.health || {};

  const healthClass = health.status === 'healthy' ? 'healthy' : health.status === 'degraded' ? 'degraded' : 'unhealthy';

  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value">${mem.total_count || 0}</div>
      <div class="stat-label">总记忆数</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${sys.total_users || 0}</div>
      <div class="stat-label">用户数</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${kg.nodes || 0} / ${kg.edges || 0}</div>
      <div class="stat-label">KG 节点 / 边</div>
    </div>
    <div class="stat-card">
      <div class="stat-value"><span class="health-dot ${healthClass}"></span> ${health.status || 'unknown'}</div>
      <div class="stat-label">系统状态</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${mem.by_layer?.working || 0}</div>
      <div class="stat-label">工作记忆</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${mem.by_layer?.episodic || 0}</div>
      <div class="stat-label">情景记忆</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${mem.by_layer?.semantic || 0}</div>
      <div class="stat-label">语义记忆</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${sys.total_sessions || 0} / ${sys.active_sessions || 0}</div>
      <div class="stat-label">总/活跃会话</div>
    </div>
  `;

  const typeDiv = document.getElementById('type-distribution');
  const byType = mem.by_type || {};
  if (Object.keys(byType).length > 0) {
    typeDiv.innerHTML = Object.entries(byType).map(([t, c]) =>
      `<div class="stat-card" style="flex:1;min-width:120px;"><div class="stat-value" style="font-size:20px;">${c}</div><div class="stat-label">${typeLabels[t] || t}</div></div>`
    ).join('');
  } else {
    typeDiv.innerHTML = '<span style="color:var(--text2)">暂无数据</span>';
  }

  loadTrend();
}

async function loadTrend() {
  const days = document.getElementById('trend-days').value;
  const resp = await api(`/dashboard/trend?days=${days}`);
  if (!resp || resp.status !== 'ok') return;

  const data = resp.data || [];
  if (data.length === 0) {
    document.getElementById('trend-bars').innerHTML = '<span style="color:var(--text2);padding:40px;">暂无趋势数据</span>';
    document.getElementById('trend-labels').innerHTML = '';
    return;
  }

  const maxCount = Math.max(...data.map(d => d.count), 1);
  const barsHtml = data.map(d => {
    const h = Math.max(2, (d.count / maxCount) * 180);
    return `<div class="chart-bar" style="height:${h}px"><div class="tooltip">${d.date}: ${d.count}</div></div>`;
  }).join('');

  const step = Math.max(1, Math.floor(data.length / 10));
  const labelsHtml = data.map((d, i) => {
    const label = i % step === 0 ? d.date.slice(5) : '';
    return `<span>${label}</span>`;
  }).join('');

  document.getElementById('trend-bars').innerHTML = barsHtml;
  document.getElementById('trend-labels').innerHTML = labelsHtml;
}
