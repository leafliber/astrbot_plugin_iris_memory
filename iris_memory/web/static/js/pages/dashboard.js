/**
 * 统计面板页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtDuration, fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';

export async function loadDashboard() {
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = '<div class="stat-card"><div class="spinner"></div></div>';

  const res = await api.get('/dashboard/stats');
  if (!res || res.status !== 'ok') { grid.innerHTML = '<div class="stat-card">加载失败</div>'; return; }

  const d = res.data;
  const sys = d.system || {};
  const mem = d.memories || {};
  const kg = d.knowledge_graph || {};
  const health = d.health || {};

  grid.innerHTML = `
    <div class="stat-card"><div class="stat-value">${esc(mem.total ?? 0)}</div><div class="stat-label">记忆总数</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.working ?? 0)}</div><div class="stat-label">工作记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.episodic ?? 0)}</div><div class="stat-label">情景记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(mem.by_layer?.semantic ?? 0)}</div><div class="stat-label">语义记忆</div></div>
    <div class="stat-card"><div class="stat-value">${esc(sys.total_sessions ?? 0)}</div><div class="stat-label">会话数</div></div>
    <div class="stat-card"><div class="stat-value">${esc(sys.total_personas ?? 0)}</div><div class="stat-label">用户画像</div></div>
    <div class="stat-card"><div class="stat-value">${esc(kg.nodes ?? 0)}</div><div class="stat-label">图谱节点</div></div>
    <div class="stat-card"><div class="stat-value">${esc(kg.edges ?? 0)}</div><div class="stat-label">图谱边</div></div>
  `;

  const distEl = document.getElementById('type-distribution');
  if (distEl && mem.by_type) {
    const total = Object.values(mem.by_type).reduce((a, b) => a + b, 0) || 1;
    const labels = { fact: '事实', emotion: '情感', relationship: '关系', interaction: '交互', inferred: '推断' };
    const colors = { fact: 'var(--accent)', emotion: 'var(--danger)', relationship: 'var(--success)', interaction: 'var(--warning)', inferred: '#9d6cf0' };
    distEl.innerHTML = Object.entries(mem.by_type).map(([k, v]) =>
      `<div class="stat-card" style="flex:1;min-width:100px;border-left:3px solid ${colors[k] || 'var(--accent)'}">
        <div class="stat-value" style="font-size:20px;">${v}</div>
        <div class="stat-label">${esc(labels[k] || k)} (${Math.round(v / total * 100)}%)</div>
      </div>`
    ).join('');
  }

  loadSystemOverview();
  loadLlmOverview();
  loadTrend();
}

export async function loadTrend() {
  const days = Number(document.getElementById('trend-days')?.value || 30);
  const res = await api.get('/dashboard/trend', { days });
  if (!res || res.status !== 'ok') return;

  const items = res.data || [];
  if (!items.length) return;

  const maxVal = Math.max(...items.map(i => i.count), 1);
  const barsEl = document.getElementById('trend-bars');
  const labelsEl = document.getElementById('trend-labels');

  barsEl.innerHTML = items.map(i => {
    const h = Math.max(2, (i.count / maxVal) * 100);
    return `<div class="chart-bar" style="height:${h}%"><div class="tooltip">${esc(i.date)}: ${i.count}</div></div>`;
  }).join('');

  labelsEl.innerHTML = items.map((i, idx) => {
    const show = items.length <= 14 || idx % Math.ceil(items.length / 10) === 0;
    return `<span>${show ? esc(i.date?.slice(5) || '') : ''}</span>`;
  }).join('');
}

async function loadSystemOverview() {
  const container = el('system-overview-container');
  if (!container) return;

  container.innerHTML = '<div class="card"><div class="loading"><div class="spinner"></div></div></div>';

  const res = await api.get('/system/overview');
  let health = {};
  let storage = {};
  let pid = null;

  if (res?.status === 'ok') {
    health = res.data?.health || res.data || {};
    storage = res.data?.storage || {};
    pid = res.data?.pid;
  } else {
    const [healthRes, storageRes] = await Promise.all([
      api.get('/system/health'),
      api.get('/system/storage'),
    ]);
    health = healthRes?.data || {};
    storage = storageRes?.data || {};
  }

  const healthStatus = health.status || 'ok';
  const statusColor = healthStatus === 'healthy' || healthStatus === 'ok' ? 'var(--success)' :
    healthStatus === 'degraded' ? 'var(--warning)' : 'var(--danger)';
  const statusText = healthStatus === 'healthy' || healthStatus === 'ok' ? '健康' :
    healthStatus === 'degraded' ? '降级' : healthStatus || '未知';

  const chroma = storage.chroma || {};
  const kg = storage.kg || {};

  container.innerHTML = `
    <div class="card">
      <div class="card-title">🖥️ 系统状态</div>
      <div class="system-overview-grid">
        <div class="system-stat">
          <div class="system-stat-icon" style="background:${statusColor}20;color:${statusColor}">
            <span class="health-dot ${healthStatus === 'healthy' || healthStatus === 'ok' ? 'healthy' : 'unhealthy'}"></span>
          </div>
          <div class="system-stat-content">
            <div class="system-stat-value" style="color:${statusColor}">${esc(statusText)}</div>
            <div class="system-stat-label">系统状态</div>
          </div>
        </div>
        <div class="system-stat">
          <div class="system-stat-icon">⏱️</div>
          <div class="system-stat-content">
            <div class="system-stat-value">${health.uptime_seconds ? fmtDuration(health.uptime_seconds) : '-'}</div>
            <div class="system-stat-label">运行时间</div>
          </div>
        </div>
        <div class="system-stat">
          <div class="system-stat-icon">${health.initialized ? '✅' : '❌'}</div>
          <div class="system-stat-content">
            <div class="system-stat-value">${health.initialized ? '已初始化' : '未初始化'}</div>
            <div class="system-stat-label">初始化状态</div>
          </div>
        </div>
        <div class="system-stat">
          <div class="system-stat-icon">🔢</div>
          <div class="system-stat-content">
            <div class="system-stat-value">${pid ?? '-'}</div>
            <div class="system-stat-label">进程 PID</div>
          </div>
        </div>
      </div>
      <div class="storage-overview-grid">
        <div class="storage-item">
          <div class="storage-header">
            <span class="storage-icon">🗄️</span>
            <span class="storage-name">ChromaDB</span>
            <span class="storage-status ${chroma.ready ? 'ok' : 'err'}">${chroma.ready ? '就绪' : '离线'}</span>
          </div>
          ${chroma.ready ? `<div class="storage-detail">文档数: ${chroma.count ?? '-'}</div>` : ''}
          ${chroma.error ? `<div class="storage-error">${esc(chroma.error)}</div>` : ''}
        </div>
        <div class="storage-item">
          <div class="storage-header">
            <span class="storage-icon">🔗</span>
            <span class="storage-name">知识图谱</span>
            <span class="storage-status ${kg.enabled ? 'ok' : 'off'}">${kg.enabled ? '启用' : '禁用'}</span>
          </div>
          ${kg.enabled && kg.nodes != null ? `<div class="storage-detail">节点: ${kg.nodes} | 边: ${kg.edges ?? '-'}</div>` : ''}
          ${kg.error ? `<div class="storage-error">${esc(kg.error)}</div>` : ''}
        </div>
      </div>
      ${renderComponentHealthCompact(health.components)}
    </div>`;
}

function renderComponentHealthCompact(components) {
  if (!components || !Object.keys(components).length) return '';
  const entries = Object.entries(components).slice(0, 6);
  return `<div class="component-health-compact">
    ${entries.map(([k, v]) => {
      const ok = v.status === 'ok' || v.status === 'healthy';
      return `<div class="component-item">
        <span class="health-dot ${ok ? 'healthy' : 'unhealthy'}"></span>
        <span class="component-name">${esc(k)}</span>
      </div>`;
    }).join('')}
  </div>`;
}

async function loadLlmOverview() {
  const container = el('llm-overview-container');
  if (!container) return;

  container.innerHTML = '<div class="card"><div class="loading"><div class="spinner"></div></div></div>';

  try {
    const [sumRes, aggRes] = await Promise.all([
      api.get('/llm/summary'),
      api.get('/llm/aggregated'),
    ]);

    if (!sumRes || sumRes.status !== 'ok') {
      container.innerHTML = '<div class="card"><div class="card-title">🤖 LLM 监控</div><div style="text-align:center;color:var(--text2);padding:20px">LLM 统计不可用</div></div>';
      return;
    }

    const s = sumRes.data || {};
    const agg = aggRes?.data || {};

    if (s.available === false) {
      container.innerHTML = '<div class="card"><div class="card-title">🤖 LLM 监控</div><div style="text-align:center;color:var(--text2);padding:20px">LLM 统计未启用</div></div>';
      return;
    }

    container.innerHTML = `
      <div class="card">
        <div class="card-title">🤖 LLM 监控</div>
        <div class="llm-overview-grid">
          <div class="llm-stat">
            <div class="llm-stat-value">${s.total_calls ?? 0}</div>
            <div class="llm-stat-label">总调用</div>
          </div>
          <div class="llm-stat">
            <div class="llm-stat-value">${s.total_tokens ?? 0}</div>
            <div class="llm-stat-label">Token 消耗</div>
          </div>
          <div class="llm-stat">
            <div class="llm-stat-value" style="color:${getSuccessRateColor(s.success_rate)}">${s.success_rate != null ? (s.success_rate * 100).toFixed(1) + '%' : '-'}</div>
            <div class="llm-stat-label">成功率</div>
          </div>
          <div class="llm-stat">
            <div class="llm-stat-value">${s.avg_duration_ms != null ? s.avg_duration_ms.toFixed(0) + 'ms' : '-'}</div>
            <div class="llm-stat-label">平均耗时</div>
          </div>
        </div>
        ${renderProviderStatsCompact(agg)}
        ${renderSourceStatsCompact(agg)}
      </div>`;

    if (s.total_calls > 0) {
      loadLlmRecent();
    }
  } catch (e) {
    container.innerHTML = '<div class="card"><div class="card-title">🤖 LLM 监控</div><div style="text-align:center;color:var(--text2);padding:20px">加载失败</div></div>';
  }
}

function getSuccessRateColor(rate) {
  if (rate == null) return 'var(--text)';
  if (rate >= 0.95) return 'var(--success)';
  if (rate >= 0.8) return 'var(--warning)';
  return 'var(--danger)';
}

function renderProviderStatsCompact(agg) {
  const providers = agg.by_provider || agg.calls_by_provider;
  if (!providers || !Object.keys(providers).length) return '';
  const entries = Object.entries(providers).slice(0, 3);
  return `<div class="llm-provider-compact">
    <div class="compact-title">Provider</div>
    <div class="compact-items">${entries.map(([k, v]) => {
      const calls = typeof v === 'object' ? v.calls : v;
      return `<span class="compact-tag">${esc(k)}: ${calls}</span>`;
    }).join('')}</div>
  </div>`;
}

function renderSourceStatsCompact(agg) {
  const sources = agg.by_source || agg.calls_by_source;
  if (!sources || !Object.keys(sources).length) return '';
  const entries = Object.entries(sources).slice(0, 4);
  return `<div class="llm-source-compact">
    <div class="compact-title">调用来源</div>
    <div class="compact-items">${entries.map(([k, v]) => {
      const count = typeof v === 'object' ? v.calls : v;
      return `<span class="compact-tag">${esc(k)}: ${count}</span>`;
    }).join('')}</div>
  </div>`;
}

async function loadLlmRecent() {
  const res = await api.get('/llm/recent', { limit: 10 });
  const section = el('llm-recent-section');
  const container = el('llm-recent-container');
  
  if (!res || res.status !== 'ok' || !res.data?.length) {
    if (section) section.style.display = 'none';
    return;
  }

  if (section) section.style.display = 'block';
  const items = res.data;

  container.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>时间</th><th>来源</th><th>Provider</th><th>状态</th><th>Token</th><th>耗时</th></tr></thead>
    <tbody>${items.map(i => `<tr>
      <td>${esc(fmtTime(i.timestamp || i.created_at))}</td>
      <td>${esc(i.source || '-')}</td>
      <td>${esc(i.provider_id || '-')}</td>
      <td>${i.success ? '<span style="color:var(--success)">✓</span>' : '<span style="color:var(--danger)">✗</span>'}</td>
      <td>${i.tokens_used ?? '-'}</td>
      <td>${i.duration_ms != null ? i.duration_ms + 'ms' : '-'}</td>
    </tr>`).join('')}</tbody>
  </table></div>`;
}

function el(id) { return document.getElementById(id); }
