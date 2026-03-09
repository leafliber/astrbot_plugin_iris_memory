/**
 * 主动回复页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { toast } from '../components/toast.js';
import { showConfirm } from '../components/modal.js';

const state = { loaded: false };

export function getState() { return state; }

export async function loadProactiveStatus() {
  state.loaded = true;
  const container = el('proactive-status-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const res = await api.get('/proactive/status');
  if (!res || res.status !== 'ok') { container.innerHTML = '<div class="card">加载失败</div>'; return; }

  const d = res.data;
  renderStatus(d);
  loadWhitelist();
}

function renderStatus(d) {
  const container = el('proactive-status-container');
  const stats = d.stats || {};
  const config = d.config || {};

  container.innerHTML = `
    <div class="card">
      <div class="card-title">📊 主动回复状态</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value" style="color:${d.enabled ? 'var(--success)' : 'var(--danger)'}">
            ${d.enabled ? '已启用' : '已禁用'}
          </div>
          <div class="stat-label">功能状态</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${esc(stats.sent ?? 0)}</div>
          <div class="stat-label">已发送</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${esc(stats.skipped ?? 0)}</div>
          <div class="stat-label">已跳过</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${esc(stats.failed ?? 0)}</div>
          <div class="stat-label">失败</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${esc(stats.pending ?? 0)}</div>
          <div class="stat-label">待处理</div>
        </div>
      </div>
      <div style="font-size:13px;color:var(--text2);margin-top:8px;">
        白名单模式: <strong style="color:var(--text)">${d.whitelist_mode ? '是' : '否'}</strong>
        ${config.cooldown_seconds ? ` | 冷却: ${config.cooldown_seconds}s` : ''}
        ${config.max_signals_per_day ? ` | 日上限: ${config.max_signals_per_day}` : ''}
      </div>
    </div>`;
}

async function loadWhitelist() {
  const res = await api.get('/proactive/whitelist');
  if (!res || res.status !== 'ok') return;

  const groups = res.data?.groups || res.data || [];
  const container = el('proactive-whitelist-container');

  if (!groups.length) {
    container.innerHTML = '<div style="color:var(--text2);text-align:center;padding:20px">白名单为空</div>';
    return;
  }

  container.innerHTML = `<div class="whitelist-list">${groups.map(g => {
    const gid = typeof g === 'string' ? g : g.group_id || g;
    return `<div class="whitelist-item">
      <span class="whitelist-id">${esc(gid)}</span>
      <button class="btn btn-danger btn-sm" onclick="window.__proactive.removeWhitelist('${esc(gid)}')">移除</button>
    </div>`;
  }).join('')}</div>`;
}

export async function addWhitelist() {
  const gid = val('proactive-group-input');
  if (!gid) { toast.err('请输入群聊 ID'); return; }

  const res = await api.post('/proactive/whitelist', { group_id: gid });
  if (res?.status === 'ok') {
    toast.ok('已添加');
    el('proactive-group-input').value = '';
    loadWhitelist();
  } else toast.err(res?.message || '添加失败');
}

export function removeWhitelist(gid) {
  showConfirm('移除白名单', `确定要将 ${gid} 从白名单中移除吗？`, async () => {
    const res = await api.del(`/proactive/whitelist/${encodeURIComponent(gid)}`);
    if (res?.status === 'ok') { toast.ok('已移除'); loadWhitelist(); }
    else toast.err(res?.message || '移除失败');
  });
}

export async function checkWhitelist() {
  const gid = val('proactive-check-input');
  const resultEl = el('proactive-check-result');
  if (!gid) { resultEl.innerHTML = ''; return; }

  const res = await api.get(`/proactive/whitelist/${encodeURIComponent(gid)}/check`);
  if (res?.status === 'ok') {
    const inList = res.data?.in_whitelist;
    resultEl.innerHTML = inList
      ? '<span style="color:var(--success)">✓ 在白名单中</span>'
      : '<span style="color:var(--text2)">✗ 不在白名单中</span>';
  }
}

export function refreshProactiveTab() { loadProactiveStatus(); }

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }

window.__proactive = { removeWhitelist };
