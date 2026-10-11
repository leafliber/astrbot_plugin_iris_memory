const bridge = window.AstrBotPluginPage;
const el = (id) => document.getElementById(id);
const tabs = ['logs', 'prepares', 'sessions'];
const judgment = { applied: '已应用', disabled: '未启用', degraded: '降级', unknown: '未报告' };
const outcomes = { ok: '返回成功', failed: '失败', cancelled: '取消／超时' };
const stages = { message: '消息门控', scheduler: '定时检查', decision_gate: '决策入口门控', calls: '模型调用（动机）', sends: '成功发言（动机）', outcome: '决策结果', reminder: '提醒门控' };
const partitions = { persona: '人格', memories: '记忆', recent_messages: '近期消息', state: '状态', goals: '目标', hints: '提示', entry_profile: '入口画像' };
function activate(name) {
  for (const tab of tabs) {
    el(`diagnostics-panel-${tab}`).hidden = tab !== name;
    el(`diagnostics-tab-${tab}`).setAttribute('aria-selected', String(tab === name));
    el(`diagnostics-tab-${tab}`).setAttribute('tabindex', tab === name ? '0' : '-1');
  }
}
for (const [index, tab] of tabs.entries()) {
  const button = el(`diagnostics-tab-${tab}`);
  button.addEventListener('click', () => activate(tab));
  button.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    activate(tabs[next]); el(`diagnostics-tab-${tabs[next]}`).focus();
  });
}
function rows(id, records, columns, empty) {
  const body = el(id); body.replaceChildren();
  if (!records.length) {
    const tr = document.createElement('tr'); const td = document.createElement('td');
    td.setAttribute('colspan', columns.length); td.textContent = empty; tr.append(td); body.append(tr); return;
  }
  for (const record of records) {
    const tr = document.createElement('tr');
    for (const column of columns) {
      const td = document.createElement('td'); td.textContent = column(record); tr.append(td);
    }
    body.append(tr);
  }
}
function render(data) {
  el('diagnostics-summary').textContent = data.enabled ?
    `本次采集开始于 ${data.started_at}；最多保留 ${data.limits.logs} 条日志、${data.limits.prepares} 次 prepare、${data.limits.sessions} 个活跃诊断会话。重启清空，页面刷新不调用 Iris。` :
    '诊断采集已关闭；在插件配置开启 diagnostics_enabled 后重载。';
  rows('diagnostics-logs', [...data.logs].reverse(), [r => r.at, r => r.code], '尚无运行事件。');
  rows('diagnostics-prepares', [...data.prepares].reverse(), [
    r => r.at,
    r => r.entry_ref,
    r => `${r.purpose === 'proactive' ? '主动决策' : '普通回复'} · ${outcomes[r.outcome] || '未知'}`,
    r => `${r.duration_ms} ms`,
    r => judgment[r.judgment] || '未报告',
    r => Object.entries(partitions).map(([key, title]) => `${title} ${r.partitions[key] || 0}`).join(' · '),
    r => r.entry_profile_version == null ? '无版本' : `v${r.entry_profile_version}${r.entry_profile_stale ? ' · 可能过时' : ''}`,
  ], '尚无 prepare 请求。');
  const root = el('diagnostics-sessions'); root.replaceChildren();
  if (!data.sessions.length) root.textContent = '尚无主动门控事件。';
  for (const session of data.sessions) {
    const article = document.createElement('article'); article.className = 'session-card';
    const heading = document.createElement('h3'); heading.textContent = `会话 ${session.session_ref} · 入口 ${session.entry_ref}`;
    const since = document.createElement('p'); since.className = 'muted'; since.textContent = `本轮计数开始于 ${session.since}`;
    const counts = document.createElement('pre');
    counts.textContent = Object.entries(stages).map(([stage, title]) =>
      `${title}：${Object.entries(session.counts[stage] || {}).map(([key, value]) => `${key}=${value}`).join('，') || '0'}`).join('\n');
    article.append(heading, since, counts); root.append(article);
  }
}
async function refreshDiagnostics() {
  el('diagnostics-refresh').disabled = true;
  try { render(await bridge.apiGet('diagnostics')); el('diagnostics-notice').textContent = '诊断已刷新'; }
  catch { el('diagnostics-notice').textContent = '无法读取诊断，请检查 AstrBot 登录与插件；已有内容来自上次刷新。'; }
  finally { el('diagnostics-refresh').disabled = false; }
}
el('diagnostics-refresh').addEventListener('click', refreshDiagnostics);
activate('logs');
if (bridge) {
  try { await bridge.ready(); await refreshDiagnostics(); }
  catch { el('diagnostics-notice').textContent = '无法读取诊断，请从 AstrBot 插件 Pages 打开。'; }
} else { el('diagnostics-notice').textContent = '请从 AstrBot 插件 Pages 打开此页。'; }
