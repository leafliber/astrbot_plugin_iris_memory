function reminderTransport(value) {
  if (!value) return 'stopped · disabled';
  return `${value.mode} · ${value.reason}${value.hint ? ' · ' + value.hint : ''}`;
}
const bridge = window.AstrBotPluginPage;
const el = (id) => document.getElementById(id);
const label = { connected: '已连接', unavailable: '服务暂不可用', disabled: '未启用或未配置' };
async function refresh() {
  el('refresh').disabled = true;
  try {
    const data = await bridge.apiGet('status');
    el('connection').replaceChildren();
    for (const [name, value] of [['Iris 服务', label[data.connection] || '未知'], ['令牌', data.token_configured ? '已配置' : '未配置'], ['Docker 主机名', data.allow_container_host ? '已允许' : '未允许'], ['提醒通道', reminderTransport(data.reminder_transport)]]) {
      const dt = document.createElement('dt'); dt.textContent = name;
      const dd = document.createElement('dd'); dd.textContent = value;
      el('connection').append(dt, dd);
    }
    el('management').textContent = data.management_url;
    el('management').href = data.management_url;
    el('management-hint').textContent = data.allow_container_host ?
      'Docker 服务名通常只能在容器网络解析；请用宿主机映射端口打开管理页，例如在同机浏览器访问 http://127.0.0.1:8080。' :
      '在运行 Iris 的电脑浏览器中打开上面的地址。';
    el('health').textContent = data.health ? JSON.stringify(data.health, null, 2) : '尚未取得服务状态';
    el('rules').textContent = JSON.stringify(data.entry_rules, null, 2);
    el('warnings').replaceChildren(...data.warnings.map((text) => { const li = document.createElement('li'); li.textContent = text; return li; }));
    el('entries').replaceChildren(...data.entries.map((entry) => {
      const tr = document.createElement('tr');
      for (const value of [entry.origin, entry.entry_id, entry.kind, `${entry.entry_ref || "—"} / ${entry.session_ref || "—"}`]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
      return tr;
    }));
    renderProactive(data.proactive);
    el('notice').textContent = '状态已刷新';
  } catch { el('notice').textContent = '无法读取状态，请检查 AstrBot 登录和插件是否加载。'; }
  finally { el('refresh').disabled = false; }
}
function field(form, title, control) {
  const label = document.createElement('label');
  const span = document.createElement('span'); span.textContent = title;
  label.append(span, control); form.append(label); return control;
}
function input(type, value) {
  const node = document.createElement('input'); node.type = type; node.value = value; return node;
}
function renderProactive(data) {
  const root = el('proactive-sessions'); root.replaceChildren();
  el('proactive-summary').textContent = data ?
    `总开关：${data.enabled ? '开启' : '关闭'} · 群自动发起：${data.initiate_enabled ? '开启' : '关闭'} · 私聊关心：${data.private_care_enabled ? '开启' : '关闭'}` : 'Iris 未连接，暂不能设置主动回复。';
  if (!data) return;
  if (!data.sessions.length) { root.textContent = '尚无已观察到的会话，请先发送一条消息。'; return; }
  for (const session of data.sessions) {
    const card = document.createElement('article'); card.className = 'session-card';
    const heading = document.createElement('h3'); heading.textContent = session.origin;
    const info = document.createElement('p'); info.className = 'muted';
    info.textContent = `${session.entry_id} · ${session.live ? '本次启动已观察' : '等待新消息'}`;
    const form = document.createElement('form'); form.className = 'session-form';
    const enabled = field(form, '允许主动回复', input('checkbox', '')); enabled.checked = session.enabled;
    const willingness = document.createElement('select');
    for (const [value, text] of [['low', '低'], ['medium', '中'], ['high', '高']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; willingness.append(option);
    }
    willingness.value = session.willingness; field(form, '回复意愿', willingness);
    const cooldown = field(form, '冷却分钟（0 清除）', input('number', session.cooldown_minutes));
    cooldown.min = '0'; cooldown.max = '120'; cooldown.step = '1'; cooldown.required = true;
    const start = field(form, '静音开始', input('time', session.quiet_start)); start.required = true;
    const end = field(form, '静音结束', input('time', session.quiet_end)); end.required = true;
    const save = document.createElement('button'); save.type = 'submit'; save.textContent = '保存设置';
    const reset = document.createElement('button'); reset.type = 'button'; reset.textContent = '重置统计、限额与冷却';
    reset.className = 'secondary'; form.append(save, reset);
    const stats = document.createElement('pre');
    stats.textContent = `今日决策 ${session.calls_today}/${session.daily_limit} · 累计调用 ${session.calls} · 发言 ${session.sends}\n` +
      `滚动 24 小时发起 ${session.initiates_24h}/${session.initiate_limit} · 关注锚点 ${session.anchor_count}\n` +
      `最近结果：${session.last_result} · 预约：${session.pending_at ? new Date(session.pending_at * 1000).toLocaleString() : '无'}\n` +
      `调用分布：${JSON.stringify(session.calls_by_motive)}\n发言分布：${JSON.stringify(session.sends_by_motive)}`;
    async function update(changes) {
      save.disabled = reset.disabled = true;
      try { await bridge.apiPost('proactive', {origin: session.origin, changes}); await refresh(); }
      catch { el('notice').textContent = '保存失败，请检查登录、会话和参数。'; }
      finally { save.disabled = reset.disabled = false; }
    }
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (form.reportValidity()) update({enabled: enabled.checked, willingness: willingness.value,
        cooldown_minutes: Number(cooldown.value), quiet_start: start.value, quiet_end: end.value});
    });
    reset.addEventListener('click', () => update({reset: true}));
    card.append(heading, info, form, stats); root.append(card);
  }
}
el('refresh').addEventListener('click', refresh);
if (bridge) { await bridge.ready(); await refresh(); }
else { el('notice').textContent = '请从 AstrBot 插件 Pages 打开此页。'; }
