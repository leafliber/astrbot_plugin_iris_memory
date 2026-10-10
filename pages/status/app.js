const bridge = window.AstrBotPluginPage;
const el = (id) => document.getElementById(id);
const label = { connected: '已连接', unavailable: '服务暂不可用', disabled: '未启用或未配置' };
async function refresh() {
  el('refresh').disabled = true;
  try {
    const data = await bridge.apiGet('status');
    el('connection').replaceChildren();
    for (const [name, value] of [['Iris 服务', label[data.connection] || '未知'], ['令牌', data.token_configured ? '已配置' : '未配置']]) {
      const dt = document.createElement('dt'); dt.textContent = name;
      const dd = document.createElement('dd'); dd.textContent = value;
      el('connection').append(dt, dd);
    }
    el('management').textContent = data.management_url;
    el('management').href = data.management_url;
    el('health').textContent = data.health ? JSON.stringify(data.health, null, 2) : '尚未取得服务状态';
    el('rules').textContent = JSON.stringify(data.entry_rules, null, 2);
    el('warnings').replaceChildren(...data.warnings.map((text) => { const li = document.createElement('li'); li.textContent = text; return li; }));
    el('entries').replaceChildren(...data.entries.map((entry) => {
      const tr = document.createElement('tr');
      for (const value of [entry.origin, entry.entry_id, entry.kind]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
      return tr;
    }));
    el('notice').textContent = '状态已刷新';
  } catch { el('notice').textContent = '无法读取状态，请检查 AstrBot 登录和插件是否加载。'; }
  finally { el('refresh').disabled = false; }
}
el('refresh').addEventListener('click', refresh);
if (bridge) { await bridge.ready(); await refresh(); }
else { el('notice').textContent = '请从 AstrBot 插件 Pages 打开此页。'; }
