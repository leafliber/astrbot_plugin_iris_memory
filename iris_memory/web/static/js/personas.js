// ========== User Personas Section ==========

const personaState = { page: 1, pageSize: 12, loaded: false, currentQuery: '' };

async function searchPersonas() {
  const query = document.getElementById('persona-query').value;
  personaState.currentQuery = query;

  let path = `/personas?page=${personaState.page}&page_size=${personaState.pageSize}`;
  if (query) {
    path += `&q=${encodeURIComponent(query)}`;
  }

  const container = document.getElementById('personas-container');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

  const resp = await api(path);
  personaState.loaded = true;

  if (!resp || resp.status !== 'ok') {
    container.innerHTML = '<div style="color:var(--text2);text-align:center;padding:40px;">获取画像失败</div>';
    return;
  }

  const data = resp.data;
  const personas = data.items || [];

  document.getElementById('persona-total-info').textContent = `共 ${data.total} 个用户画像`;

  if (personas.length === 0) {
    const emptyMsg = query
      ? `未找到匹配 "${escHtml(query)}" 的用户画像`
      : '暂无用户画像数据<br><span style="font-size:13px;">当用户与 Bot 交互后将自动生成画像</span>';
    container.innerHTML = `<div style="color:var(--text2);text-align:center;padding:40px;">${emptyMsg}</div>`;
    document.getElementById('persona-pagination').innerHTML = '';
    return;
  }

  container.innerHTML = personas.map(p => `
    <div class="persona-card" onclick="showPersonaDetail('${escHtml(p.user_id)}')">
      <div class="persona-header">
        <span class="persona-uid">◉ ${highlightText(p.user_id, query)}</span>
        <span class="persona-meta">更新 ${p.update_count} 次</span>
      </div>
      <div style="font-size:12px;color:var(--text2);margin-bottom:8px;">
        最后更新: ${p.last_updated ? new Date(p.last_updated).toLocaleString('zh-CN') : '—'}
      </div>
      <div style="margin-bottom:8px;">
        <div style="font-size:12px;color:var(--text2);margin-bottom:4px;">信任度</div>
        <div class="persona-bar"><div class="persona-bar-fill" style="width:${(p.trust_level||0.5)*100}%;background:var(--accent);"></div></div>
      </div>
      <div style="margin-bottom:8px;">
        <div style="font-size:12px;color:var(--text2);margin-bottom:4px;">亲密度</div>
        <div class="persona-bar"><div class="persona-bar-fill" style="width:${(p.intimacy_level||0.5)*100}%;background:var(--success);"></div></div>
      </div>
      ${p.work_style ? `<div style="font-size:12px;"><strong>工作风格:</strong> ${highlightText(p.work_style, query)}</div>` : ''}
      ${p.lifestyle ? `<div style="font-size:12px;"><strong>生活方式:</strong> ${highlightText(p.lifestyle, query)}</div>` : ''}
      ${Object.keys(p.interests || {}).length > 0 ? `
        <div class="persona-traits">
          ${Object.entries(p.interests).slice(0,5).map(([k,v]) => `<span class="persona-trait">${highlightText(k, query)}</span>`).join('')}
        </div>
      ` : ''}
    </div>
  `).join('');

  // Pagination
  const totalPages = Math.ceil(data.total / personaState.pageSize);
  let pagHtml = '';
  if (totalPages > 1) {
    pagHtml += `<button class="btn btn-outline btn-sm" ${personaState.page<=1?'disabled':''} onclick="personaPage(1)">首页</button>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${personaState.page<=1?'disabled':''} onclick="personaPage(${personaState.page-1})">上一页</button>`;
    pagHtml += `<span class="page-info">${personaState.page} / ${totalPages}</span>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${personaState.page>=totalPages?'disabled':''} onclick="personaPage(${personaState.page+1})">下一页</button>`;
    pagHtml += `<button class="btn btn-outline btn-sm" ${personaState.page>=totalPages?'disabled':''} onclick="personaPage(${totalPages})">末页</button>`;
  }
  document.getElementById('persona-pagination').innerHTML = pagHtml;
}

function personaPage(p) {
  personaState.page = p;
  searchPersonas();
}

function resetPersonaFilters() {
  document.getElementById('persona-query').value = '';
  personaState.page = 1;
  searchPersonas();
}

function highlightText(text, query) {
  if (!query || !text) return escHtml(text);
  const escaped = escHtml(text);
  const queryEsc = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  try {
    return escaped.replace(new RegExp(`(${queryEsc})`, 'gi'), '<span class="highlight">$1</span>');
  } catch(e) { return escaped; }
}

// 保持向后兼容
async function loadPersonas() {
  await searchPersonas();
}

async function showPersonaDetail(userId) {
  const resp = await api(`/personas/detail?user_id=${encodeURIComponent(userId)}`);
  if (!resp || resp.status !== 'ok') { toast('获取画像详情失败', 'error'); return; }

  const p = resp.data;
  const container = document.getElementById('persona-detail-content');

  // Also try to get emotion state
  const emotionResp = await api(`/emotions?user_id=${encodeURIComponent(userId)}`);
  const emotion = (emotionResp && emotionResp.status === 'ok') ? emotionResp.data : null;

  let html = `<div style="margin-bottom:16px;font-size:14px;color:var(--text2);">用户: <strong style="color:var(--text);">${escHtml(userId)}</strong></div>`;

  // Big Five personality
  html += `<div class="card" style="margin-bottom:12px;">
    <div class="card-title">⬠ 人格特质 (Big Five)</div>
    <div style="display:grid;gap:8px;">
      ${renderPersonalityBar('开放性', p.personality_openness)}
      ${renderPersonalityBar('尽责性', p.personality_conscientiousness)}
      ${renderPersonalityBar('外向性', p.personality_extraversion)}
      ${renderPersonalityBar('亲和性', p.personality_agreeableness)}
      ${renderPersonalityBar('神经质', p.personality_neuroticism)}
    </div>
  </div>`;

  // Communication style
  html += `<div class="card" style="margin-bottom:12px;">
    <div class="card-title">⬡ 沟通风格</div>
    <div style="display:grid;gap:8px;">
      ${renderPersonalityBar('正式度', p.communication_formality)}
      ${renderPersonalityBar('直接度', p.communication_directness)}
      ${renderPersonalityBar('幽默感', p.communication_humor)}
      ${renderPersonalityBar('共情力', p.communication_empathy)}
    </div>
  </div>`;

  // Interests
  if (p.interests && Object.keys(p.interests).length > 0) {
    html += `<div class="card" style="margin-bottom:12px;">
      <div class="card-title">◇ 兴趣偏好</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
        ${Object.entries(p.interests).map(([k,v]) =>
          `<span class="persona-trait" style="font-size:13px;">${escHtml(k)}: ${(v||0).toFixed(1)}</span>`
        ).join('')}
      </div>
    </div>`;
  }

  // Relationship
  html += `<div class="card" style="margin-bottom:12px;">
    <div class="card-title">⬢ 关系指标</div>
    <div class="detail-grid">
      <div class="detail-item"><div class="detail-label">信任度</div><div class="detail-value">${(p.trust_level||0).toFixed(2)}</div></div>
      <div class="detail-item"><div class="detail-label">亲密度</div><div class="detail-value">${(p.intimacy_level||0).toFixed(2)}</div></div>
      <div class="detail-item"><div class="detail-label">社交风格</div><div class="detail-value">${escHtml(p.social_style||'—')}</div></div>
      <div class="detail-item"><div class="detail-label">情感基线</div><div class="detail-value">${escHtml(p.emotional_baseline||'neutral')}</div></div>
    </div>
  </div>`;

  // Work & Life
  html += `<div class="card" style="margin-bottom:12px;">
    <div class="card-title">◈ 工作与生活</div>
    <div class="detail-grid">
      <div class="detail-item"><div class="detail-label">工作风格</div><div class="detail-value">${escHtml(p.work_style||'—')}</div></div>
      <div class="detail-item"><div class="detail-label">生活方式</div><div class="detail-value">${escHtml(p.lifestyle||'—')}</div></div>
    </div>
    ${(p.work_goals||[]).length > 0 ? `<div style="margin-top:8px;"><strong style="font-size:12px;color:var(--text2);">工作目标:</strong> ${p.work_goals.map(g => escHtml(g)).join(', ')}</div>` : ''}
    ${(p.habits||[]).length > 0 ? `<div style="margin-top:4px;"><strong style="font-size:12px;color:var(--text2);">习惯:</strong> ${p.habits.map(h => escHtml(h)).join(', ')}</div>` : ''}
  </div>`;

  // Emotion state
  if (emotion) {
    const cur = emotion.current || {};
    const emotionColors = { happy:'#36c78d', sad:'#7c6cf0', angry:'#ef4565', fear:'#f5a623', surprise:'#9d6cf0', disgust:'#5c6082', neutral:'#8b8fa8' };
    const ec = emotionColors[cur.primary] || '#aab8c2';

    html += `<div class="card" style="margin-bottom:12px;">
      <div class="card-title">∿ 情感状态</div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
        <div class="emotion-primary" style="color:${ec};">${escHtml(cur.primary || 'neutral')}</div>
        <div style="flex:1;">
          <div style="font-size:12px;color:var(--text2);">强度: ${(cur.intensity||0).toFixed(2)}</div>
          <div class="emotion-intensity"><div class="emotion-intensity-fill" style="width:${(cur.intensity||0)*100}%;background:${ec};"></div></div>
        </div>
      </div>
      ${emotion.trajectory ? `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">趋势</div><div class="detail-value">${escHtml(emotion.trajectory.trend||'stable')}</div></div>
          <div class="detail-item"><div class="detail-label">波动性</div><div class="detail-value">${(emotion.trajectory.volatility||0).toFixed(2)}</div></div>
        </div>
      ` : ''}
    </div>`;
  }

  // Metadata
  html += `<div class="card">
    <div class="card-title">≡ 元数据</div>
    <div class="detail-grid">
      <div class="detail-item"><div class="detail-label">版本</div><div class="detail-value">${p.version || 1}</div></div>
      <div class="detail-item"><div class="detail-label">更新次数</div><div class="detail-value">${p.update_count || 0}</div></div>
      <div class="detail-item"><div class="detail-label">最后更新</div><div class="detail-value">${p.last_updated ? new Date(p.last_updated).toLocaleString('zh-CN') : '—'}</div></div>
      <div class="detail-item"><div class="detail-label">主动回复偏好</div><div class="detail-value">${(p.proactive_reply_preference||0.5).toFixed(2)}</div></div>
    </div>
  </div>`;

  container.innerHTML = html;
  document.getElementById('persona-modal').classList.add('show');
}

function renderPersonalityBar(label, value) {
  const v = value || 0.5;
  const pct = v * 100;
  const color = v > 0.7 ? 'var(--success)' : v < 0.3 ? 'var(--danger)' : 'var(--accent)';
  return `<div style="display:flex;align-items:center;gap:8px;">
    <span style="width:60px;font-size:12px;color:var(--text2);">${label}</span>
    <div class="persona-bar" style="flex:1;"><div class="persona-bar-fill" style="width:${pct}%;background:${color};"></div></div>
    <span style="width:40px;text-align:right;font-size:12px;">${v.toFixed(2)}</span>
  </div>`;
}
