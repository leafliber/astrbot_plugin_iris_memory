/**
 * 用户画像页面
 */
import { api } from '../api/client.js';
import { esc } from '../utils/escape.js';
import { fmtTime } from '../utils/format.js';
import { toast } from '../components/toast.js';
import { showConfirm, closeModal, showDetailModal } from '../components/modal.js';
import { renderPagination } from '../components/pagination.js';

const state = { page: 1, pageSize: 12, total: 0, loaded: false };

export function getState() { return state; }

export async function searchPersonas() {
  state.loaded = true;
  const q = val('persona-query');

  const res = await api.get('/personas', { query: q, page: state.page, page_size: state.pageSize });
  if (!res || res.status !== 'ok') return;

  const d = res.data;
  state.total = d.total || 0;
  el('persona-total-info').textContent = `共 ${state.total} 位用户`;

  const items = d.items || [];
  const container = el('personas-container');

  if (!items.length) {
    container.innerHTML = '<div style="text-align:center;color:var(--text2);padding:40px">暂无用户画像数据</div>';
    el('persona-pagination').innerHTML = '';
    return;
  }

  container.innerHTML = items.map(p => {
    const importantTraits = [];
    if (p.interests) {
      const entries = typeof p.interests === 'object' ? Object.keys(p.interests).slice(0, 3) : [];
      importantTraits.push(...entries);
    }
    if (p.work_style) importantTraits.push(p.work_style);
    if (p.lifestyle) importantTraits.push(p.lifestyle);

    const displayName = p.display_name || p.user_id;
    const lastUpdated = fmtTime(p.last_updated);

    return `<div class="persona-card" onclick="window.__persona.showDetail('${esc(p.user_id)}')">
      <div class="persona-header">
        <div class="persona-identity">
          <span class="persona-name">${esc(displayName)}</span>
          ${p.display_name ? `<span class="persona-uid-small">${esc(p.user_id)}</span>` : ''}
        </div>
        <div class="persona-time-badge">${esc(lastUpdated)}</div>
      </div>
      <div class="persona-stats-row">
        <div class="persona-stat">
          <span class="stat-icon">📊</span>
          <span class="stat-value">${p.update_count ?? 0}</span>
          <span class="stat-label">次更新</span>
        </div>
        <div class="persona-stat">
          <span class="stat-icon">🎯</span>
          <span class="stat-value">${p.all_interests_count ?? 0}</span>
          <span class="stat-label">个兴趣</span>
        </div>
        <div class="persona-stat">
          <span class="stat-icon">💭</span>
          <span class="stat-value">${esc(p.emotional_baseline || 'neutral')}</span>
        </div>
      </div>
      ${renderMiniPersonality(p.personality)}
      ${importantTraits.length ? `<div class="persona-traits">${importantTraits.slice(0, 4).map(t => `<span class="persona-trait">${esc(t)}</span>`).join('')}</div>` : ''}
    </div>`;
  }).join('');

  renderPagination({
    page: state.page, pageSize: state.pageSize, total: state.total,
    onChange: p => { state.page = p; searchPersonas(); },
    container: el('persona-pagination'),
  });
}

export function loadPersonas() { searchPersonas(); }

export function resetPersonaFilters() {
  el('persona-query').value = '';
  state.page = 1;
  searchPersonas();
}

export function changePageSize(v) { state.pageSize = Number(v); state.page = 1; searchPersonas(); }

export async function showDetail(userId) {
  const res = await api.get(`/personas/${encodeURIComponent(userId)}`);
  if (!res || res.status !== 'ok') { toast.err('无法加载画像详情'); return; }
  const p = res.data;

  const displayName = p.display_name || userId;
  const lastUpdated = fmtTime(p.last_updated);

  const html = `
    <div class="detail-header">
      <div class="detail-avatar">${esc(displayName.charAt(0).toUpperCase())}</div>
      <div class="detail-info">
        <h3>${esc(displayName)}</h3>
        ${p.display_name ? `<div class="detail-user-id">${esc(userId)}</div>` : ''}
        <div class="detail-meta">
          <span>v${p.version ?? 3}</span>
          <span>•</span>
          <span>${p.update_count ?? 0} 次更新</span>
          <span>•</span>
          <span>${esc(lastUpdated)}</span>
        </div>
      </div>
    </div>
    
    ${renderQuickStats(p)}
    
    <div class="detail-sections">
      ${renderPersonalitySection(p.personality)}
      ${renderCommunicationSection(p.communication_style)}
      ${renderInterestsSection(p.interests)}
      ${renderRelationshipSection(p)}
      ${renderWorkLifeSection(p)}
      ${renderEmotionSection(p)}
      ${renderBehaviorSection(p)}
      ${renderPreferencesSection(p)}
      ${renderMetaSection(p)}
    </div>
    
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="window.__persona.closeDetail()">关闭</button>
      <button class="btn btn-danger" onclick="window.__persona.deletePersona('${esc(userId)}')">删除画像</button>
    </div>`;
  showDetailModal('persona-detail-modal', html);
}

export function closeDetail() { closeModal('persona-detail-modal'); }

export function deletePersona(uid) {
  showConfirm('删除画像', `确定要删除 ${uid} 的用户画像吗？`, async () => {
    const res = await api.del(`/personas/${encodeURIComponent(uid)}`);
    if (res?.status === 'ok') { toast.ok('已删除'); closeDetail(); searchPersonas(); }
    else toast.err(res?.message || '删除失败');
  });
}

// ── 渲染组件 ──

function renderQuickStats(p) {
  const stats = [
    { icon: '🎯', label: '兴趣', value: p.interests ? Object.keys(p.interests).length : 0 },
    { icon: '🤝', label: '信任', value: formatPercent(p.trust_level), isPercent: true },
    { icon: '💕', label: '亲密', value: formatPercent(p.intimacy_level), isPercent: true },
    { icon: '💭', label: '情绪', value: p.emotional_baseline || 'neutral' },
  ];
  return `<div class="quick-stats">${stats.map(s => `
    <div class="quick-stat">
      <span class="quick-stat-icon">${s.icon}</span>
      <div class="quick-stat-content">
        <span class="quick-stat-value">${s.isPercent ? s.value : esc(String(s.value))}</span>
        <span class="quick-stat-label">${s.label}</span>
      </div>
    </div>
  `).join('')}</div>`;
}

function renderMiniPersonality(p) {
  if (!p) return '';
  const bars = [
    ['O', p.openness, '开放性'], ['C', p.conscientiousness, '尽责性'], ['E', p.extraversion, '外向性'],
    ['A', p.agreeableness, '宜人性'], ['N', p.neuroticism, '神经质'],
  ].filter(([, v]) => v != null);
  if (!bars.length) return '';
  return `<div style="display:flex;gap:4px;margin:6px 0">${bars.map(([l, v, title]) =>
    `<div style="flex:1;text-align:center" title="${title}">
      <div style="font-size:10px;color:var(--text2)">${l}</div>
      <div class="persona-bar"><div class="persona-bar-fill" style="width:${(v * 100).toFixed(0)}%;background:${getBarColor(v)}"></div></div>
    </div>`
  ).join('')}</div>`;
}

function renderPersonalitySection(p) {
  if (!p) return '';
  const items = [
    ['开放性', p.openness, '好奇心、创造力、尝试新事物'],
    ['尽责性', p.conscientiousness, '自律、组织性、可靠性'],
    ['外向性', p.extraversion, '社交活跃度、精力来源'],
    ['宜人性', p.agreeableness, '合作性、同理心、信任'],
    ['神经质', p.neuroticism, '情绪稳定性、焦虑倾向'],
  ];
  const validItems = items.filter(([, val]) => val != null);
  if (!validItems.length) return '';
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">🧠</span>大五人格</div>
    <div class="personality-grid">
      ${validItems.map(([label, val, desc]) => `
        <div class="personality-item">
          <div class="personality-header">
            <span class="personality-label">${esc(label)}</span>
            <span class="personality-value">${formatPercent(val)}</span>
          </div>
          <div class="personality-bar-wrap">
            <div class="personality-bar-bg">
              <div class="personality-bar-fill" style="width:${(val * 100).toFixed(0)}%;background:${getBarColor(val)}"></div>
            </div>
          </div>
          <div class="personality-desc">${esc(desc)}</div>
        </div>
      `).join('')}
    </div>
  </div>`;
}

function renderCommunicationSection(cs) {
  if (!cs) return '';
  const items = [
    ['正式度', cs.formality, '非正式 ←→ 正式'],
    ['直接性', cs.directness, '委婉 ←→ 直接'],
    ['幽默感', cs.humor, '严肃 ←→ 幽默'],
    ['同理心', cs.empathy, '理性 ←→ 感性'],
  ];
  const validItems = items.filter(([, v]) => v != null);
  if (!validItems.length) return '';
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">💬</span>沟通风格</div>
    <div class="comm-grid">
      ${validItems.map(([label, val, scale]) => `
        <div class="comm-item">
          <div class="comm-header">
            <span class="comm-label">${esc(label)}</span>
            <span class="comm-value">${formatPercent(val)}</span>
          </div>
          <div class="comm-bar-wrap">
            <div class="comm-bar-bg">
              <div class="comm-bar-fill" style="width:${(val * 100).toFixed(0)}%;background:${getBarColor(val)}"></div>
            </div>
          </div>
          <div class="comm-scale">${esc(scale)}</div>
        </div>
      `).join('')}
    </div>
  </div>`;
}

function renderInterestsSection(interests) {
  if (!interests || !Object.keys(interests).length) return '';
  const sorted = Object.entries(interests).sort((a, b) => b[1] - a[1]);
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">🎯</span>兴趣爱好</div>
    <div class="interests-grid">
      ${sorted.map(([name, weight]) => `
        <div class="interest-item">
          <div class="interest-content">
            <span class="interest-name">${esc(name)}</span>
            <span class="interest-weight">${formatPercent(weight)}</span>
          </div>
          <div class="interest-bar">
            <div class="interest-bar-fill" style="width:${(weight * 100).toFixed(0)}%;background:${getBarColor(weight)}"></div>
          </div>
        </div>
      `).join('')}
    </div>
  </div>`;
}

function renderRelationshipSection(p) {
  const items = [];
  if (p.trust_level != null) items.push(['信任度', p.trust_level]);
  if (p.intimacy_level != null) items.push(['亲密度', p.intimacy_level]);
  
  const textItems = [];
  if (p.social_style) textItems.push(['社交风格', p.social_style]);
  
  if (!items.length && !textItems.length) return '';
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">🤝</span>关系维度</div>
    ${items.length ? `
      <div class="relationship-grid">
        ${items.map(([label, val]) => `
          <div class="relationship-item">
            <div class="relationship-header">
              <span class="relationship-label">${esc(label)}</span>
              <span class="relationship-value">${formatPercent(val)}</span>
            </div>
            <div class="relationship-bar">
              <div class="relationship-bar-fill" style="width:${(val * 100).toFixed(0)}%;background:${getBarColor(val)}"></div>
            </div>
          </div>
        `).join('')}
      </div>
    ` : ''}
    ${textItems.length ? `
      <div class="text-items">
        ${textItems.map(([label, val]) => `
          <div class="text-item">
            <span class="text-label">${esc(label)}</span>
            <span class="text-value">${esc(val)}</span>
          </div>
        `).join('')}
      </div>
    ` : ''}
  </div>`;
}

function renderWorkLifeSection(p) {
  const sections = [];
  
  if (p.work_style) {
    sections.push(`<div class="wl-item"><span class="wl-icon">💼</span><div class="wl-content"><span class="wl-label">工作风格</span><span class="wl-value">${esc(p.work_style)}</span></div></div>`);
  }
  if (p.lifestyle) {
    sections.push(`<div class="wl-item"><span class="wl-icon">🏠</span><div class="wl-content"><span class="wl-label">生活方式</span><span class="wl-value">${esc(p.lifestyle)}</span></div></div>`);
  }
  if (p.work_goals?.length) {
    sections.push(`<div class="wl-item"><span class="wl-icon">🎯</span><div class="wl-content"><span class="wl-label">工作目标</span><div class="wl-tags">${p.work_goals.map(g => `<span class="wl-tag">${esc(g)}</span>`).join('')}</div></div></div>`);
  }
  if (p.work_challenges?.length) {
    sections.push(`<div class="wl-item"><span class="wl-icon">⚡</span><div class="wl-content"><span class="wl-label">工作挑战</span><div class="wl-tags">${p.work_challenges.map(c => `<span class="wl-tag wl-tag-warning">${esc(c)}</span>`).join('')}</div></div></div>`);
  }
  if (p.habits?.length) {
    sections.push(`<div class="wl-item"><span class="wl-icon">🔄</span><div class="wl-content"><span class="wl-label">生活习惯</span><div class="wl-tags">${p.habits.map(h => `<span class="wl-tag">${esc(h)}</span>`).join('')}</div></div></div>`);
  }
  
  if (!sections.length) return '';
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">🏠</span>工作与生活</div>
    <div class="work-life-grid">${sections.join('')}</div>
  </div>`;
}

function renderEmotionSection(p) {
  const emotion = {
    baseline: p.emotional_baseline,
    volatility: p.emotional_volatility,
    trajectory: p.emotional_trajectory,
    triggers: p.emotional_triggers,
    soothers: p.emotional_soothers,
  };
  
  if (!emotion.baseline && !emotion.volatility) return '';
  
  const emotionEmoji = getEmotionEmoji(emotion.baseline);
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">🎭</span>情感状态</div>
    <div class="emotion-display">
      <div class="emotion-main">
        <div class="emotion-icon">${emotionEmoji}</div>
        <div class="emotion-info">
          <div class="emotion-label">当前情绪</div>
          <div class="emotion-value">${esc(emotion.baseline || 'neutral')}</div>
        </div>
      </div>
      ${emotion.volatility != null ? `
        <div class="emotion-stat">
          <span class="emotion-stat-label">波动性</span>
          <div class="emotion-stat-bar">
            <div class="emotion-stat-fill" style="width:${(emotion.volatility * 100).toFixed(0)}%;background:${getVolatilityColor(emotion.volatility)}"></div>
          </div>
          <span class="emotion-stat-value">${formatPercent(emotion.volatility)}</span>
        </div>
      ` : ''}
      ${emotion.trajectory ? `
        <div class="emotion-trajectory">
          <span class="trajectory-label">趋势</span>
          <span class="trajectory-value">${getTrajectoryIcon(emotion.trajectory)} ${esc(emotion.trajectory)}</span>
        </div>
      ` : ''}
      ${emotion.triggers?.length ? `
        <div class="emotion-triggers">
          <span class="triggers-label">情感触发点</span>
          <div class="triggers-list">${emotion.triggers.slice(0, 5).map(t => `<span class="trigger-tag">${esc(t)}</span>`).join('')}</div>
        </div>
      ` : ''}
    </div>
  </div>`;
}

function renderBehaviorSection(p) {
  const hasHourly = p.hourly_distribution && p.hourly_distribution.some(v => v > 0);
  if (!hasHourly) return '';
  
  const maxVal = Math.max(...p.hourly_distribution);
  const hours = p.hourly_distribution.map((v, i) => ({
    hour: i,
    value: v,
    height: maxVal > 0 ? (v / maxVal * 100) : 0
  }));
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">📊</span>活跃时段</div>
    <div class="activity-chart">
      <div class="chart-bars">
        ${hours.map(h => `
          <div class="chart-bar-col" title="${h.hour}:00 - ${h.value.toFixed(2)}">
            <div class="chart-bar-fill" style="height:${h.height}%"></div>
          </div>
        `).join('')}
      </div>
      <div class="chart-labels">
        ${[0, 6, 12, 18, 23].map(i => `<span>${i}:00</span>`).join('')}
      </div>
    </div>
  </div>`;
}

function renderPreferencesSection(p) {
  const items = [];
  
  if (p.proactive_reply_preference != null) {
    items.push(`<div class="pref-item">
      <span class="pref-label">主动回复偏好</span>
      <div class="pref-bar">
        <div class="pref-bar-fill" style="width:${(p.proactive_reply_preference * 100).toFixed(0)}%"></div>
      </div>
      <span class="pref-value">${formatPercent(p.proactive_reply_preference)}</span>
    </div>`);
  }
  
  if (p.preferred_reply_style) {
    items.push(`<div class="pref-item">
      <span class="pref-label">回复风格偏好</span>
      <span class="pref-tag">${esc(p.preferred_reply_style)}</span>
    </div>`);
  }
  
  if (p.topic_blacklist?.length) {
    items.push(`<div class="pref-item">
      <span class="pref-label">话题黑名单</span>
      <div class="pref-tags">${p.topic_blacklist.map(t => `<span class="pref-tag-black">${esc(t)}</span>`).join('')}</div>
    </div>`);
  }
  
  if (!items.length) return '';
  
  return `<div class="detail-section">
    <div class="section-title"><span class="section-icon">⚙️</span>交互偏好</div>
    <div class="preferences-grid">${items.join('')}</div>
  </div>`;
}

function renderMetaSection(p) {
  return `<div class="detail-section meta-section">
    <div class="section-title"><span class="section-icon">📋</span>元数据</div>
    <div class="meta-grid">
      <div class="meta-item">
        <span class="meta-label">版本</span>
        <span class="meta-value">v${p.version ?? 3}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">更新次数</span>
        <span class="meta-value">${p.update_count ?? 0}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">最后更新</span>
        <span class="meta-value">${esc(fmtTime(p.last_updated))}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">变更记录</span>
        <span class="meta-value">${p.change_log?.length ?? 0} 条</span>
      </div>
    </div>
  </div>`;
}

// ── 辅助 ──
function el(id) { return document.getElementById(id); }
function val(id) { return (el(id)?.value ?? '').trim(); }

function formatPercent(v) {
  if (v == null) return '-';
  return `${(v * 100).toFixed(0)}%`;
}

function getBarColor(v) {
  if (v >= 0.7) return 'var(--success)';
  if (v >= 0.4) return 'var(--accent)';
  return 'var(--warning)';
}

function getVolatilityColor(v) {
  if (v >= 0.7) return 'var(--danger)';
  if (v >= 0.4) return 'var(--warning)';
  return 'var(--success)';
}

function getEmotionEmoji(emotion) {
  const map = {
    joy: '😊', happy: '😊', excitement: '🎉',
    sadness: '😢', sad: '😢',
    anger: '😠', angry: '😠',
    fear: '😨', anxiety: '😰',
    surprise: '😲',
    disgust: '🤢',
    neutral: '😐', calm: '😌',
    love: '🥰', affection: '💕',
    gratitude: '🙏', thanks: '🙏',
    curiosity: '🤔',
    frustration: '😤',
    hope: '🌟',
  };
  return map[emotion?.toLowerCase()] || '😐';
}

function getTrajectoryIcon(trajectory) {
  const map = {
    improving: '📈', rising: '📈',
    declining: '📉', falling: '📉',
    stable: '➡️', steady: '➡️',
    fluctuating: '〰️', volatile: '〰️',
  };
  return map[trajectory?.toLowerCase()] || '➡️';
}

window.__persona = { showDetail, closeDetail, deletePersona };
