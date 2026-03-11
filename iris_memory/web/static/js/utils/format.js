/**
 * 格式化与标签工具
 */

export const typeLabels = {
  fact: '事实', emotion: '情感', relationship: '关系',
  interaction: '交互', inferred: '推断',
};

export const layerLabels = {
  working: '工作', episodic: '情景', semantic: '语义',
};

export const relationLabels = {
  friend_of: '朋友', colleague_of: '同事', family_of: '家人', boss_of: '上级',
  subordinate_of: '下属', knows: '认识', lives_in: '居住于', works_at: '工作于',
  studies_at: '就读于', belongs_to: '属于', owns: '拥有', likes: '喜欢',
  dislikes: '不喜欢', does: '做', is: '是', has: '有', wants: '想要',
  participated_in: '参与', happened_at: '发生于', caused_by: '引起', related_to: '相关',
};

export const nodeColors = {
  person: '#7c6cf0', location: '#36c78d', organization: '#f5a623',
  object: '#ef4565', event: '#9d6cf0', concept: '#f27649',
  time: '#8b8fa8', unknown: '#5c6082',
};

export const nodeTypeLabels = {
  person: '人物', location: '地点', organization: '组织', object: '物品',
  event: '事件', concept: '概念', time: '时间', unknown: '未知',
};

/**
 * 高亮搜索词
 * @param {string} text
 * @param {string} keyword
 * @returns {string}
 */
export function highlightText(text, keyword) {
  if (!keyword || !text) return text;
  const safe = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return text.replace(new RegExp(`(${safe})`, 'gi'), '<span class="highlight">$1</span>');
}

/**
 * 触发文件下载
 * @param {Blob} blob
 * @param {string} filename
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

/**
 * 格式化日期时间
 * @param {string|number|null} input - ISO字符串或Unix时间戳(秒)
 * @returns {string}
 */
export function fmtTime(input) {
  if (!input) return '-';
  try {
    let d;
    if (typeof input === 'number') {
      d = new Date(input * 1000);
    } else if (typeof input === 'string') {
      if (/^\d+(\.\d+)?$/.test(input)) {
        d = new Date(parseFloat(input) * 1000);
      } else {
        d = new Date(input);
      }
    } else {
      return '-';
    }
    if (isNaN(d.getTime())) return '-';
    const pad = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return String(input);
  }
}

/**
 * 格式化秒数为人类可读
 * @param {number} seconds
 * @returns {string}
 */
export function fmtDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}分${Math.round(seconds % 60)}秒`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}时${m}分`;
}
