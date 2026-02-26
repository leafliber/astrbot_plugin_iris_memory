// ========== Shared Utilities ==========

/** HTML-escape a string */
function escHtml(s) {
  const div = document.createElement('div');
  div.textContent = s || '';
  return div.innerHTML;
}

/** Trigger a file download from a Blob */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

// ========== Shared Label Maps ==========
const typeLabels = { fact:'事实', emotion:'情感', relationship:'关系', interaction:'交互', inferred:'推断' };
const layerLabels = { working:'工作', episodic:'情景', semantic:'语义' };
const relationLabels = {
  friend_of:'朋友', colleague_of:'同事', family_of:'家人', boss_of:'上级',
  subordinate_of:'下属', knows:'认识', lives_in:'居住于', works_at:'工作于',
  studies_at:'就读于', belongs_to:'属于', owns:'拥有', likes:'喜欢',
  dislikes:'不喜欢', does:'做', is:'是', has:'有', wants:'想要',
  participated_in:'参与', happened_at:'发生于', caused_by:'引起', related_to:'相关'
};

// KG node colours & labels
const nodeColors = {
  person: '#7c6cf0', location: '#36c78d', organization: '#f5a623',
  object: '#ef4565', event: '#9d6cf0', concept: '#f27649',
  time: '#8b8fa8', unknown: '#5c6082'
};
const nodeTypeLabels = {
  person:'人物', location:'地点', organization:'组织', object:'物品',
  event:'事件', concept:'概念', time:'时间', unknown:'未知'
};
