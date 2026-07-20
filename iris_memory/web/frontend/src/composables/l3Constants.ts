/**
 * L3 知识图谱共享常量与工具函数
 *
 * 节点类型 → 图标 / 主题色名 / 中文标签
 * 关系类型 → 中文标签
 * 置信度 → 颜色等级
 */

// 节点类型 → MDI 图标名
export const NODE_TYPE_ICONS: Record<string, string> = {
  Person: 'mdi-account',
  Preference: 'mdi-heart',
  Skill: 'mdi-tools',
  Trait: 'mdi-emoticon',
  Goal: 'mdi-flag',
  Belief: 'mdi-lightbulb-on',
  Event: 'mdi-calendar',
  Concept: 'mdi-lightbulb',
  Location: 'mdi-map-marker',
  Item: 'mdi-package-variant',
  Topic: 'mdi-tag',
  Group: 'mdi-account-group',
  Entity: 'mdi-circle',
}

// 节点类型 → Vuetify 主题色名（用于列表 chip 与图谱节点填充色解析）
export const NODE_TYPE_COLORS: Record<string, string> = {
  Person: 'primary',
  Preference: 'pink',
  Skill: 'teal',
  Trait: 'purple',
  Goal: 'orange',
  Belief: 'indigo',
  Event: 'secondary',
  Concept: 'info',
  Location: 'success',
  Item: 'warning',
  Topic: 'accent',
  Group: 'cyan',
  Entity: 'blue-grey',
}

// 节点类型 → 中文标签
export const NODE_TYPE_LABELS: Record<string, string> = {
  Person: '人物',
  Preference: '偏好',
  Skill: '技能',
  Trait: '性格特征',
  Goal: '目标',
  Belief: '信念',
  Event: '事件',
  Concept: '概念',
  Location: '地点',
  Item: '物品',
  Topic: '话题',
  Group: '群体',
}

// 关系类型 → 中文标签
export const RELATION_TYPE_LABELS: Record<string, string> = {
  KNOWS: '认识',
  HAS_PREFERENCE: '偏好',
  HAS_SKILL: '掌握',
  HAS_TRAIT: '具有',
  HAS_GOAL: '追求',
  HAS_BELIEF: '相信',
  PARTICIPATED_IN: '参与',
  LOCATED_AT: '位于',
  HAPPENED_AT: '发生在',
  PART_OF: '属于',
  LEADS_TO: '导致',
  CONTRADICTS: '矛盾',
  SUPPORTS: '支持',
  RELATED_TO: '相关',
}

export const getNodeIcon = (label: string): string => NODE_TYPE_ICONS[label] || 'mdi-tag'

export const getTypeColor = (label: string): string => NODE_TYPE_COLORS[label] || 'blue-grey'

export const getNodeLabel = (label: string): string => NODE_TYPE_LABELS[label] || label

export const getRelationLabel = (relation: string): string =>
  RELATION_TYPE_LABELS[relation] || relation

// 置信度 → Vuetify 色名（用于 chip 展示）
export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.9) return 'success'
  if (confidence >= 0.7) return 'info'
  if (confidence >= 0.5) return 'warning'
  return 'error'
}

/**
 * 从 document 读取 Vuetify 主题色对应的 rgb 字符串。
 * G6 在 canvas 中渲染，无法直接消费 CSS 变量，需在初始化时解析为具体颜色值。
 * 解析失败时回退到 fallback。
 */
export const resolveThemeColor = (colorName: string, fallback = '#888888'): string => {
  if (typeof document === 'undefined') return fallback
  try {
    const style = getComputedStyle(document.documentElement)
    const raw = style.getPropertyValue(`--v-theme-${colorName}`).trim()
    if (raw) return `rgb(${raw})`
  } catch {
    // ignore
  }
  return fallback
}

export const formatTime = (timestamp?: string): string => {
  if (!timestamp) return '-'
  try {
    const date = new Date(timestamp)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return timestamp
  }
}
