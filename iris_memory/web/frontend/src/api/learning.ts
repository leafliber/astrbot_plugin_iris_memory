import { apiGet, apiPost } from './request'

export type LearningTable = 'jargon' | 'expression_pattern' | 'few_shot'

export type LearningStatus = 'pending_review' | 'approved' | 'disabled' | 'active'

// 圈内暗语
export interface JargonItem {
  id: number
  group_id: string
  term: string
  count: number
  meaning: string
  confidence: number
  status: string
  last_inferred_at: number
  created_at: number
}

// 表达模式
export interface ExpressionPatternItem {
  id: number
  group_id: string
  scene: string
  expression: string
  source_pair_id: number | null
  hit_count: number
  status: string
  created_at: number
  last_hit_at: number | null
}

// 对话样例
export interface FewShotItem {
  id: number
  group_id: string
  user_id: string
  user_text: string
  bot_text: string
  message_id: string
  status: string
  created_at: number
}

export type LearningItem = JargonItem | ExpressionPatternItem | FewShotItem

export interface LearningTableStats {
  total: number
  by_status: Record<string, number>
}

export interface LearningStats {
  jargon: LearningTableStats
  expression_pattern: LearningTableStats
  few_shot: LearningTableStats
}

export interface JargonTopItem {
  term: string
  group_id: string
  count: number
  meaning: string
}

export interface LearningListParams {
  table: LearningTable
  group_id?: string
  status?: string
  page?: number
  page_size?: number
}

interface ApiBaseResponse {
  success: boolean
  error?: string
}

function checkSuccess(response: ApiBaseResponse, errorMsg: string): void {
  if (!response.success) {
    throw new Error(response.error || errorMsg)
  }
}

export async function getLearningList(params: LearningListParams): Promise<{ items: LearningItem[]; total: number }> {
  const response = await apiGet<any>('learning/list', params as Record<string, any>)
  checkSuccess(response, '获取学习数据失败')
  return { items: response.items || [], total: response.total || 0 }
}

export async function getLearningGroups(): Promise<string[]> {
  const response = await apiGet<any>('learning/groups')
  checkSuccess(response, '获取群组列表失败')
  return response.groups || []
}

export async function getLearningStats(): Promise<{ stats: LearningStats; jargon_top: JargonTopItem[] }> {
  const response = await apiGet<any>('learning/stats')
  checkSuccess(response, '获取学习统计失败')
  return { stats: response.stats, jargon_top: response.jargon_top || [] }
}

export async function addLearningItem(table: LearningTable, fields: Record<string, unknown>): Promise<number> {
  const response = await apiPost<any>('learning/add', { table, ...fields })
  checkSuccess(response, '新增失败')
  return response.id
}

export async function updateLearningItem(
  table: LearningTable,
  id: number,
  fields: Record<string, unknown>
): Promise<void> {
  const response = await apiPost<any>('learning/update', { table, id, fields })
  checkSuccess(response, '更新失败')
}

export async function deleteLearningItems(table: LearningTable, ids: number[]): Promise<number> {
  const response = await apiPost<any>('learning/delete', { table, ids })
  checkSuccess(response, '删除失败')
  return response.deleted ?? ids.length
}

export async function setLearningStatus(table: LearningTable, ids: number[], status: string): Promise<void> {
  const response = await apiPost<any>('learning/status', { table, ids, status })
  checkSuccess(response, '更新状态失败')
}
