import { apiGet, apiPost } from './request'

export interface ReplyWhitelistGroup {
  group_id: string
  state: string
  willingness: string
  msg_count: number
  effective_n: number
  effective_t: number
  backoff_level: number
  consecutive_replies: number
  initiate_daily_count: number
  anchor_kind?: string
  anchor_users?: string[]
  anchor_keywords?: string[]
  anchor_reason?: string
  anchor_bot_message?: string
  initiate_pending?: boolean
  initiate_no_reply_streak?: number
}

export interface ReplyStatsGroup {
  group_id: string
  total_decisions: number
  total_replies: number
  total_skips: number
  total_errors: number
  total_drifts: number
  total_initiates: number
  total_passive_replies: number
  last_decision_time: number
  last_motive: string
  last_reply_time: number
  current_state: string
  willingness: string
  msg_count: number
  effective_n: number
  effective_t: number
  backoff_level: number
  consecutive_replies: number
  initiate_daily_count: number
}

export interface ReplyLlmLog {
  group_id: string
  motive: string
  system_prompt: string
  user_prompt: string
  response_text: string
  action: string
  message: string
  observation: string
  watch_users: string[]
  watch_keywords: string[]
  watch_reason: string
  drifted: boolean
  timestamp: number
  duration_ms: number
}

export interface ReplyConfigSubMeta {
  label: string
  type: string
  min: number
  max: number
}

export interface ReplyConfigMeta {
  label: string
  type: 'int' | 'float' | 'bool' | 'str' | 'object'
  min?: number
  max?: number
  step?: number
  hint?: string
  items?: Record<string, ReplyConfigSubMeta>
}

export async function getReplyWhitelist(): Promise<ReplyWhitelistGroup[]> {
  return apiGet<ReplyWhitelistGroup[]>('reply/whitelist/list')
}

export async function enableReplyGroup(groupId: string): Promise<void> {
  await apiPost('reply/whitelist/enable', { group_id: groupId })
}

export async function disableReplyGroup(groupId: string): Promise<void> {
  await apiPost('reply/whitelist/disable', { group_id: groupId })
}

export async function setReplyWillingness(groupId: string, willingness: string): Promise<void> {
  await apiPost('reply/group/set_willingness', { group_id: groupId, willingness })
}

export async function resetReplyGroup(groupId: string): Promise<void> {
  await apiPost('reply/group/reset', { group_id: groupId })
}

export async function getReplyStatsStatus(): Promise<{ enabled: boolean }> {
  return apiGet<{ enabled: boolean }>('reply/stats/status')
}

export async function getReplyStatsGroups(): Promise<ReplyStatsGroup[]> {
  return apiGet<ReplyStatsGroup[]>('reply/stats/groups')
}

export async function getReplyStatsLogs(params: {
  group_id?: string
  limit: number
  offset: number
}): Promise<ReplyLlmLog[]> {
  return apiGet<ReplyLlmLog[]>('reply/stats/logs', params)
}

export async function clearReplyStats(): Promise<void> {
  await apiPost('reply/stats/clear')
}

export async function getReplyConfig(): Promise<{
  values: Record<string, unknown>
  meta: Record<string, ReplyConfigMeta>
}> {
  return apiGet('reply/config/get')
}

export async function setReplyConfig(payload: Record<string, unknown>): Promise<void> {
  await apiPost('reply/config/set', payload)
}
