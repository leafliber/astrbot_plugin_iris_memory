import { apiGet, apiPost } from './request'

interface ApiBaseResponse {
  success: boolean
  error?: string
}

function checkSuccess(response: ApiBaseResponse, errorMsg: string): void {
  if (!response.success) {
    throw new Error(response.error || errorMsg)
  }
}

export async function clearL1Buffer(groupId?: string): Promise<{ cleared_count: number }> {
  // groupId 为 undefined 时清空全部；为字符串（含空字符串——遗留的空键队列）
  // 时清空对应队列，后端按 group_id 是否为 null 区分
  const response = await apiPost<any>('manage/l1/clear', {
    group_id: groupId ?? null
  })
  checkSuccess(response, '清空 L1 缓冲失败')
  return { cleared_count: response.cleared_count }
}

export async function deleteL2Memory(scope: 'all' | 'group', groupId?: string): Promise<{ deleted_count: number }> {
  const response = await apiPost<any>('manage/l2/delete', {
    scope,
    group_id: groupId
  })
  checkSuccess(response, '删除 L2 记忆失败')
  return { deleted_count: response.deleted_count }
}

export async function deleteL3KG(scope: 'all' | 'group', groupId?: string): Promise<{ deleted_count: number }> {
  const response = await apiPost<any>('manage/l3/delete', {
    scope,
    group_id: groupId
  })
  checkSuccess(response, '删除 L3 图谱失败')
  return { deleted_count: response.deleted_count }
}

export async function mergeL3Duplicates(): Promise<{ merged_count: number; deleted_count: number }> {
  const response = await apiPost<any>('manage/l3/merge-duplicates')
  checkSuccess(response, '合并重复节点失败')
  return { merged_count: response.merged_count, deleted_count: response.deleted_count }
}

export async function deleteProfile(
  scope: 'group' | 'user' | 'all',
  groupId?: string,
  userId?: string
): Promise<Record<string, unknown>> {
  const response = await apiPost<any>('manage/profile/delete', {
    scope,
    group_id: groupId,
    user_id: userId
  })
  checkSuccess(response, '删除画像失败')
  return response
}

export type TaskName = 'dream' | 'cache_cleanup'

export async function triggerTask(task: TaskName): Promise<{ message: string }> {
  const response = await apiPost<any>('manage/tasks/trigger', {
    task
  })
  checkSuccess(response, '触发任务失败')
  return { message: response.message }
}

export interface TaskStatus {
  running: boolean
}

export async function getTasksStatus(): Promise<Record<string, TaskStatus>> {
  const response = await apiGet<any>('manage/tasks/status')
  checkSuccess(response, '获取任务状态失败')
  return response.tasks
}
