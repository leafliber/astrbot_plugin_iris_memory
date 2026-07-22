import { apiGet, apiPost } from './request'

export interface RunLogEntry {
  id: number
  timestamp: string
  type: 'llm_call' | 'injection' | 'proactive'
  type_label: string
  title: string
  success: boolean
  detail: Record<string, any>
}

export interface RunLogType {
  key: string
  label: string
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

export async function getRunLogs(
  type?: string,
  limit?: number
): Promise<{ entries: RunLogEntry[]; counts: Record<string, number>; types: RunLogType[] }> {
  const params: Record<string, any> = {}
  if (type) params.type = type
  if (limit) params.limit = limit
  const response = await apiGet<any>('run-log', params)
  checkSuccess(response, '获取运行日志失败')
  return {
    entries: response.entries || [],
    counts: response.counts || {},
    types: response.types || []
  }
}

export async function clearRunLogs(type?: string): Promise<number> {
  const response = await apiPost<any>('run-log/clear', { type: type || null })
  checkSuccess(response, '清空运行日志失败')
  return response.cleared || 0
}
