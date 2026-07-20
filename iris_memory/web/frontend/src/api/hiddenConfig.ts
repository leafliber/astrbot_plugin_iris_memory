import { apiGet, apiPost } from './request'

export interface HiddenConfigItem {
  key: string
  value: unknown
  default: unknown
  type: 'int' | 'float' | 'bool' | 'string' | 'literal'
  description: string
  group: string
  options: string[]
}

export interface HiddenConfigGroup {
  name: string
  keys: string[]
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

export async function getHiddenConfig(): Promise<{ items: HiddenConfigItem[]; groups: HiddenConfigGroup[] }> {
  const response = await apiGet<any>('hidden-config')
  checkSuccess(response, '获取隐藏配置失败')
  return { items: response.items, groups: response.groups }
}

export async function updateHiddenConfig(updates: Record<string, unknown>): Promise<string[]> {
  const response = await apiPost<any>('hidden-config/update', { updates })
  checkSuccess(response, '更新隐藏配置失败')
  return response.updated_keys || []
}

export async function deleteHiddenConfig(key: string): Promise<string> {
  const response = await apiPost<any>('hidden-config/delete', { key })
  checkSuccess(response, '删除配置项失败')
  return response.message || ''
}

export async function resetHiddenConfig(): Promise<string> {
  const response = await apiPost<any>('hidden-config/reset')
  checkSuccess(response, '重置隐藏配置失败')
  return response.message || ''
}
