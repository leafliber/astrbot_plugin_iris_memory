import { apiGet, apiPost } from './request'

export interface UIPreferences {
  dark_mode: boolean
}

interface ApiBaseResponse {
  success: boolean
  error?: string
  preferences?: UIPreferences
}

function checkSuccess(response: ApiBaseResponse, errorMsg: string): void {
  if (!response.success) {
    throw new Error(response.error || errorMsg)
  }
}

export async function getUIPreferences(): Promise<UIPreferences> {
  const response = await apiGet<ApiBaseResponse>('ui-preferences')
  checkSuccess(response, '获取 UI 偏好失败')
  return response.preferences!
}

export async function updateUIPreferences(updates: Partial<UIPreferences>): Promise<UIPreferences> {
  const response = await apiPost<ApiBaseResponse>('ui-preferences/update', { updates })
  checkSuccess(response, '更新 UI 偏好失败')
  return response.preferences!
}
