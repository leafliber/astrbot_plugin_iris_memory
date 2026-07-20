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

export async function getGroupProfile(groupId: string): Promise<any> {
  const response = await apiGet<any>('profile/group', { group_id: groupId })
  checkSuccess(response, '获取群聊画像失败')
  return response.profile || {}
}

export async function updateGroupProfile(groupId: string, data: any): Promise<void> {
  const response = await apiPost<any>('profile/group/update', { ...data, group_id: groupId })
  checkSuccess(response, '更新群聊画像失败')
}

export async function getUserProfile(userId: string, groupId?: string): Promise<any> {
  const params: Record<string, any> = { user_id: userId }
  if (groupId) {
    params.group_id = groupId
  }
  const response = await apiGet<any>('profile/user', params)
  checkSuccess(response, '获取用户画像失败')
  return response.profile || {}
}

export async function updateUserProfile(userId: string, data: any, groupId?: string): Promise<void> {
  const params = groupId ? { group_id: groupId } : {}
  const response = await apiPost<any>('profile/user/update', { ...data, user_id: userId, ...params })
  checkSuccess(response, '更新用户画像失败')
}

export async function getGroupList(): Promise<any[]> {
  const response = await apiGet<any>('profile/groups')
  checkSuccess(response, '获取群聊列表失败')
  return response.groups || []
}

export async function deleteGroupProfile(groupId: string): Promise<void> {
  const response = await apiPost<any>('profile/group/delete', { group_id: groupId })
  checkSuccess(response, '删除群聊画像失败')
}

export async function deleteUserProfile(userId: string, groupId?: string): Promise<void> {
  const params: Record<string, any> = { user_id: userId }
  if (groupId) {
    params.group_id = groupId
  }
  const response = await apiPost<any>('profile/user/delete', params)
  checkSuccess(response, '删除用户画像失败')
}

export async function getUserList(groupId?: string): Promise<any[]> {
  const params = groupId ? { group_id: groupId } : {}
  const response = await apiGet<any>('profile/users', params)
  checkSuccess(response, '获取用户列表失败')
  return response.users || []
}
