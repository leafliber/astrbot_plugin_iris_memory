import { apiGet, apiPost, apiDownload } from './request'

interface ApiBaseResponse {
  success: boolean
  error?: string
}

function checkSuccess(response: ApiBaseResponse, errorMsg: string): void {
  if (!response.success) {
    throw new Error(response.error || errorMsg)
  }
}

export async function exportL2Memory(groupId?: string): Promise<void> {
  const params: Record<string, string> = groupId ? { group_id: groupId } : {}
  const timestamp = new Date().toISOString().slice(0, 10)
  await apiDownload('data/l2/export', params, `iris_l2_memory_${timestamp}.json`)
}

export async function importL2Memory(data: unknown, skipDuplicates: boolean = true): Promise<any> {
  const response = await apiPost<any>('data/l2/import', {
    data,
    skip_duplicates: skipDuplicates
  })
  checkSuccess(response, '导入 L2 记忆失败')
  return response.stats
}

export async function exportL3KG(): Promise<void> {
  const timestamp = new Date().toISOString().slice(0, 10)
  await apiDownload('data/l3/export', {}, `iris_l3_kg_${timestamp}.json`)
}

export async function importL3KG(data: unknown, skipDuplicates: boolean = true): Promise<any> {
  const response = await apiPost<any>('data/l3/import', {
    data,
    skip_duplicates: skipDuplicates
  })
  checkSuccess(response, '导入 L3 知识图谱失败')
  return response.stats
}

export async function exportProfiles(): Promise<void> {
  const timestamp = new Date().toISOString().slice(0, 10)
  await apiDownload('data/profile/export', {}, `iris_profiles_${timestamp}.json`)
}

export async function importProfiles(data: unknown, skipDuplicates: boolean = true): Promise<any> {
  const response = await apiPost<any>('data/profile/import', {
    data,
    skip_duplicates: skipDuplicates
  })
  checkSuccess(response, '导入画像失败')
  return response.stats
}

export async function exportAll(): Promise<void> {
  const timestamp = new Date().toISOString().slice(0, 10)
  await apiDownload('data/all/export', {}, `iris_full_backup_${timestamp}.json`)
}

export async function importAll(data: unknown, skipDuplicates: boolean = true): Promise<Record<string, unknown>> {
  const response = await apiPost<any>('data/all/import', {
    data,
    skip_duplicates: skipDuplicates
  })
  checkSuccess(response, '全量导入失败')
  return response.result
}
