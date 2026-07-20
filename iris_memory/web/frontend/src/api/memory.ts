import { apiGet, apiPost } from './request'

interface ApiBaseResponse {
  success: boolean
  error?: string
  message?: string
}

function checkSuccess(response: ApiBaseResponse, errorMsg: string): void {
  if (!response.success) {
    throw new Error(response.error || errorMsg)
  }
}

export async function getL1Messages(groupId?: string): Promise<any> {
  // 显式传入空字符串（遗留的空键队列）时也要携带参数，
  // 后端以此区分"未选择会话"与"查询空键队列"
  const params = groupId !== undefined ? { group_id: groupId } : {}
  const response = await apiGet<any>('memory/l1/list', params)
  checkSuccess(response, '获取L1缓冲失败')
  return {
    messages: response.messages || [],
    count: response.count || 0
  }
}

export async function getL1Queues(): Promise<any[]> {
  const response = await apiGet<any>('memory/l1/queues')
  checkSuccess(response, '获取L1队列列表失败')
  return response.queues || []
}

export async function searchL2Memory(params: any): Promise<any> {
  const response = await apiPost<any>('memory/l2/search', params)
  checkSuccess(response, '搜索L2记忆失败')
  return { results: response.results || [] }
}

export async function getL2Stats(): Promise<{ total_count: number; group_count: number }> {
  const response = await apiGet<any>('memory/l2/stats')
  checkSuccess(response, '获取L2统计失败')
  return response.stats || { total_count: 0, group_count: 0 }
}

export async function getLatestL2Memories(
  limit: number = 20,
  groupId?: string,
  sortBy: string = 'timestamp',
  sortOrder: string = 'desc',
  offset: number = 0
): Promise<any> {
  const params: Record<string, any> = { limit, sort_by: sortBy, sort_order: sortOrder, offset }
  if (groupId) {
    params.group_id = groupId
  }
  const response = await apiGet<any>('memory/l2/latest', params)
  checkSuccess(response, '获取最新L2记忆失败')
  return {
    results: response.results || [],
    total_count: response.total_count ?? response.results?.length ?? 0,
    limit: response.limit ?? limit,
    offset: response.offset ?? offset
  }
}

export async function deleteL2Entries(ids: string[]): Promise<number> {
  const response = await apiPost<any>('memory/l2/delete', { ids })
  checkSuccess(response, '删除L2记忆失败')
  return response.deleted_count
}

export async function updateL2Entry(id: string, content: string): Promise<void> {
  const response = await apiPost<any>('memory/l2/update', { id, content })
  checkSuccess(response, '更新L2记忆失败')
}

export async function getL3Graph(params?: {
  node_id?: string
  depth?: number
  max_nodes?: number
  max_edges?: number
  group_id?: string
}): Promise<any> {
  const response = await apiGet<any>('memory/l3/graph', params)
  checkSuccess(response, '获取L3图谱失败')
  return response
}

export async function getL3Stats(): Promise<{
  available: boolean
  node_count: number
  edge_count: number
  node_types: Record<string, number>
  relation_types: Record<string, number>
}> {
  const response = await apiGet<any>('memory/l3/stats')
  checkSuccess(response, '获取L3统计失败')
  return (
    response.stats || {
      available: false,
      node_count: 0,
      edge_count: 0,
      node_types: {},
      relation_types: {},
    }
  )
}

export async function searchL3Nodes(keyword: string, limit: number = 20): Promise<any[]> {
  const response = await apiGet<any>('memory/l3/search/nodes', { keyword, limit })
  checkSuccess(response, '搜索节点失败')
  return response.nodes || []
}

export async function searchL3Edges(keyword: string, limit: number = 20): Promise<any[]> {
  const response = await apiGet<any>('memory/l3/search/edges', { keyword, limit })
  checkSuccess(response, '搜索边失败')
  return response.edges || []
}

export async function getL3Nodes(limit: number = 100, keyword?: string, groupId?: string): Promise<any[]> {
  const params: Record<string, any> = { limit }
  if (keyword) {
    params.keyword = keyword
  }
  if (groupId) {
    params.group_id = groupId
  }
  const response = await apiGet<any>('memory/l3/nodes', params)
  checkSuccess(response, '获取L3节点列表失败')
  return response.nodes || []
}

export async function getL3Edges(limit: number = 100, keyword?: string, groupId?: string): Promise<any[]> {
  const params: Record<string, any> = { limit }
  if (keyword) {
    params.keyword = keyword
  }
  if (groupId) {
    params.group_id = groupId
  }
  const response = await apiGet<any>('memory/l3/edges', params)
  checkSuccess(response, '获取L3关系列表失败')
  return response.edges || []
}

export async function deleteL3Nodes(ids: string[]): Promise<number> {
  const response = await apiPost<any>('memory/l3/nodes/delete', { ids })
  checkSuccess(response, '删除L3节点失败')
  return response.deleted_count
}

export async function deleteL3Edge(sourceId: string, targetId: string, relation: string): Promise<void> {
  const response = await apiPost<any>('memory/l3/edges/delete', {
    source_id: sourceId,
    target_id: targetId,
    relation
  })
  checkSuccess(response, '删除L3关系失败')
}
