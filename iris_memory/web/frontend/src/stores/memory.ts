import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { L1Message, L2Memory, KGGraph, KGNode, KGEdge, L1QueueItem, L3NodeDetail, L3EdgeDetail, L2SortField, L2SortOrder, L3SearchNodeResult, L3SearchEdgeResult, L3Stats, L3LayoutType, L3Filters } from '@/types'
import * as memoryApi from '@/api/memory'

export const useMemoryStore = defineStore('memory', () => {
  const l1Messages = ref<L1Message[]>([])
  const l1Loading = ref(false)
  const l1Queues = ref<L1QueueItem[]>([])
  const l1QueuesLoading = ref(false)

  const l2Results = ref<L2Memory[]>([])
  const l2Loading = ref(false)
  const l2Query = ref('')
  const l2TotalCount = ref(0)
  const l2GroupCount = ref(0)

  const l2LatestResults = ref<L2Memory[]>([])
  const l2LatestLoading = ref(false)
  const l2LatestLimit = ref(20)
  const l2LatestSortBy = ref<L2SortField>('timestamp')
  const l2LatestSortOrder = ref<L2SortOrder>('desc')
  const l2LatestOffset = ref(0)
  const l2LatestTotalCount = ref(0)

  const l3Graph = ref<KGGraph>({ nodes: [], edges: [], start_node: null })
  const l3StartNode = ref<KGNode | null>(null)
  const l3Loading = ref(false)
  const l3Depth = ref(2)
  const l3MaxNodes = ref(30)
  const l3Layout = ref<L3LayoutType>('force')
  const l3Stats = ref<L3Stats>({
    available: false,
    node_count: 0,
    edge_count: 0,
    node_types: {},
    relation_types: {},
  })
  const l3StatsLoading = ref(false)
  const l3Filters = ref<L3Filters>({
    nodeTypes: [],
    relationTypes: [],
    groupId: null,
    minConfidence: 0,
  })
  // 导航历史：已展开的节点 ID 栈
  const l3NavHistory = ref<string[]>([])
  const l3NavIndex = ref(-1)

  const l3SearchResults = ref<{ nodes: L3SearchNodeResult[], edges: L3SearchEdgeResult[] }>({ nodes: [], edges: [] })
  const l3SearchLoading = ref(false)
  const l3SearchKeyword = ref('')

  const l3Nodes = ref<L3NodeDetail[]>([])
  const l3NodesLoading = ref(false)
  const l3NodesKeyword = ref('')
  const l3Edges = ref<L3EdgeDetail[]>([])
  const l3EdgesLoading = ref(false)
  const l3EdgesKeyword = ref('')

  // 过滤后的图谱数据（按类型/关系/置信度）
  const l3FilteredGraph = computed<KGGraph>(() => {
    const f = l3Filters.value
    const allowedNodeTypes = new Set(f.nodeTypes)
    const allowedRelTypes = new Set(f.relationTypes)
    const typeFilterActive = allowedNodeTypes.size > 0
    const relFilterActive = allowedRelTypes.size > 0

    const nodes = l3Graph.value.nodes.filter((n) => {
      if (typeFilterActive && !allowedNodeTypes.has(n.label)) return false
      if (n.confidence < f.minConfidence) return false
      if (f.groupId && n.group_id !== f.groupId) return false
      return true
    })
    const nodeIds = new Set(nodes.map((n) => n.id))
    const edges = l3Graph.value.edges.filter((e) => {
      if (relFilterActive && !allowedRelTypes.has(e.relation)) return false
      if ((e.confidence ?? 1) < f.minConfidence) return false
      if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) return false
      return true
    })
    return { nodes, edges, start_node: l3Graph.value.start_node }
  })

  // 当前图谱中出现的节点/关系类型（供过滤器选项）
  const l3AvailableNodeTypes = computed(() => {
    const m = new Map<string, number>()
    l3Graph.value.nodes.forEach((n) => m.set(n.label, (m.get(n.label) || 0) + 1))
    return Array.from(m.entries()).map(([type, count]) => ({ type, count }))
  })
  const l3AvailableRelationTypes = computed(() => {
    const m = new Map<string, number>()
    l3Graph.value.edges.forEach((e) => m.set(e.relation, (m.get(e.relation) || 0) + 1))
    return Array.from(m.entries()).map(([type, count]) => ({ type, count }))
  })

  const canGoBack = computed(() => l3NavIndex.value > 0)
  const canGoForward = computed(() => l3NavIndex.value < l3NavHistory.value.length - 1)

  const fetchL1Messages = async (groupId?: string) => {
    l1Loading.value = true
    try {
      const response = await memoryApi.getL1Messages(groupId)
      l1Messages.value = response.messages
    } catch (error) {
      console.error('获取L1缓冲失败:', error)
      l1Messages.value = []
    } finally {
      l1Loading.value = false
    }
  }

  const fetchL1Queues = async () => {
    l1QueuesLoading.value = true
    try {
      l1Queues.value = await memoryApi.getL1Queues()
    } catch (error) {
      console.error('获取L1队列列表失败:', error)
      l1Queues.value = []
    } finally {
      l1QueuesLoading.value = false
    }
  }

  const searchL2Memory = async (query: string, groupId?: string, topK = 10) => {
    l2Loading.value = true
    l2Query.value = query
    try {
      const response = await memoryApi.searchL2Memory({ query, group_id: groupId, top_k: topK })
      l2Results.value = response.results
    } catch (error) {
      console.error('搜索L2记忆失败:', error)
      l2Results.value = []
    } finally {
      l2Loading.value = false
    }
  }

  const fetchL2Stats = async () => {
    try {
      const stats = await memoryApi.getL2Stats()
      l2TotalCount.value = stats.total_count || 0
      l2GroupCount.value = stats.group_count || 0
    } catch (error) {
      console.error('获取L2统计失败:', error)
    }
  }

  const _loadGraph = async (nodeId?: string, recordNav = true) => {
    l3Loading.value = true
    try {
      const response = await memoryApi.getL3Graph({
        node_id: nodeId,
        depth: l3Depth.value,
        max_nodes: l3MaxNodes.value,
        group_id: l3Filters.value.groupId || undefined,
      })
      l3Graph.value = {
        nodes: response.nodes || [],
        edges: response.edges || [],
        start_node: response.start_node || null,
      }
      l3StartNode.value = response.start_node || null
      if (recordNav && response.start_node) {
        const id = response.start_node.id
        // 截断前进历史
        l3NavHistory.value = [...l3NavHistory.value.slice(0, l3NavIndex.value + 1), id]
        l3NavIndex.value = l3NavHistory.value.length - 1
      }
    } catch (error) {
      console.error('获取L3图谱失败:', error)
      l3Graph.value = { nodes: [], edges: [], start_node: null }
      l3StartNode.value = null
    } finally {
      l3Loading.value = false
    }
  }

  const fetchL3Graph = async (nodeId?: string) => _loadGraph(nodeId, true)

  /**
   * 刷新当前图谱：保留当前主节点重新加载（若无可用的主节点则后端随机选）。
   * 用于深度/最大节点数等控制参数变更后重新拉取，避免主节点被随机重置。
   * 不写入导航历史，因为这只是参数刷新而非节点跳转。
   */
  const refreshL3Graph = async () => {
    const currentNodeId = l3StartNode.value?.id
    await _loadGraph(currentNodeId, false)
  }

  const expandFromNode = async (nodeId: string) => _loadGraph(nodeId, true)

  const navBack = async () => {
    if (!canGoBack.value) return
    l3NavIndex.value -= 1
    await _loadGraph(l3NavHistory.value[l3NavIndex.value], false)
  }

  const navForward = async () => {
    if (!canGoForward.value) return
    l3NavIndex.value += 1
    await _loadGraph(l3NavHistory.value[l3NavIndex.value], false)
  }

  const fetchL3Stats = async () => {
    l3StatsLoading.value = true
    try {
      l3Stats.value = await memoryApi.getL3Stats()
    } catch (error) {
      console.error('获取L3统计失败:', error)
    } finally {
      l3StatsLoading.value = false
    }
  }

  const setDepth = (depth: number) => {
    l3Depth.value = depth
  }

  const setMaxNodes = (maxNodes: number) => {
    l3MaxNodes.value = maxNodes
  }

  const setLayout = (layout: L3LayoutType) => {
    l3Layout.value = layout
  }

  const toggleNodeTypeFilter = (type: string) => {
    const arr = l3Filters.value.nodeTypes
    const idx = arr.indexOf(type)
    if (idx >= 0) arr.splice(idx, 1)
    else arr.push(type)
  }

  const toggleRelationTypeFilter = (type: string) => {
    const arr = l3Filters.value.relationTypes
    const idx = arr.indexOf(type)
    if (idx >= 0) arr.splice(idx, 1)
    else arr.push(type)
  }

  const setMinConfidence = (v: number) => {
    l3Filters.value.minConfidence = v
  }

  const setGroupFilter = (groupId: string | null) => {
    l3Filters.value.groupId = groupId
    // 清空已加载的节点/边列表，下次切换到对应 tab 时会带新 group_id 重新拉取
    l3Nodes.value = []
    l3Edges.value = []
  }

  const resetFilters = () => {
    l3Filters.value = {
      nodeTypes: [],
      relationTypes: [],
      groupId: null,
      minConfidence: 0,
    }
  }

  const clearL2Results = () => {
    l2Results.value = []
    l2Query.value = ''
  }

  const fetchLatestL2Memories = async (limit?: number, groupId?: string, offset?: number) => {
    l2LatestLoading.value = true
    if (limit !== undefined) {
      l2LatestLimit.value = limit
    }
    if (offset !== undefined) {
      l2LatestOffset.value = offset
    }
    try {
      const response = await memoryApi.getLatestL2Memories(
        l2LatestLimit.value,
        groupId,
        l2LatestSortBy.value,
        l2LatestSortOrder.value,
        l2LatestOffset.value
      )
      l2LatestResults.value = response.results
      l2LatestTotalCount.value = response.total_count ?? 0
    } catch (error) {
      console.error('获取最新L2记忆失败:', error)
      l2LatestResults.value = []
      l2LatestTotalCount.value = 0
    } finally {
      l2LatestLoading.value = false
    }
  }

  const setL2LatestLimit = (limit: number) => {
    l2LatestLimit.value = limit
  }

  const setL2LatestSort = (sortBy: L2SortField, sortOrder: L2SortOrder) => {
    l2LatestSortBy.value = sortBy
    l2LatestSortOrder.value = sortOrder
  }

  const setL2LatestOffset = (offset: number) => {
    l2LatestOffset.value = offset
  }

  const setL2LatestPage = (page: number) => {
    l2LatestOffset.value = (page - 1) * l2LatestLimit.value
  }

  const getL2LatestCurrentPage = (): number => {
    if (l2LatestLimit.value <= 0) return 1
    return Math.floor(l2LatestOffset.value / l2LatestLimit.value) + 1
  }

  const getL2LatestTotalPages = (): number => {
    if (l2LatestLimit.value <= 0) return 1
    return Math.max(1, Math.ceil(l2LatestTotalCount.value / l2LatestLimit.value))
  }

  const deleteL2Entries = async (ids: string[]) => {
    const count = await memoryApi.deleteL2Entries(ids)
    l2LatestResults.value = l2LatestResults.value.filter(r => !ids.includes(r.id))
    l2Results.value = l2Results.value.filter(r => !ids.includes(r.id))
    return count
  }

  const updateL2Entry = async (id: string, content: string) => {
    await memoryApi.updateL2Entry(id, content)
    const updateInList = (list: L2Memory[]) => {
      const item = list.find(r => r.id === id)
      if (item) {
        item.content = content
      }
    }
    updateInList(l2LatestResults.value)
    updateInList(l2Results.value)
  }

  const searchL3 = async (keyword: string) => {
    if (!keyword.trim()) {
      l3SearchResults.value = { nodes: [], edges: [] }
      return
    }
    
    l3SearchLoading.value = true
    l3SearchKeyword.value = keyword
    
    try {
      const [nodes, edges] = await Promise.all([
        memoryApi.searchL3Nodes(keyword),
        memoryApi.searchL3Edges(keyword)
      ])
      l3SearchResults.value = { nodes, edges }
    } catch (error) {
      console.error('搜索L3图谱失败:', error)
      l3SearchResults.value = { nodes: [], edges: [] }
    } finally {
      l3SearchLoading.value = false
    }
  }

  const clearL3Search = () => {
    l3SearchResults.value = { nodes: [], edges: [] }
    l3SearchKeyword.value = ''
  }

  const fetchL3Nodes = async (keyword?: string) => {
    l3NodesLoading.value = true
    l3NodesKeyword.value = keyword || ''
    try {
      l3Nodes.value = await memoryApi.getL3Nodes(100, keyword, l3Filters.value.groupId || undefined)
    } catch (error) {
      console.error('获取L3节点列表失败:', error)
      l3Nodes.value = []
    } finally {
      l3NodesLoading.value = false
    }
  }

  const fetchL3Edges = async (keyword?: string) => {
    l3EdgesLoading.value = true
    l3EdgesKeyword.value = keyword || ''
    try {
      l3Edges.value = await memoryApi.getL3Edges(100, keyword, l3Filters.value.groupId || undefined)
    } catch (error) {
      console.error('获取L3关系列表失败:', error)
      l3Edges.value = []
    } finally {
      l3EdgesLoading.value = false
    }
  }

  const deleteL3Nodes = async (ids: string[]) => {
    const count = await memoryApi.deleteL3Nodes(ids)
    l3Nodes.value = l3Nodes.value.filter(n => !ids.includes(n.id))
    return count
  }

  const deleteL3Edge = async (sourceId: string, targetId: string, relation: string) => {
    await memoryApi.deleteL3Edge(sourceId, targetId, relation)
    l3Edges.value = l3Edges.value.filter(
      e => !(e.source.id === sourceId && e.target.id === targetId && e.relation === relation)
    )
  }

  return {
    l1Messages,
    l1Loading,
    l1Queues,
    l1QueuesLoading,
    l2Results,
    l2Loading,
    l2Query,
    l2TotalCount,
    l2GroupCount,
    l2LatestResults,
    l2LatestLoading,
    l2LatestLimit,
    l2LatestSortBy,
    l2LatestSortOrder,
    l2LatestOffset,
    l2LatestTotalCount,
    l3Graph,
    l3StartNode,
    l3Loading,
    l3Depth,
    l3MaxNodes,
    l3Layout,
    l3Stats,
    l3StatsLoading,
    l3Filters,
    l3FilteredGraph,
    l3AvailableNodeTypes,
    l3AvailableRelationTypes,
    l3NavHistory,
    l3NavIndex,
    canGoBack,
    canGoForward,
    l3SearchResults,
    l3SearchLoading,
    l3SearchKeyword,
    l3Nodes,
    l3NodesLoading,
    l3NodesKeyword,
    l3Edges,
    l3EdgesLoading,
    l3EdgesKeyword,
    fetchL1Messages,
    fetchL1Queues,
    searchL2Memory,
    fetchL2Stats,
    fetchL3Graph,
    refreshL3Graph,
    expandFromNode,
    navBack,
    navForward,
    fetchL3Stats,
    setDepth,
    setMaxNodes,
    setLayout,
    toggleNodeTypeFilter,
    toggleRelationTypeFilter,
    setMinConfidence,
    setGroupFilter,
    resetFilters,
    clearL2Results,
    fetchLatestL2Memories,
    setL2LatestLimit,
    setL2LatestSort,
    setL2LatestOffset,
    setL2LatestPage,
    getL2LatestCurrentPage,
    getL2LatestTotalPages,
    deleteL2Entries,
    updateL2Entry,
    searchL3,
    clearL3Search,
    fetchL3Nodes,
    fetchL3Edges,
    deleteL3Nodes,
    deleteL3Edge
  }
})
