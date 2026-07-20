// ============================================
// API 类型定义
// ============================================

// 通用响应
export interface ApiResponse<T = unknown> {
  success: boolean
  error?: string
  data?: T
}

// ============================================
// 组件状态类型
// ============================================

export type ComponentStatus = 'pending' | 'initializing' | 'available' | 'unavailable'

export type ErrorType = 'disabled' | 'dependency_missing' | 'connection_failed' | 'other'

export interface ComponentState {
  status: ComponentStatus
  error: string | null
  error_type: ErrorType | null
}

export type GlobalStatus = 'pending' | 'initializing' | 'available'

export interface ComponentStates {
  [key: string]: ComponentState
}

// ============================================
// 记忆相关类型
// ============================================

// L1 消息
export interface L1Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
  user_id?: string
  user_name?: string
}

// L1 队列项
export interface L1QueueItem {
  group_id: string
  group_name?: string
  is_private?: boolean
  user_id?: string
  message_count: number
  total_tokens: number
}

// L2 记忆条目
export type L2SortField = 'timestamp' | 'access_count' | 'confidence' | 'last_access_time'

export type L2SortOrder = 'asc' | 'desc'

export interface L2Memory {
  id: string
  content: string
  score: number
  metadata: Record<string, unknown>
  timestamp?: string
  access_count?: number
  last_access_time?: string
  confidence?: number
  source?: string
  group_id?: string
}

// L3 图谱节点（富字段，含详情）
export interface KGNode {
  id: string
  label: string
  name: string
  content?: string
  confidence: number
  access_count?: number
  last_access_time?: string
  created_time?: string
  source_memory_id?: string
  group_id?: string
  properties?: Record<string, string>
}

// L3 图谱边（富字段，含权重/置信度）
export interface KGEdge {
  source: string
  target: string
  relation: string
  weight?: number
  confidence?: number
  access_count?: number
  created_time?: string
}

// L3 图谱
export interface KGGraph {
  nodes: KGNode[]
  edges: KGEdge[]
  start_node?: KGNode | null
}

// L3 图谱全局统计
export interface L3Stats {
  available: boolean
  node_count: number
  edge_count: number
  node_types: Record<string, number>
  relation_types: Record<string, number>
  persist_dir?: string
}

// L3 图谱布局类型
export type L3LayoutType = 'force' | 'dagre' | 'radial' | 'concentric'

// L3 图谱过滤器
export interface L3Filters {
  nodeTypes: string[]          // 选中的节点类型（空=全部）
  relationTypes: string[]      // 选中的关系类型（空=全部）
  groupId: string | null       // 群组过滤
  minConfidence: number        // 最低置信度阈值
}

export interface L3SearchNodeResult {
  id: string
  label: string
  name: string
  confidence: number
}

export interface L3SearchEdgeResult {
  source: { id: string; name: string; label: string }
  target: { id: string; name: string; label: string }
  relation: string
}

// L3 节点详情（列表页用）
export interface L3NodeDetail {
  id: string
  label: string
  name: string
  content: string
  confidence: number
  group_id?: string
  access_count?: number
  created_time?: string
  source_memory_id?: string
  properties?: Record<string, string>
}

// L3 关系详情（列表页用）
export interface L3EdgeDetail {
  source: { id: string; label: string; name: string }
  target: { id: string; label: string; name: string }
  relation: string
  confidence: number
  weight?: number
  access_count?: number
  created_time?: string
}

// L2 搜索请求
export interface L2SearchRequest {
  query: string
  group_id?: string
  top_k?: number
}

// L2 搜索结果
export interface L2SearchResponse {
  results: L2Memory[]
}

// L1 列表响应
export interface L1ListResponse {
  messages: L1Message[]
  count: number
}

// ============================================
// 画像相关类型
// ============================================

// 群聊画像
export interface GroupProfile {
  group_id: string
  group_name?: string
  interests?: string[]
  atmosphere_tags?: string[]
  long_term_tags?: string[]
  blacklist_topics?: string[]
  custom_fields?: Record<string, string>
  version?: number
}

// 用户画像
export interface UserProfile {
  user_id: string
  user_name?: string
  historical_names?: string[]
  personality_tags?: string[]
  interests?: string[]
  occupation?: string
  language_style?: string
  communication_style?: string
  emotional_baseline?: string
  favorability?: number
  bot_relationship?: string
  important_dates?: Array<{ date: string; description: string }>
  taboo_topics?: string[]
  important_events?: string[]
  custom_fields?: Record<string, string>
  version?: number
}

// 群聊列表项
export interface GroupListItem {
  group_id: string
  group_name?: string
}

// 用户列表项
export interface UserListItem {
  user_id: string
  nickname?: string
  group_id?: string
}

// ============================================
// 统计相关类型
// ============================================

// Token 统计
export interface TokenStats {
  total_input_tokens: number
  total_output_tokens: number
  total_calls: number
}

export interface TokenStatsResponse {
  global: TokenStats
  l1_summarizer: TokenStats
  [key: string]: TokenStats
}

// L1 统计
export interface L1Stats {
  queue_count?: number
  total_messages?: number
  total_tokens?: number
  max_capacity?: number
  max_queue_length?: number
}

// L2 统计
export interface L2Stats {
  total_count?: number
  group_count?: number
}

// 记忆统计
export interface MemoryStats {
  l1: L1Stats
  l2: L2Stats
  l3: L3Stats
}

// 系统统计（新格式）
export interface SystemStats {
  components: ComponentStates
  global_status: GlobalStatus
  uptime: number
}

// 知识图谱统计
export interface KGStats {
  node_count: number
  edge_count: number
  node_types: Record<string, number>
  relation_types: Record<string, number>
}

// ============================================
// 仪表盘类型
// ============================================

export interface DashboardData {
  system: SystemStats
  memory: MemoryStats
  token: TokenStatsResponse
  kg: KGStats
}

// ============================================
// 组件状态映射
// ============================================

export const COMPONENT_DISPLAY_NAMES: Record<string, string> = {
  l1_buffer: 'L1 缓冲',
  l2_memory: 'L2 记忆',
  l3_kg: 'L3 图谱',
  profile: '画像管理',
  llm_manager: 'LLM 管理器'
}

export const ERROR_TYPE_DISPLAY_NAMES: Record<ErrorType, string> = {
  disabled: '已禁用',
  dependency_missing: '依赖缺失',
  connection_failed: '连接失败',
  other: '其他原因'
}

export const STATUS_DISPLAY_NAMES: Record<ComponentStatus, string> = {
  pending: '等待初始化',
  initializing: '正在初始化',
  available: '可用',
  unavailable: '不可用'
}

// ============================================
// 导入导出类型
// ============================================

export interface L2ImportStats {
  total_count: number
  imported_count: number
  skipped_count: number
  error_count: number
}

export interface L3ImportStats {
  imported_nodes: number
  imported_edges: number
  skipped_nodes: number
  error_count: number
}

export interface ProfileImportStats {
  imported_groups: number
  imported_users: number
  skipped: number
  error_count: number
}

export interface FullImportResult {
  l2_memory: L2ImportStats | { error: string } | null
  l3_kg: L3ImportStats | { error: string } | null
  profiles: ProfileImportStats | { error: string } | null
}

// ============================================
// 管理操作类型
// ============================================

export type TaskName = 'dream' | 'cache_cleanup'

export interface TaskStatus {
  running: boolean
}

export type TasksStatusMap = Record<TaskName, TaskStatus>

export const TASK_DISPLAY_NAMES: Record<TaskName, string> = {
  dream: '梦境任务',
  cache_cleanup: '缓存清理'
}
