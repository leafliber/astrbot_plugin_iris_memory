<template>
  <div class="l3-graph-view">
    <ComponentDisabled
      :status="status"
      :error="error"
      :error-type="errorType"
      component-name="L3 知识图谱"
      @retry="refreshState"
    >
      <!-- 顶部统计条 -->
      <v-card color="surface" variant="flat" class="iris-hero-card mb-3">
        <v-card-text class="py-2">
          <div class="stats-bar">
            <div class="stats-bar__left">
              <div class="d-flex align-center mr-3">
                <v-icon icon="mdi-graph" color="primary" class="mr-2" />
                <span class="text-h6">L3 知识图谱</span>
              </div>
              <v-chip size="small" variant="tonal" color="primary" class="ma-1">
                <v-icon icon="mdi-circle-multiple" start size="x-small" />
                {{ memoryStore.l3Stats.node_count }} 节点
              </v-chip>
              <v-chip size="small" variant="tonal" color="secondary" class="ma-1">
                <v-icon icon="mdi-link-variant" start size="x-small" />
                {{ memoryStore.l3Stats.edge_count }} 关系
              </v-chip>
              <v-chip size="small" variant="tonal" color="info" class="ma-1 d-none d-sm-inline-flex">
                <v-icon icon="mdi-eye" start size="x-small" />
                {{ memoryStore.l3FilteredGraph.nodes.length }} / {{ memoryStore.l3Graph.nodes.length }}
              </v-chip>
              <IsolationBadge
                type="memory"
                :enabled="isolationStatus.enable_group_memory_isolation"
                class="ma-1"
              />
            </div>
            <v-btn-toggle v-model="activeTab" mandatory color="primary" density="compact" class="flex-shrink-0">
              <v-btn value="graph" size="small">
                <v-icon icon="mdi-graph" class="mr-1" />
                图谱
              </v-btn>
              <v-btn value="nodes" size="small">
                <v-icon icon="mdi-circle-multiple" class="mr-1" />
                节点
              </v-btn>
              <v-btn value="edges" size="small">
                <v-icon icon="mdi-link-variant" class="mr-1" />
                关系
              </v-btn>
            </v-btn-toggle>
          </div>
        </v-card-text>
      </v-card>

      <!-- 图谱视图 -->
      <div v-show="activeTab === 'graph'" class="graph-layout">
        <!-- 左侧边栏 -->
        <div class="sidebar-col">
          <L3Sidebar
            :stats="memoryStore.l3Stats"
            :loading="memoryStore.l3Loading"
            :depth="memoryStore.l3Depth"
            :max-nodes="memoryStore.l3MaxNodes"
            :layout="memoryStore.l3Layout"
            :min-confidence="memoryStore.l3Filters.minConfidence"
            :available-node-types="memoryStore.l3AvailableNodeTypes"
            :available-relation-types="memoryStore.l3AvailableRelationTypes"
            :active-node-types="memoryStore.l3Filters.nodeTypes"
            :active-relation-types="memoryStore.l3Filters.relationTypes"
            :search-results="memoryStore.l3SearchResults"
            :search-loading="memoryStore.l3SearchLoading"
            :search-keyword="memoryStore.l3SearchKeyword"
            :group-id="memoryStore.l3Filters.groupId"
            :groups="groups"
            @search="handleSearch"
            @clear-search="handleClearSearch"
            @update:depth="handleDepthChange"
            @update:max-nodes="handleMaxNodesChange"
            @update:layout="handleLayoutChange"
            @update:min-confidence="handleMinConfidenceChange"
            @update:group-id="handleGroupChange"
            @toggle-node-type="memoryStore.toggleNodeTypeFilter"
            @toggle-relation-type="memoryStore.toggleRelationTypeFilter"
            @reset-filters="memoryStore.resetFilters"
            @random-node="handleRandomMainNode"
            @focus-node="handleFocusNode"
          />
        </div>

        <!-- 中央画布 -->
        <div class="canvas-col">
          <L3GraphCanvas
            ref="canvasRef"
            :nodes="memoryStore.l3FilteredGraph.nodes"
            :edges="memoryStore.l3FilteredGraph.edges"
            :loading="memoryStore.l3Loading"
            :start-node="memoryStore.l3StartNode"
            :layout="memoryStore.l3Layout"
            :can-go-back="memoryStore.canGoBack"
            :can-go-forward="memoryStore.canGoForward"
            @node-click="handleNodeClick"
            @node-dblclick="handleExpandNode"
            @edge-click="handleEdgeClick"
            @nav-back="memoryStore.navBack"
            @nav-forward="memoryStore.navForward"
          />
        </div>
      </div>

      <!-- 节点列表 -->
      <div v-show="activeTab === 'nodes'" class="l3-panel-container">
        <L3NodeListPanel
          :nodes="memoryStore.l3Nodes"
          :loading="memoryStore.l3NodesLoading"
          @focus-node="handleFocusNode"
          @expand="handleExpandNode"
          @delete="(id: string) => handleDeleteNodes([id])"
          @bulk-delete="handleBulkDeleteNodes"
        />
      </div>

      <!-- 关系列表 -->
      <div v-show="activeTab === 'edges'" class="l3-panel-container">
        <L3EdgeListPanel
          :edges="memoryStore.l3Edges"
          :loading="memoryStore.l3EdgesLoading"
          @focus-edge="handleEdgeFocus"
          @delete="handleDeleteEdge"
          @bulk-delete="handleBulkDeleteEdges"
        />
      </div>
    </ComponentDisabled>

    <!-- 节点详情抽屉 -->
    <L3NodeDrawer
      v-model="drawerOpen"
      :node="selectedNode"
      :edges="memoryStore.l3Graph.edges"
      :all-nodes="memoryStore.l3Graph.nodes"
      :loading="memoryStore.l3Loading"
      @expand="handleExpandNode"
      @delete="handleDrawerDelete"
      @focus-node="handleFocusNode"
    />

    <!-- 删除确认弹窗 -->
    <v-dialog v-model="deleteDialog" max-width="420">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-alert-circle" color="warning" class="mr-2" />
          确认删除
        </v-card-title>
        <v-card-text>{{ deleteMessage }}</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteDialog = false">取消</v-btn>
          <v-btn color="error" variant="tonal" :loading="deleting" @click="confirmDelete">
            确认删除
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- 删除提示 -->
    <v-snackbar v-model="snackbar" :color="snackbarColor" :timeout="3000">
      {{ snackbarText }}
    </v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useMemoryStore } from '@/stores'
import { useComponentState } from '@/composables/useComponentState'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import IsolationBadge from '@/components/IsolationBadge.vue'
import L3GraphCanvas from '@/components/l3/L3GraphCanvas.vue'
import L3Sidebar from '@/components/l3/L3Sidebar.vue'
import L3NodeDrawer from '@/components/l3/L3NodeDrawer.vue'
import L3NodeListPanel from '@/components/l3/L3NodeListPanel.vue'
import L3EdgeListPanel from '@/components/l3/L3EdgeListPanel.vue'
import { getGroupList } from '@/api/profile'
import { useIsolationStatus } from '@/composables/useIsolationStatus'
import type { KGNode, KGEdge, L3LayoutType, L3EdgeDetail } from '@/types'

const memoryStore = useMemoryStore()
const { status, error, errorType, refreshState } = useComponentState('l3_kg')
const { status: isolationStatus } = useIsolationStatus()

const activeTab = ref('graph')
const canvasRef = ref<InstanceType<typeof L3GraphCanvas> | null>(null)

// 群聊列表：供侧边栏群聊过滤下拉使用
const groups = ref<{ group_id: string; group_name?: string }[]>([])

const fetchGroups = async () => {
  try {
    const list = await getGroupList()
    groups.value = (list || []).map((g: any) => ({
      group_id: g.group_id,
      group_name: g.group_name,
    }))
  } catch (e) {
    console.error('获取群聊列表失败:', e)
    groups.value = []
  }
}

// 节点抽屉
const drawerOpen = ref(false)
const selectedNode = ref<KGNode | null>(null)

// 删除确认
const deleteDialog = ref(false)
const deleting = ref(false)
const deleteMessage = ref('')
const deleteAction = ref<() => Promise<void>>(async () => {})

// 提示条
const snackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref<'success' | 'error' | 'info'>('success')

const showSnackbar = (text: string, color: 'success' | 'error' | 'info' = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  snackbar.value = true
}

// ---- 图谱加载 ----
const loadGraph = () => memoryStore.fetchL3Graph()

// 随机主节点：显式重新让后端随机挑选一个起始节点
const handleRandomMainNode = () => memoryStore.fetchL3Graph()

const handleExpandNode = (nodeId: string) => {
  memoryStore.expandFromNode(nodeId)
}

// ---- 搜索 ----
const handleSearch = (keyword: string) => memoryStore.searchL3(keyword)
const handleClearSearch = () => memoryStore.clearL3Search()

// ---- 控制 ----
// 深度/最大节点数变更：保留当前主节点重新加载，不随机化
const handleDepthChange = (depth: number) => {
  memoryStore.setDepth(depth)
  memoryStore.refreshL3Graph()
}
const handleMaxNodesChange = (maxNodes: number) => {
  memoryStore.setMaxNodes(maxNodes)
  memoryStore.refreshL3Graph()
}
const handleLayoutChange = (layout: L3LayoutType) => {
  memoryStore.setLayout(layout)
}
const handleMinConfidenceChange = (v: number) => {
  memoryStore.setMinConfidence(v)
}

// 群聊过滤变更：若当前主节点不属于目标群则后端随机选一个，否则保留主节点
const handleGroupChange = (groupId: string | null) => {
  memoryStore.setGroupFilter(groupId)
  const currentNode = memoryStore.l3StartNode
  const keepMain = !!currentNode && (!groupId || currentNode.group_id === groupId)
  if (keepMain) {
    memoryStore.refreshL3Graph()
  } else {
    memoryStore.fetchL3Graph()
  }
  memoryStore.fetchL3Stats()
}

// ---- 画布事件 ----
const handleNodeClick = (node: KGNode) => {
  selectedNode.value = node
  drawerOpen.value = true
}

const handleEdgeClick = (_edge: KGEdge) => {
  // 边点击暂不打开抽屉，可后续扩展
}

const handleEdgeFocus = (edge: L3EdgeDetail) => {
  // 切到图谱视图并聚焦源节点
  handleFocusNode(edge.source.id)
}

// ---- 聚焦节点（来自搜索/列表/抽屉邻居）：切到图谱 Tab 并在画布上聚焦 ----
const handleFocusNode = async (nodeId: string) => {
  if (activeTab.value !== 'graph') {
    activeTab.value = 'graph'
    await nextTick()
  }
  await nextTick()
  canvasRef.value?.focusNode(nodeId)
  // 同时在抽屉中展示该节点（若存在于当前图谱）
  const node = memoryStore.l3Graph.nodes.find((n) => n.id === nodeId)
  if (node) {
    selectedNode.value = node
    drawerOpen.value = true
  }
}

// ---- 节点删除 ----
const handleDeleteNodes = (ids: string[]) => {
  deleteMessage.value =
    ids.length === 1
      ? '确定要删除该节点吗？与之关联的关系也将被删除。此操作不可撤销。'
      : `确定要删除 ${ids.length} 个节点吗？与之关联的关系也将被删除。此操作不可撤销。`
  deleteAction.value = async () => {
    await memoryStore.deleteL3Nodes(ids)
    showSnackbar(`已删除 ${ids.length} 个节点`)
    if (selectedNode.value && ids.includes(selectedNode.value.id)) {
      drawerOpen.value = false
      selectedNode.value = null
    }
    loadGraph()
    memoryStore.fetchL3Stats()
  }
  deleteDialog.value = true
}

const handleBulkDeleteNodes = (ids: string[]) => handleDeleteNodes(ids)

const handleDrawerDelete = (nodeId: string) => {
  drawerOpen.value = false
  handleDeleteNodes([nodeId])
}

// ---- 关系删除 ----
const handleDeleteEdge = (edge: L3EdgeDetail) => {
  deleteMessage.value = `确定要删除关系「${edge.source.name} → ${edge.target.name}」吗？此操作不可撤销。`
  deleteAction.value = async () => {
    await memoryStore.deleteL3Edge(edge.source.id, edge.target.id, edge.relation)
    showSnackbar('关系已删除')
    loadGraph()
    memoryStore.fetchL3Stats()
  }
  deleteDialog.value = true
}

const handleBulkDeleteEdges = (edges: L3EdgeDetail[]) => {
  deleteMessage.value = `确定要删除 ${edges.length} 条关系吗？此操作不可撤销。`
  deleteAction.value = async () => {
    for (const e of edges) {
      await memoryStore.deleteL3Edge(e.source.id, e.target.id, e.relation)
    }
    showSnackbar(`已删除 ${edges.length} 条关系`)
    loadGraph()
    memoryStore.fetchL3Stats()
  }
  deleteDialog.value = true
}

const confirmDelete = async () => {
  deleting.value = true
  try {
    await deleteAction.value()
    deleteDialog.value = false
  } catch (e) {
    console.error('删除失败:', e)
    showSnackbar('删除失败', 'error')
  } finally {
    deleting.value = false
  }
}

// ---- Tab 切换懒加载列表 ----
watch(activeTab, (tab) => {
  if (tab === 'nodes' && memoryStore.l3Nodes.length === 0) {
    memoryStore.fetchL3Nodes()
  } else if (tab === 'edges' && memoryStore.l3Edges.length === 0) {
    memoryStore.fetchL3Edges()
  }
})

const handleRefresh = () => {
  if (activeTab.value === 'graph') {
    loadGraph()
    memoryStore.fetchL3Stats()
  } else if (activeTab.value === 'nodes') {
    memoryStore.fetchL3Nodes(memoryStore.l3NodesKeyword || undefined)
  } else if (activeTab.value === 'edges') {
    memoryStore.fetchL3Edges(memoryStore.l3EdgesKeyword || undefined)
  }
}

onMounted(() => {
  loadGraph()
  memoryStore.fetchL3Stats()
  fetchGroups()
  window.addEventListener('iris:refresh', handleRefresh)
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.l3-graph-view {
  height: 100%;
  display: flex;
  flex-direction: column;
}

/* 顶部统计条：flex 布局，自动换行 */
.stats-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}

.stats-bar__left {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

/* 标题字重微调 */
.l3-graph-view :deep(.text-h6) {
  font-weight: 700;
  letter-spacing: 0.01em;
}

.graph-layout {
  display: flex;
  gap: 12px;
  align-items: stretch;
  /* 顶部条 + 外边距 + padding 综合预留 */
  height: calc(100vh - 200px);
  min-height: 480px;
  flex: 1;
}

.sidebar-col {
  width: 320px;
  flex-shrink: 0;
  overflow: visible;
}

.canvas-col {
  flex: 1;
  min-width: 0;
  display: flex;
}

.canvas-col > * {
  width: 100%;
}

/* 中等屏幕：收窄侧栏 */
@media (max-width: 1280px) {
  .graph-layout {
    flex-direction: column;
    height: auto;
    min-height: unset;
  }

  .sidebar-col {
    width: 100%;
    height: 280px;
    overflow: hidden;
  }

  .canvas-col {
    flex: none;
    height: 500px;
  }
}

/* 小屏幕：进一步压缩 */
@media (max-width: 768px) {
  .graph-layout {
    gap: 8px;
  }

  .sidebar-col {
    height: 220px;
  }

  .canvas-col {
    height: 400px;
  }
}

/* Tab 按钮组 */
.l3-graph-view :deep(.v-btn-toggle) {
  border-radius: 8px;
  overflow: hidden;
}

/* 节点/边列表面板容器：建立高度约束，使子组件的 h-100 / flex-grow-1
   能形成内部滚动表格，而非整页滚动。
   此前 v-show 的 div 无高度，导致 height:100% 解析为 auto，表格无法滚动。 */
.l3-panel-container {
  height: calc(100vh - 200px);
  min-height: 400px;
  display: flex;
  flex-direction: column;
}

@media (max-width: 1280px) {
  .l3-panel-container {
    height: auto;
    min-height: 500px;
  }
}
</style>
