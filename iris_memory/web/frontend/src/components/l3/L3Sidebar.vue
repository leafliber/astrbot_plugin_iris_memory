<template>
  <div class="l3-sidebar">
    <!-- 统一搜索 -->
    <v-card color="surface" variant="flat" class="mb-3 iris-card iris-card-hover">
      <v-card-text class="pb-2">
        <v-text-field
          v-model="keyword"
          placeholder="搜索节点 / 关系…"
          prepend-inner-icon="mdi-magnify"
          variant="outlined"
          density="compact"
          hide-details
          clearable
          @keyup.enter="handleSearch"
          @click:clear="handleClearSearch"
        />
        <v-btn
          color="primary"
          size="small"
          block
          class="mt-2"
          :loading="searchLoading"
          @click="handleSearch"
        >
          <v-icon icon="mdi-magnify" class="mr-1" />
          搜索
        </v-btn>
      </v-card-text>

      <!-- 搜索结果下拉 -->
      <div v-if="hasResults" class="search-results">
        <div v-if="searchResults.nodes.length" class="mb-2">
          <div class="text-caption text-medium-emphasis px-3 py-1">
            节点 ({{ searchResults.nodes.length }})
          </div>
          <v-list density="compact" class="pa-0 bg-transparent">
            <v-list-item
              v-for="node in searchResults.nodes"
              :key="node.id"
              density="compact"
              @click="emit('focus-node', node.id)"
            >
              <template #prepend>
                <v-icon :icon="getNodeIcon(node.label)" :color="getTypeColor(node.label)" size="small" />
              </template>
              <v-list-item-title class="text-body-2">{{ node.name || node.id }}</v-list-item-title>
              <v-list-item-subtitle class="text-caption">{{ getNodeLabel(node.label) }}</v-list-item-subtitle>
              <template #append>
                <v-chip size="x-small" :color="getConfidenceColor(node.confidence)" variant="tonal">
                  {{ (node.confidence * 100).toFixed(0) }}%
                </v-chip>
              </template>
            </v-list-item>
          </v-list>
        </div>
        <div v-if="searchResults.edges.length">
          <div class="text-caption text-medium-emphasis px-3 py-1">
            关系 ({{ searchResults.edges.length }})
          </div>
          <v-list density="compact" class="pa-0 bg-transparent">
            <v-list-item
              v-for="(edge, idx) in searchResults.edges"
              :key="idx"
              density="compact"
              @click="emit('focus-node', edge.source.id)"
            >
              <template #prepend>
                <v-icon icon="mdi-arrow-right-bold" color="secondary" size="small" />
              </template>
              <v-list-item-title class="text-body-2">{{ getRelationLabel(edge.relation) }}</v-list-item-title>
              <v-list-item-subtitle class="text-caption">
                {{ edge.source.name }} → {{ edge.target.name }}
              </v-list-item-subtitle>
            </v-list-item>
          </v-list>
        </div>
      </div>
    </v-card>

    <!-- 图谱控制 -->
    <v-card color="surface" variant="flat" class="mb-3 iris-card iris-card-hover">
      <v-card-text>
        <div class="text-subtitle-2 mb-3 iris-section-title">
          <v-icon icon="mdi-tune" size="small" class="mr-1" />
          图谱控制
        </div>

        <!-- 群聊过滤 -->
        <div class="mb-3">
          <span class="text-caption text-medium-emphasis">群聊范围</span>
          <v-select
            :model-value="groupId"
            :items="groupOptions"
            item-title="title"
            item-value="value"
            variant="outlined"
            density="compact"
            hide-details
            clearable
            placeholder="全部群聊"
            class="mt-1"
            @update:model-value="(v: string | null) => emit('update:groupId', v ?? null)"
          />
        </div>

        <div class="mb-3">
          <div class="d-flex justify-space-between align-center">
            <span class="text-caption text-medium-emphasis">拓展深度</span>
            <v-chip size="x-small" variant="tonal">{{ depth }} 跳</v-chip>
          </div>
          <v-slider
            :model-value="depth"
            :min="1"
            :max="3"
            :step="1"
            density="compact"
            hide-details
            color="primary"
            @update:model-value="(v: number) => emit('update:depth', v)"
          />
        </div>

        <div class="mb-3">
          <div class="d-flex justify-space-between align-center">
            <span class="text-caption text-medium-emphasis">最大节点数</span>
            <v-chip size="x-small" variant="tonal">{{ maxNodes }}</v-chip>
          </div>
          <v-slider
            :model-value="maxNodes"
            :min="10"
            :max="100"
            :step="5"
            density="compact"
            hide-details
            color="primary"
            @update:model-value="(v: number) => emit('update:maxNodes', v)"
          />
        </div>

        <div class="mb-1">
          <span class="text-caption text-medium-emphasis">布局算法</span>
        </div>
        <v-btn-toggle
          :model-value="layout"
          mandatory
          density="compact"
          color="primary"
          class="w-100 mb-3"
          @update:model-value="(v: any) => emit('update:layout', v as L3LayoutType)"
        >
          <v-btn value="force" size="x-small" class="flex-grow-1">力导向</v-btn>
          <v-btn value="dagre" size="x-small" class="flex-grow-1">层次</v-btn>
          <v-btn value="radial" size="x-small" class="flex-grow-1">辐射</v-btn>
          <v-btn value="concentric" size="x-small" class="flex-grow-1">同心</v-btn>
        </v-btn-toggle>

        <v-btn color="primary" variant="tonal" block size="small" :loading="loading" @click="emit('random-node')">
          <v-icon icon="mdi-shuffle" class="mr-1" />
          随机主节点
        </v-btn>
      </v-card-text>
    </v-card>

    <!-- 过滤器 -->
    <v-card color="surface" variant="flat" class="mb-3 iris-card iris-card-hover">
      <v-card-text>
        <div class="d-flex align-center mb-3">
          <v-icon icon="mdi-filter-variant" size="small" class="mr-1" />
          <span class="text-subtitle-2 iris-section-title">过滤器</span>
          <v-spacer />
          <v-btn
            v-if="hasActiveFilters"
            icon="mdi-filter-remove"
            size="x-small"
            variant="text"
            @click="emit('reset-filters')"
          >
            <v-tooltip activator="parent" location="bottom">清除过滤</v-tooltip>
          </v-btn>
        </div>

        <!-- 置信度阈值 -->
        <div class="mb-3">
          <div class="d-flex justify-space-between align-center">
            <span class="text-caption text-medium-emphasis">最低置信度</span>
            <v-chip size="x-small" :color="getConfidenceColor(minConfidence)" variant="tonal">
              {{ (minConfidence * 100).toFixed(0) }}%
            </v-chip>
          </div>
          <v-slider
            :model-value="minConfidence"
            :min="0"
            :max="1"
            :step="0.05"
            density="compact"
            hide-details
            color="primary"
            @update:model-value="(v: number) => emit('update:minConfidence', v)"
          />
        </div>

        <!-- 节点类型过滤 -->
        <div class="mb-2">
          <span class="text-caption text-medium-emphasis">节点类型</span>
        </div>
        <div class="filter-chips mb-3">
          <v-chip
            v-for="item in availableNodeTypes"
            :key="item.type"
            size="x-small"
            :color="getTypeColor(item.type)"
            :variant="isNodeTypeActive(item.type) ? 'flat' : 'tonal'"
            class="ma-1"
            @click="emit('toggle-node-type', item.type)"
          >
            <v-icon :icon="getNodeIcon(item.type)" start size="x-small" />
            {{ getNodeLabel(item.type) }}
            <span class="ml-1 text-caption">({{ item.count }})</span>
          </v-chip>
          <span v-if="availableNodeTypes.length === 0" class="text-caption text-disabled">
            无数据
          </span>
        </div>

        <!-- 关系类型过滤 -->
        <div class="mb-2">
          <span class="text-caption text-medium-emphasis">关系类型</span>
        </div>
        <div class="filter-chips">
          <v-chip
            v-for="item in availableRelationTypes"
            :key="item.type"
            size="x-small"
            variant="outlined"
            :color="isRelationTypeActive(item.type) ? 'primary' : 'default'"
            class="ma-1"
            @click="emit('toggle-relation-type', item.type)"
          >
            {{ getRelationLabel(item.type) }}
            <span class="ml-1 text-caption">({{ item.count }})</span>
          </v-chip>
          <span v-if="availableRelationTypes.length === 0" class="text-caption text-disabled">
            无数据
          </span>
        </div>
      </v-card-text>
    </v-card>

    <!-- 全局统计 -->
    <v-card color="surface" variant="flat" class="mb-3 iris-card iris-card-hover">
      <v-card-text>
        <div class="text-subtitle-2 mb-3 iris-section-title">
          <v-icon icon="mdi-chart-box" size="small" class="mr-1" />
          全局统计
        </div>
        <v-row dense>
          <v-col cols="6">
            <div class="stat-box">
              <div class="stat-value text-primary">{{ stats.node_count }}</div>
              <div class="stat-label">总节点</div>
            </div>
          </v-col>
          <v-col cols="6">
            <div class="stat-box">
              <div class="stat-value text-secondary">{{ stats.edge_count }}</div>
              <div class="stat-label">总关系</div>
            </div>
          </v-col>
        </v-row>

        <v-divider class="my-3" />

        <div class="text-caption text-medium-emphasis mb-2">节点类型分布</div>
        <div v-for="(count, type) in stats.node_types" :key="type" class="type-bar mb-1">
          <v-icon :icon="getNodeIcon(type)" :color="getTypeColor(type)" size="x-small" class="mr-1" />
          <span class="text-body-2 flex-grow-1">{{ getNodeLabel(type) }}</span>
          <span class="text-caption text-medium-emphasis">{{ count }}</span>
        </div>

        <div class="text-caption text-medium-emphasis mb-2 mt-3">关系类型分布</div>
        <div v-for="(count, type) in stats.relation_types" :key="type" class="type-bar mb-1">
          <span class="text-body-2 flex-grow-1">{{ getRelationLabel(type) }}</span>
          <span class="text-caption text-medium-emphasis">{{ count }}</span>
        </div>
      </v-card-text>
    </v-card>

    <!-- 图例 -->
    <v-card color="surface" variant="flat" class="iris-card iris-card-hover">
      <v-card-text>
        <div class="text-subtitle-2 mb-2 iris-section-title">
          <v-icon icon="mdi-palette" size="small" class="mr-1" />
          图例
        </div>
        <div class="legend-grid">
          <div v-for="(color, type) in legendTypes" :key="type" class="legend-item">
            <span class="legend-dot" :style="{ background: color }" />
            <span class="text-caption">{{ getNodeLabel(type) }}</span>
          </div>
        </div>
      </v-card-text>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import type {
  L3Stats,
  L3LayoutType,
  L3SearchNodeResult,
  L3SearchEdgeResult,
} from '@/types'
import {
  getNodeIcon,
  getTypeColor,
  getNodeLabel,
  getRelationLabel,
  getConfidenceColor,
  resolveThemeColor,
  NODE_TYPE_COLORS,
} from '@/composables/l3Constants'

const props = defineProps<{
  stats: L3Stats
  loading: boolean
  depth: number
  maxNodes: number
  layout: L3LayoutType
  minConfidence: number
  availableNodeTypes: { type: string; count: number }[]
  availableRelationTypes: { type: string; count: number }[]
  activeNodeTypes: string[]
  activeRelationTypes: string[]
  searchResults: { nodes: L3SearchNodeResult[]; edges: L3SearchEdgeResult[] }
  searchLoading: boolean
  searchKeyword: string
  groupId: string | null
  groups: { group_id: string; group_name?: string }[]
}>()

const emit = defineEmits<{
  search: [keyword: string]
  'clear-search': []
  'update:depth': [depth: number]
  'update:maxNodes': [maxNodes: number]
  'update:layout': [layout: L3LayoutType]
  'update:minConfidence': [v: number]
  'update:groupId': [groupId: string | null]
  'toggle-node-type': [type: string]
  'toggle-relation-type': [type: string]
  'reset-filters': []
  'random-node': []
  'focus-node': [nodeId: string]
}>()

const keyword = ref(props.searchKeyword)

watch(
  () => props.searchKeyword,
  (v) => {
    if (v !== keyword.value) keyword.value = v
  }
)

const hasResults = computed(
  () => props.searchResults.nodes.length > 0 || props.searchResults.edges.length > 0
)

// 群聊下拉选项：将后端 group 列表映射为 { title, value }
const groupOptions = computed(() =>
  props.groups.map((g) => ({
    title: g.group_name || g.group_id,
    value: g.group_id,
  }))
)

const hasActiveFilters = computed(
  () =>
    props.activeNodeTypes.length > 0 ||
    props.activeRelationTypes.length > 0 ||
    props.minConfidence > 0
)

const isNodeTypeActive = (type: string) => props.activeNodeTypes.includes(type)
const isRelationTypeActive = (type: string) => props.activeRelationTypes.includes(type)

// 图例：从全局统计的节点类型生成，解析主题色
const legendTypes = computed<Record<string, string>>(() => {
  const result: Record<string, string> = {}
  Object.keys(props.stats.node_types).forEach((type) => {
    result[type] = resolveThemeColor(getTypeColor(type), '#5c6bc0')
  })
  // 也补充预设类型
  Object.entries(NODE_TYPE_COLORS).forEach(([type, colorName]) => {
    if (!result[type]) result[type] = resolveThemeColor(colorName, '#5c6bc0')
  })
  return result
})

const handleSearch = () => {
  const v = keyword.value?.trim()
  if (v) emit('search', v)
}

const handleClearSearch = () => {
  keyword.value = ''
  emit('clear-search')
}
</script>

<style scoped>
.l3-sidebar {
  /* 高度由父容器控制，自身只负责滚动 */
  max-height: 100%;
  overflow-y: auto;
  padding-right: 4px;
  /* 自定义滚动条 */
  scrollbar-width: thin;
  scrollbar-color: rgba(var(--v-theme-on-surface), 0.2) transparent;
}

.l3-sidebar::-webkit-scrollbar {
  width: 6px;
}

.l3-sidebar::-webkit-scrollbar-thumb {
  background: rgba(var(--v-theme-on-surface), 0.18);
  border-radius: 3px;
}

.l3-sidebar::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--v-theme-on-surface), 0.32);
}

/* 分区标题统一风格（与 iris-section-title 协同） */
.text-subtitle-2 {
  display: flex;
  align-items: center;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: rgba(var(--v-theme-on-surface), 0.85);
}

.search-results {
  max-height: 320px;
  overflow-y: auto;
  border-top: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  background: rgba(var(--v-theme-on-surface), 0.02);
}

.filter-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
}

.filter-chips :deep(.v-chip) {
  transition: all 0.15s ease;
  font-weight: 500;
}

.filter-chips :deep(.v-chip:hover) {
  transform: translateY(-1px);
}

.stat-box {
  text-align: center;
  padding: 10px 8px;
  background: linear-gradient(
    135deg,
    rgba(var(--v-theme-primary), 0.08),
    rgba(var(--v-theme-primary), 0.02)
  );
  border-radius: 10px;
  border: 1px solid rgba(var(--v-theme-primary), 0.12);
}

.stat-value {
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}

.stat-label {
  font-size: 0.7rem;
  color: rgba(var(--v-theme-on-surface), 0.6);
  margin-top: 2px;
}

.type-bar {
  display: flex;
  align-items: center;
  padding: 3px 6px;
  border-radius: 6px;
  transition: background 0.15s ease;
}

.type-bar:hover {
  background: rgba(var(--v-theme-on-surface), 0.04);
}

.legend-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px 6px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8), 0 1px 3px rgba(0, 0, 0, 0.15);
}

/* 布局按钮组紧凑化 */
.l3-sidebar :deep(.v-btn-toggle .v-btn) {
  font-size: 0.75rem !important;
  letter-spacing: 0;
}
</style>
