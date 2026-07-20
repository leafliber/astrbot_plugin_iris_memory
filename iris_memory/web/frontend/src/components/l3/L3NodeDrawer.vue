<template>
  <v-navigation-drawer
    :model-value="modelValue"
    @update:model-value="(v: boolean) => emit('update:modelValue', v)"
    location="right"
    width="400"
    temporary
    scrim
  >
    <template v-if="node">
      <!-- 抽屉头部 -->
      <v-card color="surface" variant="flat" class="drawer-header">
        <v-card-text class="pa-4">
          <div class="d-flex align-center mb-2">
            <v-icon
              :icon="getNodeIcon(node.label)"
              :color="getTypeColor(node.label)"
              size="large"
              class="mr-3"
            />
            <div class="flex-grow-1">
              <div class="text-h6">{{ node.name || node.id }}</div>
              <div class="text-caption text-medium-emphasis">
                {{ getNodeLabel(node.label) }}
              </div>
            </div>
            <v-btn
              icon="mdi-close"
              variant="text"
              size="small"
              @click="emit('update:modelValue', false)"
            />
          </div>

          <div class="d-flex gap-2 flex-wrap">
            <v-chip size="small" :color="getConfidenceColor(node.confidence)" variant="tonal">
              <v-icon icon="mdi-shield-check" start size="x-small" />
              置信度 {{ (node.confidence * 100).toFixed(0) }}%
            </v-chip>
            <v-chip v-if="node.group_id" size="small" variant="tonal" color="info">
              <v-icon icon="mdi-account-group" start size="x-small" />
              {{ node.group_id }}
            </v-chip>
            <v-chip v-if="degree !== null" size="small" variant="tonal" color="accent">
              <v-icon icon="mdi-link-variant" start size="x-small" />
              {{ degree }} 连接
            </v-chip>
          </div>
        </v-card-text>
      </v-card>

      <v-divider />

      <!-- 操作按钮 -->
      <div class="drawer-actions pa-3 d-flex gap-2">
        <v-btn
          color="primary"
          variant="tonal"
          size="small"
          class="flex-grow-1"
          :loading="loading"
          @click="emit('expand', node.id)"
        >
          <v-icon icon="mdi-arrow-expand" class="mr-1" />
          以此展开
        </v-btn>
        <v-btn
          color="error"
          variant="tonal"
          size="small"
          class="flex-grow-1"
          @click="emit('delete', node.id)"
        >
          <v-icon icon="mdi-delete" class="mr-1" />
          删除
        </v-btn>
      </div>

      <v-divider />

      <!-- 内容描述 -->
      <div v-if="node.content" class="pa-4">
        <div class="text-subtitle-2 mb-2">
          <v-icon icon="mdi-text" size="small" class="mr-1" />
          描述
        </div>
        <div class="text-body-2 node-content">{{ node.content }}</div>
      </div>

      <v-divider v-if="node.content" />

      <!-- 邻居列表 -->
      <div class="pa-4">
        <div class="d-flex align-center mb-2">
          <v-icon icon="mdi-sitemap" size="small" class="mr-1" />
          <span class="text-subtitle-2">关联节点</span>
          <v-spacer />
          <v-chip size="x-small" variant="tonal">{{ neighbors.length }}</v-chip>
        </div>
        <div v-if="neighbors.length === 0" class="text-caption text-disabled py-2">
          无关联节点
        </div>
        <v-list density="compact" class="pa-0 bg-transparent iris-list">
          <v-list-item
            v-for="nb in neighbors"
            :key="nb.node.id"
            density="compact"
            @click="emit('focus-node', nb.node.id)"
          >
            <template #prepend>
              <v-icon :icon="getNodeIcon(nb.node.label)" :color="getTypeColor(nb.node.label)" size="small" />
            </template>
            <v-list-item-title class="text-body-2">{{ nb.node.name || nb.node.id }}</v-list-item-title>
            <v-list-item-subtitle class="text-caption">
              <v-icon :icon="nb.direction === 'out' ? 'mdi-arrow-right' : 'mdi-arrow-left'" size="x-small" />
              {{ getRelationLabel(nb.relation) }}
            </v-list-item-subtitle>
            <template #append>
              <v-chip size="x-small" :color="getConfidenceColor(nb.node.confidence)" variant="tonal">
                {{ (nb.node.confidence * 100).toFixed(0) }}%
              </v-chip>
            </template>
          </v-list-item>
        </v-list>
      </div>

      <v-divider />

      <!-- 元数据 -->
      <div class="pa-4">
        <div class="text-subtitle-2 mb-2">
          <v-icon icon="mdi-information" size="small" class="mr-1" />
          元数据
        </div>
        <v-table density="compact" class="bg-transparent iris-table">
          <tbody>
            <tr>
              <td class="text-caption text-medium-emphasis" style="width: 100px">ID</td>
              <td class="text-caption text-truncate">{{ node.id }}</td>
            </tr>
            <tr v-if="node.access_count !== undefined">
              <td class="text-caption text-medium-emphasis">访问次数</td>
              <td class="text-caption">{{ node.access_count }}</td>
            </tr>
            <tr v-if="node.created_time">
              <td class="text-caption text-medium-emphasis">创建时间</td>
              <td class="text-caption">{{ formatTime(node.created_time) }}</td>
            </tr>
            <tr v-if="node.last_access_time">
              <td class="text-caption text-medium-emphasis">最后访问</td>
              <td class="text-caption">{{ formatTime(node.last_access_time) }}</td>
            </tr>
            <tr v-if="node.source_memory_id">
              <td class="text-caption text-medium-emphasis">来源记忆</td>
              <td class="text-caption text-truncate">{{ node.source_memory_id }}</td>
            </tr>
          </tbody>
        </v-table>
      </div>

      <!-- 扩展属性 -->
      <template v-if="node.properties && Object.keys(node.properties).length > 0">
        <v-divider />
        <div class="pa-4">
          <div class="text-subtitle-2 mb-2">
            <v-icon icon="mdi-code-json" size="small" class="mr-1" />
            扩展属性
          </div>
          <div v-for="(value, key) in node.properties" :key="key" class="prop-row">
            <span class="text-caption text-medium-emphasis">{{ key }}</span>
            <span class="text-body-2">{{ value }}</span>
          </div>
        </div>
      </template>
    </template>
  </v-navigation-drawer>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { KGNode, KGEdge } from '@/types'
import {
  getNodeIcon,
  getTypeColor,
  getNodeLabel,
  getRelationLabel,
  getConfidenceColor,
  formatTime,
} from '@/composables/l3Constants'

const props = defineProps<{
  modelValue: boolean
  node: KGNode | null
  edges: KGEdge[]
  allNodes: KGNode[]
  loading: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [v: boolean]
  expand: [nodeId: string]
  delete: [nodeId: string]
  'focus-node': [nodeId: string]
}>()

interface Neighbor {
  node: KGNode
  relation: string
  direction: 'out' | 'in'
}

// 从边数据计算邻居
const neighbors = computed<Neighbor[]>(() => {
  if (!props.node) return []
  const nodeId = props.node.id
  const result: Neighbor[] = []
  const nodeMap = new Map(props.allNodes.map((n) => [n.id, n]))

  props.edges.forEach((e) => {
    if (e.source === nodeId) {
      const target = nodeMap.get(e.target)
      if (target) result.push({ node: target, relation: e.relation, direction: 'out' })
    } else if (e.target === nodeId) {
      const source = nodeMap.get(e.source)
      if (source) result.push({ node: source, relation: e.relation, direction: 'in' })
    }
  })
  return result
})

const degree = computed(() => (props.node ? neighbors.value.length : null))
</script>

<style scoped>
.drawer-header {
  background: linear-gradient(
    135deg,
    rgba(var(--v-theme-primary), 0.08),
    rgba(var(--v-theme-surface), 0)
  );
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.06);
}

.drawer-header :deep(.text-h6) {
  font-weight: 700;
  word-break: break-word;
}

.gap-2 {
  gap: 8px;
}

/* 操作按钮区粘性置顶 */
.drawer-actions {
  position: sticky;
  top: 0;
  z-index: 2;
  background: rgb(var(--v-theme-surface));
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.06);
}

.node-content {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.7;
  color: rgba(var(--v-theme-on-surface), 0.85);
  background: rgba(var(--v-theme-on-surface), 0.03);
  padding: 10px 12px;
  border-radius: 8px;
  border-left: 3px solid rgba(var(--v-theme-primary), 0.4);
}

.prop-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.04);
  gap: 12px;
}

.prop-row:last-child {
  border-bottom: none;
}

.prop-row span:first-child {
  flex-shrink: 0;
  font-weight: 500;
}

.prop-row span:last-child {
  text-align: right;
  word-break: break-all;
}
</style>
