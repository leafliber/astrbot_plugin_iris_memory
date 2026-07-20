<template>
  <div class="l3-node-list">
    <v-card color="surface" variant="flat" class="iris-card h-100 d-flex flex-column">
      <!-- 工具栏 -->
      <v-card-title class="py-2 px-3 d-flex align-center iris-section-title">
        <v-icon icon="mdi-file-tree" size="small" class="mr-2" />
        <span class="text-subtitle-1">节点列表</span>
        <v-spacer />
        <v-text-field
          v-model="localFilter"
          placeholder="过滤…"
          prepend-inner-icon="mdi-magnify"
          variant="outlined"
          density="compact"
          hide-details
          clearable
          class="filter-input"
        />
        <v-btn
          v-if="selected.length > 0"
          color="error"
          variant="tonal"
          size="small"
          class="ml-2"
          @click="emit('bulk-delete', selected)"
        >
          <v-icon icon="mdi-delete" class="mr-1" />
          删除 ({{ selected.length }})
        </v-btn>
      </v-card-title>

      <v-divider />

      <!-- 表格 -->
      <v-data-table
        :headers="headers"
        :items="filteredItems"
        :items-per-page="10"
        :loading="loading"
        item-value="id"
        show-select
        v-model="selected"
        density="compact"
        hover
        class="flex-grow-1 iris-table"
        @click:row="handleRowClick"
      >
        <template #item.label="{ item }">
          <v-icon :icon="getNodeIcon(item.label)" :color="getTypeColor(item.label)" size="small" class="mr-1" />
          <span class="text-body-2">{{ getNodeLabel(item.label) }}</span>
        </template>

        <template #item.name="{ item }">
          <div class="text-body-2 font-weight-medium">{{ item.name || item.id }}</div>
          <div v-if="item.content" class="text-caption text-medium-emphasis text-truncate" style="max-width: 240px">
            {{ item.content }}
          </div>
        </template>

        <template #item.confidence="{ item }">
          <v-progress-linear
            :model-value="item.confidence * 100"
            :color="getConfidenceColor(item.confidence)"
            height="6"
            rounded
            class="my-2"
            style="width: 60px"
          />
          <span class="text-caption ml-1">{{ (item.confidence * 100).toFixed(0) }}%</span>
        </template>

        <template #item.access_count="{ item }">
          <v-chip size="x-small" variant="tonal">{{ item.access_count ?? 0 }}</v-chip>
        </template>

        <template #item.group_id="{ item }">
          <v-chip v-if="item.group_id" size="x-small" variant="tonal" color="info">{{ item.group_id }}</v-chip>
          <span v-else class="text-disabled">—</span>
        </template>

        <template #item.created_time="{ item }">
          <span class="text-caption">{{ formatTime(item.created_time) }}</span>
        </template>

        <template #item.actions="{ item }">
          <v-btn
            icon="mdi-arrow-expand"
            size="x-small"
            variant="text"
            @click.stop="emit('expand', item.id)"
          >
            <v-tooltip activator="parent" location="bottom">以此节点展开</v-tooltip>
          </v-btn>
          <v-btn
            icon="mdi-delete"
            size="x-small"
            variant="text"
            color="error"
            @click.stop="emit('delete', item.id)"
          >
            <v-tooltip activator="parent" location="bottom">删除节点</v-tooltip>
          </v-btn>
        </template>

        <template #no-data>
          <div class="iris-empty-state">
            <v-icon icon="mdi-database-off" size="56" />
            <div class="iris-empty-state__title">暂无节点数据</div>
          </div>
        </template>
      </v-data-table>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { L3NodeDetail } from '@/types'
import {
  getNodeIcon,
  getTypeColor,
  getNodeLabel,
  getConfidenceColor,
  formatTime,
} from '@/composables/l3Constants'

const props = defineProps<{
  nodes: L3NodeDetail[]
  loading: boolean
}>()

const emit = defineEmits<{
  'focus-node': [nodeId: string]
  expand: [nodeId: string]
  delete: [nodeId: string]
  'bulk-delete': [ids: string[]]
}>()

const localFilter = ref('')
const selected = ref<string[]>([])

const handleRowClick = (_: unknown, row: { item: L3NodeDetail }) => {
  emit('focus-node', row.item.id)
}

const headers = [
  { title: '类型', key: 'label', width: '120px', sortable: true },
  { title: '名称 / 内容', key: 'name', sortable: true },
  { title: '置信度', key: 'confidence', width: '140px', sortable: true },
  { title: '访问', key: 'access_count', width: '80px', sortable: true },
  { title: '群组', key: 'group_id', width: '100px', sortable: true },
  { title: '创建时间', key: 'created_time', width: '150px', sortable: true },
  { title: '操作', key: 'actions', width: '90px', sortable: false },
]

const filteredItems = computed(() => {
  const kw = localFilter.value?.trim().toLowerCase()
  if (!kw) return props.nodes
  return props.nodes.filter((n) =>
    [n.name, n.id, n.content, n.label]
      .filter(Boolean)
      .some((v) => String(v).toLowerCase().includes(kw))
  )
})
</script>

<style scoped>
.l3-node-list {
  height: 100%;
}

.filter-input {
  max-width: 200px;
}
</style>
