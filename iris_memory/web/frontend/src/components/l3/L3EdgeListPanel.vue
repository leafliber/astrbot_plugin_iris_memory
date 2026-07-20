<template>
  <div class="l3-edge-list">
    <v-card color="surface" variant="flat" class="iris-card h-100 d-flex flex-column">
      <v-card-title class="py-2 px-3 d-flex align-center iris-section-title">
        <v-icon icon="mdi-link-variant" size="small" class="mr-2" />
        <span class="text-subtitle-1">关系列表</span>
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

      <v-data-table
        :headers="headers"
        :items="filteredItems"
        :items-per-page="10"
        :loading="loading"
        :item-value="(item: L3EdgeDetail) => `${item.source.id}->${item.target.id}->${item.relation}`"
        show-select
        v-model="selected"
        density="compact"
        hover
        class="flex-grow-1 iris-table"
        @click:row="handleRowClick"
      >
        <template #item.relation="{ item }">
          <v-chip size="x-small" variant="outlined" color="secondary">
            {{ getRelationLabel(item.relation) }}
          </v-chip>
        </template>

        <template #item.source="{ item }">
          <div class="d-flex align-center">
            <v-icon :icon="getNodeIcon(item.source.label)" :color="getTypeColor(item.source.label)" size="x-small" class="mr-1" />
            <span class="text-body-2">{{ item.source.name || item.source.id }}</span>
          </div>
        </template>

        <template #item.target="{ item }">
          <div class="d-flex align-center">
            <v-icon :icon="getNodeIcon(item.target.label)" :color="getTypeColor(item.target.label)" size="x-small" class="mr-1" />
            <span class="text-body-2">{{ item.target.name || item.target.id }}</span>
          </div>
        </template>

        <template #item.weight="{ item }">
          <v-progress-linear
            :model-value="Math.min((item.weight ?? 1) * 50, 100)"
            color="secondary"
            height="6"
            rounded
            class="my-2"
            style="width: 60px"
          />
          <span class="text-caption ml-1">{{ (item.weight ?? 1).toFixed(2) }}</span>
        </template>

        <template #item.confidence="{ item }">
          <v-chip size="x-small" :color="getConfidenceColor(item.confidence ?? 1)" variant="tonal">
            {{ ((item.confidence ?? 1) * 100).toFixed(0) }}%
          </v-chip>
        </template>

        <template #item.created_time="{ item }">
          <span class="text-caption">{{ formatTime(item.created_time) }}</span>
        </template>

        <template #item.actions="{ item }">
          <v-btn
            icon="mdi-delete"
            size="x-small"
            variant="text"
            color="error"
            @click.stop="emit('delete', item)"
          >
            <v-tooltip activator="parent" location="bottom">删除关系</v-tooltip>
          </v-btn>
        </template>

        <template #no-data>
          <div class="iris-empty-state">
            <v-icon icon="mdi-link-off" size="56" />
            <div class="iris-empty-state__title">暂无关系数据</div>
          </div>
        </template>
      </v-data-table>
    </v-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { L3EdgeDetail } from '@/types'
import {
  getNodeIcon,
  getTypeColor,
  getRelationLabel,
  getConfidenceColor,
  formatTime,
} from '@/composables/l3Constants'

const props = defineProps<{
  edges: L3EdgeDetail[]
  loading: boolean
}>()

const emit = defineEmits<{
  'focus-edge': [edge: L3EdgeDetail]
  delete: [edge: L3EdgeDetail]
  'bulk-delete': [edges: L3EdgeDetail[]]
}>()

const localFilter = ref('')
const selected = ref<L3EdgeDetail[]>([])

const handleRowClick = (_: unknown, row: { item: L3EdgeDetail }) => {
  emit('focus-edge', row.item)
}

const headers = [
  { title: '关系', key: 'relation', width: '120px', sortable: true },
  { title: '源节点', key: 'source', width: '180px', sortable: true },
  { title: '目标节点', key: 'target', width: '180px', sortable: true },
  { title: '权重', key: 'weight', width: '140px', sortable: true },
  { title: '置信度', key: 'confidence', width: '100px', sortable: true },
  { title: '创建时间', key: 'created_time', width: '150px', sortable: true },
  { title: '操作', key: 'actions', width: '70px', sortable: false },
]

const filteredItems = computed(() => {
  const kw = localFilter.value?.trim().toLowerCase()
  if (!kw) return props.edges
  return props.edges.filter((e) => {
    return [
      e.relation,
      e.source.name,
      e.source.id,
      e.target.name,
      e.target.id,
      getRelationLabel(e.relation),
    ].some((v) => String(v).toLowerCase().includes(kw))
  })
})
</script>

<style scoped>
.l3-edge-list {
  height: 100%;
}

.filter-input {
  max-width: 200px;
}
</style>
