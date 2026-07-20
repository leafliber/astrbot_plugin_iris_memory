<template>
  <div class="l2-memory-view">
    <ComponentDisabled
      :status="status"
      :error="error"
      :error-type="errorType"
      component-name="L2 记忆"
      @retry="refreshState"
    >
      <v-row>
        <v-col cols="12">
          <v-card color="surface" variant="flat" class="iris-card">
            <v-card-text class="d-flex align-center flex-wrap ga-2 py-2">
              <div class="d-flex align-center">
                <v-icon icon="mdi-database" color="primary" class="mr-2" />
                <span class="text-h6">L2 记忆库</span>
              </div>
              <v-spacer />
              <IsolationBadge
                type="memory"
                :enabled="isolationStatus.enable_group_memory_isolation"
              />
            </v-card-text>
            <v-divider />
            <v-tabs v-model="activeTab" color="primary" grow>
              <v-tab value="latest">
                <v-icon icon="mdi-clock-outline" class="mr-2" />
                最新记忆
              </v-tab>
              <v-tab value="search">
                <v-icon icon="mdi-magnify" class="mr-2" />
                记忆搜索
              </v-tab>
            </v-tabs>
          </v-card>
        </v-col>
      </v-row>

      <v-window v-model="activeTab" class="mt-4">
        <v-window-item value="latest">
          <v-row>
            <v-col cols="12">
              <v-card color="surface" variant="flat" class="iris-card">
                <v-card-title class="d-flex align-center flex-wrap ga-2 iris-section-title">
                  <v-icon icon="mdi-clock-outline" color="secondary" class="mr-2" />
                  最新记忆
                  <v-spacer />
                  <v-select
                    v-model="latestGroupIdFilter"
                    :items="groupOptions"
                    item-title="title"
                    item-value="value"
                    placeholder="全部群聊"
                    prepend-inner-icon="mdi-account-group"
                    variant="outlined"
                    density="compact"
                    hide-details
                    clearable
                    style="max-width: 160px"
                    @update:model-value="handleLatestGroupChange"
                  />
                  <v-select
                    v-model="selectedSortBy"
                    :items="sortByOptions"
                    label="排序字段"
                    variant="outlined"
                    density="compact"
                    hide-details
                    style="max-width: 150px"
                    @update:model-value="handleSortChange"
                  />
                  <v-btn
                    :icon="memoryStore.l2LatestSortOrder === 'desc' ? 'mdi-arrow-down' : 'mdi-arrow-up'"
                    variant="outlined"
                    density="compact"
                    size="small"
                    @click="toggleSortOrder"
                  />
                  <v-select
                    v-model="selectedLimit"
                    :items="limitOptions"
                    label="显示数量"
                    variant="outlined"
                    density="compact"
                    hide-details
                    style="max-width: 120px"
                    @update:model-value="handleLimitChange"
                  />
                </v-card-title>
                <v-card-text>
                  <v-progress-linear
                    v-if="memoryStore.l2LatestLoading"
                    indeterminate
                    color="primary"
                  />

                  <div v-else-if="memoryStore.l2LatestResults.length > 0">
                    <div class="d-flex align-center mb-2">
                      <v-checkbox
                        v-model="selectAllLatest"
                        label="全选"
                        density="compact"
                        hide-details
                        class="mr-2"
                        @update:model-value="toggleSelectAllLatest"
                      />
                      <v-spacer />
                      <v-btn
                        v-if="selectedLatestIds.length > 0"
                        color="error"
                        variant="tonal"
                        size="small"
                        :loading="deletingL2"
                        @click="handleDeleteSelectedLatest"
                      >
                        <v-icon icon="mdi-delete" class="mr-1" />
                        删除选中 ({{ selectedLatestIds.length }})
                      </v-btn>
                    </div>
                    <v-card
                      v-for="(result, index) in memoryStore.l2LatestResults"
                      :key="result.id || index"
                      variant="outlined"
                      class="mb-3 iris-card iris-card-hover"
                    >
                      <v-card-text>
                        <div class="d-flex align-start">
                          <v-checkbox
                            :model-value="selectedLatestIds.includes(result.id)"
                            density="compact"
                            hide-details
                            class="mr-2 mt-0"
                            @update:model-value="toggleSelectLatest(result.id)"
                          />
                          <v-chip
                            color="secondary"
                            size="small"
                            class="mr-3 mt-1"
                          >
                            #{{ memoryStore.l2LatestOffset + index + 1 }}
                          </v-chip>
                          <div class="flex-grow-1">
                            <div class="text-body-1 text-wrap">{{ result.content }}</div>
                            <div class="d-flex flex-wrap align-center mt-2 text-caption text-medium-emphasis ga-3">
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-clock-outline" size="small" class="mr-1" />
                                {{ formatTime(result.timestamp) }}
                              </span>
                              <span v-if="result.group_id" class="d-flex align-center">
                                <v-icon icon="mdi-account-group" size="small" class="mr-1" />
                                {{ result.group_id }}
                              </span>
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-eye-outline" size="small" class="mr-1" />
                                {{ result.access_count ?? 0 }} 次访问
                              </span>
                              <span v-if="result.last_access_time" class="d-flex align-center">
                                <v-icon icon="mdi-clock-check-outline" size="small" class="mr-1" />
                                最近访问 {{ formatTime(result.last_access_time) }}
                              </span>
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-star-outline" size="small" class="mr-1" />
                                置信度 {{ ((result.confidence ?? 0.5) * 100).toFixed(0) }}%
                              </span>
                              <v-chip
                                v-if="result.source"
                                size="x-small"
                                variant="tonal"
                                :color="getSourceColor(result.source)"
                              >
                                {{ getSourceLabel(result.source) }}
                              </v-chip>
                            </div>
                          </div>
                          <div class="d-flex ml-2">
                            <v-btn
                              icon="mdi-pencil"
                              variant="text"
                              size="small"
                              @click="openEditDialog(result)"
                            />
                            <v-btn
                              icon="mdi-delete"
                              variant="text"
                              size="small"
                              color="error"
                              @click="handleDeleteSingle(result.id)"
                            />
                          </div>
                        </div>
                      </v-card-text>
                    </v-card>
                  </div>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-database-outline" size="64" />
                    <div class="iris-empty-state__title">暂无记忆数据</div>
                    <div class="iris-empty-state__desc">
                      L2 记忆库为空或数据加载失败
                    </div>
                  </div>

                  <div
                    v-if="l2LatestTotalPages > 1"
                    class="d-flex flex-column align-center mt-4"
                  >
                    <v-pagination
                      :model-value="l2LatestCurrentPage"
                      :length="l2LatestTotalPages"
                      :total-visible="7"
                      density="comfortable"
                      @update:model-value="handlePageChange"
                    />
                    <div class="text-caption text-medium-emphasis mt-1">
                      共 {{ memoryStore.l2LatestTotalCount }} 条记忆，
                      第 {{ l2LatestCurrentPage }} / {{ l2LatestTotalPages }} 页
                    </div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>

        <v-window-item value="search">
          <v-row>
            <v-col cols="12">
              <v-card color="surface" variant="flat" class="iris-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <v-icon icon="mdi-database-search" color="secondary" class="mr-2" />
                  L2 记忆搜索
                </v-card-title>
                <v-card-text>
                  <v-row>
                    <v-col cols="12" md="6">
                      <v-text-field
                        v-model="searchQuery"
                        placeholder="输入关键词搜索记忆..."
                        prepend-inner-icon="mdi-magnify"
                        variant="outlined"
                        density="comfortable"
                        hide-details
                        clearable
                        @keyup.enter="handleSearch"
                      />
                    </v-col>
                    <v-col cols="12" md="3">
                      <v-select
                        v-model="groupIdFilter"
                        :items="groupOptions"
                        item-title="title"
                        item-value="value"
                        placeholder="群聊（可选）"
                        prepend-inner-icon="mdi-account-group"
                        variant="outlined"
                        density="comfortable"
                        hide-details
                        clearable
                      />
                    </v-col>
                    <v-col cols="12" md="3">
                      <v-btn
                        color="primary"
                        size="large"
                        block
                        :loading="memoryStore.l2Loading"
                        @click="handleSearch"
                      >
                        <v-icon icon="mdi-magnify" class="mr-1" />
                        搜索
                      </v-btn>
                    </v-col>
                  </v-row>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>

          <v-row class="mt-4">
            <v-col cols="12">
              <v-card color="surface" variant="flat" class="iris-card">
                <v-card-title class="d-flex align-center iris-section-title">
                  <span>搜索结果</span>
                  <v-spacer />
                  <v-chip v-if="memoryStore.l2Results.length > 0" size="small" color="secondary" variant="tonal">
                    {{ memoryStore.l2Results.length }} 条结果
                  </v-chip>
                </v-card-title>
                <v-card-text>
                  <v-progress-linear
                    v-if="memoryStore.l2Loading"
                    indeterminate
                    color="primary"
                  />

                  <div v-else-if="memoryStore.l2Results.length > 0">
                    <div class="d-flex align-center mb-2">
                      <v-checkbox
                        v-model="selectAllSearch"
                        label="全选"
                        density="compact"
                        hide-details
                        class="mr-2"
                        @update:model-value="toggleSelectAllSearch"
                      />
                      <v-spacer />
                      <v-btn
                        v-if="selectedSearchIds.length > 0"
                        color="error"
                        variant="tonal"
                        size="small"
                        :loading="deletingL2"
                        @click="handleDeleteSelectedSearch"
                      >
                        <v-icon icon="mdi-delete" class="mr-1" />
                        删除选中 ({{ selectedSearchIds.length }})
                      </v-btn>
                    </div>
                    <v-card
                      v-for="(result, index) in memoryStore.l2Results"
                      :key="result.id || index"
                      variant="outlined"
                      class="mb-3 iris-card iris-card-hover"
                    >
                      <v-card-text>
                        <div class="d-flex align-start">
                          <v-checkbox
                            :model-value="selectedSearchIds.includes(result.id)"
                            density="compact"
                            hide-details
                            class="mr-2 mt-0"
                            @update:model-value="toggleSelectSearch(result.id)"
                          />
                          <v-chip
                            :color="getScoreColor(result.score)"
                            size="small"
                            class="mr-3 mt-1"
                          >
                            {{ (result.score * 100).toFixed(0) }}%
                          </v-chip>
                          <div class="flex-grow-1">
                            <div class="text-body-1 text-wrap">{{ result.content }}</div>
                            <div class="d-flex flex-wrap align-center mt-2 text-caption text-medium-emphasis ga-3">
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-clock-outline" size="small" class="mr-1" />
                                {{ formatTime(result.timestamp) }}
                              </span>
                              <span v-if="result.group_id" class="d-flex align-center">
                                <v-icon icon="mdi-account-group" size="small" class="mr-1" />
                                {{ result.group_id }}
                              </span>
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-eye-outline" size="small" class="mr-1" />
                                {{ result.access_count ?? 0 }} 次访问
                              </span>
                              <span v-if="result.last_access_time" class="d-flex align-center">
                                <v-icon icon="mdi-clock-check-outline" size="small" class="mr-1" />
                                最近访问 {{ formatTime(result.last_access_time) }}
                              </span>
                              <span class="d-flex align-center">
                                <v-icon icon="mdi-star-outline" size="small" class="mr-1" />
                                置信度 {{ ((result.confidence ?? 0.5) * 100).toFixed(0) }}%
                              </span>
                              <v-chip
                                v-if="result.source"
                                size="x-small"
                                variant="tonal"
                                :color="getSourceColor(result.source)"
                              >
                                {{ getSourceLabel(result.source) }}
                              </v-chip>
                            </div>
                          </div>
                          <div class="d-flex ml-2">
                            <v-btn
                              icon="mdi-pencil"
                              variant="text"
                              size="small"
                              @click="openEditDialog(result)"
                            />
                            <v-btn
                              icon="mdi-delete"
                              variant="text"
                              size="small"
                              color="error"
                              @click="handleDeleteSingle(result.id)"
                            />
                          </div>
                        </div>
                      </v-card-text>
                    </v-card>
                  </div>

                  <div v-else class="iris-empty-state">
                    <v-icon icon="mdi-database-search-outline" size="64" />
                    <div class="iris-empty-state__title">
                      {{ memoryStore.l2Query ? '未找到相关记忆' : '输入关键词搜索记忆' }}
                    </div>
                    <div class="iris-empty-state__desc">
                      {{ memoryStore.l2Query ? '尝试使用其他关键词' : 'L2 记忆支持语义检索' }}
                    </div>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-window-item>
      </v-window>

      <v-row class="mt-4">
        <v-col cols="12">
          <v-card color="surface" variant="flat" class="iris-card">
            <v-card-title class="iris-section-title">
              <v-icon icon="mdi-information" class="mr-2" />
              L2 记忆说明
            </v-card-title>
            <v-card-text>
              <v-alert type="info" variant="tonal" density="compact">
                <div class="text-body-2">
                  <strong>L2 记忆（Episodic Memory）</strong> 是长期记忆存储，基于 RIF 评分动态管理，支持选择性遗忘。
                  使用向量检索技术，可以语义相似度搜索历史对话内容。
                </div>
              </v-alert>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </ComponentDisabled>

    <v-dialog v-model="editDialog" max-width="600" class="iris-dialog">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-pencil" class="mr-2" />
          编辑记忆
        </v-card-title>
        <v-card-text>
          <v-textarea
            v-model="editContent"
            variant="outlined"
            rows="5"
            label="记忆内容"
            hide-details
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="editDialog = false">取消</v-btn>
          <v-btn color="primary" variant="tonal" :loading="updatingL2" @click="handleUpdateEntry">
            保存
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-dialog v-model="deleteDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-alert-circle" color="warning" class="mr-2" />
          确认删除
        </v-card-title>
        <v-card-text>
          确定要删除{{ deleteTargetIds.length > 1 ? ` ${deleteTargetIds.length} 条` : '该' }}记忆吗？此操作不可撤销。
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteDialog = false">取消</v-btn>
          <v-btn color="error" variant="tonal" :loading="deletingL2" @click="confirmDelete">
            确认删除
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useMemoryStore } from '@/stores'
import { useComponentState } from '@/composables/useComponentState'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import IsolationBadge from '@/components/IsolationBadge.vue'
import { getGroupList } from '@/api/profile'
import { useIsolationStatus } from '@/composables/useIsolationStatus'
import type { L2Memory, L2SortField, L2SortOrder } from '@/types'

const memoryStore = useMemoryStore()
const { status, error, errorType, refreshState } = useComponentState('l2_memory')
const { status: isolationStatus } = useIsolationStatus()

const activeTab = ref('latest')
const searchQuery = ref('')
const groupIdFilter = ref<string | null>(null)
const latestGroupIdFilter = ref<string | null>(null)
const selectedLimit = ref(20)
const selectedSortBy = ref<L2SortField>('timestamp')

// 群聊下拉选项
const groups = ref<{ group_id: string; group_name?: string }[]>([])
const groupOptions = computed(() =>
  groups.value.map((g) => ({
    title: g.group_name || g.group_id,
    value: g.group_id,
  }))
)

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

const selectedLatestIds = ref<string[]>([])
const selectedSearchIds = ref<string[]>([])
const selectAllLatest = ref(false)
const selectAllSearch = ref(false)

const deletingL2 = ref(false)
const deleteDialog = ref(false)
const deleteTargetIds = ref<string[]>([])

const editDialog = ref(false)
const editId = ref('')
const editContent = ref('')
const updatingL2 = ref(false)

const limitOptions = [
  { title: '10 条', value: 10 },
  { title: '20 条', value: 20 },
  { title: '50 条', value: 50 },
  { title: '100 条', value: 100 }
]

const sortByOptions = [
  { title: '创建时间', value: 'timestamp' },
  { title: '访问次数', value: 'access_count' },
  { title: '置信度', value: 'confidence' },
  { title: '最近访问', value: 'last_access_time' }
]

const l2LatestCurrentPage = computed(() => memoryStore.getL2LatestCurrentPage())
const l2LatestTotalPages = computed(() => memoryStore.getL2LatestTotalPages())

// 统一拉取最新记忆：附加当前群聊过滤
const reloadLatest = (limit?: number) =>
  memoryStore.fetchLatestL2Memories(limit, latestGroupIdFilter.value || undefined)

const handleLatestGroupChange = () => {
  memoryStore.setL2LatestPage(1)
  reloadLatest()
}

const handlePageChange = (page: number) => {
  memoryStore.setL2LatestPage(page)
  reloadLatest()
}

const handleLimitChange = (value: number) => {
  memoryStore.setL2LatestLimit(value)
  memoryStore.setL2LatestPage(1)
  reloadLatest(value)
}

const handleSortChange = (value: L2SortField) => {
  memoryStore.setL2LatestSort(value, memoryStore.l2LatestSortOrder)
  memoryStore.setL2LatestPage(1)
  reloadLatest()
}

const toggleSortOrder = () => {
  const newOrder: L2SortOrder = memoryStore.l2LatestSortOrder === 'desc' ? 'asc' : 'desc'
  memoryStore.setL2LatestSort(memoryStore.l2LatestSortBy, newOrder)
  memoryStore.setL2LatestPage(1)
  reloadLatest()
}

const handleSearch = () => {
  if (searchQuery.value.trim()) {
    memoryStore.searchL2Memory(
      searchQuery.value.trim(),
      groupIdFilter.value || undefined
    )
  }
}

const getScoreColor = (score: number): string => {
  if (score >= 0.9) return 'success'
  if (score >= 0.7) return 'info'
  if (score >= 0.5) return 'warning'
  return 'error'
}

const getSourceColor = (source?: string): string => {
  if (source === 'summary') return 'info'
  if (source === 'tool') return 'success'
  return 'secondary'
}

const getSourceLabel = (source?: string): string => {
  if (source === 'summary') return '总结'
  if (source === 'tool') return '工具'
  return source || '未知'
}

const formatTime = (timestamp?: string): string => {
  if (!timestamp) return '未知时间'
  try {
    const date = new Date(timestamp)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return timestamp
  }
}

const handleRefresh = () => {
  if (activeTab.value === 'latest') {
    reloadLatest()
  } else if (searchQuery.value.trim()) {
    handleSearch()
  }
}

const toggleSelectLatest = (id: string) => {
  const idx = selectedLatestIds.value.indexOf(id)
  if (idx >= 0) {
    selectedLatestIds.value.splice(idx, 1)
  } else {
    selectedLatestIds.value.push(id)
  }
}

const toggleSelectAllLatest = (checked: boolean | null) => {
  if (checked) {
    selectedLatestIds.value = memoryStore.l2LatestResults.map(r => r.id)
  } else {
    selectedLatestIds.value = []
  }
}

const toggleSelectSearch = (id: string) => {
  const idx = selectedSearchIds.value.indexOf(id)
  if (idx >= 0) {
    selectedSearchIds.value.splice(idx, 1)
  } else {
    selectedSearchIds.value.push(id)
  }
}

const toggleSelectAllSearch = (checked: boolean | null) => {
  if (checked) {
    selectedSearchIds.value = memoryStore.l2Results.map(r => r.id)
  } else {
    selectedSearchIds.value = []
  }
}

const handleDeleteSingle = (id: string) => {
  deleteTargetIds.value = [id]
  deleteDialog.value = true
}

const handleDeleteSelectedLatest = () => {
  deleteTargetIds.value = [...selectedLatestIds.value]
  deleteDialog.value = true
}

const handleDeleteSelectedSearch = () => {
  deleteTargetIds.value = [...selectedSearchIds.value]
  deleteDialog.value = true
}

const confirmDelete = async () => {
  deletingL2.value = true
  try {
    await memoryStore.deleteL2Entries(deleteTargetIds.value)
    selectedLatestIds.value = selectedLatestIds.value.filter(id => !deleteTargetIds.value.includes(id))
    selectedSearchIds.value = selectedSearchIds.value.filter(id => !deleteTargetIds.value.includes(id))
    deleteDialog.value = false
    deleteTargetIds.value = []
    // 刷新以更新总数和分页
    await reloadLatest()
  } catch (error) {
    console.error('删除记忆失败:', error)
  } finally {
    deletingL2.value = false
  }
}

const openEditDialog = (item: L2Memory) => {
  editId.value = item.id
  editContent.value = item.content
  editDialog.value = true
}

const handleUpdateEntry = async () => {
  if (!editId.value || !editContent.value.trim()) return
  updatingL2.value = true
  try {
    await memoryStore.updateL2Entry(editId.value, editContent.value.trim())
    editDialog.value = false
  } catch (error) {
    console.error('更新记忆失败:', error)
  } finally {
    updatingL2.value = false
  }
}

watch(activeTab, (newTab) => {
  if (newTab === 'latest' && memoryStore.l2LatestResults.length === 0) {
    reloadLatest()
  }
})

watch(() => memoryStore.l2LatestSortBy, (val) => {
  selectedSortBy.value = val
})

onMounted(() => {
  window.addEventListener('iris:refresh', handleRefresh)
  selectedSortBy.value = memoryStore.l2LatestSortBy
  fetchGroups()
  if (activeTab.value === 'latest') {
    reloadLatest()
  }
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>
