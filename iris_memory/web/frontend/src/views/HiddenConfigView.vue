<template>
  <div class="hidden-config-view">
    <v-card color="surface" variant="flat" class="iris-hero-card mb-3">
      <v-card-title class="d-flex align-center iris-section-title">
        <v-icon icon="mdi-cog-outline" color="primary" class="mr-2" />
        隐藏参数配置
        <v-spacer />
        <v-btn
          color="primary"
          variant="tonal"
          size="small"
          prepend-icon="mdi-refresh"
          :loading="loading"
          @click="loadData"
        >
          刷新
        </v-btn>
      </v-card-title>
      <v-card-text class="pt-0">
        <v-alert type="warning" variant="tonal" density="compact">
          <div class="text-body-2">
            这些参数控制内部行为，修改后需点击底部「保存修改」才会生效并持久化。不当修改可能导致功能异常，请谨慎操作。
          </div>
        </v-alert>
      </v-card-text>
    </v-card>

    <v-progress-circular v-if="loading" indeterminate color="primary" size="48" class="d-block mx-auto my-8" />

    <template v-else>
      <v-expansion-panels v-model="expandedGroups" multiple variant="accordion" class="mb-16">
        <v-expansion-panel
          v-for="group in groupsWithItems"
          :key="group.name"
          :value="group.name"
          class="bg-surface"
        >
          <v-expansion-panel-title class="py-2">
            <div class="d-flex align-center ga-2">
              <v-icon :icon="getGroupIcon(group.name)" color="primary" size="20" />
              <span class="text-subtitle-1 font-weight-medium">{{ group.name }}</span>
              <v-chip size="x-small" variant="tonal" color="primary">
                {{ group.items.length }}
              </v-chip>
              <v-chip v-if="getGroupChangedCount(group.name) > 0" size="x-small" variant="tonal" color="warning">
                {{ getGroupChangedCount(group.name) }} 项已修改
              </v-chip>
            </div>
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <div class="config-grid">
              <div
                v-for="item in group.items"
                :key="item.key"
                class="config-item"
                :class="{ 'config-item--changed': isChanged(item.key) }"
              >
                <div class="config-item__header">
                  <div class="d-flex align-center ga-2">
                    <code class="text-body-2 font-weight-medium">{{ item.key }}</code>
                    <v-chip v-if="isChanged(item.key)" size="x-small" variant="tonal" color="warning" label>
                      已修改
                    </v-chip>
                  </div>
                  <div class="d-flex align-center ga-1">
                    <v-tooltip v-if="isChanged(item.key) || isServerChanged(item)" text="恢复默认值" location="top">
                      <template #activator="{ props }">
                        <v-btn
                          icon="mdi-undo-variant"
                          variant="text"
                          size="x-small"
                          color="warning"
                          v-bind="props"
                          @click="handleResetItem(item)"
                        />
                      </template>
                    </v-tooltip>
                    <v-tooltip text="默认值" location="top">
                      <template #activator="{ props }">
                        <span class="text-caption text-medium-emphasis" v-bind="props">
                          默认: {{ formatDefaultValue(item) }}
                        </span>
                      </template>
                    </v-tooltip>
                  </div>
                </div>
                <p v-if="item.description" class="config-item__desc text-body-2 text-medium-emphasis mb-2">
                  {{ item.description }}
                </p>
                <div class="config-item__input">
                  <v-select
                    v-if="item.type === 'literal' && item.options.length > 0"
                    :model-value="getCurrentValue(item) as string | null | undefined"
                    :items="item.options"
                    density="compact"
                    variant="outlined"
                    hide-details
                    @update:model-value="(v: unknown) => onValueChange(item, v)"
                  />
                  <v-switch
                    v-else-if="item.type === 'bool'"
                    :model-value="getCurrentValue(item)"
                    density="compact"
                    color="primary"
                    hide-details
                    :label="getCurrentValue(item) ? '启用' : '禁用'"
                    @update:model-value="(v: unknown) => onValueChange(item, v)"
                  />
                  <v-text-field
                    v-else
                    :model-value="getCurrentValue(item)"
                    :type="item.type === 'int' || item.type === 'float' ? 'number' : 'text'"
                    density="compact"
                    variant="outlined"
                    hide-details
                    :placeholder="`默认: ${formatDefaultValue(item)}`"
                    @update:model-value="(v: unknown) => onValueChange(item, parseInputValue(item, String(v ?? '')))"
                  />
                </div>
              </div>
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
    </template>

    <div class="sticky-bottom-bar" :class="{ 'sticky-bottom-bar--elevated': hasChanges }">
      <v-card color="surface" :variant="hasChanges ? 'elevated' : 'flat'" class="pa-3">
        <div class="d-flex align-center ga-3 flex-wrap">
          <v-btn
            color="primary"
            prepend-icon="mdi-content-save"
            :loading="saving"
            :disabled="!hasChanges"
            @click="handleSaveAll"
          >
            保存修改
            <v-chip v-if="hasChanges" size="small" variant="tonal" color="on-primary" class="ml-2">
              {{ changedKeys.length }}
            </v-chip>
          </v-btn>
          <v-btn
            color="grey"
            variant="tonal"
            prepend-icon="mdi-undo"
            :disabled="!hasChanges"
            @click="handleDiscardChanges"
          >
            放弃修改
          </v-btn>
          <v-spacer />
          <v-btn
            color="error"
            variant="tonal"
            prepend-icon="mdi-restore"
            size="small"
            :loading="resetting"
            @click="handleResetAll"
          >
            全部重置为默认值
          </v-btn>
        </div>
      </v-card>
    </div>

    <v-dialog v-model="confirmDialog" max-width="400" class="iris-dialog">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-alert" color="warning" class="mr-2" />
          确认操作
        </v-card-title>
        <v-card-text>{{ confirmMessage }}</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="confirmDialog = false">取消</v-btn>
          <v-btn color="error" variant="tonal" @click="confirmAction">确认</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-snackbar
      v-model="showSnackbar"
      :color="snackbarColor"
      :timeout="4000"
      location="top"
    >
      {{ snackbarText }}
    </v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import {
  getHiddenConfig,
  updateHiddenConfig,
  deleteHiddenConfig,
  resetHiddenConfig,
  type HiddenConfigItem,
  type HiddenConfigGroup
} from '@/api/hiddenConfig'

const loading = ref(false)
const saving = ref(false)
const resetting = ref(false)

const items = ref<HiddenConfigItem[]>([])
const groups = ref<HiddenConfigGroup[]>([])

const pendingChanges = ref<Record<string, unknown>>({})

const expandedGroups = ref<string[]>([])

const confirmDialog = ref(false)
const confirmMessage = ref('')
const confirmCallback = ref<(() => void) | null>(null)

const showSnackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const changedKeys = computed(() => Object.keys(pendingChanges.value))

const hasChanges = computed(() => changedKeys.value.length > 0)

const groupsWithItems = computed(() => {
  return groups.value.map(group => ({
    ...group,
    items: items.value.filter(item => item.group === group.name)
  })).filter(group => group.items.length > 0)
})

const isChanged = (key: string): boolean => {
  return key in pendingChanges.value
}

const isServerChanged = (item: HiddenConfigItem): boolean => {
  return item.value !== item.default
}

const getCurrentValue = (item: HiddenConfigItem): unknown => {
  if (item.key in pendingChanges.value) {
    return pendingChanges.value[item.key]
  }
  return item.value
}

const getGroupChangedCount = (groupName: string): number => {
  const groupItems = items.value.filter(item => item.group === groupName)
  return groupItems.filter(item => isChanged(item.key)).length
}

const formatDefaultValue = (item: HiddenConfigItem): string => {
  const val = item.default
  if (typeof val === 'boolean') return val ? 'true' : 'false'
  if (val === null || val === undefined) return '(空)'
  return String(val)
}

const parseInputValue = (item: HiddenConfigItem, raw: string): unknown => {
  if (item.type === 'int') {
    const parsed = parseInt(raw, 10)
    return isNaN(parsed) ? raw : parsed
  }
  if (item.type === 'float') {
    const parsed = parseFloat(raw)
    return isNaN(parsed) ? raw : parsed
  }
  return raw
}

const onValueChange = (item: HiddenConfigItem, newValue: unknown) => {
  if (newValue === item.value || (item.type !== 'bool' && newValue === '' && item.value === null)) {
    const updated = { ...pendingChanges.value }
    delete updated[item.key]
    pendingChanges.value = updated
  } else {
    pendingChanges.value = { ...pendingChanges.value, [item.key]: newValue }
  }
}

const getGroupIcon = (name: string): string => {
  const icons: Record<string, string> = {
    'L1 缓冲': 'mdi-lightning-bolt',
    'Token 预算': 'mdi-counter',
    '遗忘算法': 'mdi-delete-clock',
    '调试配置': 'mdi-bug',
    '性能调优': 'mdi-speedometer',
    'L3 知识图谱': 'mdi-graph',
    'LLM 调用管理': 'mdi-robot',
    '定时任务': 'mdi-clock-outline',
    '知识图谱提取任务': 'mdi-graph-outline',
    'Tool 配置': 'mdi-tools',
    'Web 安全': 'mdi-shield-lock',
    '画像系统': 'mdi-account-group',
    '图片解析': 'mdi-image-search',
    '输入清理': 'mdi-filter',
    '遗忘确认': 'mdi-check-decagram',
    'L2 查询改写': 'mdi-text-search'
  }
  return icons[name] || 'mdi-cog'
}

const handleResetItem = async (item: HiddenConfigItem) => {
  const updated = { ...pendingChanges.value }
  delete updated[item.key]
  pendingChanges.value = updated

  try {
    await deleteHiddenConfig(item.key)
    notify(`${item.key} 已恢复为默认值`)
    loadData()
  } catch (e: unknown) {
    notify((e as Error).message || '操作失败', 'error')
  }
}

const handleSaveAll = async () => {
  if (!hasChanges.value) return

  saving.value = true
  try {
    const updatedKeys = await updateHiddenConfig(pendingChanges.value)
    pendingChanges.value = {}
    notify(`已保存 ${updatedKeys.length} 项配置`)
    loadData()
  } catch (e: unknown) {
    notify((e as Error).message || '保存失败', 'error')
  } finally {
    saving.value = false
  }
}

const handleDiscardChanges = () => {
  pendingChanges.value = {}
  notify('已放弃修改', 'info')
}

const handleResetAll = () => {
  showConfirm('确认要将所有隐藏参数重置为默认值吗？此操作不可逆。', async () => {
    resetting.value = true
    try {
      await resetHiddenConfig()
      pendingChanges.value = {}
      notify('所有隐藏参数已重置为默认值')
      loadData()
    } catch (e: unknown) {
      notify((e as Error).message || '重置失败', 'error')
    } finally {
      resetting.value = false
    }
  })
}

const showConfirm = (message: string, callback: () => void) => {
  confirmMessage.value = message
  confirmCallback.value = callback
  confirmDialog.value = true
}

const confirmAction = () => {
  confirmDialog.value = false
  if (confirmCallback.value) {
    confirmCallback.value()
    confirmCallback.value = null
  }
}

const notify = (text: string, color: string = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  showSnackbar.value = true
}

const loadData = async () => {
  loading.value = true
  try {
    const result = await getHiddenConfig()
    items.value = result.items
    groups.value = result.groups
    if (expandedGroups.value.length === 0) {
      expandedGroups.value = groups.value.map(g => g.name)
    }
  } catch (e: unknown) {
    notify((e as Error).message || '加载失败', 'error')
  } finally {
    loading.value = false
  }
}

const handleRefresh = () => {
  loadData()
}

onMounted(() => {
  loadData()
  window.addEventListener('iris:refresh', handleRefresh)
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 12px;
  padding: 4px 0;
}

.config-item {
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  transition: border-color 0.2s, background-color 0.2s;
}

.config-item:hover {
  border-color: rgba(var(--v-theme-primary), 0.3);
}

.config-item--changed {
  border-color: rgba(var(--v-theme-warning), 0.5);
  background: rgba(var(--v-theme-warning), 0.04);
}

.config-item__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
  gap: 8px;
  flex-wrap: wrap;
}

.config-item__desc {
  margin: 0;
  line-height: 1.4;
}

.config-item__input {
  margin-top: 4px;
}

.sticky-bottom-bar {
  position: fixed;
  bottom: 0;
  left: var(--v-layout-left, 0px);
  right: 0;
  z-index: 100;
  padding: 0 16px 16px;
  pointer-events: none;
  transition: left 0.2s cubic-bezier(0.4, 0, 0.2, 1), all 0.2s;
}

.sticky-bottom-bar > .v-card {
  pointer-events: auto;
  max-width: 900px;
  margin: 0 auto;
}

.sticky-bottom-bar--elevated > .v-card {
  border: 1px solid rgba(var(--v-theme-primary), 0.3);
}
</style>
