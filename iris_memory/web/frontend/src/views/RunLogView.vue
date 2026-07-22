<template>
  <div class="run-log-view">
    <v-card color="surface" variant="flat" class="iris-hero-card mb-3">
      <v-card-title class="d-flex align-center iris-section-title">
        <v-icon icon="mdi-text-box-search-outline" color="primary" class="mr-2" />
        运行日志
        <v-spacer />
        <v-btn
          color="error"
          variant="tonal"
          size="small"
          prepend-icon="mdi-delete-sweep-outline"
          class="mr-2"
          :disabled="!entries.length"
          @click="handleClear"
        >
          清空
        </v-btn>
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
        <v-alert type="info" variant="tonal" density="compact">
          <div class="text-body-2">
            记录最近的 LLM 调用、上下文注入（含截断详情）与主动回复决策，用于排查问题。
            每类保留条数可在「隐藏参数 → 运行日志」中调整（默认 10 条）。
          </div>
        </v-alert>
      </v-card-text>
    </v-card>

    <div class="d-flex align-center flex-wrap ga-2 mb-3">
      <v-chip
        :variant="selectedType === '' ? 'flat' : 'tonal'"
        :color="selectedType === '' ? 'primary' : 'default'"
        size="small"
        @click="selectedType = ''"
      >
        全部 ({{ totalCount }})
      </v-chip>
      <v-chip
        v-for="t in types"
        :key="t.key"
        :variant="selectedType === t.key ? 'flat' : 'tonal'"
        :color="selectedType === t.key ? typeColor(t.key) : 'default'"
        size="small"
        @click="selectedType = t.key"
      >
        {{ t.label }} ({{ counts[t.key] || 0 }})
      </v-chip>
    </div>

    <v-progress-circular v-if="loading" indeterminate color="primary" size="48" class="d-block mx-auto my-8" />

    <v-card v-else-if="!entries.length" color="surface" variant="flat" class="pa-8 text-center">
      <v-icon icon="mdi-text-box-outline" size="48" color="medium-emphasis" class="mb-2" />
      <div class="text-body-1 text-medium-emphasis">暂无运行日志</div>
    </v-card>

    <v-expansion-panels v-else v-model="expanded" multiple variant="accordion">
      <v-expansion-panel
        v-for="entry in entries"
        :key="entry.id"
        :value="entry.id"
        class="bg-surface mb-1 log-panel"
      >
        <v-expansion-panel-title class="py-2">
          <div class="d-flex align-center ga-2 log-title-row">
            <v-chip :color="typeColor(entry.type)" size="x-small" variant="tonal" label>
              {{ entry.type_label }}
            </v-chip>
            <v-icon
              :icon="entry.success ? 'mdi-check-circle-outline' : 'mdi-alert-circle-outline'"
              :color="entry.success ? 'success' : 'error'"
              size="18"
            />
            <span class="text-body-2 log-title">{{ entry.title }}</span>
            <v-spacer />
            <span class="text-caption text-medium-emphasis log-time">
              {{ formatTime(entry.timestamp) }}
            </span>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <template v-for="block in detailBlocks(entry)" :key="block.key">
            <div v-if="block.kind === 'row'" class="detail-row">
              <span class="detail-key">{{ block.label }}</span>
              <span class="detail-value">{{ block.value }}</span>
            </div>
            <div v-else class="detail-text-block">
              <div class="detail-key mb-1">{{ block.label }}</div>
              <pre class="detail-pre">{{ block.value }}</pre>
            </div>
          </template>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <v-snackbar v-model="showSnackbar" :color="snackbarColor" :timeout="3000" location="top">
      {{ snackbarText }}
    </v-snackbar>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { getRunLogs, clearRunLogs, type RunLogEntry, type RunLogType } from '@/api/runLog'

const FIELD_LABELS: Record<string, string> = {
  module: '模块',
  path: '调用方式',
  provider_id: 'Provider',
  prompt: 'Prompt',
  prompt_chars: 'Prompt 长度',
  system_prompt: 'System Prompt',
  contexts_count: '上下文条数',
  image_count: '图片数',
  response: '响应',
  response_chars: '响应长度',
  input_tokens: '输入 Tokens',
  output_tokens: '输出 Tokens',
  duration_ms: '耗时 (ms)',
  error: '错误',
  group_id: '群聊',
  session_id: '会话',
  user_message: '触发消息',
  injected_sections: '注入 section 数',
  total_chars: '注入总字符',
  sections: 'Section 详情',
  image: '图片解析',
  content: '注入内容',
  wake: '触发源',
  motive: '动机',
  quiet_minutes: '冷场分钟数',
  result: '决策结果',
  should_speak: '是否发言',
  message: '发言内容',
  observation: '观察',
  watch_users: '关注用户',
  watch_keywords: '关注关键词',
  watch_reason: '关注原因',
  drifted: '话题漂移',
  cooldown_minutes: '冷却 (分钟)',
  parse_failed: '解析失败',
  stage: '阶段',
  raw_response: 'LLM 原始响应'
}

const TEXT_KEYS = new Set([
  'prompt',
  'system_prompt',
  'response',
  'raw_response',
  'content',
  'message',
  'user_message',
  'sections',
  'image'
])

const HIDDEN_KEYS = new Set(['error'])

interface DetailBlock {
  key: string
  label: string
  kind: 'row' | 'text'
  value: string
}

const entries = ref<RunLogEntry[]>([])
const counts = ref<Record<string, number>>({})
const types = ref<RunLogType[]>([])
const selectedType = ref('')
const expanded = ref<number[]>([])
const loading = ref(false)

const showSnackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const totalCount = computed(() =>
  Object.values(counts.value).reduce((sum, n) => sum + n, 0)
)

const typeColor = (type: string) => {
  switch (type) {
    case 'llm_call':
      return 'primary'
    case 'injection':
      return 'teal'
    case 'proactive':
      return 'deep-purple'
    default:
      return 'default'
  }
}

const formatTime = (ts: string) => {
  if (!ts) return ''
  const t = ts.replace('T', ' ')
  return t.length > 23 ? t.slice(0, 23) : t
}

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'object') return JSON.stringify(value, null, 2)
  return String(value)
}

const detailBlocks = (entry: RunLogEntry): DetailBlock[] => {
  const blocks: DetailBlock[] = []
  for (const [key, value] of Object.entries(entry.detail || {})) {
    if (HIDDEN_KEYS.has(key) && !value) continue
    if (value === null || value === undefined || value === '') continue
    const label = FIELD_LABELS[key] || key
    const isObject = typeof value === 'object'
    const isLongText = typeof value === 'string' && (value.length > 80 || value.includes('\n'))
    if (TEXT_KEYS.has(key) || isObject || isLongText) {
      blocks.push({ key, label, kind: 'text', value: formatValue(value) })
    } else {
      blocks.push({ key, label, kind: 'row', value: formatValue(value) })
    }
  }
  return blocks
}

const notify = (text: string, color = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  showSnackbar.value = true
}

const loadData = async () => {
  loading.value = true
  try {
    const data = await getRunLogs(selectedType.value || undefined)
    entries.value = data.entries
    counts.value = data.counts
    types.value = data.types
  } catch (e: any) {
    notify(e?.message || '获取运行日志失败', 'error')
  } finally {
    loading.value = false
  }
}

const handleClear = async () => {
  try {
    const cleared = await clearRunLogs(selectedType.value || undefined)
    notify(`已清空 ${cleared} 条日志`)
    await loadData()
  } catch (e: any) {
    notify(e?.message || '清空运行日志失败', 'error')
  }
}

watch(selectedType, () => {
  expanded.value = []
  loadData()
})

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
.log-panel {
  border-radius: 8px !important;
}

.log-title-row {
  width: 100%;
  min-width: 0;
}

.log-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.log-time {
  flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}

.detail-row {
  display: flex;
  gap: 12px;
  padding: 3px 0;
  font-size: 0.85rem;
}

.detail-key {
  color: rgba(var(--v-theme-on-surface), 0.6);
  flex-shrink: 0;
  min-width: 110px;
  font-size: 0.8rem;
}

.detail-value {
  word-break: break-all;
}

.detail-text-block {
  margin-top: 8px;
}

.detail-pre {
  background: rgba(var(--v-theme-on-surface), 0.05);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 0.78rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 320px;
  overflow-y: auto;
}
</style>
