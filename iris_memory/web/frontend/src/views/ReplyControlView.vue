<template>
  <div class="reply-control-view">
    <v-card color="surface" variant="flat" class="iris-hero-card mb-3">
      <v-card-title class="d-flex align-center iris-section-title">
        <v-icon icon="mdi-robot" color="primary" class="mr-2" />
        主动回复
        <v-spacer />
        <v-btn
          color="primary"
          variant="tonal"
          size="small"
          prepend-icon="mdi-refresh"
          :loading="loading"
          @click="reloadActiveTab"
        >
          刷新
        </v-btn>
      </v-card-title>
      <v-card-text class="pt-0">
        <div class="text-body-2 text-medium-emphasis">
          管理白名单群聊的主动回复行为，查看 LLM 决策统计与调用日志，并调整触发与主动发起参数。
        </div>
      </v-card-text>
    </v-card>

    <v-card color="surface" variant="flat" class="iris-card">
      <v-tabs v-model="tab" color="primary" density="comfortable">
        <v-tab value="manage" prepend-icon="mdi-account-group">群聊管理</v-tab>
        <v-tab value="stats" prepend-icon="mdi-chart-pie">统计监控</v-tab>
        <v-tab value="settings" prepend-icon="mdi-cog">设置</v-tab>
      </v-tabs>

      <v-divider />

      <v-window v-model="tab">
        <!-- ============ 群聊管理 ============ -->
        <v-window-item value="manage">
          <div class="pa-4">
            <div class="d-flex align-center ga-3 mb-4 flex-wrap">
              <v-text-field
                v-model="newGroupId"
                label="输入群ID以添加到白名单"
                density="compact"
                variant="outlined"
                hide-details
                style="max-width: 280px"
                @keydown.enter="handleEnableGroup"
              />
              <v-btn
                color="primary"
                variant="tonal"
                prepend-icon="mdi-plus"
                :loading="manageActionLoading"
                :disabled="!newGroupId.trim()"
                @click="handleEnableGroup"
              >
                启用群聊
              </v-btn>
            </div>

            <v-progress-circular
              v-if="loading"
              indeterminate
              color="primary"
              size="48"
              class="d-block mx-auto my-8"
            />

            <div v-else-if="!whitelist.length" class="iris-empty-state">
              <v-icon icon="mdi-inbox-outline" size="48" />
              <div class="iris-empty-state__title">暂无已启用的群聊</div>
              <div class="iris-empty-state__desc">在上方输入群ID并点击「启用群聊」</div>
            </div>

            <div v-else class="table-wrapper">
              <v-table density="compact" class="iris-table">
                <thead>
                  <tr>
                    <th>群ID</th>
                    <th>状态机</th>
                    <th>意愿</th>
                    <th>消息</th>
                    <th>阈值 N/T</th>
                    <th>退避</th>
                    <th>连续</th>
                    <th>锚点</th>
                    <th>今日发起</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="g in whitelist" :key="g.group_id">
                    <td class="font-weight-medium">{{ g.group_id }}</td>
                    <td>
                      <v-chip
                        size="x-small"
                        variant="tonal"
                        :color="stateChip(g.state).color"
                      >
                        {{ stateChip(g.state).text }}
                      </v-chip>
                    </td>
                    <td>
                      <v-select
                        :model-value="g.willingness"
                        :items="willingnessItems"
                        density="compact"
                        variant="outlined"
                        hide-details
                        style="min-width: 84px"
                        @update:model-value="(v: string) => handleSetWillingness(g.group_id, v)"
                      />
                    </td>
                    <td>{{ g.msg_count }}</td>
                    <td>{{ g.effective_n }}/{{ g.effective_t }}m</td>
                    <td>{{ g.backoff_level }}</td>
                    <td>{{ g.consecutive_replies }}</td>
                    <td style="max-width: 260px">
                      <div v-if="g.anchor_kind" class="d-flex flex-wrap ga-1 align-center">
                        <v-chip size="x-small" variant="tonal" color="info">{{ g.anchor_kind }}</v-chip>
                        <v-chip
                          v-for="u in g.anchor_users || []"
                          :key="u"
                          size="x-small"
                          variant="tonal"
                          color="primary"
                        >
                          {{ u }}
                        </v-chip>
                        <v-chip
                          v-for="k in g.anchor_keywords || []"
                          :key="k"
                          size="x-small"
                          variant="tonal"
                          color="warning"
                        >
                          {{ k }}
                        </v-chip>
                        <span v-if="g.anchor_reason" class="text-caption text-medium-emphasis">
                          {{ g.anchor_reason }}
                        </span>
                      </div>
                      <span v-else class="text-medium-emphasis">-</span>
                    </td>
                    <td>
                      {{ g.initiate_daily_count }}
                      <v-chip v-if="g.initiate_pending" size="x-small" variant="tonal" color="warning" class="ml-1">
                        等待接话
                      </v-chip>
                    </td>
                    <td>
                      <v-tooltip text="重置群状态" location="top">
                        <template #activator="{ props }">
                          <v-btn
                            icon="mdi-restore"
                            variant="text"
                            size="x-small"
                            v-bind="props"
                            @click="handleResetGroup(g.group_id)"
                          />
                        </template>
                      </v-tooltip>
                      <v-tooltip text="禁用该群主动回复" location="top">
                        <template #activator="{ props }">
                          <v-btn
                            icon="mdi-block-helper"
                            variant="text"
                            size="x-small"
                            color="error"
                            v-bind="props"
                            @click="handleDisableGroup(g.group_id)"
                          />
                        </template>
                      </v-tooltip>
                    </td>
                  </tr>
                </tbody>
              </v-table>
            </div>
          </div>
        </v-window-item>

        <!-- ============ 统计监控 ============ -->
        <v-window-item value="stats">
          <div class="pa-4">
            <v-progress-circular
              v-if="loading"
              indeterminate
              color="primary"
              size="48"
              class="d-block mx-auto my-8"
            />

            <v-alert v-else-if="!statsEnabled" type="info" variant="tonal" class="my-4">
              <div class="text-subtitle-1 font-weight-medium mb-1">统计监控未启用</div>
              <div class="text-body-2">请在插件配置中开启「启用统计监控」选项后刷新此页面</div>
            </v-alert>

            <template v-else>
              <div class="d-flex align-center ga-3 mb-4 flex-wrap">
                <v-select
                  v-model="selectedGroup"
                  :items="groupFilterItems"
                  label="群聊筛选"
                  density="compact"
                  variant="outlined"
                  hide-details
                  style="max-width: 240px"
                />
                <v-spacer />
                <v-btn
                  color="error"
                  variant="tonal"
                  size="small"
                  prepend-icon="mdi-delete-sweep"
                  @click="handleClearStats"
                >
                  清除记录
                </v-btn>
              </div>

              <div class="text-subtitle-1 font-weight-medium mb-2">群聊概览</div>
              <div v-if="!statsGroups.length" class="iris-empty-state">
                <v-icon icon="mdi-inbox-outline" size="48" />
                <div class="iris-empty-state__title">暂无群聊数据</div>
              </div>
              <div v-else class="table-wrapper mb-6">
                <v-table density="compact" class="iris-table">
                  <thead>
                    <tr>
                      <th>群ID</th>
                      <th>状态</th>
                      <th>意愿</th>
                      <th>决策</th>
                      <th>回复</th>
                      <th>跳过</th>
                      <th>错误</th>
                      <th>偏移</th>
                      <th>发起</th>
                      <th>被动</th>
                      <th>消息</th>
                      <th>阈值 N/T</th>
                      <th>退避</th>
                      <th>连续</th>
                      <th>最近决策</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="g in filteredStatsGroups" :key="g.group_id">
                      <td class="font-weight-medium">{{ g.group_id }}</td>
                      <td>
                        <v-chip size="x-small" variant="tonal" :color="stateChip(g.current_state).color">
                          {{ stateChip(g.current_state).text }}
                        </v-chip>
                      </td>
                      <td :class="`text-${willingnessColor(g.willingness)}`">
                        {{ willingnessText(g.willingness) }}
                      </td>
                      <td>{{ g.total_decisions }}</td>
                      <td>{{ g.total_replies }}</td>
                      <td>{{ g.total_skips }}</td>
                      <td>{{ g.total_errors }}</td>
                      <td>{{ g.total_drifts }}</td>
                      <td>{{ g.total_initiates }}</td>
                      <td>{{ g.total_passive_replies }}</td>
                      <td>{{ g.msg_count }}</td>
                      <td>{{ g.effective_n }}/{{ g.effective_t }}m</td>
                      <td>{{ g.backoff_level }}</td>
                      <td>{{ g.consecutive_replies }}</td>
                      <td class="text-medium-emphasis">{{ formatTime(g.last_decision_time) }}</td>
                    </tr>
                  </tbody>
                </v-table>
              </div>

              <div class="text-subtitle-1 font-weight-medium mb-2">LLM 调用日志</div>
              <div v-if="!logs.length" class="iris-empty-state">
                <v-icon icon="mdi-file-document-outline" size="48" />
                <div class="iris-empty-state__title">暂无调用日志</div>
              </div>
              <v-expansion-panels v-else variant="accordion" class="mb-4">
                <v-expansion-panel
                  v-for="log in logs"
                  :key="`${log.timestamp}-${log.group_id}`"
                  class="bg-surface"
                >
                  <v-expansion-panel-title class="py-2">
                    <div class="d-flex align-center ga-2 flex-wrap log-title-row">
                      <v-chip size="x-small" variant="tonal" :color="resultChip(log).color">
                        {{ resultChip(log).text }}
                      </v-chip>
                      <span class="text-body-2 font-weight-medium">{{ log.group_id }}</span>
                      <v-chip size="x-small" variant="tonal" color="info">
                        {{ motiveText(log.motive) }}
                      </v-chip>
                      <span class="text-caption text-medium-emphasis">{{ formatDuration(log.duration_ms) }}</span>
                      <v-spacer />
                      <span class="text-caption text-medium-emphasis">{{ formatTime(log.timestamp) }}</span>
                    </div>
                  </v-expansion-panel-title>
                  <v-expansion-panel-text>
                    <template v-if="log.message">
                      <div class="log-label">发言内容</div>
                      <pre class="log-pre">{{ log.message }}</pre>
                    </template>
                    <div class="log-label">观察摘要</div>
                    <pre class="log-pre">{{ log.observation || '-' }}</pre>
                    <template v-if="log.watch_users && log.watch_users.length">
                      <div class="log-label">关注用户</div>
                      <pre class="log-pre">{{ log.watch_users.join(', ') }}</pre>
                    </template>
                    <template v-if="log.watch_keywords && log.watch_keywords.length">
                      <div class="log-label">关注关键词</div>
                      <pre class="log-pre">{{ log.watch_keywords.join(', ') }}</pre>
                    </template>
                    <template v-if="log.watch_reason">
                      <div class="log-label">关注原因</div>
                      <pre class="log-pre">{{ log.watch_reason }}</pre>
                    </template>
                    <div class="log-label">LLM 原始响应</div>
                    <pre class="log-pre">{{ log.response_text }}</pre>
                    <div class="log-label">System Prompt</div>
                    <pre class="log-pre">{{ log.system_prompt }}</pre>
                    <div class="log-label">User Prompt</div>
                    <pre class="log-pre">{{ log.user_prompt }}</pre>
                  </v-expansion-panel-text>
                </v-expansion-panel>
              </v-expansion-panels>

              <div class="d-flex align-center justify-center ga-3">
                <v-btn
                  variant="tonal"
                  size="small"
                  prepend-icon="mdi-chevron-left"
                  :disabled="logPage === 0"
                  @click="logPage--; loadLogs()"
                >
                  上一页
                </v-btn>
                <span class="text-body-2 text-medium-emphasis">第 {{ logPage + 1 }} 页</span>
                <v-btn
                  variant="tonal"
                  size="small"
                  append-icon="mdi-chevron-right"
                  :disabled="logs.length < LOG_PAGE_SIZE"
                  @click="logPage++; loadLogs()"
                >
                  下一页
                </v-btn>
              </div>
            </template>
          </div>
        </v-window-item>

        <!-- ============ 设置 ============ -->
        <v-window-item value="settings">
          <div class="pa-4">
            <v-progress-circular
              v-if="loading"
              indeterminate
              color="primary"
              size="48"
              class="d-block mx-auto my-8"
            />

            <div v-else-if="configLoadFailed" class="iris-empty-state">
              <v-icon icon="mdi-alert-circle" size="48" />
              <div class="iris-empty-state__title">加载配置失败</div>
            </div>

            <template v-else>
              <v-card
                v-for="section in configSections"
                :key="section.title"
                color="surface"
                variant="outlined"
                class="mb-4 iris-card"
              >
                <v-card-title class="d-flex align-center iris-section-title py-3">
                  <v-icon :icon="section.icon" color="primary" class="mr-2" size="20" />
                  {{ section.title }}
                </v-card-title>
                <v-card-text>
                  <div class="config-grid">
                    <div v-for="key in section.keys" :key="key" class="config-item">
                      <div class="text-body-2 font-weight-medium mb-1">
                        {{ configMeta[key].label }}
                      </div>
                      <p v-if="configMeta[key].hint" class="text-caption text-medium-emphasis mb-2">
                        {{ configMeta[key].hint }}
                      </p>

                      <div v-if="configMeta[key].type === 'object'" class="config-sub-grid">
                        <v-text-field
                          v-for="(subMeta, subKey) in configMeta[key].items"
                          :key="subKey"
                          :model-value="getSubValue(key, String(subKey))"
                          :label="subMeta.label"
                          type="number"
                          density="compact"
                          variant="outlined"
                          hide-details
                          :min="subMeta.min"
                          :max="subMeta.max"
                          @update:model-value="(v: unknown) => setSubValue(key, String(subKey), v)"
                        />
                      </div>

                      <v-switch
                        v-else-if="configMeta[key].type === 'bool'"
                        v-model="configForm[key]"
                        density="compact"
                        color="primary"
                        hide-details
                        :label="configForm[key] ? '启用' : '禁用'"
                      />

                      <v-text-field
                        v-else-if="configMeta[key].type === 'str'"
                        v-model="configForm[key]"
                        density="compact"
                        variant="outlined"
                        hide-details
                      />

                      <v-text-field
                        v-else
                        v-model="configForm[key]"
                        type="number"
                        density="compact"
                        variant="outlined"
                        hide-details
                        :min="configMeta[key].min"
                        :max="configMeta[key].max"
                        :step="configMeta[key].step || (configMeta[key].type === 'float' ? 0.01 : 1)"
                      />
                    </div>
                  </div>
                </v-card-text>
              </v-card>

              <div class="d-flex align-center ga-3">
                <v-btn
                  color="primary"
                  prepend-icon="mdi-content-save"
                  :loading="configSaving"
                  @click="handleSaveConfig"
                >
                  保存配置
                </v-btn>
              </div>
            </template>
          </div>
        </v-window-item>
      </v-window>
    </v-card>

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
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import {
  getReplyWhitelist,
  enableReplyGroup,
  disableReplyGroup,
  setReplyWillingness,
  resetReplyGroup,
  getReplyStatsStatus,
  getReplyStatsGroups,
  getReplyStatsLogs,
  clearReplyStats,
  getReplyConfig,
  setReplyConfig,
  type ReplyWhitelistGroup,
  type ReplyStatsGroup,
  type ReplyLlmLog,
  type ReplyConfigMeta
} from '@/api/reply'

const LOG_PAGE_SIZE = 20

const tab = ref('manage')
const loading = ref(false)

const whitelist = ref<ReplyWhitelistGroup[]>([])
const newGroupId = ref('')
const manageActionLoading = ref(false)

const statsEnabled = ref(false)
const statsGroups = ref<ReplyStatsGroup[]>([])
const logs = ref<ReplyLlmLog[]>([])
const selectedGroup = ref('')
const logPage = ref(0)

const configMeta = ref<Record<string, ReplyConfigMeta>>({})
const configForm = ref<Record<string, any>>({})
const configSaving = ref(false)
const configLoadFailed = ref(false)

const confirmDialog = ref(false)
const confirmMessage = ref('')
const confirmCallback = ref<(() => void) | null>(null)

const showSnackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const willingnessItems = [
  { title: '低', value: 'low' },
  { title: '中', value: 'medium' },
  { title: '高', value: 'high' }
]

const basicKeys = [
  'mute_period', 'window_size', 'default_n', 'default_t', 'max_token',
  'quality_threshold', 'follow_up_ttl', 'follow_up_aggregate_window',
  'trigger_min_interval', 'boost_factor', 'boost_duration', 'max_boosted_replies'
]
const proactiveKeys = [
  'proactive_enabled', 'proactive_check_interval', 'proactive_quiet_minutes',
  'proactive_max_per_day', 'proactive_min_interval', 'proactive_drift_delay',
  'proactive_pending_timeout', 'proactive_max_streak', 'proactive_instruction',
  'proactive_max_message_len'
]

const configSections = computed(() => [
  {
    title: '基本参数',
    icon: 'mdi-tune',
    keys: basicKeys.filter(k => configMeta.value[k])
  },
  {
    title: '主动发起',
    icon: 'mdi-robot',
    keys: proactiveKeys.filter(k => configMeta.value[k])
  }
])

const groupFilterItems = computed(() => [
  { title: '全部群聊', value: '' },
  ...statsGroups.value.map(g => ({ title: g.group_id, value: g.group_id }))
])

const filteredStatsGroups = computed(() => {
  if (!selectedGroup.value) return statsGroups.value
  return statsGroups.value.filter(g => g.group_id === selectedGroup.value)
})

const stateChip = (state: string): { text: string; color: string } => {
  const map: Record<string, { text: string; color: string }> = {
    idle: { text: '空闲', color: 'grey' },
    cooldown: { text: '冷却', color: 'warning' },
    following: { text: '跟进', color: 'info' }
  }
  return map[state] || { text: state, color: 'grey' }
}

const willingnessText = (level: string): string => {
  const map: Record<string, string> = { low: '低', medium: '中', high: '高' }
  return map[level] || level
}

const willingnessColor = (level: string): string => {
  const map: Record<string, string> = { low: 'error', medium: 'warning', high: 'success' }
  return map[level] || 'medium-emphasis'
}

const resultChip = (log: ReplyLlmLog): { text: string; color: string } => {
  if (log.drifted) return { text: '话题偏移', color: 'error' }
  if (log.action === 'speak') return { text: '回复', color: 'success' }
  return { text: '跳过', color: 'warning' }
}

const motiveText = (motive: string): string => {
  const map: Record<string, string> = {
    chime_in: '跟话',
    follow_up: '跟进',
    initiate: '发起',
    watch: '评估'
  }
  return map[motive] || motive
}

const formatTime = (ts: number): string => {
  if (!ts) return '-'
  const d = new Date(ts * 1000)
  return d.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const formatDuration = (ms: number): string => {
  if (!ms) return '-'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

const getSubValue = (key: string, subKey: string): unknown => {
  const val = configForm.value[key]
  if (val && typeof val === 'object') return val[subKey]
  return configMeta.value[key]?.items?.[subKey]?.min
}

const setSubValue = (key: string, subKey: string, value: unknown) => {
  const current = configForm.value[key]
  const base = current && typeof current === 'object' ? { ...current } : {}
  base[subKey] = value === '' || value === null || value === undefined ? value : Number(value)
  configForm.value = { ...configForm.value, [key]: base }
}

const notify = (text: string, color: string = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  showSnackbar.value = true
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

const loadManage = async () => {
  loading.value = true
  try {
    whitelist.value = await getReplyWhitelist()
  } catch (e: unknown) {
    notify((e as Error).message || '加载白名单失败', 'error')
  } finally {
    loading.value = false
  }
}

const handleEnableGroup = async () => {
  const gid = newGroupId.value.trim()
  if (!gid) return
  manageActionLoading.value = true
  try {
    await enableReplyGroup(gid)
    newGroupId.value = ''
    notify(`已启用群 ${gid}`)
    loadManage()
  } catch (e: unknown) {
    notify((e as Error).message || '启用失败', 'error')
  } finally {
    manageActionLoading.value = false
  }
}

const handleDisableGroup = (gid: string) => {
  showConfirm(`确定要禁用群 ${gid} 的主动回复吗？`, async () => {
    try {
      await disableReplyGroup(gid)
      notify(`已禁用群 ${gid}`)
      loadManage()
    } catch (e: unknown) {
      notify((e as Error).message || '禁用失败', 'error')
    }
  })
}

const handleResetGroup = (gid: string) => {
  showConfirm(`确定要重置群 ${gid} 的状态吗？`, async () => {
    try {
      await resetReplyGroup(gid)
      notify(`已重置群 ${gid}`)
      loadManage()
    } catch (e: unknown) {
      notify((e as Error).message || '重置失败', 'error')
    }
  })
}

const handleSetWillingness = async (gid: string, willingness: string) => {
  try {
    await setReplyWillingness(gid, willingness)
    notify(`群 ${gid} 意愿已调整为「${willingnessText(willingness)}」`)
  } catch (e: unknown) {
    notify((e as Error).message || '设置失败', 'error')
    loadManage()
  }
}

const loadStats = async () => {
  loading.value = true
  try {
    const status = await getReplyStatsStatus()
    statsEnabled.value = status.enabled
    if (!status.enabled) return
    statsGroups.value = await getReplyStatsGroups()
    await loadLogs()
  } catch (e: unknown) {
    notify((e as Error).message || '加载统计失败', 'error')
  } finally {
    loading.value = false
  }
}

const loadLogs = async () => {
  try {
    logs.value = await getReplyStatsLogs({
      group_id: selectedGroup.value || undefined,
      limit: LOG_PAGE_SIZE,
      offset: logPage.value * LOG_PAGE_SIZE
    })
  } catch (e: unknown) {
    notify((e as Error).message || '加载日志失败', 'error')
  }
}

const handleClearStats = () => {
  showConfirm('确定要清除所有统计数据吗？', async () => {
    try {
      await clearReplyStats()
      notify('统计数据已清除')
      logPage.value = 0
      loadStats()
    } catch (e: unknown) {
      notify((e as Error).message || '清除失败', 'error')
    }
  })
}

const loadSettings = async () => {
  loading.value = true
  configLoadFailed.value = false
  try {
    const data = await getReplyConfig()
    configMeta.value = data.meta
    configForm.value = JSON.parse(JSON.stringify(data.values || {}))
  } catch {
    configLoadFailed.value = true
  } finally {
    loading.value = false
  }
}

const handleSaveConfig = async () => {
  configSaving.value = true
  try {
    const payload: Record<string, unknown> = {}
    for (const [key, meta] of Object.entries(configMeta.value)) {
      const val = configForm.value[key]
      if (meta.type === 'object') {
        const sub: Record<string, number> = {}
        for (const subKey of Object.keys(meta.items || {})) {
          const raw = val && typeof val === 'object' ? (val as Record<string, unknown>)[subKey] : undefined
          sub[subKey] = parseInt(String(raw ?? ''), 10) || 0
        }
        payload[key] = sub
      } else if (meta.type === 'bool') {
        payload[key] = !!val
      } else if (meta.type === 'str') {
        payload[key] = String(val ?? '')
      } else {
        payload[key] = parseFloat(String(val ?? '')) || 0
      }
    }
    await setReplyConfig(payload)
    notify('配置保存成功')
  } catch (e: unknown) {
    notify((e as Error).message || '保存失败', 'error')
  } finally {
    configSaving.value = false
  }
}

const reloadActiveTab = () => {
  if (tab.value === 'manage') loadManage()
  else if (tab.value === 'stats') loadStats()
  else if (tab.value === 'settings') loadSettings()
}

watch(tab, () => {
  reloadActiveTab()
})

watch(selectedGroup, () => {
  logPage.value = 0
  if (tab.value === 'stats' && statsEnabled.value) loadLogs()
})

const handleRefresh = () => {
  reloadActiveTab()
}

onMounted(() => {
  loadManage()
  window.addEventListener('iris:refresh', handleRefresh)
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.table-wrapper {
  overflow-x: auto;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 12px;
}

.config-item {
  padding: 12px;
  border-radius: 8px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  transition: border-color 0.2s;
}

.config-item:hover {
  border-color: rgba(var(--v-theme-primary), 0.3);
}

.config-sub-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
}

.log-title-row {
  width: 100%;
}

.log-label {
  margin-top: 10px;
  margin-bottom: 4px;
  color: rgba(var(--v-theme-on-surface), 0.6);
  font-size: 12px;
  font-weight: 600;
}

.log-label:first-child {
  margin-top: 0;
}

.log-pre {
  background: rgba(var(--v-theme-on-surface), 0.04);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 6px;
  padding: 10px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 400px;
  overflow-y: auto;
}
</style>
