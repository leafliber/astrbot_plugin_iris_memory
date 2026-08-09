<template>
  <div class="learning-view">
    <ComponentDisabled
      :status="status"
      :error="error"
      :error-type="errorType"
      component-name="学习管理"
      @retry="refreshState"
    >
      <!-- 顶部 hero 卡 -->
      <v-card color="surface" variant="flat" class="iris-hero-card mb-3">
        <v-card-title class="d-flex align-center iris-section-title">
          <v-icon icon="mdi-school" color="primary" class="mr-2" />
          学习管理
          <v-spacer />
          <v-btn
            color="primary"
            variant="tonal"
            size="small"
            prepend-icon="mdi-refresh"
            :loading="refreshing"
            @click="handleManualRefresh"
          >
            刷新
          </v-btn>
        </v-card-title>
        <v-card-text class="pt-0">
          <v-alert type="info" variant="tonal" density="compact">
            <div class="text-body-2">
              学习模块从群聊对话中自动学习表达模式、对话样例与圈内暗语，可在此手动管理与审查。
            </div>
          </v-alert>
        </v-card-text>
      </v-card>

      <!-- 已学习内容：页面主任务，优先展示 -->
      <v-card color="surface" variant="flat" class="iris-card mb-3">
        <v-card-title class="d-flex align-center iris-section-title">
          <v-icon icon="mdi-database-check-outline" color="primary" class="mr-2" />
          已学习内容
        </v-card-title>
        <v-divider />
        <v-tabs v-model="activeTab" color="primary">
          <v-tab value="jargon">
            <v-icon icon="mdi-tag" class="mr-1" />
            圈内暗语
          </v-tab>
          <v-tab value="expression_pattern">
            <v-icon icon="mdi-emoticon" class="mr-1" />
            表达模式
          </v-tab>
          <v-tab value="few_shot">
            <v-icon icon="mdi-message-text-outline" class="mr-1" />
            对话样例
          </v-tab>
        </v-tabs>
        <v-divider />

        <v-window v-model="activeTab">
          <!-- 圈内暗语 -->
          <v-window-item value="jargon">
            <div class="d-flex align-center flex-wrap ga-2 pa-3">
              <v-select
                v-model="tabStates.jargon.groupId"
                :items="groupSelectItems"
                label="群"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('jargon')"
              />
              <v-select
                v-model="tabStates.jargon.status"
                :items="jargonStatusOptions"
                label="状态"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('jargon')"
              />
              <v-spacer />
              <v-btn color="primary" variant="tonal" size="small" prepend-icon="mdi-plus" @click="openAdd">
                新增
              </v-btn>
            </div>
            <v-data-table-server
              :headers="jargonHeaders"
              :items="tabStates.jargon.items"
              :items-length="tabStates.jargon.total"
              :loading="tabStates.jargon.loading"
              :page="tabStates.jargon.page"
              :items-per-page="tabStates.jargon.pageSize"
              :items-per-page-options="[10, 20, 50]"
              item-value="id"
              density="compact"
              hover
              class="iris-table"
              @update:options="(opts) => handleOptions('jargon', opts)"
            >
              <template #item.term="{ item }">
                <span class="font-weight-medium">{{ item.term }}</span>
              </template>
              <template #item.group_id="{ item }">
                <v-chip size="x-small" variant="tonal" color="info">{{ item.group_id }}</v-chip>
              </template>
              <template #item.meaning="{ item }">
                <div class="cell-ellipsis" :title="item.meaning">{{ item.meaning || '—' }}</div>
              </template>
              <template #item.confidence="{ item }">
                {{ formatConfidence(item.confidence) }}
              </template>
              <template #item.status="{ item }">
                <v-chip size="x-small" variant="tonal" :color="statusColor(item.status)">
                  {{ statusLabel(item.status) }}
                </v-chip>
              </template>
              <template #item.actions="{ item }">
                <div class="d-flex align-center flex-nowrap ga-1">
                  <v-btn
                    v-if="item.status === 'active'"
                    size="x-small"
                    variant="tonal"
                    color="warning"
                    @click="quickStatus('jargon', item, 'disabled')"
                  >
                    禁用
                  </v-btn>
                  <v-btn
                    v-if="item.status === 'disabled' || item.status === 'dormant'"
                    size="x-small"
                    variant="tonal"
                    color="info"
                    @click="quickStatus('jargon', item, 'active')"
                  >
                    启用
                  </v-btn>
                  <v-btn icon="mdi-pencil" size="x-small" variant="text" @click="openEdit('jargon', item)">
                    <v-tooltip activator="parent" location="bottom">编辑</v-tooltip>
                  </v-btn>
                  <v-btn icon="mdi-delete" size="x-small" variant="text" color="error" @click="confirmDelete('jargon', item)">
                    <v-tooltip activator="parent" location="bottom">删除</v-tooltip>
                  </v-btn>
                </div>
              </template>
              <template #no-data>
                <div class="iris-empty-state">
                  <v-icon icon="mdi-inbox-outline" size="56" />
                  <div class="iris-empty-state__title">暂无暗语数据</div>
                </div>
              </template>
            </v-data-table-server>
          </v-window-item>

          <!-- 表达模式 -->
          <v-window-item value="expression_pattern">
            <div class="d-flex align-center flex-wrap ga-2 pa-3">
              <v-select
                v-model="tabStates.expression_pattern.groupId"
                :items="groupSelectItems"
                label="群"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('expression_pattern')"
              />
              <v-select
                v-model="tabStates.expression_pattern.status"
                :items="commonStatusOptions"
                label="状态"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('expression_pattern')"
              />
              <v-spacer />
              <v-btn color="primary" variant="tonal" size="small" prepend-icon="mdi-plus" @click="openAdd">
                新增
              </v-btn>
            </div>
            <v-data-table-server
              :headers="patternHeaders"
              :items="tabStates.expression_pattern.items"
              :items-length="tabStates.expression_pattern.total"
              :loading="tabStates.expression_pattern.loading"
              :page="tabStates.expression_pattern.page"
              :items-per-page="tabStates.expression_pattern.pageSize"
              :items-per-page-options="[10, 20, 50]"
              item-value="id"
              density="compact"
              hover
              class="iris-table"
              @update:options="(opts) => handleOptions('expression_pattern', opts)"
            >
              <template #item.scene="{ item }">
                <div class="cell-ellipsis" :title="item.scene">{{ item.scene }}</div>
              </template>
              <template #item.expression="{ item }">
                <div class="cell-ellipsis" :title="item.expression">{{ item.expression }}</div>
              </template>
              <template #item.group_id="{ item }">
                <v-chip size="x-small" variant="tonal" color="info">{{ item.group_id }}</v-chip>
              </template>
              <template #item.status="{ item }">
                <v-chip size="x-small" variant="tonal" :color="statusColor(item.status)">
                  {{ statusLabel(item.status) }}
                </v-chip>
              </template>
              <template #item.actions="{ item }">
                <div class="d-flex align-center flex-nowrap ga-1">
                  <v-btn
                    v-if="item.status === 'pending_review'"
                    size="x-small"
                    variant="tonal"
                    color="success"
                    @click="quickStatus('expression_pattern', item, 'approved')"
                  >
                    通过
                  </v-btn>
                  <v-btn
                    v-if="item.status === 'pending_review'"
                    size="x-small"
                    variant="tonal"
                    color="warning"
                    @click="quickStatus('expression_pattern', item, 'disabled')"
                  >
                    禁用
                  </v-btn>
                  <v-btn icon="mdi-pencil" size="x-small" variant="text" @click="openEdit('expression_pattern', item)">
                    <v-tooltip activator="parent" location="bottom">编辑</v-tooltip>
                  </v-btn>
                  <v-btn icon="mdi-delete" size="x-small" variant="text" color="error" @click="confirmDelete('expression_pattern', item)">
                    <v-tooltip activator="parent" location="bottom">删除</v-tooltip>
                  </v-btn>
                </div>
              </template>
              <template #no-data>
                <div class="iris-empty-state">
                  <v-icon icon="mdi-inbox-outline" size="56" />
                  <div class="iris-empty-state__title">暂无表达模式数据</div>
                </div>
              </template>
            </v-data-table-server>
          </v-window-item>

          <!-- 对话样例 -->
          <v-window-item value="few_shot">
            <div class="d-flex align-center flex-wrap ga-2 pa-3">
              <v-select
                v-model="tabStates.few_shot.groupId"
                :items="groupSelectItems"
                label="群"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('few_shot')"
              />
              <v-select
                v-model="tabStates.few_shot.status"
                :items="commonStatusOptions"
                label="状态"
                density="compact"
                variant="outlined"
                hide-details
                class="filter-select"
                @update:model-value="applyFilter('few_shot')"
              />
              <v-spacer />
              <v-btn color="primary" variant="tonal" size="small" prepend-icon="mdi-plus" @click="openAdd">
                新增
              </v-btn>
            </div>
            <v-data-table-server
              :headers="fewshotHeaders"
              :items="tabStates.few_shot.items"
              :items-length="tabStates.few_shot.total"
              :loading="tabStates.few_shot.loading"
              :page="tabStates.few_shot.page"
              :items-per-page="tabStates.few_shot.pageSize"
              :items-per-page-options="[10, 20, 50]"
              item-value="id"
              density="compact"
              hover
              class="iris-table"
              @update:options="(opts) => handleOptions('few_shot', opts)"
            >
              <template #item.user_text="{ item }">
                <div class="cell-ellipsis" :title="item.user_text">{{ item.user_text }}</div>
              </template>
              <template #item.bot_text="{ item }">
                <div class="cell-ellipsis" :title="item.bot_text">{{ item.bot_text }}</div>
              </template>
              <template #item.group_id="{ item }">
                <v-chip size="x-small" variant="tonal" color="info">{{ item.group_id }}</v-chip>
              </template>
              <template #item.status="{ item }">
                <v-chip size="x-small" variant="tonal" :color="statusColor(item.status)">
                  {{ statusLabel(item.status) }}
                </v-chip>
              </template>
              <template #item.actions="{ item }">
                <div class="d-flex align-center flex-nowrap ga-1">
                  <v-btn
                    v-if="item.status === 'pending_review'"
                    size="x-small"
                    variant="tonal"
                    color="success"
                    @click="quickStatus('few_shot', item, 'approved')"
                  >
                    通过
                  </v-btn>
                  <v-btn
                    v-if="item.status === 'pending_review'"
                    size="x-small"
                    variant="tonal"
                    color="warning"
                    @click="quickStatus('few_shot', item, 'disabled')"
                  >
                    禁用
                  </v-btn>
                  <v-btn icon="mdi-pencil" size="x-small" variant="text" @click="openEdit('few_shot', item)">
                    <v-tooltip activator="parent" location="bottom">编辑</v-tooltip>
                  </v-btn>
                  <v-btn icon="mdi-delete" size="x-small" variant="text" color="error" @click="confirmDelete('few_shot', item)">
                    <v-tooltip activator="parent" location="bottom">删除</v-tooltip>
                  </v-btn>
                </div>
              </template>
              <template #no-data>
                <div class="iris-empty-state">
                  <v-icon icon="mdi-inbox-outline" size="56" />
                  <div class="iris-empty-state__title">暂无对话样例数据</div>
                </div>
              </template>
            </v-data-table-server>
          </v-window-item>
        </v-window>
      </v-card>

      <!-- 辅助信息：统计概览 -->
      <div class="d-flex align-center mb-2 mt-4 text-subtitle-1 font-weight-medium">
        <v-icon icon="mdi-chart-box-outline" color="primary" class="mr-2" />
        学习概览
      </div>
      <v-row class="mb-3">
        <v-col v-for="card in statCards" :key="card.table" cols="12" md="4">
          <v-card color="surface" variant="flat" class="iris-card h-100">
            <v-card-title class="d-flex align-center iris-section-title">
              <v-icon :icon="card.icon" color="primary" class="mr-2" />
              {{ card.label }}
              <v-spacer />
              <span class="text-h6">{{ card.total }}</span>
            </v-card-title>
            <v-card-text class="pt-0">
              <div class="d-flex align-center flex-wrap ga-2">
                <v-chip size="small" variant="tonal" color="warning">待审查 {{ card.pending }}</v-chip>
                <v-chip size="small" variant="tonal" color="success">已通过 {{ card.approved }}</v-chip>
                <v-chip size="small" variant="tonal">已禁用 {{ card.disabled }}</v-chip>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 辅助信息：自动暗语漏斗 -->
      <v-card color="surface" variant="flat" class="iris-card mb-3">
        <v-card-title class="d-flex align-center iris-section-title">
          <v-icon icon="mdi-filter-variant" color="primary" class="mr-2" />
          自动暗语漏斗
          <v-spacer />
          <v-chip size="small" variant="tonal">
            今日 LLM {{ jargonLlmUsage.call_count }} 次 / 审查 {{ jargonLlmUsage.candidate_count }} 簇
          </v-chip>
        </v-card-title>
        <v-card-text class="pt-0">
          <v-table v-if="jargonCandidates.length" density="compact" class="iris-table">
            <thead><tr><th>候选簇</th><th>群</th><th>消息</th><th>用户</th><th>统计分</th><th>状态</th><th>判定原因</th></tr></thead>
            <tbody>
              <tr v-for="item in jargonCandidates" :key="item.cluster_id">
                <td>
                  <div class="d-flex align-center ga-2">
                    <span class="font-weight-medium">{{ item.term }}</span>
                    <v-chip
                      v-if="item.cluster_size > 1"
                      size="x-small"
                      variant="outlined"
                      :title="item.cluster_terms.join(' / ')"
                    >
                      已折叠 {{ item.cluster_size }} 个片段
                    </v-chip>
                  </div>
                </td>
                <td>{{ item.group_id }}</td>
                <td>{{ item.message_count }}</td>
                <td>{{ item.user_count }}</td>
                <td>{{ item.local_score.toFixed(2) }}</td>
                <td><v-chip size="x-small" variant="tonal">{{ item.state }}</v-chip></td>
                <td class="cell-ellipsis" :title="item.verdict_reason || ''">{{ item.verdict_reason || '—' }}</td>
              </tr>
            </tbody>
          </v-table>
          <div v-else class="text-body-2 text-medium-emphasis">暂无候选</div>
        </v-card-text>
      </v-card>

      <!-- 新增对话框 -->
      <v-dialog v-model="addDialog" max-width="560" class="iris-dialog">
        <v-card>
          <v-card-title class="d-flex align-center">
            <v-icon icon="mdi-plus" color="primary" class="mr-2" />
            新增{{ TABLE_LABELS[activeTab] }}
          </v-card-title>
          <v-card-text>
            <v-combobox
              v-model="addForm.group_id"
              :items="groups"
              label="群 *"
              placeholder="选择或输入群 ID"
              variant="outlined"
              density="compact"
              class="mb-2"
            />
            <template v-if="activeTab === 'jargon'">
              <v-text-field v-model="addForm.term" label="词条 *" variant="outlined" density="compact" class="mb-2" />
              <v-textarea v-model="addForm.meaning" label="含义" variant="outlined" density="compact" rows="2" class="mb-2" />
              <v-text-field
                v-model="addForm.confidence"
                label="置信度（0-1，可选）"
                type="number"
                variant="outlined"
                density="compact"
                hide-details
              />
            </template>
            <template v-else-if="activeTab === 'expression_pattern'">
              <v-text-field v-model="addForm.scene" label="场景 *" variant="outlined" density="compact" class="mb-2" />
              <v-textarea v-model="addForm.expression" label="表达 *" variant="outlined" density="compact" rows="2" hide-details />
            </template>
            <template v-else>
              <v-text-field v-model="addForm.user_id" label="用户 ID *" variant="outlined" density="compact" class="mb-2" />
              <v-textarea v-model="addForm.user_text" label="用户消息 *" variant="outlined" density="compact" rows="2" class="mb-2" />
              <v-textarea v-model="addForm.bot_text" label="机器人回复 *" variant="outlined" density="compact" rows="2" hide-details />
            </template>
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn variant="text" @click="addDialog = false">取消</v-btn>
            <v-btn color="primary" variant="tonal" :loading="addSaving" @click="saveAdd">新增</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- 编辑对话框 -->
      <v-dialog v-model="editDialog" max-width="560" class="iris-dialog">
        <v-card>
          <v-card-title class="d-flex align-center">
            <v-icon icon="mdi-pencil" color="primary" class="mr-2" />
            编辑{{ TABLE_LABELS[editingTable] }}
          </v-card-title>
          <v-card-text>
            <div v-if="editingItem" class="text-caption text-medium-emphasis mb-3">
              创建于 {{ formatTime(editingItem.created_at) }}
              <template v-if="editingItem.approved_at">
                · 批准于 {{ formatTime(editingItem.approved_at) }}
              </template>
              <template v-if="editingItem.last_hit_at">
                · 最近命中 {{ formatTime(editingItem.last_hit_at) }}
              </template>
            </div>
            <template v-if="editingTable === 'jargon'">
              <v-text-field v-model="editForm.term" label="词条" variant="outlined" density="compact" class="mb-2" />
              <v-textarea v-model="editForm.meaning" label="含义" variant="outlined" density="compact" rows="2" class="mb-2" />
              <v-text-field
                v-model="editForm.confidence"
                label="置信度（0-1）"
                type="number"
                variant="outlined"
                density="compact"
                class="mb-2"
              />
            </template>
            <template v-else-if="editingTable === 'expression_pattern'">
              <v-text-field v-model="editForm.scene" label="场景" variant="outlined" density="compact" class="mb-2" />
              <v-textarea v-model="editForm.expression" label="表达" variant="outlined" density="compact" rows="2" class="mb-2" />
            </template>
            <template v-else>
              <v-textarea v-model="editForm.user_text" label="用户消息" variant="outlined" density="compact" rows="2" class="mb-2" />
              <v-textarea v-model="editForm.bot_text" label="机器人回复" variant="outlined" density="compact" rows="2" class="mb-2" />
            </template>
            <v-select
              v-model="editForm.status"
              :items="editStatusOptions"
              label="状态"
              variant="outlined"
              density="compact"
              hide-details
            />
          </v-card-text>
          <v-card-actions>
            <v-spacer />
            <v-btn variant="text" @click="editDialog = false">取消</v-btn>
            <v-btn color="primary" variant="tonal" :loading="editSaving" @click="saveEdit">保存</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>

      <!-- 确认对话框 -->
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
    </ComponentDisabled>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, onUnmounted } from 'vue'
import { useComponentState } from '@/composables/useComponentState'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import {
  getLearningList,
  getLearningGroups,
  getLearningStats,
  getJargonCandidates,
  addLearningItem,
  updateLearningItem,
  deleteLearningItems,
  setLearningStatus,
  type LearningTable,
  type LearningStats,
  type JargonItem,
  type ExpressionPatternItem,
  type FewShotItem,
  type JargonCandidateItem,
  type JargonLlmUsage
} from '@/api/learning'

const { status, error, errorType, refreshState } = useComponentState('learning')

// ============================================
// 常量与工具
// ============================================

const TABLE_LABELS: Record<LearningTable, string> = {
  jargon: '圈内暗语',
  expression_pattern: '表达模式',
  few_shot: '对话样例'
}

const STATUS_LABELS: Record<string, string> = {
  pending_review: '待审查',
  approved: '已通过',
  disabled: '已禁用',
  active: '生效中',
  dormant: '休眠'
}

const STATUS_COLORS: Record<string, string> = {
  pending_review: 'warning',
  approved: 'success',
  disabled: 'default',
  active: 'info',
  dormant: 'warning'
}

const statusLabel = (s: string): string => STATUS_LABELS[s] || s

const statusColor = (s: string): string => STATUS_COLORS[s] || 'default'

const formatTime = (ts?: number | null): string => {
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

const formatConfidence = (v?: number): string => {
  return typeof v === 'number' ? v.toFixed(2) : '—'
}

// 兼容三表字段的行类型（模板渲染用）
type LearningRow = Partial<JargonItem> &
  Partial<ExpressionPatternItem> &
  Partial<FewShotItem> & {
    id: number
    group_id: string
    status: string
    created_at: number
  }

// ============================================
// Tab 状态（每 tab 独立，切 tab 懒加载）
// ============================================

const activeTab = ref<LearningTable>('jargon')

interface TabState {
  items: LearningRow[]
  loading: boolean
  loaded: boolean
  page: number
  pageSize: number
  total: number
  groupId: string
  status: string
}

const createTabState = (): TabState => ({
  items: [],
  loading: false,
  loaded: false,
  page: 1,
  pageSize: 20,
  total: 0,
  groupId: '',
  status: ''
})

const tabStates = reactive<Record<LearningTable, TabState>>({
  jargon: createTabState(),
  expression_pattern: createTabState(),
  few_shot: createTabState()
})

// ============================================
// 筛选选项
// ============================================

const groups = ref<string[]>([])

const groupSelectItems = computed(() => [
  { title: '全部', value: '' },
  ...groups.value.map(g => ({ title: g, value: g }))
])

const commonStatusOptions = [
  { title: '全部', value: '' },
  { title: '待审查', value: 'pending_review' },
  { title: '已通过', value: 'approved' },
  { title: '已禁用', value: 'disabled' }
]

const jargonStatusOptions = [
  { title: '全部', value: '' },
  { title: '生效中', value: 'active' },
  { title: '休眠', value: 'dormant' },
  { title: '已禁用', value: 'disabled' }
]

// ============================================
// 表头
// ============================================

const jargonHeaders = [
  { title: '词条', key: 'term', minWidth: '140px', sortable: false },
  { title: '群', key: 'group_id', width: '110px', sortable: false },
  { title: '证据数', key: 'evidence_count', width: '90px', sortable: false },
  { title: '含义', key: 'meaning', minWidth: '200px', sortable: false },
  { title: '置信度', key: 'confidence', width: '90px', sortable: false },
  { title: '状态', key: 'status', width: '100px', sortable: false },
  { title: '操作', key: 'actions', width: '250px', sortable: false }
]

const patternHeaders = [
  { title: '场景', key: 'scene', minWidth: '140px', sortable: false },
  { title: '表达', key: 'expression', minWidth: '220px', sortable: false },
  { title: '群', key: 'group_id', width: '110px', sortable: false },
  { title: '命中数', key: 'hit_count', width: '90px', sortable: false },
  { title: '状态', key: 'status', width: '100px', sortable: false },
  { title: '操作', key: 'actions', width: '220px', sortable: false }
]

const fewshotHeaders = [
  { title: '用户消息', key: 'user_text', minWidth: '200px', sortable: false },
  { title: '机器人回复', key: 'bot_text', minWidth: '200px', sortable: false },
  { title: '用户', key: 'user_id', width: '110px', sortable: false },
  { title: '群', key: 'group_id', width: '110px', sortable: false },
  { title: '状态', key: 'status', width: '100px', sortable: false },
  { title: '操作', key: 'actions', width: '220px', sortable: false }
]

// ============================================
// 统计
// ============================================

const stats = ref<LearningStats | null>(null)
const jargonCandidates = ref<JargonCandidateItem[]>([])
const jargonLlmUsage = ref<JargonLlmUsage>({ day: '', call_count: 0, candidate_count: 0 })
const statsLoading = ref(false)

const statCards = computed(() => {
  const mk = (table: LearningTable, label: string, icon: string) => {
    const ts = stats.value ? stats.value[table] : undefined
    const by = ts?.by_status || {}
    return {
      table,
      label,
      icon,
      total: ts?.total ?? 0,
      pending: table === 'jargon'
        ? ((stats.value?.jargon_candidate?.by_status?.collecting ?? 0)
          + (stats.value?.jargon_candidate?.by_status?.queued ?? 0)
          + (stats.value?.jargon_candidate?.by_status?.deferred ?? 0))
        : (by.pending_review ?? 0),
      // 暗语无 approved 状态，其"已通过"一栏显示生效中（active）数量
      approved: table === 'jargon' ? (by.active ?? 0) : (by.approved ?? 0),
      disabled: by.disabled ?? 0
    }
  }
  return [
    mk('jargon', '圈内暗语', 'mdi-tag'),
    mk('expression_pattern', '表达模式', 'mdi-emoticon'),
    mk('few_shot', '对话样例', 'mdi-message-text-outline')
  ]
})

const loadStats = async () => {
  statsLoading.value = true
  try {
    const result = await getLearningStats()
    stats.value = result.stats
    jargonLlmUsage.value = result.jargon_llm_usage
    const candidates = await getJargonCandidates(20)
    jargonCandidates.value = candidates.items
  } catch (e: unknown) {
    notify((e as Error).message || '获取学习统计失败', 'error')
  } finally {
    statsLoading.value = false
  }
}

const loadGroups = async () => {
  try {
    groups.value = await getLearningGroups()
  } catch (e: unknown) {
    notify((e as Error).message || '获取群组列表失败', 'error')
  }
}

// ============================================
// Snackbar 与确认对话框
// ============================================

const showSnackbar = ref(false)
const snackbarText = ref('')
const snackbarColor = ref('success')

const notify = (text: string, color: string = 'success') => {
  snackbarText.value = text
  snackbarColor.value = color
  showSnackbar.value = true
}

const confirmDialog = ref(false)
const confirmMessage = ref('')
const confirmCallback = ref<(() => void) | null>(null)

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

// ============================================
// 列表加载
// ============================================

const loadList = async (table: LearningTable) => {
  const st = tabStates[table]
  st.loading = true
  try {
    const result = await getLearningList({
      table,
      group_id: st.groupId || undefined,
      status: st.status || undefined,
      page: st.page,
      page_size: st.pageSize
    })
    st.items = result.items as LearningRow[]
    st.total = result.total
    st.loaded = true
    if (result.items.length === 0 && st.page > 1) {
      st.page -= 1
      await loadList(table)
    }
  } catch (e: unknown) {
    notify((e as Error).message || '加载数据失败', 'error')
  } finally {
    st.loading = false
  }
}

const applyFilter = (table: LearningTable) => {
  const st = tabStates[table]
  st.page = 1
  loadList(table)
}

const handleOptions = (table: LearningTable, options: { page: number; itemsPerPage: number }) => {
  if (activeTab.value !== table) return
  const st = tabStates[table]
  const page = options.page > 0 ? options.page : 1
  const pageSize = options.itemsPerPage > 0 ? options.itemsPerPage : st.pageSize
  if (st.page === page && st.pageSize === pageSize) {
    if (!st.loaded && !st.loading) loadList(table)
    return
  }
  st.page = page
  st.pageSize = pageSize
  loadList(table)
}

const refreshAfterChange = async (table: LearningTable) => {
  await Promise.all([loadStats(), loadList(table)])
}

// ============================================
// 行操作：状态快捷按钮 / 删除
// ============================================

const quickStatus = async (table: LearningTable, item: LearningRow, newStatus: string) => {
  try {
    await setLearningStatus(table, [item.id], newStatus)
    notify(`已将状态更新为「${statusLabel(newStatus)}」`)
    await refreshAfterChange(table)
  } catch (e: unknown) {
    notify((e as Error).message || '更新状态失败', 'error')
  }
}

const itemTitle = (item: LearningRow): string => {
  const raw = item.term || item.scene || item.user_text || String(item.id)
  return raw.length > 24 ? raw.slice(0, 24) + '…' : raw
}

const confirmDelete = (table: LearningTable, item: LearningRow) => {
  showConfirm(`确认要删除${TABLE_LABELS[table]}「${itemTitle(item)}」吗？此操作不可逆。`, async () => {
    try {
      await deleteLearningItems(table, [item.id])
      notify('删除成功')
      await refreshAfterChange(table)
    } catch (e: unknown) {
      notify((e as Error).message || '删除失败', 'error')
    }
  })
}

// ============================================
// 编辑
// ============================================

const editDialog = ref(false)
const editSaving = ref(false)
const editingTable = ref<LearningTable>('jargon')
const editingItem = ref<LearningRow | null>(null)

const editForm = reactive({
  term: '',
  meaning: '',
  confidence: '',
  status: '',
  scene: '',
  expression: '',
  user_text: '',
  bot_text: ''
})

const editStatusOptions = computed(() => {
  if (editingTable.value === 'jargon') {
    // 暗语无审查语义，仅 生效中/已禁用
    return [
      { title: '生效中', value: 'active' },
      { title: '休眠', value: 'dormant' },
      { title: '已禁用', value: 'disabled' }
    ]
  }
  return [
    { title: '待审查', value: 'pending_review' },
    { title: '已通过', value: 'approved' },
    { title: '已禁用', value: 'disabled' }
  ]
})

const openEdit = (table: LearningTable, item: LearningRow) => {
  editingTable.value = table
  editingItem.value = item
  editForm.status = item.status
  editForm.term = item.term || ''
  editForm.meaning = item.meaning || ''
  editForm.confidence = typeof item.confidence === 'number' ? String(item.confidence) : ''
  editForm.scene = item.scene || ''
  editForm.expression = item.expression || ''
  editForm.user_text = item.user_text || ''
  editForm.bot_text = item.bot_text || ''
  editDialog.value = true
}

const saveEdit = async () => {
  const table = editingTable.value
  const item = editingItem.value
  if (!item) return

  const fields: Record<string, unknown> = { status: editForm.status }
  if (table === 'jargon') {
    if (!editForm.term.trim()) {
      notify('词条不能为空', 'error')
      return
    }
    fields.term = editForm.term.trim()
    fields.meaning = editForm.meaning.trim()
    const confidence = parseFloat(editForm.confidence)
    if (!isNaN(confidence)) fields.confidence = confidence
  } else if (table === 'expression_pattern') {
    if (!editForm.scene.trim() || !editForm.expression.trim()) {
      notify('场景和表达不能为空', 'error')
      return
    }
    fields.scene = editForm.scene.trim()
    fields.expression = editForm.expression.trim()
  } else {
    if (!editForm.user_text.trim() || !editForm.bot_text.trim()) {
      notify('用户消息和机器人回复不能为空', 'error')
      return
    }
    fields.user_text = editForm.user_text.trim()
    fields.bot_text = editForm.bot_text.trim()
  }

  editSaving.value = true
  try {
    await updateLearningItem(table, item.id, fields)
    notify('更新成功')
    editDialog.value = false
    await refreshAfterChange(table)
  } catch (e: unknown) {
    notify((e as Error).message || '更新失败', 'error')
  } finally {
    editSaving.value = false
  }
}

// ============================================
// 新增
// ============================================

const addDialog = ref(false)
const addSaving = ref(false)

const addForm = reactive({
  group_id: null as string | null,
  term: '',
  meaning: '',
  confidence: '',
  scene: '',
  expression: '',
  user_id: '',
  user_text: '',
  bot_text: ''
})

const openAdd = () => {
  addForm.group_id = null
  addForm.term = ''
  addForm.meaning = ''
  addForm.confidence = ''
  addForm.scene = ''
  addForm.expression = ''
  addForm.user_id = ''
  addForm.user_text = ''
  addForm.bot_text = ''
  addDialog.value = true
}

const saveAdd = async () => {
  const table = activeTab.value
  const groupId = (addForm.group_id || '').trim()
  if (!groupId) {
    notify('请选择或输入群 ID', 'error')
    return
  }

  const fields: Record<string, unknown> = { group_id: groupId }
  if (table === 'jargon') {
    if (!addForm.term.trim()) {
      notify('词条不能为空', 'error')
      return
    }
    fields.term = addForm.term.trim()
    if (addForm.meaning.trim()) fields.meaning = addForm.meaning.trim()
    const confidence = parseFloat(addForm.confidence)
    if (!isNaN(confidence)) fields.confidence = confidence
  } else if (table === 'expression_pattern') {
    if (!addForm.scene.trim() || !addForm.expression.trim()) {
      notify('场景和表达不能为空', 'error')
      return
    }
    fields.scene = addForm.scene.trim()
    fields.expression = addForm.expression.trim()
  } else {
    if (!addForm.user_id.trim() || !addForm.user_text.trim() || !addForm.bot_text.trim()) {
      notify('用户 ID、用户消息和机器人回复均不能为空', 'error')
      return
    }
    fields.user_id = addForm.user_id.trim()
    fields.user_text = addForm.user_text.trim()
    fields.bot_text = addForm.bot_text.trim()
  }

  addSaving.value = true
  try {
    await addLearningItem(table, fields)
    notify('新增成功')
    addDialog.value = false
    await refreshAfterChange(table)
  } catch (e: unknown) {
    notify((e as Error).message || '新增失败', 'error')
  } finally {
    addSaving.value = false
  }
}

// ============================================
// 刷新与生命周期
// ============================================

const refreshing = ref(false)

const handleManualRefresh = async () => {
  refreshing.value = true
  try {
    await Promise.all([loadStats(), loadGroups(), loadList(activeTab.value)])
  } finally {
    refreshing.value = false
  }
}

const handleRefresh = () => {
  loadStats()
  loadGroups()
  loadList(activeTab.value)
}

watch(activeTab, (table) => {
  const st = tabStates[table]
  if (!st.loaded && !st.loading) loadList(table)
})

onMounted(() => {
  loadStats()
  loadGroups()
  const st = tabStates[activeTab.value]
  if (!st.loaded && !st.loading) loadList(activeTab.value)
  window.addEventListener('iris:refresh', handleRefresh)
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.filter-select {
  max-width: 180px;
}

.cell-ellipsis {
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
