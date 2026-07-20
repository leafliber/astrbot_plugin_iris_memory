<template>
  <div class="l1-buffer-view">
    <ComponentDisabled
      :status="status"
      :error="error"
      :error-type="errorType"
      component-name="L1 缓冲"
      @retry="refreshState"
    >
      <v-row>
        <v-col cols="12" md="4">
          <v-card color="surface" variant="flat" class="iris-list-card">
            <v-card-title class="d-flex align-center iris-section-title">
              <span>会话列表</span>
              <v-spacer />
              <v-btn
                icon="mdi-refresh"
                variant="text"
                size="small"
                :loading="memoryStore.l1QueuesLoading"
                @click="loadL1Queues"
              />
            </v-card-title>
            <v-card-text class="pa-0">
              <v-progress-linear
                v-if="memoryStore.l1QueuesLoading"
                indeterminate
                color="primary"
              />

              <v-list v-else-if="memoryStore.l1Queues.length > 0" lines="two" class="iris-list">
                <v-list-item
                  v-for="queue in memoryStore.l1Queues"
                  :key="queue.group_id"
                  :active="selectedGroupId === queue.group_id"
                  @click="selectGroup(queue.group_id)"
                >
                  <template #prepend>
                    <v-avatar color="primary" variant="tonal">
                      <v-icon :icon="queue.is_private ? 'mdi-account' : 'mdi-account-group'" />
                    </v-avatar>
                  </template>

                  <v-list-item-title>{{ queue.group_name || (queue.is_private ? queue.user_id : queue.group_id) || '未知会话' }}</v-list-item-title>
                  <v-list-item-subtitle>
                    <template v-if="queue.group_name">
                      <code class="text-caption">{{ queue.is_private ? queue.user_id : queue.group_id }}</code> ·
                    </template>
                    {{ queue.message_count }} 条消息 · {{ queue.total_tokens }} tokens
                  </v-list-item-subtitle>
                </v-list-item>
              </v-list>

              <div v-else class="iris-empty-state">
                <v-icon icon="mdi-inbox-outline" size="48" />
                <div class="iris-empty-state__title">暂无会话数据</div>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" md="8">
          <v-card color="surface" variant="flat" class="iris-list-card">
            <v-card-title class="d-flex align-center iris-section-title">
              <span>消息缓冲</span>
              <v-chip v-if="selectedGroupId !== null" size="small" color="primary" variant="tonal" class="ml-2">
                {{ selectedGroupName || selectedGroupId || '未知会话' }}
              </v-chip>
              <v-spacer />
              <v-btn
                v-if="selectedGroupId !== null"
                icon="mdi-delete-sweep"
                variant="text"
                size="small"
                color="error"
                :loading="clearingBuffer"
                @click="handleClearBuffer"
              />
              <v-btn
                v-if="selectedGroupId !== null"
                icon="mdi-refresh"
                variant="text"
                size="small"
                :loading="memoryStore.l1Loading"
                @click="loadL1Messages"
              />
            </v-card-title>
            <v-card-text>
              <template v-if="selectedGroupId !== null">
                <v-progress-linear
                  v-if="memoryStore.l1Loading"
                  indeterminate
                  color="primary"
                />

                <v-list v-else-if="memoryStore.l1Messages.length > 0" lines="three" class="iris-list">
                  <v-list-item
                    v-for="(msg, index) in memoryStore.l1Messages"
                    :key="index"
                    :class="getRoleClass(msg.role)"
                  >
                    <template #prepend>
                      <v-avatar :color="getRoleColor(msg.role)" variant="tonal">
                        <v-icon :icon="getRoleIcon(msg.role)" size="small" />
                      </v-avatar>
                    </template>

                    <v-list-item-title class="font-weight-medium">
                      {{ getSenderName(msg) }}
                    </v-list-item-title>

                    <v-list-item-subtitle class="text-wrap mt-1">
                      <span v-for="(segment, si) in renderContent(msg.content)" :key="si">
                        <v-chip
                          v-if="segment.type === 'image'"
                          size="x-small"
                          variant="tonal"
                          color="info"
                          class="mr-1"
                        >
                          <v-icon icon="mdi-image" start size="x-small" />
                          {{ segment.text }}
                        </v-chip>
                        <v-chip
                          v-else-if="segment.type === 'pending'"
                          size="x-small"
                          variant="tonal"
                          color="warning"
                          class="mr-1"
                        >
                          <v-icon icon="mdi-image-outline" start size="x-small" />
                          解析中
                        </v-chip>
                        <span v-else>{{ segment.text }}</span>
                      </span>
                    </v-list-item-subtitle>

                    <template #append>
                      <span class="text-caption text-medium-emphasis">
                        {{ formatTime(msg.timestamp) }}
                      </span>
                    </template>
                  </v-list-item>
                </v-list>

                <div v-else class="iris-empty-state">
                  <v-icon icon="mdi-message-outline" size="56" />
                  <div class="iris-empty-state__title">暂无缓冲消息</div>
                </div>
              </template>

              <div v-else class="iris-empty-state">
                <v-icon icon="mdi-hand-pointing-up" size="56" />
                <div class="iris-empty-state__title">请从左侧选择一个会话</div>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <v-row class="mt-4">
        <v-col cols="12">
          <v-card color="surface" variant="flat" class="iris-card">
            <v-card-title class="iris-section-title">
              <v-icon icon="mdi-information" class="mr-2" />
              L1 缓冲说明
            </v-card-title>
            <v-card-text>
              <v-alert type="info" variant="tonal" density="compact">
                <div class="text-body-2">
                  <strong>L1 缓冲（Working Memory）</strong> 是会话内的临时存储，采用 LRU 缓存策略。
                  这里显示的是当前会话中的消息列表，消息会在会话结束后被清理或转移到 L2 长期记忆。
                </div>
              </v-alert>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </ComponentDisabled>

    <v-dialog v-model="showClearDialog" max-width="400">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-alert-circle" color="warning" class="mr-2" />
          确认清空
        </v-card-title>
        <v-card-text>
          确定要清空{{ clearTarget === 'group' ? '当前会话' : '所有会话' }}的 L1 缓冲吗？此操作不可撤销。
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showClearDialog = false">取消</v-btn>
          <v-btn color="error" variant="tonal" :loading="clearingBuffer" @click="confirmClearBuffer">
            确认清空
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useMemoryStore } from '@/stores'
import { useComponentState } from '@/composables/useComponentState'
import ComponentDisabled from '@/components/ComponentDisabled.vue'
import { clearL1Buffer } from '@/api/manage'

const memoryStore = useMemoryStore()
const { status, error, errorType, refreshState } = useComponentState('l1_buffer')

const selectedGroupId = ref<string | null>(null)
const clearingBuffer = ref(false)
const showClearDialog = ref(false)
const clearTarget = ref<'group' | 'all'>('group')

const selectedGroupName = computed(() => {
  if (selectedGroupId.value === null) return ''
  const queue = memoryStore.l1Queues.find(q => q.group_id === selectedGroupId.value)
  return queue?.group_name || (queue?.is_private ? queue?.user_id : '') || ''
})

const loadL1Queues = () => {
  memoryStore.fetchL1Queues()
}

const selectGroup = (groupId: string) => {
  selectedGroupId.value = groupId
  memoryStore.fetchL1Messages(groupId)
}

const loadL1Messages = () => {
  if (selectedGroupId.value !== null) {
    memoryStore.fetchL1Messages(selectedGroupId.value)
  }
}

const handleClearBuffer = () => {
  clearTarget.value = 'group'
  showClearDialog.value = true
}

const confirmClearBuffer = async () => {
  clearingBuffer.value = true
  try {
    const groupId = clearTarget.value === 'group' ? selectedGroupId.value ?? undefined : undefined
    await clearL1Buffer(groupId)
    showClearDialog.value = false
    loadL1Queues()
    if (selectedGroupId.value !== null) {
      memoryStore.fetchL1Messages(selectedGroupId.value)
    }
  } catch (error) {
    console.error('清空缓冲失败:', error)
  } finally {
    clearingBuffer.value = false
  }
}

const getRoleClass = (role: string): string => {
  return role === 'user' ? 'border-l-primary' : role === 'assistant' ? 'border-l-secondary' : 'border-l-accent'
}

const getRoleColor = (role: string): string => {
  return role === 'user' ? 'primary' : role === 'assistant' ? 'secondary' : 'accent'
}

const getRoleIcon = (role: string): string => {
  return role === 'user' ? 'mdi-account' : role === 'assistant' ? 'mdi-robot' : 'mdi-cog'
}

const getSenderName = (msg: { role: string; user_name?: string; user_id?: string }): string => {
  if (msg.role === 'assistant') return '助手'
  if (msg.role === 'system') return '系统'
  if (msg.user_name) return msg.user_name
  if (msg.user_id) return msg.user_id
  return '用户'
}

interface ContentSegment {
  type: 'text' | 'image' | 'pending'
  text: string
}

const renderContent = (content: string): ContentSegment[] => {
  if (!content) return []
  const segments: ContentSegment[] = []
  const imgDescRe = /\[图:([^\]]*)\]/g
  const imgPendingRe = /\[IMG:[^\]]*\]/g
  const combinedRe = /(\[图:([^\]]*)\]|\[IMG:[^\]]*\])/g
  let lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = combinedRe.exec(content)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', text: content.slice(lastIndex, match.index) })
    }
    if (match[2] !== undefined) {
      segments.push({ type: 'image', text: match[2] })
    } else {
      segments.push({ type: 'pending', text: '' })
    }
    lastIndex = combinedRe.lastIndex
  }
  if (lastIndex < content.length) {
    segments.push({ type: 'text', text: content.slice(lastIndex) })
  }
  return segments
}

const formatTime = (timestamp?: string): string => {
  if (!timestamp) return ''
  try {
    const date = new Date(timestamp)
    return date.toLocaleString('zh-CN', {
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
  loadL1Queues()
}

onMounted(() => {
  loadL1Queues()
  window.addEventListener('iris:refresh', handleRefresh)
})

onUnmounted(() => {
  window.removeEventListener('iris:refresh', handleRefresh)
})
</script>

<style scoped>
.border-l-primary {
  border-left: 3px solid rgb(var(--v-theme-primary));
}
.border-l-secondary {
  border-left: 3px solid rgb(var(--v-theme-secondary));
}
.border-l-accent {
  border-left: 3px solid rgb(var(--v-theme-accent));
}
</style>
