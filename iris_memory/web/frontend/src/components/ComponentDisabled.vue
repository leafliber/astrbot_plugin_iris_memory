<template>
  <div class="component-disabled-overlay" v-if="showOverlay">
    <div class="disabled-content text-center">
      <v-icon :icon="statusIcon" :color="statusColor" size="64" />
      <div class="text-h6 mt-4">{{ title }}</div>
      <div class="text-body-2 text-medium-emphasis mt-2">{{ message }}</div>
      <v-btn
        v-if="showRetry"
        color="primary"
        variant="tonal"
        class="mt-4"
        @click="$emit('retry')"
      >
        重试
      </v-btn>
    </div>
  </div>
  <div v-else class="h-100">
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ComponentStatus, ErrorType } from '@/types'
import { ERROR_TYPE_DISPLAY_NAMES, STATUS_DISPLAY_NAMES } from '@/types'

interface Props {
  status: ComponentStatus
  error?: string | null
  errorType?: ErrorType | null
  componentName: string
  showRetry?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  error: null,
  errorType: null,
  showRetry: false
})

defineEmits<{
  retry: []
}>()

const showOverlay = computed(() => {
  return props.status !== 'available'
})

const statusIcon = computed(() => {
  switch (props.status) {
    case 'pending':
      return 'mdi-clock-outline'
    case 'initializing':
      return 'mdi-loading'
    case 'unavailable':
      return 'mdi-alert-circle'
    default:
      return 'mdi-help-circle'
  }
})

const statusColor = computed(() => {
  switch (props.status) {
    case 'pending':
    case 'initializing':
      return 'warning'
    case 'unavailable':
      return 'error'
    default:
      return 'grey'
  }
})

const title = computed(() => {
  return `${props.componentName} ${STATUS_DISPLAY_NAMES[props.status] || '不可用'}`
})

const message = computed(() => {
  switch (props.status) {
    case 'pending':
      return '正在等待初始化...'
    case 'initializing':
      return '组件正在初始化中，请稍候...'
    case 'unavailable':
      if (props.errorType) {
        return ERROR_TYPE_DISPLAY_NAMES[props.errorType] || props.error || '未知错误'
      }
      return props.error || '组件不可用'
    default:
      return props.error || '组件不可用'
  }
})
</script>

<style scoped>
.component-disabled-overlay {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  background: linear-gradient(
    180deg,
    rgba(var(--v-theme-surface), 0.4),
    rgba(var(--v-theme-surface), 0.7)
  );
  border-radius: 12px;
  border: 1px dashed rgba(var(--v-theme-on-surface), 0.12);
}

.disabled-content {
  padding: 40px 32px;
  max-width: 420px;
  animation: iris-fade-in 0.3s ease;
}

.disabled-content :deep(.v-icon) {
  filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.1));
}

.disabled-content :deep(.text-h6) {
  font-weight: 700;
  letter-spacing: 0.01em;
}

@keyframes iris-fade-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
