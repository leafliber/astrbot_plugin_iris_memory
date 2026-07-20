<template>
  <v-chip
    :color="enabled ? 'success' : 'default'"
    size="small"
    variant="tonal"
    class="isolation-badge"
  >
    <v-icon
      :icon="enabled ? 'mdi-shield-check' : 'mdi-shield-off-outline'"
      start
      size="x-small"
    />
    {{ label }}
    <span class="ml-1 font-weight-bold">{{ enabled ? '开' : '关' }}</span>
    <v-tooltip activator="parent" location="bottom" max-width="320">
      <span class="text-caption">{{ tooltip }}</span>
    </v-tooltip>
  </v-chip>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  type: 'memory' | 'profile' | 'persona'
  enabled: boolean
}>()

const META: Record<
  typeof props.type,
  { label: string; tooltip: string }
> = {
  memory: {
    label: '群记忆隔离',
    tooltip:
      '开启后 L2/L3 记忆按群聊独立存储与检索；关闭后跨群共享记忆库。仅在群记忆隔离开启时，按群过滤才有实际意义。',
  },
  profile: {
    label: '群画像隔离',
    tooltip:
      '开启后用户/群画像按群聊独立维护；关闭后所有群聊共用同一份画像（以 default 标识）。',
  },
  persona: {
    label: '人设隔离',
    tooltip:
      '开启后不同群聊使用各自的人设；关闭后全局共用同一人设。',
  },
}

const label = computed(() => META[props.type].label)
const tooltip = computed(() => META[props.type].tooltip)
</script>

<style scoped>
.isolation-badge {
  cursor: help;
}
</style>
