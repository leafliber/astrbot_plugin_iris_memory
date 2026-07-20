import { computed, onMounted, onUnmounted } from 'vue'
import { useStatsStore } from '@/stores'
import type { ComponentState } from '@/types'

export function useComponentState(componentName: string) {
  const statsStore = useStatsStore()

  const state = computed<ComponentState>(() => {
    return statsStore.getComponentState(componentName)
  })

  const status = computed(() => state.value.status)

  const isAvailable = computed(() => status.value === 'available')

  const isLoading = computed(() => 
    status.value === 'pending' || status.value === 'initializing'
  )

  const isUnavailable = computed(() => status.value === 'unavailable')

  const error = computed(() => state.value.error)

  const errorType = computed(() => state.value.error_type)

  const refreshState = async () => {
    await statsStore.fetchSystemStats()
  }

  const handleRefresh = () => {
    refreshState()
  }

  onMounted(() => {
    if (!statsStore.systemStats) {
      statsStore.fetchSystemStats()
    }
    window.addEventListener('iris:refresh', handleRefresh)
  })

  onUnmounted(() => {
    window.removeEventListener('iris:refresh', handleRefresh)
  })

  return {
    state,
    status,
    isAvailable,
    isLoading,
    isUnavailable,
    error,
    errorType,
    refreshState
  }
}
