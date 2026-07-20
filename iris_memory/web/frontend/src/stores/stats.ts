import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { MemoryStats, TokenStatsResponse, KGStats, SystemStats, ComponentState, ComponentStatus } from '@/types'
import { getAllStats, getMemoryStats, getTokenStats, getKGStats, getSystemStats } from '@/api/stats'

export const useStatsStore = defineStore('stats', () => {
  const loading = ref(false)

  const memoryStats = ref<MemoryStats | null>(null)

  const tokenStats = ref<TokenStatsResponse | null>(null)

  const kgStats = ref<KGStats | null>(null)

  const systemStats = ref<SystemStats | null>(null)

  const componentStates = computed(() => systemStats.value?.components || {})

  const globalStatus = computed(() => systemStats.value?.global_status || 'pending')

  const getComponentState = (name: string): ComponentState => {
    return componentStates.value[name] || { status: 'pending', error: null, error_type: null }
  }

  const isComponentAvailable = (name: string): boolean => {
    return getComponentState(name).status === 'available'
  }

  const isComponentLoading = (name: string): boolean => {
    const status = getComponentState(name).status
    return status === 'pending' || status === 'initializing'
  }

  const isSystemReady = computed(() => globalStatus.value === 'available')

  const fetchAllStats = async () => {
    loading.value = true
    try {
      const data = await getAllStats()
      memoryStats.value = data.memory
      tokenStats.value = data.token
      kgStats.value = data.kg
      systemStats.value = data.system
    } catch (error) {
      console.error('获取统计数据失败:', error)
    } finally {
      loading.value = false
    }
  }

  const fetchMemoryStats = async () => {
    try {
      memoryStats.value = await getMemoryStats()
    } catch (error) {
      console.error('获取记忆统计失败:', error)
    }
  }

  const fetchTokenStats = async () => {
    try {
      tokenStats.value = await getTokenStats()
    } catch (error) {
      console.error('获取Token统计失败:', error)
    }
  }

  const fetchKGStats = async () => {
    try {
      kgStats.value = await getKGStats()
    } catch (error) {
      console.error('获取图谱统计失败:', error)
    }
  }

  const fetchSystemStats = async () => {
    try {
      systemStats.value = await getSystemStats()
    } catch (error) {
      console.error('获取系统统计失败:', error)
    }
  }

  return {
    loading,
    memoryStats,
    tokenStats,
    kgStats,
    systemStats,
    componentStates,
    globalStatus,
    getComponentState,
    isComponentAvailable,
    isComponentLoading,
    isSystemReady,
    fetchAllStats,
    fetchMemoryStats,
    fetchTokenStats,
    fetchKGStats,
    fetchSystemStats
  }
})
