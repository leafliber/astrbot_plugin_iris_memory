import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getUIPreferences, updateUIPreferences } from '@/api'

export const useAppStore = defineStore('app', () => {
  // 加载状态
  const loading = ref(false)
  const error = ref<string | null>(null)

  // 选中的群聊
  const selectedGroupId = ref<string | null>(null)

  // 主题（默认夜间模式，initTheme 从后端加载实际偏好）
  const darkMode = ref(true)
  const themeLoaded = ref(false)

  // 设置加载状态
  const setLoading = (value: boolean) => {
    loading.value = value
  }

  // 设置错误
  const setError = (msg: string | null) => {
    error.value = msg
  }

  // 清除错误
  const clearError = () => {
    error.value = null
  }

  // 选中群聊
  const selectGroup = (groupId: string | null) => {
    selectedGroupId.value = groupId
  }

  // 从后端加载主题偏好
  // 背景：AstrBot 插件页面以 iframe 嵌入 Dashboard，localStorage 不可用，
  // 因此 UI 偏好存储在后端 JSON 文件中。
  const initTheme = async () => {
    if (themeLoaded.value) return
    try {
      const prefs = await getUIPreferences()
      darkMode.value = prefs.dark_mode
    } catch {
      // 加载失败时保持默认值（夜间模式）
    } finally {
      themeLoaded.value = true
    }
  }

  // 切换主题并持久化到后端
  const toggleTheme = async () => {
    darkMode.value = !darkMode.value
    try {
      await updateUIPreferences({ dark_mode: darkMode.value })
    } catch {
      // 持久化失败时仅影响当前会话，不回滚 UI 状态
    }
  }

  return {
    loading,
    error,
    selectedGroupId,
    darkMode,
    themeLoaded,
    setLoading,
    setError,
    clearError,
    selectGroup,
    initTheme,
    toggleTheme
  }
})
