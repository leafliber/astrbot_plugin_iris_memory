import { ref, onMounted } from 'vue'
import { getIsolationStatus, type IsolationStatus } from '@/api/stats'

const DEFAULT_STATUS: IsolationStatus = {
  enable_group_memory_isolation: false,
  enable_group_isolation: false,
  enable_persona_isolation: false,
}

// 模块级缓存：多个页面共享同一次请求结果
let _cached: IsolationStatus | null = null
let _promise: Promise<IsolationStatus> | null = null

async function loadIsolation(): Promise<IsolationStatus> {
  if (_cached) return _cached
  if (_promise) return _promise
  _promise = getIsolationStatus()
    .then((s) => {
      _cached = s
      return s
    })
    .catch((e) => {
      console.error('获取隔离状态失败:', e)
      return DEFAULT_STATUS
    })
    .finally(() => {
      _promise = null
    })
  return _promise
}

export function useIsolationStatus() {
  const status = ref<IsolationStatus>({ ...DEFAULT_STATUS })
  const loading = ref(false)

  const refresh = async () => {
    loading.value = true
    try {
      _cached = null
      status.value = await loadIsolation()
    } finally {
      loading.value = false
    }
  }

  onMounted(async () => {
    loading.value = true
    try {
      status.value = await loadIsolation()
    } finally {
      loading.value = false
    }
  })

  return { status, loading, refresh }
}
