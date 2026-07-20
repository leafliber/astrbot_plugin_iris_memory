const bridge = window.AstrBotPluginPage as any

let readyPromise: Promise<any> | null = null

function ensureReady(): Promise<any> {
  if (!readyPromise) {
    readyPromise = bridge.ready()!
  }
  return readyPromise!
}

/**
 * 将 Vue 响应式代理（ref/reactive）转换为纯 JSON 对象。
 *
 * AstrBot 插件桥接层通过 window.postMessage 与宿主通信，结构化克隆算法
 * 无法克隆 Vue 的 Proxy 对象，会抛出 "could not be cloned" 异常。
 * 所有传入桥接层的数据必须先经过此函数剥离响应性。
 */
function toPlain<T>(value: T): T {
  if (value === undefined || value === null) {
    return value
  }
  return JSON.parse(JSON.stringify(value))
}

async function apiGet<T = any>(endpoint: string, params?: Record<string, any>): Promise<T> {
  await ensureReady()
  return bridge.apiGet(endpoint, toPlain(params))
}

async function apiPost<T = any>(endpoint: string, body?: any): Promise<T> {
  await ensureReady()
  return bridge.apiPost(endpoint, toPlain(body))
}

async function apiDownload(endpoint: string, params?: Record<string, string>, filename?: string): Promise<void> {
  await ensureReady()
  return bridge.download(endpoint, params, filename)
}

async function apiUpload<T = any>(endpoint: string, file: File): Promise<T> {
  await ensureReady()
  return bridge.upload(endpoint, file)
}

export { apiGet, apiPost, apiDownload, apiUpload, ensureReady }
