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

/** 桥接下载的无响应超时（毫秒）：超时后降级为 apiGet 拉取 */
const BRIDGE_DOWNLOAD_TIMEOUT_MS = 20000

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((_resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} 超时（${ms}ms）`))
    }, ms)
    promise.then(
      (value) => {
        clearTimeout(timer)
        _resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error instanceof Error ? error : new Error(String(error)))
      },
    )
  })
}

/**
 * 在 iframe 内直接构造文件下载。
 *
 * 插件页运行在 Dashboard 的 iframe 中，不能直接 fetch API 端点（会绕过
 * 宿主认证），也不可用 localStorage；数据必须经桥接 apiGet 取回后，
 * 在本地构造 Blob 触发下载。
 */
function downloadBlobAsFile(data: unknown, filename: string): void {
  const text = typeof data === 'string' ? data : JSON.stringify(data, null, 2)
  const blob = new Blob([text], { type: 'application/json' })
  const blobUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = blobUrl
  anchor.download = filename || 'download.json'
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  setTimeout(() => {
    URL.revokeObjectURL(blobUrl)
  }, 0)
}

/**
 * 下载端点数据为本地文件。
 *
 * 优先走宿主桥接 download（流式 blob，支持 Content-Disposition）；
 * 宿主桥接缺失 / 报错 / 无响应超时时，降级为 apiGet 拉取 JSON 后在
 * iframe 内构造 Blob 下载。当前所有导出端点均返回纯 JSON，降级路径
 * 覆盖全部调用方。双路失败则抛错，由调用方统一提示。
 */
async function apiDownload(
  endpoint: string,
  params?: Record<string, string>,
  filename?: string,
): Promise<void> {
  await ensureReady()

  if (typeof bridge.download === 'function') {
    try {
      await withTimeout(
        bridge.download(endpoint, toPlain(params), filename),
        BRIDGE_DOWNLOAD_TIMEOUT_MS,
        '桥接下载',
      )
      return
    } catch (e) {
      console.warn(
        `[iris] 桥接下载失败，降级为 apiGet 拉取：${endpoint}`,
        e,
      )
    }
  } else {
    console.warn(
      `[iris] 当前宿主桥接不支持 download（AstrBot < 4.23.6？），` +
        `降级为 apiGet 拉取：${endpoint}`,
    )
  }

  try {
    const data = await apiGet(endpoint, params)
    downloadBlobAsFile(data, filename || 'export.json')
  } catch (e) {
    throw e instanceof Error ? e : new Error(String(e))
  }
}

async function apiUpload<T = any>(endpoint: string, file: File): Promise<T> {
  await ensureReady()
  return bridge.upload(endpoint, file)
}

export { apiGet, apiPost, apiDownload, apiUpload, ensureReady }
