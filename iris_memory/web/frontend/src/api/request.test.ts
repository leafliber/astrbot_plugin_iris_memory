/** apiDownload 降级链路测试(桥接 download 缺失/失败/超时 → apiGet 拉取) */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface BridgeMock {
  ready: ReturnType<typeof vi.fn>
  apiGet: ReturnType<typeof vi.fn>
  download?: ReturnType<typeof vi.fn>
  clickSpy?: ReturnType<typeof vi.fn>
}

let clickSpy: ReturnType<typeof vi.fn>
let downloadAttr: { value?: string }

function makeBridge(overrides: Partial<BridgeMock> = {}) {
  return {
    ready: vi.fn().mockResolvedValue({ pluginName: 'iris' }),
    apiGet: vi.fn().mockResolvedValue({ version: '1.0', nodes: [] }),
    ...overrides,
  }
}

/** 安装桥接桩后动态导入模块(bridge 在模块顶层被捕获) */
async function importRequest(bridgeMock: unknown) {
  ;(globalThis as any).window = { AstrBotPluginPage: bridgeMock }

  clickSpy = vi.fn()
  downloadAttr = {}
  const anchor = {
    set href(v: string) {
      downloadAttr.value = v
    },
    get href() {
      return downloadAttr.value || ''
    },
    set download(v: string) {
      ;(anchor as any)._download = v
    },
    get download() {
      return (anchor as any)._download
    },
    click: clickSpy,
    remove: vi.fn(),
  }
  ;(globalThis as any).document = {
    createElement: vi.fn(() => anchor),
    body: { appendChild: vi.fn() },
  }
  ;(globalThis as any).URL = Object.assign(
    Object.getPrototypeOf(URL) === null ? URL : URL,
    {
      createObjectURL: vi.fn().mockReturnValue('blob:mock'),
      revokeObjectURL: vi.fn(),
    },
  )

  const mod = await import('./request')
  return mod
}

describe('apiDownload 降级链路', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    delete (globalThis as any).window
    delete (globalThis as any).document
  })

  it('桥接 download 可用时优先走桥接,不调用 apiGet', async () => {
    const bridge = makeBridge({ download: vi.fn().mockResolvedValue(undefined) })
    const { apiDownload } = await importRequest(bridge)

    await apiDownload('data/l3/export', {}, 'iris_l3_kg.json')

    expect(bridge.download).toHaveBeenCalledWith(
      'data/l3/export',
      {},
      'iris_l3_kg.json',
    )
    expect(bridge.apiGet).not.toHaveBeenCalled()
    expect(clickSpy).not.toHaveBeenCalled()
  })

  it('桥接 download 缺失时降级为 apiGet + Blob 下载', async () => {
    const bridge = makeBridge() // 无 download 方法
    const { apiDownload } = await importRequest(bridge)

    await apiDownload('data/l3/export', {}, 'iris_l3_kg.json')

    expect(bridge.apiGet).toHaveBeenCalledWith('data/l3/export', {})
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('桥接 download 抛错时降级', async () => {
    const bridge = makeBridge({
      download: vi.fn().mockRejectedValue(new Error('host error')),
    })
    const { apiDownload } = await importRequest(bridge)

    await apiDownload('data/l2/export', {}, 'iris_l2.json')

    expect(bridge.apiGet).toHaveBeenCalledWith('data/l2/export', {})
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('桥接 download 无响应超时后降级', async () => {
    const bridge = makeBridge({
      download: vi.fn().mockImplementation(() => new Promise(() => {})),
    })
    const { apiDownload } = await importRequest(bridge)

    const pending = apiDownload('data/profile/export', {}, 'iris_p.json')
    // 推进虚拟时钟越过 20s 桥接超时
    await vi.advanceTimersByTimeAsync(21000)
    await pending

    expect(bridge.apiGet).toHaveBeenCalledWith('data/profile/export', {})
    expect(clickSpy).toHaveBeenCalledOnce()
  })

  it('双路失败时抛错,交由调用方提示', async () => {
    const bridge = makeBridge({
      download: vi.fn().mockRejectedValue(new Error('host error')),
    })
    bridge.apiGet = vi.fn().mockRejectedValue(new Error('apiGet error'))
    const { apiDownload } = await importRequest(bridge)

    await expect(
      apiDownload('data/all/export', {}, 'iris_full.json'),
    ).rejects.toThrow('apiGet error')
  })
})
