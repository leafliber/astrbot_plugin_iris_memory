interface AstrBotPluginPageBridge {
  ready(): Promise<{ pluginName: string; displayName: string }>
  getContext(): { pluginName: string; displayName: string }
  apiGet(endpoint: string, params?: Record<string, any>): Promise<any>
  apiPost(endpoint: string, body?: any): Promise<any>
  upload(endpoint: string, file: File): Promise<any>
  download(endpoint: string, params?: Record<string, string>, filename?: string): Promise<void>
  subscribeSSE(endpoint: string, handlers: any, params?: Record<string, any>): string
  unsubscribeSSE(subscriptionId: string): void
}

interface Window {
  AstrBotPluginPage: AstrBotPluginPageBridge
}
