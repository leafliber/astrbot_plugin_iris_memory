import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import { initSync, parse } from 'es-module-lexer'
import { resolve, join } from 'path'
import { readdirSync, readFileSync, statSync } from 'fs'

const collectSourceTokens = (root: string): Set<string> => {
  const tokens = new Set<string>()
  const walk = (p: string) => {
    const st = statSync(p)
    if (st.isDirectory()) {
      for (const name of readdirSync(p)) walk(join(p, name))
      return
    }
    if (!/\.(vue|ts|css|html)$/.test(p)) return
    for (const m of readFileSync(p, 'utf8').matchAll(/[A-Za-z_][A-Za-z0-9_-]+/g)) {
      tokens.add(m[0])
    }
  }
  walk(root)
  return tokens
}

const collectVuetifyTokens = (root: string): Set<string> => {
  const tokens = new Set<string>()
  const re =
    /\b(?:d|flex|align|justify|ga|ma|pa|mt|mb|ml|mr|ms|me|pt|pb|pl|pr|ps|pe|text|w|h|overflow|position|rounded|border|elevation|bg|font-weight|cursor|opacity)-[A-Za-z0-9_-]+/g
  const walk = (p: string) => {
    const st = statSync(p)
    if (st.isDirectory()) {
      for (const name of readdirSync(p)) walk(join(p, name))
      return
    }
    if (!p.endsWith('.js')) return
    for (const m of readFileSync(p, 'utf8').matchAll(re)) tokens.add(m[0])
  }
  walk(root)
  return tokens
}

const splitTopLevel = (str: string, sep: string): string[] => {
  const parts: string[] = []
  let depth = 0
  let last = 0
  for (let k = 0; k < str.length; k++) {
    const ch = str[k]
    if (ch === '(' || ch === '[') depth++
    else if (ch === ')' || ch === ']') depth--
    else if (ch === sep && depth === 0) {
      parts.push(str.slice(last, k))
      last = k + 1
    }
  }
  parts.push(str.slice(last))
  return parts
}

const pruneVuetifyCss = (css: string, used: Set<string>): string => {
  const keepClass = (cls: string) =>
    used.has(cls) || /^v-|^elevation-|^rounded|^position-|^border-|transition/.test(cls)

  const prune = (input: string): string => {
    let out = ''
    let i = 0
    const n = input.length
    while (i < n) {
      if (input.startsWith('/*', i)) {
        const end = input.indexOf('*/', i + 2)
        const j = end === -1 ? n : end + 2
        out += input.slice(i, j)
        i = j
        continue
      }
      const brace = input.indexOf('{', i)
      const semi = input.indexOf(';', i)
      if (brace === -1 || (semi !== -1 && semi < brace)) {
        const j = semi === -1 ? n : semi + 1
        out += input.slice(i, j)
        i = j
        continue
      }
      const prelude = input.slice(i, brace)
      let depth = 1
      let j = brace + 1
      while (j < n && depth > 0) {
        if (input[j] === '/' && input[j + 1] === '*') {
          const end = input.indexOf('*/', j + 2)
          j = end === -1 ? n : end + 2
          continue
        }
        if (input[j] === '{') depth++
        else if (input[j] === '}') depth--
        j++
      }
      const block = input.slice(brace + 1, j - 1)
      const head = prelude.trim()
      if (head.startsWith('@')) {
        const name = head.split(/[\s(]/)[0]
        if (name === '@media' || name === '@supports' || name === '@layer' || name === '@container') {
          const inner = prune(block)
          if (inner.trim()) out += prelude + '{' + inner + '}'
        } else {
          out += input.slice(i, j)
        }
      } else {
        const selectors = splitTopLevel(prelude, ',')
          .map(s => s.trim())
          .filter(Boolean)
        const allDroppable =
          selectors.length > 0 &&
          selectors.every(sel => {
            const m = sel.match(/^\.[A-Za-z_][A-Za-z0-9_-]*$/)
            return m !== null && !keepClass(m[0].slice(1))
          })
        if (!allDroppable) out += input.slice(i, j)
      }
      i = j
    }
    return out
  }

  return prune(css)
}

const vuetifyCssPrune = (): Plugin => {
  let used: Set<string> | null = null
  return {
    name: 'iris:vuetify-css-prune',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('vuetify/lib/styles/main.css')) return null
      if (!used) {
        used = collectSourceTokens(resolve(import.meta.dirname, 'src'))
        for (const t of collectSourceTokens(resolve(import.meta.dirname, 'index.html'))) used.add(t)
        for (const t of collectVuetifyTokens(resolve(import.meta.dirname, 'node_modules/vuetify/lib'))) {
          used.add(t)
        }
      }
      return { code: pruneVuetifyCss(code, used), map: null }
    }
  }
}

/**
 * AstrBot 会在下发插件 Page 的 JS 时重写相对 import，并给资源 URL 追加
 * asset_token。其扫描器要求 import/export/from 与相邻语法之间存在空白；
 * Terser 默认生成的 `import{...}from"./chunk.js"` 无法被识别。
 *
 * 这里利用模块词法分析结果只调整真实静态 import/export 声明，不触碰字符串、
 * 正则或业务代码。动态 import() 本身已符合 AstrBot 的扫描规则。
 */
const astrBotPluginPageImportCompat = (): Plugin => {
  initSync()
  return {
    name: 'iris:astrbot-plugin-page-import-compat',
    enforce: 'post',
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue
        const [imports] = parse(output.code)
        let code = output.code
        for (const imported of [...imports].sort((a, b) => b.ss - a.ss)) {
          if (
            imported.d !== -1 ||
            !imported.n ||
            !(imported.n.startsWith('./') || imported.n.startsWith('../') || imported.n.startsWith('/'))
          ) {
            continue
          }
          const prefix = code.slice(imported.ss, imported.s)
          const compatiblePrefix = prefix
            .replace(/^(import|export)(?=[{*])/, '$1 ')
            .replace(/\bfrom(?=["']$)/, ' from ')
            .replace(/^import(?=["']$)/, 'import ')
          const isAstrBotRewritable =
            /^(?:import|export)\s+[\s\S]*\s+from\s+["']$/.test(compatiblePrefix) ||
            /^import\s+["']$/.test(compatiblePrefix)
          if (!isAstrBotRewritable) {
            throw new Error(`无法生成 AstrBot 可重写的模块导入：${imported.n}`)
          }
          code = code.slice(0, imported.ss) + compatiblePrefix + code.slice(imported.s)
        }
        output.code = code
      }
    }
  }
}

export default defineConfig({
  plugins: [
    vue(),
    vuetify({ autoImport: true }),
    vuetifyCssPrune(),
    astrBotPluginPageImportCompat()
  ],
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, 'src')
    }
  },
  base: './',
  esbuild: {
    drop: ['console', 'debugger'],
    legalComments: 'none'
  },
  build: {
    target: 'es2020',
    // modulepreload 由运行时代码拼接 URL，无法继承插件 Page 的 asset_token。
    modulePreload: false,
    outDir: resolve(import.meta.dirname, '../../../pages/iris'),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 600,
    cssCodeSplit: false,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        passes: 3,
        pure_funcs: ['console.log']
      },
      mangle: {
        safari10: false
      },
      format: {
        comments: false,
        ecma: 2020
      }
    },
    rollupOptions: {
      output: {
        entryFileNames: 'iris.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'iris.[ext]',
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('cytoscape')) return 'graph-vendor'
          if (id.includes('vuetify')) return 'vuetify-vendor'
          if (
            id.includes('/node_modules/vue/') ||
            id.includes('/node_modules/vue-router/') ||
            id.includes('/node_modules/pinia/')
          ) {
            return 'vue-vendor'
          }
          return 'vendor'
        }
      }
    }
  }
})
