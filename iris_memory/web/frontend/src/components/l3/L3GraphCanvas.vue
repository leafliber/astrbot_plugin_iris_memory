<template>
  <div class="canvas-wrapper">
    <!-- 顶部工具栏 -->
    <div class="canvas-toolbar">
      <div class="toolbar-left">
        <v-btn-group density="compact" variant="tonal">
          <v-btn icon="mdi-undo-variant" size="small" :disabled="!canGoBack" @click="emit('nav-back')">
            <v-tooltip activator="parent" location="bottom">后退</v-tooltip>
          </v-btn>
          <v-btn icon="mdi-redo-variant" size="small" :disabled="!canGoForward" @click="emit('nav-forward')">
            <v-tooltip activator="parent" location="bottom">前进</v-tooltip>
          </v-btn>
        </v-btn-group>

        <v-btn-group density="compact" variant="tonal" class="ml-2">
          <v-btn icon="mdi-magnify-plus" size="small" @click="zoomBy(1.25)" />
          <v-btn icon="mdi-magnify-minus" size="small" @click="zoomBy(0.8)" />
          <v-btn icon="mdi-fit-to-screen" size="small" @click="fitView" />
          <v-btn icon="mdi-image-filter-center-focus" size="small" @click="fitCenter" />
        </v-btn-group>

        <v-chip v-if="startNode" size="small" color="accent" variant="tonal" class="ml-2">
          <v-icon :icon="getNodeIcon(startNode.label)" start size="small" />
          {{ startNode.name }}
        </v-chip>
      </div>

      <div class="toolbar-right">
        <v-chip size="small" variant="text">
          <v-icon icon="mdi-circle-multiple" start size="small" color="primary" />
          {{ nodes.length }}
        </v-chip>
        <v-chip size="small" variant="text">
          <v-icon icon="mdi-arrow-right-bold" start size="small" color="secondary" />
          {{ edges.length }}
        </v-chip>
      </div>
    </div>

    <!-- G6 画布容器 -->
    <div ref="containerRef" class="graph-container">
      <div v-if="loading" class="overlay">
        <v-progress-circular indeterminate color="primary" size="56" width="4" />
        <div class="text-caption mt-3 text-medium-emphasis">加载图谱中…</div>
      </div>
      <div v-else-if="nodes.length === 0" class="overlay">
        <v-icon icon="mdi-graph-outline" size="72" class="mb-3 text-medium-emphasis" />
        <div class="text-h6 text-medium-emphasis">暂无图谱数据</div>
        <div class="text-body-2 text-medium-emphasis mt-1">L3 知识图谱为空或未启用</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, shallowRef, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { Graph } from '@antv/g6'
import type { GraphData, NodeData, EdgeData } from '@antv/g6'
import type { KGNode, KGEdge, L3LayoutType } from '@/types'
import {
  getNodeIcon,
  getTypeColor,
  getNodeLabel,
  getRelationLabel,
  resolveThemeColor,
} from '@/composables/l3Constants'

const props = defineProps<{
  nodes: KGNode[]
  edges: KGEdge[]
  loading: boolean
  startNode: KGNode | null
  layout: L3LayoutType
  canGoBack: boolean
  canGoForward: boolean
}>()

const emit = defineEmits<{
  'node-click': [node: KGNode]
  'node-dblclick': [nodeId: string]
  'edge-click': [edge: KGEdge]
  'nav-back': []
  'nav-forward': []
}>()

const containerRef = ref<HTMLElement | null>(null)
const graphRef = shallowRef<Graph | null>(null)
let resizeObserver: ResizeObserver | null = null

// ---- 主题色缓存（canvas 无法消费 CSS 变量）----
const colorCache = new Map<string, { fill: string; stroke: string; light: string }>()
const nodeColors = (label: string) => {
  if (!colorCache.has(label)) {
    const base = resolveThemeColor(getTypeColor(label), '#5c6bc0')
    // 解析 rgb 值用于生成透明度变体
    const match = base.match(/rgb\(([^)]+)\)/)
    if (match) {
      const [r, g, b] = match[1].split(',').map((s) => parseInt(s.trim(), 10))
      colorCache.set(label, {
        fill: `rgb(${r}, ${g}, ${b})`,
        stroke: `rgb(${Math.max(0, r - 40)}, ${Math.max(0, g - 40)}, ${Math.max(0, b - 40)})`,
        light: `rgba(${r}, ${g}, ${b}, 0.15)`,
      })
    } else {
      colorCache.set(label, { fill: base, stroke: base, light: base + '26' })
    }
  }
  return colorCache.get(label)!
}

// ---- 判断是否深色主题 ----
const isDarkTheme = (): boolean => {
  try {
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--v-theme-surface').trim()
    if (bg) {
      const [r, g, b] = bg.split(',').map((s) => parseInt(s.trim(), 10))
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
      return luminance < 0.5
    }
  } catch {
    // ignore
  }
  return false
}

// ---- 计算节点度数 ----
const computeDegrees = (): Map<string, number> => {
  const deg = new Map<string, number>()
  props.edges.forEach((e) => {
    deg.set(e.source, (deg.get(e.source) || 0) + 1)
    deg.set(e.target, (deg.get(e.target) || 0) + 1)
  })
  return deg
}

// ---- 布局配置：d3-force 稳定无闪烁 ----
const layoutConfig = (type: L3LayoutType, nodeCount: number) => {
  // 根据节点数自适应间距
  const linkDist = nodeCount > 40 ? 100 : nodeCount > 20 ? 130 : 160
  const repulsion = nodeCount > 40 ? -120 : -180

  switch (type) {
    case 'dagre':
      return {
        type: 'dagre',
        rankdir: 'LR',
        nodesep: 30,
        ranksep: 60,
        nodeSize: 40,
      } as any
    case 'radial':
      return {
        type: 'radial',
        unitRadius: 110,
        preventOverlap: true,
        nodeSize: 40,
        linkDistance: linkDist,
      } as any
    case 'concentric':
      return {
        type: 'concentric',
        minNodeSpacing: 40,
        preventOverlap: true,
        nodeSize: 40,
      } as any
    case 'force':
    default:
      return {
        type: 'd3-force',
        animation: true,
        alpha: 1,
        alphaDecay: 0.05,
        alphaMin: 0.001,
        alphaTarget: 0,
        velocityDecay: 0.4,
        link: {
          distance: linkDist,
          strength: 0.3,
          iterations: 1,
        },
        manyBody: {
          strength: repulsion,
          theta: 0.9,
        },
        center: {
          strength: 0.06,
        },
        collide: {
          radius: (node: any) => (node.size || 32) / 2 + 6,
          strength: 0.8,
          iterations: 2,
        },
      } as any
  }
}

// ---- 数据转换 ----
const toGraphData = (): GraphData => {
  const degrees = computeDegrees()
  const maxDeg = Math.max(1, ...degrees.values())
  const showAllLabels = props.nodes.length <= 15

  return {
    nodes: props.nodes.map<NodeData>((n) => {
      const deg = degrees.get(n.id) || 0
      // 度数映射到 28~46
      const size = 28 + (deg / maxDeg) * 18
      return {
        id: n.id,
        data: {
          label: n.label,
          name: n.name,
          confidence: n.confidence,
          degree: deg,
          size,
          content: n.content,
          showLabel: showAllLabels || deg >= 2,
        },
      }
    }),
    edges: props.edges.map<EdgeData>((e) => ({
      source: e.source,
      target: e.target,
      data: { relation: e.relation, weight: e.weight ?? 1 },
    })),
  }
}

// ---- Tooltip 内容生成 ----
const tooltipContent = async (_evt: any, items: any[]): Promise<string> => {
  if (!items.length) return ''
  const item = items[0]
  const data = item.data || {}
  if (item.type === 'node') {
    const label = getNodeLabel(data.label || 'Entity')
    const name = data.name || item.id
    const deg = data.degree ?? 0
    const conf = ((data.confidence ?? 0) * 100).toFixed(0)
    return `<div class="l3-tip">
      <div class="l3-tip-title">${escapeHtml(name)}</div>
      <div class="l3-tip-row"><span>类型</span><b>${label}</b></div>
      <div class="l3-tip-row"><span>连接</span><b>${deg}</b></div>
      <div class="l3-tip-row"><span>置信度</span><b>${conf}%</b></div>
    </div>`
  }
  // edge
  const rel = getRelationLabel(data.relation || '')
  const w = (data.weight ?? 1).toFixed(2)
  return `<div class="l3-tip">
    <div class="l3-tip-title">${escapeHtml(rel)}</div>
    <div class="l3-tip-row"><span>权重</span><b>${w}</b></div>
  </div>`
}

const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

// ---- 初始化 G6 ----
const initGraph = () => {
  const container = containerRef.value
  if (!container) return
  const width = container.clientWidth || 800
  const height = container.clientHeight || 500
  const dark = isDarkTheme()
  const textColor = dark ? '#e0e0e0' : '#424242'
  const labelBg = dark ? 'rgba(33,33,33,0.9)' : 'rgba(255,255,255,0.92)'
  const edgeColor = dark ? '#555555' : '#bdbdbd'
  const gridColor = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'

  const useForce = props.layout === 'force'

  const graph = new Graph({
    container,
    width,
    height,
    autoFit: 'view',
    data: toGraphData(),
    layout: layoutConfig(props.layout, props.nodes.length),
    theme: dark ? 'dark' : 'light',
    node: {
      type: 'circle',
      style: (d: NodeData) => {
        const label = (d.data?.label as string) || 'Entity'
        const name = (d.data?.name as string) || String(d.id)
        const size = (d.data?.size as number) || 32
        const colors = nodeColors(label)
        const showLabel = (d.data?.showLabel as boolean) ?? true
        const deg = (d.data?.degree as number) || 0
        return {
          size,
          fill: colors.fill,
          stroke: '#ffffff',
          lineWidth: 2.5,
          shadowColor: 'rgba(0,0,0,0.25)',
          shadowBlur: 8,
          shadowOffsetX: 0,
          shadowOffsetY: 2,
          cursor: 'pointer',
          icon: false,
          // 度数 >= 3 的节点显示度数徽标
          badge: deg >= 3,
          badges: deg >= 3
            ? [{ text: String(deg), placement: 'right-top', fill: '#ff9800', color: '#fff' }]
            : [],
          labelText: showLabel ? name : '',
          labelPlacement: 'bottom',
          labelFontSize: 11,
          labelFontWeight: 500,
          labelFill: textColor,
          labelBackground: true,
          labelBackgroundFill: labelBg,
          labelBackgroundOpacity: 0.95,
          labelBackgroundRadius: 4,
          labelPadding: [3, 6],
        }
      },
      state: {
        active: {
          lineWidth: 3.5,
          stroke: '#ff9800',
          shadowColor: 'rgba(255,152,0,0.5)',
          shadowBlur: 16,
          labelOpacity: 1,
        },
        inactive: {
          opacity: 0.15,
          labelOpacity: 0,
        },
        selected: {
          lineWidth: 3.5,
          stroke: '#ff9800',
          shadowColor: 'rgba(255,152,0,0.6)',
          shadowBlur: 18,
        },
      },
    },
    edge: {
      type: 'quadratic',
      style: (d: EdgeData) => {
        const w = (d.data?.weight as number) ?? 1
        return {
          stroke: edgeColor,
          lineWidth: Math.min(1 + w * 0.6, 3),
          strokeOpacity: 0.55,
          endArrow: true as any,
          endArrowSize: 6,
          endArrowFill: edgeColor,
          curveOffset: 20,
          curvePosition: 0.5,
          cursor: 'pointer',
        }
      },
      state: {
        active: {
          stroke: '#ff9800',
          lineWidth: 2.5,
          strokeOpacity: 1,
          endArrowFill: '#ff9800',
        },
        inactive: { opacity: 0.05 },
      },
    },
    behaviors: [
      'zoom-canvas',
      'drag-canvas',
      useForce
        ? { type: 'drag-element-force', fixed: true }
        : 'drag-element',
      {
        type: 'hover-activate',
        degree: 1,
        state: 'active',
        inactiveState: 'inactive',
      } as any,
    ],
    plugins: [
      {
        type: 'grid-line',
        size: 24,
        stroke: gridColor,
        lineWidth: 1,
        border: false,
      } as any,
      {
        type: 'tooltip',
        trigger: 'hover',
        position: 'top',
        offset: [0, 12],
        getContent: tooltipContent,
      } as any,
      {
        type: 'minimap',
        size: [180, 120],
        position: 'right-bottom',
        className: 'l3-minimap',
      } as any,
    ],
  })

  // 节点单击：通知父组件打开抽屉
  graph.on('node:click', (evt: any) => {
    const id = evt.target?.id as string | undefined
    if (!id) return
    const node = props.nodes.find((n) => n.id === id)
    if (node) emit('node-click', node)
  })

  // 节点双击：以此节点展开
  graph.on('node:dblclick', (evt: any) => {
    const id = evt.target?.id as string | undefined
    if (id) emit('node-dblclick', id)
  })

  // 边点击
  graph.on('edge:click', (evt: any) => {
    const id = evt.target?.id as string | undefined
    if (!id) return
    const edgeData = graph.getEdgeData(id)
    const edge = props.edges.find(
      (e) => e.source === edgeData?.source && e.target === edgeData?.target
    )
    if (edge) emit('edge-click', edge)
  })

  graphRef.value = graph
  graph.render().catch((e) => console.error('G6 render failed:', e))
}

// ---- 数据更新 ----
const updateData = async () => {
  const graph = graphRef.value
  if (!graph) return
  graph.setData(toGraphData())
  try {
    await graph.render()
  } catch (e) {
    console.error('G6 update failed:', e)
  }
}

watch(
  () => [props.nodes, props.edges],
  () => nextTick(() => updateData()),
  { deep: true }
)

// ---- 布局切换：重建图实例 ----
watch(
  () => props.layout,
  () => {
    destroyGraph()
    nextTick(() => initGraph())
  }
)

// ---- 工具栏操作 ----
const zoomBy = (factor: number) => {
  const graph = graphRef.value
  if (!graph) return
  const cur = graph.getZoom?.() ?? 1
  graph.zoomTo(Math.min(Math.max(cur * factor, 0.2), 4)).catch(() => {})
}

const fitView = () => {
  graphRef.value?.fitView({ when: 'always' }).catch(() => {})
}

const fitCenter = () => {
  graphRef.value?.fitCenter().catch(() => {})
}

// ---- 暴露给父组件 ----
const focusNode = async (nodeId: string) => {
  const graph = graphRef.value
  if (!graph) return
  if (graph.hasNode(nodeId)) {
    clearSelected()
    graph.setElementState(nodeId, 'selected')
    await graph.focusElement(nodeId).catch(() => {})
  } else {
    emit('node-dblclick', nodeId)
  }
}

const highlightNode = (nodeId: string) => {
  const graph = graphRef.value
  if (!graph || !graph.hasNode(nodeId)) return
  clearSelected()
  graph.setElementState(nodeId, 'selected')
}

const clearSelected = () => {
  const graph = graphRef.value
  if (!graph) return
  try {
    graph.getNodeData().forEach((n) => {
      if (graph.getElementState(n.id).includes('selected')) {
        graph.setElementState(n.id, [])
      }
    })
  } catch {
    // ignore
  }
}

defineExpose({ focusNode, highlightNode, clearSelected })

// ---- 生命周期 ----
const handleResize = () => {
  const graph = graphRef.value
  const container = containerRef.value
  if (!graph || !container) return
  graph.setSize(container.clientWidth, container.clientHeight)
}

const destroyGraph = () => {
  graphRef.value?.destroy()
  graphRef.value = null
}

onMounted(() => {
  initGraph()
  resizeObserver = new ResizeObserver(() => handleResize())
  if (containerRef.value) resizeObserver.observe(containerRef.value)
})

onUnmounted(() => {
  resizeObserver?.disconnect()
  resizeObserver = null
  destroyGraph()
})
</script>

<style scoped>
.canvas-wrapper {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 400px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}

.canvas-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: rgb(var(--v-theme-surface));
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  flex-wrap: wrap;
  gap: 8px;
}

.toolbar-left,
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 4px;
}

.graph-container {
  position: relative;
  flex: 1;
  width: 100%;
  min-height: 0; /* 允许 flex 子项收缩，配合 min-height: 360px 的下限 */
  background: rgb(var(--v-theme-surface));
  overflow: hidden;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: rgb(var(--v-theme-surface));
  z-index: 10;
}

:deep(.l3-minimap) {
  border-radius: 8px !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
  background: rgba(var(--v-theme-surface), 0.95) !important;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.1) !important;
}
</style>
