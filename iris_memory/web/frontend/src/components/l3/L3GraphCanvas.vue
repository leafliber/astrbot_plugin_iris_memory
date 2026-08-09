<template>
  <div class="canvas-wrapper">
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

      <div
        v-show="tooltip.visible"
        class="l3-tooltip"
        :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
        v-html="tooltip.content"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, shallowRef, reactive, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useTheme } from 'vuetify'
import cytoscape from 'cytoscape'
import type { Core, ElementDefinition, StylesheetStyle, LayoutOptions } from 'cytoscape'
import fcose from 'cytoscape-fcose'
import dagre from 'cytoscape-dagre'
import type { KGNode, KGEdge, L3LayoutType } from '@/types'
import {
  getNodeIcon,
  getNodeLabel,
  getRelationLabel,
  resolveThemeColor,
  NODE_TYPE_COLORS,
} from '@/composables/l3Constants'

cytoscape.use(fcose)
cytoscape.use(dagre)

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
const cyRef = shallowRef<Core | null>(null)
let resizeObserver: ResizeObserver | null = null
let resizeTimer: ReturnType<typeof setTimeout> | null = null

const vuetifyTheme = useTheme()

const tooltip = reactive({ visible: false, x: 0, y: 0, content: '' })

// ---- 主题 ----
const isDark = (): boolean => {
  try {
    const bg = getComputedStyle(document.documentElement)
      .getPropertyValue('--v-theme-surface')
      .trim()
    if (bg) {
      const [r, g, b] = bg.split(',').map((s) => parseInt(s.trim(), 10))
      return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5
    }
  } catch { /* ignore */ }
  return vuetifyTheme.global.current.value.dark ?? false
}

// ---- 度数计算 ----
const computeDegrees = (): Map<string, number> => {
  const deg = new Map<string, number>()
  props.edges.forEach((e) => {
    deg.set(e.source, (deg.get(e.source) || 0) + 1)
    deg.set(e.target, (deg.get(e.target) || 0) + 1)
  })
  return deg
}

// ---- 样式表 ----
const buildStylesheet = (): StylesheetStyle[] => {
  const dark = isDark()
  const textColor = dark ? '#e0e0e0' : '#424242'
  const labelBg = dark ? 'rgba(33,33,33,0.92)' : 'rgba(255,255,255,0.92)'
  const edgeColor = dark ? '#666666' : '#bdbdbd'

  // cytoscape 3.34 自带类型对 style 字段约束过严（如 transition-duration 要求 number），
  // 此处按运行时实际接受的写法构造，返回时再收敛为 StylesheetStyle[]
  const sheet: any[] = [
    {
      selector: 'node',
      style: {
        width: 'data(size)' as any,
        height: 'data(size)' as any,
        'background-color': '#5c6bc0',
        'border-width': 2.5,
        'border-color': dark ? 'rgba(255,255,255,0.85)' : '#ffffff',
        label: 'data(displayName)' as any,
        'text-valign': 'bottom',
        'text-halign': 'center',
        'font-size': 11,
        'font-weight': 500,
        color: textColor,
        'text-background-color': labelBg,
        'text-background-opacity': 0.95,
        'text-background-shape': 'roundrectangle',
        'text-margin-y': 4,
        'text-wrap': 'ellipsis',
        'text-max-width': '90px',
        'overlay-padding': '4px',
        'transition-property': 'background-color, border-color, border-width, opacity',
        'transition-duration': '0.2s',
      },
    },
    ...Object.entries(NODE_TYPE_COLORS).map(
      ([type, colorName]) => ({
        selector: `node[type="${type}"]`,
        style: {
          'background-color': resolveThemeColor(colorName, '#5c6bc0'),
        },
      })
    ),
    {
      selector: 'edge',
      style: {
        width: 'data(edgeWidth)' as any,
        'line-color': edgeColor,
        'target-arrow-color': edgeColor,
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.85,
        'curve-style': 'bezier',
        'control-point-step-size': 20,
        opacity: 0.55,
        'transition-property': 'line-color, target-arrow-color, opacity, width',
        'transition-duration': '0.2s',
      },
    },
    {
      selector: 'node.highlighted',
      style: {
        'border-width': 3.5,
        'border-color': '#ff9800',
        'overlay-color': 'rgba(255,152,0,0.12)',
        'overlay-opacity': 1,
      },
    },
    {
      selector: 'edge.highlighted',
      style: {
        'line-color': '#ff9800',
        'target-arrow-color': '#ff9800',
        width: 2.5,
        opacity: 1,
      },
    },
    {
      selector: 'node.faded',
      style: {
        opacity: 0.25,
        'text-opacity': 0,
      },
    },
    {
      selector: 'edge.faded',
      style: {
        opacity: 0.08,
      },
    },
    {
      selector: 'node.selected',
      style: {
        'border-width': 3.5,
        'border-color': '#ff9800',
        'overlay-color': 'rgba(255,152,0,0.18)',
        'overlay-opacity': 1,
      },
    },
  ]
  return sheet
}

// ---- 布局配置 ----
const layoutConfig = (type: L3LayoutType, nodeCount: number): LayoutOptions => {
  const edgeLen = nodeCount > 40 ? 80 : nodeCount > 20 ? 110 : 140
  const repulsion = nodeCount > 40 ? 3000 : 5000

  switch (type) {
    case 'dagre':
      return {
        name: 'dagre',
        rankDir: 'LR',
        nodeSep: 30,
        rankSep: 60,
        padding: 30,
        animate: true,
        animationDuration: 400,
      } as any
    case 'radial':
      return {
        name: 'breadthfirst',
        circles: true,
        directed: false,
        padding: 30,
        spacingFactor: 1.15,
        animate: true,
        animationDuration: 400,
      } as any
    case 'concentric':
      return {
        name: 'concentric',
        minNodeSpacing: 40,
        padding: 30,
        animate: true,
        animationDuration: 400,
        concentric: (ele: any) => ele.data('degree') ?? 0,
        levelWidth: () => 2,
      } as any
    case 'force':
    default:
      return {
        name: 'fcose',
        animate: true,
        animationDuration: 500,
        fit: true,
        padding: 30,
        nodeRepulsion: repulsion,
        idealEdgeLength: edgeLen,
        edgeElasticity: 0.45,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
        randomize: true,
      } as any
  }
}

// ---- 数据转换 ----
const toElements = (): ElementDefinition[] => {
  const degrees = computeDegrees()
  const maxDeg = Math.max(1, ...degrees.values())
  const showAllLabels = props.nodes.length <= 15

  const nodes: ElementDefinition[] = props.nodes.map((n) => {
    const deg = degrees.get(n.id) || 0
    const size = 28 + (deg / maxDeg) * 18
    const showLabel = showAllLabels || deg >= 2
    return {
      data: {
        id: n.id,
        type: n.label,
        name: n.name,
        displayName: showLabel ? n.name : '',
        confidence: n.confidence,
        degree: deg,
        size,
        content: n.content,
      },
    }
  })

  const edges: ElementDefinition[] = props.edges.map((e, i) => ({
    data: {
      id: `e-${e.source}-${e.target}-${i}`,
      source: e.source,
      target: e.target,
      relation: e.relation,
      weight: e.weight ?? 1,
      edgeWidth: Math.min(1 + (e.weight ?? 1) * 0.6, 3),
    },
  }))

  return [...nodes, ...edges]
}

// ---- Tooltip ----
const escapeHtml = (s: string): string =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const nodeTooltipHtml = (data: any): string => {
  const label = getNodeLabel(data.type || 'Entity')
  const name = data.name || data.id
  const deg = data.degree ?? 0
  const conf = ((data.confidence ?? 0) * 100).toFixed(0)
  return `<div class="l3-tip">
    <div class="l3-tip-title">${escapeHtml(name)}</div>
    <div class="l3-tip-row"><span>类型</span><b>${escapeHtml(label)}</b></div>
    <div class="l3-tip-row"><span>连接</span><b>${deg}</b></div>
    <div class="l3-tip-row"><span>置信度</span><b>${conf}%</b></div>
  </div>`
}

const edgeTooltipHtml = (data: any): string => {
  const rel = getRelationLabel(data.relation || '')
  const w = (data.weight ?? 1).toFixed(2)
  return `<div class="l3-tip">
    <div class="l3-tip-title">${escapeHtml(rel)}</div>
    <div class="l3-tip-row"><span>权重</span><b>${w}</b></div>
  </div>`
}

const showTooltip = (renderedPos: { x: number; y: number }, html: string) => {
  tooltip.x = renderedPos.x + 14
  tooltip.y = renderedPos.y - 14
  tooltip.content = html
  tooltip.visible = true
}

const hideTooltip = () => {
  tooltip.visible = false
}

// ---- Hover 高亮 ----
const clearHoverStates = () => {
  const cy = cyRef.value
  if (!cy) return
  cy.elements().removeClass('highlighted faded')
  hideTooltip()
}

// ---- 初始化 ----
const initGraph = () => {
  const container = containerRef.value
  if (!container) return

  const cy = cytoscape({
    container,
    elements: toElements(),
    style: buildStylesheet(),
    layout: layoutConfig(props.layout, props.nodes.length),
    minZoom: 0.15,
    maxZoom: 4,
    wheelSensitivity: 0.3,
    boxSelectionEnabled: false,
  })

  // 节点悬停：高亮邻居、淡化其余
  cy.on('mouseover', 'node', (e) => {
    const node = e.target
    const neighborhood = node.closedNeighborhood()
    cy.elements().not(neighborhood).addClass('faded')
    neighborhood.addClass('highlighted')
    showTooltip(e.renderedPosition, nodeTooltipHtml(node.data()))
  })

  cy.on('mouseout', 'node', () => {
    clearHoverStates()
  })

  // 边悬停：仅 tooltip
  cy.on('mouseover', 'edge', (e) => {
    showTooltip(e.renderedPosition, edgeTooltipHtml(e.target.data()))
  })

  cy.on('mouseout', 'edge', () => {
    hideTooltip()
  })

  // 画布空白区域：兜底清除
  cy.on('mouseout', (e) => {
    if (e.target === cy) clearHoverStates()
  })

  // 节点单击
  cy.on('tap', 'node', (e) => {
    const id = e.target.id() as string
    const node = props.nodes.find((n) => n.id === id)
    if (node) emit('node-click', node)
  })

  // 节点双击：展开
  cy.on('dbltap', 'node', (e) => {
    emit('node-dblclick', e.target.id() as string)
  })

  // 边点击
  cy.on('tap', 'edge', (e) => {
    const d = e.target.data()
    const edge = props.edges.find(
      (ed) => ed.source === d.source && ed.target === d.target
    )
    if (edge) emit('edge-click', edge)
  })

  cyRef.value = cy
}

// ---- 数据更新 ----
const updateData = () => {
  const cy = cyRef.value
  if (!cy) return
  cy.elements().remove()
  cy.add(toElements())
  runLayout()
}

const runLayout = () => {
  const cy = cyRef.value
  if (!cy) return
  const count = cy.nodes().length
  if (count === 0) return
  const l = cy.layout(layoutConfig(props.layout, count))
  l.on('layoutstop', () => cy.fit(undefined, 30))
  l.run()
}

watch(
  () => [props.nodes, props.edges] as const,
  () => nextTick(() => updateData())
)

// ---- 布局切换 ----
watch(
  () => props.layout,
  () => runLayout()
)

// ---- 主题切换：重建样式 ----
watch(
  () => vuetifyTheme.global.name.value,
  () => {
    const cy = cyRef.value
    if (!cy) return
    cy.style().fromJson(buildStylesheet() as any).update()
  }
)

// ---- 工具栏 ----
const zoomBy = (factor: number) => {
  const cy = cyRef.value
  if (!cy) return
  const z = cy.zoom()
  cy.zoom({ level: Math.min(Math.max(z * factor, 0.15), 4), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } })
}

const fitView = () => {
  cyRef.value?.fit(undefined, 30)
}

const fitCenter = () => {
  cyRef.value?.center()
}

// ---- 暴露给父组件 ----
const focusNode = async (nodeId: string) => {
  const cy = cyRef.value
  if (!cy) return
  const ele = cy.getElementById(nodeId)
  if (ele.length) {
    clearSelected()
    ele.addClass('selected')
    cy.animate({
      center: { eles: ele },
      duration: 350,
    })
  } else {
    emit('node-dblclick', nodeId)
  }
}

const highlightNode = (nodeId: string) => {
  const cy = cyRef.value
  if (!cy) return
  const ele = cy.getElementById(nodeId)
  if (!ele.length) return
  clearSelected()
  ele.addClass('selected')
}

const clearSelected = () => {
  cyRef.value?.nodes().removeClass('selected')
}

defineExpose({ focusNode, highlightNode, clearSelected })

// ---- 生命周期 ----
const handleResize = () => {
  if (resizeTimer) clearTimeout(resizeTimer)
  resizeTimer = setTimeout(() => {
    cyRef.value?.resize()
  }, 150)
}

onMounted(() => {
  initGraph()
  resizeObserver = new ResizeObserver(handleResize)
  if (containerRef.value) resizeObserver.observe(containerRef.value)
})

onUnmounted(() => {
  if (resizeTimer) clearTimeout(resizeTimer)
  resizeObserver?.disconnect()
  resizeObserver = null
  cyRef.value?.destroy()
  cyRef.value = null
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
  min-height: 0;
  background: rgb(var(--v-theme-surface));
  background-image:
    linear-gradient(rgba(var(--v-theme-on-surface), 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(var(--v-theme-on-surface), 0.04) 1px, transparent 1px);
  background-size: 24px 24px;
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

.l3-tooltip {
  position: absolute;
  z-index: 20;
  pointer-events: none;
  max-width: 260px;
  transform: translateY(-100%);
}
</style>

<style>
.l3-tip {
  background: rgb(var(--v-theme-surface));
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.14);
  font-size: 13px;
  line-height: 1.5;
}

.l3-tip-title {
  font-weight: 700;
  font-size: 14px;
  margin-bottom: 6px;
  color: rgb(var(--v-theme-on-surface));
}

.l3-tip-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 1px 0;
}

.l3-tip-row span {
  color: rgba(var(--v-theme-on-surface), 0.55);
}

.l3-tip-row b {
  color: rgb(var(--v-theme-on-surface));
  font-weight: 600;
}
</style>
