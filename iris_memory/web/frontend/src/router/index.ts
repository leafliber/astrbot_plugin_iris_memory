import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'

const viewLoaders = {
  dashboard: () => import('@/views/DashboardView.vue'),
  l1Buffer: () => import('@/views/L1BufferView.vue'),
  l2Memory: () => import('@/views/L2MemoryView.vue'),
  l3Graph: () => import('@/views/L3GraphView.vue'),
  profile: () => import('@/views/ProfileView.vue'),
  learning: () => import('@/views/LearningView.vue'),
  personaEvolution: () => import('@/views/PersonaEvolutionView.vue'),
  dataManage: () => import('@/views/DataManageView.vue'),
  replyControl: () => import('@/views/ReplyControlView.vue'),
  runLog: () => import('@/views/RunLogView.vue'),
  hiddenConfig: () => import('@/views/HiddenConfigView.vue')
}

// AstrBot 插件 Page 的资源 asset_token 有效期很短。生产环境启动后立即在
// 后台预热路由模块，确保所有 chunk 在 token 有效期内进入浏览器模块缓存；
// 页面渲染不等待预热完成。
if (import.meta.env.PROD) {
  void Promise.allSettled(Object.values(viewLoaders).map(load => load()))
}

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: viewLoaders.dashboard,
    meta: { title: '仪表盘', icon: 'mdi-view-dashboard' }
  },
  {
    path: '/l1-buffer',
    name: 'L1Buffer',
    component: viewLoaders.l1Buffer,
    meta: { title: 'L1 缓冲', icon: 'mdi-lightning-bolt' }
  },
  {
    path: '/l2-memory',
    name: 'L2Memory',
    component: viewLoaders.l2Memory,
    meta: { title: 'L2 记忆', icon: 'mdi-database-search' }
  },
  {
    path: '/l3-graph',
    name: 'L3Graph',
    component: viewLoaders.l3Graph,
    meta: { title: 'L3 图谱', icon: 'mdi-graph' }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: viewLoaders.profile,
    meta: { title: '画像管理', icon: 'mdi-account-group' }
  },
  {
    path: '/learning',
    name: 'Learning',
    component: viewLoaders.learning,
    meta: { title: '学习管理', icon: 'mdi-school' }
  },
  {
    path: '/persona-evolution',
    name: 'PersonaEvolution',
    component: viewLoaders.personaEvolution,
    meta: { title: '人格自迭代', icon: 'mdi-account-cog' }
  },
  {
    path: '/data-manage',
    name: 'DataManage',
    component: viewLoaders.dataManage,
    meta: { title: '数据管理', icon: 'mdi-swap-vertical' }
  },
  {
    path: '/reply-control',
    name: 'ReplyControl',
    component: viewLoaders.replyControl,
    meta: { title: '主动回复', icon: 'mdi-robot' }
  },
  {
    path: '/run-log',
    name: 'RunLog',
    component: viewLoaders.runLog,
    meta: { title: '运行日志', icon: 'mdi-text-box-search-outline' }
  },
  {
    path: '/hidden-config',
    name: 'HiddenConfig',
    component: viewLoaders.hiddenConfig,
    meta: { title: '隐藏参数', icon: 'mdi-cog-outline' }
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
