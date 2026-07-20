import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: { title: '仪表盘', icon: 'mdi-view-dashboard' }
  },
  {
    path: '/l1-buffer',
    name: 'L1Buffer',
    component: () => import('@/views/L1BufferView.vue'),
    meta: { title: 'L1 缓冲', icon: 'mdi-lightning-bolt' }
  },
  {
    path: '/l2-memory',
    name: 'L2Memory',
    component: () => import('@/views/L2MemoryView.vue'),
    meta: { title: 'L2 记忆', icon: 'mdi-database-search' }
  },
  {
    path: '/l3-graph',
    name: 'L3Graph',
    component: () => import('@/views/L3GraphView.vue'),
    meta: { title: 'L3 图谱', icon: 'mdi-graph' }
  },
  {
    path: '/profile',
    name: 'Profile',
    component: () => import('@/views/ProfileView.vue'),
    meta: { title: '画像管理', icon: 'mdi-account-group' }
  },
  {
    path: '/data-manage',
    name: 'DataManage',
    component: () => import('@/views/DataManageView.vue'),
    meta: { title: '数据管理', icon: 'mdi-swap-vertical' }
  },
  {
    path: '/hidden-config',
    name: 'HiddenConfig',
    component: () => import('@/views/HiddenConfigView.vue'),
    meta: { title: '隐藏参数', icon: 'mdi-cog-outline' }
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
