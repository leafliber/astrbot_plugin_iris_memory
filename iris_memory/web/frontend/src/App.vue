<template>
  <v-app>
    <v-navigation-drawer v-model="drawer" :rail="rail" permanent>
      <v-list-item
        prepend-icon="mdi-flower-tulip"
        title="Iris Memory"
        nav
        @click="rail = !rail"
      >
        <template #append>
          <v-btn
            :icon="rail ? 'mdi-chevron-right' : 'mdi-chevron-left'"
            variant="text"
            size="small"
          />
        </template>
      </v-list-item>

      <v-divider />

      <v-list density="compact" nav>
        <v-list-item
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          :prepend-icon="item.icon"
          :title="item.title"
          :value="item.to"
          color="primary"
        />
      </v-list>
    </v-navigation-drawer>

    <v-app-bar color="surface" elevation="0" border="b">
      <v-app-bar-title class="text-h6">
        {{ currentTitle }}
      </v-app-bar-title>

      <template #append>
        <v-btn
          icon="mdi-refresh"
          variant="text"
          :loading="loading"
          @click="handleRefresh"
        />

        <v-btn
          :icon="darkMode ? 'mdi-weather-sunny' : 'mdi-weather-night'"
          variant="text"
          @click="toggleTheme"
        />
      </template>
    </v-app-bar>

    <v-main>
      <v-container fluid class="pa-4">
        <router-view v-slot="{ Component }">
          <keep-alive>
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </v-container>
    </v-main>

    <v-snackbar
      v-model="showError"
      color="error"
      :timeout="3000"
      location="top"
    >
      {{ error }}
    </v-snackbar>
  </v-app>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useTheme } from 'vuetify'
import { storeToRefs } from 'pinia'
import { useAppStore } from '@/stores'

const route = useRoute()
const appStore = useAppStore()
const theme = useTheme()

const { loading, error, darkMode } = storeToRefs(appStore)

const drawer = ref(true)
const rail = ref(false)
const showError = ref(false)

watch(darkMode, (val) => {
  theme.global.name.value = val ? 'dark' : 'light'
}, { immediate: true })

onMounted(() => {
  appStore.initTheme()
})

const navItems = [
  { to: '/dashboard', title: '仪表盘', icon: 'mdi-view-dashboard' },
  { to: '/l1-buffer', title: 'L1 缓冲', icon: 'mdi-lightning-bolt' },
  { to: '/l2-memory', title: 'L2 记忆', icon: 'mdi-database-search' },
  { to: '/l3-graph', title: 'L3 图谱', icon: 'mdi-graph' },
  { to: '/profile', title: '画像管理', icon: 'mdi-account-group' },
  { to: '/data-manage', title: '数据管理', icon: 'mdi-swap-vertical' },
  { to: '/reply-control', title: '主动回复', icon: 'mdi-robot' },
  { to: '/hidden-config', title: '隐藏参数', icon: 'mdi-cog-outline' }
]

const currentTitle = computed(() => {
  const item = navItems.find(i => i.to === route.path)
  return item?.title || 'Iris Memory'
})

const handleRefresh = () => {
  window.dispatchEvent(new CustomEvent('iris:refresh'))
}

const toggleTheme = () => {
  appStore.toggleTheme()
}

watch(error, (val) => {
  showError.value = !!val
})
</script>

<style>
/* 全局应用样式 */
.v-navigation-drawer {
  border-right: 1px solid rgba(var(--v-theme-on-surface), 0.06) !important;
}

/* 品牌头部 */
.v-navigation-drawer .v-list-item:first-child {
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.04);
}

.v-navigation-drawer .v-list-item:first-child .v-list-item-title {
  font-weight: 700;
  letter-spacing: 0.02em;
}

/* 侧边栏导航项 */
.v-navigation-drawer .v-list-item {
  border-radius: 8px;
  margin: 2px 8px;
  transition: background 0.2s ease, transform 0.15s ease;
}

.v-navigation-drawer .v-list-item:hover {
  transform: translateX(2px);
}

/* 活动导航项：纯色高亮，覆盖 Vuetify 默认渐变 */
.v-navigation-drawer .v-list-item--active {
  background: rgba(var(--v-theme-primary), 0.12) !important;
  color: rgb(var(--v-theme-primary)) !important;
}

/* 移除 Vuetify 默认的 ::before 渐变层 */
.v-navigation-drawer .v-list-item--active::before,
.v-navigation-drawer .v-list-item--active .v-list-item__overlay {
  opacity: 0 !important;
}

.v-navigation-drawer .v-list-item--active .v-icon {
  color: rgb(var(--v-theme-primary)) !important;
}

/* 顶栏 */
.v-app-bar {
  backdrop-filter: blur(8px);
  background: rgba(var(--v-theme-surface), 0.85) !important;
}

.v-app-bar .v-app-bar-title {
  font-weight: 700;
  letter-spacing: 0.01em;
}

/* 主内容容器 */
.v-main > .v-container {
  max-width: 1600px;
}

/* 全局：所有 v-card 圆角统一 */
/* !important 覆盖 Vuetify 的 .v-card { border-radius: 4px }
   （同为 0,1,0，Vuetify 按需注入的组件样式在构建产物中排列在后会胜出） */
.v-card {
  border-radius: var(--iris-card-radius) !important;
}

/* 全局：v-tabs 下划线 */
.v-tabs .v-tab {
  font-weight: 500;
  letter-spacing: 0.01em;
  text-transform: none;
}

/* 全局：滚动条 */
.v-navigation-drawer__content::-webkit-scrollbar {
  width: 4px;
}

.v-navigation-drawer__content::-webkit-scrollbar-thumb {
  background: rgba(var(--v-theme-on-surface), 0.15);
  border-radius: 2px;
}
</style>
