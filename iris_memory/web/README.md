# Iris Chat Memory - Web 模块

为 Iris Chat Memory 插件提供 Web 可视化管理界面。

## 架构

- **后端**：通过 `context.register_web_api()` 注册 API，由 AstrBot Dashboard 统一托管
- **前端**：Vue.js 3 + Vuetify 3 + TypeScript SPA，构建为单文件输出到 `pages/iris/`
- **认证**：由 AstrBot Dashboard 统一处理，无需额外配置
- **路由**：Hash 模式（`createWebHashHistory`），兼容 AstrBot Plugin Pages

## 快速开始

### 使用 Web 界面

插件安装后，Web 界面自动注册到 AstrBot Dashboard。在 AstrBot 管理面板的插件页面中即可访问。

### 前端开发

#### 1. 安装依赖

```bash
cd iris_memory/web/frontend
npm install
```

#### 2. 开发模式

```bash
npm run dev
```

访问 `http://localhost:5173`

#### 3. 构建生产版本

```bash
npm run build
```

构建产物通过 `vite-plugin-singlefile` 输出为单文件到 `pages/iris/index.html`，由 AstrBot 自动托管。

## 项目结构

```
iris_memory/web/
├── __init__.py                  # Web 模块入口，注册所有路由
├── routes/                      # 后端 API 路由
│   ├── __init__.py              # 路由导出
│   ├── memory.py                # 记忆 API（L1/L2/L3 查询、搜索、编辑、删除）
│   ├── profile.py               # 画像 API（用户/群聊画像的查询、编辑、删除）
│   ├── stats.py                 # 统计 API（Token、记忆、图谱、系统状态）
│   ├── data_routes.py           # 数据导入导出 API（L2/L3/画像/全量）
│   ├── manage_routes.py         # 管理操作 API（清空、删除、任务触发）
│   └── hidden_config_routes.py  # 隐藏配置 API（查询、更新、重置）
└── frontend/                    # Vue.js 3 前端项目
    ├── src/
    │   ├── api/                 # API 请求封装
    │   │   ├── index.ts         # API 入口
    │   │   ├── request.ts       # 请求工具
    │   │   ├── memory.ts        # 记忆 API
    │   │   ├── profile.ts       # 画像 API
    │   │   ├── stats.ts         # 统计 API
    │   │   ├── data.ts          # 数据导入导出 API
    │   │   ├── manage.ts        # 管理操作 API
    │   │   └── hiddenConfig.ts  # 隐藏配置 API
    │   ├── views/               # 页面组件
    │   │   ├── DashboardView.vue      # 仪表盘（系统状态总览）
    │   │   ├── L1BufferView.vue       # L1 消息缓冲查看
    │   │   ├── L2MemoryView.vue       # L2 记忆库搜索与管理
    │   │   ├── L3GraphView.vue        # L3 知识图谱可视化
    │   │   ├── ProfileView.vue        # 画像查看与编辑
    │   │   ├── DataManageView.vue     # 数据导入导出
    │   │   └── HiddenConfigView.vue   # 隐藏配置热修改
    │   ├── components/          # 通用组件
    │   │   └── ComponentDisabled.vue  # 组件不可用提示
    │   ├── composables/         # 组合式函数
    │   │   └── useComponentState.ts   # 组件状态管理
    │   ├── stores/              # Pinia 状态管理
    │   │   ├── index.ts
    │   │   ├── app.ts           # 应用全局状态
    │   │   ├── memory.ts        # 记忆数据状态
    │   │   ├── profile.ts       # 画像数据状态
    │   │   └── stats.ts         # 统计数据状态
    │   ├── router/              # Vue Router 配置
    │   │   └── index.ts         # 路由定义（Hash 模式）
    │   ├── types/               # TypeScript 类型定义
    │   │   └── index.ts
    │   ├── App.vue              # 根组件
    │   ├── main.ts              # 入口文件
    │   ├── icons.ts             # 图标注册
    │   ├── MdiSvgIcon.ts        # MDI SVG 图标组件
    │   └── astrbot-bridge.d.ts  # AstrBot 桥接类型声明
    ├── index.html               # HTML 入口
    ├── package.json
    ├── vite.config.ts           # Vite 配置（单文件输出）
    ├── tsconfig.json
    └── tsconfig.node.json
```

## API 文档

所有 API 路径前缀为 `/{plugin_name}/`，其中 `plugin_name` = `astrbot_plugin_iris_chat_memory`。

### Memory API

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/memory/l2/search` | POST | 搜索 L2 记忆 |
| `/{prefix}/memory/l2/latest` | GET | 获取最新 L2 记忆 |
| `/{prefix}/memory/l2/stats` | GET | 获取 L2 统计 |
| `/{prefix}/memory/l2/delete` | POST | 删除 L2 记忆条目 |
| `/{prefix}/memory/l2/update` | POST | 更新 L2 记忆条目 |
| `/{prefix}/memory/l1/list` | GET | 获取 L1 缓冲列表 |
| `/{prefix}/memory/l1/queues` | GET | 获取 L1 队列列表 |
| `/{prefix}/memory/l3/graph` | GET | 获取 L3 图谱数据（路径扩展） |
| `/{prefix}/memory/l3/search/nodes` | GET | 搜索 L3 节点 |
| `/{prefix}/memory/l3/search/edges` | GET | 搜索 L3 边 |
| `/{prefix}/memory/l3/nodes` | GET | 获取 L3 节点列表 |
| `/{prefix}/memory/l3/edges` | GET | 获取 L3 关系列表 |
| `/{prefix}/memory/l3/nodes/delete` | POST | 删除 L3 节点 |
| `/{prefix}/memory/l3/edges/delete` | POST | 删除 L3 关系 |

### Profile API

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/profile/group` | GET | 获取群聊画像 |
| `/{prefix}/profile/group/update` | POST | 更新群聊画像 |
| `/{prefix}/profile/group/delete` | POST | 删除群聊画像 |
| `/{prefix}/profile/user` | GET | 获取用户画像 |
| `/{prefix}/profile/user/update` | POST | 更新用户画像 |
| `/{prefix}/profile/user/delete` | POST | 删除用户画像 |
| `/{prefix}/profile/groups` | GET | 获取群聊列表 |
| `/{prefix}/profile/users` | GET | 获取用户列表 |

### Stats API

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/stats/token` | GET | 获取 Token 使用统计 |
| `/{prefix}/stats/memory` | GET | 获取记忆统计（L1/L2/L3） |
| `/{prefix}/stats/kg` | GET | 获取知识图谱统计 |
| `/{prefix}/stats/system` | GET | 获取系统状态（组件状态、运行时间） |
| `/{prefix}/stats/all` | GET | 获取所有统计（合并以上全部） |

### Data API（导入导出）

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/data/l2/export` | GET | 导出 L2 记忆（JSON 文件下载） |
| `/{prefix}/data/l2/import` | POST | 导入 L2 记忆 |
| `/{prefix}/data/l3/export` | GET | 导出 L3 知识图谱 |
| `/{prefix}/data/l3/import` | POST | 导入 L3 知识图谱 |
| `/{prefix}/data/profile/export` | GET | 导出画像 |
| `/{prefix}/data/profile/import` | POST | 导入画像 |
| `/{prefix}/data/all/export` | GET | 全量导出（L2 + L3 + 画像） |
| `/{prefix}/data/all/import` | POST | 全量导入 |

### Manage API（管理操作）

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/manage/l1/clear` | POST | 清空 L1 缓冲 |
| `/{prefix}/manage/l2/delete` | POST | 删除 L2 记忆（按 scope） |
| `/{prefix}/manage/l3/delete` | POST | 删除 L3 图谱（按 scope） |
| `/{prefix}/manage/l3/merge-duplicates` | POST | 合并 L3 重复节点 |
| `/{prefix}/manage/profile/delete` | POST | 删除画像（按 scope） |
| `/{prefix}/manage/tasks/trigger` | POST | 手动触发定时任务 |
| `/{prefix}/manage/tasks/status` | GET | 获取任务运行状态 |

### Hidden Config API（隐藏配置）

| 路径 | 方法 | 说明 |
|------|------|------|
| `/{prefix}/hidden-config` | GET | 获取所有隐藏配置（含默认值、类型、分组） |
| `/{prefix}/hidden-config/update` | POST | 批量更新隐藏配置 |
| `/{prefix}/hidden-config/delete` | POST | 删除单个隐藏配置项（恢复默认值） |
| `/{prefix}/hidden-config/reset` | POST | 重置所有隐藏配置为默认值 |

## 前端页面

| 路由 | 页面 | 功能 |
|------|------|------|
| `/dashboard` | DashboardView | 系统状态总览（组件状态、Token 统计、记忆统计） |
| `/l1-buffer` | L1BufferView | L1 消息缓冲查看（队列列表、消息浏览） |
| `/l2-memory` | L2MemoryView | L2 记忆库搜索与管理（搜索、浏览、编辑、删除） |
| `/l3-graph` | L3GraphView | L3 知识图谱可视化（节点/边浏览、路径扩展） |
| `/profile` | ProfileView | 画像查看与编辑（用户画像、群聊画像） |
| `/data-manage` | DataManageView | 数据导入导出（L2/L3/画像/全量） |
| `/hidden-config` | HiddenConfigView | 隐藏配置热修改（分组展示、实时更新） |

## 开发指南

### 添加新 API

1. 在 `routes/` 目录创建新路由文件
2. 定义处理函数和 `register_xxx_routes(context)` 注册函数
3. 在 `routes/__init__.py` 中导出注册函数
4. 在 `web/__init__.py` 的 `register_all_routes()` 中调用注册函数

### 添加前端页面

1. 在 `frontend/src/views/` 创建 Vue 组件
2. 在 `frontend/src/router/index.ts` 添加路由配置
3. 在 `frontend/src/api/` 创建对应的 API 封装
4. 在 `frontend/src/stores/` 创建 Pinia Store（如需要）
5. 运行 `npm run build` 构建到 `pages/iris/`

### 构建部署

```bash
cd iris_memory/web/frontend
npm run build
```

构建产物自动输出到 `pages/iris/index.html`（单文件模式），AstrBot 自动托管该页面。
