# AstrBot Iris Memory Plugin Docker 测试环境

本目录包含用于本地测试 AstrBot Iris Memory 插件的 Docker 配置。

## 快速开始

### 1. 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0

### 2. 启动测试环境

```bash
# 进入 docker 目录
cd docker

# 启动服务
./manager.sh start
```

### 3. 访问服务

- **AstrBot Web UI**: http://localhost:6185

### 4. 查看日志

```bash
./manager.sh logs
```

### 5. 进入容器调试

```bash
./manager.sh shell
```

## 管理脚本命令

| 命令 | 说明 |
|------|------|
| `./manager.sh start` | 启动测试环境 |
| `./manager.sh stop` | 停止测试环境 |
| `./manager.sh restart` | 重启测试环境 |
| `./manager.sh logs` | 查看实时日志 |
| `./manager.sh shell` | 进入容器 Shell |
| `./manager.sh shell-root` | 以 root 身份进入容器 |
| `./manager.sh status` | 查看容器状态 |
| `./manager.sh test` | 在容器中运行插件测试 |
| `./manager.sh build` | 重新构建镜像 |
| `./manager.sh update` | 更新镜像并重启 |
| `./manager.sh clean` | 清理所有数据和镜像 |
| `./manager.sh help` | 显示帮助信息 |

## 目录结构

```
docker/
├── Dockerfile              # AstrBot + Iris Memory 镜像定义
├── docker-compose.yml      # 服务编排配置
├── manager.sh              # 管理脚本
├── .dockerignore           # Docker 构建忽略文件
└── README.md               # 本文件
```

## 开发模式

Docker 环境使用卷挂载（volume mount）将本地代码映射到容器中：

- `iris_memory/` → 插件核心代码
- `main.py` → 插件入口
- `metadata.yaml` → 插件元数据
- `_conf_schema.json` → 配置模式
- `requirements.txt` → 依赖列表

这意味着你可以**实时修改本地代码**，无需重建镜像即可生效。

## 数据持久化

以下数据通过 Docker 卷持久化：

- `astrbot_data` - AstrBot 数据和配置

## 故障排除

### 端口冲突

如果 6185 或 6186 端口被占用，修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8080:6185"  # 将主机 8080 映射到容器 6185
```

### 权限问题

如果遇到权限错误，使用 root 身份进入容器：

```bash
./manager.sh shell-root
```

### 插件未加载

1. 检查日志：`./manager.sh logs`
2. 确认文件已正确挂载到 `/AstrBot/data/plugins/astrbot_plugin_iris_memory/`
3. 重启服务：`./manager.sh restart`

### 内存不足

如果容器因内存不足崩溃，增加 Docker 内存限制或在 `docker-compose.yml` 中添加：

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## 高级用法

### 自定义配置

创建 `docker-compose.override.yml` 覆盖默认配置：

```yaml
version: '3.8'
services:
  astrbot:
    environment:
      - CUSTOM_ENV=value
    volumes:
      - /path/to/custom/config:/AstrBot/data/config
```


