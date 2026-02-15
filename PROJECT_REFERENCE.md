# Iris Memory 项目参考手册（维护向）

> 目标：沉淀后续开发/排障最常用的项目背景、接口约束与检查清单。
> 适用版本：AstrBot `v4.16.0`，插件版本 `v1.3.0`。

## 1. 项目定位与核心能力

- 插件：`astrbot_plugin_iris_memory`
- 核心模型：三层记忆（Working / Episodic / Semantic）
- 关键流程：记忆捕获 → RIF/多维评分 → 存储分层 → 检索注入 → LLM 响应后再捕获
- 核心增强：
  - 近期聊天上下文注入（`ChatHistoryBuffer`）
  - 多维度评分（`MultidimensionalScorer`）
  - 主动回复（`proactive_manager` + `message_sender`）

## 2. 关键模块速查

### 2.1 入口与主流程
- `main.py`：事件入口、LLM 请求前后处理、消息录入时机
- `iris_memory/services/memory_service.py`：记忆注入、构建提示词上下文、核心编排

### 2.2 分析与评分
- `iris_memory/analysis/rif_scorer.py`：传统 RIF + 多维评分切换与回退
- `iris_memory/analysis/multidimensional_scorer.py`：五维评分主体实现

### 2.3 存储
- `iris_memory/storage/chroma_manager.py`：Chroma 初始化、collection 管理、向量维度一致性
- `iris_memory/storage/chat_history_buffer.py`：会话级聊天记录滑动窗口

### 2.4 主动回复
- `iris_memory/proactive/proactive_manager.py`：任务调度、触发判定、发送调用
- `iris_memory/proactive/message_sender.py`：发送方法检测与发送适配

## 3. 近期关键变更（v1.3.0）

1. 新增主动回复群聊白名单模式及管理员指令
  - 新增 `proactive_reply.group_whitelist_mode` 配置，开启后仅在管理员通过指令启用的群聊触发主动回复
  - 管理员指令：`/proactive_reply on|off|status|list`，用于在群聊中控制主动回复开关
2. 新增近期聊天上下文注入（群聊/私聊）
3. 修复 Chroma 向量维度冲突（自动重建 collection）
4. 修复主动回复 `Context.send_message` 缺失 session 参数
5. 修复 `MessageChain` 类型错误（避免 `'list' object has no attribute 'chain'`）
6. 修复 `fallback_to_rif` 参数重复传递冲突
7. 修复 @Bot 消息重复写入聊天缓冲区

## 4. AstrBot 接口约束（高频踩坑）

## 4.1 Context.send_message

- 正确签名（AstrBot v4.16）：
  - `send_message(session, message_chain)`
- 关键约束：
  - `session` 需使用 `unified_msg_origin`（简称 `umo`）或 `MessageSession`
  - `message_chain` 必须是 `MessageChain` 对象，不可直接传 `list`

### 4.2 主动回复发送优先级（当前实现）

`provider_send` → `platform_send` → `context_send` → `service_send` → `event_send`

建议：
- 能用 provider/platform 时优先使用；
- 使用 `context_send` 时必须带 `umo`。

## 5. 已知问题模式与定位提示

### 5.1 向量维度错误
- 典型日志：`Collection expecting embedding with dimension of X, got Y`
- 根因：历史 collection 维度与当前 embedding provider 维度不一致
- 处理：`chroma_manager.initialize()` 中检测冲突后删除并重建 collection

### 5.2 主动回复发送失败（session）
- 典型日志：`Context.send_message() missing 1 required positional argument: 'session'`
- 根因：调用 `send_message` 时未传 `umo`
- 处理：保证 `proactive_manager -> message_sender.send(..., umo=task.umo)` 链路完整

### 5.3 主动回复发送失败（message_chain）
- 典型日志：`'list' object has no attribute 'chain'`
- 根因：把 `list[Plain]` 当成 `MessageChain` 传入
- 处理：构造 `MessageChain()` 并通过 `.message(content)` 添加文本

### 5.4 多维评分初始化冲突
- 典型日志：`got multiple values for keyword argument 'fallback_to_rif'`
- 根因：显式参数与 `**kwargs` 同时包含 `fallback_to_rif`
- 处理：在 `rif_scorer` 转发前过滤该键

## 6. 配置与版本文件

- 插件元数据版本：`metadata.yaml`（`version: vX.Y.Z`）
- Python 包版本：`pyproject.toml`（`version = "X.Y.Z"`）
- 发布说明：`CHANGELOG.md`

发版时三处必须同步。

## 7. 测试与回归命令

### 7.1 关键模块测试
```bash
.venv/bin/python -m pytest tests/proactive/test_message_sender.py -v
```

### 7.2 全量回归（当前项目常用）
```bash
.venv/bin/python -m pytest tests/ --ignore=tests/embedding -q
```

当前基线：`1000 passed`（含 warnings）。

## 8. 变更前后最小检查清单

1. 涉及发送链路时：检查是否传入 `umo`
2. 涉及 `context_send` 时：检查 `MessageChain` 类型
3. 涉及 embedding/provider 切换时：检查 collection 维度
4. 涉及评分器参数时：检查 `kwargs` 是否和显式参数冲突
5. 更新版本时：同步 `metadata.yaml` / `pyproject.toml` / `CHANGELOG.md`
6. 提交前至少跑一次：`tests/proactive` + 全量回归

## 9. 推荐阅读顺序（新人上手）

1. `README.md`（功能与安装）
2. `main.py`（插件事件生命周期）
3. `memory_service.py`（核心编排）
4. `message_sender.py` + `proactive_manager.py`（主动回复）
5. `rif_scorer.py` + `multidimensional_scorer.py`（评分机制）
6. `chroma_manager.py`（存储一致性）

---

如后续再出现新的“高频报错 + 已确认修复”，建议在本文件的「已知问题模式」追加一条，作为团队排障知识库。