# 更新日志

## 4.0.0-dev — 2026-10-11

- Pages 新增运行日志、prepare 检索元数据和主动门控／决策统计；采集有界、仅内存、无正文和凭据，兼容缺失入口画像，复用登录校验。
- 增加默认关闭的 `allow_container_host`，支持同一可信 Docker 网络的单一主机名 HTTP(S) origin；严格校验地址，日志与 Pages 提示 HTTP 明文令牌传输。
- README 测试版统一安装 `dev.4`，补充 Iris 0.0.1 镜像的宿主机／容器两种部署、首次设置码与管理页令牌配置，更新入口画像、批量操作和定时备份状态。
- 清空 dev.4 旧实现并保留 Git 历史，基于 iris-connectors `a532e2b` 的 Python 客户端与 AstrBot 接线重建。
- 客户端 vendored；插件名保持 `astrbot_plugin_iris_memory`，要求 AstrBot >=4.28.2,<5、Python >=3.12。
- 接入消息接收、prepare 六分区注入、成功发送后的输出回流、可选使用反馈、持久提醒收件箱及图片上传。
- 增加上下文配置检查、Iris 近期上下文接管、入口映射与严格 ID 校验、QQ 官方身份空间适配。
- 提供七个入口范围工具与 status/search/persona/goals/learn/help 指令；跟进由 Iris 目标承接，使用 host_key 与修订保护。
- 重写纯 @ 回复、错误消息友好化与 Markdown 清理，增加 AstrBot Pages 最小状态页。
- 基于五个生命周期钩子实现主动回复：逐会话默认关闭，本地信号／冷却／静音／意愿门控，单次 AstrBot Provider 决策并直接生成内容。
- 支持自然插话、Iris 目标跟进、只观察、静默概率预约、滚动限额，以及独立开关的私聊主动关心；发送前复查新消息与控制变更，失败不回流。
- 增加管理员 `/iris_reply` 指令、`set_cooldown` 工具、每日决策上限、持久统计和 Pages 会话设置；主动提醒与会话开关、静音和冷却一致。
- prepare 可选 `$entry_profile` 注入普通回复与主动决策；兼容缺失／null 的旧服务。
- 补全 v3 能力与延期对照，标明 Iris 入口画像、批量遗忘／删除、定时备份的归属；v3 数据迁移明确不做。删除临时 PL-2 交接文档。
- 不再内置 L2／L3／梦境数据库；不提供独立知识图谱、旧表达／黑话学习管线、对话样例或 v3 数据迁移。
- 增加客户端回归、AstrBot 替身、假 HTTP 与真实空模型 iris-core 集成测试。
- 补齐 GNU AGPL-3.0 许可全文，明确插件为 AGPL-3.0-only；记录 vendored 来源、改动及 iris-connectors 作者的同许可证授权（`a3784d8`）。
