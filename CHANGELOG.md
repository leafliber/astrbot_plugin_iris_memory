# 更新日志

## 4.0.0-dev — 2026-10-10

- 清空 dev.4 旧实现并保留 Git 历史，基于 iris-connectors `a532e2b` 的 Python 客户端与 AstrBot 接线重建。
- 客户端 vendored；插件名保持 `astrbot_plugin_iris_memory`，要求 AstrBot >=4.28.2,<5、Python >=3.12。
- 接入消息接收、prepare 六分区注入、成功发送后的输出回流、可选使用反馈、持久提醒收件箱及图片上传。
- 增加上下文配置检查、Iris 近期上下文接管、入口映射与严格 ID 校验、QQ 官方身份空间适配。
- 提供七个入口范围工具与 status/search/persona/goals/learn/help 指令；跟进由 Iris 目标承接，使用 host_key 与修订保护。
- 重写纯 @ 回复、错误消息友好化与 Markdown 清理，增加 AstrBot Pages 最小状态页。
- 为 PL-2 主动回复预留独立 factory 与生命周期挂接；本版不提供主动回复策略和面板。
- 不再内置 L2／L3／梦境数据库；不提供独立知识图谱、表达学习、黑话、对话样例或 v3 数据迁移。
- 增加客户端回归、AstrBot 替身、假 HTTP 与真实空模型 iris-core 集成测试；记录来源与上游许可缺口。
