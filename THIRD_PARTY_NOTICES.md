# 来源与许可兼容说明

## iris-connectors

- 来源仓库：[AirObject/iris-connectors](https://github.com/AirObject/iris-connectors)
- Python 客户端同步提交：[`78633912a0466f9787724f90173c6c6bf3335d26`](https://github.com/AirObject/iris-connectors/commit/78633912a0466f9787724f90173c6c6bf3335d26)（main，含客户端提醒事件流订阅 PR #6）。
- AstrBot 接线、设计文档及原有契约／测试参考提交：`a532e2bfd5313030c12f99982c013a4438547c03`（简称 `a532e2b`）。
- Python 客户端原路径 `python/src/iris_client/`，vendored 到 `iris_memory/vendor/iris_client/`。使用包内相对导入，不覆盖环境里的其他 `iris_client`。
- AstrBot 原路径 `astrbot/astrbot_plugin_iris/`，桥接、配置、映射、媒体、提示、收件箱移入 `iris_memory/`。保留原始设计文档于 `docs/CONNECTOR_DESIGN.md`，其中安装路径及环境变量说明属于上游快照，本插件行为以 README 为准。
- 客户端／AstrBot 回归测试及公开 `contracts/host-v1-examples.json` 一并移植，后者位于 `tests/fixtures/`。

客户端目录八个文件完整同步上述 `7863391` 提交，包含 `events.py` 与 `py.typed`；仅按本仓库 Ruff 规则整理格式，没有保留旧 vendored 本地行为补丁。严格入口 ID 校验移至插件配置／映射边界，Python 3.12 类型别名语法保留。桥接增加 prepare 预算字段、成功后的 contexts 接管、有限的入口状态映射、独立消息提交预算与主动提醒文本清理；提示保留生命周期、白名单来源元数据与供目标工具使用的目标编号。配置改为只读 AstrBot 配置令牌，增加工具／小工具开关；QQ 官方身份空间和图片组件属性兜底作适配。SSE 的持久状态适配、提醒队列去重、后台生命周期和通道诊断由插件实现；其余新增接线、服务、命令、状态页和小工具均为本次实现。

AirObject/iris-connectors 的全部代码已由作者以 **AGPL-3.0-only** 授权；授权提交为 [`a3784d8`](https://github.com/AirObject/iris-connectors/commit/a3784d818e73da93bfada1f4ef348cfce486aa0b)，其中加入 LICENSE，并在 Python／TypeScript 包元数据中声明许可证。本插件按上述两个固定来源提交维护客户端与原有接线，按作者授权以 AGPL-3.0-only 分发，与本插件许可一致。许可全文见根目录 [LICENSE](LICENSE)。

## v3 与 AstrBot

本插件以 **AGPL-3.0-only** 分发。插件 v3 README 声明 AGPL-3.0；v3 仅作为功能与行为参考，未整体搬运其实现。AstrBot 4.28.2 声明 AGPL-3.0-or-later，只用于 API 核对与运行宿主，没有 vendoring AstrBot 源码。

运行依赖 httpx 按其自身许可证分发。升级 vendor 时应同时保留并复核第三方原有版权与许可声明。
