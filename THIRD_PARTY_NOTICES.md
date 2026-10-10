# 来源与许可兼容说明

## iris-connectors

- 来源仓库：[AirObject/iris-connectors](https://github.com/AirObject/iris-connectors)
- 固定提交：`a532e2bfd5313030c12f99982c013a4438547c03`（简称 `a532e2b`）。
- Python 客户端原路径 `python/src/iris_client/`，vendored 到 `iris_memory/vendor/iris_client/`。使用包内相对导入，不覆盖环境里的其他 `iris_client`。
- AstrBot 原路径 `astrbot/astrbot_plugin_iris/`，桥接、配置、映射、媒体、提示、收件箱移入 `iris_memory/`。保留原始设计文档于 `docs/CONNECTOR_DESIGN.md`，其中安装路径及环境变量说明属于上游快照，本插件行为以 README 为准。
- 客户端／AstrBot 回归测试及公开 `contracts/host-v1-examples.json` 一并移植，后者位于 `tests/fixtures/`。

本次客户端除格式整理外，仅加强入口 ID 校验及 search 入口参数校验，以适配 core 上线前约束；Python 3.12 类型别名语法保留。桥接增加 prepare 预算字段、成功后的 contexts 接管、有限的入口状态映射、独立消息提交预算与主动提醒文本清理；提示保留生命周期、白名单来源元数据与供目标工具使用的目标编号。配置改为只读 AstrBot 配置令牌，增加工具／小工具开关；QQ 官方身份空间和图片组件属性兜底作适配。其余新增接线、服务、命令、状态页和小工具均为本次实现。

该固定上游快照没有 LICENSE、NOTICE 文件，Python 包元数据也未声明许可证。**不能据此推断上游已按 AGPL、MIT 或其他许可证授权，不能宣称许可兼容已被证明。** 本次按仓库所有者明确要求集成，保留来源与此许可缺口说明；公开再分发前需由权利人确认上游授权或补充兼容许可证，本仓库不替上游擅自重新许可。

## v3 与 AstrBot

插件 v3 README 声明 AGPL-3.0；本次新编写的插件实现沿用该声明，不将其自动延伸为上游 vendored 代码的许可结论。v3 仅作为功能与行为参考，未整体搬运其实现。AstrBot 4.28.2 声明 AGPL-3.0-or-later，只用于 API 核对与运行宿主，没有 vendoring AstrBot 源码。

运行依赖 httpx 按其自身许可证分发。第三方原有版权与许可声明如后续补齐，升级 vendor 时应同时保留并复核。
