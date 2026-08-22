# qq_official 平台适配器完整支持 — 实施计划

## 背景

`factory.py:43` 中 `qq_official` 注册为 `None`,降级到 GenericAdapter,丢失了本可用的能力:引用消息解析、图片提取、被@用户提取、message_id 元数据(L1 回填配对依赖)。调研结论:AstrBot 转换 qq_official 消息时已把引用内容、附件、@信息解析进消息链,新适配器采用与 OneBot 相同的"链优先"策略即可拿到大部分能力。

已确认的决策:
- 群/单聊无昵称场景,适配器返回 openid 派生稳定标签(群 `成员_{member_openid[:6]}`、单聊 `用户_{user_openid[:6]}`),群聊记忆可区分说话人;
- proactive 频控降级延后单独做(本次不改 proactive 模块);
- 同时注册 `qq_official` 与 `qq_official_webhook` 两个平台键(webhook 版复用同一解析逻辑,数据形状一致)。

## 改动清单(3 个新文件 + 3 个修改)

### 1. 新建 `iris_memory/platform/qq_official.py` — QQOfficialAdapter

不 import botpy(插件不依赖它),通过 raw 载荷特征字段探测场景:

- `_raw_payload(event)`:读 `event.message_obj.raw_message.raw_data`(AstrBot Patched* 消息对象挂载;botpy 对象全部 `__slots__` 无 `__dict__`,通用 `__dict__` 回退必然失败);raw_data 缺失时降级返回 {}。
- `_detect_scene(raw_data)`:`group_openid` → group;`author.user_openid` → c2c;`direct_message` → guild_dm;`channel_id` → guild_channel。

逐方法实现:

| 方法 | 实现 |
|---|---|
| get_user_id / get_group_id / is_group_message / get_session_id | 标准 message_obj 字段,与 Generic 一致;群=group_openid、频道=channel_id、单聊/私信=`private:{openid}` |
| get_user_name / get_user_nickname | 频道/频道私信场景返回真实 `sender.nickname`(username);群/单聊场景 nickname 为空时返回稳定标签 `成员_{openid[:6]}` / `用户_{openid[:6]}`(openid 取自 raw_data.author);raw 不可用回退空串 |
| get_group_name | 返回 ""(平台无 API;保留 `_structured_group_name` 首选) |
| get_user_role | 私聊 "private";guild_channel 从 raw_data.member.roles 映射("4"→owner、"2"/"3"→admin、其余 member,防御式解析);群/单聊恒 "member" |
| get_raw_message | 返回 raw_data 拷贝,**注入 `message_id` = raw 的 `id` 键**(消费点 `message_hook.py:334/548` 读 "message_id";在适配器侧归一化,不动共享路径) |
| get_reply_info | 链优先:Reply 组件(id/message_str/chain,sender_id=0、sender_nickname="" 清洗,复用 OneBot 的链解析模式);fallback raw_data.message_reference.message_id。群引用(msg_elements type=103)已被 AstrBot 归一化进链,无需自解析;被引者名字拿不到时由现有 `_backfill_reply_from_buffer` 兜底 |
| get_mentioned_users | 链上 At 组件,**排除标记 At**(`qq=="qq_official"`、`qq==self_id`、`qq=="all"`——AstrBot 会塞机器人标记 At,不排除会把机器人记成被@用户);guild_channel 场景从 raw_data.mentions 补充提取(排除 `bot:true` 与 self);群场景 mentions 只含机器人,自然为空 |
| get_images | 链上 Image 组件(source="user")+ Reply.chain 内 Image(source="forward"),url 取 `component.url or component.file`(AstrBot 已归一化 https 前缀) |
| get_msg_by_id | 仅 guild_channel/guild_dm:`event.bot.api.get_message(channel_id, message_id)`(asyncio.wait_for 5s),解析返回的 author/content/attachments;群/单聊无此 API,直接返回空 |
| get_forward_messages | 继承基类返回 [](平台无合并转发 API) |

### 2. 修改 `iris_memory/platform/factory.py`

- import QQOfficialAdapter;registry:`"qq_official": QQOfficialAdapter`、`"qq_official_webhook": QQOfficialAdapter`;
- 更新模块 docstring(移除"待实现,降级到通用适配器")。

### 3. 修改 `iris_memory/platform/__init__.py`

- 导出 QQOfficialAdapter、加入 `__all__`、更新模块 docstring 平台列表。

### 4. 扩展 `tests/platform/fakes.py`

沿用"形状漂移探针"思路:

- `FakeBotpyRawMessage`:带 `__slots__`、**刻意无 `__dict__`**、挂 raw_data/message_type/msg_elements 三属性,对齐 AstrBot Patched* 形状;
- `make_qq_official_event(scene=...)` 工厂:四种场景各自构造对齐 botpy 载荷的 raw_data(群:group_openid+member_openid+mentions 只含机器人;单聊:user_openid;频道:channel_id+author.id/username+member.roles+mentions 含真实用户;私信:direct_message),链使用真实 astrbot 组件(Reply/标记 At/Plain/Image);
- fake bot:`api.get_message` 为 AsyncMock。

### 5. 新建 `tests/platform/test_qq_official_adapter.py`

按仓库风格(类分组 + test_snake_case + 中文 docstring):

- 四场景基础 getters(user_id/group_id/session_id/真实用户名 vs openid 标签的稳定性);
- get_raw_message:slotted 探针走 raw_data、message_id 键归一化("id"→"message_id")、raw_data 缺失降级 {};
- get_reply_info:链 Reply 解析、sender_id=0 清洗、非引用返回空;
- get_mentioned_users:标记 At 排除、频道 mentions 提取(排除 bot)、群场景为空;
- get_images:主消息与 Reply.chain 图片的 source 区分;
- get_msg_by_id(asyncio mark + AsyncMock):频道场景成功并断言调用参数、群/单聊不调 API 直接空、超时/异常降级;
- get_user_role:频道角色映射、群恒 member、私聊 private;
- 工厂:两个平台键均返回 QQOfficialAdapter 实例(现有 TestGetAdapter 用 "wechat" 测降级路径,不受影响)。

### 6. 文档

- README「安装前须知」后新增简短「平台支持说明」:aiocqhttp 全功能;qq_official 支持 L1/引用/图片/频道@,并明示平台限制(无昵称→成员标签、无群名、无合并转发、同一用户群/单聊/频道身份隔离、主动消息约 4 条/用户/月);
- CHANGELOG `[Unreleased]`:Added(适配器)+ Tests 条目。

## 明确不做(范围外)

- proactive 平台频控降级(已确认延后,AstrBot 侧已有 msg_id 缓存与主动降级,不会报错);
- message_hook 共享路径改造(适配器侧键归一化替代,零回归风险);
- 频道群名 API 拉取、合并转发(平台无能力)。

## 验证

1. `./.venv/bin/python -m pytest tests/platform/ -q`,通过后全量 `./.venv/bin/python -m pytest tests/ -q`;
2. 手动验收清单(需真实 qq_official 部署,由你执行):群@消息入 L1 且 message_id 正确、引用消息内容与图片回填、频道消息 mentions 与角色映射、单聊标签跨消息稳定。

## 风险与缓解

- AstrBot 版本差异:raw_data 挂载是 AstrBot 4.x 的 Patched 行为;旧版缺失时适配器降级到 message_obj 标准字段(链数据来自 AstrBot 转换不受影响,仅 raw 补充路径失效);
- openid 标签进入画像 historical_names:如后续平台下发真名,现有 update_user_name 会自动覆盖更新。

工作量估计:适配器约 300 行 + 测试约 400 行,合计 1.5~2 天。