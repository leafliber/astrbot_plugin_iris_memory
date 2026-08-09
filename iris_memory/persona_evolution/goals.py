"""
Iris Chat Memory - 人格自迭代目标预设

文档 §5.2 的 8 个默认目标方向。预设内容版本化：
每个 Revision 保存当次使用的目标完整快照（build_goal_snapshot），
后续修改预设不会改变历史 Revision 的解释。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from iris_memory.core import get_logger

logger = get_logger("persona_evolution.goals")

# 目标预设内容版本号：预设文本发生变更时递增
GOAL_PRESET_VERSION = 1

CUSTOM_PRESET_ID = "custom"


@dataclass(frozen=True)
class GoalPreset:
    """目标方向预设

    Attributes:
        preset_id: 预设标识（存储与 API 使用）
        display_name: 显示名称
        text: 完整目标文本（送模与快照使用）
    """

    preset_id: str
    display_name: str
    text: str


GOAL_PRESETS: Dict[str, GoalPreset] = {
    "natural": GoalPreset(
        preset_id="natural",
        display_name="自然拟人",
        text=(
            "减少模板感和机器腔，让表达更像自然的群聊发言，"
            "同时保持角色身份与人格设定稳定不变。"
        ),
    ),
    "warm": GoalPreset(
        preset_id="warm",
        display_name="温暖共情",
        text=(
            "更关注对方情绪，先回应感受再给出内容，"
            "语气温暖包容，避免过度说教和冷冰冰的纠正。"
        ),
    ),
    "concise": GoalPreset(
        preset_id="concise",
        display_name="简洁直接",
        text=(
            "缩短回复长度，先给结论再补充必要细节，"
            "减少重复、铺垫和无关的客套。"
        ),
    ),
    "humorous": GoalPreset(
        preset_id="humorous",
        display_name="幽默活泼",
        text=(
            "学习参考语料中轻松幽默的表达节奏，让回复更活泼，"
            "但不复制具体梗、不拿群友开玩笑、不使用攻击性表达。"
        ),
    ),
    "professional": GoalPreset(
        preset_id="professional",
        display_name="专业严谨",
        text=(
            "表达结构清楚、措辞准确，"
            "对不确定的内容明确说明不确定性，不编造事实。"
        ),
    ),
    "proactive": GoalPreset(
        preset_id="proactive",
        display_name="主动好奇",
        text=(
            "更善于追问细节、承接话题，"
            "在合适时主动提供下一步建议或延伸方向。"
        ),
    ),
    "group_style": GoalPreset(
        preset_id="group_style",
        display_name="贴近群风格",
        text=(
            "以参考语料的综合表达风格为主，整体贴近群聊氛围，"
            "不额外强化某个单一方向，保持角色身份稳定。"
        ),
    ),
    "custom": GoalPreset(
        preset_id="custom",
        display_name="自定义",
        text="",  # 使用 Job 的 custom_goal 文本
    ),
}


def get_goal_preset(preset_id: str) -> Optional[GoalPreset]:
    """按 ID 获取目标预设

    Returns:
        预设对象，不存在返回 None
    """
    return GOAL_PRESETS.get(preset_id)


def list_goal_presets() -> list[Dict[str, str]]:
    """列出全部目标预设（供 Web UI 下拉框）"""
    return [
        {
            "preset_id": p.preset_id,
            "display_name": p.display_name,
            "text": p.text,
        }
        for p in GOAL_PRESETS.values()
    ]


def build_goal_snapshot(preset_id: str, custom_goal: str = "") -> Dict[str, Any]:
    """构建目标完整快照（Revision 持久化用）

    custom 预设使用 custom_goal 文本；其他预设使用当前版本预设文本。
    preset_id 未知时降级为 custom 并记录警告。

    Returns:
        快照字典：{preset_id, preset_version, display_name, text, custom_goal}
    """
    preset = GOAL_PRESETS.get(preset_id)
    if preset is None:
        logger.warning(f"未知目标预设 {preset_id}，按 custom 处理")
        preset = GOAL_PRESETS[CUSTOM_PRESET_ID]
        preset_id = CUSTOM_PRESET_ID

    text = custom_goal.strip() if preset_id == CUSTOM_PRESET_ID else preset.text
    return {
        "preset_id": preset_id,
        "preset_version": GOAL_PRESET_VERSION,
        "display_name": preset.display_name,
        "text": text,
        "custom_goal": custom_goal.strip(),
    }
