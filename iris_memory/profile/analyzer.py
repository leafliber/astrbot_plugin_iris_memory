"""
Iris Chat Memory - 画像分析器

使用 LLM 分析画像特征。
仅保留中长期字段分析。
"""

from typing import List, Dict, Union, TYPE_CHECKING
import json

from iris_memory.core import get_logger
from iris_memory.config import get_config
from .models import UpdateTier

if TYPE_CHECKING:
    from iris_memory.llm import LLMManager

logger = get_logger("profile")


# LLM 分析时从画像字典中剥离的纯元数据字段（对分析无价值，徒增 token）
_ANALYSIS_STRIP_KEYS = frozenset({"field_meta", "update_tracker", "version"})


def _slim_profile_dict(profile: Dict) -> Dict:
    """剥离 LLM 分析无用的元数据字段并过滤空值，减少 prompt token。"""
    slimmed: Dict = {}
    for k, v in profile.items():
        if k in _ANALYSIS_STRIP_KEYS:
            continue
        if v in (None, "", [], {}):
            continue
        slimmed[k] = v
    return slimmed


def _truncate_messages(messages: List[str], max_chars: int) -> List[str]:
    """按 max_chars 截断每条消息（0 表示不截断）。"""
    if max_chars <= 0:
        return messages
    return [(m[:max_chars] + "…" if len(m) > max_chars else m) for m in messages]


def _get_max_chars() -> int:
    """读取画像分析单条消息最大字符数配置。"""
    try:
        return int(get_config().get("profile_message_max_chars", 150))
    except RuntimeError:
        return 150


class ProfileAnalyzer:
    """画像分析器

    使用 LLM 分析画像特征。
    仅保留中长期字段分析。

    Attributes:
        _llm_manager: LLM 调用管理器实例
    """

    def __init__(self, llm_manager: "LLMManager"):
        self._llm_manager = llm_manager

    async def analyze_group_profile(
        self,
        messages: List[str],
        current_profile: Dict,
        tier: UpdateTier = UpdateTier.MID,
        combined: bool = False,
    ) -> Dict[str, List[str]]:
        """分析群聊画像

        Args:
            messages: 近期对话消息列表
            current_profile: 当前画像数据（字典格式）
            tier: 更新层级，决定分析深度
            combined: 是否合并 MID+LONG 为单次调用

        Returns:
            分析结果字典
        """
        prompt = self._build_group_analysis_prompt(
            messages, current_profile, tier, combined=combined
        )

        try:
            response = await self._llm_manager.generate_direct(
                prompt=prompt, module="profile_analysis"
            )

            result = self._parse_json_response(response)
            label = "combined" if combined else tier.value
            logger.info(f"群聊画像分析完成 (tier={label})")
            return result

        except Exception as e:
            logger.error(f"群聊画像分析失败: {e}")
            return {}

    async def analyze_user_profile(
        self,
        messages: List[str],
        current_profile: Dict,
        tier: UpdateTier = UpdateTier.MID,
        combined: bool = False,
    ) -> Dict[str, Union[str, List[str]]]:
        """分析用户画像

        Args:
            messages: 用户近期对话列表
            current_profile: 当前画像数据（字典格式）
            tier: 更新层级，决定分析深度
            combined: 是否合并 MID+LONG 为单次调用

        Returns:
            分析结果字典
        """
        prompt = self._build_user_analysis_prompt(
            messages, current_profile, tier, combined=combined
        )

        try:
            response = await self._llm_manager.generate_direct(
                prompt=prompt, module="profile_analysis"
            )

            result = self._parse_json_response(response)
            label = "combined" if combined else tier.value
            logger.info(f"用户画像分析完成 (tier={label})")
            return result

        except Exception as e:
            logger.error(f"用户画像分析失败: {e}")
            return {}

    def _build_group_analysis_prompt(
        self,
        messages: List[str],
        current_profile: Dict,
        tier: UpdateTier = UpdateTier.MID,
        combined: bool = False,
    ) -> str:
        """构建群聊画像分析 prompt

        Args:
            messages: 对话消息列表
            current_profile: 当前画像数据
            tier: 更新层级
            combined: 是否合并 MID+LONG

        Returns:
            分析提示词
        """
        try:
            config = get_config()
            max_messages = config.get("profile_max_messages_for_analysis", 50)
        except RuntimeError:
            max_messages = 50
        limited_messages = messages[-max_messages:]
        if len(messages) > max_messages:
            logger.debug(
                f"群聊画像分析消息截断：原始 {len(messages)} 条 → 保留最近 {max_messages} 条"
            )

        if combined:
            return self._build_group_combined_prompt(limited_messages, current_profile)
        if tier == UpdateTier.LONG:
            return self._build_group_long_prompt(limited_messages, current_profile)

        max_chars = _get_max_chars()
        truncated = _truncate_messages(limited_messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)

        return f"""分析以下群聊对话，提取群聊画像特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "interests": ["兴趣点"],
    "atmosphere_tags": ["氛围标签"],
    "custom_fields": {{"字段名": "值"}}
}}

1. interests 返回完整列表（旧兴趣+新发现），最多5个，不再活跃的不要保留
2. atmosphere_tags 最多3个
3. custom_fields 最多3个，key 简短明确，复用已有 key
4. 不确定的返回空数组

仅返回JSON。"""

    def _build_group_long_prompt(
        self, messages: List[str], current_profile: Dict
    ) -> str:
        """构建群聊画像长期分析 prompt

        长期分析更关注核心特征和禁忌话题。

        Args:
            messages: 对话消息列表
            current_profile: 当前画像数据

        Returns:
            长期分析提示词
        """
        max_chars = _get_max_chars()
        truncated = _truncate_messages(messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)

        return f"""深度分析以下群聊对话，提取核心长期特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "long_term_tags": ["核心特征标签"],
    "blacklist_topics": ["禁忌话题"],
    "interests": ["兴趣（如有变化，返回完整替换列表）"],
    "atmosphere_tags": ["氛围标签（如有变化）"],
    "custom_fields": {{"字段名": "值"}}
}}

1. long_term_tags 描述群聊核心身份（如"技术交流群"），最多3个，宁可少标
2. blacklist_topics 必须非常确定才填写
3. "如有变化"字段：无显著变化返回空数组，有变化返回完整替换列表
4. custom_fields 最多3个，key 简短明确，复用已有 key
5. 不确定的返回空数组

仅返回JSON。"""

    def _build_group_combined_prompt(
        self, messages: List[str], current_profile: Dict
    ) -> str:
        """构建群聊画像合并分析 prompt（MID+LONG 单次调用）

        Args:
            messages: 对话消息列表
            current_profile: 当前画像数据

        Returns:
            合并分析提示词
        """
        max_chars = _get_max_chars()
        truncated = _truncate_messages(messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)

        return f"""深度分析以下群聊对话，一次性提取中期与长期特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "interests": ["兴趣（完整替换列表）"],
    "atmosphere_tags": ["氛围标签"],
    "long_term_tags": ["核心特征标签"],
    "blacklist_topics": ["禁忌话题"],
    "custom_fields": {{"字段名": "值"}}
}}

1. interests 返回完整列表，最多5个，不再活跃的不要保留
2. atmosphere_tags 最多3个
3. long_term_tags 描述群聊核心身份（如"技术交流群"），最多3个，宁可少标
4. blacklist_topics 必须非常确定才填写
5. custom_fields 最多3个，key 简短明确，复用已有 key
6. 不确定的返回空数组

仅返回JSON。"""

    def _build_user_analysis_prompt(
        self,
        messages: List[str],
        current_profile: Dict,
        tier: UpdateTier = UpdateTier.MID,
        combined: bool = False,
    ) -> str:
        """构建用户画像分析 prompt

        Args:
            messages: 用户对话列表
            current_profile: 当前画像数据
            tier: 更新层级
            combined: 是否合并 MID+LONG

        Returns:
            分析提示词
        """
        try:
            config = get_config()
            max_messages = config.get("profile_max_messages_for_user_analysis", 30)
        except RuntimeError:
            max_messages = 30
        limited_messages = messages[-max_messages:]
        if len(messages) > max_messages:
            logger.debug(
                f"用户画像分析消息截断：原始 {len(messages)} 条 → 保留最近 {max_messages} 条"
            )

        if combined:
            return self._build_user_combined_prompt(limited_messages, current_profile)
        if tier == UpdateTier.LONG:
            return self._build_user_long_prompt(limited_messages, current_profile)

        max_chars = _get_max_chars()
        truncated = _truncate_messages(limited_messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)
        favorability_enabled = self._favorability_enabled()

        return f"""分析以下用户对话，提取用户画像特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

用户近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "personality_tags": ["性格标签"],
    "interests": ["兴趣"],
    "language_style": "语言风格",
    "communication_style": "沟通偏好",
    "emotional_baseline": "情感基线",
{self._favorability_schema_line(favorability_enabled)}    "custom_fields": {{"字段名": "值"}}
}}

1. personality_tags 最多3个
2. interests 返回完整列表，最多5个，不再活跃的不要保留
3. communication_style 选：简洁/详细/随意/正式，无法判断留空
4. emotional_baseline 选：稳定/敏感/乐观/低落/焦虑，无法判断留空
{self._favorability_instruction_line(favorability_enabled)}{self._mid_tail_instructions(favorability_enabled)}
仅返回JSON。"""

    def _build_user_long_prompt(
        self, messages: List[str], current_profile: Dict
    ) -> str:
        """构建用户画像长期分析 prompt

        长期分析更关注职业、关系、重要事件等稳定特征。

        Args:
            messages: 用户对话列表
            current_profile: 当前画像数据

        Returns:
            长期分析提示词
        """
        max_chars = _get_max_chars()
        truncated = _truncate_messages(messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)

        return f"""深度分析以下用户对话，提取长期稳定特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

用户近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "occupation": "职业/身份",
    "bot_relationship": "对AI的称呼或关系设定",
    "important_events": ["重要事件"],
    "taboo_topics": ["禁忌话题"],
    "important_dates": [{{"date": "日期", "description": "描述"}}],
    "personality_tags": ["性格标签（如有变化）"],
    "interests": ["兴趣（如有变化）"],
    "language_style": "语言风格（如有变化）",
    "communication_style": "沟通偏好（如有变化）",
    "emotional_baseline": "情感基线（如有变化）",
    "custom_fields": {{"字段名": "值"}}
}}

1. 长期特征必须高度可靠，宁可留空不要猜测
2. occupation/bot_relationship 仅在有明确线索时填写
3. important_events 最多5个（工作变动、人生里程碑等）
4. "如有变化"字段：无变化返回空数组/空字符串，有变化返回完整替换值
5. communication_style 选：简洁/详细/随意/正式
6. emotional_baseline 选：稳定/敏感/乐观/低落/焦虑
7. custom_fields 最多3个，key 简短明确，复用已有 key

仅返回JSON。"""

    def _build_user_combined_prompt(
        self, messages: List[str], current_profile: Dict
    ) -> str:
        """构建用户画像合并分析 prompt（MID+LONG 单次调用）

        Args:
            messages: 用户对话列表
            current_profile: 当前画像数据

        Returns:
            合并分析提示词
        """
        max_chars = _get_max_chars()
        truncated = _truncate_messages(messages, max_chars)
        slim_profile = _slim_profile_dict(current_profile)
        favorability_enabled = self._favorability_enabled()

        return f"""深度分析以下用户对话，一次性提取中期与长期特征。

当前画像：
{json.dumps(slim_profile, ensure_ascii=False, indent=2)}

用户近期对话：
{chr(10).join(truncated)}

返回JSON：
{{
    "personality_tags": ["性格标签"],
    "interests": ["兴趣"],
    "language_style": "语言风格",
    "communication_style": "沟通偏好",
    "emotional_baseline": "情感基线",
{self._favorability_schema_line(favorability_enabled)}    "occupation": "职业/身份",
    "bot_relationship": "对AI的称呼或关系设定",
    "important_events": ["重要事件"],
    "taboo_topics": ["禁忌话题"],
    "important_dates": [{{"date": "日期", "description": "描述"}}],
    "custom_fields": {{"字段名": "值"}}
}}

1. personality_tags 最多3个
2. interests 返回完整列表，最多5个，不再活跃的不要保留
3. communication_style 选：简洁/详细/随意/正式，无法判断留空
4. emotional_baseline 选：稳定/敏感/乐观/低落/焦虑，无法判断留空
{self._favorability_instruction_line(favorability_enabled)}{self._combined_tail_instructions(favorability_enabled)}
仅返回JSON。"""

    @staticmethod
    def _favorability_enabled() -> bool:
        """读取好感度功能开关。"""
        try:
            return bool(get_config().get("profile.favorability_enable", True))
        except RuntimeError:
            return True

    @staticmethod
    def _favorability_schema_line(enabled: bool) -> str:
        """返回 JSON schema 中 favorability_delta 行（含缩进与换行）。"""
        if not enabled:
            return ""
        return '    "favorability_delta": 整数,\n'

    @staticmethod
    def _favorability_instruction_line(enabled: bool) -> str:
        """返回 favorability_delta 的说明行（已含编号与换行）。"""
        if not enabled:
            return ""
        return (
            "5. favorability_delta：用户对AI好感度变化量（-20~+20整数），"
            "根据语气友好度/互动积极性/冲突情况调整；无明显变化返回0\n"
        )

    @staticmethod
    def _mid_tail_instructions(favorability_enabled: bool) -> str:
        """构建 MID prompt 尾部说明（custom_fields + 不确定），编号随好感度开关调整。"""
        n = 6 if favorability_enabled else 5
        return (
            f"{n}. custom_fields 最多3个，key 简短明确，复用已有 key\n"
            f"{n + 1}. 不确定的返回空数组或空字符串\n"
        )

    @staticmethod
    def _combined_tail_instructions(favorability_enabled: bool) -> str:
        """构建 combined prompt 尾部说明（长期特征之后），编号随好感度开关调整。"""
        n = 6 if favorability_enabled else 5
        return (
            f"{n}. 长期特征（occupation/bot_relationship/important_events/taboo_topics/important_dates）必须高度可靠，宁可留空不要猜测\n"
            f"{n + 1}. occupation/bot_relationship 仅在有明确线索时填写\n"
            f"{n + 2}. important_events 最多5个（工作变动、人生里程碑等）\n"
            f"{n + 3}. custom_fields 最多3个，key 简短明确，复用已有 key\n"
            f"{n + 4}. 不确定的返回空数组或空字符串\n"
        )

    def _parse_json_response(self, response: str) -> Dict:
        """解析 JSON 响应

        尝试从 LLM 响应中提取 JSON 内容。

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典，失败返回空字典
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning(f"无法解析JSON响应: {response[:100]}")
            return {}
