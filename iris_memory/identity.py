"""
Iris Chat Memory - 用户身份映射

AstrBot 内置 webchat 把登录名同时填入 user_id 和 nickname，同一个人在
网页面板与 IM 平台上因此是两个 user_id。平台不提供两者的对应关系，
只能由管理员通过 ``identity.user_id_aliases`` 显式声明。
"""

import json
from functools import lru_cache
from typing import Any, Dict

from .core import get_logger

logger = get_logger("platform.identity")


def _normalize_aliases(raw: dict) -> Dict[str, str]:
    aliases = {str(k).strip(): str(v).strip() for k, v in raw.items()}
    return {source: target for source, target in aliases.items() if source and target}


@lru_cache(maxsize=8)
def _parse_alias_text(text: str) -> Dict[str, str]:
    # 每条消息会多次解析身份；按配置文本缓存，配置无效时也只告警一次
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"identity.user_id_aliases 不是合法 JSON，已忽略：{e}")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("identity.user_id_aliases 不是 JSON 对象，已忽略")
        return {}
    return _normalize_aliases(parsed)


def parse_user_id_aliases(raw: Any) -> Dict[str, str]:
    """解析别名配置，配置无效时返回空表，不影响消息链路"""
    if isinstance(raw, dict):
        return _normalize_aliases(raw)
    if isinstance(raw, str) and raw.strip():
        return _parse_alias_text(raw)
    return {}


def resolve_user_id(raw_id: str) -> str:
    """按别名配置返回规范 user_id，未命中时原样返回"""
    from .config import get_config

    try:
        aliases = parse_user_id_aliases(get_config().get("identity.user_id_aliases", ""))
    except RuntimeError:
        return raw_id  # 配置尚未初始化
    return aliases.get(raw_id, raw_id)
