"""Provider 相关工具函数。"""

from typing import Any, Optional, Tuple


def normalize_provider_id(provider_value: Any) -> str:
    """将 provider 配置值规范化为字符串 ID。"""
    if provider_value is None:
        return ""

    if isinstance(provider_value, dict):
        for key in ("value", "provider_id", "id"):
            value = provider_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    provider_id = str(provider_value).strip()
    if provider_id.lower() in {"", "none", "null"}:
        return ""
    return provider_id


def extract_provider_id(provider: Any) -> Optional[str]:
    """从 provider 对象中尽可能提取 provider_id。"""
    if provider is None:
        return None

    direct_id = normalize_provider_id(
        getattr(provider, "id", None) or getattr(provider, "provider_id", None)
    )
    if direct_id:
        return direct_id

    try:
        meta = provider.meta() if hasattr(provider, "meta") else None
        meta_id = normalize_provider_id(
            getattr(meta, "id", None) or getattr(meta, "provider_id", None)
        )
        if meta_id:
            return meta_id
    except Exception:
        pass

    return None


def get_provider_by_id(context: Any, provider_id: Any) -> Tuple[Optional[Any], Optional[str]]:
    """根据 provider_id 获取 provider 对象（适配多种 AstrBot API 形态）。"""
    pid = normalize_provider_id(provider_id)
    if not pid or pid == "default" or not context:
        return None, None

    try:
        if hasattr(context, "get_provider_by_id"):
            provider = context.get_provider_by_id(provider_id=pid)
            if provider is not None:
                return provider, (extract_provider_id(provider) or pid)
    except TypeError:
        try:
            provider = context.get_provider_by_id(pid)
            if provider is not None:
                return provider, (extract_provider_id(provider) or pid)
        except Exception:
            pass
    except Exception:
        pass

    try:
        providers = context.get_all_providers() if hasattr(context, "get_all_providers") else []
        for provider in providers:
            candidate = extract_provider_id(provider)
            if candidate == pid:
                return provider, candidate
            if candidate and candidate.lower() == pid.lower():
                return provider, candidate
    except Exception:
        pass

    return None, None


def get_default_provider(context: Any, umo: str = "") -> Tuple[Optional[Any], Optional[str]]:
    """获取默认 provider（适配 get_using_provider 的不同签名）。"""
    if not context:
        return None, None

    provider = None
    try:
        provider = context.get_using_provider(umo=umo)
    except TypeError:
        try:
            provider = context.get_using_provider()
        except Exception:
            provider = None
    except Exception:
        provider = None

    if provider is not None:
        return provider, extract_provider_id(provider)

    try:
        providers = context.get_all_providers() if hasattr(context, "get_all_providers") else []
        if providers:
            provider = providers[0]
            return provider, extract_provider_id(provider)
    except Exception:
        pass

    return None, None