"""Closed, bounded input validation shared by local controls and protocol DTOs."""

import json
import re
from urllib.parse import urlsplit

from .errors import ControlError

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ControlError("INVALID_IDENTIFIER")
    return value


def exact_fields(value, required, optional=()):
    if (
        type(value) is not dict
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ControlError("INVALID_FIELDS")
    return value


def integer(value, low=0, high=2**53 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ControlError("INVALID_INTEGER")
    return value


def boolean(value):
    if type(value) is not bool:
        raise ControlError("INVALID_BOOLEAN")
    return value


def origin(value):
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or any(c.isspace() or ord(c) < 32 for c in value)
    ):
        raise ControlError("INVALID_ORIGIN")
    try:
        p = urlsplit(value)
        if (
            p.scheme not in ("http", "https")
            or not p.hostname
            or p.username is not None
            or p.password is not None
            or p.path not in ("", "/")
            or p.query
            or p.fragment
            or "\\" in value
            or p.port == 0
        ):
            raise ValueError
        host = p.hostname.encode("idna").decode("ascii").lower()
        if ":" not in host and not re.fullmatch(r"[a-z0-9.-]+", host):
            raise ValueError
        if ":" in host:
            import ipaddress

            ipaddress.IPv6Address(host)
            host = f"[{host}]"
        port = p.port
        suffix = (
            f":{port}" if port and port != (443 if p.scheme == "https" else 80) else ""
        )
        return f"{p.scheme}://{host}{suffix}"
    except (ValueError, UnicodeError):
        raise ControlError("INVALID_ORIGIN") from None


def secret(value):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 4096
        or not re.fullmatch(r"[A-Za-z0-9._~+/=-]+", value)
    ):
        raise ControlError("INVALID_CREDENTIAL")
    return value


def encode(value, limit=32768):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(text.encode()) > limit:
            raise ControlError("PAYLOAD_TOO_LARGE", 413)
        return text
    except (TypeError, ValueError, RecursionError):
        raise ControlError("INVALID_JSON") from None


def decode(raw, limit=262144):
    if len(raw) > limit:
        raise ControlError("PAYLOAD_TOO_LARGE", 413)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ControlError("INVALID_JSON") from None


def reject_secret_echo(value, secrets):
    """Do not persist or display credentials reflected by a remote peer."""
    if isinstance(value, str):
        if any(token and token in value for token in secrets):
            raise ControlError("REMOTE_CREDENTIAL_ECHO", 502)
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_secret_echo(key, secrets)
            reject_secret_echo(item, secrets)
    elif isinstance(value, list):
        for item in value:
            reject_secret_echo(item, secrets)
