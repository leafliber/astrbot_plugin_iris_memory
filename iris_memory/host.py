"""AstrBot public API integration. Shared providers remain owned by AstrBot."""

import asyncio
import math

from .budget import tokens
from .errors import IrisError
from .identity import digest


class AstrBotHost:
    def __init__(self, context, logger):
        self.context, self.logger = context, logger
        self.persona_lock = asyncio.Lock()
        self.model_lock = asyncio.Semaphore(1)

    def provider_list(self):
        def describe(provider):
            meta = provider.meta()
            return {"id": meta.id, "model": provider.get_model()}

        return {
            "chat": [describe(p) for p in self.context.get_all_providers()],
            "embedding": [
                describe(p) for p in self.context.get_all_embedding_providers()
            ],
        }

    async def providers(self, control):
        settings = control.settings
        if settings["cognitive_provider"]:
            raise IrisError(
                "core_cognitive_scope_missing",
                "当前 Core 异步认知桥接不能绑定候选 Scope；请清空认知 Provider，详见接入报告",
            )
        if not settings["embedding_provider"]:
            return None, None
        from iris_memory_core.embedded_providers import (
            AsyncEmbeddingAdapter,
            VectorSpaceConfig,
        )

        provider = self.context.get_provider_by_id(settings["embedding_provider"])
        if provider is None or not callable(getattr(provider, "get_embeddings", None)):
            raise IrisError("provider_missing", "Embedding Provider 不存在")

        async def embed(texts):
            await control.budget.reserve(tokens(list(texts)))
            async with asyncio.timeout(settings["model_timeout"]):
                async with self.model_lock:
                    values = await provider.get_embeddings(list(texts))
            if len(values) != len(texts):
                raise IrisError("invalid_embedding", "Embedding 数量不匹配")
            normalized = []
            for vector in values:
                if len(vector) != settings["embedding_dimension"] or not all(
                    math.isfinite(v) for v in vector
                ):
                    raise IrisError("invalid_embedding", "Embedding 维度或数值无效")
                norm = math.sqrt(sum(v * v for v in vector))
                if norm == 0:
                    raise IrisError("invalid_embedding", "Embedding 返回零向量")
                normalized.append([v / norm for v in vector])
            control.logs.emit(
                "provider.embedding",
                count=len(texts),
                dimensions=settings["embedding_dimension"],
            )
            return normalized

        return AsyncEmbeddingAdapter(
            VectorSpaceConfig(
                model=settings["embedding_provider"]
                + "/"
                + provider.get_model()
                + "@"
                + settings["embedding_revision"],
                dimension=settings["embedding_dimension"],
            ),
            embed,
            timeout_seconds=settings["model_timeout"],
        ), None

    async def generate(
        self, control, prompt, *, system="", images=None, image_budget=0
    ):
        settings = control.settings
        provider_id = settings["chat_provider"]
        if not provider_id:
            raise IrisError("provider_missing", "请先选择插件模型 Provider")
        reservation = (
            tokens(prompt) + tokens(system) + settings["output_reserve"] + image_budget
        )
        if reservation > settings["model_window"]:
            raise IrisError("context_full", "模型请求超出配置窗口，未截断发送")
        await control.budget.reserve(reservation)
        control.logs.emit(
            "provider.request",
            provider=provider_id,
            reserved=reservation,
            body={"prompt": prompt, "system": system},
        )
        async with asyncio.timeout(settings["model_timeout"]):
            async with self.model_lock:
                response = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system,
                    image_urls=images or [],
                    max_tokens=settings["output_reserve"],
                )
        text = response.completion_text
        if not isinstance(text, str) or not text.strip():
            raise IrisError("model_empty", "模型没有返回可用正文")
        control.logs.emit("provider.response", provider=provider_id, body=text)
        return text

    async def send(self, umo, text):
        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Plain

        return await self.context.send_message(umo, MessageChain(chain=[Plain(text)]))

    async def check_persona_support(self):
        manager = getattr(self.context, "persona_manager", None)
        if manager is None or not all(
            callable(getattr(manager, name, None))
            for name in ("resolve_selected_persona", "create_persona", "update_persona")
        ):
            raise IrisError(
                "host_incompatible", "AstrBot 缺少公开人格管理接口，请升级到已验证版本"
            )
        from astrbot.api import sp

        if not callable(getattr(sp, "put_async", None)):
            raise IrisError("host_incompatible", "AstrBot 缺少会话配置接口")

    async def selected_persona(self, identity):
        cm = self.context.conversation_manager
        cid = await cm.get_curr_conversation_id(identity.umo)
        conversation = await cm.get_conversation(identity.umo, cid) if cid else None
        return await self.context.persona_manager.resolve_selected_persona(
            umo=identity.umo,
            conversation_persona_id=getattr(conversation, "persona_id", None),
            platform_name=identity.platform,
        )

    async def adopt_persona(self, control, identity, persona):
        from astrbot.api import sp

        async with self.persona_lock:
            manager = self.context.persona_manager
            row = await control.store.get("host_carriers", identity.umo)
            config = (
                await sp.get_async("umo", identity.umo, "session_service_config", {})
                or {}
            )
            selected, source, _, _ = await self.selected_persona(identity)
            carrier_id = "iris_v4_" + digest(identity.umo)[:24]
            if row:
                value = row["value"]
                if config.get("persona_id") != carrier_id:
                    raise IrisError(
                        "host_persona_conflict",
                        "宿主人格选择已被外部修改，请在 Pages 解除后重新绑定",
                    )
                restrictions = value["restrictions"]
                current = manager.get_persona_v3_by_id(carrier_id)
                if current and current.get("prompt") != value["text"]:
                    raise IrisError(
                        "host_persona_conflict", "人格执行镜像被外部修改，未覆盖"
                    )
            else:
                if manager.get_persona_v3_by_id(carrier_id):
                    raise IrisError(
                        "host_persona_conflict", "存在未登记的人格执行镜像，未覆盖"
                    )
                source = source or {}
                restrictions = {
                    k: source.get(k)
                    for k in ("tools", "skills", "custom_error_message")
                }
                restrictions["begin_dialogs"] = source.get("begin_dialogs", [])
                value = {
                    "carrier_id": carrier_id,
                    "original_present": "persona_id" in config,
                    "original": config.get("persona_id"),
                    "restrictions": restrictions,
                    "source": selected,
                }
            if (
                row
                and value.get("plugin_id") == persona["id"]
                and value.get("plugin_revision") == persona["revision"]
            ):
                return
            # Persist intent first so a restart can restore the original selector.
            new = {
                **value,
                "text": persona["text"],
                "plugin_id": persona["id"],
                "plugin_revision": persona["revision"],
            }
            saved = await control.store.put(
                "host_carriers",
                identity.umo,
                new,
                expected_revision=row["revision"] if row else 0,
            )
            try:
                if manager.get_persona_v3_by_id(carrier_id):
                    await manager.update_persona(
                        carrier_id, system_prompt=persona["text"], **restrictions
                    )
                else:
                    await manager.create_persona(
                        carrier_id, persona["text"], **restrictions
                    )
                await sp.put_async(
                    "umo",
                    identity.umo,
                    "session_service_config",
                    {**config, "persona_id": carrier_id},
                )
            except BaseException:
                control.logs.emit(
                    "persona.adoption_failed",
                    level="ERROR",
                    reason="Carrier intent retained for recovery",
                    revision=saved["revision"],
                )
                raise
            control.logs.emit(
                "persona.adopted",
                persona_id=persona["id"],
                revision=persona["revision"],
                conversation=identity.key,
            )

    async def verify_persona(self, control, identity, request):
        persona = await control.personas.resolve(
            identity, control.settings["default_persona"]
        )
        if not persona:
            return
        selected, _, _, _ = await self.selected_persona(identity)
        row = await control.store.get("host_carriers", identity.umo)
        if (
            not row
            or selected != row["value"]["carrier_id"]
            or row["value"]["plugin_revision"] != persona["revision"]
            or persona["text"] not in request.system_prompt
        ):
            raise IrisError(
                "persona_not_adopted", "当前请求未采用已发布人格，暂停 Iris 上下文注入"
            )

    async def release_personas(self, control, only_umo=None):
        from astrbot.api import sp

        async with self.persona_lock:
            rows = await control.store.list("host_carriers", limit=500)
            for row in rows:
                umo, value = row["key"], row["value"]
                if only_umo and only_umo != umo:
                    continue
                config = (
                    await sp.get_async("umo", umo, "session_service_config", {}) or {}
                )
                if config.get("persona_id") == value["carrier_id"]:
                    if value["original_present"]:
                        config["persona_id"] = value["original"]
                    else:
                        config.pop("persona_id", None)
                    await sp.put_async("umo", umo, "session_service_config", config)
                manager = self.context.persona_manager
                current = manager.get_persona_v3_by_id(value["carrier_id"])
                if current and current.get("prompt") == value["text"]:
                    await manager.delete_persona(value["carrier_id"])
                elif current:
                    control.logs.emit(
                        "persona.release_conflict",
                        level="WARNING",
                        reason="Externally edited carrier retained",
                        carrier=value["carrier_id"],
                    )
                await control.store.run(
                    lambda db, key=umo: db.execute(
                        "DELETE FROM documents WHERE collection='host_carriers' AND key=?",
                        (key,),
                    )
                )
