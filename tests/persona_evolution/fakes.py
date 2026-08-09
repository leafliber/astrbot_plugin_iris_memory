"""人格自迭代阶段 2 测试契约层：PersonaManager / LLM fake

FakePersonaManager：内存 Persona 存储，记录 update_persona 调用参数
（供断言只传 persona_id + system_prompt），支持注入外部编辑。
FakeLLMManager：按 module 可编程返回好 JSON / 坏 JSON / 异常 / 超时。
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple


class FakePersona:
    """内存 Persona 对象（模拟 AstrBot Persona 属性访问）"""

    def __init__(
        self,
        persona_id: str,
        system_prompt: str,
        begin_dialogs: Optional[List[str]] = None,
    ):
        self.persona_id = persona_id
        self.system_prompt = system_prompt
        self.begin_dialogs = begin_dialogs or []


class FakePersonaManager:
    """PersonaManager fake

    - get_persona / update_persona / get_all_personas 异步接口；
    - update_calls 记录每次调用的 (args, kwargs)，供断言参数个数；
    - external_edit 模拟 AstrBot 面板侧的外部修改（不记入 update_calls）。
    """

    def __init__(self):
        self._personas: Dict[str, FakePersona] = {}
        self.update_calls: List[Tuple[tuple, dict]] = []

    def add_persona(self, persona_id: str, system_prompt: str) -> None:
        self._personas[persona_id] = FakePersona(persona_id, system_prompt)

    def external_edit(self, persona_id: str, new_prompt: str) -> None:
        """模拟外部编辑：直接改库，不经过 update_persona"""
        self._personas[persona_id].system_prompt = new_prompt

    def get_prompt(self, persona_id: str) -> Optional[str]:
        persona = self._personas.get(persona_id)
        return persona.system_prompt if persona else None

    async def get_persona(self, persona_id: str) -> Optional[FakePersona]:
        await asyncio.sleep(0)
        return self._personas.get(persona_id)

    async def get_all_personas(self) -> List[FakePersona]:
        await asyncio.sleep(0)
        return list(self._personas.values())

    async def update_persona(self, *args, **kwargs) -> None:
        await asyncio.sleep(0)
        self.update_calls.append((args, kwargs))
        persona_id = args[0]
        persona = self._personas.get(persona_id)
        if persona is None:
            raise ValueError(f"Persona 不存在：{persona_id}")
        if "system_prompt" in kwargs:
            persona.system_prompt = kwargs["system_prompt"]

    async def create_persona(
        self,
        persona_id: str,
        system_prompt: Optional[str] = None,
        begin_dialogs: Optional[List[str]] = None,
        **kwargs,
    ) -> FakePersona:
        """模拟 AstrBot 4.x create_persona：已存在抛 ValueError"""
        await asyncio.sleep(0)
        if persona_id in self._personas:
            raise ValueError(f"Persona with ID {persona_id} already exists.")
        persona = FakePersona(persona_id, system_prompt or "", begin_dialogs)
        self._personas[persona_id] = persona
        return persona


class FakeContext:
    """AstrBot Context fake（只携带 persona_manager）"""

    def __init__(self, persona_manager: Optional[FakePersonaManager] = None):
        self.persona_manager = persona_manager


class FakeLLMManager:
    """LLMManager fake

    按 module 编程响应队列（FIFO），元素可以是：
    - str：作为 generate_direct 返回值；
    - Exception 实例：调用时抛出（如 asyncio.TimeoutError()）。

    队列耗尽时按 default_responses 中的 module 默认值返回，
    否则抛 RuntimeError（暴露测试中漏配的调用）。
    """

    is_available = True

    def __init__(self):
        self._queues: Dict[str, List[Any]] = {}
        self.default_responses: Dict[str, str] = {}
        self.calls: List[Dict[str, Any]] = []
        # 人为延迟（秒），用于并发测试制造重叠窗口
        self.delay = 0.0

    def push(self, module: str, response: Any) -> None:
        self._queues.setdefault(module, []).append(response)

    def set_default(self, module: str, response: str) -> None:
        self.default_responses[module] = response

    async def generate_direct(
        self,
        prompt: str,
        module: str = "default",
        provider_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        await asyncio.sleep(self.delay)
        self.calls.append(
            {"module": module, "provider_id": provider_id, "prompt": prompt}
        )
        queue = self._queues.get(module) or []
        if queue:
            response = queue.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if module in self.default_responses:
            return self.default_responses[module]
        raise RuntimeError(f"FakeLLMManager 未配置 module={module} 的响应")


# ----------------------------------------------------------------------
# 各模块的"好 JSON"响应构造
# ----------------------------------------------------------------------

ANALYSIS_MODULE = "persona_evolution_analysis"
GENERATION_MODULE = "persona_evolution_generate"
REVIEW_MODULE = "persona_evolution_review"


def good_analysis_json(confidence: float = 0.85) -> str:
    return json.dumps(
        {
            "tone": ["轻松", "直接"],
            "verbosity": "short",
            "sentence_rhythm": "短句为主，偶尔追问",
            "punctuation": ["较少句号"],
            "emoji_style": "low",
            "interaction_patterns": ["先回应情绪再给建议"],
            "humor_style": ["轻微自嘲"],
            "avoid_patterns": ["复读"],
            "confidence": confidence,
            "evidence_summary": "基于多个用户和时间段的共同特征",
        },
        ensure_ascii=False,
    )


def good_generation_json(candidate_prompt: str, confidence: float = 0.86) -> str:
    return json.dumps(
        {
            "candidate_prompt": candidate_prompt,
            "change_summary": ["回复更短", "更常追问"],
            "rationale": "在保持身份设定不变的前提下贴近参考风格",
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


def good_review_json(**overrides) -> str:
    data = {
        "identity_consistency": 0.95,
        "constraint_preservation": 0.98,
        "goal_alignment": 0.85,
        "privacy_safety": 1.0,
        "prompt_injection_suspected": False,
        "pass": True,
        "reasons": ["身份保持一致", "约束全部保留"],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)
