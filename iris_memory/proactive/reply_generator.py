"""
主动回复生成器
使用LLM结合记忆生成个性化回复
"""
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from iris_memory.utils.logger import get_logger
from iris_memory.retrieval.retrieval_engine import MemoryRetrievalEngine
from iris_memory.models.emotion_state import EmotionalState

logger = get_logger("reply_generator")


@dataclass
class GeneratedReply:
    """生成的回复"""
    content: str
    emotion_tone: str
    referenced_memories: List[str]
    confidence: float
    metadata: Dict[str, Any]


class ProactiveReplyGenerator:
    """主动回复生成器"""
    
    def __init__(
        self,
        astrbot_context=None,
        retrieval_engine: Optional[MemoryRetrievalEngine] = None,
        config: Optional[Dict] = None
    ):
        self.astrbot_context = astrbot_context
        self.retrieval_engine = retrieval_engine
        self.config = config or {}
        
        self.llm_api = None
        self.max_tokens = self.config.get("max_reply_tokens", 150)
        self.temperature = self.config.get("reply_temperature", 0.7)
        
        # 回复风格模板
        self.tone_templates = {
            "supportive": "温暖支持，表达关心和理解",
            "cheerful": "愉快活泼，分享喜悦",
            "neutral": "平和自然，保持友好",
            "empathetic": "共情理解，深入交流",
            "encouraging": "鼓励激励，给予力量"
        }
    
    async def initialize(self) -> bool:
        """初始化LLM API"""
        if not self.astrbot_context:
            logger.warning("No astrbot_context provided")
            return False
        
        logger.info("Reply generator initialized with context")
        return True
    
    async def generate_reply(
        self,
        messages: List[str],
        user_id: str,
        group_id: Optional[str] = None,
        emotional_state: Optional[EmotionalState] = None,
        umo: str = "",
        reply_context: Optional[Dict] = None
    ) -> Optional[GeneratedReply]:
        """生成主动回复"""
        if not self.astrbot_context or not messages:
            logger.warning("No astrbot_context or messages provided")
            return None
        
        try:
            # 获取LLM provider
            provider = self.astrbot_context.get_using_provider(umo=umo)
            if not provider:
                logger.warning("No LLM provider available for reply generation")
                return None
            
            # 检索相关记忆
            relevant_memories = []
            if self.retrieval_engine and len(messages) > 0:
                query = messages[-1]
                memories = await self.retrieval_engine.retrieve(
                    query=query,
                    user_id=user_id,
                    group_id=group_id,
                    top_k=3,
                    emotional_state=emotional_state
                )
                relevant_memories = [m.content for m in memories]
            
            # 确定回复语调
            emotion_tone = self._determine_tone(
                reply_context.get("emotion", {}) if reply_context else {},
                emotional_state
            )
            
            # 构建提示词
            prompt = self._build_reply_prompt(
                messages=messages,
                memories=relevant_memories,
                tone=emotion_tone,
                reply_context=reply_context
            )
            
            # 调用LLM生成回复
            response = await self._call_llm(provider, prompt)
            
            if not response:
                return None
            
            # 解析回复
            reply_content = self._extract_reply(response)
            
            return GeneratedReply(
                content=reply_content,
                emotion_tone=emotion_tone,
                referenced_memories=relevant_memories,
                confidence=0.8,
                metadata={
                    "prompt_tokens": len(prompt) // 4,
                    "reply_tokens": len(reply_content) // 4
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate reply: {e}")
            return None
    
    def _determine_tone(
        self,
        emotion_analysis: Dict[str, Any],
        emotional_state: Optional[EmotionalState]
    ) -> str:
        """确定回复语调"""
        primary_emotion = emotion_analysis.get("primary", "neutral")
        intensity = emotion_analysis.get("intensity", 0.5)
        
        emotion_tone_map = {
            "happy": "cheerful",
            "excited": "cheerful",
            "sad": "supportive",
            "angry": "empathetic",
            "anxious": "supportive",
            "lonely": "empathetic",
            "grateful": "cheerful",
            "neutral": "neutral"
        }
        
        tone = emotion_tone_map.get(primary_emotion, "neutral")
        
        if intensity > 0.8:
            if tone == "supportive":
                tone = "empathetic"
            elif tone == "cheerful":
                tone = "encouraging"
        
        return tone
    
    def _build_reply_prompt(
        self,
        messages: List[str],
        memories: List[str],
        tone: str,
        reply_context: Optional[Dict] = None
    ) -> str:
        """构建回复生成提示词"""
        
        conversation_history = "\n".join([
            f"用户: {msg}" for msg in messages[-5:]
        ])
        
        memory_context = ""
        if memories:
            memory_context = "\n相关记忆：\n" + "\n".join([
                f"- {m}" for m in memories[:3]
            ])
        
        tone_description = self.tone_templates.get(tone, self.tone_templates["neutral"])
        
        reason = ""
        if reply_context:
            reason = reply_context.get("reason", "")
        
        prompt = f"""你是一个贴心的AI助手。用户发送了以下消息，需要你主动回复。

对话历史：
{conversation_history}{memory_context}

回复要求：
1. 语调：{tone_description}
2. 简洁自然，不超过3句话
3. 结合相关记忆，让回复更个性化
4. 不要重复用户的话，提供有价值的回应
5. 回复原因：{reason}

请直接生成回复内容，不要添加说明："""
        
        return prompt
    
    async def _call_llm(self, provider, prompt: str) -> Optional[str]:
        """调用LLM"""
        if not provider:
            return None
        
        try:
            # 使用text_chat方法，类似image_analyzer的做法
            response = await provider.text_chat(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if isinstance(response, dict):
                return response.get("text", "") or response.get("content", "")
            return str(response) if response else ""
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _extract_reply(self, response: str) -> str:
        """从LLM响应中提取回复内容"""
        if not response:
            return "我在听，请继续说。"
        
        reply = response.strip()
        reply = reply.strip('"').strip("'")
        
        max_length = 200
        if len(reply) > max_length:
            reply = reply[:max_length-3] + "..."
        
        return reply
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.astrbot_context is not None
