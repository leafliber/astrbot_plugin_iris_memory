"""
知识图谱三元组提取器

从消息文本中提取 (主语, 谓语, 宾语) 三元组，并将其映射到图节点和边。

支持两种模式：
- rule : 纯规则/正则提取（零 LLM 开销）
- llm  : 使用 LLM 进行语义级关系提取（精度高）
- hybrid: 规则预筛 + LLM 补充（推荐）
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from iris_memory.knowledge_graph.kg_models import (
    KGEdge,
    KGNode,
    KGNodeType,
    KGRelationType,
    KGTriple,
)
from iris_memory.knowledge_graph.kg_storage import KGStorage
from iris_memory.utils.logger import get_logger
from iris_memory.utils.rate_limiter import DailyCallLimiter

logger = get_logger("kg_extractor")


# ── LLM Prompt ──
_TRIPLE_EXTRACTION_PROMPT = """从以下文本中提取实体关系三元组。

## 提取规则
1. 提取所有 (主语, 关系, 宾语) 三元组
2. 主语和宾语应是具体的实体（人名、地名、组织、事物、概念等）
3. 关系应是动词或动词短语
4. 忽略代词（我、你、他），用实际名称替换（如已知）
5. 最多提取 5 个最重要的三元组

## 关系类型参考
- 人际: friend_of, colleague_of, family_of, boss_of, subordinate_of, knows
- 属性: lives_in, works_at, studies_at, belongs_to, owns
- 行为: likes, dislikes, does, is, has, wants
- 事件: participated_in, happened_at, caused_by
- 通用: related_to

## 实体类型参考
- person, location, organization, object, event, concept, time, unknown

## 文本
{text}

## 发送者信息
发送者: {sender_name}（用户ID: {user_id}）

## 输出格式
```json
{{
  "triples": [
    {{
      "subject": "实体A",
      "subject_type": "person",
      "predicate": "关系描述",
      "relation_type": "likes",
      "object": "实体B",
      "object_type": "concept",
      "confidence": 0.8
    }}
  ]
}}
```
仅返回JSON，不要有其他文字。"""


# ── 规则模式匹配 ──

# 文本长度阈值：超过此长度跳过规则提取（避免大量正则遍历长文本）
RULE_TEXT_MAX_LENGTH: int = 2000

# hybrid 模式默认每日 LLM 调用上限
_DEFAULT_DAILY_LIMIT: int = 100

# 关系信号关键词：文本含有这些词时更可能包含可提取的关系
_RELATIONSHIP_SIGNAL_KEYWORDS: frozenset[str] = frozenset([
    # 身份/属性描述
    "是一", "是个", "当过", "做过", "担任",
    # 事件参与
    "参加", "参与", "一起", "带着", "陪着",
    # 归属/所有
    "属于", "来自", "出生", "毕业于",
    # 描述句式
    "叫做", "名叫", "人称", "外号", "绰号",
    # 英文
    "is a", "works as", "born in", "graduated from",
    "known as", "called", "belongs to",
])

# 快速关键词预过滤集合：只有包含这些关键词的文本才进入精细正则匹配
_QUICK_FILTER_KEYWORDS: frozenset[str] = frozenset([
    # 人际关系
    "朋友", "好友", "闺蜜", "哥们", "同事", "同学", "家人", "亲戚", "夫妻",
    "上司", "领导", "老板", "下属", "手下", "认识",
    "父亲", "母亲", "父母", "兄弟", "姐妹",
    "爸爸", "妈妈", "儿子", "女儿", "老公", "老婆", "丈夫", "妻子",
    "哥哥", "姐姐", "弟弟", "妹妹",
    "导师", "老师", "师父", "师傅", "教练", "学生", "徒弟", "弟子",
    "室友", "邻居", "搭档",
    "谈恋爱", "交往", "恋爱", "约会",
    # 属性
    "住在", "居住在", "家在", "工作", "上班", "就职", "上学", "读书", "学习", "就读",
    # 喜好
    "喜欢", "爱", "热爱", "钟爱", "迷恋", "讨厌", "不喜欢", "厌恶", "反感",
    "有", "拥有", "养了", "想要", "想买", "想学", "想去",
    # 身份
    "是一",
    # 英文
    "friend", "buddy", "boss", "manager", "supervisor",
    "works", "worked", "lives", "lived",
    "likes", "loves", "hates", "dislikes",
])

# 中文关系模式（预编译）
_CN_RELATION_PATTERNS: List[Tuple[re.Pattern, KGRelationType, str]] = [
    # 人际关系
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:是|为)(?P<o>[\u4e00-\u9fa5A-Za-z]+)的(?:上司|领导|老板)"), KGRelationType.BOSS_OF, "是...的上司"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:是|为)(?P<o>[\u4e00-\u9fa5A-Za-z]+)的(?:下属|手下)"), KGRelationType.SUBORDINATE_OF, "是...的下属"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:和|与|跟)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:是)?(?:朋友|好友|闺蜜|哥们)"), KGRelationType.FRIEND_OF, "是朋友"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:和|与|跟)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:是)?(?:同事|同学)"), KGRelationType.COLLEAGUE_OF, "是同事"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:和|与|跟)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:是)?(?:一家人|家人|亲戚|父母|兄弟|姐妹|夫妻|爸爸|妈妈|儿子|女儿)"), KGRelationType.FAMILY_OF, "是家人"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)认识(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.KNOWS, "认识"),

    # 补充人际关系模式
    (re.compile(r"(?P<o>[\u4e00-\u9fa5A-Za-z]+)是(?P<s>[\u4e00-\u9fa5A-Za-z]+)的(?:朋友|好友|闺蜜|哥们|兄弟)"), KGRelationType.FRIEND_OF, "是...的朋友"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:和|与|跟)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:在)?(?:谈恋爱|交往|恋爱|约会)"), KGRelationType.RELATED_TO, "谈恋爱"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)是(?P<o>[\u4e00-\u9fa5A-Za-z]+)的(?:导师|老师|师父|师傅|教练)"), KGRelationType.RELATED_TO, "是...的导师"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)是(?P<o>[\u4e00-\u9fa5A-Za-z]+)的(?:学生|徒弟|弟子)"), KGRelationType.RELATED_TO, "是...的学生"),
    (re.compile(r"(?P<o>[\u4e00-\u9fa5A-Za-z]+)是(?P<s>[\u4e00-\u9fa5A-Za-z]+)的(?:家人|亲戚|父亲|母亲|哥哥|姐姐|弟弟|妹妹|老公|老婆|丈夫|妻子|爸爸|妈妈|儿子|女儿)"), KGRelationType.FAMILY_OF, "是...的家人"),
    (re.compile(r"(?P<o>[\u4e00-\u9fa5A-Za-z]+)是(?P<s>[\u4e00-\u9fa5A-Za-z]+)的(?:同事|同学|室友|邻居|搭档)"), KGRelationType.COLLEAGUE_OF, "是...的同事"),

    # 属性关系
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:住在|居住在|家在)(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.LIVES_IN, "住在"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:在|就职于)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:工作|上班|就职)"), KGRelationType.WORKS_AT, "在...工作"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:在|就读于)(?P<o>[\u4e00-\u9fa5A-Za-z]+)(?:上学|读书|学习|就读)"), KGRelationType.STUDIES_AT, "在...上学"),

    # 喜好
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:喜欢|爱|热爱|钟爱|迷恋)(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.LIKES, "喜欢"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:讨厌|不喜欢|厌恶|反感)(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.DISLIKES, "讨厌"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:有|拥有|养了)(?:一只|一个|一条)?(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.HAS, "有"),
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)(?:想要|想买|想学|想去)(?P<o>[\u4e00-\u9fa5A-Za-z]+)"), KGRelationType.WANTS, "想要"),

    # 身份
    (re.compile(r"(?P<s>[\u4e00-\u9fa5A-Za-z]+)是(?:一[名个位])?(?P<o>[\u4e00-\u9fa5A-Za-z]+(?:师|员|家|生|手|者))"), KGRelationType.IS, "是"),
]

# 英文关系模式（预编译）
_EN_RELATION_PATTERNS: List[Tuple[re.Pattern, KGRelationType, str]] = [
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:is|was) (?:a )?(?:friend|buddy) of (?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"), KGRelationType.FRIEND_OF, "friend of"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:is|was) (?:the )?(?:boss|manager|supervisor) of (?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"), KGRelationType.BOSS_OF, "boss of"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:works|worked) at (?P<o>[A-Z][\w\s]+)"), KGRelationType.WORKS_AT, "works at"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:lives|lived) in (?P<o>[A-Z][\w\s]+)"), KGRelationType.LIVES_IN, "lives in"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:likes?|loves?) (?P<o>[\w\s]+)"), KGRelationType.LIKES, "likes"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:hates?|dislikes?) (?P<o>[\w\s]+)"), KGRelationType.DISLIKES, "dislikes"),
]

# 实体类型猜测规则
_ENTITY_TYPE_HINTS: Dict[str, KGNodeType] = {}

# ── 概念关键词 → CONCEPT ──
_CONCEPT_KEYWORDS: set = {
    # 兴趣/技能
    "编程", "音乐", "运动", "阅读", "游戏", "美食", "旅行", "摄影", "绘画", "写作",
    "烹饪", "健身", "瑜伽", "跑步", "游泳", "篮球", "足球", "乒乓球", "羽毛球",
    "钓鱼", "登山", "滑雪", "冲浪", "骑行", "舞蹈", "唱歌", "弹琴", "吉他",
    # 学科/技术领域
    "机器学习", "深度学习", "人工智能", "区块链", "大数据", "云计算",
    "数学", "物理", "化学", "生物", "历史", "地理", "哲学", "心理学", "经济学",
    "计算机科学", "数据科学", "自然语言处理", "计算机视觉",
    "前端开发", "后端开发", "全栈开发", "移动开发", "嵌入式开发",
    # 通用概念
    "自由", "民主", "科学", "艺术", "文化", "教育", "环保", "健康",
    # 英文概念
    "programming", "music", "sports", "reading", "gaming", "cooking",
    "machine learning", "deep learning", "artificial intelligence",
    "blockchain", "data science",
}

# ── 事件关键词 → EVENT ──
_EVENT_KEYWORDS: set = {
    "会议", "聚会", "婚礼", "生日", "毕业", "面试", "考试", "比赛",
    "旅行", "出差", "搬家", "入职", "离职", "升职", "年会",
    "春节", "中秋", "国庆", "元旦", "圣诞",
}

# ── 物品关键词 → OBJECT ──
_OBJECT_KEYWORDS: set = {
    "手机", "电脑", "笔记本", "平板", "耳机", "相机",
    "汽车", "自行车", "摩托车",
    "猫", "狗", "宠物", "仓鼠", "兔子", "鹦鹉", "金鱼",
    "书", "书籍",
}

# 中文职业/身份后缀 → PERSON
_PERSON_SUFFIXES = [
    "师", "员", "家", "生", "手", "者", "长",
    "教授", "医生", "老师", "同学", "朋友",
    # 新兴职业/网络用语
    "博主", "主播", "大佬", "萌新", "大神", "小白",
]

# 中文人名/昵称模式 → PERSON
_PERSON_PATTERNS = [
    # 小X/老X/阿X (1~2字)
    r'^[小老阿][\u4e00-\u9fa5]{1,2}$',
    # UP主、KOL 等特殊身份
    r'^(?:UP主|KOL|CEO|CTO|CFO|COO)$',
]

# 中文地名后缀 → LOCATION
_LOCATION_SUFFIXES = ["市", "省", "区", "县", "镇", "村", "路", "街", "大学", "学校", "医院"]

# 组织后缀 → ORGANIZATION
_ORG_SUFFIXES = ["公司", "集团", "科技", "有限", "股份", "银行", "基金", "协会", "委员会"]


def _guess_node_type(text: str) -> KGNodeType:
    """根据文本猜测节点类型

    优先级：关键词精确匹配 > 后缀匹配 > 正则模式匹配 > UNKNOWN
    """
    # ── 1. 关键词精确匹配（最高优先级）──
    text_lower = text.lower()
    if text in _CONCEPT_KEYWORDS or text_lower in _CONCEPT_KEYWORDS:
        return KGNodeType.CONCEPT
    if text in _EVENT_KEYWORDS:
        return KGNodeType.EVENT
    if text in _OBJECT_KEYWORDS:
        return KGNodeType.OBJECT

    # ── 2. 后缀匹配 ──
    for suffix in _PERSON_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.PERSON
    for suffix in _LOCATION_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.LOCATION
    for suffix in _ORG_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.ORGANIZATION

    # ── 3. 正则模式匹配 ──
    # 纯英文首字母大写 → 大概率是名字
    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):
        return KGNodeType.PERSON

    # 中文 2~4 字，常见姓氏开头 → PERSON
    surnames = '王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹'
    if re.match(f'^[{surnames}][\u4e00-\u9fa5]{{1,2}}$', text):
        return KGNodeType.PERSON
    # 昵称模式
    for pattern in _PERSON_PATTERNS:
        if re.match(pattern, text):
            return KGNodeType.PERSON

    return KGNodeType.UNKNOWN


class KGExtractor:
    """三元组提取器

    从文本中提取实体关系，并写入 KGStorage。

    支持三种模式：
    - rule: 正则/规则提取
    - llm: LLM 语义提取
    - hybrid: 规则 + LLM
    """

    def __init__(
        self,
        storage: KGStorage,
        mode: str = "rule",
        astrbot_context: Any = None,
        provider_id: Optional[str] = None,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
    ) -> None:
        self.storage = storage
        self.mode = mode  # "rule" | "llm" | "hybrid"
        self._astrbot_context = astrbot_context
        self._provider_id = provider_id
        self._provider = None
        self._resolved_provider_id: Optional[str] = None
        self._provider_initialized = False

        # 每日 LLM 调用限制
        self._limiter = DailyCallLimiter(daily_limit)

        # Hybrid 决策统计
        self._stats: Dict[str, int] = {
            "rule_extractions": 0,
            "llm_extractions": 0,
            "llm_skipped_sufficient": 0,
            "llm_skipped_limit": 0,
            "llm_skipped_no_signal": 0,
            "hybrid_decisions": 0,
            "total_triples": 0,
        }

    # ================================================================
    # 主入口
    # ================================================================

    async def extract_and_store(
        self,
        text: str,
        user_id: str,
        group_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        sender_name: Optional[str] = None,
        existing_entities: Optional[List[str]] = None,
        persona_id: Optional[str] = None,
    ) -> List[KGTriple]:
        """从文本中提取三元组并存入图谱

        Args:
            text: 消息文本
            user_id: 用户 ID
            group_id: 群组 ID
            memory_id: 关联的记忆 ID
            sender_name: 发送者名称
            existing_entities: 已提取的实体列表（来自 EntityExtractor）
            persona_id: 人格 ID（始终写入节点/边，用于 persona 隔离）

        Returns:
            提取到的三元组列表
        """
        if not text or len(text.strip()) < 4:
            return []

        triples: List[KGTriple] = []

        if self.mode in ("rule", "hybrid"):
            rule_triples = self._extract_by_rules(text, sender_name)
            triples.extend(rule_triples)
            if rule_triples:
                self._stats["rule_extractions"] += 1

        if self.mode == "llm":
            # 纯 LLM 模式：直接调用
            llm_triples = await self._extract_by_llm(text, user_id, sender_name)
            triples = self._merge_triples(triples, llm_triples)
            if llm_triples:
                self._stats["llm_extractions"] += 1
        elif self.mode == "hybrid":
            # hybrid 模式：条件触发 LLM
            self._stats["hybrid_decisions"] += 1
            should_call, reason = self._should_call_llm_hybrid(text, triples)
            if should_call:
                if self._limiter.is_within_limit():
                    llm_triples = await self._extract_by_llm(
                        text, user_id, sender_name
                    )
                    if llm_triples:
                        self._limiter.increment()
                        self._stats["llm_extractions"] += 1
                    triples = self._merge_triples(triples, llm_triples)
                    logger.debug(
                        f"Hybrid LLM triggered ({reason}): "
                        f"got {len(llm_triples) if llm_triples else 0} triples"
                    )
                else:
                    self._stats["llm_skipped_limit"] += 1
                    logger.debug("Hybrid LLM skipped: daily limit reached")
            else:
                logger.debug(f"Hybrid LLM skipped: {reason}")

        if not triples:
            # 尝试从 existing_entities 构建隐含关系
            if existing_entities and sender_name:
                triples = self._build_implicit_triples(
                    text, sender_name, existing_entities
                )

        # 写入 KGStorage
        for triple in triples:
            await self._store_triple(triple, user_id, group_id, memory_id, persona_id)

        if triples:
            self._stats["total_triples"] += len(triples)
            logger.debug(
                f"Extracted {len(triples)} triples from text (mode={self.mode}): "
                + ", ".join(str(t) for t in triples[:3])
            )

        return triples

    # ================================================================
    # Hybrid 决策逻辑
    # ================================================================

    def _should_call_llm_hybrid(
        self,
        text: str,
        rule_triples: List[KGTriple],
    ) -> Tuple[bool, str]:
        """判断 hybrid 模式下是否需要调用 LLM

        决策逻辑：
        1. 规则已提取到 ≥2 个高置信度三元组 → 跳过
        2. 文本超长（规则跳过了）→ 触发 LLM
        3. 文本含有关系信号但规则未提取到 → 触发 LLM
        4. 规则只提取到低置信度结果 → 触发 LLM 补充

        Returns:
            (should_call, reason) 元组
        """
        # 规则已提取到足够多的高置信度结果
        high_conf = [t for t in rule_triples if t.confidence >= 0.6]
        if len(high_conf) >= 2:
            self._stats["llm_skipped_sufficient"] += 1
            return False, "rule_sufficient"

        # 文本超长（规则因长度限制跳过了）
        if len(text) > RULE_TEXT_MAX_LENGTH:
            return True, "text_too_long_for_rules"

        # 规则未提取到但文本有关系信号
        if not rule_triples and self._has_relationship_signals(text):
            return True, "relationship_signals_detected"

        # 规则提取到了但全部低置信度
        if rule_triples and all(t.confidence < 0.5 for t in rule_triples):
            return True, "low_confidence_rules"

        # 规则提取到 1 个高置信度结果，但文本可能还有更多关系
        if len(high_conf) == 1 and self._has_relationship_signals(text):
            return True, "partial_rules_with_signals"

        # 其余情况不调用
        if not rule_triples:
            self._stats["llm_skipped_no_signal"] += 1
            return False, "no_signal"

        self._stats["llm_skipped_sufficient"] += 1
        return False, "rule_acceptable"

    @staticmethod
    def _has_relationship_signals(text: str) -> bool:
        """检测文本是否包含关系描述的信号词

        除了 _QUICK_FILTER_KEYWORDS（用于规则匹配的精确关键词），
        这里额外覆盖一些规则正则无法匹配但 LLM 能理解的模式。
        """
        # 先检查规则关键词（这些文本至少可能包含关系）
        if any(kw in text for kw in _QUICK_FILTER_KEYWORDS):
            return True
        # 再检查扩展信号词
        if any(kw in text for kw in _RELATIONSHIP_SIGNAL_KEYWORDS):
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取 Hybrid 决策统计"""
        return {
            **self._stats,
            "daily_limit": self._limiter._daily_limit,
            "remaining_calls": self._limiter.remaining,
            "mode": self.mode,
        }

    @property
    def remaining_daily_calls(self) -> int:
        """剩余每日 LLM 可用次数"""
        return self._limiter.remaining

    # ================================================================
    # 规则提取
    # ================================================================

    def _extract_by_rules(
        self,
        text: str,
        sender_name: Optional[str] = None,
    ) -> List[KGTriple]:
        """基于预编译正则模式提取三元组
        
        优化策略：
        1. 文本长度阈值：超长文本跳过规则提取
        2. 关键词预过滤：快速检测文本是否可能包含关系表述
        3. 预编译正则：避免重复编译开销
        """
        # 超长文本跳过规则提取（交给 LLM 处理）
        if len(text) > RULE_TEXT_MAX_LENGTH:
            logger.debug(
                f"Text too long ({len(text)} chars > {RULE_TEXT_MAX_LENGTH}), "
                f"skipping rule extraction"
            )
            return []
        
        # 快速关键词预过滤
        if not any(kw in text for kw in _QUICK_FILTER_KEYWORDS):
            return []
        
        triples: List[KGTriple] = []
        seen: set = set()

        all_patterns = _CN_RELATION_PATTERNS + _EN_RELATION_PATTERNS

        for compiled_pattern, relation_type, label in all_patterns:
            for m in compiled_pattern.finditer(text):
                subject = m.group("s").strip()
                obj = m.group("o").strip()

                # 替换代词
                subject = self._resolve_pronoun(subject, sender_name)
                obj = self._resolve_pronoun(obj, sender_name)

                if not subject or not obj or subject == obj:
                    continue

                key = (subject.lower(), relation_type.value, obj.lower())
                if key in seen:
                    continue
                seen.add(key)

                triple = KGTriple(
                    subject=subject,
                    predicate=label,
                    object=obj,
                    subject_type=_guess_node_type(subject),
                    object_type=_guess_node_type(obj),
                    relation_type=relation_type,
                    confidence=0.7,
                    source_text=text,
                )
                triples.append(triple)

        return triples

    def _resolve_pronoun(self, text: str, sender_name: Optional[str]) -> str:
        """将代词替换为发送者名称"""
        pronouns_cn = {"我", "本人", "自己", "俺", "吾"}
        pronouns_en = {"i", "me", "myself"}

        if text in pronouns_cn or text.lower() in pronouns_en:
            return sender_name or text
        return text

    def _build_implicit_triples(
        self,
        text: str,
        sender_name: str,
        entities: List[str],
    ) -> List[KGTriple]:
        """从已提取实体构建隐含关系"""
        triples: List[KGTriple] = []

        for entity in entities:
            if entity == sender_name:
                continue
            # 默认创建 "related_to" 关系
            triple = KGTriple(
                subject=sender_name,
                predicate="提到了",
                object=entity,
                subject_type=KGNodeType.PERSON,
                object_type=_guess_node_type(entity),
                relation_type=KGRelationType.RELATED_TO,
                confidence=0.3,
                source_text=text,
            )
            triples.append(triple)

        return triples[:3]  # 限制隐含关系数量

    # ================================================================
    # LLM 提取
    # ================================================================

    async def _extract_by_llm(
        self,
        text: str,
        user_id: str,
        sender_name: Optional[str],
    ) -> List[KGTriple]:
        """使用 LLM 提取三元组"""
        if not self._astrbot_context:
            return []

        try:
            if not await self._ensure_provider():
                return []

            prompt = _TRIPLE_EXTRACTION_PROMPT.format(
                text=text,
                sender_name=sender_name or "未知",
                user_id=user_id,
            )

            from iris_memory.utils.llm_helper import call_llm, parse_llm_json
            result = await call_llm(
                self._astrbot_context,
                self._provider,
                self._resolved_provider_id,
                prompt,
                parse_json=True,
            )

            if not result.success or not result.content:
                return []

            data = result.parsed_json or parse_llm_json(result.content)
            if not data or "triples" not in data:
                return []

            triples: List[KGTriple] = []
            for item in data["triples"][:5]:
                subject = item.get("subject", "").strip()
                obj = item.get("object", "").strip()
                predicate = item.get("predicate", "").strip()
                if not subject or not obj or not predicate:
                    continue

                # 解析 relation_type
                rt_str = item.get("relation_type", "related_to")
                try:
                    relation_type = KGRelationType(rt_str)
                except ValueError:
                    relation_type = KGRelationType.RELATED_TO

                # 解析 node types
                st_str = item.get("subject_type", "unknown")
                ot_str = item.get("object_type", "unknown")
                try:
                    subject_type = KGNodeType(st_str)
                except ValueError:
                    subject_type = _guess_node_type(subject)
                try:
                    object_type = KGNodeType(ot_str)
                except ValueError:
                    object_type = _guess_node_type(obj)

                triple = KGTriple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    subject_type=subject_type,
                    object_type=object_type,
                    relation_type=relation_type,
                    confidence=float(item.get("confidence", 0.6)),
                    source_text=text,
                )
                triples.append(triple)

            return triples

        except Exception as e:
            logger.warning(f"LLM triple extraction failed: {e}")
            return []

    async def _ensure_provider(self) -> bool:
        """确保 LLM provider 可用"""
        if self._provider_initialized:
            return self._provider is not None

        self._provider_initialized = True
        try:
            from iris_memory.utils.llm_helper import resolve_llm_provider
            provider, resolved_provider_id = resolve_llm_provider(
                self._astrbot_context,
                self._provider_id or "",
                label="KGExtractor",
            )
            if provider:
                self._provider = provider
                self._resolved_provider_id = resolved_provider_id
                return True
        except Exception as e:
            logger.warning(f"Failed to resolve LLM provider for KGExtractor: {e}")

        return False

    # ================================================================
    # 三元组 → 图节点/边
    # ================================================================

    async def _store_triple(
        self,
        triple: KGTriple,
        user_id: str,
        group_id: Optional[str],
        memory_id: Optional[str],
        persona_id: Optional[str] = None,
    ) -> None:
        """将三元组写入 KGStorage"""
        _persona = persona_id or "default"
        # 创建/更新主语节点
        subject_node = KGNode(
            name=triple.subject,
            display_name=triple.subject,
            node_type=triple.subject_type,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        subject_node = await self.storage.upsert_node(subject_node)

        # 创建/更新宾语节点
        object_node = KGNode(
            name=triple.object,
            display_name=triple.object,
            node_type=triple.object_type,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        object_node = await self.storage.upsert_node(object_node)

        # 创建/更新边
        edge = KGEdge(
            source_id=subject_node.id,
            target_id=object_node.id,
            relation_type=triple.relation_type,
            relation_label=triple.predicate,
            memory_id=memory_id,
            user_id=user_id,
            group_id=group_id,
            persona_id=_persona,
            confidence=triple.confidence,
        )
        await self.storage.upsert_edge(edge)

    # ================================================================
    # 工具方法
    # ================================================================

    def _merge_triples(
        self,
        rule_triples: List[KGTriple],
        llm_triples: List[KGTriple],
    ) -> List[KGTriple]:
        """合并规则和 LLM 提取的三元组（去重）"""
        seen: set = set()
        merged: List[KGTriple] = []

        for t in rule_triples:
            key = (t.subject.lower(), t.relation_type.value, t.object.lower())
            seen.add(key)
            merged.append(t)

        for t in llm_triples:
            key = (t.subject.lower(), t.relation_type.value, t.object.lower())
            if key not in seen:
                seen.add(key)
                merged.append(t)

        return merged
