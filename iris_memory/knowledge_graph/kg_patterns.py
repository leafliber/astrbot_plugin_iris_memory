"""
kg_patterns.py - 知识图谱提取规则与模式数据

从 kg_extractor.py 提取的模块级常量：关键词集合、正则模式、实体类型猜测。
遵循 SRP 原则，将 ~200 行静态数据与提取逻辑分离。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from iris_memory.knowledge_graph.kg_models import (
    KGNodeType,
    KGRelationType,
)


# ── LLM Prompt ──
TRIPLE_EXTRACTION_PROMPT = """从以下文本中提取实体关系三元组。

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
{{{{
  "triples": [
    {{{{
      "subject": "实体A",
      "subject_type": "person",
      "predicate": "关系描述",
      "relation_type": "likes",
      "object": "实体B",
      "object_type": "concept",
      "confidence": 0.8
    }}}}
  ]
}}}}
```
仅返回JSON，不要有其他文字。"""


# ── 规则模式匹配阈值 ──

# 文本长度阈值：超过此长度跳过规则提取
RULE_TEXT_MAX_LENGTH: int = 2000

# hybrid 模式默认每日 LLM 调用上限
DEFAULT_DAILY_LIMIT: int = 100


# ── 关系信号关键词 ──

RELATIONSHIP_SIGNAL_KEYWORDS: frozenset[str] = frozenset([
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

# 快速关键词预过滤集合
QUICK_FILTER_KEYWORDS: frozenset[str] = frozenset([
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


# ── 中文关系模式（预编译）──

CN_RELATION_PATTERNS: List[Tuple[re.Pattern, KGRelationType, str]] = [
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
EN_RELATION_PATTERNS: List[Tuple[re.Pattern, KGRelationType, str]] = [
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:is|was) (?:a )?(?:friend|buddy) of (?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"), KGRelationType.FRIEND_OF, "friend of"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:is|was) (?:the )?(?:boss|manager|supervisor) of (?P<o>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"), KGRelationType.BOSS_OF, "boss of"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:works|worked) at (?P<o>[A-Z][\w\s]+)"), KGRelationType.WORKS_AT, "works at"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:lives|lived) in (?P<o>[A-Z][\w\s]+)"), KGRelationType.LIVES_IN, "lives in"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:likes?|loves?) (?P<o>[\w\s]+)"), KGRelationType.LIKES, "likes"),
    (re.compile(r"(?P<s>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) (?:hates?|dislikes?) (?P<o>[\w\s]+)"), KGRelationType.DISLIKES, "dislikes"),
]


# ── 实体类型猜测数据 ──

CONCEPT_KEYWORDS: frozenset[str] = frozenset({
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
})

EVENT_KEYWORDS: frozenset[str] = frozenset({
    "会议", "聚会", "婚礼", "生日", "毕业", "面试", "考试", "比赛",
    "旅行", "出差", "搬家", "入职", "离职", "升职", "年会",
    "春节", "中秋", "国庆", "元旦", "圣诞",
})

OBJECT_KEYWORDS: frozenset[str] = frozenset({
    "手机", "电脑", "笔记本", "平板", "耳机", "相机",
    "汽车", "自行车", "摩托车",
    "猫", "狗", "宠物", "仓鼠", "兔子", "鹦鹉", "金鱼",
    "书", "书籍",
})

# 中文职业/身份后缀 → PERSON
PERSON_SUFFIXES: Tuple[str, ...] = (
    "师", "员", "家", "生", "手", "者", "长",
    "教授", "医生", "老师", "同学", "朋友",
    "博主", "主播", "大佬", "萌新", "大神", "小白",
)

# 中文人名/昵称模式 → PERSON
PERSON_PATTERNS: Tuple[str, ...] = (
    r'^[小老阿][\u4e00-\u9fa5]{1,2}$',
    r'^(?:UP主|KOL|CEO|CTO|CFO|COO)$',
)

# 中文地名后缀 → LOCATION
LOCATION_SUFFIXES: Tuple[str, ...] = (
    "市", "省", "区", "县", "镇", "村", "路", "街", "大学", "学校", "医院",
)

# 组织后缀 → ORGANIZATION
ORG_SUFFIXES: Tuple[str, ...] = (
    "公司", "集团", "科技", "有限", "股份", "银行", "基金", "协会", "委员会",
)

# 常见中文姓氏（用于人名判断）
COMMON_SURNAMES: str = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹"


def guess_node_type(text: str) -> KGNodeType:
    """根据文本猜测节点类型

    优先级：关键词精确匹配 > 后缀匹配 > 正则模式匹配 > UNKNOWN
    """
    # ── 1. 关键词精确匹配（最高优先级）──
    text_lower = text.lower()
    if text in CONCEPT_KEYWORDS or text_lower in CONCEPT_KEYWORDS:
        return KGNodeType.CONCEPT
    if text in EVENT_KEYWORDS:
        return KGNodeType.EVENT
    if text in OBJECT_KEYWORDS:
        return KGNodeType.OBJECT

    # ── 2. 后缀匹配 ──
    for suffix in PERSON_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.PERSON
    for suffix in LOCATION_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.LOCATION
    for suffix in ORG_SUFFIXES:
        if text.endswith(suffix):
            return KGNodeType.ORGANIZATION

    # ── 3. 正则模式匹配 ──
    # 纯英文首字母大写 → 大概率是名字
    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):
        return KGNodeType.PERSON

    # 中文 2~4 字，常见姓氏开头 → PERSON
    if re.match(f'^[{COMMON_SURNAMES}][\u4e00-\u9fa5]{{1,2}}$', text):
        return KGNodeType.PERSON
    # 昵称模式
    for pattern in PERSON_PATTERNS:
        if re.match(pattern, text):
            return KGNodeType.PERSON

    return KGNodeType.UNKNOWN
