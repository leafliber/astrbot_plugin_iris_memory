"""
分层检测器

三级漏斗检测架构：
- L1: RuleDetector — 规则快速过滤
- L2: VectorDetector — 语义场景匹配
- L3: LLMDetector — LLM 确认边缘案例
"""
