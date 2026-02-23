"""
Web 模块 - 提供 Iris Memory 的 Web 管理界面

功能模块：
1. 统计面板：数据可视化、趋势分析
2. 记忆管理：查询检索、单条/批量删除
3. 知识图谱：可视化展示、节点/关系管理
4. 数据导入导出：JSON/CSV 格式支持
"""

from iris_memory.web.api_routes import IrisWebAPI

__all__ = ["IrisWebAPI"]
