"""
Web 模块 - 提供 Iris Memory 的 Web 管理界面

架构层次：
- api/          响应辅助 + 路由蓝图（dashboard / memories / kg / personas / io）
- service/      领域子服务（DashboardService / MemoryWebService / KgWebService …）
- data/         仓储实现（MemoryRepo / KgRepo / PersonaRepo / SessionRepo）
- static/       前端资源（HTML shell + CSS + JS 模块）
- web_ui.py     Web UI 管理器（插件入口使用）

功能模块：
1. 统计面板：数据可视化、趋势分析
2. 记忆管理：查询检索、单条/批量删除
3. 知识图谱：可视化展示、节点/关系管理
4. 数据导入导出：JSON/CSV 格式支持

通过独立端口和访问密钥进行认证，与 AstrBot 认证系统解耦。
"""

from iris_memory.web.standalone_server import StandaloneWebServer
from iris_memory.web.web_service import WebService
from iris_memory.web.web_ui import WebUIManager

__all__ = [
    "StandaloneWebServer",
    "WebService",
    "WebUIManager",
]
