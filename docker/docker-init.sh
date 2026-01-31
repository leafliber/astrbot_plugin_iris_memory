#!/bin/bash
# AstrBot Docker 初始化脚本

set -e

echo "==================================="
echo "AstrBot with Iris Memory Plugin"
echo "==================================="
echo ""

# 确保插件目录存在
if [ ! -d "/app/astrbot/data/plugins/astrbot_plugin_iris_memory" ]; then
    echo "⚠️  警告: 插件目录不存在，正在创建..."
    mkdir -p /app/astrbot/data/plugins
fi

# 检查插件是否正确挂载
if [ -f "/app/astrbot/data/plugins/astrbot_plugin_iris_memory/metadata.yaml" ]; then
    echo "✓ 插件已正确挂载"
    echo "  - 名称: astrbot_plugin_iris_memory"
    echo "  - 版本: v1.0.0"
    echo "  - 描述: 基于companion-memory框架的三层记忆插件"
else
    echo "⚠️  警告: 插件元数据文件不存在，请检查插件挂载"
fi

echo ""
echo "==================================="
echo "安装插件依赖..."
echo "==================================="

# 安装插件依赖
if [ -f "/app/astrbot/data/plugins/astrbot_plugin_iris_memory/requirements.txt" ]; then
    echo "✓ 安装插件依赖..."
    /root/.local/bin/uv pip install -r /app/astrbot/data/plugins/astrbot_plugin_iris_memory/requirements.txt || echo "⚠️ 依赖安装可能失败，但不影响基础功能"
else
    echo "⚠️  requirements.txt 未找到"
fi

echo ""
echo "==================================="
echo "启动 AstrBot..."
echo "==================================="

# 启动AstrBot
exec uv run python main.py
