#!/bin/bash
# AstrBot Iris Memory Plugin 启动脚本
# 确保插件目录正确挂载并加载

set -e

echo "AstrBot with Iris Memory Plugin"

# 等待插件目录挂载（给 Docker 卷挂载一些时间）
PLUGIN_DIR="/AstrBot/data/plugins/astrbot_plugin_iris_memory"
METADATA_FILE="$PLUGIN_DIR/metadata.yaml"

# 检查插件目录
if [ ! -d "$PLUGIN_DIR" ]; then
    echo "⚠️  警告: 插件目录不存在，正在创建..."
    mkdir -p "$PLUGIN_DIR"
fi

# 检查元数据文件是否存在（用于判断挂载是否成功）
if [ ! -f "$METADATA_FILE" ]; then
    echo "⚠️  警告: 插件元数据文件不存在，请检查插件挂载"
    echo "    预期路径: $METADATA_FILE"
    echo "    目录内容:"
    ls -la "$PLUGIN_DIR" 2>/dev/null || echo "    (目录为空或无法访问)"
fi

# 确保目录权限正确
chmod -R 777 /AstrBot/data/plugins 2>/dev/null || true

# 安装插件依赖（如果 requirements.txt 存在）
if [ -f "$PLUGIN_DIR/requirements.txt" ]; then
    echo "安装插件依赖..."
    pip install -q -r "$PLUGIN_DIR/requirements.txt" || echo "⚠️  部分依赖安装失败"
fi

# 启动 AstrBot
echo "启动 AstrBot..."
cd /AstrBot
exec python main.py
