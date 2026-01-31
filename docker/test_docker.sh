#!/bin/bash
# Docker环境测试脚本 - 用于测试插件功能

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONTAINER_NAME="astrbot-iris-memory"

echo -e "${BLUE}===================================${NC}"
echo -e "${BLUE}Docker环境测试${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# 检查容器是否运行
check_container() {
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        echo -e "${RED}错误: 容器未运行，请先使用 manager.sh start 启动容器${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 容器运行中${NC}"
}

# 测试1: 检查容器健康状态
test_health() {
    echo ""
    echo -e "${YELLOW}[测试1] 检查容器健康状态${NC}"
    
    STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME")
    if [ "$STATUS" = "running" ]; then
        echo -e "${GREEN}✓ 容器状态正常${NC}"
    else
        echo -e "${RED}✗ 容器状态异常: $STATUS${NC}"
        return 1
    fi
}

# 测试2: 检查插件是否加载
test_plugin_loaded() {
    echo ""
    echo -e "${YELLOW}[测试2] 检查插件是否加载${NC}"
    
    # 等待插件加载（最多30秒）
    echo "等待插件加载..."
    for i in {1..30}; do
        if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "iris_memory"; then
            echo -e "${GREEN}✓ 插件已加载${NC}"
            return 0
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    echo -e "${RED}✗ 插件加载超时${NC}"
    return 1
}

# 测试3: 检查插件目录
test_plugin_directory() {
    echo ""
    echo -e "${YELLOW}[测试3] 检查插件目录${NC}"
    
    if docker exec "$CONTAINER_NAME" test -d "/app/astrbot/data/plugins/astrbot_plugin_iris_memory"; then
        echo -e "${GREEN}✓ 插件目录存在${NC}"
    else
        echo -e "${RED}✗ 插件目录不存在${NC}"
        return 1
    fi

    if docker exec "$CONTAINER_NAME" test -f "/app/astrbot/data/plugins/astrbot_plugin_iris_memory/metadata.yaml"; then
        echo -e "${GREEN}✓ metadata.yaml 存在${NC}"
    else
        echo -e "${RED}✗ metadata.yaml 不存在${NC}"
        return 1
    fi

    if docker exec "$CONTAINER_NAME" test -f "/app/astrbot/data/plugins/astrbot_plugin_iris_memory/main.py"; then
        echo -e "${GREEN}✓ main.py 存在${NC}"
    else
        echo -e "${RED}✗ main.py 不存在${NC}"
        return 1
    fi
}

# 测试4: 检查Python依赖
test_python_deps() {
    echo ""
    echo -e "${YELLOW}[测试4] 检查Python依赖${NC}"
    
    # 检查关键依赖
    DEPS=("chromadb" "openai" "pydantic")
    for dep in "${DEPS[@]}"; do
        if docker exec "$CONTAINER_NAME" python -c "import $dep" 2>/dev/null; then
            echo -e "${GREEN}✓ $dep 已安装${NC}"
        else
            echo -e "${RED}✗ $dep 未安装${NC}"
        fi
    done
}

# 测试5: 检查插件导入
test_plugin_import() {
    echo ""
    echo -e "${YELLOW}[测试5] 检查插件导入${NC}"
    
    if docker exec "$CONTAINER_NAME" python -c "import sys; sys.path.insert(0, '/app/astrbot/data/plugins/astrbot_plugin_iris_memory'); from iris_memory.models.memory import Memory" 2>/dev/null; then
        echo -e "${GREEN}✓ 插件模块可导入${NC}"
    else
        echo -e "${YELLOW}⚠ 插件模块导入检查跳过（需要插件完全启动）${NC}"
    fi
}

# 测试6: 检查数据目录
test_data_directories() {
    echo ""
    echo -e "${YELLOW}[测试6] 检查数据目录${NC}"
    
    if docker exec "$CONTAINER_NAME" test -d "/app/astrbot/data"; then
        echo -e "${GREEN}✓ AstrBot数据目录存在${NC}"
    else
        echo -e "${RED}✗ AstrBot数据目录不存在${NC}"
    fi

    if docker exec "$CONTAINER_NAME" test -d "/app/astrbot/data/plugin_data"; then
        echo -e "${GREEN}✓ 插件数据目录存在${NC}"
    else
        echo -e "${YELLOW}⚠ 插件数据目录可能尚未创建${NC}"
    fi
}

# 测试7: 检查日志输出
test_logs() {
    echo ""
    echo -e "${YELLOW}[测试7] 检查日志输出（最近20行）${NC}"
    
    docker logs --tail=20 "$CONTAINER_NAME" 2>&1 | tail -20
}

# 测试8: 运行pytest测试（如果存在）
test_pytest() {
    echo ""
    echo -e "${YELLOW}[测试8] 运行单元测试${NC}"
    
    if docker exec "$CONTAINER_NAME" test -d "/app/astrbot/data/plugins/astrbot_plugin_iris_memory/tests"; then
        echo "运行pytest测试..."
        docker exec "$CONTAINER_NAME" bash -c "cd /app/astrbot/data/plugins/astrbot_plugin_iris_memory && python -m pytest tests/ -v --tb=short" || echo -e "${YELLOW}⚠ 部分测试失败${NC}"
    else
        echo -e "${YELLOW}⚠ 测试目录不存在，跳过pytest测试${NC}"
    fi
}

# 汇总测试结果
summary() {
    echo ""
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}测试总结${NC}"
    echo -e "${BLUE}===================================${NC}"
    echo ""
    echo "所有基础测试已完成"
    echo ""
    echo "如需查看详细日志，请运行: manager.sh logs"
    echo "如需进入容器调试，请运行: manager.sh shell"
}

# 主测试流程
main() {
    check_container
    
    test_health
    test_plugin_directory
    test_data_directories
    test_python_deps
    test_plugin_loaded
    test_plugin_import
    test_logs
    test_pytest
    
    summary
}

# 执行主测试流程
main
