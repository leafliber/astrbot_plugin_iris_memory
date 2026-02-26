#!/bin/bash
# AstrBot Iris Memory Plugin Docker 管理脚本
# 用于本地测试环境的快速管理

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 显示帮助信息
show_help() {
    echo -e "${BLUE}AstrBot Iris Memory Plugin Docker 管理脚本${NC}"
    echo ""
    echo "用法: ./manager.sh [命令]"
    echo ""
    echo "命令:"
    echo "  start           启动测试环境"
    echo "  stop            停止测试环境"
    echo "  restart         重启测试环境"
    echo "  logs            查看日志"
    echo "  shell           进入容器 Shell"
    echo "  shell-root      以 root 身份进入容器 Shell"
    echo "  update          更新镜像并重启"
    echo "  clean           清理所有数据和镜像"
    echo "  status          查看容器状态"
    echo "  test            在容器中运行插件测试"
    echo "  build           重新构建镜像"
    echo ""
    echo "示例:"
    echo "  ./manager.sh start      # 启动测试环境"
    echo "  ./manager.sh logs       # 查看日志"
    echo "  ./manager.sh shell      # 进入容器调试"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: Docker 未安装${NC}"
        echo "请先安装 Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}错误: Docker Compose 未安装${NC}"
        echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
}

# 获取 docker compose 命令
get_compose_cmd() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

# 启动服务
start_service() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo -e "${BLUE}正在启动 AstrBot Iris Memory 测试环境...${NC}"
    cd "$SCRIPT_DIR"
    
    $compose_cmd up -d
    echo -e "${GREEN}服务已启动${NC}"
    echo -e "  AstrBot Web UI: ${YELLOW}http://localhost:6185${NC}"
    
    echo ""
    echo -e "${BLUE}提示:${NC}"
    echo "  - 首次启动可能需要几分钟下载镜像"
    echo "  - 使用 './manager.sh logs' 查看启动日志"
    echo "  - 使用 './manager.sh shell' 进入容器调试"
}

# 停止服务
stop_service() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo -e "${BLUE}正在停止服务...${NC}"
    cd "$SCRIPT_DIR"
    $compose_cmd down
    echo -e "${GREEN}服务已停止${NC}"
}

# 重启服务
restart_service() {
    stop_service
    sleep 2
    start_service
}

# 查看日志
show_logs() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    cd "$SCRIPT_DIR"
    echo -e "${BLUE}正在显示日志 (按 Ctrl+C 退出)...${NC}"
    $compose_cmd logs -f astrbot
}

# 进入容器 Shell
enter_shell() {
    check_docker
    echo -e "${BLUE}正在进入 AstrBot 容器...${NC}"
    docker exec -it astrbot-iris-memory-test /bin/bash
}

# 以 root 身份进入容器 Shell
enter_shell_root() {
    check_docker
    echo -e "${BLUE}正在以 root 身份进入 AstrBot 容器...${NC}"
    docker exec -it -u root astrbot-iris-memory-test /bin/bash
}

# 更新镜像
update_service() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo -e "${BLUE}正在更新镜像...${NC}"
    cd "$SCRIPT_DIR"
    $compose_cmd pull
    $compose_cmd up -d --build
    echo -e "${GREEN}更新完成${NC}"
}

# 清理环境
clean_environment() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo -e "${YELLOW}警告: 这将删除所有容器、镜像和数据卷！${NC}"
    read -p "是否继续? (y/N): " confirm
    
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        cd "$SCRIPT_DIR"
        $compose_cmd down -v --rmi all
        echo -e "${GREEN}清理完成${NC}"
    else
        echo -e "${BLUE}已取消${NC}"
    fi
}

# 查看状态
show_status() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    cd "$SCRIPT_DIR"
    echo -e "${BLUE}容器状态:${NC}"
    $compose_cmd ps
    
    echo ""
    echo -e "${BLUE}资源使用:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.Status}}" 2>/dev/null || true
}

# 运行测试
run_tests() {
    check_docker
    echo -e "${BLUE}正在运行插件测试...${NC}"
    
    # 安装测试依赖
    docker exec astrbot-iris-memory-test pip install pytest pytest-asyncio pytest-cov -q
    
    # 运行测试
    docker exec -w /AstrBot/data/plugins/astrbot_plugin_iris_memory astrbot-iris-memory-test \
        python -m pytest tests/ -v --tb=short
}

# 重新构建镜像
build_image() {
    check_docker
    local compose_cmd=$(get_compose_cmd)
    
    echo -e "${BLUE}正在重新构建镜像...${NC}"
    cd "$SCRIPT_DIR"
    $compose_cmd build --no-cache
    echo -e "${GREEN}构建完成${NC}"
}

# 主逻辑
case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    logs)
        show_logs
        ;;
    shell)
        enter_shell
        ;;
    shell-root)
        enter_shell_root
        ;;
    update)
        update_service
        ;;
    clean)
        clean_environment
        ;;
    status)
        show_status
        ;;
    test)
        run_tests
        ;;
    build)
        build_image
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}未知命令: ${1:-}${NC}"
        show_help
        exit 1
        ;;
esac
