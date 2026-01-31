#!/bin/bash
# Docker管理脚本 - 用于管理AstrBot + Iris Memory本地开发环境

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_NAME="astrbot-iris-memory"
CONTAINER_NAME="${PROJECT_NAME}"
COMPOSE_FILE="docker-compose.yml"

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: Docker未安装，请先安装Docker${NC}"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}错误: Docker Compose未安装，请先安装Docker Compose${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Docker环境检查通过${NC}"
}

# 检查必要文件
check_files() {
    if [ ! -f "Dockerfile" ]; then
        echo -e "${RED}错误: Dockerfile不存在${NC}"
        exit 1
    fi

    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}错误: $COMPOSE_FILE不存在${NC}"
        exit 1
    fi

    if [ ! -f "docker-init.sh" ]; then
        echo -e "${RED}错误: docker-init.sh不存在${NC}"
        exit 1
    fi

    if [ ! -f "../metadata.yaml" ]; then
        echo -e "${RED}错误: metadata.yaml不存在${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 必要文件检查通过${NC}"
}

# 启动容器
start() {
    echo -e "${YELLOW}正在构建并启动Docker环境...${NC}"
    docker-compose -f "$COMPOSE_FILE" up -d --build

    echo ""
    echo -e "${GREEN}✓ Docker环境已启动${NC}"
    echo ""
    echo "访问地址:"
    echo "  - WebUI: http://localhost:6185"
    echo ""
    echo "使用 'manager.sh logs' 查看日志"
    echo "使用 'manager.sh shell' 进入容器"
}

# 停止容器
stop() {
    echo -e "${YELLOW}正在停止Docker环境...${NC}"
    docker-compose -f "$COMPOSE_FILE" stop
    echo -e "${GREEN}✓ 容器已停止${NC}"
}

# 重启容器
restart() {
    echo -e "${YELLOW}正在重启容器...${NC}"
    docker-compose -f "$COMPOSE_FILE" restart
    echo -e "${GREEN}✓ 容器已重启${NC}"
}

# 停止并清理
down() {
    echo -e "${YELLOW}正在停止并清理Docker环境...${NC}"
    read -p "是否删除持久化数据? (y/N): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "$COMPOSE_FILE" down -v
        echo -e "${GREEN}✓ 已停止并删除所有数据${NC}"
    else
        docker-compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✓ 已停止容器（数据保留）${NC}"
    fi
}

# 查看日志
logs() {
    echo -e "${YELLOW}查看AstrBot日志（Ctrl+C退出）...${NC}"
    docker-compose -f "$COMPOSE_FILE" logs -f astrbot
}

# 进入容器shell
shell() {
    echo -e "${YELLOW}进入容器shell...${NC}"
    docker exec -it "$CONTAINER_NAME" bash
}

# 查看状态
status() {
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}容器状态${NC}"
    echo -e "${BLUE}===================================${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""

    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo -e "${BLUE}资源使用:${NC}"
        docker stats --no-stream "$CONTAINER_NAME"
        echo ""

        echo -e "${BLUE}日志摘要（最后10行）:${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=10 astrbot
    else
        echo -e "${YELLOW}容器未运行${NC}"
    fi
}

# 运行测试
test() {
    echo -e "${YELLOW}运行Docker环境测试...${NC}"
    if [ -f "test_docker.sh" ]; then
        bash test_docker.sh
    else
        echo -e "${RED}错误: test_docker.sh不存在${NC}"
        exit 1
    fi
}

# 快速测试
quick_test() {
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}Iris Memory Docker 快速测试${NC}"
    echo -e "${BLUE}===================================${NC}"
    echo ""

    echo -e "${YELLOW}步骤1: 构建并启动Docker环境${NC}"
    echo "-----------------------------------"
    start

    echo ""
    echo -e "${YELLOW}等待容器启动（约30秒）...${NC}"
    sleep 30

    echo ""
    echo -e "${YELLOW}步骤2: 检查容器状态${NC}"
    echo "-----------------------------------"
    status

    echo ""
    echo -e "${YELLOW}步骤3: 查看启动日志${NC}"
    echo "-----------------------------------"
    docker logs --tail=20 "$CONTAINER_NAME"

    echo ""
    echo -e "${YELLOW}步骤4: 运行自动化测试${NC}"
    echo "-----------------------------------"
    if [ -f "test_docker.sh" ]; then
        bash test_docker.sh
    else
        echo -e "${YELLOW}测试脚本不存在，跳过${NC}"
    fi

    echo ""
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}测试环境已就绪！${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    echo "访问地址:"
    echo "  - WebUI: http://localhost:6185"
    echo ""
    echo "测试步骤:"
    echo "  1. 打开WebUI"
    echo "  2. 启用 'Iris Memory' 插件"
    echo "  3. 测试指令:"
    echo "     /memory_save 我喜欢编程"
    echo "     /memory_search 我喜欢什么"
    echo "     /memory_stats"
    echo ""
    echo "管理命令:"
    echo "  ./manager.sh logs    # 查看日志"
    echo "  ./manager.sh test    # 运行测试"
    echo "  ./manager.sh stop    # 停止环境"
    echo "  ./manager.sh help    # 查看所有命令"
    echo ""
}

# 备份数据
backup() {
    echo -e "${YELLOW}备份容器数据...${NC}"
    BACKUP_DIR="../backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # 直接备份本地data目录
    if [ -d "../data" ]; then
        tar czf "$BACKUP_DIR/data.tar.gz" -C .. data
        echo -e "${GREEN}✓ 数据已备份到 $BACKUP_DIR${NC}"
        echo "备份文件: data.tar.gz"
    else
        echo -e "${YELLOW}⚠ data目录不存在，跳过备份${NC}"
    fi
}

# 恢复数据
restore() {
    BACKUP_DIR="$1"

    if [ -z "$BACKUP_DIR" ]; then
        echo "请输入备份目录路径:"
        read BACKUP_DIR
    fi

    if [ ! -d "$BACKUP_DIR" ]; then
        echo -e "${RED}错误: 备份目录不存在${NC}"
        exit 1
    fi

    if [ ! -f "$BACKUP_DIR/data.tar.gz" ]; then
        echo -e "${RED}错误: 备份文件 data.tar.gz 不存在${NC}"
        exit 1
    fi

    echo -e "${YELLOW}恢复容器数据...${NC}"
    read -p "确认要恢复数据吗？这将覆盖当前data目录！(y/N): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 备份当前数据
        if [ -d "../data" ]; then
            echo -e "${YELLOW}备份当前data目录...${NC}"
            mv ../data ../data.backup.$(date +%Y%m%d_%H%M%S)
        fi

        # 恢复数据
        tar xzf "$BACKUP_DIR/data.tar.gz" -C ..
        echo -e "${GREEN}✓ 数据已恢复${NC}"
        echo "请重启容器: ./manager.sh restart"
    else
        echo "已取消恢复"
    fi
}

# 重建镜像
rebuild() {
    echo -e "${YELLOW}正在重建Docker镜像...${NC}"
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    echo -e "${GREEN}✓ 镜像已重建${NC}"
    echo "请使用 'manager.sh start' 启动容器"
}

# 显示帮助信息
help() {
    echo "Docker管理脚本 - 用于管理AstrBot + Iris Memory本地开发环境"
    echo ""
    echo "用法: manager.sh [命令]"
    echo ""
    echo "命令:"
    echo "  start      构建并启动Docker环境"
    echo "  stop       停止容器"
    echo "  restart    重启容器"
    echo "  down       停止并清理容器"
    echo "  logs       查看容器日志（实时）"
    echo "  shell      进入容器shell"
    echo "  status     查看容器状态"
    echo "  test       运行Docker环境测试"
    echo "  quick_test 快速测试（启动+验证）"
    echo "  backup     备份容器数据"
    echo "  restore    恢复容器数据"
    echo "  rebuild    重建Docker镜像"
    echo "  help       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  manager.sh start       # 启动环境"
    echo "  manager.sh logs        # 查看日志"
    echo "  manager.sh test        # 运行测试"
    echo "  manager.sh quick_test  # 快速测试"
    echo ""
    echo "不提供参数时将进入交互模式"
}

# 交互式菜单
interactive_mode() {
    while true; do
        echo ""
        echo -e "${BLUE}===================================${NC}"
        echo -e "${BLUE}Docker管理菜单${NC}"
        echo -e "${BLUE}===================================${NC}"
        echo ""
        echo "1) 启动环境"
        echo "2) 停止容器"
        echo "3) 重启容器"
        echo "4) 停止并清理"
        echo "5) 查看日志"
        echo "6) 进入容器shell"
        echo "7) 查看状态"
        echo "8) 运行测试"
        echo "9) 快速测试"
        echo "10) 备份数据"
        echo "11) 恢复数据"
        echo "12) 重建镜像"
        echo "0) 退出"
        echo ""

        read -p "请输入选项 (0-12): " choice

        case $choice in
            1) start ;;
            2) stop ;;
            3) restart ;;
            4) down ;;
            5) logs ;;
            6) shell ;;
            7) status ;;
            8) test ;;
            9) quick_test ;;
            10) backup ;;
            11)
                echo "请输入备份目录路径:"
                read BACKUP_DIR
                restore "$BACKUP_DIR"
                ;;
            12) rebuild ;;
            0)
                echo "退出"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选项，请重新选择${NC}"
                ;;
        esac

        echo ""
        read -p "按Enter键继续..."
    done
}

# 主函数
main() {
    check_docker

    # 如果没有参数，进入交互模式
    if [ $# -eq 0 ]; then
        check_files
        interactive_mode
    else
        # 根据参数执行对应操作
        case "$1" in
            start)
                check_files
                start
                ;;
            stop)
                stop
                ;;
            restart)
                restart
                ;;
            down)
                down
                ;;
            logs)
                logs
                ;;
            shell)
                shell
                ;;
            status)
                check_files
                status
                ;;
            test)
                test
                ;;
            quick_test)
                check_files
                quick_test
                ;;
            backup)
                check_docker
                backup
                ;;
            restore)
                check_docker
                restore "$2"
                ;;
            rebuild)
                rebuild
                ;;
            help|--help|-h)
                help
                ;;
            *)
                echo -e "${RED}未知命令: $1${NC}"
                echo ""
                help
                exit 1
                ;;
        esac
    fi
}

# 执行主函数
main "$@"
