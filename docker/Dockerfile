# AstrBot Iris Memory Plugin 测试环境
# 基于 AstrBot 官方镜像，集成 Iris Memory 插件进行本地测试

FROM soulter/astrbot:latest

# 设置工作目录
WORKDIR /AstrBot

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制插件依赖文件（从 build context 复制）
COPY requirements.txt /tmp/iris_memory_requirements.txt

# 安装插件依赖
RUN pip install --no-cache-dir -r /tmp/iris_memory_requirements.txt

# 创建插件目录结构
RUN mkdir -p /AstrBot/data/plugins/astrbot_plugin_iris_memory

# 复制 entrypoint 脚本
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 设置权限
RUN chmod -R 777 /AstrBot/data

# 暴露端口
EXPOSE 6185 6186

# 使用 entrypoint 脚本启动
ENTRYPOINT ["/entrypoint.sh"]
