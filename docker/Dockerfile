# AstrBot with Iris Memory Plugin Docker镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install uv

# 克隆AstrBot
RUN git clone https://github.com/AstrBotDevs/AstrBot.git /app/astrbot

# 设置工作目录到AstrBot
WORKDIR /app/astrbot

# 创建插件目录
RUN mkdir -p /app/astrbot/data/plugins

# 安装AstrBot依赖
RUN uv sync

# 暴露端口
EXPOSE 7862

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# 启动脚本
COPY docker-init.sh /app/astrbot/docker-init.sh
RUN chmod +x /app/astrbot/docker-init.sh

# 默认启动命令
CMD ["/bin/bash", "-c", "./docker-init.sh"]
