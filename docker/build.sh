#!/bin/bash

# 设置代理变量 - 请根据您的实际环境修改以下值
# 例如: export PROXY_SERVER="http://10.0.0.1:8080" 或 "http://proxy.example.com:8080"
export PROXY_SERVER="http://172.18.32.1:18899"
export NO_PROXY="localhost,127.0.0.1,.local,192.168.0.0/16"

# 直接设置代理变量，与 Dockerfile 中的变量名称保持一致
export HTTP_PROXY="$PROXY_SERVER"
export HTTPS_PROXY="$PROXY_SERVER"
export NO_PROXY="$NO_PROXY"

# 显示当前设置的代理变量
echo "使用代理设置:"
echo "PROXY_SERVER=$PROXY_SERVER"
echo "NO_PROXY=$NO_PROXY"

# # 运行 docker compose 构建
# # 如果您想在后台运行，可以添加 -d 参数
# docker compose -f docker-compose.yml up --build

# 如果您只想构建而不启动容器，可以使用以下命令（取消注释使用）:
docker compose -f docker-compose.yml build
