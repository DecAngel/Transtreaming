#!/bin/bash

# 检查是否安装了uv
if ! command -v uv &> /dev/null; then
    echo "uv未安装，正在安装..."

    # 使用官方安装脚本安装uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 将uv添加到当前shell的PATH中
    export PATH="$HOME/.local/bin:$PATH"

    # 验证安装是否成功
    if ! command -v uv &> /dev/null; then
        echo "uv安装失败，请手动安装"
        exit 1
    fi

    echo "uv安装成功"
else
    echo "uv已安装"
fi

# 执行uv sync命令
uv sync --extra=cu124
