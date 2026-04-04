#!/bin/bash
set -e

echo "============================================================"
echo "  Qwen3-TTS 本地声音克隆服务 - 一键安装 (Linux/macOS)"
echo "============================================================"
echo

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo "[!] uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi
echo "[OK] uv 版本: $(uv --version)"

# 检查 ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[!] ffmpeg 未安装，尝试安装..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "[!] 请手动安装 ffmpeg: https://ffmpeg.org/download.html"
    fi
else
    echo "[OK] ffmpeg 已安装"
fi

# 创建虚拟环境
echo
echo "[*] 创建 Python 3.11 虚拟环境..."
uv venv --python 3.11 .venv
source .venv/bin/activate

# 安装 PyTorch (CUDA 12.8)
echo
echo "[*] 安装 PyTorch 2.11 + CUDA 12.8..."
uv pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安装依赖
echo
echo "[*] 安装项目依赖..."
uv pip install -r requirements.txt

# 环境变量
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[OK] 已创建 .env 文件"
fi

echo
echo "============================================================"
echo "  [OK] 安装完成！"
echo "  运行服务: bash start.sh"
echo "============================================================"
