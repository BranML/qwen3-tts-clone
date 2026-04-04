@echo off
chcp 65001 >nul
echo ============================================================
echo   Qwen3-TTS 本地声音克隆服务 - 一键安装
echo ============================================================
echo.

:: 检查 uv 是否安装
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] uv 未安装，正在安装...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo [OK] uv 安装完成，请重新打开命令行后再次运行此脚本
    pause
    exit /b 0
)

echo [OK] 检测到 uv: 
uv --version
echo.

:: 检查 ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ffmpeg 未检测到，正在尝试用 winget 安装...
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    echo [*] 安装后请重新打开命令行以使 PATH 生效
) else (
    echo [OK] ffmpeg 已安装
)
echo.

:: 创建虚拟环境
echo [*] 创建 Python 3.11 虚拟环境...
uv venv --python 3.11 .venv
echo.

:: 安装 PyTorch (CUDA 12.8)
echo [*] 安装 PyTorch 2.11 + CUDA 12.8...
uv pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
echo.

:: 安装其余依赖
echo [*] 安装项目依赖...
uv pip install -r requirements.txt
echo.

:: 复制环境变量
if not exist .env (
    copy .env.example .env
    echo [OK] 已创建 .env 文件，请根据需要修改配置
) else (
    echo [*] .env 文件已存在，跳过
)

echo.
echo ============================================================
echo   [OK] 安装完成！
echo   运行服务: start.bat
echo ============================================================
pause
