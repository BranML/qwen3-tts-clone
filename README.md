# Qwen3-TTS 本地声音克隆服务

基于 **Qwen3-TTS** 的本地语音克隆服务，完全兼容 OpenAI TTS API 格式，支持上传参考音频克隆任意声音。

---

## 硬件要求

| 配置项 | 最低要求 | 推荐 |
|--------|---------|------|
| GPU | NVIDIA RTX 3060 (12GB VRAM) | RTX 3090 / 4090 |
| CUDA | 12.x | 12.8 |
| RAM | 16GB | 32GB |
| 磁盘 | 10GB 可用空间 | 20GB |
| OS | Windows 10/11 | Windows 11 / Ubuntu 22.04 |

> 💡 **显存说明**
> - 12GB 显存 → 使用 `0.6B` 模型（默认）
> - 24GB+ 显存 → 可改用 `1.7B` 模型效果更好

---

## 快速部署

### 前置依赖（手动安装一次）

#### 1. 安装 uv（Python 包管理器）

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. 安装 ffmpeg（音频格式转换必需）

**Windows（推荐用 winget）:**
```cmd
winget install -e --id Gyan.FFmpeg
```

或到 https://ffmpeg.org/download.html 手动下载，将 `bin/` 目录加入系统 PATH。

**Ubuntu / Debian:**
```bash
sudo apt-get install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

> ⚠️ 安装 ffmpeg 后需重新打开终端，确保 `ffmpeg --version` 可以正常运行。

#### 3. 安装 NVIDIA CUDA 驱动

确保 CUDA 版本 ≥ 12.0：
```bash
nvidia-smi
```

---

### 一键安装环境

克隆项目后，在项目根目录执行：

**Windows:**
```cmd
install.bat
```

**Linux / macOS:**
```bash
chmod +x install.sh && ./install.sh
```

安装脚本会自动完成：
- ✅ 创建 Python 3.11 虚拟环境（`.venv/`）
- ✅ 安装 PyTorch 2.11 + CUDA 12.8
- ✅ 安装全部 Python 依赖
- ✅ 创建 `.env` 配置文件

---

### 手动安装（如自动安装失败）

```bash
# 1. 创建虚拟环境
uv venv --python 3.11 .venv

# 2. 安装 PyTorch（CUDA 12.8）
uv pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其余依赖
uv pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
```

---

## 配置

编辑 `.env` 文件（从 `.env.example` 复制而来）：

```env
# 服务地址和端口
TTS_HOST=0.0.0.0
TTS_PORT=8880

# API Key（客户端需在请求头中携带）
TTS_API_KEY=sk-local-qwen3-tts

# 模型选择
# 12GB 显存: 使用 0.6B 模型
TTS_BASE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
# 24GB+ 显存: 使用 1.7B 模型（效果更好）
# TTS_BASE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base

# 推理设备
TTS_DEVICE=cuda:0
```

---

## 启动服务

**Windows:**
```cmd
start.bat
```

或直接：
```cmd
uv run python server.py
```

**Linux / macOS:**
```bash
bash start.sh
```

服务成功启动后输出：
```
[OK] 服务已就绪: http://0.0.0.0:8880
     API Key: sk-local-qwen3-tts
```

首次启动会自动从 HuggingFace 下载模型（约 2-5GB），需要保持网络连接。

---

## API 使用说明

### 上传音色

```bash
curl -X POST http://localhost:8880/v1/audio/voice/upload \
  -H "Authorization: Bearer sk-local-qwen3-tts" \
  -F "file=@reference.mp3" \
  -F "customName=my_voice" \
  -F "text=参考音频里实际说的内容"
```

> ⚠️ **`text` 字段非常重要**：必须填写参考音频中实际说的话，模型依此对齐音频与文本。
> 如果不知道内容，可用 [Whisper](https://github.com/openai/whisper) 先转录音频。

响应：
```json
{"uri": "voice-clone://my_voice_a1b2c3d4e5f6"}
```

### 生成语音

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Authorization: Bearer sk-local-qwen3-tts" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "你好，这是一段使用克隆声音生成的语音。",
    "voice": "voice-clone://my_voice_a1b2c3d4e5f6",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output output.mp3
```

### 查看已注册音色

```bash
curl http://localhost:8880/v1/audio/voices \
  -H "Authorization: Bearer sk-local-qwen3-tts"
```

### 健康检查

```bash
curl http://localhost:8880/health
```

---

## 项目结构

```
qwen3-tts-clone/
├── server.py          # FastAPI 服务主入口
├── tts_engine.py      # TTS 推理引擎封装
├── config.py          # 配置读取（读取 .env）
├── test_api.py        # API 测试脚本
│
├── .env.example       # 环境变量示例（提交到 git）
├── .env               # 实际配置（不提交到 git）
│
├── requirements.txt   # Python 依赖（精确版本锁定）
├── pyproject.toml     # uv 项目配置
│
├── install.bat        # Windows 一键安装脚本
├── install.sh         # Linux/macOS 一键安装脚本
├── start.bat          # Windows 启动脚本
├── start.sh           # Linux/macOS 启动脚本
│
├── uploads/           # 上传的参考音频（自动创建）
├── outputs/           # 生成的音频文件（自动创建）
└── voice_cache.json   # 已注册音色元数据
```

---

## 常见问题

### ffmpeg / SoX 相关错误

```
'sox' 不是内部或外部命令
SoX could not be found!
```

安装 ffmpeg 并确保已添加到 PATH，重启终端后重新启动服务。

### CUDA 不可用

```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch 是否识别到 GPU
uv run python -c "import torch; print(torch.cuda.is_available())"
```

如输出 `False`，请确认安装了正确的 CUDA 版本的 PyTorch。

### 克隆音色效果差

1. **检查参考文本**：`text` 字段必须是参考音频里实际说的内容，不能乱填
2. **参考音频质量**：建议使用 10-30 秒、无背景噪音的清晰录音
3. **参考音频格式**：支持 MP3、WAV、FLAC，采样率建议 ≥ 16kHz

用 Whisper 自动获取参考文本：
```bash
uv pip install openai-whisper
uv run python -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('reference.mp3', language='zh')
print(result['text'])
"
```

### 模型下载慢

在国内可以使用 HuggingFace 镜像：
```env
# 在 .env 中添加
HF_ENDPOINT=https://hf-mirror.com
```

或在启动前设置环境变量：
```cmd
:: Windows
set HF_ENDPOINT=https://hf-mirror.com
uv run python server.py
```

```bash
# Linux
HF_ENDPOINT=https://hf-mirror.com uv run python server.py
```

---

## 依赖版本说明

| 包 | 版本 | 说明 |
|----|------|------|
| torch | 2.11.0+cu128 | PyTorch CUDA 12.8 版本 |
| qwen-tts | 0.1.1 | Qwen3-TTS 推理库 |
| transformers | 4.57.3 | HuggingFace 模型加载 |
| fastapi | 0.135.3 | Web API 框架 |
| pydub | 0.25.1 | 音频格式转换（依赖 ffmpeg） |
| soundfile | 0.13.1 | WAV/FLAC 音频读写 |
| librosa | 0.11.0 | 音频特征提取 |
