"""
Qwen3-TTS 本地克隆服务 - 配置文件
"""
import os

# ============================================================
# 服务器配置
# ============================================================
HOST = os.getenv("TTS_HOST", "0.0.0.0")
PORT = int(os.getenv("TTS_PORT", "8880"))

# API Key (可选, 设为空字符串则不验证)
API_KEY = os.getenv("TTS_API_KEY", "sk-local-qwen3-tts")

# ============================================================
# 模型配置
# ============================================================
# RTX 3060 12GB → 使用 0.6B 模型以确保显存够用
# 如果你有更大显存的卡，可改为 1.7B 模型
TTS_BASE_MODEL = os.getenv(
    "TTS_BASE_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"     # 声音克隆模型
)
TTS_CUSTOM_VOICE_MODEL = os.getenv(
    "TTS_CUSTOM_VOICE_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # 内置音色模型
)

# 设备 & 精度
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
DTYPE = "bfloat16"  # bfloat16 适合 Ampere+ 架构

# FlashAttention 2: RTX 3060 可用, 但 Windows 上安装困难
# 如果安装了 flash-attn 则自动启用, 否则使用 sdpa
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = "sdpa"

# ============================================================
# 存储配置
# ============================================================
# 上传的语音文件存储目录
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
# 生成的音频存储目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
# 音色指纹缓存文件
VOICE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "voice_cache.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
