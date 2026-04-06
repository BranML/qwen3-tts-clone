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
DTYPE = "bfloat16"  # bfloat16 适合 Ampere+ 架构（RTX 30/40 系列）

# FlashAttention 2: RTX 3060 可用, 但 Windows 上需安装预编译 wheel
# 如果安装了 flash-attn 则自动启用, 否则使用 sdpa (PyTorch 内置，免安装)
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = "sdpa"

# ============================================================
# 推理加速配置
# ============================================================

# torch.compile(): 使用 CUDA Graph + 算子融合加速推理
# - 首次启动会多花 30-60s 编译
# - 后续每次推理可加速 20-40%
# - 在 Windows 上设置为 False 如遇到问题
TORCH_COMPILE = os.getenv("TTS_TORCH_COMPILE", "false").lower() == "true"

# 启动时预热: 服务启动后立即做一次推理，消除首次请求的冷启动延迟
# 需要已有注册的音色才会生效
WARMUP_ON_START = os.getenv("TTS_WARMUP", "true").lower() == "true"

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
