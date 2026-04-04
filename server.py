"""
Qwen3-TTS 本地克隆服务 - FastAPI 接口
兼容用户现有调用代码:
  - POST /v1/audio/voice/upload   上传音色 (返回 {uri: ...})
  - POST /v1/audio/speech          TTS 生成   (返回音频二进制)
"""
import os
import sys
import uuid
import time
from contextlib import asynccontextmanager

# Windows 控制台编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, Form, UploadFile, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

import config as cfg
from tts_engine import engine


# ----------------------------------------------------------
# 鉴权中间件
# ----------------------------------------------------------
def verify_api_key(request: Request):
    """验证 API Key (如果已配置)"""
    if not cfg.API_KEY:
        return  # 未配置则跳过

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = auth

    if token != cfg.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ----------------------------------------------------------
# 应用生命周期
# ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    print("=" * 60)
    print("[*] Qwen3-TTS 本地克隆服务启动中...")
    print("=" * 60)
    engine.load_models()
    print("=" * 60)
    print(f"[OK] 服务已就绪: http://{cfg.HOST}:{cfg.PORT}")
    print(f"     API Key: {cfg.API_KEY}")
    print("=" * 60)
    yield
    print("[*] 服务已关闭")


app = FastAPI(
    title="Qwen3-TTS 本地声音克隆服务",
    description="兼容 OpenAI TTS API 格式的本地 Qwen3-TTS 服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------
# API: 上传音色
# 兼容用户代码:
#   POST /v1/audio/voice/upload
#   Headers: Authorization: Bearer <key>
#   Form: file=<audio>, model=<model>, customName=<name>, text=<ref_text>
#   Response: {"uri": "voice-clone://..."}
# ----------------------------------------------------------
@app.post("/v1/audio/voice/upload")
async def upload_voice(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("qwen3-tts"),
    customName: str = Form("default"),
    text: str = Form(""),
):
    verify_api_key(request)

    audio_data = await file.read()
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    uri = engine.register_voice(
        audio_data=audio_data,
        filename=file.filename or "upload.mp3",
        custom_name=customName,
        ref_text=text,
    )

    return JSONResponse(content={"uri": uri})


# ----------------------------------------------------------
# API: TTS 生成
# 兼容用户代码:
#   POST /v1/audio/speech
#   Headers: Authorization: Bearer <key>, Content-Type: application/json
#   Body: {"model": "...", "input": "text", "voice": "uri",
#          "response_format": "mp3", "speed": 1.0}
#   Response: 音频二进制数据
# ----------------------------------------------------------
class TTSRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str
    response_format: str = "mp3"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def generate_speech(request: Request, body: TTSRequest):
    verify_api_key(request)

    if not body.input or not body.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    if not body.voice:
        raise HTTPException(status_code=400, detail="Voice URI is required")

    try:
        audio_bytes = engine.generate_speech(
            text=body.input,
            voice_uri=body.voice,
            speed=body.speed,
            response_format=body.response_format,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[ERROR] TTS 生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    # 根据格式设置 Content-Type
    content_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }
    content_type = content_types.get(body.response_format, "audio/wav")

    return Response(content=audio_bytes, media_type=content_type)


# ----------------------------------------------------------
# API: 列出已注册的音色
# ----------------------------------------------------------
@app.get("/v1/audio/voices")
async def list_voices(request: Request):
    verify_api_key(request)
    voices = []
    for uri, meta in engine.voice_meta.items():
        voices.append({
            "uri": uri,
            "custom_name": meta.get("custom_name", ""),
            "created_at": meta.get("created_at", ""),
        })
    return JSONResponse(content={"voices": voices})


# ----------------------------------------------------------
# 健康检查
# ----------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine.base_model is not None,
        "registered_voices": len(engine.voice_meta),
    }


# ----------------------------------------------------------
# 启动入口
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=cfg.HOST,
        port=cfg.PORT,
        workers=1,  # 单 GPU 只用 1 个 worker
    )
