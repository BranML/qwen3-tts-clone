"""
Qwen3-TTS 引擎 - 模型加载 & 推理封装
"""
import io
import os
import sys
import json
import time
import hashlib
import threading
from typing import Optional

import torch
import numpy as np
import soundfile as sf

import config as cfg

# Windows 控制台编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TTSEngine:
    """封装 Qwen3-TTS 模型的加载和推理"""

    def __init__(self):
        self.base_model = None
        self.voice_cache: dict[str, dict] = {}   # uri -> prompt_items
        self.voice_meta: dict[str, str] = {}      # uri -> custom_name
        self._lock = threading.Lock()
        self._load_voice_cache()

    def load_models(self):
        """加载 TTS 模型 (Base 模型用于声音克隆)"""
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, cfg.DTYPE)

        print(f"[LOAD] 加载 Base 模型: {cfg.TTS_BASE_MODEL}")
        print(f"       设备: {cfg.DEVICE}, 精度: {cfg.DTYPE}, 注意力: {cfg.ATTN_IMPL}")
        t0 = time.time()

        self.base_model = Qwen3TTSModel.from_pretrained(
            cfg.TTS_BASE_MODEL,
            device_map=cfg.DEVICE,
            dtype=dtype,
            attn_implementation=cfg.ATTN_IMPL,
        )

        print(f"[OK] 模型加载完成, 耗时 {time.time() - t0:.1f}s")

        # -------------------------------------------------------
        # torch.compile() 加速: 使用 CUDA Graph + 算子融合
        # mode="reduce-overhead" 在 Windows 上无需 Triton 即可生效
        # -------------------------------------------------------
        if cfg.TORCH_COMPILE and torch.cuda.is_available():
            print("[OPT] 正在编译模型 (torch.compile)，首次推理会额外花 30-60s...")
            try:
                # 只编译 LM 部分（生成 token 的核心），避免编译整个模型失败
                if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "forward"):
                    self.base_model.model = torch.compile(
                        self.base_model.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[OK] torch.compile 完成")
                else:
                    print("[SKIP] 模型结构不支持 torch.compile，跳过")
            except Exception as e:
                print(f"[WARN] torch.compile 失败，跳过: {e}")

        # -------------------------------------------------------
        # Warmup: 预热 CUDA 算子，消除首次推理的冷启动延迟
        # -------------------------------------------------------
        if cfg.WARMUP_ON_START:
            self._warmup()

    def _warmup(self):
        """推理预热 - 让 CUDA 提前编译 kernel，首次真实请求不再卡顿"""
        print("[WARM] 正在预热模型...")
        t0 = time.time()
        try:
            # 用一个短文本做一次完整推理，但不保存结果
            dummy_meta = list(self.voice_meta.values())
            if dummy_meta:
                # 用已有的第一个音色做预热
                first_uri = list(self.voice_meta.keys())[0]
                with torch.inference_mode():
                    prompt_items = self._get_or_build_prompt(first_uri)
                    self.base_model.generate_voice_clone(
                        text="你好",
                        language="Auto",
                        voice_clone_prompt=prompt_items,
                    )
                print(f"[OK] 预热完成, 耗时 {time.time() - t0:.1f}s")
            else:
                print("[WARM] 无已注册音色，跳过预热（首次请求时会稍慢）")
        except Exception as e:
            print(f"[WARN] 预热失败，不影响正常使用: {e}")

    # ----------------------------------------------------------
    # 音色管理
    # ----------------------------------------------------------
    def _load_voice_cache(self):
        """从磁盘加载已注册的音色元数据"""
        if os.path.exists(cfg.VOICE_CACHE_FILE):
            with open(cfg.VOICE_CACHE_FILE, "r", encoding="utf-8") as f:
                self.voice_meta = json.load(f)

    def _save_voice_cache(self):
        """保存音色元数据到磁盘"""
        with open(cfg.VOICE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.voice_meta, f, indent=2, ensure_ascii=False)

    def register_voice(
        self,
        audio_data: bytes,
        filename: str,
        custom_name: str,
        ref_text: str,
    ) -> str:
        """
        注册一个音色: 保存音频文件, 生成 URI 标识符
        返回: voice URI (格式: voice-clone://<hash>)
        """
        # 生成唯一 hash 作为 voice id
        content_hash = hashlib.md5(audio_data).hexdigest()[:12]
        voice_uri = f"voice-clone://{custom_name}_{content_hash}"

        # 保存音频文件
        ext = os.path.splitext(filename)[1] or ".mp3"
        save_path = os.path.join(cfg.UPLOAD_DIR, f"{custom_name}_{content_hash}{ext}")
        with open(save_path, "wb") as f:
            f.write(audio_data)

        # 保存元数据
        with self._lock:
            self.voice_meta[voice_uri] = {
                "custom_name": custom_name,
                "audio_path": save_path,
                "ref_text": ref_text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save_voice_cache()

        print(f"[OK] 音色已注册: {custom_name} -> {voice_uri}")
        return voice_uri

    def _get_or_build_prompt(self, voice_uri: str):
        """获取或构建 voice_clone_prompt (带缓存)"""
        if voice_uri in self.voice_cache:
            return self.voice_cache[voice_uri]

        meta = self.voice_meta.get(voice_uri)
        if meta is None:
            raise ValueError(f"未找到音色: {voice_uri}")

        audio_path = meta["audio_path"]
        ref_text = meta.get("ref_text", "")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        print(f"[BUILD] 构建克隆 prompt: {meta['custom_name']}")

        # 使用 x_vector_only_mode 当没有参考文本时
        use_xvector_only = not ref_text or ref_text.strip() == ""

        prompt_items = self.base_model.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=ref_text if not use_xvector_only else None,
            x_vector_only_mode=use_xvector_only,
        )

        with self._lock:
            self.voice_cache[voice_uri] = prompt_items

        return prompt_items

    # ----------------------------------------------------------
    # TTS 生成
    # ----------------------------------------------------------
    def generate_speech(
        self,
        text: str,
        voice_uri: str,
        language: str = "Auto",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> bytes:
        """
        使用克隆的音色生成语音
        返回: 音频字节数据
        """
        if self.base_model is None:
            raise RuntimeError("模型未加载, 请先调用 load_models()")

        # 获取克隆 prompt
        prompt_items = self._get_or_build_prompt(voice_uri)

        t0 = time.time()

        # torch.inference_mode: 禁用梯度计算，节省显存 & 提速
        with torch.inference_mode():
            wavs, sr = self.base_model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )

        infer_ms = (time.time() - t0) * 1000
        audio_len_s = len(wavs[0]) / sr if sr > 0 else 0
        rtf = (infer_ms / 1000) / audio_len_s if audio_len_s > 0 else 0
        print(f"[PERF] 推理耗时 {infer_ms:.0f}ms | 音频时长 {audio_len_s:.1f}s | RTF {rtf:.2f}")

        wav_data = wavs[0]

        # 转换格式
        audio_bytes = self._convert_audio(wav_data, sr, response_format)
        return audio_bytes

    def _convert_audio(
        self,
        wav_data: np.ndarray,
        sr: int,
        fmt: str,
    ) -> bytes:
        """将 numpy 音频数据转换为指定格式的字节"""
        buf = io.BytesIO()

        if fmt in ("wav", "flac", "ogg"):
            sf.write(buf, wav_data, sr, format=fmt.upper())
        elif fmt == "mp3":
            # 先写成 WAV, 再用 pydub 转 MP3
            sf.write(buf, wav_data, sr, format="WAV")
            buf.seek(0)
            try:
                from pydub import AudioSegment
                audio_seg = AudioSegment.from_wav(buf)
                mp3_buf = io.BytesIO()
                audio_seg.export(mp3_buf, format="mp3", bitrate="192k")
                return mp3_buf.getvalue()
            except Exception:
                # 如果 pydub/ffmpeg 不可用, 返回 WAV
                buf.seek(0)
                return buf.read()
        else:
            sf.write(buf, wav_data, sr, format="WAV")

        buf.seek(0)
        return buf.read()


# 全局单例
engine = TTSEngine()
