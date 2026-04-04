"""
测试脚本 - 验证 Qwen3-TTS 本地克隆服务
用法: python test_api.py
"""
import os
import sys
import json
import requests
import time

# ============================================================
# 配置
# ============================================================
BASE_URL = "http://127.0.0.1:8880"
API_KEY = "sk-local-qwen3-tts"

HEADERS_AUTH = {"Authorization": f"Bearer {API_KEY}"}
HEADERS_JSON = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def test_health():
    """测试 1: 健康检查"""
    print("\n🔍 测试 1: 健康检查")
    resp = requests.get(f"{BASE_URL}/health")
    data = resp.json()
    print(f"   状态: {data['status']}")
    print(f"   模型已加载: {data['model_loaded']}")
    print(f"   已注册音色: {data['registered_voices']}")
    assert data["status"] == "ok", "健康检查失败"
    assert data["model_loaded"] == True, "模型未加载"
    print("   ✅ 通过")


def test_upload_voice(audio_path: str, custom_name: str, ref_text: str = "") -> str:
    """测试 2: 上传音色"""
    print(f"\n🔍 测试 2: 上传音色 ({custom_name})")

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data = {
            "model": "qwen3-tts",
            "customName": custom_name,
            "text": ref_text,
        }
        resp = requests.post(
            f"{BASE_URL}/v1/audio/voice/upload",
            headers=HEADERS_AUTH,
            files=files,
            data=data,
        )

    assert resp.status_code == 200, f"上传失败: {resp.text}"
    uri = resp.json().get("uri")
    print(f"   URI: {uri}")
    print("   ✅ 通过")
    return uri


def test_generate_tts(text: str, voice_uri: str, output_file: str = "test_output.mp3"):
    """测试 3: 生成语音"""
    print(f"\n🔍 测试 3: 生成语音")
    print(f"   文本: {text[:50]}...")
    print(f"   音色: {voice_uri}")

    payload = {
        "model": "qwen3-tts",
        "input": text,
        "voice": voice_uri,
        "response_format": "mp3",
        "speed": 1.0,
    }

    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json=payload,
        headers=HEADERS_JSON,
        timeout=120,
    )
    elapsed = time.time() - t0

    assert resp.status_code == 200, f"生成失败: {resp.text}"

    with open(output_file, "wb") as f:
        f.write(resp.content)

    print(f"   耗时: {elapsed:.1f}s")
    print(f"   文件大小: {len(resp.content) / 1024:.1f} KB")
    print(f"   已保存: {output_file}")
    print("   ✅ 通过")


def test_list_voices():
    """测试 4: 列出音色"""
    print("\n🔍 测试 4: 列出已注册音色")
    resp = requests.get(
        f"{BASE_URL}/v1/audio/voices",
        headers=HEADERS_AUTH,
    )
    assert resp.status_code == 200
    voices = resp.json().get("voices", [])
    for v in voices:
        print(f"   - {v['custom_name']} → {v['uri']}")
    print(f"   总计: {len(voices)} 个音色")
    print("   ✅ 通过")


if __name__ == "__main__":
    print("=" * 60)
    print("  Qwen3-TTS 本地克隆服务 - API 测试")
    print("=" * 60)

    # 1. 健康检查
    test_health()

    # 2. 上传音色 (需要一个参考音频文件)
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        ref_text = sys.argv[2] if len(sys.argv) > 2 else ""
        custom_name = os.path.basename(audio_path).split(".")[0]

        uri = test_upload_voice(audio_path, custom_name, ref_text)

        # 3. 生成语音
        test_generate_tts(
            text="大家好，欢迎来到我的频道，今天我们来聊一聊人工智能的最新发展。",
            voice_uri=uri,
            output_file="test_output.mp3",
        )
    else:
        print("\n⚠️ 跳过上传/生成测试 (未提供音频文件)")
        print("   用法: python test_api.py <音频文件路径> [参考文本]")

    # 4. 列出音色
    test_list_voices()

    print("\n" + "=" * 60)
    print("  全部测试完成 ✅")
    print("=" * 60)
