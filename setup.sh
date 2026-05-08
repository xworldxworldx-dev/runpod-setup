#!/bin/bash
# NAU RunPod Setup Script
# 사용법: bash <(curl -s https://raw.githubusercontent.com/xworldxworldx-dev/runpod-setup/main/setup.sh)

set -e

export HF_HUB_DISABLE_XET=1

COMFY_MODELS="/workspace/ComfyUI/models"
SLIM_MODELS="/workspace/runpod-slim/ComfyUI/models"
CUSTOM_NODES="/workspace/runpod-slim/ComfyUI/custom_nodes"

echo "=============================="
echo " NAU RunPod Setup 시작"
echo "=============================="

# ─────────────────────────────────
# 1. 디렉토리 생성
# ─────────────────────────────────
echo "[1/5] 디렉토리 생성..."
mkdir -p $COMFY_MODELS/diffusion_models
mkdir -p $COMFY_MODELS/text_encoders
mkdir -p $COMFY_MODELS/vae
mkdir -p $COMFY_MODELS/clip_vision
mkdir -p $COMFY_MODELS/loras
mkdir -p $COMFY_MODELS/upscale_models

# ─────────────────────────────────
# 2. 모델 다운로드 (병렬, XET 비활성화)
# ─────────────────────────────────
echo "[2/5] 모델 다운로드 시작 (병렬)..."

python3 << 'PYEOF'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

downloads = [
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors', '/workspace/ComfyUI/models/diffusion_models'),
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors', '/workspace/ComfyUI/models/diffusion_models'),
    ('FX-FeiHou/wan2.2-Remix', 'NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors', '/workspace/ComfyUI/models/diffusion_models'),
    ('FX-FeiHou/wan2.2-Remix', 'NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors', '/workspace/ComfyUI/models/diffusion_models'),
    ('NSFW-API/NSFW-Wan-UMT5-XXL', 'nsfw_wan_umt5-xxl_fp8_scaled.safetensors', '/workspace/ComfyUI/models/text_encoders'),
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', '/workspace/ComfyUI/models/text_encoders'),
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'split_files/vae/wan_2.1_vae.safetensors', '/workspace/ComfyUI/models/vae'),
    ('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/clip_vision/clip_vision_h.safetensors', '/workspace/ComfyUI/models/clip_vision'),
    ('lightx2v/Wan2.2-Distill-Loras', 'wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors', '/workspace/ComfyUI/models/loras'),
    ('lightx2v/Wan2.2-Distill-Loras', 'wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors', '/workspace/ComfyUI/models/loras'),
]

def download(args):
    repo, filename, local_dir = args
    name = filename.split('/')[-1]
    print(f"  → {name}")
    hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(f"  ✓ {name}")

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(download, d): d for d in downloads}
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            repo, filename, _ = futures[future]
            print(f"  ✗ 실패: {filename.split('/')[-1]} — {e}")

print("HuggingFace 다운로드 완료")
PYEOF

echo "  → RealESRGAN_x4plus.pth"
wget -q --show-progress \
    -O $COMFY_MODELS/upscale_models/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

echo "[2/5] 다운로드 완료"

# ─────────────────────────────────
# 3. 심볼릭 링크
# ─────────────────────────────────
echo "[3/5] 심볼릭 링크 설정..."

ln -sf $COMFY_MODELS/diffusion_models/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors $SLIM_MODELS/diffusion_models/
ln -sf $COMFY_MODELS/diffusion_models/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors $SLIM_MODELS/diffusion_models/
ln -sf $COMFY_MODELS/diffusion_models/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors $SLIM_MODELS/diffusion_models/
ln -sf $COMFY_MODELS/diffusion_models/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors $SLIM_MODELS/diffusion_models/
ln -sf $COMFY_MODELS/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors $SLIM_MODELS/text_encoders/
ln -sf $COMFY_MODELS/text_encoders/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $SLIM_MODELS/text_encoders/
ln -sf $COMFY_MODELS/vae/split_files/vae/wan_2.1_vae.safetensors $SLIM_MODELS/vae/
ln -sf $COMFY_MODELS/clip_vision/split_files/clip_vision/clip_vision_h.safetensors $SLIM_MODELS/clip_vision/
ln -sf $COMFY_MODELS/loras/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors $SLIM_MODELS/loras/
ln -sf $COMFY_MODELS/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors $SLIM_MODELS/loras/
ln -sf $COMFY_MODELS/upscale_models/RealESRGAN_x4plus.pth $SLIM_MODELS/upscale_models/

echo "[3/5] 심볼릭 링크 완료"

# ─────────────────────────────────
# 4. 커스텀 노드 설치
# ─────────────────────────────────
echo "[4/5] 커스텀 노드 설치..."

cd $CUSTOM_NODES

if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
    echo "  → ComfyUI-WanVideoWrapper"
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
    pip install -r ComfyUI-WanVideoWrapper/requirements.txt -q
else
    echo "  ✓ ComfyUI-WanVideoWrapper (이미 설치됨)"
fi

if [ ! -d "ComfyUI-Frame-Interpolation" ]; then
    echo "  → ComfyUI-Frame-Interpolation"
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
    pip install -r ComfyUI-Frame-Interpolation/requirements-no-cupy.txt -q
else
    echo "  ✓ ComfyUI-Frame-Interpolation (이미 설치됨)"
fi

if [ ! -d "ComfyUI-Custom-Scripts" ]; then
    echo "  → ComfyUI-Custom-Scripts"
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
else
    echo "  ✓ ComfyUI-Custom-Scripts (이미 설치됨)"
fi

if [ ! -d "ComfyUI-Easy-Use" ]; then
    echo "  → ComfyUI-Easy-Use"
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git
    pip install -r ComfyUI-Easy-Use/requirements.txt -q 2>/dev/null || true
else
    echo "  ✓ ComfyUI-Easy-Use (이미 설치됨)"
fi

echo "  → diffusers 업그레이드"
/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/pip install -U diffusers huggingface_hub -q

echo "[4/5] 커스텀 노드 설치 완료"

# ─────────────────────────────────
# 5. ComfyUI 재시작
# ─────────────────────────────────
echo "[5/5] ComfyUI 재시작..."
kill $(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 3
cd /workspace/runpod-slim/ComfyUI
source .venv-cu128/bin/activate
python main.py --listen 0.0.0.0 --port 8188 > /tmp/comfyui.log 2>&1 &

sleep 5
echo ""
echo "=============================="
echo " 셋업 완료!"
echo " 포트 8188 접속 후 Wan22-I2V-Remix.json 드래그 앤 드롭"
echo " 로그 확인: tail -f /tmp/comfyui.log"
echo "=============================="
