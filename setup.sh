#!/bin/bash
# NAU RunPod Setup Script
# 사용법: bash <(curl -s https://raw.githubusercontent.com/xworldxworldx-dev/runpod-setup/main/setup.sh)
#
# [전제조건]
# - Pod 템플릿: ComfyUI CUDA 13 (runpod-workers/comfyui-base)
# - GPU: RTX 5090
# - Container disk: 50GB 이상 (로컬 모델 복사용 — 3.5단계)
# - Network volume: /workspace 마운트
#
# [ComfyUI 워크플로우 권장 설정 (5090 기준)]
# - Model Loader quantization: fp8_e4m3fn (양쪽)
# - Sampler force_offload: false (양쪽)
# - Block Swap blocks_to_swap: 0
# - fps: 32 / num_frames: 81 / cfg: 1.0 / motion_amplitude: 1.4

set -e

COMFY_MODELS="/workspace/ComfyUI/models"
SLIM_MODELS="/workspace/runpod-slim/ComfyUI/models"
CUSTOM_NODES="/workspace/runpod-slim/ComfyUI/custom_nodes"
LOCAL_MODELS="/root/models_local"

echo "=============================="
echo " NAU RunPod Setup 시작"
echo "=============================="

# ─────────────────────────────────
# 1. 디렉토리 생성
# ─────────────────────────────────
echo "[1/6] 디렉토리 생성..."
mkdir -p $COMFY_MODELS/diffusion_models
mkdir -p $COMFY_MODELS/text_encoders
mkdir -p $COMFY_MODELS/vae
mkdir -p $COMFY_MODELS/clip_vision
mkdir -p $COMFY_MODELS/loras
mkdir -p $COMFY_MODELS/upscale_models

# 타겟(runpod-slim) 디렉토리도 생성 — 템플릿에 따라 없을 수 있음
mkdir -p $SLIM_MODELS/diffusion_models
mkdir -p $SLIM_MODELS/text_encoders
mkdir -p $SLIM_MODELS/vae
mkdir -p $SLIM_MODELS/clip_vision
mkdir -p $SLIM_MODELS/loras
mkdir -p $SLIM_MODELS/upscale_models
mkdir -p $CUSTOM_NODES

# ─────────────────────────────────
# 2. 모델 다운로드 (hf_transfer + 병렬)
# ─────────────────────────────────
echo "[2/6] huggingface_hub / hf_transfer 설치..."
pip install huggingface_hub hf_transfer -q

echo "[2/6] 모델 다운로드 시작 (병렬)..."

python3 << 'PYEOF'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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

# 업스케일 모델
echo "  → RealESRGAN_x4plus.pth"
wget -q --show-progress -O $COMFY_MODELS/upscale_models/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

echo "  → 4xUltraSharp.pth"
wget -q --show-progress -O $COMFY_MODELS/upscale_models/4xUltraSharp.pth \
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"

echo "  → 4xNMKD-Superscale.pth"
wget -q --show-progress -O $COMFY_MODELS/upscale_models/4xNMKD-Superscale.pth \
    "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth"

echo "[2/6] 다운로드 완료"

# ─────────────────────────────────
# 3. 심볼릭 링크 (network volume 기준)
# ─────────────────────────────────
echo "[3/6] 심볼릭 링크 설정..."

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
ln -sf $COMFY_MODELS/upscale_models/4xUltraSharp.pth $SLIM_MODELS/upscale_models/
ln -sf $COMFY_MODELS/upscale_models/4xNMKD-Superscale.pth $SLIM_MODELS/upscale_models/

echo "[3/6] 심볼릭 링크 완료"

# ─────────────────────────────────
# 4. 모델 로컬 디스크 복사 (프로젝트 A)
# ─────────────────────────────────
# 목적: network volume에서 모델 로드 시 3~8분 병목 → 로컬 디스크에서 1초
# 주의: 컨테이너 디스크는 Pod 종료 시 삭제됨 → Pod 시작마다 이 단계 필요
#       Container disk 50GB 이상 필요 (Remix 모델 2개 = 27GB)
echo "[4/6] 모델 로컬 디스크 복사 (로딩 속도 최적화)..."

mkdir -p $LOCAL_MODELS

if [ ! -f "$LOCAL_MODELS/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors" ]; then
    echo "  → high_lighting 로컬 복사 중... (14GB)"
    cp $COMFY_MODELS/diffusion_models/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors $LOCAL_MODELS/ &
else
    echo "  ✓ high_lighting (이미 로컬에 있음)"
fi

if [ ! -f "$LOCAL_MODELS/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors" ]; then
    echo "  → low_lighting 로컬 복사 중... (14GB)"
    cp $COMFY_MODELS/diffusion_models/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors $LOCAL_MODELS/ &
else
    echo "  ✓ low_lighting (이미 로컬에 있음)"
fi

wait
echo "  ✓ 로컬 복사 완료"

# Remix 심볼릭 링크를 로컬 파일로 교체 (3단계의 volume 링크를 덮어씀)
ln -sf $LOCAL_MODELS/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_v2.0.safetensors $SLIM_MODELS/diffusion_models/
ln -sf $LOCAL_MODELS/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_v2.0.safetensors $SLIM_MODELS/diffusion_models/

echo "[4/6] 로컬 디스크 최적화 완료 (모델 로딩: 분 단위 → 1초)"

# ─────────────────────────────────
# 5. 커스텀 노드 설치
# ─────────────────────────────────
echo "[5/6] 커스텀 노드 설치..."

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

if [ ! -d "ComfyUI-PainterI2VforKJ" ]; then
    echo "  → ComfyUI-PainterI2VforKJ"
    git clone https://github.com/princepainter/ComfyUI-PainterI2VforKJ.git
else
    echo "  ✓ ComfyUI-PainterI2VforKJ (이미 설치됨)"
fi

echo "  → diffusers 업그레이드"
if [ -f "/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/pip" ]; then
    /workspace/runpod-slim/ComfyUI/.venv-cu128/bin/pip install -U diffusers huggingface_hub hf_transfer -q
elif [ -f "/workspace/ComfyUI/venv/bin/pip" ]; then
    /workspace/ComfyUI/venv/bin/pip install -U diffusers huggingface_hub hf_transfer -q
else
    pip install -U diffusers huggingface_hub hf_transfer -q
fi

echo "[5/6] 커스텀 노드 설치 완료"

# ─────────────────────────────────
# 6. ComfyUI 재시작
# ─────────────────────────────────
echo "[6/6] ComfyUI 재시작..."
kill $(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 3

if [ -f "/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/activate" ]; then
    cd /workspace/runpod-slim/ComfyUI
    source .venv-cu128/bin/activate
elif [ -f "/workspace/ComfyUI/venv/bin/activate" ]; then
    cd /workspace/ComfyUI
    source venv/bin/activate
else
    echo "  ⚠️ 가상환경을 찾지 못함 — ComfyUI 수동 시작 필요"
    exit 1
fi

python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header > /tmp/comfyui.log 2>&1 &

sleep 5
echo ""
echo "=============================="
echo " 셋업 완료!"
echo " 포트 8188 접속 후 Wan22-I2V-Remix.json 드래그 앤 드롭"
echo " 로그 확인: tail -f /tmp/comfyui.log"
echo ""
echo " [ComfyUI 워크플로우 확인사항 - 5090]"
echo "  quantization: fp8_e4m3fn / force_offload: false"
echo "  blocks_to_swap: 0 / fps: 32 / cfg: 1.0"
echo "=============================="
