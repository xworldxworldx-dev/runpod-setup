#!/bin/bash
# MiniMax H3 - NSFW LoRA 다운로드
# 사용법: Pod 부팅 후 Jupyter 터미널에서
#   bash <(curl -s https://raw.githubusercontent.com/xworldxworldx-dev/runpod-setup/main/h3_loras.sh)
#
# 전제: HF_TOKEN 환경변수 설정됨 (Pod 배포 시 입력)
#       없으면 아래 줄 주석 해제하고 토큰 직접 입력
# export HF_TOKEN="hf_여기에토큰"

set -e

LORA_DIR="/workspace/ComfyUI/models/loras"
mkdir -p "$LORA_DIR"

echo "=============================="
echo " H3 NSFW LoRA 다운로드"
echo "=============================="

if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN 없음 — rate limit(429) 발생 가능"
fi

pip install -U huggingface_hub hf_transfer -q

python3 << 'PYEOF'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

TOKEN = os.environ.get("HF_TOKEN")
REPO = "Hearmeman/minimax-h3-loras"
DEST = "/workspace/ComfyUI/models/loras"

# ── 메인: 올인원 NSFW LoRA (2026-08-23 검증, 이게 핵심) ──
files = [
    "HMNSFW_AIO_V2.safetensors",
]

# ── 부위별 전문 LoRA (필요시 주석 해제) ──
# AIO로 부족한 부위가 있을 때만 추가. 1~2개까지만 (많이 쌓으면 간섭)
# files += [
#     "HMBreasts_085e0750_e40.safetensors",
#     "hmpussy_v6_epoch30.safetensors",
#     "HMPenis_v2_e35.safetensors",
#     "HMInnie_v1_e50.safetensors",
#     "HMCumshot_V2.safetensors",
#     "vagassist_e40.safetensors",
# ]

for f in files:
    print(f"  → {f}")
    try:
        hf_hub_download(repo_id=REPO, filename=f, local_dir=DEST, token=TOKEN)
        print("    ✓")
    except Exception as e:
        print(f"    ✗ {str(e)[:150]}")

print("완료")
PYEOF

echo ""
echo "=============================="
echo " 다운로드 완료"
echo ""
echo " [ComfyUI에서 설정]"
echo " 1. 새로고침 (Cmd+Shift+R)"
echo " 2. Power Lora Loader 노드 →"
echo "    빈 슬롯 드롭다운에서 HMNSFW_AIO_V2 선택"
echo " 3. 왼쪽 토글 켜기"
echo " 4. Strength 1.0 시작 (과하면 0.7~0.8)"
echo "=============================="
ls -lh "$LORA_DIR" | grep -i hm || true
