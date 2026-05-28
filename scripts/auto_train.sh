#!/bin/bash
# 자율 학습 스크립트 — Phase 2~4 자동 실행
# Phase 1 (finetune) 완료 후 이 스크립트 실행

set -e

export LD_LIBRARY_PATH=/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/curand/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cufft/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cublas/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/nccl/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib

source /home/jeongwoo/adas-env/bin/activate

PROJECT="/mnt/c/Users/Daniel Park/Downloads/SoundPJ-rebuilt"
cd "$PROJECT"

echo "========================================"
echo "Phase 2: Embedding 재추출 (finetuned YAMNet)"
echo "========================================"
python3 -m src.embedding --config config.yaml

echo "========================================"
echo "Phase 3: Classifier 재학습"
echo "========================================"
python3 -m src.classifier --config config.yaml

echo "========================================"
echo "Phase 4: Finetune Round 2 (60% unfreeze)"
echo "========================================"
python3 -m src.finetune --config config.yaml

echo "========================================"
echo "ALL DONE"
echo "========================================"
