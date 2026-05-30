#!/bin/bash
export LD_LIBRARY_PATH=/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/curand/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cufft/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cublas/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/nccl/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/jeongwoo/adas-env/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/lib/wsl/lib

cd '/mnt/c/Users/Daniel Park/Desktop/SoundPJ-rebuilt'
echo "=== ADAS Finetune Start: $(date) ===" | tee -a logs/finetune.log
/home/jeongwoo/adas-env/bin/python3 -m src.finetune --config config.yaml 2>&1 | tee -a logs/finetune.log
echo "=== ADAS Finetune End: $(date) ===" | tee -a logs/finetune.log