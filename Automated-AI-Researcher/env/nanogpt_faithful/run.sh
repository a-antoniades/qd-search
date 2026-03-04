#!/bin/bash
wandb_name=faithful_run

export WANDB_NAME=$wandb_name
export WANDB_PROJECT=${WANDB_PROJECT:-nanogpt_ES_faithful}

# Detect available GPUs from CUDA_VISIBLE_DEVICES
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

echo "Detected $NGPUS GPU(s), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ "$NGPUS" -gt 1 ]; then
    conda run --live-stream -n aira-dojo torchrun \
        --standalone --nproc_per_node=$NGPUS train.py
else
    conda run --live-stream -n aira-dojo python -u train.py
fi

echo "run.sh has finished."
