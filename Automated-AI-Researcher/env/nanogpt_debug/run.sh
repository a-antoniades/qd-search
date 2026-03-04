#!/bin/bash
wandb_name=debug_1gpu

export WANDB_NAME=$wandb_name
export WANDB_PROJECT=nanogpt_ES_claude_debug

conda run --live-stream -n aira-dojo python -u \
    train.py \
    --batch_size 32

echo "run.sh has finished."
