#!/bin/bash
# Multi-epoch evolutionary NanoGPT experiment on dipsy (8 GPUs free)
# Runs LLM-driven evolutionary search: generate ideas → train → evolve → repeat
#
# Usage: bash scripts/run_evo_dipsy.sh [GPU_IDS]
#   GPU_IDS: comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)
#
# Expected runtime: ~60 min (5 epochs × 20 ideas × 300s training on 8 GPUs)

set -e

cd "$(dirname "$0")/.."

# Check data exists
if [ ! -e "env/nanogpt_debug/fineweb10B" ]; then
    echo "Error: Debug data not found."
    echo "Run first: bash scripts/prepare_debug_data.sh"
    exit 1
fi

# Check API key
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set."
    exit 1
fi

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
RUN_NAME="nanogpt_evo_5epoch_dipsy"
EPOCHS=5
IDEAS_PER_EPOCH=20
TIMEOUT=300
MODEL="gemini-3-flash-preview"
PROJECT="nanogpt_ES_claude_evo"

echo "=== NanoGPT Evolutionary Search Experiment (dipsy) ==="
echo "  GPUs: $GPU_IDS"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Ideas/epoch: $IDEAS_PER_EPOCH"
echo "  Timeout: ${TIMEOUT}s per job"
echo "  Run name: $RUN_NAME"
echo "  Project: $PROJECT"
echo "  batch_size: 32 (full 40GB GPUs)"
echo ""
echo "Starting at $(date)"
echo ""

python -u -m agent.full_pipeline \
    --epochs $EPOCHS \
    --num_ideas_per_epoch $IDEAS_PER_EPOCH \
    --run_name "$RUN_NAME" \
    --env_dir env/nanogpt_debug \
    --model_name "$MODEL" \
    --entity woanderers \
    --project "$PROJECT" \
    --local \
    --gpu_ids "$GPU_IDS" \
    --timeout $TIMEOUT \
    --total_workers 8 \
    --wandb_sync_wait 5 \
    --wandb_min_completion 0.0

echo ""
echo "Experiment completed at $(date)"
