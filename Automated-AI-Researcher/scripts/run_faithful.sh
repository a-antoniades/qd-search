#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Check data exists
if [ ! -e "env/nanogpt_faithful/fineweb10B" ]; then
    echo "Error: FineWeb data not found."
    echo "Run: bash scripts/prepare_fineweb_data.sh"
    exit 1
fi

if ! ls env/nanogpt_faithful/fineweb10B/fineweb_train_*.bin 1>/dev/null 2>&1; then
    echo "Error: No train shards found in env/nanogpt_faithful/fineweb10B/"
    echo "Run: bash scripts/prepare_fineweb_data.sh"
    exit 1
fi

# Check API key
if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: No API key set. Export GEMINI_API_KEY or GOOGLE_API_KEY."
    exit 1
fi

GPU_IDS=${1:-0,1,2,3,4,5,6,7}
RUN_NAME="nanogpt_faithful_4gpu"
EPOCHS=8
IDEAS_PER_EPOCH=20
TIMEOUT=2400        # 40 min (1500s training + compile + val + overhead)
GPUS_PER_JOB=4
MODEL="gemini-3-flash-preview"
PROJECT="nanogpt_ES_faithful"

echo "=== NanoGPT Faithful Replication ==="
echo "  GPUs: $GPU_IDS ($GPUS_PER_JOB per idea)"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS, Ideas/epoch: $IDEAS_PER_EPOCH"
echo "  Timeout: ${TIMEOUT}s"
echo "  Run name: $RUN_NAME"
echo ""

python -u -m agent.full_pipeline \
    --epochs $EPOCHS \
    --num_ideas_per_epoch $IDEAS_PER_EPOCH \
    --run_name "$RUN_NAME" \
    --env_dir env/nanogpt_faithful \
    --model_name "$MODEL" \
    --entity woanderers \
    --project "$PROJECT" \
    --local \
    --gpu_ids "$GPU_IDS" \
    --gpus_per_job $GPUS_PER_JOB \
    --timeout $TIMEOUT \
    --total_workers 8 \
    --wandb_sync_wait 5 \
    --wandb_min_completion 0.0
