#!/bin/bash
# Fast debug pipeline: 1 epoch, 10 ideas (1 LLM batch), 1 GPU, 2 min timeout
# Runs the entire LLM-driven evolutionary search end-to-end in ~5-10 minutes.
#
# Usage: cd Automated-AI-Researcher && bash scripts/run_debug.sh [GPU_ID]
#   GPU_ID defaults to 7
#
# Prerequisites:
#   1. bash scripts/prepare_debug_data.sh   (generates ~40MB synthetic data)
#   2. export GEMINI_API_KEY=...            (or GOOGLE_API_KEY)

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
    echo "Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set."
    echo "LLM calls will fail unless VertexAI is configured."
fi

GPU_ID=${1:-7}

echo "=== Debug Pipeline ==="
echo "  GPU: $GPU_ID"
echo "  Model: gemini-3-flash-preview"
echo "  Ideas: 3"
echo "  Timeout: 120s per job"
echo "  Env: env/nanogpt_debug"
echo ""

python -m agent.full_pipeline \
    --epochs 1 \
    --num_ideas_per_epoch 10 \
    --run_name "debug_nanogpt" \
    --env_dir env/nanogpt_debug \
    --model_name gemini-3-flash-preview \
    --entity woanderers \
    --project nanogpt_ES_claude_debug \
    --local \
    --gpu_ids "$GPU_ID" \
    --timeout 120 \
    --total_workers 3 \
    --wandb_sync_wait 10 \
    --wandb_min_completion 0.0
