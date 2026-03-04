#!/bin/bash
# Run EVO with RLM selector for QD study
# Usage: ./run_evo_rlm.sh <task_name> <gpu_id> <seed>

set -e

TASK=${1:-"tabular-playground-series-dec-2021"}
GPU=${2:-0}
SEED=${3:-1}

cd /share/edc/home/antonis/qd-search/aira-dojo
source /opt/conda/etc/profile.d/conda.sh
conda activate aira-dojo

export CUDA_VISIBLE_DEVICES=$GPU

# Clear expired HuggingFace tokens
unset HUGGINGFACE_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_API_KEY 2>/dev/null || true
for _hf_dir in "${HF_HOME:-}" "${HOME}/.cache/huggingface" "${HOME}/.huggingface"; do
    [ -f "${_hf_dir}/token" ] && mv "${_hf_dir}/token" "${_hf_dir}/token.disabled" 2>/dev/null || true
done

echo "Starting EVO+RLM run: task=$TASK gpu=$GPU seed=$SEED"
echo "Time: $(date)"

python -m dojo.main_run \
    solver=mlebench/evo \
    solver.num_generations=6 \
    solver.individuals_per_generation=5 \
    solver.num_islands=1 \
    solver.crossover_prob=0.5 \
    solver.num_generations_till_crossover=2 \
    solver.execution_timeout=1800 \
    solver.use_test_score=true \
    selector@solver.selector=rlm \
    solver/client@solver.operators.analyze.llm.client=gdm \
    solver/client@solver.operators.debug.llm.client=gdm \
    solver/client@solver.operators.draft.llm.client=gdm \
    solver/client@solver.operators.improve.llm.client=gdm \
    solver/client@solver.operators.crossover.llm.client=gdm \
    task=mlebench/_default \
    task.name=$TASK \
    metadata.seed=$SEED \
    metadata.git_issue_id=QD_STUDY_evo_rlm \
    logger.use_wandb=False \
    interpreter=python \
    2>&1 | tee logs/qd_study_evo_rlm_${TASK}_seed${SEED}.log

echo "Completed: $(date)"
