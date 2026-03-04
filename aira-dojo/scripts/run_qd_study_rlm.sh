#!/bin/bash
# QD Study with RLM Selector: EVO runs using iterative LLM-based node selection
# Tests the updated RLM pipeline (llm_query_batched, semantic approach comparison)
#
# Usage:
#   ./scripts/run_qd_study_rlm.sh                          # Run all 5 tasks on GPUs 4-7
#   ./scripts/run_qd_study_rlm.sh --dry                    # Dry run (print config)
#   ./scripts/run_qd_study_rlm.sh --single tabular-playground-series-dec-2021
#   ./scripts/run_qd_study_rlm.sh --parallel 4,5,6,7       # Specify GPUs

set -e

# Clear expired HuggingFace tokens
unset HUGGINGFACE_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_API_KEY 2>/dev/null || true
for _hf_dir in "${HF_HOME:-}" "${HOME}/.cache/huggingface" "${HOME}/.huggingface"; do
    [ -f "${_hf_dir}/token" ] && mv "${_hf_dir}/token" "${_hf_dir}/token.disabled" 2>/dev/null || true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_BASE}"

MODE=""
EXTRA_ARG=""
SINGLE_TASK=""
GPU_IDS="4,5,6,7"  # Default to free GPUs

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry)
      MODE="--dry"; shift ;;
    --single)
      SINGLE_TASK="$2"; shift 2 ;;
    --parallel)
      GPU_IDS="${2:-4,5,6,7}"; shift 2 ;;
    *)
      shift ;;
  esac
done
MODE=${MODE:-"--parallel"}

# All 5 QD study tasks
TASKS=(
  "tabular-playground-series-dec-2021"
  "spooky-author-identification"
  "dog-breed-identification"
  "learning-agency-lab-automated-essay-scoring-2"
  "stanford-covid-vaccine"
)

# Only seed 1 for initial comparison
SEEDS=(1)

# Common client overrides for gdm (Google genai SDK)
CLIENT_OVERRIDES=(
  "solver/client@solver.operators.analyze.llm.client=gdm"
  "solver/client@solver.operators.debug.llm.client=gdm"
  "solver/client@solver.operators.draft.llm.client=gdm"
  "solver/client@solver.operators.improve.llm.client=gdm"
  "solver/client@solver.operators.crossover.llm.client=gdm"
)

# EVO solver with RLM selector (direct override of selector config)
SOLVER_OVERRIDES=(
  "solver=mlebench/evo"
  # Override selector to use RLM instead of fitness
  "solver.selector._target_=dojo.config_dataclasses.selector.rlm.RLMSelectorConfig"
  "solver.selector.selector_type=rlm"
  "+solver.selector.backend=gemini"
  "+solver.selector.model_name=gemini-2.0-flash"
  "+solver.selector.api_key_env=GEMINI_API_KEY"
  "+solver.selector.max_iterations=6"
  "+solver.selector.timeout=120"
  "+solver.selector.max_nodes=50"
  "+solver.selector.full_context=true"
  "+solver.selector.fallback_to_fitness=true"
  "solver.selector.verbose=true"
  # EVO parameters
  "solver.num_generations=6"
  "solver.individuals_per_generation=5"
  "solver.num_islands=1"
  "solver.crossover_prob=0.5"
  "solver.num_generations_till_crossover=2"
  "solver.execution_timeout=1800"
  "solver.use_test_score=true"
  "metadata.git_issue_id=QD_STUDY_evo_rlm"
)

run_single() {
  local task=$1
  local seed=$2
  local gpu=${3:-""}

  echo "--- [EVO+RLM] Task: ${task}, Seed: ${seed}${gpu:+ (GPU $gpu)} ---"

  local gpu_prefix=""
  if [ -n "$gpu" ]; then
    gpu_prefix="CUDA_VISIBLE_DEVICES=${gpu}"
  fi

  env ${gpu_prefix} python -m dojo.main_run \
    task=mlebench/_default \
    "task.name=${task}" \
    interpreter=python \
    "${CLIENT_OVERRIDES[@]}" \
    "${SOLVER_OVERRIDES[@]}" \
    "metadata.seed=${seed}" \
    logger.use_wandb=False
}

run_dry() {
  local task="${SINGLE_TASK:-${TASKS[0]}}"

  echo "=== Dry run for EVO+RLM (printing resolved config) ==="
  python -m dojo.main_run \
    task=mlebench/_default \
    "task.name=${task}" \
    interpreter=python \
    "${CLIENT_OVERRIDES[@]}" \
    "${SOLVER_OVERRIDES[@]}" \
    metadata.seed=1 \
    logger.use_wandb=False \
    --cfg job
}

run_parallel() {
  local gpu_str=${1:-$GPU_IDS}
  local task_list=("${TASKS[@]}")

  if [ -n "$SINGLE_TASK" ]; then
    task_list=("$SINGLE_TASK")
  fi

  IFS=',' read -ra GPUS <<< "$gpu_str"
  local max_jobs=${#GPUS[@]}

  # Build job list: (task, seed) pairs
  local jobs=()
  for task in "${task_list[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("${task}:${seed}")
    done
  done

  local total=${#jobs[@]}
  echo "=== Running EVO+RLM in parallel (${total} runs, ${max_jobs} workers on GPUs: ${gpu_str}) ==="
  echo "=== RLM Selector: iterative pipeline with llm_query_batched() ==="
  echo "=== Execution timeout: 1800s (30 min) per step ==="
  echo "=== Log files: ${LOG_BASE}/qd_study_evo_rlm_*.log ==="

  local running=0
  local gpu_idx=0
  local pids=()
  local pid_info=()

  for job in "${jobs[@]}"; do
    local task="${job%%:*}"
    local seed="${job##*:}"
    local gpu="${GPUS[$gpu_idx]}"
    local log_file="${LOG_BASE}/qd_study_evo_rlm_${task}_seed${seed}.log"

    echo "[$(date '+%H:%M:%S')] Launching: ${task} seed=${seed} on GPU ${gpu} -> ${log_file}"
    run_single "$task" "$seed" "$gpu" > "$log_file" 2>&1 &
    local pid=$!
    pids+=($pid)
    pid_info[$pid]="${task}/seed${seed}/GPU${gpu}"

    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    running=$((running + 1))

    # If we've filled all GPU slots, wait for one to finish
    if [ "$running" -ge "$max_jobs" ]; then
      wait -n "${pids[@]}" 2>/dev/null || true
      # Clean up finished PIDs
      local new_pids=()
      for p in "${pids[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
          new_pids+=($p)
        else
          wait "$p" 2>/dev/null
          local exit_code=$?
          echo "[$(date '+%H:%M:%S')] Finished: ${pid_info[$p]} (exit code: ${exit_code})"
          running=$((running - 1))
        fi
      done
      pids=("${new_pids[@]}")
    fi
  done

  # Wait for remaining jobs
  echo "[$(date '+%H:%M:%S')] All jobs launched. Waiting for ${#pids[@]} remaining..."
  for p in "${pids[@]}"; do
    wait "$p" 2>/dev/null
    local exit_code=$?
    echo "[$(date '+%H:%M:%S')] Finished: ${pid_info[$p]} (exit code: ${exit_code})"
  done

  echo "=== All ${total} EVO+RLM runs completed ==="
}

case "${MODE}" in
  --dry)
    run_dry
    ;;
  --parallel|*)
    run_parallel "${GPU_IDS}"
    ;;
esac
