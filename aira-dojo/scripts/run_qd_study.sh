#!/bin/bash
# QD Study: Compare AIRA_GREEDY vs AIRA_EVO on diverse MLE-bench tasks
# Uses gemini-3-flash-preview via gdm client (native Google genai SDK with retries)
#
# Usage:
#   ./scripts/run_qd_study.sh greedy                     # Run greedy solver (local, sequential)
#   ./scripts/run_qd_study.sh evo                        # Run EVO solver (local, sequential)
#   ./scripts/run_qd_study.sh greedy --dry               # Dry run (print resolved config)
#   ./scripts/run_qd_study.sh all                        # Run both solvers locally
#   ./scripts/run_qd_study.sh greedy --parallel          # Run 2 parallel on GPUs 6,7
#   ./scripts/run_qd_study.sh greedy --parallel 5,6,7    # Run 3 parallel on GPUs 5,6,7
#   ./scripts/run_qd_study.sh greedy --parallel 6@5,6,7  # Run 6 workers sharing GPUs 5,6,7
#
# For single-task runs:
#   ./scripts/run_qd_study.sh greedy --single tabular-playground-series-dec-2021

set -e

# Clear expired HuggingFace tokens — env vars (inherited from tmux) and token files
# Public models (timm, torchvision) work without auth; an expired token causes 401 errors
unset HUGGINGFACE_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_API_KEY 2>/dev/null || true
for _hf_dir in "${HF_HOME:-}" "${HOME}/.cache/huggingface" "${HOME}/.huggingface"; do
    [ -f "${_hf_dir}/token" ] && mv "${_hf_dir}/token" "${_hf_dir}/token.disabled" 2>/dev/null || true
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_BASE="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_BASE}"

SOLVER=${1:-greedy}
MODE=""
EXTRA_ARG=""
SINGLE_TASK=""

# Parse remaining args
shift 1 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry|--local)
      MODE="$1"; shift ;;
    --single)
      SINGLE_TASK="$2"; shift 2 ;;
    --parallel)
      MODE="--parallel"; EXTRA_ARG="${2:-6,7}"; shift 2 ;;
    *)
      shift ;;
  esac
done
MODE=${MODE:---local}

TASKS=(
  "tabular-playground-series-dec-2021"
  "spooky-author-identification"
  "dog-breed-identification"
  "learning-agency-lab-automated-essay-scoring-2"
  "stanford-covid-vaccine"
  # "google-research-identify-contrails-reduce-global-warming"  # Skipped: 35GB dataset, extraction too slow
)
SEEDS=(1 2 3)

# Common solver client overrides for gdm (native Google genai SDK)
CLIENT_OVERRIDES=(
  "solver/client@solver.operators.analyze.llm.client=gdm"
  "solver/client@solver.operators.debug.llm.client=gdm"
  "solver/client@solver.operators.draft.llm.client=gdm"
  "solver/client@solver.operators.improve.llm.client=gdm"
)

get_solver_overrides() {
  local solver_type=$1
  local overrides=()

  if [ "$solver_type" == "greedy" ]; then
    overrides=(
      "solver=mlebench/greedy"
      "solver.step_limit=30"
      "solver.execution_timeout=1800"
      "solver.use_test_score=true"
      "metadata.git_issue_id=QD_STUDY_greedy_gdm"
    )
  elif [ "$solver_type" == "evo" ]; then
    overrides=(
      "solver=mlebench/evo"
      "solver.num_generations=6"
      "solver.individuals_per_generation=5"
      "solver.num_islands=1"
      "solver.crossover_prob=0.5"
      "solver.num_generations_till_crossover=2"
      "solver.execution_timeout=1800"
      "solver.use_test_score=true"
      "metadata.git_issue_id=QD_STUDY_evo_gdm"
      "solver/client@solver.operators.crossover.llm.client=gdm"
    )
  fi

  echo "${overrides[@]}"
}

run_single() {
  local solver_type=$1
  local task=$2
  local seed=$3
  local gpu=${4:-""}
  local solver_overrides=($(get_solver_overrides "$solver_type"))

  echo "--- [${solver_type}] Task: ${task}, Seed: ${seed}${gpu:+ (GPU $gpu)} ---"

  local gpu_prefix=""
  if [ -n "$gpu" ]; then
    gpu_prefix="CUDA_VISIBLE_DEVICES=${gpu}"
  fi

  env ${gpu_prefix} python -m dojo.main_run \
    task=mlebench/_default \
    "task.name=${task}" \
    interpreter=python \
    "${CLIENT_OVERRIDES[@]}" \
    "${solver_overrides[@]}" \
    "metadata.seed=${seed}" \
    logger.use_wandb=False
}

run_dry() {
  local solver_type=$1
  local task="${TASKS[0]}"
  local solver_overrides=($(get_solver_overrides "$solver_type"))

  echo "=== Dry run for ${solver_type} (printing resolved config) ==="
  python -m dojo.main_run \
    task=mlebench/_default \
    "task.name=${task}" \
    interpreter=python \
    "${CLIENT_OVERRIDES[@]}" \
    "${solver_overrides[@]}" \
    metadata.seed=1 \
    logger.use_wandb=False \
    --cfg job
}

run_local() {
  local solver_type=$1
  local task_list=("${TASKS[@]}")

  # If single task mode, override task list
  if [ -n "$SINGLE_TASK" ]; then
    task_list=("$SINGLE_TASK")
  fi

  echo "=== Running ${solver_type} locally (${#task_list[@]} tasks × ${#SEEDS[@]} seeds) ==="
  for task in "${task_list[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_single "$solver_type" "$task" "$seed"
    done
  done
}

run_parallel() {
  local solver_type=$1
  local parallel_spec=${2:-"6,7"}
  local task_list=("${TASKS[@]}")

  if [ -n "$SINGLE_TASK" ]; then
    task_list=("$SINGLE_TASK")
  fi

  # Parse parallel spec: WORKERS@GPU_IDS or just GPU_IDS
  local gpu_str max_jobs
  if [[ "$parallel_spec" == *"@"* ]]; then
    max_jobs="${parallel_spec%%@*}"
    gpu_str="${parallel_spec#*@}"
  else
    gpu_str="$parallel_spec"
  fi
  IFS=',' read -ra GPUS <<< "$gpu_str"
  max_jobs=${max_jobs:-${#GPUS[@]}}

  # Build job list: (task, seed) pairs
  local jobs=()
  for task in "${task_list[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("${task}:${seed}")
    done
  done

  local total=${#jobs[@]}
  echo "=== Running ${solver_type} in parallel (${total} runs, ${max_jobs} workers on GPUs: ${gpu_str}) ==="
  echo "=== Execution timeout: 1800s (30 min) per step ==="
  echo "=== Log files: ${LOG_BASE}/qd_study_${solver_type}_*.log ==="

  local running=0
  local gpu_idx=0
  local pids=()
  local pid_info=()

  for job in "${jobs[@]}"; do
    local task="${job%%:*}"
    local seed="${job##*:}"
    local gpu="${GPUS[$gpu_idx]}"
    local log_file="${LOG_BASE}/qd_study_${solver_type}_${task}_seed${seed}.log"

    echo "[$(date '+%H:%M:%S')] Launching: ${task} seed=${seed} on GPU ${gpu} -> ${log_file}"
    run_single "$solver_type" "$task" "$seed" "$gpu" > "$log_file" 2>&1 &
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

  echo "=== All ${total} runs completed ==="
}

case "${SOLVER}" in
  greedy|evo)
    ;;
  all)
    for s in greedy evo; do
      $0 "$s" "$MODE" "$EXTRA_ARG"
    done
    exit 0
    ;;
  *)
    echo "Usage: $0 {greedy|evo|all} [--local|--dry|--single <task>|--parallel [gpu_ids]]"
    exit 1
    ;;
esac

case "${MODE}" in
  --dry)
    run_dry "$SOLVER"
    ;;
  --single)
    run_local "$SOLVER"
    ;;
  --parallel)
    run_parallel "$SOLVER" "${EXTRA_ARG:-6,7}"
    ;;
  --local|*)
    run_local "$SOLVER"
    ;;
esac
