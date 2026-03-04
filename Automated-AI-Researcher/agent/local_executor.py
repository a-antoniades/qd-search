"""Local GPU executor for training jobs.

Manages a pool of GPUs and runs training jobs (e.g., GRPO) concurrently
using ThreadPoolExecutor. Replaces the upload-to-HuggingFace + sleep workflow
with direct local execution.
"""

import argparse
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class JobResult:
    idea_dir: str
    gpu_ids: list[int]
    exit_code: int
    duration_seconds: float
    log_path: str
    timed_out: bool = False
    error: str = ""


class GPUPool:
    """Thread-safe pool of GPU IDs, supporting multi-GPU jobs."""

    def __init__(self, gpu_ids: list[int], gpus_per_job: int = 1):
        if not gpu_ids:
            raise ValueError("gpu_ids must be a non-empty list")
        if len(gpu_ids) < gpus_per_job:
            raise ValueError(f"Need at least {gpus_per_job} GPUs, got {len(gpu_ids)}")
        self._lock = threading.Lock()
        self._gpus_per_job = gpus_per_job
        # Pre-chunk GPUs into groups
        self._groups = [
            gpu_ids[i:i + gpus_per_job]
            for i in range(0, len(gpu_ids) - len(gpu_ids) % gpus_per_job, gpus_per_job)
        ]
        self._available = threading.Semaphore(len(self._groups))
        self._free_groups = list(self._groups)

    @property
    def num_slots(self) -> int:
        """Number of concurrent job slots."""
        return len(self._groups)

    def acquire(self) -> list[int]:
        """Block until a GPU group is available, then return list of GPU IDs."""
        self._available.acquire()
        with self._lock:
            return self._free_groups.pop(0)

    def release(self, gpu_ids: list[int]) -> None:
        """Return a GPU group to the pool."""
        with self._lock:
            self._free_groups.append(gpu_ids)
        self._available.release()


def _detect_free_gpus() -> list[int]:
    """Detect GPUs with no running processes via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

    free = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        gpu_idx = int(parts[0].strip())
        mem_used = int(parts[1].strip())
        # Consider GPU "free" if less than 500 MB used
        if mem_used < 500:
            free.append(gpu_idx)
    return free


def _patch_wandb_name(run_sh_path: str, wandb_name: str) -> None:
    """Replace the wandb_name=... line in run.sh with the desired name."""
    with open(run_sh_path, "r") as f:
        content = f.read()
    content = re.sub(r"^wandb_name=.*$", f"wandb_name={wandb_name}", content, count=1, flags=re.MULTILINE)
    with open(run_sh_path, "w") as f:
        f.write(content)


def _run_single_job(
    idea_path: str,
    gpu_ids: list[int],
    wandb_name: str,
    timeout_seconds: int,
    wandb_project: str | None = None,
) -> JobResult:
    """Execute a single training job in an idea directory."""
    run_sh = os.path.join(idea_path, "run.sh")
    if not os.path.exists(run_sh):
        return JobResult(
            idea_dir=os.path.basename(idea_path),
            gpu_ids=gpu_ids,
            exit_code=-1,
            duration_seconds=0.0,
            log_path="",
            error=f"run.sh not found in {idea_path}",
        )

    # Patch wandb name
    _patch_wandb_name(run_sh, wandb_name)

    log_path = os.path.join(idea_path, "output.log")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["VLLM_USE_V1"] = "0"
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project

    start = time.monotonic()
    timed_out = False
    error = ""

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", "run.sh"],
            cwd=idea_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = -9
            timed_out = True
            error = f"Timed out after {timeout_seconds}s"

    duration = time.monotonic() - start

    return JobResult(
        idea_dir=os.path.basename(idea_path),
        gpu_ids=gpu_ids,
        exit_code=exit_code,
        duration_seconds=duration,
        log_path=log_path,
        timed_out=timed_out,
        error=error,
    )


def execute_training_jobs(
    repo_variants_dir: str,
    run_name: str,
    epoch_num: int,
    gpu_ids: list[int] | None = None,
    wandb_project: str | None = None,
    timeout_seconds: int = 3600,
    gpus_per_job: int = 1,
) -> list[JobResult]:
    """Execute training jobs for all idea_* dirs using a GPU pool.

    Args:
        repo_variants_dir: Directory containing idea_0/, idea_1/, etc.
        run_name: Experiment run name for Wandb naming.
        epoch_num: Current epoch number.
        gpu_ids: Explicit GPU IDs to use. Auto-detects free GPUs if None.
        wandb_project: Wandb project name. If None, uses WANDB_PROJECT env var.
        timeout_seconds: Max seconds per job before killing it.
        gpus_per_job: Number of GPUs to allocate per training job (for DDP).

    Returns:
        List of JobResult for each idea directory.
    """
    if gpu_ids is None:
        gpu_ids = _detect_free_gpus()
        if not gpu_ids:
            raise RuntimeError("No free GPUs detected. Specify --gpu_ids explicitly.")
        print(f"Auto-detected free GPUs: {gpu_ids}")

    # Find all idea_* directories, sorted numerically
    idea_dirs = sorted(
        [
            d for d in os.listdir(repo_variants_dir)
            if os.path.isdir(os.path.join(repo_variants_dir, d)) and d.startswith("idea_")
        ],
        key=lambda d: int(d.split("_")[1]) if d.split("_")[1].isdigit() else float("inf"),
    )

    if not idea_dirs:
        print(f"No idea_* directories found in {repo_variants_dir}")
        return []

    gpu_pool = GPUPool(gpu_ids, gpus_per_job=gpus_per_job)
    print(f"Found {len(idea_dirs)} idea directories, using {len(gpu_ids)} GPUs: {gpu_ids} "
          f"({gpus_per_job} per job, {gpu_pool.num_slots} parallel slots)")

    results: list[JobResult] = []

    def _worker(idea_dir_name: str) -> JobResult:
        idea_path = os.path.join(repo_variants_dir, idea_dir_name)
        idea_number = idea_dir_name.split("_")[1]

        # Wandb name pattern: {run_name}_epoch{N}_b200_idea_{N}
        # Must match retrieve_training_logs.py:get_run_name()
        wandb_name = f"{run_name}_epoch{epoch_num}_b200_idea_{idea_number}"

        job_gpu_ids = gpu_pool.acquire()
        print(f"  [{idea_dir_name}] Starting on GPUs {job_gpu_ids} (wandb: {wandb_name})")
        try:
            result = _run_single_job(
                idea_path=idea_path,
                gpu_ids=job_gpu_ids,
                wandb_name=wandb_name,
                timeout_seconds=timeout_seconds,
                wandb_project=wandb_project,
            )
        finally:
            gpu_pool.release(job_gpu_ids)

        status = "OK" if result.exit_code == 0 else f"FAIL(exit={result.exit_code})"
        if result.timed_out:
            status = "TIMEOUT"
        print(f"  [{idea_dir_name}] Finished on GPUs {job_gpu_ids}: {status} ({result.duration_seconds:.0f}s)")
        return result

    with ThreadPoolExecutor(max_workers=gpu_pool.num_slots) as executor:
        futures = {executor.submit(_worker, d): d for d in idea_dirs}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results by idea number for consistent output
    results.sort(key=lambda r: int(r.idea_dir.split("_")[1]) if r.idea_dir.split("_")[1].isdigit() else float("inf"))

    # Summary
    succeeded = sum(1 for r in results if r.exit_code == 0)
    failed = sum(1 for r in results if r.exit_code != 0 and not r.timed_out)
    timed_out = sum(1 for r in results if r.timed_out)
    print(f"\nExecution summary: {succeeded} succeeded, {failed} failed, {timed_out} timed out (of {len(results)} total)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training jobs locally on GPU pool")
    parser.add_argument("--repo_variants_dir", type=str, required=True, help="Directory containing idea_* subdirs")
    parser.add_argument("--run_name", type=str, required=True, help="Experiment run name")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs (default: auto-detect free)")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per job in seconds (default: 3600)")
    parser.add_argument("--gpus_per_job", type=int, default=1, help="Number of GPUs per training job (default: 1)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None

    results = execute_training_jobs(
        repo_variants_dir=args.repo_variants_dir,
        run_name=args.run_name,
        epoch_num=args.epoch,
        gpu_ids=gpu_ids,
        wandb_project=args.wandb_project,
        timeout_seconds=args.timeout,
        gpus_per_job=args.gpus_per_job,
    )

    # Print detailed results
    print("\nDetailed results:")
    for r in results:
        status = "OK" if r.exit_code == 0 else f"FAIL(exit={r.exit_code})"
        if r.timed_out:
            status = "TIMEOUT"
        print(f"  {r.idea_dir}: {status} gpus={r.gpu_ids} duration={r.duration_seconds:.0f}s log={r.log_path}")
        if r.error:
            print(f"    error: {r.error}")
