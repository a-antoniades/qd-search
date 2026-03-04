import shutil
import os
import time
import json
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import * 
from agent.compute_idea_stats import compute_idea_stats
from agent.upload_repo_variants import zip_and_upload_repo_variants
from agent.retrieve_training_logs import retrieve_training_logs
from agent.evolutionary_search import update_database

def collect_local_training_logs(repo_variants_dir, run_dir, epoch_num, target_loss=3.28):
    """Collect training logs locally instead of downloading from wandb.

    For local execution, output.log files are already on disk in the
    repo_variants directory. This copies them to training_logs/ and
    extracts metrics, producing the same ranked_ideas.json as the
    wandb-based retrieval.
    """
    from agent.retrieve_training_logs import extract_metrics_nanogpt

    logs_dir = os.path.join(run_dir, "training_logs", f"epoch{epoch_num}") + "/"
    os.makedirs(logs_dir, exist_ok=True)

    idea_final_rewards = {}
    num_collected = 0

    idea_dirs = sorted(
        [d for d in os.listdir(repo_variants_dir)
         if os.path.isdir(os.path.join(repo_variants_dir, d)) and d.startswith("idea_")],
        key=lambda d: int(d.split("_")[1]) if d.split("_")[1].isdigit() else float("inf"),
    )

    for idea_dir in idea_dirs:
        src_log = os.path.join(repo_variants_dir, idea_dir, "output.log")
        if not os.path.exists(src_log):
            print(f"  [{idea_dir}] No output.log found, skipping")
            continue

        dst_idea_dir = os.path.join(logs_dir, idea_dir)
        os.makedirs(dst_idea_dir, exist_ok=True)
        dst_log = os.path.join(dst_idea_dir, "output.log")
        shutil.copy2(src_log, dst_log)

        # Extract metrics
        with open(dst_log, "r") as f:
            lines = f.readlines()
        eval_rewards, time_to_target = extract_metrics_nanogpt(lines, target_loss)

        # Also extract final train loss as fallback metric
        final_train_loss = None
        for line in reversed(lines):
            if "train loss" in line:
                import re
                m = re.search(r"train loss\s+([\d.]+)", line)
                if m:
                    final_train_loss = float(m.group(1))
                    break

        with open(os.path.join(dst_idea_dir, "metrics.json"), "w") as f:
            json.dump({
                "eval_rewards": eval_rewards,
                "time_to_target": time_to_target,
                "final_train_loss": final_train_loss,
            }, f)

        if eval_rewards:
            final_val_loss = min(r["val_loss"] for r in eval_rewards)
        elif final_train_loss is not None:
            final_val_loss = final_train_loss
        else:
            final_val_loss = -999.0

        idea_final_rewards[idea_dir] = final_val_loss
        num_collected += 1
        print(f"  [{idea_dir}] Collected: {len(eval_rewards)} eval points, best val_loss={final_val_loss:.4f}" if eval_rewards else f"  [{idea_dir}] Collected (no eval points)")

    # Create ranked_ideas.json
    if idea_final_rewards:
        ranked_ideas = sorted(
            idea_final_rewards.items(),
            key=lambda x: float("inf") if x[1] <= 0 else x[1]
        )
        ranked_ideas_dicts = [{idea: reward} for idea, reward in ranked_ideas]
        with open(os.path.join(logs_dir, "ranked_ideas.json"), "w") as f:
            json.dump(ranked_ideas_dicts, f, indent=4)
    else:
        ranked_ideas_dicts = []

    return num_collected, ranked_ideas_dicts


def move_diffs_and_repo_variants(src_diffs, dst_diffs, src_repo, dst_repo):
    # Move diffs directory, overwriting if destination exists
    if os.path.exists(src_diffs):
        dst_diffs_path = os.path.join(dst_diffs, os.path.basename(src_diffs))
        if os.path.exists(dst_diffs_path):
            if os.path.isdir(dst_diffs_path):
                shutil.rmtree(dst_diffs_path)
            else:
                os.remove(dst_diffs_path)
        shutil.move(src_diffs, dst_diffs)
        print(f"Moved {src_diffs} to {dst_diffs}")
    else:
        print(f"Source diffs directory {src_diffs} does not exist.")

    # Move repo_variants directory, overwriting if destination exists
    if os.path.exists(src_repo):
        dst_repo_path = os.path.join(dst_repo, os.path.basename(src_repo))
        if os.path.exists(dst_repo_path):
            if os.path.isdir(dst_repo_path):
                shutil.rmtree(dst_repo_path)
            else:
                os.remove(dst_repo_path)
        shutil.move(src_repo, dst_repo)
        print(f"Moved {src_repo} to {dst_repo}")
    else:
        print(f"Source repo_variants directory {src_repo} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_ideas_per_epoch", type=int, default=80)
    parser.add_argument("--continue_from_epoch", type=int, default=0)
    parser.add_argument("--skip_log_retrieval_when_continue", action="store_true")
    parser.add_argument("--skip_idea_generation_when_continue", action="store_true")
    parser.add_argument("--run_name", type=str, default="nanogpt_claude_opus_bsz80")
    parser.add_argument("--env_dir", type=str, default="env/nanogpt")
    parser.add_argument("--entity", type=str, default="hashimoto-group")
    parser.add_argument("--project", type=str, default="nanogpt_ES_claude")
    parser.add_argument("--model_name", type=str, default="claude-opus-4-5")
    parser.add_argument("--local", action="store_true", help="Execute training locally instead of uploading to HuggingFace")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs for local execution (default: auto-detect free)")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-job timeout in seconds for local execution (default: 3600)")
    parser.add_argument("--total_workers", type=int, default=10, help="Parallel workers for code diff generation (default: 10)")
    parser.add_argument("--wandb_sync_wait", type=int, default=60, help="Seconds to wait after local training for wandb sync (default: 60)")
    parser.add_argument("--wandb_min_completion", type=float, default=0.3, help="Fraction of runs required before proceeding with log retrieval (default: 0.3)")
    parser.add_argument("--gpus_per_job", type=int, default=1, help="Number of GPUs per training job for DDP (default: 1)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Base output directory for run artifacts (default: runs)")
    args = parser.parse_args()

    epochs = args.epochs
    num_ideas_per_epoch = args.num_ideas_per_epoch
    run_name = args.run_name
    run_dir = os.path.join(args.output_dir, run_name)
    if args.continue_from_epoch < 0:
        start_epoch = 0
    else:
        start_epoch = args.continue_from_epoch
        
    for epoch in range(start_epoch, epochs):
        if epoch >= args.continue_from_epoch and not args.skip_idea_generation_when_continue:
            # generate ideas 
            print ("Sampling ideas for epoch ", epoch)
            agent_call_idea(num_ideas = num_ideas_per_epoch, cache_file = f"{run_dir}/ideas/ideas_epoch{epoch}.json", run_name = run_name, epoch_num = epoch, prev_ideas_file = f"{run_dir}/ideas/ideas_epoch{epoch-1}.json", prev_training_logs = f"{run_dir}/training_logs/epoch{epoch-1}/", top_k=100, sample_k=100, env_dir=args.env_dir, model_name=args.model_name)

            # generate the code diff for each experiment
            print ("Generating code diffs for epoch ", epoch)
            generate_code_diff_parallel(max_trials=10, diffs_dir=f"{run_dir}/diffs/epoch{epoch}", repo_dir=f"{run_dir}/repo_variants/epoch{epoch}", env_dir=args.env_dir, idea_file=f"{run_dir}/ideas/ideas_epoch{epoch}.json", model_name=args.model_name, total_workers=args.total_workers)

            print ("Computing idea stats for epoch ", epoch)
            compute_idea_stats(idea_file = f"{run_dir}/ideas/ideas_epoch{epoch}.json", repo_variants_dir = f"{run_dir}/repo_variants/epoch{epoch}", idea_stats_file = f"{run_dir}/idea_stats/epoch{epoch}.json")

            if args.local:
                # Execute training locally instead of uploading + sleeping
                from agent.local_executor import execute_training_jobs
                gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
                print(f"Executing training jobs locally for epoch {epoch}")
                results = execute_training_jobs(
                    repo_variants_dir=f"{run_dir}/repo_variants/epoch{epoch}",
                    run_name=run_name,
                    epoch_num=epoch,
                    gpu_ids=gpu_ids,
                    wandb_project=args.project,
                    timeout_seconds=args.timeout,
                    gpus_per_job=args.gpus_per_job,
                )

                # Collect training logs locally (output.log files are already on disk)
                print(f"Collecting local training logs for epoch {epoch}")
                num_logs, ranked_ideas_dicts = collect_local_training_logs(
                    repo_variants_dir=f"{run_dir}/repo_variants/epoch{epoch}",
                    run_dir=run_dir,
                    epoch_num=epoch,
                )
                print(f"Collected {num_logs} local training logs")
            else:
                zip_and_upload_repo_variants(original_ideas = f"{run_dir}/repo_variants/epoch{epoch}", folder_path = f"/juice5b/scr5b/nlp/aihinton/repo_variants/{run_name}/epoch{epoch}", run_name = run_name, epoch_num = epoch)
                print ("Waiting before retrieving training logs")
                time.sleep(90 * 60)

            print ("Moving diffs and repo_variants of this epoch to archive")
            move_diffs_and_repo_variants(src_diffs=f"{run_dir}/diffs/epoch{epoch}", dst_diffs=f"{run_dir}/archive/diffs", src_repo=f"{run_dir}/repo_variants/epoch{epoch}", dst_repo=f"{run_dir}/archive/repo_variants")

        if epoch == args.continue_from_epoch and args.skip_log_retrieval_when_continue:
            print ("Skipping log retrieval for epoch ", epoch)
            continue

        if not args.local:
            # Remote mode: retrieve logs from wandb with polling
            print ("Retrieving training logs for epoch ", epoch)
            num_logs_retrieved = 0
            idea_stats_path = f"{run_dir}/idea_stats/epoch{epoch}.json"
            with open(idea_stats_path, "r") as f:
                idea_stats = json.load(f)
            num_ideas_submitted = idea_stats.get("success_count", num_ideas_per_epoch)

            last_num_logs_retrieved = 0
            min_logs_needed = int(num_ideas_submitted * args.wandb_min_completion)
            while num_logs_retrieved <= min_logs_needed:
                num_logs_retrieved, ranked_ideas_dicts = retrieve_training_logs(run_name = run_name, epoch_num = epoch, env_dir = args.env_dir, entity = args.entity, project = args.project, run_dir=run_dir)
                print (f"Number of logs retrieved: {num_logs_retrieved} out of {num_ideas_submitted} submitted ideas (need > {min_logs_needed})")
                ## terminate conditions: most runs are finished
                ## this means we might waste some runs that are still running but it's going to be faster
                if num_logs_retrieved > min_logs_needed:
                    break
                last_num_logs_retrieved = num_logs_retrieved
                print ("Waiting before retrieving training logs again")
                time.sleep(20 * 60)
    
    ## do a final round of update_dataset 
    update_database(run_name = run_name, epoch_num = epochs - 1, output_dir=args.output_dir)
