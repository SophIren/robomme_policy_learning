"""Decode a few fixed train and val samples for saved checkpoints of a training run and save GT-vs-PRED plan images
(same pictures as scripts/preview_plan_predictions.py) plus summary metrics. Not started by the training script;
run it by hand after training (--once), or in parallel only on a GPU the training does not use:
    CUDA_VISIBLE_DEVICES=0 python scripts/watch_plan_previews.py \
        --output_dir runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan_lora --run_dir <run> --once

Results for the run dir <OUTPUT_DIR>/v<k>-<time>/:
    preview/checkpoint-<N>/train/*_vis.png, preview/checkpoint-<N>/val/*_vis.png, .../{train,val}.jsonl
    preview/metrics.jsonl     one line per (checkpoint, split): text_exact, same_len, mean_L1px, first_point_L1px
    runs/ (tensorboard)       the same metrics as preview/<split>/* scalars, if tensorboard is installed
The samples are fixed (same seed) for all checkpoints, so the images can be compared across steps.
"""

import argparse
import gc
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview_plan_predictions import DEFAULT_EXCLUDE_TASKS, load_engine, load_rows, run_preview  # noqa: E402

READY_FILE = "trainer_state.json"  # written by the HF Trainer after the weights of a checkpoint


def step_of(path: str) -> int:
    return int(re.search(r"checkpoint-(\d+)", path).group(1))


def find_run_dir(output_dir: str, started: float, run_dir: str | None) -> str | None:
    """The swift run dir <output_dir>/v<k>-<time> created by the current training (newest, not older than start)."""
    if run_dir:
        return run_dir
    runs = [d for d in glob.glob(os.path.join(output_dir, "v*")) if os.path.isdir(d)]
    runs = [d for d in runs if os.path.getmtime(d) >= started - 60]
    return max(runs, key=os.path.getmtime) if runs else None


def parent_alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def make_tb_writer(run_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(os.path.join(run_dir, "runs", "preview"))


def process(ckpt: str, run_dir: str, splits: dict, args, tb) -> None:
    step = step_of(ckpt)
    print(f"[preview] checkpoint-{step}: loading {ckpt}", flush=True)
    import torch

    engine = load_engine(ckpt)
    out_root = os.path.join(run_dir, "preview", f"checkpoint-{step}")
    for split, rows in splits.items():
        if not rows:
            continue
        summary = run_preview(engine, rows, vis_dir=os.path.join(out_root, split),
                              out_jsonl=os.path.join(out_root, f"{split}.jsonl"),
                              max_tokens=args.max_tokens, verbose=args.verbose)
        rec = {"step": step, "split": split, **summary}
        print(f"[preview] {json.dumps(rec)}", flush=True)
        with open(os.path.join(run_dir, "preview", "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if tb:
            for k, v in summary.items():
                if k != "samples":
                    tb.add_scalar(f"preview/{split}/{k}", v, step)
            tb.flush()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="OUTPUT_DIR of finetune_vlm_plan_predictor.sh")
    p.add_argument("--run_dir", default=None, help="explicit <output_dir>/v<k>-<time>; default = the newest one")
    p.add_argument("--data_dir", default="data/robomme_preprocessed_data/qwenvl_plan")
    p.add_argument("--train_jsonl", default=None, help="default <data_dir>/plan_train_split.jsonl")
    p.add_argument("--val_jsonl", default=None, help="default <data_dir>/plan_val.jsonl ('' to skip)")
    p.add_argument("--n_train", type=int, default=6)
    p.add_argument("--n_val", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_tasks", default=DEFAULT_EXCLUDE_TASKS)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--poll", type=int, default=30, help="seconds between checks for new checkpoints")
    p.add_argument("--parent_pid", type=int, default=None, help="exit (after a final pass) when this pid is gone")
    p.add_argument("--stop_file", default=None, help="exit (after a final pass) when this file appears")
    p.add_argument("--once", action="store_true", help="process existing checkpoints and exit (no watching)")
    p.add_argument("--verbose", action="store_true", help="also print the per-sample GT/PRED text")
    args = p.parse_args()

    train_jsonl = args.train_jsonl or os.path.join(args.data_dir, "plan_train_split.jsonl")
    val_jsonl = os.path.join(args.data_dir, "plan_val.jsonl") if args.val_jsonl is None else args.val_jsonl
    splits = {"train": load_rows(train_jsonl, args.n_train, 1, args.seed, args.exclude_tasks)}
    if val_jsonl and os.path.exists(val_jsonl):
        splits["val"] = load_rows(val_jsonl, args.n_val, 1, args.seed, args.exclude_tasks)
    print(f"[preview] samples: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)

    started = 0.0 if (args.once or args.run_dir) else time.time()
    done: set[str] = set()
    run_dir, tb, announced = None, None, False
    while True:
        # evaluated before scanning, so the scan right after training ends still picks up the last checkpoints
        alive = parent_alive(args.parent_pid) and not (args.stop_file and os.path.exists(args.stop_file))
        run_dir = find_run_dir(args.output_dir, started, args.run_dir) or run_dir
        if run_dir:
            if not announced:
                os.makedirs(os.path.join(run_dir, "preview"), exist_ok=True)
                tb = make_tb_writer(run_dir)
                announced = True
                print(f"[preview] watching {run_dir}", flush=True)
            ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")), key=step_of)
            ready = [c for c in ckpts if c not in done and os.path.exists(os.path.join(c, READY_FILE))]
            # newest first: with save_total_limit old checkpoints may be deleted before we get to them
            for ckpt in reversed(ready):
                if not os.path.isdir(ckpt):
                    continue
                time.sleep(5)  # let the trainer finish writing
                try:
                    process(ckpt, run_dir, splits, args, tb)
                except Exception as e:  # checkpoint deleted mid-way, OOM, ...: log and go on
                    print(f"[preview] {ckpt} failed: {type(e).__name__}: {e}", flush=True)
                done.add(ckpt)
                break  # re-scan: a newer checkpoint may have appeared meanwhile
            else:
                if args.once or not alive:
                    break
                time.sleep(args.poll)
                continue
            continue
        if args.once or not alive:
            break
        time.sleep(args.poll)
    print("[preview] done", flush=True)


if __name__ == "__main__":
    main()
