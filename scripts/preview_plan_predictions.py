"""Quick look at full-plan VLM predictions vs ground truth on a few jsonl rows.

Usage (robomme env, 1 GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/preview_plan_predictions.py \
        --adapter runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/<run>/checkpoint-XXX \
        --jsonl data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl \
        --n 10 --per_task 1 --out runs/plan_preview.jsonl
"""

import os

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")
os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")

import argparse
import json
import random
import re
from collections import defaultdict

BOX_RE = re.compile(r"<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>")
IMAGE_SIZE = 256  # robomme front_rgb; Qwen3-VL emits coords normalized to 0..1000


def parse_pred(text: str) -> tuple[list[str], list[list[int]]]:
    """Model output -> (steps with <bbox> placeholders, pixel points)."""
    pts = [[int(int(a) * IMAGE_SIZE / 1000), int(int(b) * IMAGE_SIZE / 1000)] for a, b in BOX_RE.findall(text)]
    text = BOX_RE.sub("<bbox>", text)
    steps = [re.sub(r"^\d+\.\s*", "", s.strip()) for s in text.split(";") if s.strip()]
    return steps, pts


def parse_gt(row: dict) -> tuple[list[str], list[list[int]]]:
    text = row["messages"][2]["content"]
    steps = [re.sub(r"^\d+\.\s*", "", s.strip()) for s in text.split(";") if s.strip()]
    return steps, row["objects"]["bbox"]


def task_of(row: dict) -> str:
    # image name: <Task>_ep<i>_step<j>.png
    return os.path.basename(row["images"][0]).split("_ep")[0]


def select_rows(rows: list[dict], n: int, per_task: int, seed: int) -> list[dict]:
    random.seed(seed)
    by_task = defaultdict(list)
    for r in rows:
        by_task[task_of(r)].append(r)
    picked = []
    for t in sorted(by_task):
        picked += random.sample(by_task[t], min(per_task, len(by_task[t])))
    random.shuffle(picked)
    return picked[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--jsonl", default="data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--per_task", type=int, default=1, help="rows per task before truncating to n")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--out", default=None, help="optional jsonl with pred/gt per sample")
    args = p.parse_args()

    from swift.llm import InferRequest, PtEngine, RequestConfig

    rows = [json.loads(l) for l in open(args.jsonl)]
    rows = select_rows(rows, args.n, args.per_task, args.seed)

    engine = PtEngine(model_id_or_path="Qwen/Qwen3-VL-4B-Instruct", adapters=[args.adapter], attn_impl="sdpa")
    cfg = RequestConfig(max_tokens=args.max_tokens, temperature=0)

    reqs = []
    for r in rows:
        d = {"messages": r["messages"][:2], "images": r["images"]}
        if "videos" in r:
            d["videos"] = r["videos"]
        reqs.append(InferRequest(**d))

    outs = []
    for r, req in zip(rows, reqs):  # batch=1: rows have different video/image sizes
        outs.append(engine.infer([req], request_config=cfg)[0].choices[0].message.content)

    n_exact_text, n_len_ok, px_errs = 0, 0, []
    fout = open(args.out, "w") if args.out else None
    for r, raw in zip(rows, outs):
        gt_steps, gt_pts = parse_gt(r)
        pr_steps, pr_pts = parse_pred(raw)
        exact = gt_steps == pr_steps
        n_exact_text += exact
        n_len_ok += len(gt_steps) == len(pr_steps)
        errs = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(gt_pts, pr_pts)]
        px_errs += errs

        print("=" * 100)
        print(f"{task_of(r)} | {os.path.basename(r['images'][0])} | video={'videos' in r}")
        print("goal:", r["messages"][1]["content"].split("\n")[0].replace("<video>", ""))
        print(f"steps gt={len(gt_steps)} pred={len(pr_steps)} text_exact={exact} "
              f"bbox gt={len(gt_pts)} pred={len(pr_pts)} L1px={errs}")
        for i in range(max(len(gt_steps), len(pr_steps))):
            g = gt_steps[i] if i < len(gt_steps) else "-"
            q = pr_steps[i] if i < len(pr_steps) else "-"
            mark = "  " if g == q else "!!"
            print(f"  {mark} {i + 1:2d}. GT  : {g}\n         PRED: {q}")
        print("  GT pts  :", gt_pts)
        print("  PRED pts:", pr_pts)
        print("  RAW     :", raw)
        if fout:
            fout.write(json.dumps({"task": task_of(r), "image": r["images"][0], "raw": raw,
                                   "gt_steps": gt_steps, "pred_steps": pr_steps,
                                   "gt_pts": gt_pts, "pred_pts": pr_pts}) + "\n")
    if fout:
        fout.close()

    n = len(rows)
    print("=" * 100)
    print(f"samples={n} text_exact={n_exact_text}/{n} same_len={n_len_ok}/{n} "
          f"mean_L1px={sum(px_errs) / max(len(px_errs), 1):.1f} (over {len(px_errs)} matched points)")


if __name__ == "__main__":
    main()
