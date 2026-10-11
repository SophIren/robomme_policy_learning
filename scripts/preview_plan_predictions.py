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


def point_step_ids(steps: list[str]) -> list[int]:
    """Step number (1-based) for every <bbox> in order of appearance."""
    return [i + 1 for i, s in enumerate(steps) for _ in range(s.count("<bbox>"))]


def save_vis(row: dict, gt_steps, gt_pts, pr_steps, pr_pts, out_path: str, scale: int = 3) -> None:
    """Input frame (upscaled) with GT points (green) and predicted points (red), numbered by plan step,
    a yellow line between GT and prediction with the same index, and a text panel with both plans."""
    import cv2
    import imageio
    import numpy as np

    img = imageio.imread(row["images"][0])[..., :3]
    h, w = img.shape[:2]
    canvas = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    GT, PR, LINE = (0, 255, 0), (255, 40, 40), (255, 220, 0)  # RGB

    def xy(p):  # [y, x] in 0..255 -> (x, y) on canvas
        return int(p[1] * scale), int(p[0] * scale)

    for a, b in zip(gt_pts, pr_pts):
        cv2.line(canvas, xy(a), xy(b), LINE, 1, cv2.LINE_AA)
    for pts, sids, color, marker in ((gt_pts, point_step_ids(gt_steps), GT, "o"),
                                     (pr_pts, point_step_ids(pr_steps), PR, "x")):
        for k, p in enumerate(pts):
            c = xy(p)
            if marker == "o":
                cv2.circle(canvas, c, 7, color, 2, cv2.LINE_AA)
            else:
                cv2.drawMarker(canvas, c, color, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            label = str(sids[k]) if k < len(sids) else "?"
            cv2.putText(canvas, label, (c[0] + 8, c[1] - 8), font, 0.5, color, 1, cv2.LINE_AA)

    # text panel: legend + both plans, with pixel coords and per-point L1
    lines = [(f"{task_of(row)}  {os.path.basename(row['images'][0])}", (255, 255, 255)),
             ("green o = GT, red x = PRED, number = plan step, yellow = GT<->PRED pair", (200, 200, 200))]
    gi, pi_ = iter(gt_pts), iter(pr_pts)
    for i in range(max(len(gt_steps), len(pr_steps))):
        g = gt_steps[i] if i < len(gt_steps) else "-"
        q = pr_steps[i] if i < len(pr_steps) else "-"
        g_pts = [next(gi, None) for _ in range(g.count("<bbox>"))]
        q_pts = [next(pi_, None) for _ in range(q.count("<bbox>"))]
        lines.append((f"{i + 1}. GT  : {g}  {[p for p in g_pts if p is not None]}", GT))
        err = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(g_pts, q_pts) if a is not None and b is not None]
        lines.append((f"   PRED: {q}  {[p for p in q_pts if p is not None]}" + (f"  L1={err}" if err else ""),
                      PR if g != q else (255, 160, 160)))
    line_h = 18
    panel = np.zeros((line_h * len(lines) + 10, canvas.shape[1], 3), dtype=np.uint8)
    for k, (text, color) in enumerate(lines):
        cv2.putText(panel, text[:140], (6, 14 + k * line_h), font, 0.42, color, 1, cv2.LINE_AA)
    imageio.imwrite(out_path, np.concatenate([canvas, panel], axis=0))


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
    p.add_argument("--adapter", required=True, help="checkpoint dir: LoRA adapter or full fine-tune")
    p.add_argument("--jsonl", default="data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--per_task", type=int, default=1, help="rows per task before truncating to n")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--out", default=None, help="optional jsonl with pred/gt per sample")
    p.add_argument("--vis_dir", default="runs/plan_preview_vis", help="where to save GT-vs-PRED images ('' to skip)")
    p.add_argument("--exclude_tasks", default="ButtonUnmaskSwap,PickHighlight", help="comma-separated, '' = keep all")
    args = p.parse_args()
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    exclude = {t for t in args.exclude_tasks.split(",") if t}

    from swift.llm import InferRequest, PtEngine, RequestConfig

    rows = [json.loads(l) for l in open(args.jsonl)]
    rows = [r for r in rows if task_of(r) not in exclude]
    rows = select_rows(rows, args.n, args.per_task, args.seed)

    if os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        engine = PtEngine(model_id_or_path="Qwen/Qwen3-VL-4B-Instruct", adapters=[args.adapter], attn_impl="sdpa")
    else:  # full fine-tune checkpoint
        engine = PtEngine(model_id_or_path=args.adapter, attn_impl="sdpa")
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
        if args.vis_dir:
            vis_path = os.path.join(args.vis_dir, os.path.basename(r["images"][0]).replace(".png", "_vis.png"))
            save_vis(r, gt_steps, gt_pts, pr_steps, pr_pts, vis_path)
            print("  VIS     :", vis_path)
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
