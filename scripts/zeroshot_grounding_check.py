"""Zero-shot grounding check of the BASE Qwen3-VL (no adapter): can it point at the cube / button / target?

For a few plan rows it asks the base model, in its own native format, to locate each object mentioned in the GT plan
(one question per object), and draws the answers next to the GT points with the same picture as
scripts/preview_plan_predictions.py. Only the objects whose GT point is on the *input* frame are meaningful
(the first step, buttons, static targets); later "moved" positions cannot be seen on this frame.

    CUDA_VISIBLE_DEVICES=0 python scripts/zeroshot_grounding_check.py \
        --jsonl data/robomme_preprocessed_data/qwenvl_plan_native/plan_val.jsonl --n 10 --out_dir runs/zeroshot_grounding

Native output format: Qwen3-VL is trained to answer grounding questions with JSON like
    [{"point_2d": [x, y], "label": "red cube"}]   or   [{"bbox_2d": [x1, y1, x2, y2], "label": ...}]
with coordinates relative to 0..1000 and x first. The parser below accepts point_2d, bbox_2d (uses the center) and
plain "(x, y)" / "[x, y]" fallbacks, and converts to our [y, x] pixels in 0..255.
Use --prompt_format to try a different phrasing; --coords abs if the model answers in absolute pixels instead.
"""

import os

os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "256")

import argparse
import json
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview_plan_predictions import (  # noqa: E402
    BASE_MODEL, DEFAULT_EXCLUDE_TASKS, load_rows, parse_gt, save_vis, task_of,
)

IMAGE_SIZE = 256

PROMPTS = {
    # Qwen3-VL cookbook style
    "point": 'Locate {obj} in the image and output its center point as JSON: [{{"point_2d": [x, y], "label": "{obj}"}}]',
    "bbox": 'Locate {obj} in the image and output its bounding box as JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]',
    "plain": "Point to {obj}.",
}

# object phrases we can ask about, extracted from GT steps like "pick up the red cube at <bbox> ..."
# (last "the ..." before "at <bbox>": "place the red cube onto the target at <bbox>" -> "the target")
GENERIC_RE = re.compile(r"(the (?:(?!\bthe\b)[a-z\- ])+?)\s+at <bbox>")


def objects_in_step(step: str) -> list[str]:
    """One object phrase per <bbox> in the step, in order (best effort)."""
    found = GENERIC_RE.findall(step)
    if len(found) == step.count("<bbox>"):
        return [f.strip() for f in found]
    return [step.replace("<bbox>", "").strip()] * step.count("<bbox>")


def parse_answer(text: str, coords: str) -> list[int] | None:
    """Model answer -> [y, x] in 0..255, or None."""
    def to_px(x, y):
        if coords == "rel":
            x, y = x * IMAGE_SIZE / 1000, y * IMAGE_SIZE / 1000
        return [int(round(y)), int(round(x))]

    m = re.search(r'"point_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        return to_px(float(m.group(1)), float(m.group(2)))
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        x1, y1, x2, y2 = map(float, m.groups())
        return to_px((x1 + x2) / 2, (y1 + y2) / 2)
    m = re.search(r"<\|box_start\|>\((\d+),(\d+)\)", text)  # Qwen2-VL style, (x, y) 0..1000
    if m:
        return to_px(float(m.group(1)), float(m.group(2)))
    m = re.search(r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]", text)
    if m:
        return to_px(float(m.group(1)), float(m.group(2)))
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="data/robomme_preprocessed_data/qwenvl_plan_native/plan_val.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_tasks", default=DEFAULT_EXCLUDE_TASKS)
    p.add_argument("--model", default=BASE_MODEL, help="base model id or path; no adapter is loaded")
    p.add_argument("--prompt_format", default="point", choices=list(PROMPTS))
    p.add_argument("--coords", default="rel", choices=["rel", "abs"], help="rel = 0..1000 (Qwen3-VL), abs = pixels")
    p.add_argument("--first_only", action="store_true", help="ask only about the first object of each plan")
    p.add_argument("--out_dir", default="runs/zeroshot_grounding")
    args = p.parse_args()

    from swift.llm import InferRequest, PtEngine, RequestConfig

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.jsonl, args.n, 1, args.seed, args.exclude_tasks)
    engine = PtEngine(model_id_or_path=args.model, attn_impl="sdpa")
    cfg = RequestConfig(max_tokens=128, temperature=0)

    all_errs, first_errs, n_fail = [], [], 0
    fout = open(os.path.join(args.out_dir, "answers.jsonl"), "w")
    for r in rows:
        gt_steps, gt_pts = parse_gt(r)
        queries = []  # (step_idx, object phrase, gt point)
        k = 0
        for i, s in enumerate(gt_steps):
            for obj in objects_in_step(s):
                if k < len(gt_pts):
                    queries.append((i, obj, gt_pts[k]))
                k += 1
        if args.first_only:
            queries = queries[:1]

        pred_steps, pred_pts = [], []
        print("=" * 100)
        print(f"{task_of(r)} | {os.path.basename(r['images'][0])}")
        for i, obj, gt in queries:
            # image only, no video and no task context: a pure grounding question
            req = InferRequest(messages=[{"role": "user",
                                          "content": "<image>" + PROMPTS[args.prompt_format].format(obj=obj)}],
                               images=r["images"])
            ans = engine.infer([req], request_config=cfg)[0].choices[0].message.content
            pt = parse_answer(ans, args.coords)
            err = abs(pt[0] - gt[0]) + abs(pt[1] - gt[1]) if pt else None
            print(f"  step {i + 1}: {obj!r:35s} GT {gt}  PRED {pt}  L1 {err}  | {ans.strip()[:120]!r}")
            fout.write(json.dumps({"image": r["images"][0], "step": i + 1, "object": obj, "gt": gt,
                                   "pred": pt, "l1": err, "answer": ans}) + "\n")
            if pt is None:
                n_fail += 1
                continue
            all_errs.append(err)
            if (i, obj) == (queries[0][0], queries[0][1]):
                first_errs.append(err)
            pred_steps.append(f"{obj} at <bbox>")
            pred_pts.append(pt)
        # picture: GT plan points (green) vs zero-shot points (red), paired by object order
        gt_vis_steps = [f"{obj} at <bbox>" for _, obj, _ in queries]
        gt_vis_pts = [g for _, _, g in queries]
        save_vis(r, gt_vis_steps, gt_vis_pts, pred_steps, pred_pts,
                 os.path.join(args.out_dir, os.path.basename(r["images"][0]).replace(".png", "_zeroshot.png")))
    fout.close()

    mean = lambda xs: sum(xs) / max(len(xs), 1)
    print("=" * 100)
    print(f"queries={len(all_errs) + n_fail} unparsed={n_fail} mean_L1px={mean(all_errs):.1f} "
          f"first_object_L1px={mean(first_errs):.1f}  -> pictures and answers.jsonl in {args.out_dir}")


if __name__ == "__main__":
    main()
