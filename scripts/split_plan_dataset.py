"""Split plan_train.jsonl into train/val by episode (not by row), dropping excluded tasks.

    python scripts/split_plan_dataset.py --val_per_task 10
-> <dir>/plan_train_split.jsonl, <dir>/plan_val.jsonl

By default ButtonUnmaskSwap and PickHighlight are excluded from both train and val (they do not fit the
"plan from the first frame" setup yet). Pass --exclude_tasks "" to keep all tasks.
"""

import argparse
import json
import os
import random
from collections import defaultdict

DEFAULT_EXCLUDE_TASKS = "ButtonUnmaskSwap,PickHighlight"


def episode_key(row: dict) -> tuple[str, int]:
    # image name: <Task>_ep<i>_step<j>.png
    name = os.path.basename(row["images"][0]).split("_step")[0]
    task, ep = name.rsplit("_ep", 1)
    return task, int(ep)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/robomme_preprocessed_data/qwenvl_plan_native")
    p.add_argument("--val_per_task", type=int, default=10, help="held-out episodes per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_tasks", default=DEFAULT_EXCLUDE_TASKS, help="comma-separated, '' = keep all")
    args = p.parse_args()
    exclude = {t for t in args.exclude_tasks.split(",") if t}

    rows = [json.loads(l) for l in open(os.path.join(args.dir, "plan_train.jsonl"))]
    n_all = len(rows)
    rows = [r for r in rows if episode_key(r)[0] not in exclude]
    print(f"excluded tasks: {sorted(exclude) or 'none'} ({n_all - len(rows)} rows dropped)")
    eps = defaultdict(set)
    for r in rows:
        t, e = episode_key(r)
        eps[t].add(e)

    rng = random.Random(args.seed)
    val = set()
    for t in sorted(eps):
        chosen = rng.sample(sorted(eps[t]), min(args.val_per_task, len(eps[t]) - 1))
        val |= {(t, e) for e in chosen}

    n_tr = n_va = 0
    with open(os.path.join(args.dir, "plan_train_split.jsonl"), "w") as ftr, \
         open(os.path.join(args.dir, "plan_val.jsonl"), "w") as fva:
        for r in rows:
            if episode_key(r) in val:
                fva.write(json.dumps(r) + "\n"); n_va += 1
            else:
                ftr.write(json.dumps(r) + "\n"); n_tr += 1

    with open(os.path.join(args.dir, "plan_val_episodes.json"), "w") as f:
        json.dump(sorted([f"{t}_ep{e}" for t, e in val]), f, indent=1)

    for t in sorted(eps):
        nv = sum(1 for (tt, _) in val if tt == t)
        print(f"{t:18s} episodes={len(eps[t])} val={nv}")
    print(f"rows: train={n_tr} val={n_va}")


if __name__ == "__main__":
    main()
