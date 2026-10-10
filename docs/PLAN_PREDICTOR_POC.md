# Full-plan VLM predictor (PoC)

One sample = one episode. Input: video demo (if any) + task goal + first execution frame.
Output: the whole plan with pixel coordinates, e.g.
`1. pick up the cube at <bbox>; 2. place the cube onto the correct target at <bbox>; ...`
Coordinates are `[y, x]` in 0..255 and go through ms-swift `objects.bbox`, same as the existing grounded-subgoal dataset.

New files:
- `src/mme_vla_suite/dataset_builder/build_vlm_plan_dataset_qwenvl.py`
- `scripts/finetune_vlm_plan_predictor.sh`
- `scripts/build_dataset.py` (new `--dataset_type vlm_plan_qwenvl`)

Nothing was run locally (no data, no GPU). Run all steps below in the cloud, in order.

## 0. Env

```bash
git status --short                      # expect the 3 files above
uv sync                                 # or the project's usual env setup
uv run python -c "import h5py, imageio, swift; print('ok')"
nvidia-smi
```

## 1. Raw data

```bash
ls data/robomme_h5_data | head
ls data/robomme_h5_data | wc -l         # number of task files
uv run python - <<'EOF'
import h5py, glob
f = sorted(glob.glob("data/robomme_h5_data/*.h5"))[0]
with h5py.File(f) as d:
    eps = [k for k in d if k.startswith("episode_")]
    ep = d[eps[0]]
    ts = sorted([k for k in ep if k.startswith("timestep_")], key=lambda s: int(s.split("_")[1]))
    print(f, len(eps), "episodes;", len(ts), "timesteps in first")
    info = ep[ts[0]]["info"]
    print("info keys:", list(info.keys()))
    print("task_goal:", ep["setup"]["task_goal"][()])
EOF
```

Check that `info` has `simple_subgoal`, `grounded_subgoal`, `is_video_demo`.

## 2. Build dataset (smoke test first)

```bash
uv run python scripts/build_dataset.py --dataset_type vlm_plan_qwenvl --max_episodes 2
```

Output: `data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl` and `.../images/`.
Check after the smoke run:

```bash
D=data/robomme_preprocessed_data/qwenvl_plan
wc -l $D/plan_train.jsonl               # ~ (#task files) * 2
ls $D/images | head
ls $D/images | grep -c mp4              # videos only for tasks with a demo
head -n 1 $D/plan_train.jsonl | python -m json.tool
```

Inspect a few plans, with one line per sample:

```bash
uv run python - <<'EOF'
import json
for l in open("data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl"):
    r = json.loads(l)
    a = r["messages"][2]["content"]
    nb = a.count("<bbox>")
    nobj = len(r["objects"]["bbox"])
    flag = "" if nb == nobj else "  <-- BBOX MISMATCH"
    print(r["images"][0].split("/")[-1], "| bboxes", nb, nobj, "|", a[:160], flag)
EOF
```

Must hold:
- `count("<bbox>") == len(objects.bbox)` for every row (hard requirement for ms-swift).
- Plans look sane: steps in the right order, no `complete`, coordinates inside 0..255.
- Video path exists for tasks that have a demo; the user prompt contains `<video>` iff `videos` is present.
- The image is the first execution frame (robot at start pose, not a demo frame).

Known risk to eyeball: consecutive segments with the same `simple_subgoal` (e.g. "pick up the cube" twice in a row)
are merged into one step by `_extract_plan`. If you see plans shorter than expected on counting/repeat tasks,
change the segment boundary to also trigger on a `grounded_subgoal` change.

Then the full build:

```bash
uv run python scripts/build_dataset.py --dataset_type vlm_plan_qwenvl
wc -l data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl
```

Dataset is small: one row per episode. For augmentation pass `frame_offsets=(0, 8, 16)` and
`bbox_noise=True` to `PlanDatasetBuilder` in `scripts/build_dataset.py` (not exposed as CLI flags).

## 3. Training smoke test (before the real run)

Quick check that the swift pipeline tokenizes the data and loss goes down.
Run 1 GPU, a few steps:

```bash
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=256 VIDEO_MAX_TOKEN_NUM=64 FPS_MAX_FRAMES=10 \
swift sft \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl \
  --split_dataset_ratio 0.0 --train_type lora --torch_dtype bfloat16 \
  --max_steps 10 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit true --freeze_aligner true --gradient_checkpointing true \
  --max_length 4096 --logging_steps 1 --save_steps 1000 \
  --output_dir runs/ckpts/vlm_plan_predictor/debug
```

Watch for:
- No "token indices / bbox count mismatch" or truncation warnings. If samples are truncated, raise `--max_length`
  or lower `VIDEO_MAX_TOKEN_NUM` / `FPS_MAX_FRAMES`.
- Loss is finite and decreasing; no OOM (if OOM, `--per_device_train_batch_size 1`).
- Look at the logged max seq length; plan answers are longer than the single-subgoal ones, so 4096 should be enough but verify.

## 4. Real training

```bash
bash scripts/finetune_vlm_plan_predictor.sh
```

Differences from `finetune_vlm_subgoal_predictor.sh`: dataset/output paths, `--max_length 4096`,
`--save_steps 50`, `--logging_steps 5` (small dataset). Output: `runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/`.
With ~one row per episode, 2 epochs may be too few; raise `--num_train_epochs` (e.g. 5-10) if the plan is not memorized/generalized.

```bash
tail -f runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/*/logging.jsonl
ls runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/*/
```

## 5. Check predictions

Inference on training/held-out rows, bypassing the env. Use the latest checkpoint dir:

```bash
CKPT=$(ls -d runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v*/checkpoint-* | sort -V | tail -n 1)
echo $CKPT
IMAGE_MAX_TOKEN_NUM=256 VIDEO_MAX_TOKEN_NUM=64 FPS_MAX_FRAMES=10 CUDA_VISIBLE_DEVICES=0 \
swift infer --adapters $CKPT --infer_backend pt --temperature 0 --max_new_tokens 512 \
  --val_dataset data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl --max_batch_size 1
```

Compare the output against `messages[2].content` in the jsonl:
- Step count and wording match.
- `<bbox>` count matches; the model emits coordinates in the objects/bbox format that swift uses (check the raw
  output: it should contain `<bbox>` placeholders replaced by `[y, x]`-style boxes, not free-form numbers).
- Pixel error vs ground truth (should be a few px on the train set).

To hold out data for a real generalization check, split by episode before training
(e.g. take the last N lines of the jsonl into `plan_val.jsonl` and pass it as `--val_dataset`).

## 6. Using the plan manually (PoC)

The plan string is `"1. <step>; 2. <step>; ..."`. Split on `"; "` and feed the current step as the subgoal to the VLA
(`symbolic-grounded-subgoal` config) yourself, switching steps by hand. Remember the VLA expects the `at <y, x>` text
form with real coordinates, so substitute the predicted pixel pair back into each `<bbox>`.
