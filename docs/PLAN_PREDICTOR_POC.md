# Full-plan VLM predictor (PoC)

One sample = one episode. Input: video demo (if any) + task goal + first execution frame.
Output: the whole plan with pixel coordinates, e.g.
`1. pick up the cube at <bbox>; 2. place the cube onto the correct target at <bbox>; ...`
Coordinates are `[y, x]` in 0..255 and go through ms-swift `objects.bbox`, same as the existing grounded-subgoal dataset.

New files:
- `src/mme_vla_suite/dataset_builder/build_vlm_plan_dataset_qwenvl.py`
- `scripts/finetune_vlm_plan_predictor.sh`
- `scripts/build_dataset.py` (new `--dataset_type vlm_plan_qwenvl`)
- `scripts/split_plan_dataset.py` (train/val split by episode)
- `scripts/preview_plan_predictions.py` (GT vs prediction on a few samples)

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

## 4. Split train / val (by episode)

There are three disjoint sources of episodes, and it matters which one you look at:

| Source | Where | Seeds | Used for |
|---|---|---|---|
| benchmark `train` (100 ep/task) | `robomme_data_h5` -> `plan_train.jsonl` | `env_metadata/train` | training |
| our held-out part of it | `plan_val.jsonl` (from `split_plan_dataset.py`) | subset of train seeds | val metrics during training, step 7 |
| benchmark `test` (50 ep/task) | `examples/robomme/eval.py` (`dataset="test"`) | `env_metadata/test` | closed-loop PoC |

I checked `third_party/robomme_benchmark/src/robomme/env_metadata/{train,val,test}`: for all 16 tasks the seed sets
are disjoint (train 100, val 50, test 50, zero overlap). So `eval.py` never runs on training episodes.
There is also a benchmark `val` split (50 ep/task) in the env metadata, but the released h5 contains only train
demos. `eval.py` hardcodes `dataset="test"` in `examples/robomme/env_runner.py`; change it to `"val"` there if you want
a closed-loop set for tuning that is separate from test.

Without a split the training script trains on every row (`--split_dataset_ratio 0.0`), so any offline check on
`plan_train.jsonl` measures only memorization. Make the split before training:

```bash
python scripts/split_plan_dataset.py --val_per_task 10
wc -l data/robomme_preprocessed_data/qwenvl_plan/plan_{train_split,val}.jsonl
cat data/robomme_preprocessed_data/qwenvl_plan/plan_val_episodes.json | head
```

It splits by **episode**, not by row (with `frame_offsets` several rows share one episode/plan), randomly with a
fixed seed, `--val_per_task` episodes per task. Same task distribution in train and val.

## 5. Real training (with val metrics)

```bash
GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh
```

Defaults: train on `plan_train_split.jsonl`, eval on `plan_val.jsonl` every `EVAL_STEPS=25` steps, checkpoint at
the same steps (keep last 3). Overrides via env vars:

```bash
EVAL_STEPS=10 GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh             # eval more often
DATASET_PATH=data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl VAL_DATASET= \
  GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh                         # final run on all data, no eval
```

Other differences from `finetune_vlm_subgoal_predictor.sh`: `--max_length 4096`, `--logging_steps 5`, GPU count from
`GPU_IDS`. With ~one row per episode, 2 epochs may be too few; raise `--num_train_epochs` (e.g. 5-10) and watch val.

What is logged (`runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v*/logging.jsonl`, and tensorboard in `runs/`):
- train: `loss`, `token_acc`
- val: `eval_loss`, `eval_token_acc` (teacher-forced, on the answer tokens incl. coordinates)

```bash
RUN=$(ls -d runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v* | sort -V | tail -n 1)
grep eval_loss $RUN/logging.jsonl | tail
tensorboard --logdir $RUN/runs --port 6006
```

Read it: `eval_loss` going up while `loss` goes down = overfitting, pick the checkpoint at the `eval_loss` minimum.
`eval_token_acc` is teacher-forced, so it is optimistic; the real generation quality (whole plan exact, step
count, pixel error) is measured by step 7 on `plan_val.jsonl`. Run step 7 on 1-2 checkpoints, it takes a minute.

## 6. Check predictions with swift infer (optional)

Inference on val rows, bypassing the env. Use the latest checkpoint dir:

```bash
CKPT=$(ls -d runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v*/checkpoint-* | sort -V | tail -n 1)
echo $CKPT
IMAGE_MAX_TOKEN_NUM=256 VIDEO_MAX_TOKEN_NUM=64 FPS_MAX_FRAMES=10 CUDA_VISIBLE_DEVICES=0 \
swift infer --adapters $CKPT --infer_backend pt --temperature 0 --max_new_tokens 512 \
  --val_dataset data/robomme_preprocessed_data/qwenvl_plan/plan_val.jsonl --max_batch_size 1
```

Raw dump of predictions; step 7 gives a readable GT-vs-PRED diff with metrics.

## 7. Quick look at ~10 predictions vs ground truth

```bash
CKPT=$(ls -d runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v*/checkpoint-* | sort -V | tail -n 1)
CUDA_VISIBLE_DEVICES=0 python scripts/preview_plan_predictions.py \
  --adapter $CKPT \
  --jsonl data/robomme_preprocessed_data/qwenvl_plan/plan_val.jsonl \
  --n 10 --per_task 1 --out runs/plan_preview.jsonl
```

(`--jsonl plan_train.jsonl` if you did not make a split; that only checks train fit.)

It picks `--per_task` rows per task (so the 10 samples cover different tasks), runs greedy decoding, and prints for
each sample: task, whether it has a video, step-by-step GT vs PRED (`!!` marks a mismatching step), GT and predicted
pixel points `[y, x]`, per-point L1 error in pixels, and the raw model output. The last line is a summary:
`text_exact` (whole plan text equal), `same_len` (same number of steps), `mean_L1px`.

How to read it:
- `RAW` should contain `<|box_start|>(y,x)<|box_end|>` tokens (Qwen3-VL format, 0..1000); the script converts them to
  0..255 pixels the same way `examples/robomme/subgoal_prediction/qwenvl/api.py` does. No box tokens in `RAW`
  means the model did not learn the grounding format.
- On train rows expect almost exact text and a few px of error; on val rows, step text should still match for
  fixed-structure tasks, while video/memory tasks (VideoPlaceOrder, VideoRepick, ...) and coordinates of objects
  that will move later are where errors show up.
- Coordinates of later steps (after objects move / are revealed) cannot be known from the first frame; large L1 on
  those steps is expected and is the reason a tracker is needed later.

## 8. Interactive PoC: watch the sim and switch steps by hand

`examples/robomme/interactive_plan.py` runs one or a few episodes (not the whole benchmark):
1. resets the env, predicts the full plan **once** with your VLM (video demo + task goal + first execution frame,
   the same prompt as in training);
2. starts **paused** and serves a small web page with the live front + wrist view, the predicted points drawn on the
   frame, the plan list with the current step highlighted, and the oracle subgoal as a hint;
3. sends the current step to the VLA (`symbolic-grounded-subgoal`, which expects `... at <y, x>` text) until you
   switch to the next step.

You need the VLA checkpoint `runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999` (from
`huggingface.co/Yinpei/mme_vla_suite`, see README) and 2 GPUs (VLA server on one, sim + VLM on the other).

```bash
# terminal 0: VLA policy server (openpi env)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --seed=7 --port=8006 policy:checkpoint \
  --policy.dir=runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999 --policy.config=mme_vla_suite
# wait until it prints that the server is listening

# terminal 1: sim + plan VLM + web UI (robomme env)
micromamba activate robomme
CKPT=$(ls -d runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan/v*/checkpoint-* | sort -V | tail -n 1)
CUDA_VISIBLE_DEVICES=1 python examples/robomme/interactive_plan.py --port 8006 \
  --adapter $CKPT --task MoveCube --episodes 0,1 --ui_port 8890

# your laptop: tunnel the UI and open http://localhost:8890
ssh -L 8890:localhost:8890 <cloud-host>
```

Controls (keys work when the page is focused):

| key / button | action |
|---|---|
| `space` | pause / run (starts paused, so read the plan first) |
| `n` or `→` | next step (switch the subgoal) |
| `p` or `←` | previous step (e.g. the grasp failed, retry the pick) |
| click a plan line | jump to that step |
| `o` | one sim step while paused |
| `-` / `+` | slower / faster (default 30 ms per sim step) |
| text field + `set` | override the current step text, e.g. fix a bad coordinate: `pick up the cube at <71, 94>` |
| `s` | stop the episode (go to the next one in `--episodes`) |

When to press `next`: when the current step is visibly done, e.g. the cube is lifted after `pick up`, the gripper
opened over the target after `drop/place`, the button went down after `press`. The `oracle_hint` field on the page
shows the env's own current subgoal (`grounded_subgoal_online`); it is a convenient reference for timing, but switch
by what you see if you want an honest PoC. `env_status` shows `success` / `fail` when the env decides.

Notes on behavior (differences from `eval.py`):
- On every switch / edit the remaining queued actions of the current VLA chunk are dropped and the VLA is queried
  right away with the new subgoal. In `eval.py` the subgoal can change only every 16 steps (one action chunk).
- The VLA call happens every 16 sim steps as in `eval.py` (`--obs_horizon 16`), with the same history buffer.
- The plan is predicted once and never updated; coordinates of later steps (objects that moved / were revealed) can
  be wrong. Use the text field to correct a step if you want to separate "planner coordinates wrong" from
  "switching / VLA problem".

Useful flags:
- `--episodes 0,3,7`: test-split episode ids (0..49), `--task` one of the 16 tasks.
- `--dataset val`: use the benchmark val split instead of test (both disjoint from training, see section 4).
- `--plan_file my_plan.json`: skip the VLM, load a plan from a json list of strings (`["pick up the cube at <71, 94>",
  ...]`). Use it to debug the switching/VLA side without the planner, e.g. with the oracle plan copied from the page.
- `--max_steps 1300`, `--no_history` (only for VLA checkpoints trained without history).

Outputs in `runs/evaluation/plan_poc/<Task>_ep<i>/`:
- `<result>.mp4`: rollout video with the active subgoal and its point drawn on every frame;
- `log.json`: raw VLM output, parsed plan, final result, and every switch with its sim step
  (`{"step": 233, "from": 0, "to": 1, ...}`), so you can compare your switch timings with the oracle ones later;
- `plan_image.png`, `plan_video.mp4`: exactly what the planner saw.

Suggested PoC set (a few episodes each): one without video (`PickXtimes`, `ButtonUnmask`), one with video
(`MoveCube`, `VideoPlaceOrder`), one long counting task (`BinFill`). For each, compare: success with your plan vs the
same episode with `--plan_file` built from the oracle hints; that isolates planner errors from switching errors.

Troubleshooting:
- page loads but no image: the planner is still running (first call loads the model); watch terminal 1.
- `connection refused` on start: the VLA server in terminal 0 is not up yet or the port differs.
- empty plan / no points on the frame: the VLM did not emit `<|box_start|>` tokens, check `RAW plan` in terminal 1
  and step 7.
