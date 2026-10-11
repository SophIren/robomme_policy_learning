# Full-plan VLM predictor (PoC). See docs/PLAN_PREDICTOR_POC.md.
# More usage of Swift: https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html
#
# Examples:
#   GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh                    # LoRA (default)
#   TRAIN_TYPE=full GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh    # full fine-tune of the LLM
#   EPOCHS=40 LR=3e-4 GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh  # override anything below
#
# The dataset is small (~1 row per episode), so what matters is the number of optimizer steps:
#   steps ~= rows * EPOCHS / GLOBAL_BATCH.  ~1300 rows, 20 epochs, batch 16 -> ~1600 steps
#   (the original subgoal script: 2 epochs, batch 64 -> only ~40 steps on this dataset).

# ---- data ----
# Train on plan_train_split.jsonl / eval on plan_val.jsonl (made by scripts/split_plan_dataset.py, which also drops
# the excluded tasks). VAL_DATASET='' trains without eval.
DATA_DIR=${DATA_DIR:-'data/robomme_preprocessed_data/qwenvl_plan'}
DATASET_PATH=${DATASET_PATH:-$DATA_DIR/plan_train_split.jsonl}
VAL_DATASET=${VAL_DATASET-$DATA_DIR/plan_val.jsonl}

# ---- what to train ----
TRAIN_TYPE=${TRAIN_TYPE:-lora}           # lora | full
if [ "$TRAIN_TYPE" == "full" ]; then
    # Full fine-tune of the LLM (+ aligner); the ViT stays frozen. ~8 GB per saved checkpoint.
    LR=${LR:-1e-5}
    EPOCHS=${EPOCHS:-15}
    FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
    DEEPSPEED=${DEEPSPEED:-zero3}        # zero2 may OOM for a 4B full FT on <=48 GB cards
    PER_DEVICE=${PER_DEVICE:-2}
    TYPE_ARGS="--train_type full"
else
    LR=${LR:-2e-4}
    EPOCHS=${EPOCHS:-20}
    FREEZE_ALIGNER=${FREEZE_ALIGNER:-true}
    DEEPSPEED=${DEEPSPEED:-zero2}
    PER_DEVICE=${PER_DEVICE:-4}
    LORA_RANK=${LORA_RANK:-64}
    LORA_ALPHA=${LORA_ALPHA:-128}
    TYPE_ARGS="--train_type lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --target_modules all-linear"
fi
OUTPUT_DIR=${OUTPUT_DIR:-runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan_$TRAIN_TYPE}

# ---- GPUs / batch ----
# GPU_IDS defaults to all visible GPUs. GLOBAL_BATCH = PER_DEVICE * GRAD_ACC * NUM_GPUS.
GPU_IDS=${GPU_IDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
GRAD_ACC=${GRAD_ACC:-$(( GLOBAL_BATCH / (PER_DEVICE * NUM_GPUS) ))}
[ "$GRAD_ACC" -lt 1 ] && GRAD_ACC=1

# ---- eval / save ----
EVAL_STEPS=${EVAL_STEPS:-100}
# EVAL_ON_START=true runs one full eval before the first training step (HF Trainer `eval_on_start`):
# a quick way to check that eval itself works without waiting EVAL_STEPS steps.
EVAL_ON_START=${EVAL_ON_START:-false}
if [ -n "$VAL_DATASET" ]; then
    # eval_loss / eval_token_acc on val every EVAL_STEPS; checkpoints saved at the same steps
    EVAL_ARGS="--val_dataset $VAL_DATASET --eval_strategy steps --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS --per_device_eval_batch_size $PER_DEVICE --eval_on_start $EVAL_ON_START"
else
    EVAL_ARGS="--save_steps $EVAL_STEPS"
fi

echo "TRAIN_TYPE=$TRAIN_TYPE LR=$LR EPOCHS=$EPOCHS DEEPSPEED=$DEEPSPEED FREEZE_ALIGNER=$FREEZE_ALIGNER"
echo "GPU_IDS=$GPU_IDS NUM_GPUS=$NUM_GPUS PER_DEVICE=$PER_DEVICE GRAD_ACC=$GRAD_ACC (global batch $((PER_DEVICE * GRAD_ACC * NUM_GPUS)))"
echo "train=$DATASET_PATH val=${VAL_DATASET:-none} out=$OUTPUT_DIR"
[ -f "$DATASET_PATH" ] && echo "train rows: $(wc -l < "$DATASET_PATH") -> ~$(( $(wc -l < "$DATASET_PATH") * EPOCHS / (PER_DEVICE * GRAD_ACC * NUM_GPUS) )) optimizer steps"

PYTORCH_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=$NUM_GPUS \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset $DATASET_PATH \
    --split_dataset_ratio 0.0 \
    --load_from_cache_file true \
    --packing false \
    $TYPE_ARGS \
    --torch_dtype bfloat16 \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACC \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --freeze_vit true \
    --freeze_aligner $FREEZE_ALIGNER \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    $EVAL_ARGS \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --deepspeed $DEEPSPEED \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
