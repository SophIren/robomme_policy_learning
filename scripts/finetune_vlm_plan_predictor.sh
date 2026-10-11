# More usage of Swift, please refer to https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html


# Choose the dataset path from the following list, and change the OUTPUT_DIR accordingly:
# data/robomme_preprocessed_data/qwenvl/simple_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl/grounded_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl (full-plan PoC)

# With a held-out split (scripts/split_plan_dataset.py) train on plan_train_split.jsonl and eval on plan_val.jsonl.
# Set VAL_DATASET='' to train on everything without eval.
DATA_DIR='data/robomme_preprocessed_data/qwenvl_plan'
DATASET_PATH=${DATASET_PATH:-$DATA_DIR/plan_train_split.jsonl}
VAL_DATASET=${VAL_DATASET-$DATA_DIR/plan_val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-'runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan'}
EVAL_STEPS=${EVAL_STEPS:-25}

if [ -n "$VAL_DATASET" ]; then
    # eval_loss / eval_token_acc on val every EVAL_STEPS; checkpoints saved at the same steps
    EVAL_ARGS="--val_dataset $VAL_DATASET --eval_strategy steps --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS --per_device_eval_batch_size 4"
else
    EVAL_ARGS="--save_steps 50"
fi
echo "train=$DATASET_PATH val=${VAL_DATASET:-none}"

# GPUs: override from the shell, e.g. `GPU_IDS=0,1 bash scripts/finetune_vlm_plan_predictor.sh`.
# Default: all visible GPUs. NUM_GPUS must equal the number of ids in GPU_IDS.
GPU_IDS=${GPU_IDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')
# Keep the global batch (per_device * grad_accum * NUM_GPUS) at 64 like the 4-GPU original: 4 * GRAD_ACC * NUM_GPUS
GRAD_ACC=${GRAD_ACC:-$((16 / NUM_GPUS))}
echo "GPU_IDS=$GPU_IDS NUM_GPUS=$NUM_GPUS GRAD_ACC=$GRAD_ACC"

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
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    $EVAL_ARGS \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4