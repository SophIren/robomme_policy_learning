# More usage of Swift, please refer to https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html


# Choose the dataset path from the following list, and change the OUTPUT_DIR accordingly:
# data/robomme_preprocessed_data/qwenvl/simple_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl/grounded_subgoal_train.jsonl
# data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl (full-plan PoC)

DATASET_PATH='data/robomme_preprocessed_data/qwenvl_plan/plan_train.jsonl'
OUTPUT_DIR='runs/ckpts/vlm_plan_predictor/qwenvl/grounded_plan'

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
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4