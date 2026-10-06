#!/bin/bash
set -e

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=6

MODEL_PATH="PATH_TO_QWEN3VL_OUTPUTS/stage1_projector/checkpoint-XXXX"
OUTPUT_DIR="PATH_TO_QWEN3VL_OUTPUTS/stage1_step2"
DEEPSPEED_CONFIG="./scripts/zero3.json"
DATASETS="surg_segment_captions,surg_step_descriptions,surg_ocr_phases,surg_video_summaries,surg_title_descriptions"
# DATASETS="surg_segment_captions_open,surg_step_descriptions_open,surg_ocr_phases_open,surg_video_summaries_open,surg_title_descriptions_open"
# DATASETS="surg_segment_captions_mis,surg_step_descriptions_mis,surg_ocr_phases_mis,surg_video_summaries_mis,surg_title_descriptions_mis"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use "${DATASETS}" \
  --data_flatten True \
  --tune_mm_llm True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --mm_projector_lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --model_max_length 8192 \
  --video_fps 2 \
  --video_min_frames 4 \
  --video_max_frames 16 \
  --gradient_checkpointing True \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 10 \
  --report_to none \
  --run_name "stage1_step2"
