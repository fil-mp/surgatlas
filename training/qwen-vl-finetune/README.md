# SurgAtlas training

This folder contains the code used to fine-tune
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
for SurgAtlas. Training has three caption-pretraining steps followed by
SurgAtlas-only instruction tuning.

## Setup

Install the dependencies in requirements.txt and FlashAttention 2 in a Linux
environment with NVIDIA GPUs:

    python -m pip install -r requirements.txt
    python -m pip install flash-attn==2.7.4.post1 --no-build-isolation

## Data

Download the annotation JSONL files from
[filbel/SurgAtlas](https://huggingface.co/datasets/filbel/SurgAtlas). Set
SURGATLAS_ANNOTATION_DIR in qwenvl/data/__init__.py to the directory containing
those files.

The dataset release contains annotations and YouTube identifiers, not video
files. Download the source videos and create the training clips using the
start_sec and end_sec timestamps provided in the annotations. Save each clip at
the location recorded in that example's video field, or update that field to
the corresponding local path before training.

The four stages use these released files:

| Stage | Annotation files |
|---|---|
| Stage 1, steps 1–2 | captions_train.jsonl, steps_train.jsonl, ocr_phases_train.jsonl, summaries.jsonl, metadata.jsonl |
| Stage 1, step 3 | captions_train.jsonl, steps_train.jsonl, ocr_phases_train.jsonl, summaries.jsonl |
| Stage 2 | train_vqa.jsonl, ocr_phases_vqa_train.jsonl |

## Training

Edit MODEL_PATH and OUTPUT_DIR in each script for your system, then run the
stages in order from this directory:

    bash scripts/stage1_projector.sh
    bash scripts/stage1_step2.sh
    bash scripts/stage1_step3.sh
    bash scripts/stage2.sh

Each later script should point MODEL_PATH to the desired checkpoint-XXXX
directory from the preceding stage. Keep qwen3 in the checkpoint path because
the original Qwen training entrypoint uses the path name to select the model
class. The scripts use six GPUs by default; change NPROC_PER_NODE as needed.
Commented DATASETS lines in each script select the Open-only or MIS-only
SurgAtlas annotations used for the corresponding ablations.

The training framework is based on
[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). Its Apache-2.0 license is included in
this folder.
