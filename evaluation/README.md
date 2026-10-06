# SurgAtlas evaluation

This folder contains the evaluation code used for SurgAtlas video question
answering. It supports Qwen3-VL checkpoints, OpenAI VLMs, Gemini, and
the text metrics reported from their prediction files.

## Setup

Create a Python environment and install the evaluation dependencies:

```bash
cd evaluation
python -m pip install -r requirements.txt
```

The OpenAI and Gemini evaluators also require `ffmpeg` and `ffprobe` on
`PATH`. They sample four frames uniformly from each clip before making an API
request.

## Evaluation data

Download the evaluation annotations from
[filbel/SurgAtlas](https://huggingface.co/datasets/filbel/SurgAtlas):

- `test_vqa_full_open.jsonl`
- `test_vqa_full_mis.jsonl`
- `expert_validated_open.jsonl`
- `expert_validated_mis.jsonl`

The expert-validated files are subsets of the corresponding full evaluation
sets; do not combine them as disjoint examples.

The dataset distributes annotations and YouTube identifiers, not videos.
Download the source videos and create the clips using each annotation's
`start_sec` and `end_sec` timestamps. Save each clip at the path in that row's
`video` field, or update the field to its local path.

Each JSONL row should contain a video path and a Qwen-style conversation:

```json
{
  "video": "/path/to/clip.mp4",
  "segment_id": "example_segment",
  "conversations": [
    {"from": "human", "value": "<video>\nWhat action is being performed?"},
    {"from": "gpt", "value": "The surgeon is dissecting tissue."}
  ]
}
```

Multiple-choice rows may also provide `choices`.

## Qwen3-VL

Evaluate a model or fine-tuned checkpoint on one GPU:

```bash
python qwen3vl_eval.py \
  --input_jsonl /path/to/expert_validated_mis.jsonl \
  --output_jsonl predictions/qwen3vl_surgatlas_mis.jsonl \
  --model_path /path/to/checkpoint \
  --processor_path Qwen/Qwen3-VL-8B-Instruct \
  --video_fps 2 \
  --video_min_frames 4 \
  --video_max_frames 32
```

For data-parallel evaluation, launch one model copy per GPU. Each process
writes its own file; rank 0 merges them after all ranks finish:

```bash
torchrun --nproc_per_node=4 qwen3vl_eval.py \
  --input_jsonl /path/to/expert_validated_mis.jsonl \
  --output_jsonl predictions/qwen3vl_surgatlas_mis.jsonl \
  --model_path /path/to/checkpoint \
  --processor_path Qwen/Qwen3-VL-8B-Instruct \
  --rank_suffix_output \
  --merge_ranks_at_end
```

For a model that must be sharded across GPUs, use a plain Python process with
a device map:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python qwen3vl_eval.py \
  --input_jsonl /path/to/expert_validated_mis.jsonl \
  --output_jsonl predictions/qwen3vl_32b_mis.jsonl \
  --model_path Qwen/Qwen3-VL-32B-Instruct \
  --device_map auto
```

Do not combine `torchrun` and `--device_map`.

## OpenAI and Gemini

The API evaluators use four uniformly sampled frames per clip by default:

```bash
export OPENAI_API_KEY="..."
python gpt_eval.py \
  --input_jsonl /path/to/expert_validated_open.jsonl \
  --output_jsonl predictions/openai_open.jsonl \
  --model gpt-5.1-2025-11-13 \
  --num_frames 4
```

```bash
export GEMINI_API_KEY="..."
python gemini_eval_sampling.py \
  --input_jsonl /path/to/expert_validated_open.jsonl \
  --output_jsonl predictions/gemini_open.jsonl \
  --model gemini-2.5-pro \
  --num_frames 4
```

All evaluators append predictions as they run and resume from successful rows.
Use `--overwrite` to start a new output file. Failed rows remain eligible for a
later retry.

## Metrics

Compute exact match, token F1, and ROUGE-L:

```bash
python compute_eval_metrics.py \
  --pred_jsonl predictions/qwen3vl_surgatlas_mis.merged.jsonl \
  --save_summary_json metrics/qwen3vl_surgatlas_mis.json
```

Compute the reported LLM-judge accuracy:

```bash
export OPENAI_API_KEY="..."
python compute_eval_metrics.py \
  --pred_jsonl predictions/qwen3vl_surgatlas_mis.merged.jsonl \
  --save_summary_json metrics/qwen3vl_surgatlas_mis.json \
  --llm_judge \
  --judge_model gpt-5.4-nano \
  --judged_jsonl predictions_judged/qwen3vl_surgatlas_mis.jsonl
```

`run_metrics.sh` accepts one or more prediction files. It computes text metrics
without paid API calls by default:

```bash
bash run_metrics.sh predictions/*.jsonl
```

Set `LLM_JUDGE=1` to enable the semantic LLM judge. Set `JUDGE_MODEL` to override
the default judge model.
When an evaluator has multiple attempts for one example, the metrics script
uses the latest attempt.
