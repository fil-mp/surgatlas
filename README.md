# Surgical Video Detection Pipeline

## SurgAtlas model training

The [training guide](training/qwen-vl-finetune/README.md) provides Qwen3-VL-8B fine-tuning on
[SurgAtlas](https://huggingface.co/datasets/filbel/SurgAtlas): three captioning
pretraining steps followed by SurgAtlas-only instruction tuning. It includes
the data layout, dependencies, and four stage launch scripts.

## SurgAtlas evaluation

The [evaluation guide](evaluation/README.md) provides inference scripts for
Qwen3-VL checkpoints, OpenAI VLMs, and Gemini, together with exact match,
token F1, ROUGE-L, and LLM-judge metrics. It uses the full and
expert-validated Open and MIS evaluation files released with SurgAtlas.

## Video processing pipeline

This repository contains a set of scripts for processing surgical videos into cleaner, machine-learning-ready artifacts. The pipeline can:

1. **Detect and crop the main surgical procedure window** from raw videos.
2. **Select surgery videos or narration-phase videos from annotation sheets**.
3. **Extract audio and generate Whisper transcripts** with segment, word, sentence, JSON, and VTT outputs.
4. **Generate surgical video question-answer (VQA) examples** from captioned surgical clips.

The code is designed for batch processing local surgical video datasets and producing structured outputs for downstream surgical video understanding, narration analysis, and VLM/VQA dataset creation.

---

## Repository Scripts

| Script | Purpose |
|---|---|
| `cropping.py` | Crops all videos in an input folder to the detected surgical procedure window. |
| `cropping_batch.py` | Batch wrapper around `cropping.py`; crops only videos marked `Crop == Yes` in annotation sheets. |
| `run_pipeline_batch.py` | Processes one video: extracts audio, chunks long audio, transcribes with Whisper, and saves transcript outputs. |
| `run_batch_asr.py` | Batch ASR wrapper; selects rows from annotation spreadsheets where the item is surgery-related surgeon narration, finds matching local videos, and runs transcription. |
| `generate_vqa.py` | Generates grounded surgical VQA examples from QwenVL-style surgical clip caption JSONL files. |

---

## Pipeline Overview

### 1. Crop surgical videos

`cropping.py` detects the likely surgical procedure window by sampling frames and building a surgery-aware saliency map. The saliency map combines:

- red/pink/warm tissue color priors
- saturation
- texture and edge information
- motion
- a soft center prior

The script only writes a cropped video when the detected crop removes a meaningful border or overlay. Otherwise, it skips the video.

### 2. Batch crop selected videos

`cropping_batch.py` reads spreadsheets from `Final_Annotations/`, selects rows where `Crop` is marked `Yes`, looks for matching local videos in `surgery_data/`, and writes cropped videos to:

```text
surgery_data/cropped/
```

It also writes logs and a CSV summary to:

```text
surgery_data/cropped/logs/
```

### 3. Transcribe surgical narration

`run_pipeline_batch.py` defines `SurgeryTranscriptionPipeline`, which processes one video by:

1. extracting audio,
2. chunking audio if needed,
3. transcribing with either OpenAI Whisper API or local Whisper,
4. saving transcript artifacts,
5. writing a timing report,
6. deleting temporary extracted audio/chunks while preserving the original video.

Outputs are saved under:

```text
<OUTPUT_DIR>/narrations/<video_name>/
```

For each video, the pipeline can write:

```text
<video_name>_transcript.txt
<video_name>_transcript_timestamped.txt
<video_name>_transcript_words.txt
<video_name>_transcript_sentences.txt
<video_name>_transcript_detailed.json
<video_name>_transcript.vtt
<video_name>_timing_report.json
```

### 4. Batch transcribe selected narration videos

`run_batch_asr.py` reads annotation sheets from:

```text
/surgery_research/auto_phases/inputs
```

It selects rows where:

```text
Surgery/No Surgery == surgery
Phase Info contains surgeon narration
```

It then finds matching local videos in:

```text
/videos/data
```

The match is done by comparing the spreadsheet `title` value to the local video filename stem.

Already-processed videos are skipped when this file exists:

```text
<OUTPUT_DIR>/narrations/<video_name>/<video_name>_transcript_detailed.json
```

### 5. Generate surgical VQA data

`generate_vqa.py` takes a QwenVL-style JSONL file where each line represents a surgical clip with fields such as:

```json
{
  "video": "path/to/video.mp4",
  "video_start": 0.0,
  "video_end": 5.0,
  "youtube_id": "example_id",
  "segment_id": "example_segment",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "caption describing the surgical clip"}
  ]
}
```

It also reads a procedure metadata JSON keyed by `youtube_id`, for example:

```json
{
  "example_id": {
    "surgery": "robotic",
    "specialty": "General Surgery",
    "procedure": "Robotic TAPP inguinal hernia repair"
  }
}
```

For each selected clip, the script runs a three-stage LLM pipeline:

1. **Plan**: identify the Semantic Grounding Moment (SGM), salient entities, broad categories, and fine-grained categories.
2. **Generate**: create open-ended or multiple-choice surgical QA candidates.
3. **Validate**: check grounding, temporal alignment, answerability, non-triviality, and hallucination risk.

Accepted QA pairs are written as JSONL. Rejected examples can optionally be saved for inspection.

---

## Requirements

### System dependencies

Install FFmpeg so that both `ffmpeg` and `ffprobe` are available on your `PATH`:

```bash
ffmpeg -version
ffprobe -version
```

### Python dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the likely Python dependencies:

```bash
pip install pandas openpyxl opencv-python numpy openai tqdm
```

For local Whisper transcription, also install:

```bash
pip install openai-whisper torch
```

Depending on your project structure, the ASR pipeline also expects project modules such as:

```text
src.settings.config.Config
src.audio.extractor.AudioExtractor
```

Make sure those modules are available in the repository or on `PYTHONPATH`.

---

## Configuration

The transcription pipeline loads settings using:

```python
Config.from_env()
```

At minimum, configure values for:

| Setting | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for OpenAI Whisper API and VQA generation. |
| `output_dir` | Root folder where transcript outputs are written. |
| `whisper_backend` | Use `openai`/API mode or `local` mode, depending on your `Config` implementation. |
| `whisper_model_name` | Local Whisper model name when using local mode. |
| `whisper_language` | Language hint for local Whisper transcription. |

Because the exact environment variable names are defined in your missing `Config` class, check `src/settings/config.py` and set the corresponding variables before running ASR.

Example:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OUTPUT_DIR="/surgery_research/outputs"
export WHISPER_BACKEND="openai"
```

---

## How to Run

### Crop all videos in a folder

```bash
python cropping.py \
  --input_dir ./videos \
  --output_dir ./cropped_videos \
  --recursive \
  --samples 24 \
  --padding-pct 0.04 \
  --min-border-pct 0.025
```

Dry run without writing cropped videos:

```bash
python cropping.py \
  --input_dir ./videos \
  --output_dir ./cropped_videos \
  --recursive \
  --dry-run \
  --print-filter
```

### Crop selected videos from annotation sheets

Expected default folders:

```text
Final_Annotations/      # CSV/XLSX/XLS annotation files
surgery_data/           # local source videos named by id
surgery_data/cropped/   # cropped output videos
```

Run:

```bash
python cropping_batch.py
```

Dry run:

```bash
python cropping_batch.py --dry-run --print-filter
```

### Transcribe a single video

The script currently prints this usage string:

```bash
python -m scripts.run_pipeline <video_path>
```

If running directly from the file in this repository layout, use:

```bash
python run_pipeline_batch.py /path/to/video.mp4
```

Outputs will be written to:

```text
<OUTPUT_DIR>/narrations/<video_name>/
```

### Batch transcribe surgeon narration videos

Expected default folders:

```text
/surgery_research/auto_phases/inputs/   # annotation CSV/XLSX/XLS files
/videos/data/                           # local videos
```

Run:

```bash
python run_batch_asr.py
```

The batch ASR script filters annotation rows to surgery videos whose `Phase Info` contains `surgeon narration`, matches rows to local video files by title, skips already-processed videos, and prints counts for processed, missing, skipped, and failed videos.

### Generate surgical VQA examples

Basic run:

```bash
python generate_vqa.py \
  --input_jsonl /path/to/stage1_segment_caption_subclip_qwen.jsonl \
  --procedure_map_json /path/to/merged.json \
  --output reasoning_vqa.jsonl \
  --rejected_output rejected.jsonl \
  --model gpt-5.4-mini \
  --segments_per_video 7 \
  --question_style mixed \
  --save_rejected
```

Resume a previous run:

```bash
python generate_vqa.py \
  --input_jsonl /path/to/stage1_segment_caption_subclip_qwen.jsonl \
  --procedure_map_json /path/to/merged.json \
  --output reasoning_vqa.jsonl \
  --resume
```

Filter to selected YouTube IDs:

```bash
python generate_vqa.py \
  --input_jsonl /path/to/input.jsonl \
  --procedure_map_json /path/to/merged.json \
  --id_list_txt selected_ids.txt \
  --output selected_reasoning_vqa.jsonl
```

Filter by metadata:

```bash
python generate_vqa.py \
  --input_jsonl /path/to/input.jsonl \
  --procedure_map_json /path/to/merged.json \
  --specialty_filter "General Surgery" \
  --surgery_type_filter Robotic \
  --output robotic_general_surgery_vqa.jsonl
```

Skip validation to reduce LLM calls:

```bash
python generate_vqa.py \
  --input_jsonl /path/to/input.jsonl \
  --procedure_map_json /path/to/merged.json \
  --output reasoning_vqa_unvalidated.jsonl \
  --skip_validator
```

---

## Input Data Expectations

### Annotation sheets for ASR

`run_batch_asr.py` expects columns including:

```text
title
Surgery/No Surgery
Phase Info
```

The local video filename stem must match the `title` field.

### Annotation sheets for cropping batch

`cropping_batch.py` expects columns including:

```text
id
Crop
```

The local video filename stem must match `id`. Values such as `yes`, `y`, `true`, or `1` are treated as crop-positive.

### VQA input JSONL

`generate_vqa.py` expects each line to include:

```text
video
video_start
video_end
segment_id
youtube_id
conversations
```

The GPT turn inside `conversations` is treated as the surgical clip caption.

---

## Output Summary

| Stage | Output |
|---|---|
| Cropping | Cropped videos and crop logs. |
| Batch cropping | Cropped videos, console log, crop log, and `run_summary.csv`. |
| Single-video ASR | Plain transcript, timestamped transcript, word transcript, sentence transcript, detailed JSON, VTT subtitles, timing report. |
| Batch ASR | Same ASR outputs for each selected narration video. |
| VQA generation | Accepted QA JSONL, optional rejected-candidates JSONL, chat-formatted `messages` field for SFT-style use. |

---

## Practical Notes

- Several batch scripts use hardcoded default directories. Edit the constants near the top of each script if your dataset lives elsewhere.
- Run `cropping.py --dry-run --print-filter` first when testing crop settings on a new dataset.
- Use `--resume` in `generate_vqa.py` for long-running VQA generation jobs.
- Use `--save_rejected` during development to inspect why VQA candidates were rejected.
- Local Whisper can require a GPU and enough memory depending on the model size. The current local backend uses CUDA device `cuda:1` when more than one CUDA device is available, otherwise CPU.

---

## Suggested End-to-End Workflow

```bash
# 1. Crop selected videos from annotation sheets
python cropping_batch.py --print-filter

# 2. Transcribe narration-phase surgical videos
python run_batch_asr.py

# 3. Generate VQA from existing surgical clip captions
python generate_vqa.py \
  --input_jsonl /path/to/stage1_segment_caption_subclip_qwen.jsonl \
  --procedure_map_json /path/to/merged.json \
  --output reasoning_vqa.jsonl \
  --save_rejected \
  --resume
```

This produces cropped videos, timestamped surgical narration transcripts, and structured surgical VQA examples for downstream model training or evaluation.
