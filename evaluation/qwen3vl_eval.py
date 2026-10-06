#!/usr/bin/env python3
"""Run Qwen3-VL inference on SurgAtlas video-question JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] skipping malformed line {line_idx}: {e}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def get_question_and_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    question = ""
    answer = ""

    conv = row.get("conversations", [])

    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue

            role = msg.get("from")
            value = clean_str(msg.get("value", ""))

            if role in {"human", "user"} and not question:
                question = value

            if role in {"gpt", "assistant"} and not answer:
                answer = value

    if not question:
        question = clean_str(row.get("question", row.get("prompt", "")))

    if not answer:
        answer = clean_str(
            row.get(
                "gt_answer",
                row.get("reference", row.get("answer", "")),
            )
        )

    question = question.replace("<video>", "").replace("<image>", "").strip()

    return question, answer


def make_row_key(row: Dict[str, Any]) -> str:
    for k in ["id", "uid", "sample_id", "question_id"]:
        if row.get(k):
            return str(row[k])

    segment_id = clean_str(row.get("segment_id", ""))
    question, _ = get_question_and_answer(row)

    if segment_id or question:
        return f"{segment_id}|||{question}"

    return clean_str(row.get("video", row.get("video_path", "")))


def load_done_keys(path: str | Path) -> set[str]:
    """Return keys for successful rows so failed examples can be retried."""
    path = Path(path)

    if not path.exists():
        return set()

    done: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue

            if row.get("error") is None and clean_str(row.get("prediction")):
                done.add(make_row_key(row))

    return done


def get_rank_world_size() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    return rank, local_rank, world_size


def select_rows_for_rank(
    rows: List[Dict[str, Any]],
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return rows

    return rows[rank::world_size]


def maybe_add_rank_suffix(
    output_jsonl: str,
    rank: int,
    world_size: int,
    enabled: bool,
) -> str:
    if not enabled or world_size <= 1:
        return output_jsonl

    p = Path(output_jsonl)

    if p.suffix == ".jsonl":
        return str(p.with_name(f"{p.stem}.rank{rank}.jsonl"))

    return str(p) + f".rank{rank}.jsonl"


def merged_output_path(output_jsonl: str) -> str:
    p = Path(output_jsonl)

    if p.suffix == ".jsonl":
        return str(p.with_name(f"{p.stem}.merged.jsonl"))

    return str(p) + ".merged.jsonl"


def rank_output_path(base_output_jsonl: str, rank: int) -> Path:
    base = Path(base_output_jsonl)

    if base.suffix == ".jsonl":
        return base.with_name(f"{base.stem}.rank{rank}.jsonl")

    return Path(str(base) + f".rank{rank}.jsonl")


def merge_rank_outputs(base_output_jsonl: str, world_size: int) -> str:
    merged_path = merged_output_path(base_output_jsonl)
    rank_paths = [rank_output_path(base_output_jsonl, r) for r in range(world_size)]

    missing = [str(p) for p in rank_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot merge rank outputs because these files are missing:\n"
            + "\n".join(missing)
        )

    total_lines = 0

    with open(merged_path, "w", encoding="utf-8") as fout:
        for rp in rank_paths:
            with open(rp, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    fout.write(line + "\n")
                    total_lines += 1

    print("=" * 80)
    print("MERGED RANK OUTPUTS")
    print("=" * 80)
    print(f"Merged path: {merged_path}")
    print(f"Rank files:  {len(rank_paths)}")
    print(f"Total lines: {total_lines}")
    print("=" * 80)

    return merged_path


def build_prompt(question: str, choices: List[str]) -> str:
    if choices:
        choice_text = "\n".join([f"- {c}" for c in choices])
        return f"""Answer the surgical video question based only on the video.

Question:
{question}

Choices:
{choice_text}

Return only the best answer choice and a very brief rationale."""
    else:
        return f"""Answer the surgical video question based only on the video.

Question:
{question}

Return a concise answer."""


def build_messages(
    video_path: str,
    prompt: str,
    video_fps: float,
    video_min_frames: int,
    video_max_frames: int,
    max_pixels: int,
) -> List[Dict[str, Any]]:
    video_item: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "fps": video_fps,
        "min_frames": video_min_frames,
        "max_frames": video_max_frames,
    }

    if max_pixels > 0:
        video_item["max_pixels"] = max_pixels

    messages = [
        {
            "role": "user",
            "content": [
                video_item,
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return messages


def get_first_model_device(model: Any) -> str:
    """
    For a normal model, this returns the model's device.
    For a device_map-sharded model, this returns the first parameter's device.
    Inputs should usually be placed on this device.
    """
    try:
        return str(model.device)
    except Exception:
        pass

    for p in model.parameters():
        return str(p.device)

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_max_memory(
    max_memory_per_gpu: Optional[str],
) -> Optional[Dict[int, str]]:
    if not max_memory_per_gpu:
        return None

    n_visible = torch.cuda.device_count()
    return {i: max_memory_per_gpu for i in range(n_visible)}


@torch.inference_mode()
def run_qwen_one(
    model: Any,
    processor: Any,
    row: Dict[str, Any],
    input_device: str,
    args: argparse.Namespace,
) -> str:
    video_path = clean_str(row.get("video", row.get("video_path", "")))
    question, _ = get_question_and_answer(row)

    choices = row.get("choices") or []
    if not isinstance(choices, list):
        choices = []

    prompt = build_prompt(question, choices)

    messages = build_messages(
        video_path=video_path,
        prompt=prompt,
        video_fps=args.video_fps,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        max_pixels=args.max_pixels,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        video_inputs, video_metadata = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadata = list(video_metadata)
    else:
        video_metadata = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    inputs = inputs.to(input_device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }

    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    generated_ids_trimmed = []
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
        generated_ids_trimmed.append(output_ids[len(input_ids):])

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a Qwen3-VL model or checkpoint on SurgAtlas videos."
    )

    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)

    p.add_argument(
        "--model_path",
        required=True,
        help="HF model name, local base model path, or fine-tuned checkpoint path.",
    )

    p.add_argument(
        "--processor_path",
        default=None,
        help=(
            "Processor/tokenizer path. If not set, uses --model_path. "
            "For fine-tuned checkpoints, usually set this to the base instruct model."
        ),
    )

    p.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )

    p.add_argument("--video_fps", type=float, default=2.0)
    p.add_argument("--video_min_frames", type=int, default=4)
    p.add_argument("--video_max_frames", type=int, default=32)

    p.add_argument(
        "--max_pixels",
        type=int,
        default=0,
        help="Optional max_pixels passed to video item. Use 0 to omit.",
    )

    p.add_argument("--max_new_tokens", type=int, default=128)

    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.0)

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing output_jsonl before running.",
    )

    p.add_argument(
        "--rank_suffix_output",
        action="store_true",
        help=(
            "If set with torchrun, each rank writes to output_jsonl with .rank{RANK}.jsonl suffix. "
            "This avoids multiple ranks appending to the same file."
        ),
    )

    p.add_argument(
        "--merge_ranks_at_end",
        action="store_true",
        help=(
            "If used with --rank_suffix_output, rank 0 merges all rank*.jsonl files "
            "into output_jsonl with .merged.jsonl suffix after all ranks finish."
        ),
    )

    p.add_argument(
        "--device_map",
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help=(
            "Use this for model parallelism. Example: --device_map auto. "
            "When set, run with plain python, not torchrun."
        ),
    )

    p.add_argument(
        "--max_memory_per_gpu",
        default=None,
        help=(
            "Optional memory cap per visible GPU when using --device_map. "
            "Example: --max_memory_per_gpu 75GiB."
        ),
    )

    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained if needed.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rank, local_rank, world_size = get_rank_world_size()

    if args.device_map is not None and world_size > 1:
        raise RuntimeError(
            "You set --device_map for model parallelism, but this script is running under torchrun "
            f"with WORLD_SIZE={world_size}. Do not use torchrun with --device_map. Run with plain python, e.g.:\n\n"
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python qwen3vl_eval.py ... --device_map auto\n"
        )

    if world_size > 1 and not args.rank_suffix_output:
        raise RuntimeError(
            "torchrun requires --rank_suffix_output so ranks do not write to "
            "the same JSONL file."
        )

    if args.merge_ranks_at_end and not args.rank_suffix_output:
        raise ValueError("--merge_ranks_at_end requires --rank_suffix_output")

    output_jsonl = maybe_add_rank_suffix(
        output_jsonl=args.output_jsonl,
        rank=rank,
        world_size=world_size,
        enabled=args.rank_suffix_output,
    )

    if torch.cuda.is_available():
        if args.device_map is None:
            torch.cuda.set_device(local_rank)
            input_device = f"cuda:{local_rank}"
        else:
            input_device = "cuda:0"
    else:
        input_device = "cpu"

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if args.overwrite:
        if args.rank_suffix_output:
            if Path(output_jsonl).exists():
                Path(output_jsonl).unlink()
        else:
            if rank == 0 and Path(output_jsonl).exists():
                Path(output_jsonl).unlink()

    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("PyTorch distributed support is unavailable")
        torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier(device_ids=[local_rank])

    rows = read_jsonl(args.input_jsonl)

    if args.limit is not None:
        rows = rows[: args.limit]

    rows_for_rank = select_rows_for_rank(
        rows=rows,
        rank=rank,
        world_size=world_size,
    )

    done_keys = load_done_keys(output_jsonl)

    processor_path = args.processor_path or args.model_path

    print("=" * 80)
    print("QWEN3-VL VIDEO VQA EVAL")
    print("=" * 80)
    print(f"Input JSONL:          {args.input_jsonl}")
    print(f"Output JSONL:         {output_jsonl}")
    print(f"Model path:           {args.model_path}")
    print(f"Processor path:       {processor_path}")
    print(f"Input device:         {input_device}")
    print(f"Rank/world_size:      {rank}/{world_size}")
    print(f"Total rows:           {len(rows)}")
    print(f"Rows for this rank:   {len(rows_for_rank)}")
    print(f"Already done rows:    {len(done_keys)}")
    print(f"Video fps:            {args.video_fps}")
    print(f"Video frames:         {args.video_min_frames}..{args.video_max_frames}")
    print(f"Max new tokens:       {args.max_new_tokens}")
    print(f"Attention impl:       {args.attn_implementation}")
    print(f"Dtype:                {args.dtype}")
    print(f"Device map:           {args.device_map}")
    print(f"Max memory / GPU:     {args.max_memory_per_gpu}")
    print("=" * 80)

    model_load_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "attn_implementation": args.attn_implementation,
    }

    if args.trust_remote_code:
        model_load_kwargs["trust_remote_code"] = True

    if args.device_map is not None:
        model_load_kwargs["device_map"] = args.device_map

        max_memory = parse_max_memory(args.max_memory_per_gpu)
        if max_memory is not None:
            model_load_kwargs["max_memory"] = max_memory

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            **model_load_kwargs,
        )

        input_device = get_first_model_device(model)

        print("=" * 80)
        print("MODEL-PARALLEL LOAD")
        print("=" * 80)
        print(f"Model device map mode: {args.device_map}")
        print(f"Input tensors moved to: {input_device}")
        if hasattr(model, "hf_device_map"):
            print("HF device map:")
            print(model.hf_device_map)
        print("=" * 80)

    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            **model_load_kwargs,
        ).to(input_device)

    model.eval()

    processor_kwargs: Dict[str, Any] = {}
    if args.trust_remote_code:
        processor_kwargs["trust_remote_code"] = True

    processor = AutoProcessor.from_pretrained(
        processor_path,
        **processor_kwargs,
    )

    for row in tqdm(rows_for_rank, desc=f"rank {rank}"):
        row_key = make_row_key(row)

        if row_key in done_keys:
            continue

        video_path = clean_str(row.get("video", row.get("video_path", "")))
        question, gt_answer = get_question_and_answer(row)

        if not video_path or not Path(video_path).exists():
            out = {
                **row,
                "qwen_model": args.model_path,
                "qwen_processor": processor_path,
                "prediction": None,
                "gt_answer": gt_answer,
                "error": f"Missing video file: {video_path}",
            }
            append_jsonl(output_jsonl, out)
            continue

        if not question:
            out = {
                **row,
                "qwen_model": args.model_path,
                "qwen_processor": processor_path,
                "prediction": None,
                "gt_answer": gt_answer,
                "error": "Missing question.",
            }
            append_jsonl(output_jsonl, out)
            continue

        try:
            pred = run_qwen_one(
                model=model,
                processor=processor,
                row=row,
                input_device=input_device,
                args=args,
            )

            if not pred:
                raise RuntimeError("Qwen3-VL returned an empty response")

            out = {
                **row,
                "qwen_model": args.model_path,
                "qwen_processor": processor_path,
                "prediction": pred,
                "gt_answer": gt_answer,
                "error": None,
            }

        except Exception as e:
            out = {
                **row,
                "qwen_model": args.model_path,
                "qwen_processor": processor_path,
                "prediction": None,
                "gt_answer": gt_answer,
                "error": repr(e),
            }

        append_jsonl(output_jsonl, out)

        if args.sleep > 0:
            time.sleep(args.sleep)

    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])

        if rank == 0 and args.merge_ranks_at_end:
            merge_rank_outputs(
                base_output_jsonl=args.output_jsonl,
                world_size=world_size,
            )

        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
