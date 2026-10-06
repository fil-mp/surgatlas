#!/usr/bin/env python3
"""Evaluate Gemini on uniformly sampled SurgAtlas video frames."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types
from tqdm import tqdm


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

            if row.get("error") is None and clean_str(row.get("prediction", "")):
                done.add(make_row_key(row))

    return done


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDERR:\n"
            + p.stderr
        )
    return p.stdout


def get_video_duration(video_path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        str(video_path),
    ]
    out = run_cmd(cmd).strip()
    return float(out)


def extract_even_frames(
    video_path: str | Path,
    out_dir: str | Path,
    num_frames: int = 4,
    jpg_quality: int = 3,
) -> List[Path]:
    """Extract num_frames frames uniformly across the clip.

    Uses midpoint sampling per temporal bin, so for 4 frames the timestamps are
    roughly 12.5%, 37.5%, 62.5%, and 87.5% through the clip. This avoids exact
    first/last frames, which are often black/fades.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    duration = get_video_duration(video_path)

    if duration <= 0:
        raise ValueError(f"Invalid duration for {video_path}: {duration}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = [(i + 0.5) * duration / num_frames for i in range(num_frames)]
    frame_paths: List[Path] = []

    for idx, t in enumerate(times):
        out_path = out_dir / f"frame_{idx:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{t:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", str(jpg_quality),
            str(out_path),
        ]
        run_cmd(cmd)

        if out_path.exists() and out_path.stat().st_size > 0:
            frame_paths.append(out_path)

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frame_paths


def frame_to_part(path: str | Path) -> types.Part:
    return types.Part.from_bytes(
        data=Path(path).read_bytes(),
        mime_type="image/jpeg",
    )


def build_prompt(row: Dict[str, Any], question: str) -> str:
    choices = row.get("choices") or []
    if not isinstance(choices, list):
        choices = []

    prompt = f"""You are evaluating a surgical visual question-answering task.

You are given uniformly sampled frames from a surgical video clip in temporal order.
Answer the question based only on these frames.

Question:
{question}
"""

    if choices:
        prompt += "\nChoices:\n"
        for c in choices:
            prompt += f"- {c}\n"
        prompt += "\nReturn only the best answer choice and a very brief rationale.\n"
    else:
        prompt += "\nReturn a concise answer. Do not mention uncertainty unless the frames are genuinely unclear.\n"

    return prompt


def _enum_name(x: Any) -> str:
    """Return a readable enum/value name from SDK objects."""
    if x is None:
        return ""
    return str(getattr(x, "name", x))


def _jsonable(x: Any) -> Any:
    """Best-effort conversion of SDK response objects to JSON-safe debug info."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    if hasattr(x, "to_json_dict"):
        try:
            return x.to_json_dict()
        except Exception:
            pass
    return repr(x)


def response_debug_dict(response: Any) -> Dict[str, Any]:
    """Small metadata dump so empty predictions are diagnosable."""
    debug: Dict[str, Any] = {}

    try:
        debug["usage_metadata"] = _jsonable(getattr(response, "usage_metadata", None))
    except Exception as e:
        debug["usage_metadata_error"] = repr(e)

    try:
        debug["prompt_feedback"] = _jsonable(getattr(response, "prompt_feedback", None))
    except Exception as e:
        debug["prompt_feedback_error"] = repr(e)

    candidates_debug = []
    try:
        candidates = getattr(response, "candidates", []) or []
        for cand in candidates:
            cand_info: Dict[str, Any] = {
                "finish_reason": _enum_name(getattr(cand, "finish_reason", None)),
                "finish_message": clean_str(getattr(cand, "finish_message", "")),
                "safety_ratings": _jsonable(getattr(cand, "safety_ratings", None)),
            }

            content = getattr(cand, "content", None)
            part_infos = []
            if content is not None:
                for part in getattr(content, "parts", []) or []:
                    part_infos.append(
                        {
                            "has_text": bool(getattr(part, "text", None)),
                            "text_preview": clean_str(getattr(part, "text", ""))[:200],
                            "part_repr_preview": repr(part)[:300],
                        }
                    )
            cand_info["parts"] = part_infos
            candidates_debug.append(cand_info)
    except Exception as e:
        debug["candidates_error"] = repr(e)

    debug["candidates"] = candidates_debug
    return debug


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    try:
        parts = []
        candidates = getattr(response, "candidates", []) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if t:
                    parts.append(str(t))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def call_gemini(
    client: Any,
    model: str,
    prompt: str,
    frame_paths: List[Path],
    max_output_tokens: int,
    temperature: float,
    thinking_budget: int | None,
) -> Tuple[str, Dict[str, Any]]:
    parts: List[types.Part] = [types.Part.from_text(text=prompt)]

    for i, frame_path in enumerate(frame_paths, start=1):
        parts.append(types.Part.from_text(text=f"Frame {i}/{len(frame_paths)}:"))
        parts.append(frame_to_part(frame_path))

    config_kwargs: Dict[str, Any] = {"max_output_tokens": max_output_tokens}

    if temperature is not None:
        config_kwargs["temperature"] = temperature

    # Gemini 2.5/3 models may spend output budget on internal thinking. A low
    # max_output_tokens can therefore produce an empty visible answer. Keep this
    # optional because not every model / SDK version supports it.
    if thinking_budget is not None:
        if not hasattr(types, "ThinkingConfig"):
            raise RuntimeError(
                "This google-genai version does not expose types.ThinkingConfig; "
                "upgrade google-genai or omit --thinking_budget."
            )
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    pred = extract_response_text(response)
    debug = response_debug_dict(response)
    return pred, debug


def safe_dir_name(x: Any, max_len: int = 120) -> str:
    s = clean_str(x) or "row"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:max_len].strip("_") or "row"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Gemini on uniformly sampled SurgAtlas frames."
    )

    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)

    p.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
    )

    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--jpg_quality", type=int, default=3)
    p.add_argument("--max_output_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--thinking_budget", type=int, default=None)
    p.add_argument("--save_response_metadata", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.0)

    p.add_argument(
        "--keep_frames_dir",
        default=None,
        help="Optional directory to keep extracted frames. If omitted, temp frames are deleted.",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_jsonl before running.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY first.")

    if args.overwrite and Path(args.output_jsonl).exists():
        Path(args.output_jsonl).unlink()

    client = genai.Client(api_key=api_key)

    rows = read_jsonl(args.input_jsonl)

    if args.limit is not None:
        rows = rows[: args.limit]

    done_keys = load_done_keys(args.output_jsonl)

    print("=" * 80)
    print("GEMINI FRAME-SAMPLED VQA EVAL")
    print("=" * 80)
    print(f"Input JSONL:       {args.input_jsonl}")
    print(f"Output JSONL:      {args.output_jsonl}")
    print(f"Model:             {args.model}")
    print(f"Rows loaded:       {len(rows)}")
    print(f"Already done:      {len(done_keys)}")
    print(f"Frames per clip:   {args.num_frames}")
    print(f"Max output tokens: {args.max_output_tokens}")
    print(f"Temperature:       {args.temperature}")
    print(f"Thinking budget:   {args.thinking_budget}")
    print("=" * 80)

    for row in tqdm(rows):
        row_key = make_row_key(row)

        if row_key in done_keys:
            continue

        video_path = clean_str(row.get("video", row.get("video_path", "")))
        question, gt_answer = get_question_and_answer(row)

        if not video_path or not Path(video_path).exists():
            out = {
                **row,
                "gemini_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": f"Missing video file: {video_path}",
            }
            append_jsonl(args.output_jsonl, out)
            continue

        if not question:
            out = {
                **row,
                "gemini_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": "Missing question.",
            }
            append_jsonl(args.output_jsonl, out)
            continue

        tmp_dir: Path | None = None

        try:
            if args.keep_frames_dir:
                seg = row.get("segment_id") or row.get("id") or row_key
                tmp_dir = Path(args.keep_frames_dir) / safe_dir_name(seg)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                cleanup = False
            else:
                tmp_dir = Path(tempfile.mkdtemp(prefix="gemini_frames_"))
                cleanup = True

            frame_paths = extract_even_frames(
                video_path=video_path,
                out_dir=tmp_dir,
                num_frames=args.num_frames,
                jpg_quality=args.jpg_quality,
            )

            prompt = build_prompt(row, question)

            pred, response_debug = call_gemini(
                client=client,
                model=args.model,
                prompt=prompt,
                frame_paths=frame_paths,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                thinking_budget=args.thinking_budget,
            )

            out = {
                **row,
                "gemini_model": args.model,
                "prediction": pred,
                "gt_answer": gt_answer,
                "num_frames": len(frame_paths),
                "error": None,
            }

            if not pred:
                raise RuntimeError("Gemini returned an empty response")

            if args.save_response_metadata:
                out["gemini_response_debug"] = response_debug

        except Exception as e:
            out = {
                **row,
                "gemini_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": repr(e),
            }

        finally:
            if tmp_dir is not None and cleanup:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        append_jsonl(args.output_jsonl, out)

        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
