#!/usr/bin/env python3
"""Evaluate an OpenAI VLM on uniformly sampled SurgAtlas frames."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"[warn] skipping malformed line {line_number}: {error}")
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def make_row_key(row: dict[str, Any]) -> str:
    for key in ("id", "uid", "sample_id", "question_id"):
        if row.get(key):
            return str(row[key])

    question, _ = get_question_and_answer(row)
    segment_id = clean_str(row.get("segment_id"))
    if segment_id or question:
        return f"{segment_id}|||{question}"
    return clean_str(row.get("video", row.get("video_path")))


def load_done_keys(path: str | Path) -> set[str]:
    done: set[str] = set()
    if not Path(path).exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if row.get("error") is None and clean_str(row.get("prediction")):
                done.add(make_row_key(row))

    return done


def get_question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    question = ""
    answer = ""

    for msg in row.get("conversations", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("from") in {"human", "user"} and not question:
            question = clean_str(msg.get("value"))
        elif msg.get("from") in {"gpt", "assistant"} and not answer:
            answer = clean_str(msg.get("value"))

    if not question:
        question = clean_str(row.get("question", row.get("prompt")))
    if not answer:
        answer = clean_str(
            row.get("gt_answer", row.get("reference", row.get("answer")))
        )

    return question.replace("<video>", "").replace("<image>", "").strip(), answer


def run_cmd(cmd):
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


def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1",
        video_path,
    ]
    out = run_cmd(cmd).strip()
    return float(out)


def extract_even_frames(video_path, out_dir, num_frames=4, jpg_quality=3):
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    duration = get_video_duration(video_path)

    if duration <= 0:
        raise ValueError(f"Invalid duration for {video_path}: {duration}")

    times = [(i + 0.5) * duration / num_frames for i in range(num_frames)]

    frame_paths = []

    for idx, t in enumerate(times):
        out_path = Path(out_dir) / f"frame_{idx:03d}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{t:.3f}",
            "-i", video_path,
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


def image_to_data_url(path):
    b = Path(path).read_bytes()
    enc = base64.b64encode(b).decode("utf-8")
    return f"data:image/jpeg;base64,{enc}"


def build_prompt(row, question):
    choices = row.get("choices") or []

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


def call_openai(client, model, prompt, frame_paths, max_output_tokens):
    content = [{"type": "input_text", "text": prompt}]

    for p in frame_paths:
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(p),
        })

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={"verbosity": "low"},
        max_output_tokens=max_output_tokens,
    )

    pred = getattr(response, "output_text", "") or ""

    if not pred:
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    text = getattr(c, "text", None)
                    if text:
                        parts.append(text)
        pred = "\n".join(parts).strip()

    return pred.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI VLM on SurgAtlas videos."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.1-2025-11-13")
    )
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--keep_frames_dir", default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete the output before running."
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first.")

    client = OpenAI()

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.overwrite and Path(args.output_jsonl).exists():
        Path(args.output_jsonl).unlink()

    done = load_done_keys(args.output_jsonl)

    print(f"[INFO] Rows loaded: {len(rows)}")
    print(f"[INFO] Already done: {len(done)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Frames per clip: {args.num_frames}")

    for row_index, row in enumerate(tqdm(rows)):
        row_key = make_row_key(row)
        if row_key in done:
            continue

        video_path = clean_str(row.get("video", row.get("video_path")))
        question, gt_answer = get_question_and_answer(row)

        if not video_path or not Path(video_path).exists():
            out = {
                **row,
                "openai_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": f"Missing video file: {video_path}",
            }
            append_jsonl(out, args.output_jsonl)
            continue

        if not question:
            out = {
                **row,
                "openai_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": "Missing human question in conversations.",
            }
            append_jsonl(out, args.output_jsonl)
            continue

        tmp_dir = None

        try:
            if args.keep_frames_dir:
                tmp_dir = Path(args.keep_frames_dir) / f"sample_{row_index:06d}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                cleanup = False
            else:
                tmp_dir = Path(tempfile.mkdtemp(prefix="openai_frames_"))
                cleanup = True

            frame_paths = extract_even_frames(
                video_path=video_path,
                out_dir=tmp_dir,
                num_frames=args.num_frames,
            )

            prompt = build_prompt(row, question)

            pred = call_openai(
                client=client,
                model=args.model,
                prompt=prompt,
                frame_paths=frame_paths,
                max_output_tokens=args.max_output_tokens,
            )

            if not pred:
                raise RuntimeError("OpenAI returned an empty response")

            out = {
                **row,
                "openai_model": args.model,
                "prediction": pred,
                "gt_answer": gt_answer,
                "num_frames": len(frame_paths),
                "error": None,
            }
            append_jsonl(out, args.output_jsonl)

        except Exception as e:
            out = {
                **row,
                "openai_model": args.model,
                "prediction": None,
                "gt_answer": gt_answer,
                "num_frames": args.num_frames,
                "error": repr(e),
            }
            append_jsonl(out, args.output_jsonl)

        finally:
            if tmp_dir is not None and not args.keep_frames_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
