#!/usr/bin/env python3
"""Compute text metrics and an LLM-judge score for VQA outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
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


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_row_key(row: Dict[str, Any]) -> str:
    for k in ["id", "uid", "sample_id", "question_id"]:
        if row.get(k):
            return str(row[k])

    segment_id = str(row.get("segment_id", ""))
    question = get_question(row)

    if segment_id or question:
        return f"{segment_id}|||{question}"

    video = str(row.get("video", row.get("video_path", "")))
    return f"{video}|||{question}"


def load_existing_judged(path: str | Path) -> Dict[str, Dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}

    rows = load_jsonl(path)
    out = {}

    for row in rows:
        key = make_row_key(row)
        out[key] = row

    return out


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the latest attempt for each example while preserving input order."""
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for index, row in enumerate(rows):
        key = make_row_key(row) or f"__row_{index}"
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]

def clean_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def get_prediction(row: Dict[str, Any]) -> str:
    """
    Robust prediction extractor.
    """
    for key in [
        "prediction",
        "pred",
        "model_prediction",
        "model_answer",
        "response",
        "output_text",
        "generated_text",
    ]:
        val = row.get(key)
        if val is not None:
            return clean_str(val)

    return ""


def get_reference(row: Dict[str, Any]) -> str:
    """
    Robust GT extractor.
    """
    for key in [
        "gt_answer",
        "reference",
        "ground_truth",
        "target",
        "label",
        "gold_answer",
    ]:
        val = row.get(key)
        if val is not None:
            return clean_str(val)

    conv = row.get("conversations", [])
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            if msg.get("from") in {"gpt", "assistant"}:
                return clean_str(msg.get("value", ""))

    return ""


def get_question(row: Dict[str, Any]) -> str:
    """
    Robust question extractor.
    """
    for key in ["question", "prompt", "query"]:
        val = row.get(key)
        if val is not None:
            return clean_str(val).replace("<video>", "").replace("<image>", "").strip()

    conv = row.get("conversations", [])
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            if msg.get("from") in {"human", "user"}:
                return clean_str(msg.get("value", "")).replace("<video>", "").replace("<image>", "").strip()

    return ""


def get_error(row: Dict[str, Any]) -> Optional[str]:
    err = row.get("error", None)
    if err is None:
        return None

    err_s = clean_str(err)
    if err_s == "" or err_s.lower() in {"none", "null"}:
        return None

    return err_s


def normalize_text(s: str) -> str:
    s = clean_str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: Dict[str, int] = defaultdict(int)
    for tok in ref_tokens:
        ref_counts[tok] += 1

    overlap = 0
    for tok in pred_tokens:
        if ref_counts[tok] > 0:
            overlap += 1
            ref_counts[tok] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)

    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    if not pred_tokens and not ref_tokens:
        return 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = clean_str(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {
        "score": None,
        "reason": f"Could not parse judge JSON. Raw: {text[:500]}",
    }


def build_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return f"""
You are judging a surgical video question-answering model.

Your task:
Decide whether the model prediction is semantically correct with respect to the reference answer.

Use the reference answer as the ground truth.
The prediction does NOT need to match the wording exactly.
Mark correct if it preserves the key clinical/procedural meaning.
Mark incorrect if it contradicts the reference, misses the main point, or adds unsupported important claims.

Return ONLY valid JSON with this schema:
{{
  "score": 1 or 0,
  "reason": "brief explanation"
}}

Question:
{question}

Reference answer:
{reference}

Model prediction:
{prediction}
""".strip()


def call_openai_judge(
    client: Any,
    model: str,
    question: str,
    reference: str,
    prediction: str,
    max_output_tokens: int = 256,
) -> Dict[str, Any]:
    prompt = build_judge_prompt(
        question=question,
        reference=reference,
        prediction=prediction,
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )

    text = getattr(response, "output_text", "") or ""

    if not text:
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    c_text = getattr(c, "text", None)
                    if c_text:
                        parts.append(c_text)
        text = "\n".join(parts)

    parsed = extract_json_from_text(text)

    score = parsed.get("score", None)

    try:
        score = int(score)
    except Exception:
        score = None

    if score not in {0, 1}:
        score = None

    return {
        "llm_judge_score": score,
        "llm_judge_reason": clean_str(parsed.get("reason", "")),
        "llm_judge_model": model,
        "llm_judge_raw": text,
    }


def maybe_run_llm_judge(
    rows: List[Dict[str, Any]],
    judged_jsonl: str,
    judge_model: str,
    judge_sleep: float,
    judge_max_output_tokens: int,
) -> List[Dict[str, Any]]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI package is required for --llm_judge. Install with: pip install openai"
        ) from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before using --llm_judge.")

    client = OpenAI()

    existing = load_existing_judged(judged_jsonl)
    judged_rows: List[Dict[str, Any]] = []

    print(f"[judge] Existing judged rows: {len(existing)}")
    print(f"[judge] Output judged JSONL:  {judged_jsonl}")
    print(f"[judge] Judge model:          {judge_model}")

    for idx, row in enumerate(rows, start=1):
        key = make_row_key(row)

        if key in existing and existing[key].get("llm_judge_score") in {0, 1}:
            judged_rows.append(existing[key])
            continue

        out = dict(row)

        err = get_error(row)
        pred = get_prediction(row)
        ref = get_reference(row)
        question = get_question(row)

        if err is not None:
            out["llm_judge_score"] = None
            out["llm_judge_reason"] = f"Skipped due to row error: {err}"
            out["llm_judge_model"] = judge_model
            append_jsonl(judged_jsonl, out)
            judged_rows.append(out)
            continue

        if not pred:
            out["llm_judge_score"] = 0
            out["llm_judge_reason"] = "Empty prediction."
            out["llm_judge_model"] = judge_model
            append_jsonl(judged_jsonl, out)
            judged_rows.append(out)
            continue

        if not ref:
            out["llm_judge_score"] = None
            out["llm_judge_reason"] = "Missing reference answer."
            out["llm_judge_model"] = judge_model
            append_jsonl(judged_jsonl, out)
            judged_rows.append(out)
            continue

        try:
            judge = call_openai_judge(
                client=client,
                model=judge_model,
                question=question,
                reference=ref,
                prediction=pred,
                max_output_tokens=judge_max_output_tokens,
            )
            out.update(judge)

        except Exception as e:
            out["llm_judge_score"] = None
            out["llm_judge_reason"] = f"Judge error: {repr(e)}"
            out["llm_judge_model"] = judge_model

        append_jsonl(judged_jsonl, out)
        judged_rows.append(out)

        if idx % 25 == 0:
            print(f"[judge] processed {idx}/{len(rows)}")

        if judge_sleep > 0:
            time.sleep(judge_sleep)

    return judged_rows


def init_acc() -> Dict[str, Any]:
    return {
        "n_total": 0,
        "n_success": 0,
        "n_error": 0,
        "n_missing_prediction": 0,
        "n_missing_reference": 0,

        "exact_match_sum": 0.0,
        "token_f1_sum": 0.0,
        "rouge_l_sum": 0.0,

        "llm_judge_n": 0,
        "llm_judge_correct": 0,
        "llm_judge_missing": 0,
    }


def update_acc(acc: Dict[str, Any], row: Dict[str, Any]) -> None:
    acc["n_total"] += 1

    err = get_error(row)
    pred = get_prediction(row)
    ref = get_reference(row)

    if err is not None:
        acc["n_error"] += 1
        return

    if not pred:
        acc["n_missing_prediction"] += 1

    if not ref:
        acc["n_missing_reference"] += 1
        return

    acc["n_success"] += 1

    acc["exact_match_sum"] += exact_match(pred, ref)
    acc["token_f1_sum"] += token_f1(pred, ref)
    acc["rouge_l_sum"] += rouge_l(pred, ref)

    judge_score = row.get("llm_judge_score", None)

    if judge_score in {0, 1}:
        acc["llm_judge_n"] += 1
        acc["llm_judge_correct"] += int(judge_score)
    elif "llm_judge_score" in row:
        acc["llm_judge_missing"] += 1


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def finalize_acc(acc: Dict[str, Any]) -> Dict[str, Any]:
    n = acc["n_success"]

    return {
        "n_total": acc["n_total"],
        "n_success": acc["n_success"],
        "n_error": acc["n_error"],
        "n_missing_prediction": acc["n_missing_prediction"],
        "n_missing_reference": acc["n_missing_reference"],

        "exact_match": safe_div(acc["exact_match_sum"], n),
        "token_f1": safe_div(acc["token_f1_sum"], n),
        "rouge_l": safe_div(acc["rouge_l_sum"], n),

        "llm_judge_n": acc["llm_judge_n"],
        "llm_judge_accuracy": safe_div(acc["llm_judge_correct"], acc["llm_judge_n"]),
        "llm_judge_missing": acc["llm_judge_missing"],
    }


def compute_metrics(
    rows: List[Dict[str, Any]],
    group_by: List[str],
) -> Dict[str, Any]:
    overall = init_acc()

    grouped = {
        field: defaultdict(init_acc)
        for field in group_by
    }

    for row in rows:
        update_acc(overall, row)

        for field in group_by:
            if field in row and row.get(field) is not None:
                value = str(row.get(field))
                update_acc(grouped[field][value], row)

    summary = {
        "overall": finalize_acc(overall),
        "grouped": {},
    }

    for field, group_accs in grouped.items():
        if not group_accs:
            continue

        summary["grouped"][field] = {}

        for value, acc in sorted(
            group_accs.items(),
            key=lambda kv: finalize_acc(kv[1])["n_success"],
            reverse=True,
        ):
            summary["grouped"][field][value] = finalize_acc(acc)

    return summary


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def print_metric_block(title: str, metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"n_total:              {metrics['n_total']}")
    print(f"n_success:            {metrics['n_success']}")
    print(f"n_error:              {metrics['n_error']}")
    print(f"missing prediction:   {metrics['n_missing_prediction']}")
    print(f"missing reference:    {metrics['n_missing_reference']}")
    print("-" * 80)
    print(f"Exact Match:          {fmt(metrics['exact_match'])}")
    print(f"Token F1:             {fmt(metrics['token_f1'])}")
    print(f"ROUGE-L:              {fmt(metrics['rouge_l'])}")
    print(f"LLM-J n:              {metrics['llm_judge_n']}")
    print(f"LLM-J Accuracy:       {fmt(metrics['llm_judge_accuracy'])}")
    print(f"LLM-J missing:        {metrics['llm_judge_missing']}")


def print_summary(summary: Dict[str, Any]) -> None:
    print_metric_block("Overall metrics", summary["overall"])

    for field, groups in summary.get("grouped", {}).items():
        print("\n" + "#" * 80)
        print(f"Grouped by: {field}")
        print("#" * 80)

        for value, metrics in groups.items():
            print(f"\n[{field} = {value}]")
            print(f"  n_success:       {metrics['n_success']}")
            print(f"  exact_match:     {fmt(metrics['exact_match'])}")
            print(f"  token_f1:        {fmt(metrics['token_f1'])}")
            print(f"  rouge_l:         {fmt(metrics['rouge_l'])}")
            print(f"  llm_judge_n:     {metrics['llm_judge_n']}")
            print(f"  llm_judge_acc:   {fmt(metrics['llm_judge_accuracy'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute SurgAtlas VQA metrics from a prediction JSONL."
    )

    p.add_argument("--pred_jsonl", type=str, required=True)

    p.add_argument(
        "--save_summary_json",
        type=str,
        default="",
    )

    p.add_argument(
        "--group_by",
        nargs="*",
        default=[
            "category",
            "broad_category",
            "open_split",
            "merged_specialty",
            "specialty",
            "surgery_type",
            "merged_surgery_type",
            "format",
        ],
    )

    p.add_argument(
        "--llm_judge",
        action="store_true",
        help="Run OpenAI LLM judge before computing metrics.",
    )

    p.add_argument(
        "--judged_jsonl",
        type=str,
        default="",
        help="Where to save/load judged rows. Required if --llm_judge is used.",
    )

    p.add_argument(
        "--judge_model",
        type=str,
        default=os.environ.get("JUDGE_MODEL", "gpt-5.4-nano"),
    )

    p.add_argument(
        "--judge_sleep",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--judge_max_output_tokens",
        type=int,
        default=256,
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    loaded_rows = load_jsonl(args.pred_jsonl)
    rows = deduplicate_rows(loaded_rows)

    print(f"Loaded rows: {len(loaded_rows)}")
    if len(rows) != len(loaded_rows):
        print(f"Unique examples after keeping latest attempts: {len(rows)}")
    print(f"Prediction JSONL: {args.pred_jsonl}")

    if args.llm_judge:
        if not args.judged_jsonl:
            raise ValueError("--judged_jsonl is required when using --llm_judge")

        rows = maybe_run_llm_judge(
            rows=rows,
            judged_jsonl=args.judged_jsonl,
            judge_model=args.judge_model,
            judge_sleep=args.judge_sleep,
            judge_max_output_tokens=args.judge_max_output_tokens,
        )

        metric_source = args.judged_jsonl
    else:
        metric_source = args.pred_jsonl

    summary = compute_metrics(rows, group_by=args.group_by)
    summary["pred_jsonl"] = args.pred_jsonl
    summary["metric_source_jsonl"] = metric_source
    summary["used_llm_judge"] = args.llm_judge
    summary["judge_model"] = args.judge_model if args.llm_judge else None

    print_summary(summary)

    if args.save_summary_json:
        write_json(args.save_summary_json, summary)
        print(f"\nSaved summary JSON to: {args.save_summary_json}")


if __name__ == "__main__":
    main()
