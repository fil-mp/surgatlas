#!/usr/bin/env python3
"""
This version uses a QwenVL-style JSONL file as input, where each line is a
surgical clip record with:
  - video path
  - video_start / video_end
  - youtube_id
  - segment_id
  - conversations, where the GPT turn contains the surgical clip description

It also reads a separate procedure mapping JSON keyed by youtube_id, e.g.:
{
  "_1_Ws6nmFJE": {
    "surgery": "robotic",
    "specialty": "General Surgery",
    "procedure": "Robotic TAPP inguinal hernia repair"
  }
}

If procedure/specialty/surgery is "unknown", it is treated as empty.

Pipeline:
  1) PLAN      -> choose the most applicable broad and fine-grained question
                  categories and summarize the Semantic Grounding Moment (SGM)
  2) GENERATE  -> create structured QA candidates (open or MCQ)
  3) VALIDATE  -> verify grounding, temporal alignment, answerability,
                  non-triviality, and hallucination risk (optional)

Output:
  - JSONL with one accepted QA per line
  - Optional rejected-candidates JSONL
  - Optional chat-formatted samples for VLM SFT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# TWO-LEVEL TAXONOMY
# ============================================================

BROAD_CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "perception_identification": {
        "description": "Recognizing, characterizing, or locating anatomy, tools, devices, findings, or scene elements visible in the current clip.",
        "subcategories": [
            "entity_existence",
            "entity_state",
            "spatial_relation",
        ],
    },
    "action_procedural_state": {
        "description": "Describing what is being done in the current clip, including instrument-tissue interaction and the local procedural step.",
        "subcategories": [
            "instrument_tissue_interaction",
            "operative_action",
        ],
    },
    "operative_reasoning": {
        "description": "Explaining why an action or operative choice is being made at this moment.",
        "subcategories": [
            "maneuver_rationale",
            "decision_justification",
        ],
    },
    "temporal_predictive": {
        "description": "Reasoning about sequence, short-term procedural context, or what is likely to happen next.",
        "subcategories": [
            "procedural_sequence",
            "next_step_prediction",
        ],
    },
    "risk_anatomy_identification": {
        "description": "Identifying anatomical structures requiring protection and explicit safety practices in the current clip.",
        "subcategories": [
            "risk_anatomy_identification",
        ],
    },
}

CATEGORY_SPECS: Dict[str, Dict[str, Any]] = {
    "entity_existence": {
        "description": "Whether a tool, anatomy, device, or finding is present in the current scene.",
        "examples": [
            "Is the cystic duct clearly identified in this moment?",
            "Is a clip applier currently being used?"
        ],
    },
    "entity_state": {
        "description": "Clinically meaningful state of an entity (anatomy or instrument), such as exposure, dissection plane, tension, or instrument-state transitions like clip deployment or energy activation.",
        "examples": [
            "What is notable about the exposure of the vessel here?",
            "How is the tissue being presented at this point?"
        ],
    },
    "spatial_relation": {
        "description": "Where an entity is relative to another structure, quadrant, side, or operative field.",
        "examples": [
            "Where is the stapler being positioned relative to the vessel?",
            "Where is the lesion located in the operative field?"
        ],
    },
    "instrument_tissue_interaction": {
        "description": "How a named instrument is being used and on what tissue or anatomical target.",
        "examples": [
            "What is the bipolar being used to do here?",
            "How is the grasper interacting with the tissue in this step?"
        ],
    },
    "operative_action": {
        "description": "Description of the action being performed, ranging from generic visual action (dissection, cauterization, suturing) to procedure-specific maneuvers when context is sufficient.",
        "examples": [
            "What action is occurring in this segment?",
            "What operative step is being performed here?"
        ],
    },
    "maneuver_rationale": {
        "description": "Immediate clinical rationale for the current maneuver at this local moment.",
        "examples": [
            "Why is this structure being retracted in this direction?",
            "Why is the surgeon dissecting in this plane now?"
        ],
    },
    "decision_justification": {
        "description": "Higher-level operative reasoning or decision based on anatomy, exposure, or risk.",
        "examples": [
            "Why is this approach chosen instead of proceeding directly?",
            "What operative consideration explains this choice?"
        ],
    },
    "procedural_sequence": {
        "description": "Reasoning across a short temporal window: summarizing what has just been accomplished or ordering events that span the clip and its neighbors.",
        "examples": [
            "What sequence of actions is being completed across this moment?",
            "What likely happened immediately before this step?"
        ],
    },
    "next_step_prediction": {
        "description": "What is most likely to happen next if the operation proceeds normally.",
        "examples": [
            "What is the most likely next step after this maneuver?",
            "What would the surgeon typically do next?"
        ],
    },
    "risk_anatomy_identification": {
        "description": "Critical anatomical structures requiring protection or explicit safety practice (e.g., critical view of safety) at this moment.",
        "examples": [
            "What safety principle is important in this moment?",
            "Which structure needs to be protected here and why?"
        ],
    },
}

CATEGORY_NAMES = list(CATEGORY_SPECS.keys())
BROAD_CATEGORY_NAMES = list(BROAD_CATEGORY_SPECS.keys())

CATEGORY_TO_BROAD: Dict[str, str] = {}
for broad_name, spec in BROAD_CATEGORY_SPECS.items():
    for subcat in spec["subcategories"]:
        CATEGORY_TO_BROAD[subcat] = broad_name


# ============================================================
# PROMPTS
# ============================================================

PLANNER_SYSTEM = """You are a surgical VQA dataset planner.
Your job is to identify the most suitable broad and fine-grained question
categories for a surgical clip description and summarize the semantic
grounding moment (SGM).

You must be conservative:
- Only choose categories that are actually supported by the current caption/context.
- Prefer categories that refer to the present local scene.
- Avoid categories that would require guessing hidden patient details or unsupported facts.
- If the caption is too vague, choose fewer categories.
- Do not assume details that are not stated in the current clip description or neighboring clip descriptions.
- Fine-grained categories must be consistent with the selected broad categories.

Return ONLY valid JSON matching the requested schema.
"""

GENERATOR_SYSTEM = """You are generating high-quality VQA examples for surgical video learning.

Generate question-answer pairs that are:
- grounded in the provided clip description/context
- temporally local to the current clip
- educational but not speculative
- concise and clinically plausible
- answerable from the current clip description, not broad textbook recall alone

Be conservative:
- If a category is weakly supported, skip it.
- Do not invent patient details, complications, or anatomy not supported by the context.
- Questions should sound natural, not templated.

Return ONLY valid JSON.
"""

VALIDATOR_SYSTEM = """You are validating surgical VQA candidates.

Evaluate each candidate using these standards:
1. groundedness: supported by the clip description/context
2. temporal_alignment: answer refers to the current clip moment
3. answerability: question is reasonably answerable from this moment
4. non_triviality: not too obvious or empty
5. hallucination_risk: low risk of unsupported surgical claims

Reject candidates that are generic, speculative, duplicated, or weakly grounded.
You may lightly revise the answer for clarity while keeping meaning unchanged.

Return ONLY valid JSON.
"""


def build_planner_prompt(
    surgery_type: str,
    current_text: str,
    context_text: str,
    max_categories: int,
    question_style: str,
) -> str:
    broad_block = []
    for broad_name, broad_spec in BROAD_CATEGORY_SPECS.items():
        subcats = ", ".join(broad_spec["subcategories"])
        broad_block.append(
            f"- {broad_name}: {broad_spec['description']} Subcategories: {subcats}"
        )

    fine_block = []
    for name, spec in CATEGORY_SPECS.items():
        eg = spec["examples"][0]
        fine_block.append(f"- {name}: {spec['description']} Example: {eg}")

    surgery_line = f"Surgery type: {surgery_type}\n" if surgery_type else ""

    return f"""
{surgery_line}Question style requested: {question_style}

Current surgical clip description:
{json.dumps(current_text, ensure_ascii=False)}

Local context (neighboring clip descriptions):
{json.dumps(context_text, ensure_ascii=False)}

Broad categories:
{chr(10).join(broad_block)}

Fine-grained categories:
{chr(10).join(fine_block)}

Return a JSON object with exactly these keys:
{{
  "sgm_summary": string,
  "salient_entities": [string, ...],
  "applicable_broad_categories": [string, ...],
  "applicable_categories": [string, ...],
  "notes_for_generator": [string, ...]
}}

Rules:
- Choose at most {max_categories} fine-grained categories.
- Fine-grained categories must be consistent with the selected broad categories.
- Prefer present-scene categories over broad/general ones.
- If next_step_prediction or procedural_sequence is selected, it must still be inferable from the context.
""".strip()


def build_generator_prompt(
    surgery_type: str,
    current_text: str,
    context_text: str,
    plan: Dict[str, Any],
    question_style: str,
) -> str:
    style_rules = {
        "open": "Generate only open-ended QA pairs. choices must be [] and correct_choice must be null.",
        "mcq": "Generate multiple-choice QA pairs. Provide 4 answer choices and set correct_choice to 0-based index.",
        "mixed": "Use whichever format best fits each category. Open questions are allowed; MCQ is allowed when distractors can be made safely."
    }[question_style]

    surgery_line = f"Surgery type: {surgery_type}\n\n" if surgery_type else ""

    return f"""
{surgery_line}Current surgical clip description:
{json.dumps(current_text, ensure_ascii=False)}

Local context:
{json.dumps(context_text, ensure_ascii=False)}

Planner output:
{json.dumps(plan, ensure_ascii=False, indent=2)}

{style_rules}

Return a JSON array of objects. Generate at most 1 item per category in plan["applicable_categories"].

Each object must have exactly these keys:
{{
  "category": string,
  "question": string,
  "answer": string,
  "rationale": string,
  "evidence": string,
  "format": "open" | "mcq",
  "choices": [string, ...],
  "correct_choice": integer | null
}}

Rules:
- Answers should usually be 1-2 sentences.
- Do not generate unsupported patient factors.
- Avoid asking about visibility if the clip description gives no clue.
- MCQ distractors must be plausible but clearly wrong from the given context.
- Make questions natural and varied.
""".strip()


def build_validator_prompt(
    surgery_type: str,
    current_text: str,
    context_text: str,
    qa_item: Dict[str, Any],
) -> str:
    surgery_line = f"Surgery type: {surgery_type}\n\n" if surgery_type else ""

    return f"""
{surgery_line}Current surgical clip description:
{json.dumps(current_text, ensure_ascii=False)}

Local context:
{json.dumps(context_text, ensure_ascii=False)}

Candidate QA:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return a JSON object with exactly these keys:
{{
  "keep": true | false,
  "reason": string,
  "scores": {{
    "groundedness": integer,
    "temporal_alignment": integer,
    "answerability": integer,
    "non_triviality": integer,
    "hallucination_risk": integer
  }},
  "revised_answer": string,
  "revised_question": string,
  "dedup_key_hint": string
}}

Scoring rules:
- Each score is 1-5.
- hallucination_risk = 5 means very low hallucination risk.
- Reject if the item depends on assumptions not present in the clip description/context.
- Reject if the question could be asked of almost any surgery clip.
""".strip()


# ============================================================
# UTILS
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def extract_json_substring(text: str) -> str:
    text = strip_code_fences(text)

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        return text

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidate = text[first_arr:last_arr + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidate = text[first_obj:last_obj + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return text


def parse_json_response(text: str) -> Any:
    candidate = extract_json_substring(text)
    return json.loads(candidate)


def safe_sleep(seconds: float) -> None:
    time.sleep(seconds)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def record_uid(video_path: str, start_sec: float, end_sec: float, category: str, question: str, answer: str) -> str:
    key = f"{video_path}|{start_sec:.3f}|{end_sec:.3f}|{category}|{clean_text(question)}|{clean_text(answer)}"
    return sha1_text(key)


def choose_progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def default_chat_messages(question: str, answer: str, include_rationale: bool, rationale: str) -> List[Dict[str, str]]:
    assistant = answer.strip()
    if include_rationale and rationale.strip():
        assistant = f"{assistant}\n\nReasoning: {rationale.strip()}"
    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant},
    ]


def load_id_filter_txt(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ID filter txt not found: {p}")

    ids: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            vid = line.strip()
            if vid:
                ids.add(vid)
    return ids


def parse_filter_values(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        parts = [x.strip() for x in v.split(",")]
        out.extend([x for x in parts if x])
    return out


def normalize_surgery_type(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""

    low = v.lower()
    mapping = {
        "robotic": "Robotic",
        "laparoscopic": "Laparoscopic",
        "open": "Open",
        "open surgery": "Open",
        "other": "Other",
    }
    return mapping.get(low, v)


def load_type_map_from_excel_dir(excel_dir: str | Path) -> Dict[str, str]:
    excel_dir = Path(excel_dir)
    if not excel_dir.exists():
        raise FileNotFoundError(f"Excel dir not found: {excel_dir}")
    if not excel_dir.is_dir():
        raise ValueError(f"Expected a directory, got: {excel_dir}")

    excel_paths = sorted(
        list(excel_dir.glob("*.xlsx")) +
        list(excel_dir.glob("*.xls"))
    )

    if not excel_paths:
        raise ValueError(f"No Excel files found in: {excel_dir}")

    type_map: Dict[str, str] = {}

    for excel_path in excel_paths:
        df = pd.read_excel(excel_path)

        id_col = "id"
        type_col = "Type"

        missing = [c for c in [id_col, type_col] if c not in df.columns]
        if missing:
            print(f"[warn] skipping {excel_path} because columns are missing: {missing}")
            continue

        for _, row in df.iterrows():
            youtube_id = str(row[id_col]).strip()
            if not youtube_id or youtube_id.lower() == "nan":
                continue

            raw_type = ""
            if pd.notna(row[type_col]):
                raw_type = str(row[type_col]).strip()

            if raw_type.lower() in {"n/a", "na", "unknown"}:
                raw_type = ""

            norm_type = normalize_surgery_type(raw_type)
            if not norm_type:
                continue

            if youtube_id not in type_map:
                type_map[youtube_id] = norm_type

    return type_map


def get_surgery_type_for_filter(
    youtube_id: str,
    procedure_map: Dict[str, Dict[str, Any]],
    type_source: str,
    type_map_from_excel: Dict[str, str],
) -> str:
    if type_source == "excel":
        return normalize_surgery_type(type_map_from_excel.get(youtube_id, ""))

    meta = procedure_map.get(youtube_id, {})
    if not isinstance(meta, dict):
        return ""

    return normalize_surgery_type(meta.get("surgery", ""))


def matches_metadata_filters(
    youtube_id: str,
    procedure_map: Dict[str, Dict[str, Any]],
    procedure_filters: List[str],
    specialty_filters: List[str],
    surgery_type_filters: List[str],
    type_source: str,
    type_map_from_excel: Dict[str, str],
) -> bool:
    if not procedure_filters and not specialty_filters and not surgery_type_filters:
        return True

    meta = procedure_map.get(youtube_id, {})
    if not isinstance(meta, dict):
        meta = {}

    procedure_val = str(meta.get("procedure", "") or "").strip()
    specialty_val = str(meta.get("specialty", "") or "").strip()
    surgery_type_val = get_surgery_type_for_filter(
        youtube_id=youtube_id,
        procedure_map=procedure_map,
        type_source=type_source,
        type_map_from_excel=type_map_from_excel,
    )

    if procedure_val.lower() == "unknown":
        procedure_val = ""
    if specialty_val.lower() == "unknown":
        specialty_val = ""

    if procedure_filters and procedure_val not in procedure_filters:
        return False

    if specialty_filters and specialty_val not in specialty_filters:
        return False

    if surgery_type_filters:
        normalized_requested = [normalize_surgery_type(x) for x in surgery_type_filters]
        if surgery_type_val not in normalized_requested:
            return False

    return True


# ============================================================
# PROCEDURE MAP
# ============================================================

def load_procedure_map(path: str | Path) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Procedure map JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Procedure map JSON must be a top-level object keyed by youtube_id.")
    return data


def resolve_surgery_type(youtube_id: str, procedure_map: Dict[str, Dict[str, Any]]) -> str:
    meta = procedure_map.get(youtube_id, {})
    if not isinstance(meta, dict):
        return ""

    for key in ("procedure", "procedure_name", "title", "specialty"):
        value = meta.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value and value.lower() != "unknown":
                return value

    return ""


# ============================================================
# QWENVL INPUT LOADING
# ============================================================

def extract_gpt_caption(conversations: List[Dict[str, Any]]) -> str:
    for turn in conversations:
        role = str(turn.get("from", "")).strip().lower()
        if role == "gpt":
            return clean_text(str(turn.get("value", "")))
    return ""


def load_qwenvl_jsonl(path: str | Path, min_caption_len: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[warn] skipping malformed JSONL line {line_idx}: {e}")
                continue

            try:
                video_path = str(item["video"]).strip()
                start_sec = float(item["video_start"])
                end_sec = float(item["video_end"])
                segment_id = str(item["segment_id"]).strip()
            except Exception as e:
                print(f"[warn] skipping line {line_idx} due to missing required fields: {e}")
                continue

            youtube_id = str(item.get("youtube_id", "")).strip()
            if not youtube_id:
                youtube_id = Path(video_path).stem

            conversations = item.get("conversations", [])
            if not isinstance(conversations, list):
                conversations = []

            caption = extract_gpt_caption(conversations)
            if not caption or len(caption) < min_caption_len:
                continue

            records.append({
                "video_path": video_path,
                "youtube_id": youtube_id,
                "segment_id": segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(max(0.0, end_sec - start_sec), 3),
                "caption": caption,
                "raw_item": item,
            })

    return records


def group_qwenvl_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = rec["youtube_id"] or Path(rec["video_path"]).stem
        grouped.setdefault(key, []).append(rec)

    for key in grouped:
        grouped[key].sort(key=lambda x: (x["start_sec"], x["end_sec"], x["segment_id"]))

    return grouped


def build_caption_context(records: List[Dict[str, Any]], idx: int, radius: int = 1) -> Tuple[str, str]:
    current = clean_text(records[idx].get("caption", ""))
    neighbors = []

    start = max(0, idx - radius)
    end = min(len(records), idx + radius + 1)

    for j in range(start, end):
        if j == idx:
            continue
        txt = clean_text(records[j].get("caption", ""))
        if not txt:
            continue
        prefix = "PREV" if j < idx else "NEXT"
        neighbors.append(f"{prefix}: {txt}")

    return current, " | ".join(neighbors)


# ============================================================
# OPENAI WRAPPERS
# ============================================================

_client = OpenAI()


def call_openai(prompt: str, system: str, model: str) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def call_llm_json(
    prompt: str,
    system: str,
    model: str,
    max_retries: int = 3,
    initial_backoff: float = 1.5,
) -> Any:
    last_err = None
    for attempt in range(max_retries):
        try:
            raw = call_openai(prompt, system, model)
            return parse_json_response(raw)
        except Exception as e:
            last_err = e
            wait_s = initial_backoff * (2 ** attempt)
            print(f"[warn] LLM call failed on attempt {attempt + 1}/{max_retries}: {e}")
            safe_sleep(wait_s)
    raise RuntimeError(f"Failed to obtain valid JSON after {max_retries} attempts: {last_err}")


# ============================================================
# CORE PIPELINE
# ============================================================

def plan_segment(
    current_text: str,
    context_text: str,
    surgery_type: str,
    model: str,
    max_categories: int,
    question_style: str,
) -> Dict[str, Any]:
    prompt = build_planner_prompt(
        surgery_type=surgery_type,
        current_text=current_text,
        context_text=context_text,
        max_categories=max_categories,
        question_style=question_style,
    )
    plan = call_llm_json(prompt, PLANNER_SYSTEM, model)

    if not isinstance(plan, dict):
        raise ValueError("Planner response is not an object.")

    broad = plan.get("applicable_broad_categories", [])
    if not isinstance(broad, list):
        broad = []
    normalized_broad = [b for b in broad if b in BROAD_CATEGORY_NAMES]

    cats = plan.get("applicable_categories", [])
    if not isinstance(cats, list):
        cats = []
    normalized_cats = [c for c in cats if c in CATEGORY_NAMES][:max_categories]

    if not normalized_broad:
        normalized_broad = []
        for c in normalized_cats:
            bc = CATEGORY_TO_BROAD.get(c)
            if bc and bc not in normalized_broad:
                normalized_broad.append(bc)

    return {
        "sgm_summary": clean_text(str(plan.get("sgm_summary", current_text[:180]))),
        "salient_entities": [clean_text(str(x)) for x in plan.get("salient_entities", []) if str(x).strip()],
        "applicable_broad_categories": normalized_broad,
        "applicable_categories": normalized_cats,
        "notes_for_generator": [clean_text(str(x)) for x in plan.get("notes_for_generator", []) if str(x).strip()],
    }


def generate_candidates(
    current_text: str,
    context_text: str,
    surgery_type: str,
    plan: Dict[str, Any],
    model: str,
    question_style: str,
) -> List[Dict[str, Any]]:
    if not plan.get("applicable_categories"):
        return []

    prompt = build_generator_prompt(
        surgery_type=surgery_type,
        current_text=current_text,
        context_text=context_text,
        plan=plan,
        question_style=question_style,
    )
    items = call_llm_json(prompt, GENERATOR_SYSTEM, model)
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    seen_categories = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        category = str(item.get("category", "")).strip()
        question = clean_text(str(item.get("question", "")))
        answer = clean_text(str(item.get("answer", "")))
        rationale = clean_text(str(item.get("rationale", "")))
        evidence = clean_text(str(item.get("evidence", "")))
        fmt = str(item.get("format", "open")).strip().lower()

        if category not in plan["applicable_categories"]:
            continue
        if category in seen_categories:
            continue
        if not question or not answer:
            continue
        if len(answer) < 8 or len(question) < 8:
            continue

        choices = item.get("choices", [])
        correct_choice = item.get("correct_choice", None)

        if fmt not in ("open", "mcq"):
            fmt = "open"

        if fmt == "mcq":
            if not isinstance(choices, list) or len(choices) != 4:
                continue
            choices = [clean_text(str(c)) for c in choices]
            if not isinstance(correct_choice, int) or correct_choice < 0 or correct_choice >= len(choices):
                continue
        else:
            choices = []
            correct_choice = None

        seen_categories.add(category)
        out.append({
            "category": category,
            "question": question,
            "answer": answer,
            "rationale": rationale,
            "evidence": evidence,
            "format": fmt,
            "choices": choices,
            "correct_choice": correct_choice,
        })

    return out


def validate_candidate(
    current_text: str,
    context_text: str,
    surgery_type: str,
    candidate: Dict[str, Any],
    model: str,
    min_score: int,
) -> Dict[str, Any]:
    prompt = build_validator_prompt(
        surgery_type=surgery_type,
        current_text=current_text,
        context_text=context_text,
        qa_item=candidate,
    )
    result = call_llm_json(prompt, VALIDATOR_SYSTEM, model)
    if not isinstance(result, dict):
        return {"keep": False, "reason": "validator_not_dict"}

    scores = result.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    g = int(scores.get("groundedness", 1))
    t = int(scores.get("temporal_alignment", 1))
    a = int(scores.get("answerability", 1))
    n = int(scores.get("non_triviality", 1))
    h = int(scores.get("hallucination_risk", 1))

    keep = bool(result.get("keep", False))
    if min(g, t, a, n, h) < min_score:
        keep = False

    return {
        "keep": keep,
        "reason": clean_text(str(result.get("reason", ""))),
        "scores": {
            "groundedness": g,
            "temporal_alignment": t,
            "answerability": a,
            "non_triviality": n,
            "hallucination_risk": h,
        },
        "revised_answer": clean_text(str(result.get("revised_answer", candidate["answer"]))),
        "revised_question": clean_text(str(result.get("revised_question", candidate["question"]))),
        "dedup_key_hint": clean_text(str(result.get("dedup_key_hint", ""))),
    }


def process_qwenvl_group(
    group_key: str,
    records: List[Dict[str, Any]],
    procedure_map: Dict[str, Dict[str, Any]],
    model: str,
    question_style: str,
    max_segments_per_video: int,
    max_categories_per_segment: int,
    min_validation_score: int,
    include_rationale_in_chat: bool,
    segment_radius: int,
    processed_segment_ids: set[str],
    accepted_record_ids: set[str],
    save_rejected: bool,
    skip_validator: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []

    surgery_type = resolve_surgery_type(group_key, procedure_map)

    eligible_idx = [i for i, rec in enumerate(records) if clean_text(rec.get("caption", ""))]
    if not eligible_idx:
        return [], []

    if max_segments_per_video > 0 and len(eligible_idx) > max_segments_per_video:
        chosen_idx = random.sample(eligible_idx, max_segments_per_video)
        chosen_idx.sort()
    else:
        chosen_idx = eligible_idx

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for idx in chosen_idx:
        rec = records[idx]
        seg_id = str(rec["segment_id"]).strip()
        if not seg_id:
            continue
        if seg_id in processed_segment_ids:
            continue

        current_text, context_text = build_caption_context(records, idx, radius=segment_radius)
        if not current_text:
            processed_segment_ids.add(seg_id)
            continue

        video_path = rec["video_path"]
        start_sec = float(rec["start_sec"])
        end_sec = float(rec["end_sec"])
        duration_sec = round(max(0.0, end_sec - start_sec), 3)

        try:
            plan = plan_segment(
                current_text=current_text,
                context_text=context_text,
                surgery_type=surgery_type,
                model=model,
                max_categories=max_categories_per_segment,
                question_style=question_style,
            )
        except Exception as e:
            if save_rejected:
                rejected.append({
                    "video_path": video_path,
                    "youtube_id": group_key,
                    "segment_id": seg_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                    "source_caption": current_text,
                    "stage": "plan",
                    "error": str(e),
                })
            processed_segment_ids.add(seg_id)
            continue

        try:
            candidates = generate_candidates(
                current_text=current_text,
                context_text=context_text,
                surgery_type=surgery_type,
                plan=plan,
                model=model,
                question_style=question_style,
            )
        except Exception as e:
            if save_rejected:
                rejected.append({
                    "video_path": video_path,
                    "youtube_id": group_key,
                    "segment_id": seg_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                    "source_caption": current_text,
                    "stage": "generate",
                    "plan": plan,
                    "error": str(e),
                })
            processed_segment_ids.add(seg_id)
            continue

        if not candidates:
            if save_rejected:
                rejected.append({
                    "video_path": video_path,
                    "youtube_id": group_key,
                    "segment_id": seg_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                    "source_caption": current_text,
                    "context_caption": context_text,
                    "stage": "generate",
                    "plan": plan,
                    "reason": "no_candidates",
                })
            processed_segment_ids.add(seg_id)
            continue

        for candidate in candidates:
            if skip_validator:
                verdict = {
                    "keep": True,
                    "reason": "validator_skipped",
                    "scores": {
                        "groundedness": None,
                        "temporal_alignment": None,
                        "answerability": None,
                        "non_triviality": None,
                        "hallucination_risk": None,
                    },
                    "revised_answer": candidate["answer"],
                    "revised_question": candidate["question"],
                    "dedup_key_hint": "",
                }
            else:
                try:
                    verdict = validate_candidate(
                        current_text=current_text,
                        context_text=context_text,
                        surgery_type=surgery_type,
                        candidate=candidate,
                        model=model,
                        min_score=min_validation_score,
                    )
                except Exception as e:
                    if save_rejected:
                        rejected.append({
                            "video_path": video_path,
                            "youtube_id": group_key,
                            "segment_id": seg_id,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration_sec": duration_sec,
                            "source_caption": current_text,
                            "context_caption": context_text,
                            "stage": "validate",
                            "candidate": candidate,
                            "error": str(e),
                        })
                    continue

            final_question = verdict["revised_question"] or candidate["question"]
            final_answer = verdict["revised_answer"] or candidate["answer"]

            rec_id = record_uid(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                category=candidate["category"],
                question=final_question,
                answer=final_answer,
            )
            if rec_id in accepted_record_ids:
                continue

            if verdict["keep"]:
                accepted_record_ids.add(rec_id)
                accepted.append({
                    "id": rec_id,
                    "video_path": video_path,
                    "youtube_id": group_key,
                    "segment_id": seg_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                    "surgery_type": surgery_type,
                    "broad_category": CATEGORY_TO_BROAD.get(candidate["category"], ""),
                    "category": candidate["category"],
                    "question": final_question,
                    "answer": final_answer,
                    "format": candidate["format"],
                    "choices": candidate["choices"],
                    "correct_choice": candidate["correct_choice"],
                    "source_caption": current_text,
                    "context_caption": context_text,
                    "sgm_summary": plan.get("sgm_summary", ""),
                    "salient_entities": plan.get("salient_entities", []),
                    "applicable_broad_categories": plan.get("applicable_broad_categories", []),
                    "applicable_categories": plan.get("applicable_categories", []),
                    "generator_rationale": candidate.get("rationale", ""),
                    "generator_evidence": candidate.get("evidence", ""),
                    "validator_reason": verdict["reason"],
                    "validator_scores": verdict["scores"],
                    "messages": default_chat_messages(
                        question=final_question,
                        answer=final_answer,
                        include_rationale=include_rationale_in_chat,
                        rationale=candidate.get("rationale", ""),
                    ),
                })
            else:
                if save_rejected:
                    rejected.append({
                        "video_path": video_path,
                        "youtube_id": group_key,
                        "segment_id": seg_id,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": duration_sec,
                        "surgery_type": surgery_type,
                        "source_caption": current_text,
                        "context_caption": context_text,
                        "plan": plan,
                        "candidate": candidate,
                        "validator": verdict,
                        "stage": "validate",
                    })

        processed_segment_ids.add(seg_id)

    return accepted, rejected


# ============================================================
# MAIN
# ============================================================

DEFAULT_PROCEDURE_MAP_JSON = "/surgery_research/dataset_helpers/merged.json"
DEFAULT_TYPE_EXCEL_DIR = "/surgery_research/auto_phases/Annotations_April11"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate surgical VQA from QwenVL surgical clip captions.")
    p.add_argument(
        "--input_jsonl",
        type=str,
        default="/surgery_research/dataloaders/output_jsons_april11/stage1_segment_caption_subclip_qwen.jsonl",
        help="QwenVL-style JSONL with one surgical clip record per line.",
    )
    p.add_argument(
        "--procedure_map_json",
        type=str,
        default=DEFAULT_PROCEDURE_MAP_JSON,
        help="JSON mapping youtube_id -> metadata like surgery/procedure/specialty.",
    )
    p.add_argument("--output", type=str, default="reasoning_vqa_1.jsonl")
    p.add_argument("--rejected_output", type=str, default="rejected.jsonl")
    p.add_argument("--model", type=str, default="gpt-5.4-mini")
    p.add_argument("--max_videos", type=int, default=0, help="Maximum number of unique videos/groups to process.")
    p.add_argument("--segments_per_video", type=int, default=7, help="Maximum clip records per video to process. 0 means all.")
    p.add_argument("--min_caption_len", type=int, default=12)
    p.add_argument("--max_categories_per_segment", type=int, default=3)
    p.add_argument("--question_style", type=str, default="mixed", choices=["open", "mcq", "mixed"])
    p.add_argument("--min_validation_score", type=int, default=4, choices=[1, 2, 3, 4, 5])
    p.add_argument("--segment_radius", type=int, default=1, help="How many previous/next clip descriptions to include as context.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_rejected", action="store_true")
    p.add_argument("--include_rationale_in_chat", action="store_true")
    p.add_argument("--flush_every", type=int, default=10, help="Write buffered results every N accepted samples.")

    p.add_argument(
        "--id_list_txt",
        type=str,
        default="",
        help="Optional txt file with one youtube_id per line. If provided, only these ids are processed.",
    )
    p.add_argument(
        "--procedure_filter",
        type=str,
        nargs="*",
        default=[],
        help="Optional procedure filter(s). Only ids whose procedure matches will be processed.",
    )
    p.add_argument(
        "--specialty_filter",
        type=str,
        nargs="*",
        default=[],
        help="Optional specialty filter(s). Only ids whose specialty matches will be processed.",
    )
    p.add_argument(
        "--surgery_type_filter",
        type=str,
        nargs="*",
        default=[],
        help="Optional surgery type filter(s). Only ids whose surgery type matches will be processed.",
    )
    p.add_argument(
        "--type_source",
        type=str,
        choices=["json", "excel"],
        default="json",
        help="Where to read surgery type from for filtering: json or excel.",
    )
    p.add_argument(
        "--type_excel",
        type=str,
        default=DEFAULT_TYPE_EXCEL_DIR,
        help="Excel file used only to read columns: id, Type. Required if --type_source excel.",
    )
    p.add_argument(
        "--skip_validator",
        action="store_true",
        help="If set, skip validator calls and accept generated candidates directly.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.type_source == "excel" and not args.type_excel:
        raise ValueError("--type_excel is required when --type_source excel")

    out_path = Path(args.output)
    rej_path = Path(args.rejected_output) if args.rejected_output else out_path.with_name(out_path.stem + "_rejected.jsonl")

    existing = load_existing_records(out_path) if args.resume else []
    processed_segment_ids: set[str] = set()
    accepted_record_ids: set[str] = set()

    for item in existing:
        if isinstance(item, dict):
            rid = item.get("id")
            sid = item.get("segment_id")
            if isinstance(rid, str) and rid:
                accepted_record_ids.add(rid)
            if isinstance(sid, str) and sid:
                processed_segment_ids.add(sid)

    procedure_map = load_procedure_map(args.procedure_map_json)

    type_map_from_excel: Dict[str, str] = {}
    if args.type_source == "excel":
        type_map_from_excel = load_type_map_from_excel_dir(args.type_excel)

    all_records = load_qwenvl_jsonl(args.input_jsonl, min_caption_len=args.min_caption_len)
    grouped = group_qwenvl_records(all_records)
    group_items = sorted(grouped.items(), key=lambda kv: kv[0])

    allowed_ids: set[str] | None = None
    if args.id_list_txt:
        allowed_ids = load_id_filter_txt(args.id_list_txt)

    procedure_filters = parse_filter_values(args.procedure_filter)
    specialty_filters = parse_filter_values(args.specialty_filter)
    surgery_type_filters = parse_filter_values(args.surgery_type_filter)

    filtered_group_items = []
    for youtube_id, records in group_items:
        if allowed_ids is not None and youtube_id not in allowed_ids:
            continue

        if not matches_metadata_filters(
            youtube_id=youtube_id,
            procedure_map=procedure_map,
            procedure_filters=procedure_filters,
            specialty_filters=specialty_filters,
            surgery_type_filters=surgery_type_filters,
            type_source=args.type_source,
            type_map_from_excel=type_map_from_excel,
        ):
            continue

        filtered_group_items.append((youtube_id, records))

    group_items = filtered_group_items

    random.shuffle(group_items)
    if args.max_videos > 0:
        group_items = group_items[:args.max_videos]

    num_total_clips = sum(
        len(records) if args.segments_per_video <= 0 else min(len(records), args.segments_per_video)
        for _, records in group_items
    )

    print("=" * 72)
    print("=" * 72)
    print(f"Input JSONL            : {args.input_jsonl}")
    print(f"Procedure map          : {args.procedure_map_json}")
    print(f"Type source            : {args.type_source}")
    print(f"Type Excel             : {args.type_excel if args.type_excel else '(none)'}")
    print(f"ID filter txt          : {args.id_list_txt if args.id_list_txt else '(none)'}")
    print(f"Procedure filter       : {procedure_filters if procedure_filters else '(none)'}")
    print(f"Specialty filter       : {specialty_filters if specialty_filters else '(none)'}")
    print(f"Surgery type filter    : {surgery_type_filters if surgery_type_filters else '(none)'}")
    print(f"Output                 : {out_path}")
    print(f"Rejected output        : {rej_path if args.save_rejected else '(disabled)'}")
    print(f"Model                  : {args.model}")
    print(f"Skip validator         : {args.skip_validator}")
    print(f"Videos to process      : {len(group_items)}")
    print(f"Clip records selected  : {num_total_clips}")
    print(f"Segments per video     : {'all' if args.segments_per_video <= 0 else args.segments_per_video}")
    print(f"Question style         : {args.question_style}")
    print(f"Min validation score   : {args.min_validation_score}")
    print(f"Resume mode            : {args.resume}")
    print(f"Already accepted       : {len(existing)}")
    print(f"Already processed segs : {len(processed_segment_ids)}")
    print("=" * 72)

    buffer_accept: List[Dict[str, Any]] = []
    buffer_reject: List[Dict[str, Any]] = []
    total_accept = len(existing)
    total_reject = 0

    iterator = choose_progress(enumerate(group_items, start=1), total=len(group_items), desc="Videos")

    for vid_idx, (group_key, records) in iterator:
        accepted, rejected = process_qwenvl_group(
            group_key=group_key,
            records=records,
            procedure_map=procedure_map,
            model=args.model,
            question_style=args.question_style,
            max_segments_per_video=args.segments_per_video,
            max_categories_per_segment=args.max_categories_per_segment,
            min_validation_score=args.min_validation_score,
            include_rationale_in_chat=args.include_rationale_in_chat,
            segment_radius=args.segment_radius,
            processed_segment_ids=processed_segment_ids,
            accepted_record_ids=accepted_record_ids,
            save_rejected=args.save_rejected,
            skip_validator=args.skip_validator,
        )

        buffer_accept.extend(accepted)
        buffer_reject.extend(rejected)

        total_accept += len(accepted)
        total_reject += len(rejected)

        if len(buffer_accept) >= args.flush_every:
            append_jsonl(out_path, buffer_accept)
            buffer_accept = []

        if args.save_rejected and len(buffer_reject) >= max(10, args.flush_every):
            append_jsonl(rej_path, buffer_reject)
            buffer_reject = []

        if vid_idx % 25 == 0:
            print(
                f"[progress] videos={vid_idx}/{len(group_items)} "
                f"accepted_total={total_accept} rejected_total={total_reject}"
            )

    append_jsonl(out_path, buffer_accept)
    if args.save_rejected:
        append_jsonl(rej_path, buffer_reject)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Accepted QA pairs : {total_accept}")
    if args.save_rejected:
        print(f"Rejected records  : {total_reject}")
    print(f"Output path       : {out_path}")

    final_records = load_existing_records(out_path)
    final_by_broad: Dict[str, int] = {}
    final_by_category: Dict[str, int] = {}
    for item in final_records:
        if isinstance(item, dict):
            broad = item.get("broad_category")
            cat = item.get("category")
            if isinstance(broad, str) and broad:
                final_by_broad[broad] = final_by_broad.get(broad, 0) + 1
            if isinstance(cat, str) and cat:
                final_by_category[cat] = final_by_category.get(cat, 0) + 1

    if final_records:
        print("\nBy broad category:")
        for cat, count in sorted(final_by_broad.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = 100.0 * count / len(final_records)
            print(f"  {cat:35s} {count:7d} ({pct:5.1f}%)")

        print("\nBy fine-grained category:")
        for cat, count in sorted(final_by_category.items(), key=lambda kv: (-kv[1], kv[0])):
            pct = 100.0 * count / len(final_records)
            print(f"  {cat:35s} {count:7d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()