import sys
import logging
from pathlib import Path

import pandas as pd

from src.settings.config import Config
from scripts.run_pipeline_batch import SurgeryTranscriptionPipeline


# VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


# def safe_folder_name(name: str) -> str:
#     invalid_chars = '<>:"/\\|?*'
#     for ch in invalid_chars:
#         name = name.replace(ch, "_")
#     name = " ".join(name.split()).strip()
#     return name[:200]


# def main():
#     # if len(sys.argv) != 2:
#     #     print("Usage: python scripts/run_batch_asr.py <sheet_path>")
#     #     sys.exit(1)

#     # sheet_path = Path(sys.argv[1])

#     # Hardcoded spreadsheet paths
#     sheet_paths = [
#         Path("/videos/youtube_videos_1.xlsx"),
#         Path("/videos/youtube_videos_2.xlsx"),
#         Path("/videos/youtube_videos_3.xlsx"),
#     ]

#     if not sheet_path.exists():
#         raise FileNotFoundError(f"Sheet not found: {sheet_path}")

#     # Hardcoded video folders
#     # videos_dirs = [Path(p) for p in sys.argv[2:]]
#     videos_dirs = [
#         Path("/videos/data"),
#     ]

#     for d in videos_dirs:
#         if not d.exists():
#             print(f"WARNING videos dir not found: {d}")

#     logging.basicConfig(level=logging.INFO)

#     if sheet_path.suffix.lower() == ".csv":
#         df = pd.read_csv(sheet_path)
#     elif sheet_path.suffix.lower() in {".xlsx", ".xls"}:
#         df = pd.read_excel(sheet_path)
#     else:
#         raise ValueError(f"Unsupported sheet format: {sheet_path.suffix}")

#     df.columns = [c.strip() for c in df.columns]

#     selected = df[
#         df["Surgery/No Surgery"].astype(str).str.strip().str.lower().eq("surgery") &
#         df["Phase Info"].astype(str).str.strip().str.lower().eq("surgeon narration")
#     ].copy()

#     print(f"Selected rows: {len(selected)}")

#     video_map = {}
#     for videos_dir in videos_dirs:
#         if not videos_dir.exists():
#             continue
#         for p in videos_dir.rglob("*"):
#             if p.suffix.lower() in VIDEO_EXTS:
#                 if p.stem not in video_map:
#                     video_map[p.stem] = p
#                 else:
#                     print(f"WARNING duplicate title found, keeping first: {p.stem}")

#     print(f"Found {len(video_map)} local videos across {len(videos_dirs)} folders")

#     config = Config.from_env()
#     pipeline = SurgeryTranscriptionPipeline(config)

#     root_output_dir = Path(config.output_dir)

#     processed_now = 0
#     skipped_missing = 0
#     skipped_done = 0
#     failed = 0

#     for _, row in selected.iterrows():
#         title = str(row["title"]).strip()

#         if not title:
#             print("SKIP missing title in sheet row")
#             skipped_missing += 1
#             continue

#         video_path = video_map.get(title)

#         if video_path is None:
#             print(f"SKIP missing video: {title}")
#             skipped_missing += 1
#             continue

#         folder_name = safe_folder_name(video_path.stem)
#         done_file = (
#             root_output_dir
#             / "narrations"
#             / folder_name
#             / f"{folder_name}_transcript_detailed.json"
#         )

#         if done_file.exists():
#             print(f"SKIP already processed: {video_path.name}")
#             skipped_done += 1
#             continue

#         print(f"RUNNING: {video_path}")

#         try:
#             pipeline.process_video(video_path)
#             processed_now += 1
#         except Exception as e:
#             print(f"FAILED: {video_path.name} -> {e}")
#             failed += 1

#     print("\nDone.")
#     print(f"Processed now: {processed_now}")
#     print(f"Skipped missing video: {skipped_missing}")
#     print(f"Skipped already processed: {skipped_done}")
#     print(f"Failed: {failed}")


# if __name__ == "__main__":
#     main()


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
SHEET_EXTS = {".csv", ".xlsx", ".xls"}


def safe_folder_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    name = " ".join(name.split()).strip()
    return name[:200]


def load_sheet(sheet_path: Path) -> pd.DataFrame:
    if sheet_path.suffix.lower() == ".csv":
        return pd.read_csv(sheet_path)
    if sheet_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(sheet_path)
    raise ValueError(f"Unsupported sheet format: {sheet_path.suffix}")


def collect_sheet_paths() -> list[Path]:
    # Option 1: explicit sheet files
    explicit_sheet_paths = [
        # Path("/videos/youtube_videos.xlsx"),
        # Path("/videos/youtube_videos_2.xlsx"),
    ]

    # Option 2: load all sheets from these folders
    sheet_dirs = [
        Path("/surgery_research/auto_phases/inputs"),
    ]

    collected = []

    for p in explicit_sheet_paths:
        if p.exists() and p.suffix.lower() in SHEET_EXTS:
            collected.append(p)
        else:
            print(f"WARNING explicit sheet not found or unsupported: {p}")

    for sheet_dir in sheet_dirs:
        if not sheet_dir.exists():
            print(f"WARNING sheet dir not found: {sheet_dir}")
            continue

        for p in sorted(sheet_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SHEET_EXTS:
                collected.append(p)

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in collected:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


def main():
    sheet_paths = collect_sheet_paths()

    videos_dirs = [
        Path("/videos/data"),
    ]

    if not sheet_paths:
        raise FileNotFoundError("No valid sheets were found.")

    print("Using sheets:")
    for s in sheet_paths:
        print(f"  - {s}")

    for d in videos_dirs:
        if not d.exists():
            print(f"WARNING videos dir not found: {d}")

    logging.basicConfig(level=logging.INFO)

    dfs = []
    for sheet_path in sheet_paths:
        df = load_sheet(sheet_path)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    selected = df[
        df["Surgery/No Surgery"].astype(str).str.strip().str.lower().eq("surgery") &
        df["Phase Info"].astype(str).str.lower().str.contains("surgeon narration", na=False)
        # df["Phase Info"].astype(str).str.strip().str.lower().eq("surgeon narration")
    ].copy()

    print(f"Selected rows across all sheets: {len(selected)}")

    video_map = {}
    for videos_dir in videos_dirs:
        if not videos_dir.exists():
            continue
        for p in videos_dir.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                if p.stem not in video_map:
                    video_map[p.stem] = p
                else:
                    print(f"WARNING duplicate title found, keeping first: {p.stem}")

    print(f"Found {len(video_map)} local videos across {len(videos_dirs)} folders")

    config = Config.from_env()
    pipeline = SurgeryTranscriptionPipeline(config)

    root_output_dir = Path(config.output_dir)

    processed_now = 0
    skipped_missing = 0
    skipped_done = 0
    failed = 0

    for _, row in selected.iterrows():
        title = str(row["title"]).strip()

        if not title:
            print("SKIP missing title in sheet row")
            skipped_missing += 1
            continue

        video_path = video_map.get(title)

        if video_path is None:
            print(f"SKIP missing video: {title}")
            skipped_missing += 1
            continue

        folder_name = safe_folder_name(video_path.stem)
        done_file = (
            root_output_dir
            / "narrations"
            / folder_name
            / f"{folder_name}_transcript_detailed.json"
        )

        if done_file.exists():
            print(f"SKIP already processed: {video_path.name}")
            skipped_done += 1
            continue

        print(f"RUNNING: {video_path}")

        try:
            pipeline.process_video(video_path)
            processed_now += 1
        except Exception as e:
            print(f"FAILED: {video_path.name} -> {e}")
            failed += 1

    print("\nDone.")
    print(f"Processed now: {processed_now}")
    print(f"Skipped missing video: {skipped_missing}")
    print(f"Skipped already processed: {skipped_done}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()