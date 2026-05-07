#!/usr/bin/env python3

import argparse
import contextlib
from pathlib import Path
import sys
import pandas as pd

import cropping as cropper


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
SHEET_EXTS = {".csv", ".xlsx", ".xls"}

SHEET_DIRS = [
    Path("Final_Annotations/"),
]

VIDEO_DIRS = [
    Path("surgery_data/"),
]

OUTPUT_DIR = Path("surgery_data/cropped")


def load_sheet(sheet_path: Path) -> pd.DataFrame:
    if sheet_path.suffix.lower() == ".csv":
        return pd.read_csv(sheet_path)
    if sheet_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(sheet_path)
    raise ValueError(f"Unsupported sheet format: {sheet_path.suffix}")


def collect_sheet_paths():
    paths = []
    for sheet_dir in SHEET_DIRS:
        if not sheet_dir.exists():
            print(f"WARNING sheet dir not found: {sheet_dir}")
            continue
        for p in sorted(sheet_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SHEET_EXTS:
                paths.append(p)
    return paths


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_yes(val) -> bool:
    return str(val).strip().lower() in {"yes", "y", "true", "1"}


def get_crop_yes_items(sheet_paths: list[Path]) -> list[dict]:
    dfs = []
    for sheet_path in sheet_paths:
        df = load_sheet(sheet_path)
        df = normalize_df_columns(df)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required_cols = ["id", "Crop"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    if "title" not in df.columns:
        df["title"] = ""

    df["id"] = df["id"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()

    selected = df[df["Crop"].apply(is_yes)].copy()

    items = []
    seen = set()
    for _, row in selected.iterrows():
        yt_id = str(row["id"]).strip()
        title = str(row["title"]).strip()
        if not yt_id or yt_id in seen:
            continue
        seen.add(yt_id)
        items.append({"id": yt_id, "title": title})

    return items


def build_video_map_by_id(video_dirs: list[Path]) -> dict[str, Path]:
    video_map = {}
    for videos_dir in video_dirs:
        if not videos_dir.exists():
            print(f"WARNING videos dir not found: {videos_dir}")
            continue
        for p in videos_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                stem = p.stem.strip()
                if stem not in video_map:
                    video_map[stem] = p
    return video_map


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout_stderr(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.__stdout__, log_file)
        stderr_tee = TeeStream(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--max-analysis-dim", type=int, default=640)
    parser.add_argument("--saliency-quantile", type=float, default=0.78)
    parser.add_argument("--padding-pct", type=float, default=0.04)
    parser.add_argument("--black-threshold", type=int, default=18)
    parser.add_argument("--min-border-px", type=int, default=16)
    parser.add_argument("--min-border-pct", type=float, default=0.025)
    parser.add_argument("--print-filter", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = logs_dir / "console_output.txt"
    run_csv_path = logs_dir / "run_summary.csv"

    with tee_stdout_stderr(console_log_path):
        cropper.check_dependencies()
        logger = cropper.setup_logger(OUTPUT_DIR, "crop_selected_ids_log.txt")

        sheet_paths = collect_sheet_paths()
        if not sheet_paths:
            raise FileNotFoundError("No valid sheets found.")

        items = get_crop_yes_items(sheet_paths)
        print(f"Selected ids with Crop == Yes: {len(items)}")

        video_map = build_video_map_by_id(VIDEO_DIRS)
        print(f"Found local videos by id: {len(video_map)}")

        cropped = 0
        skipped_missing = 0
        skipped_existing = 0
        skipped_no_crop_needed = 0
        dry_run_only = 0
        failed = 0
        run_rows = []

        for i, item in enumerate(items, 1):
            yt_id = item["id"]
            title = item["title"]

            logger.info("=" * 80)
            logger.info("Item %s of %s | id=%s | title=%s", i, len(items), yt_id, title)

            video_path = video_map.get(yt_id)
            output_path = None
            status = ""
            error_message = ""

            if video_path is None:
                logger.info("SKIP missing local video for id=%s", yt_id)
                skipped_missing += 1
                status = "missing_video"
            else:
                output_path = OUTPUT_DIR / f"{yt_id}_cropped{video_path.suffix}"
                if output_path.exists():
                    logger.info("SKIP already cropped: %s", output_path)
                    skipped_existing += 1
                    status = "existing_output"
                else:
                    try:
                        result = cropper.process_video(video_path, output_path, args, logger)
                        if result == "cropped":
                            cropped += 1
                            status = "cropped"
                        elif result == "skipped":
                            skipped_no_crop_needed += 1
                            status = "crop_not_needed"
                        elif result == "dry_run":
                            dry_run_only += 1
                            status = "dry_run"
                        else:
                            status = f"unknown_result:{result}"
                    except Exception as e:
                        logger.exception("FAILED id=%s path=%s error=%s", yt_id, video_path, e)
                        failed += 1
                        status = "failed"
                        error_message = str(e)

            run_rows.append(
                {
                    "item_index": i,
                    "total_items": len(items),
                    "id": yt_id,
                    "title": title,
                    "video_path": str(video_path) if video_path is not None else "",
                    "output_path": str(output_path) if output_path is not None else "",
                    "status": status,
                    "error_message": error_message,
                    "dry_run": args.dry_run,
                    "samples": args.samples,
                    "max_analysis_dim": args.max_analysis_dim,
                    "saliency_quantile": args.saliency_quantile,
                    "padding_pct": args.padding_pct,
                    "black_threshold": args.black_threshold,
                    "min_border_px": args.min_border_px,
                    "min_border_pct": args.min_border_pct,
                    "print_filter": args.print_filter,
                }
            )

        pd.DataFrame(run_rows).to_csv(run_csv_path, index=False)

        print("\nDone.")
        print(f"Cropped: {cropped}")
        print(f"Skipped missing video: {skipped_missing}")
        print(f"Skipped existing output: {skipped_existing}")
        print(f"Skipped because crop not needed: {skipped_no_crop_needed}")
        print(f"Dry run only: {dry_run_only}")
        print(f"Failed: {failed}")
        print(f"Console log saved to: {console_log_path}")
        print(f"Run summary CSV saved to: {run_csv_path}")


if __name__ == "__main__":
    main()

