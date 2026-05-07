#!/usr/bin/env python3
"""
Automatically crop surgical videos in a folder to the main procedure window.

This version is tuned for surgical footage where the true region of interest is
often centered and contains red / pink / warm tissue colors. It looks for the
most likely procedure window across sampled frames using a surgery-aware saliency
map built from:
- tissue color prior (red / pink / warm regions)
- color saturation
- texture / edges
- motion (lower weight than before)
- soft center prior

It then crops only when the detected window removes a meaningful amount of
non-procedure overlay.

Examples:
  python cropping.py \
    --input_dir ./videos \
    --output_dir ./cropped_videos

  python cropping.py \
    --input_dir ./videos \
    --output_dir ./cropped_videos \
    --recursive \
    --samples 24 \
    --padding-pct 0.04 \
    --min-border-pct 0.025

  python cropping.py \
    --input_dir ./videos \
    --output_dir ./cropped_videos \
    --dry-run \
    --print-filter
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv", ".mpg", ".mpeg", ".webm"
}


def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def check_dependencies():
    for exe in ["ffprobe", "ffmpeg"]:
        if shutil.which(exe) is None:
            raise RuntimeError(f"Missing dependency: {exe}. Please install FFmpeg first.")


def setup_logger(output_dir: Path, log_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("surgical_cropper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / log_name, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    out = run_cmd(cmd)
    data = json.loads(out)

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {video_path}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])

    duration = stream.get("duration")
    if duration in (None, "N/A"):
        duration = data.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        raise RuntimeError("Could not determine video duration.")

    return width, height, float(duration)


def even_int(x):
    x = int(round(x))
    if x < 2:
        x = 2
    return x if x % 2 == 0 else x - 1


def clamp(val, low, high):
    return max(low, min(high, val))


def build_sample_times(duration, samples):
    usable_start = min(1.0, duration * 0.05)
    usable_end = max(usable_start, duration - min(1.0, duration * 0.05))

    if duration <= 2.0 or samples <= 1 or usable_end <= usable_start:
        return [max(0.0, duration / 2.0)]

    step = (usable_end - usable_start) / (samples - 1)
    return [usable_start + i * step for i in range(samples)]


def read_sampled_frames(video_path, timestamps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("Could not read any sample frames from the video.")
    return frames


def resize_for_analysis(frame, max_dim):
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return frame.copy(), 1.0
    resized = cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def make_center_prior(h, w, sigma=0.40):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - (w - 1) / 2.0) / max(w / 2.0, 1.0)
    yy = (yy - (h - 1) / 2.0) / max(h / 2.0, 1.0)
    rr2 = xx * xx + yy * yy
    prior = np.exp(-rr2 / (2.0 * sigma * sigma)).astype(np.float32)
    maxv = float(prior.max())
    if maxv > 1e-6:
        prior /= maxv
    return prior


def tissue_color_map_bgr(frame):
    """
    Highlight red / pink / warm surgical tissue.
    Returns a float32 map in [0, 1].
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]            # OpenCV hue in [0, 179]
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    # Red bands near 0 and 179.
    red1 = ((h >= 0) & (h <= 12)).astype(np.float32)
    red2 = ((h >= 165) & (h <= 179)).astype(np.float32)

    # Pink / magenta-ish range often seen in tissue under OR lighting.
    pink = ((h >= 145) & (h <= 179)).astype(np.float32)

    # Warm orange-red extension can help on cautery-lit tissue.
    warm = ((h >= 8) & (h <= 22)).astype(np.float32)

    hue_score = np.maximum(red1 + red2, 0.7 * pink)
    hue_score = np.maximum(hue_score, 0.45 * warm)

    # Require enough saturation/value to suppress dark borders and dull overlays.
    sv_gate = np.clip((s - 0.18) / 0.35, 0.0, 1.0) * np.clip((v - 0.12) / 0.35, 0.0, 1.0)

    # Extra BGR warm-tissue cue: red dominating green and blue.
    b = frame[:, :, 0].astype(np.float32) / 255.0
    g = frame[:, :, 1].astype(np.float32) / 255.0
    r = frame[:, :, 2].astype(np.float32) / 255.0
    warm_rgb = np.clip(r - 0.6 * g - 0.35 * b, 0.0, 1.0)

    tissue = 0.65 * hue_score * sv_gate + 0.35 * warm_rgb
    tissue = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), 3.0)

    maxv = float(tissue.max())
    if maxv > 1e-6:
        tissue /= maxv
    return tissue


def compute_saliency_map(frames, max_dim):
    small_frames = []
    scales = []
    for frame in frames:
        small, scale = resize_for_analysis(frame, max_dim)
        small_frames.append(small)
        scales.append(scale)

    analysis_scale = scales[0]
    target_h, target_w = small_frames[0].shape[:2]

    aligned_frames = []
    for frame in small_frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        aligned_frames.append(frame)

    saliency_accum = np.zeros((target_h, target_w), dtype=np.float32)
    prev_gray = None
    center_prior = make_center_prior(target_h, target_w, sigma=0.40)

    for frame in aligned_frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad = cv2.GaussianBlur(grad, (0, 0), 2.0)
        grad_max = float(grad.max())
        if grad_max > 1e-6:
            grad /= grad_max

        if prev_gray is None:
            motion = np.zeros_like(gray)
        else:
            motion = cv2.absdiff(gray, prev_gray)
            motion = cv2.GaussianBlur(motion, (0, 0), 2.0)
            motion_max = float(motion.max())
            if motion_max > 1e-6:
                motion /= motion_max
        prev_gray = gray

        tissue = tissue_color_map_bgr(frame)

        # Surgery-aware weighting: tissue strongest, motion weakest.
        saliency = (
            0.50 * tissue +
            0.20 * sat +
            0.18 * grad +
            0.12 * motion
        )

        # Soft center prior so corner overlays are less likely to dominate.
        saliency = saliency * (0.65 + 0.35 * center_prior)
        saliency = cv2.GaussianBlur(saliency, (0, 0), 4.0)
        saliency_accum += saliency

    saliency_mean = saliency_accum / len(aligned_frames)
    saliency_mean = cv2.GaussianBlur(saliency_mean, (0, 0), 6.0)

    maxv = float(saliency_mean.max())
    if maxv > 1e-6:
        saliency_mean /= maxv

    return saliency_mean, analysis_scale


def component_score(x, y, w, h, area, frame_w, frame_h):
    """
    Score a connected component. Larger, more central boxes are preferred.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = abs(cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
    dy = abs(cy - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
    center_penalty = dx * dx + dy * dy
    return float(area) * (1.0 - 0.35 * min(center_penalty, 1.0))


def best_component_bbox(mask, min_area_pixels):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    h, w = mask.shape[:2]
    best = None
    best_score = -1.0

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < min_area_pixels:
            continue
        score = component_score(x, y, bw, bh, area, w, h)
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    return best


def detect_nonblack_bbox(frame, black_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > black_threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = gray.shape[:2]
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def expand_box(x, y, w, h, frame_w, frame_h, padding_pct):
    pad_x = int(round(w * padding_pct))
    pad_y = int(round(h * padding_pct))
    left = clamp(x - pad_x, 0, frame_w - 2)
    top = clamp(y - pad_y, 0, frame_h - 2)
    right = clamp(x + w + pad_x, left + 2, frame_w)
    bottom = clamp(y + h + pad_y, top + 2, frame_h)
    return left, top, right - left, bottom - top


def detect_procedure_window(
    frames,
    original_w,
    original_h,
    max_dim,
    saliency_quantile,
    padding_pct,
    black_threshold,
):
    saliency_map, scale = compute_saliency_map(frames, max_dim)
    small_h, small_w = saliency_map.shape[:2]

    q_value = float(np.quantile(saliency_map, saliency_quantile))
    threshold = max(q_value, 0.28 * float(saliency_map.max()))
    mask = (saliency_map >= threshold).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    min_area_pixels = int(round(0.03 * small_w * small_h))
    bbox = best_component_bbox(mask, min_area_pixels)

    if bbox is None:
        return 0, 0, original_w, original_h, {"reason": "no_salient_component"}

    x, y, w, h = bbox

    # Reject implausible skinny / overlay-like detections.
    aspect = w / max(h, 1)
    area_ratio = (w * h) / max(float(small_w * small_h), 1.0)
    if aspect > 3.8 or aspect < 0.45 or area_ratio < 0.05:
        return 0, 0, original_w, original_h, {
            "reason": "implausible_bbox",
            "aspect": aspect,
            "area_ratio": area_ratio,
        }

    inv_scale = 1.0 / scale
    x = int(round(x * inv_scale))
    y = int(round(y * inv_scale))
    w = int(round(w * inv_scale))
    h = int(round(h * inv_scale))

    nb_boxes = [detect_nonblack_bbox(frame, black_threshold) for frame in frames]
    nb_left = min(b[0] for b in nb_boxes)
    nb_top = min(b[1] for b in nb_boxes)
    nb_right = max(b[0] + b[2] for b in nb_boxes)
    nb_bottom = max(b[1] + b[3] for b in nb_boxes)

    x, y, w, h = expand_box(x, y, w, h, original_w, original_h, padding_pct)

    x = clamp(x, nb_left, max(nb_left, nb_right - 2))
    y = clamp(y, nb_top, max(nb_top, nb_bottom - 2))
    right = clamp(x + w, x + 2, nb_right)
    bottom = clamp(y + h, y + 2, nb_bottom)

    x = even_int(x)
    y = even_int(y)
    w = even_int(right - x)
    h = even_int(bottom - y)

    w = clamp(w, 2, original_w - x)
    h = clamp(h, 2, original_h - y)

    details = {
        "saliency_threshold": threshold,
        "saliency_quantile_value": q_value,
        "analysis_scale": scale,
        "nonblack_box": (nb_left, nb_top, nb_right - nb_left, nb_bottom - nb_top),
        "aspect": aspect,
        "area_ratio": area_ratio,
    }
    return x, y, w, h, details


def should_crop(video_w, video_h, x, y, crop_w, crop_h, min_border_pct, min_border_px):
    left = x
    top = y
    right = video_w - (x + crop_w)
    bottom = video_h - (y + crop_h)

    borders = [left, top, right, bottom]
    border_pcts = [
        left / video_w,
        top / video_h,
        right / video_w,
        bottom / video_h,
    ]

    largest_border = max(borders)
    largest_border_pct = max(border_pcts)
    kept_area_ratio = (crop_w * crop_h) / (video_w * video_h)

    crop_needed = largest_border >= min_border_px and largest_border_pct >= min_border_pct
    return crop_needed, {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "largest_border_px": largest_border,
        "largest_border_pct": largest_border_pct,
        "kept_area_ratio": kept_area_ratio,
    }


def crop_video(input_path, output_path, x, y, w, h):
    crop_filter = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]
    run_cmd(cmd)


def find_videos(input_dir: Path, recursive: bool):
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def make_output_path(input_path: Path, input_dir: Path, output_dir: Path, keep_structure: bool):
    if keep_structure:
        relative_parent = input_path.parent.relative_to(input_dir)
        target_dir = output_dir / relative_parent
    else:
        target_dir = output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{input_path.stem}_cropped{input_path.suffix}"


def process_video(input_path: Path, output_path: Path, args, logger):
    logger.info("Processing video: %s", input_path)

    video_w, video_h, duration = get_video_info(input_path)
    timestamps = build_sample_times(duration, args.samples)
    frames = read_sampled_frames(input_path, timestamps)

    x, y, crop_w, crop_h, details = detect_procedure_window(
        frames=frames,
        original_w=video_w,
        original_h=video_h,
        max_dim=args.max_analysis_dim,
        saliency_quantile=args.saliency_quantile,
        padding_pct=args.padding_pct,
        black_threshold=args.black_threshold,
    )

    crop_needed, border_info = should_crop(
        video_w, video_h, x, y, crop_w, crop_h, args.min_border_pct, args.min_border_px
    )

    logger.info("Original dimensions: %sx%s", video_w, video_h)
    logger.info("Duration: %.2fs", duration)
    logger.info("Sampled frames: %s", len(frames))
    logger.info("Crop box: x=%s, y=%s, w=%s, h=%s", x, y, crop_w, crop_h)
    logger.info(
        "Borders removed: left=%s, top=%s, right=%s, bottom=%s",
        border_info["left"], border_info["top"], border_info["right"], border_info["bottom"]
    )
    logger.info(
        "Largest border: %s px (%.3f%%)",
        border_info["largest_border_px"],
        border_info["largest_border_pct"] * 100.0,
    )
    logger.info("Kept area ratio: %.3f%%", border_info["kept_area_ratio"] * 100.0)

    if "nonblack_box" in details:
        nbx, nby, nbw, nbh = details["nonblack_box"]
        logger.info("Aggregate non-black box: x=%s, y=%s, w=%s, h=%s", nbx, nby, nbw, nbh)

    if "reason" in details:
        logger.info("Detection note: %s", details["reason"])
    if "aspect" in details:
        logger.info("Candidate aspect ratio: %.3f", details["aspect"])
    if "area_ratio" in details:
        logger.info("Candidate area ratio: %.3f%%", details["area_ratio"] * 100.0)

    crop_filter = f"crop={crop_w}:{crop_h}:{x}:{y}"
    if args.print_filter:
        logger.info("FFmpeg filter: %s", crop_filter)

    logger.info("Crop decision: %s", "apply" if crop_needed else "skip")
    logger.info("Output path: %s", output_path)

    if args.dry_run:
        logger.info("Dry run enabled. No output file written.")
        return "dry_run"

    if not crop_needed:
        logger.info("Skipping crop because detected border is below thresholds.")
        return "skipped"

    crop_video(input_path, output_path, x, y, crop_w, crop_h)
    logger.info("Saved cropped video to: %s", output_path)
    return "cropped"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically crop all videos in a folder to the main surgical procedure "
            "window, even when overlays are not black."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Path to folder containing input videos")
    parser.add_argument("--output_dir", required=True, help="Directory where output videos and log file will be saved")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for videos in subfolders of the input directory"
    )
    parser.add_argument(
        "--keep-structure",
        action="store_true",
        help="Preserve relative input subfolder structure inside the output directory"
    )
    parser.add_argument(
        "--log-name",
        default="cropping_log.txt",
        help="Name of the text log file written into the output directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of frames to sample across each video"
    )
    parser.add_argument(
        "--max-analysis-dim",
        type=int,
        default=640,
        help="Resize longest sampled-frame dimension to at most this many pixels during analysis"
    )
    parser.add_argument(
        "--saliency-quantile",
        type=float,
        default=0.78,
        help="Quantile used to threshold the saliency map; larger values make cropping tighter"
    )
    parser.add_argument(
        "--padding-pct",
        type=float,
        default=0.04,
        help="Extra padding added around the detected procedure window as a fraction of its size"
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=18,
        help="Brightness threshold used only to prevent expansion into pure black letterboxing"
    )
    parser.add_argument(
        "--min-border-px",
        type=int,
        default=16,
        help="Skip cropping unless at least one side removes this many pixels"
    )
    parser.add_argument(
        "--min-border-pct",
        type=float,
        default=0.025,
        help="Skip cropping unless at least one side removes at least this fraction of width or height"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze videos and log crop decisions, but do not save output files"
    )
    parser.add_argument(
        "--print-filter",
        action="store_true",
        help="Print the ffmpeg crop filter string for each processed video"
    )

    args = parser.parse_args()

    if not 0.0 < args.saliency_quantile < 1.0:
        print("--saliency-quantile must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)
    if args.padding_pct < 0:
        print("--padding-pct must be non-negative.", file=sys.stderr)
        sys.exit(1)
    if args.samples < 1:
        print("--samples must be at least 1.", file=sys.stderr)
        sys.exit(1)

    check_dependencies()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found or is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logger(output_dir, args.log_name)
    logger.info("Starting folder processing")
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Recursive search: %s", args.recursive)
    logger.info("Keep structure: %s", args.keep_structure)
    logger.info("Dry run: %s", args.dry_run)

    video_paths = find_videos(input_dir, args.recursive)
    if not video_paths:
        logger.info("No supported video files found.")
        return

    logger.info("Found %s video(s) to process.", len(video_paths))

    counts = {"cropped": 0, "skipped": 0, "dry_run": 0, "failed": 0}

    for index, input_path in enumerate(video_paths, start=1):
        logger.info("=" * 80)
        logger.info("Video %s of %s", index, len(video_paths))
        try:
            output_path = make_output_path(
                input_path=input_path,
                input_dir=input_dir,
                output_dir=output_dir,
                keep_structure=args.keep_structure,
            )
            result = process_video(input_path, output_path, args, logger)
            counts[result] += 1
        except Exception as exc:
            counts["failed"] += 1
            logger.exception("Failed to process %s: %s", input_path, exc)

    logger.info("=" * 80)
    logger.info("Processing complete.")
    logger.info("Cropped: %s", counts["cropped"])
    logger.info("Skipped: %s", counts["skipped"])
    logger.info("Dry run only: %s", counts["dry_run"])
    logger.info("Failed: %s", counts["failed"])
    logger.info("Log file saved to: %s", output_dir / args.log_name)


if __name__ == "__main__":
    main()
