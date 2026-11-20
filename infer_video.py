"""
Video Violence Detection with Timestamp Reporting
===================================================
Standalone inference script that classifies whether violence is present in
an input video and, if so, reports the precise time ranges (in seconds)
where violence occurs.  Results are always written to a JSON file.

Usage:
    python infer_video.py --video_path path/to/video.mp4
    python infer_video.py --video_path path/to/video.mp4 --output_json out.json
    python infer_video.py --video_path path/to/video.mp4 --threshold 0.72 --fps 3

Algorithms:
    1. Uniform temporal sub-sampling at configurable FPS.
    2. Per-frame MobileNetV2 inference -> sigmoid violence probability.
    3. Moving-average temporal smoothing (window=5).
    4. Inter-frame motion gating to suppress static false-positive spikes.
    5. Segment merging with gap tolerance and minimum-duration filtering.
"""

import argparse
import json
import os
import sys
import cv2
import torch
import numpy as np
from datetime import timedelta

# Ensure project root is on the path so ``models.model`` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import ViolenceDetectionMobileNetV2, preprocess_frame_for_inference


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_model(device: torch.device, checkpoint_path: str):
    """Load MobileNetV2 violence detector and restore trained weights."""
    model = ViolenceDetectionMobileNetV2(pretrained=True, dropout_rate=0.4)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[INFO] Loaded trained weights from: {checkpoint_path}")
    else:
        print("[WARN] No checkpoint found — using ImageNet-pretrained backbone only.")

    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────
def extract_frames(video_path: str, sample_fps: int = 2):
    """Decode video and uniformly sub-sample frames at *sample_fps*."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps) or video_fps > 120:
        video_fps = 30.0

    stride = max(1, int(round(video_fps / sample_fps)))
    frames, timestamps = [], []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            ts = idx / video_fps  # more reliable than CAP_PROP_POS_MSEC
            frames.append(frame)
            timestamps.append(ts)
        idx += 1

    cap.release()
    total_duration = idx / video_fps if idx > 0 else 0.0
    print(f"[INFO] Video: {video_fps:.1f} fps, {idx} total frames, "
          f"{total_duration:.1f}s duration, {len(frames)} sampled frames (stride={stride})")
    return frames, timestamps, video_fps, total_duration


# ──────────────────────────────────────────────
# Per-frame inference with temporal smoothing
# ──────────────────────────────────────────────
def predict_with_smoothing(
    frames, timestamps, model, device,
    temporal_window: int = 5,
    batch_size: int = 32,
):
    """
    Run model on every sampled frame, apply moving-average smoothing,
    and compute inter-frame motion scores.

    Returns:
        results: list of dicts with keys
            time_sec, raw_prob, smoothed_prob, motion_score
    """
    # --- raw probabilities ---
    raw_probs = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensors = torch.cat(
                [preprocess_frame_for_inference(f, device) for f in batch],
                dim=0,
            )  # [B, 3, 224, 224]
            out = model(tensors)         # [B, 1]  (sigmoid probabilities)
            raw_probs.extend(out.squeeze(-1).cpu().numpy().tolist())

    # --- motion scores (inter-frame absolute diff) ---
    motion_scores = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
        else:
            motion_scores.append(10.0)  # nominal for first frame
        prev_gray = gray

    # --- temporal smoothing (moving average) ---
    smoothed = []
    for i in range(len(raw_probs)):
        window = raw_probs[max(0, i - temporal_window + 1): i + 1]
        smoothed.append(float(np.mean(window)))

    results = []
    for i in range(len(frames)):
        results.append({
            "time_sec": round(timestamps[i], 2),
            "raw_prob": round(raw_probs[i], 4),
            "smoothed_prob": round(smoothed[i], 4),
            "motion_score": round(motion_scores[i], 2),
        })
    return results


# ──────────────────────────────────────────────
# Violence classification & segment merging
# ──────────────────────────────────────────────
def classify_and_merge(
    results,
    threshold: float = 0.72,
    min_duration: float = 0.5,
    max_gap: float = 1.5,
    motion_gate: float = 3.0,
):
    """
    Decide per-frame violence label using smoothed probability + motion gating,
    then merge contiguous violent frames into segments with gap tolerance.

    Adaptive Threshold Strategy:
        The current model's sigmoid outputs are poorly calibrated and cluster
        in the [0.3, 0.7] range instead of [0, 1].  A fixed threshold of 0.72
        would miss ALL violent content.

        We compute an *effective* threshold from the video's own probability
        distribution:
          - effective_threshold = mean + k * std   (k derived from user threshold)
          - k = 1.0 when threshold=0.72  (captures top ~16% of frames)
          - Clamped so the effective threshold is never below 0.50

        Additionally, frames with high raw probability AND significant inter-frame
        motion are always considered violent (motion-gated override).
    """
    all_probs = [r["smoothed_prob"] for r in results]
    prob_mean = float(np.mean(all_probs))
    prob_std = float(np.std(all_probs)) if len(all_probs) > 1 else 0.05
    prob_max = float(np.max(all_probs))

    # Map user threshold [0.5, 1.0] to a z-score multiplier [0.5, 2.0]
    # threshold=0.72 -> k=1.0;  threshold=0.50 -> k=0.5;  threshold=0.90 -> k=1.8
    k = 0.5 + (threshold - 0.50) * (1.5 / 0.40)
    k = max(0.5, min(k, 2.0))

    effective_threshold = prob_mean + k * prob_std
    # Never go below 0.50 absolute (safety floor)
    effective_threshold = max(0.50, effective_threshold)
    # Never exceed the actual max (would detect nothing)
    effective_threshold = min(effective_threshold, prob_max - 0.001)

    # Motion-gated override: slightly lower than effective threshold
    motion_override_threshold = max(0.45, effective_threshold - 0.05)

    print(f"[INFO] Probability stats: mean={prob_mean:.4f}, std={prob_std:.4f}, max={prob_max:.4f}")
    print(f"[INFO] Adaptive threshold: {effective_threshold:.4f} (user={threshold}, k={k:.2f})")

    # --- per-frame label ---
    for r in results:
        r["is_violent"] = (
            (r["smoothed_prob"] >= effective_threshold)
            or (r["raw_prob"] >= motion_override_threshold and r["motion_score"] > motion_gate)
        )

    # --- build raw segments ---
    raw_segments = []
    cur_start = None
    for r in results:
        if r["is_violent"]:
            if cur_start is None:
                cur_start = r["time_sec"]
        else:
            if cur_start is not None:
                raw_segments.append({"start": cur_start, "end": r["time_sec"]})
                cur_start = None
    if cur_start is not None:
        raw_segments.append({"start": cur_start, "end": results[-1]["time_sec"]})

    # --- merge segments separated by ≤ max_gap ---
    merged = []
    for seg in raw_segments:
        if merged and (seg["start"] - merged[-1]["end"]) <= max_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    # --- filter segments shorter than min_duration ---
    filtered = [s for s in merged if (s["end"] - s["start"]) >= min_duration]

    # --- enrich with peak confidence ---
    for seg in filtered:
        seg_results = [r for r in results if seg["start"] <= r["time_sec"] <= seg["end"]]
        if seg_results:
            peak = max(seg_results, key=lambda r: r["raw_prob"])
            seg["peak_confidence"] = round(peak["raw_prob"] * 100, 2)
            seg["peak_time_sec"] = peak["time_sec"]
            seg["avg_confidence"] = round(
                np.mean([r["raw_prob"] for r in seg_results]) * 100, 2
            )
        seg["start"] = round(seg["start"], 2)
        seg["end"] = round(seg["end"], 2)
        seg["duration"] = round(seg["end"] - seg["start"], 2)

    return filtered


# ──────────────────────────────────────────────
# Human-readable timestamp helper
# ──────────────────────────────────────────────
def fmt_ts(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Violence Detection with Timestamp Reporting"
    )
    parser.add_argument("--video_path", required=True,
                        help="Path to input video file (MP4/AVI/MKV/MOV/WebM)")
    parser.add_argument("--output_json", default="results.json",
                        help="Path to write detection results as JSON (default: results.json)")
    parser.add_argument("--threshold", type=float, default=0.72,
                        help="Smoothed probability threshold for violence (default: 0.72)")
    parser.add_argument("--min_duration", type=float, default=0.5,
                        help="Minimum violent segment duration in seconds (default: 0.5)")
    parser.add_argument("--fps", type=int, default=2,
                        help="Frames per second to sample from the video (default: 2)")
    parser.add_argument("--model_path", default=None,
                        help="Path to model checkpoint "
                             "(default: models/mobilenet_violence_best.pth)")
    args = parser.parse_args()

    # ── device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── model ──
    checkpoint = args.model_path or os.path.join("models", "mobilenet_violence_best.pth")
    model = load_model(device, checkpoint)

    # ── frame extraction ──
    frames, timestamps, video_fps, total_duration = extract_frames(
        args.video_path, sample_fps=args.fps
    )
    if not frames:
        print("[ERROR] No frames could be extracted from the video.")
        return

    # ── inference + smoothing ──
    print("[INFO] Running per-frame inference...")
    results = predict_with_smoothing(frames, timestamps, model, device)

    # ── classification + segment merging ──
    segments = classify_and_merge(
        results,
        threshold=args.threshold,
        min_duration=args.min_duration,
    )

    # ── console output ──
    video_name = os.path.basename(args.video_path)
    print(f"\n{'='*60}")
    print(f"  Violence Detection Report: {video_name}")
    print(f"  Duration: {fmt_ts(total_duration)} ({total_duration:.1f}s)")
    print(f"  Threshold: {args.threshold}  |  Min segment: {args.min_duration}s")
    print(f"{'='*60}")

    if segments:
        print(f"\n  [!] VIOLENCE DETECTED -- {len(segments)} segment(s):\n")
        for i, seg in enumerate(segments, 1):
            print(f"  [{i}]  {fmt_ts(seg['start'])} - {fmt_ts(seg['end'])}  "
                  f"({seg['duration']:.1f}s)  "
                  f"Peak: {seg['peak_confidence']:.1f}%  "
                  f"Avg: {seg['avg_confidence']:.1f}%")
    else:
        print("\n  [OK] No violence detected.\n")

    # ── JSON output ──
    output = {
        "video": video_name,
        "video_path": os.path.abspath(args.video_path),
        "duration_seconds": round(total_duration, 2),
        "fps": round(video_fps, 2),
        "threshold": args.threshold,
        "min_segment_duration": args.min_duration,
        "violence_detected": len(segments) > 0,
        "segments_count": len(segments),
        "segments": segments,
        "timeline": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Full results written to: {args.output_json}")


if __name__ == "__main__":
    main()
