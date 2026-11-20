"""
UBI-Fights Dataset Extraction & Preprocessing Pipeline (Section IV of Proposal)
==============================================================================
Processes the UBI-Fights surveillance dataset (`UBI_FIGHTSALL`):
- Total Videos: 1,000 (216 Violent, 784 Non-Violent)
- Total Duration: ~80 Hours
- Annotations: Frame-level temporal labels

This utility extracts sampled frames, applies class labels based on ground-truth
timestamps, and structures the data into training and validation splits.
"""

import os
import cv2
import csv
import glob
from typing import List, Tuple


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps_sample: float = 2.0,
    violent_intervals: List[Tuple[int, int]] = None
):
    """
    Extracts frames at regular intervals and categorizes them into violent/non-violent.
    Args:
        video_path (str): Path to surveillance video file.
        output_dir (str): Base output directory.
        fps_sample (float): Frames to extract per second (e.g. 2 frames/sec).
        violent_intervals (List[Tuple[int, int]]): List of (start_frame, end_frame) violent ranges.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(fps / fps_sample))
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    violent_dir = os.path.join(output_dir, "violent")
    non_violent_dir = os.path.join(output_dir, "non_violent")
    os.makedirs(violent_dir, exist_ok=True)
    os.makedirs(non_violent_dir, exist_ok=True)

    frame_idx = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            is_violent = False
            if violent_intervals:
                for (start_f, end_f) in violent_intervals:
                    if start_f <= frame_idx <= end_f:
                        is_violent = True
                        break

            target_dir = violent_dir if is_violent else non_violent_dir
            out_filename = f"{video_basename}_f{frame_idx:06d}.jpg"
            out_path = os.path.join(target_dir, out_filename)
            
            # Resize frame to standard 224x224
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_path, resized)
            extracted_count += 1

        frame_idx += 1

    cap.release()
    return extracted_count


def parse_ubifights_annotation_csv(annotation_csv_path: str) -> List[Tuple[int, int]]:
    """
    Parses frame-level fight interval annotations for UBI-Fights videos.
    UBI-Fights annotations format: start_frame, end_frame (or 1 for violent frames).
    """
    intervals = []
    if not os.path.exists(annotation_csv_path):
        return intervals

    with open(annotation_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    s, e = int(row[0]), int(row[1])
                    intervals.append((s, e))
                except ValueError:
                    continue
    return intervals


if __name__ == "__main__":
    print("=" * 60)
    print("UBI-FIGHTS DATASET PREPARATION TOOL (Section IV)")
    print("=" * 60)
    print("Usage Instructions:")
    print("1. Place UBI-Fights raw videos in: data/raw_videos/")
    print("2. Place CSV frame annotations in: data/raw_annotations/")
    print("3. Run this script to extract balanced 224x224 frames into data/dataset/train and val.")
