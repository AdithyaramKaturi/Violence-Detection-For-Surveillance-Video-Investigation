"""
UBI-Fights Dataset Frame Extractor & Splitter (Enhanced with Hard Negative Mining)
================================================================================
Extracts balanced, disjoint video-level splits:
1. Strictly adheres to official UBI_FIGHTS/test_videos.csv for testing.
2. Partitions remaining videos at the VIDEO LEVEL into disjoint train and val sets.
3. Mines HARD NEGATIVES: Extracts non-violent frames (label 0) from fight videos
   during calm intervals, eliminating background/setting false alarms.
4. Mines normal non-violent frames from normal surveillance videos.
5. Extracts active violent frames (label 1) from fight videos.
"""

import os
import cv2
import pandas as pd
import numpy as np
import random
from typing import Set, List, Dict

def get_test_video_ids(test_csv_path: str = "UBI_FIGHTS/test_videos.csv") -> Set[str]:
    """Reads official UBI-Fights test video identifiers."""
    if not os.path.exists(test_csv_path):
        return set()
    with open(test_csv_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def extract_ubifights_dataset(
    ubi_root: str = "UBI_FIGHTS",
    output_dir: str = "data/dataset",
    train_violent_target: int = 1800,
    val_violent_target: int = 350,
    test_violent_target: int = 350,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    # Output directory layout
    splits = ["train", "val", "test"]
    classes = ["violent", "non_violent"]
    dir_map = {}
    for s in splits:
        for c in classes:
            d = os.path.join(output_dir, s, c)
            os.makedirs(d, exist_ok=True)
            dir_map[(s, c)] = d

    videos_fight_dir = os.path.join(ubi_root, "videos", "fight")
    videos_normal_dir = os.path.join(ubi_root, "videos", "normal")
    annotation_dir = os.path.join(ubi_root, "annotation")

    test_vids = get_test_video_ids(os.path.join(ubi_root, "test_videos.csv"))
    print(f"[INFO] Loaded {len(test_vids)} official test video IDs.")

    fight_files = [f for f in os.listdir(videos_fight_dir) if f.endswith(".mp4")]
    normal_files = [f for f in os.listdir(videos_normal_dir) if f.endswith(".mp4")]

    # Split videos into train, val, test at the VIDEO level
    test_fights = [f for f in fight_files if os.path.splitext(f)[0] in test_vids]
    train_val_fights = [f for f in fight_files if os.path.splitext(f)[0] not in test_vids]
    random.shuffle(train_val_fights)

    val_fight_count = max(20, int(len(train_val_fights) * 0.20))
    val_fights = train_val_fights[:val_fight_count]
    train_fights = train_val_fights[val_fight_count:]

    test_normals = [f for f in normal_files if os.path.splitext(f)[0] in test_vids]
    train_val_normals = [f for f in normal_files if os.path.splitext(f)[0] not in test_vids]
    random.shuffle(train_val_normals)

    val_normal_count = max(40, int(len(train_val_normals) * 0.20))
    val_normals = train_val_normals[:val_normal_count]
    train_normals = train_val_normals[val_normal_count:]

    print(f"[INFO] Video Split:")
    print(f"  Fight  : {len(train_fights)} train, {len(val_fights)} val, {len(test_fights)} test")
    print(f"  Normal : {len(train_normals)} train, {len(val_normals)} val, {len(test_normals)} test")

    def process_fight_videos(v_list: List[str], split: str, violent_target: int, hard_neg_target: int):
        extracted_v = 0
        extracted_hn = 0
        random.shuffle(v_list)

        for vid_file in v_list:
            if extracted_v >= violent_target and extracted_hn >= hard_neg_target:
                break

            base_name = os.path.splitext(vid_file)[0]
            csv_path = os.path.join(annotation_dir, f"{base_name}.csv")
            if not os.path.exists(csv_path):
                continue

            try:
                labels_df = pd.read_csv(csv_path, header=None)
                labels = labels_df[0].values
            except Exception:
                continue

            violent_indices = np.where(labels == 1)[0]
            non_violent_indices = np.where(labels == 0)[0]

            # Select sample indices
            sel_v = set()
            if len(violent_indices) > 0 and extracted_v < violent_target:
                step_v = max(1, len(violent_indices) // 18)
                sel_v = set(violent_indices[::step_v])

            sel_hn = set()
            if len(non_violent_indices) > 0 and extracted_hn < hard_neg_target:
                step_hn = max(1, len(non_violent_indices) // 18)
                sel_hn = set(non_violent_indices[::step_hn])

            if not sel_v and not sel_hn:
                continue

            vid_path = os.path.join(videos_fight_dir, vid_file)
            cap = cv2.VideoCapture(vid_path)
            fidx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if fidx in sel_v and extracted_v < violent_target:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{base_name}_v_f{fidx:06d}.jpg"
                    cv2.imwrite(os.path.join(dir_map[(split, "violent")], out_name), resized)
                    extracted_v += 1

                elif fidx in sel_hn and extracted_hn < hard_neg_target:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{base_name}_hn_f{fidx:06d}.jpg"
                    cv2.imwrite(os.path.join(dir_map[(split, "non_violent")], out_name), resized)
                    extracted_hn += 1

                fidx += 1

            cap.release()

        print(f"[{split.upper()}] Extracted {extracted_v} violent and {extracted_hn} hard-negative frames from fight videos.")
        return extracted_v, extracted_hn

    def process_normal_videos(v_list: List[str], split: str, normal_target: int):
        extracted_n = 0
        random.shuffle(v_list)

        for vid_file in v_list:
            if extracted_n >= normal_target:
                break

            base_name = os.path.splitext(vid_file)[0]
            vid_path = os.path.join(videos_normal_dir, vid_file)
            cap = cv2.VideoCapture(vid_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_f <= 10:
                cap.release()
                continue

            num_to_sample = min(12, max(3, normal_target - extracted_n))
            sample_indices = set(np.linspace(10, total_f - 10, num=num_to_sample, dtype=int))
            fidx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if fidx in sample_indices and extracted_n < normal_target:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{base_name}_n_f{fidx:06d}.jpg"
                    cv2.imwrite(os.path.join(dir_map[(split, "non_violent")], out_name), resized)
                    extracted_n += 1

                fidx += 1

            cap.release()

        print(f"[{split.upper()}] Extracted {extracted_n} normal non-violent frames.")
        return extracted_n

    # 1. TRAIN SET
    print("\n--- 1. Generating TRAIN Set ---")
    train_v, train_hn = process_fight_videos(train_fights, "train", train_violent_target, train_violent_target // 2)
    train_n = process_normal_videos(train_normals, "train", train_violent_target - train_hn)

    # 2. VAL SET
    print("\n--- 2. Generating VAL Set ---")
    val_v, val_hn = process_fight_videos(val_fights, "val", val_violent_target, val_violent_target // 2)
    val_n = process_normal_videos(val_normals, "val", val_violent_target - val_hn)

    # 3. TEST SET
    print("\n--- 3. Generating TEST Set (Official Test Split) ---")
    test_v, test_hn = process_fight_videos(test_fights, "test", test_violent_target, test_violent_target // 2)
    test_n = process_normal_videos(test_normals, "test", test_violent_target - test_hn)

    print("\n" + "=" * 60)
    print("UBI-FIGHTS DATASET EXTRACTION COMPLETE!")
    print(f"Train: Violent={train_v}, Non-Violent={train_hn + train_n} ({train_hn} hard neg + {train_n} normal)")
    print(f"Val  : Violent={val_v}, Non-Violent={val_hn + val_n} ({val_hn} hard neg + {val_n} normal)")
    print(f"Test : Violent={test_v}, Non-Violent={test_hn + test_n} ({test_hn} hard neg + {test_n} normal)")
    print("=" * 60)

if __name__ == "__main__":
    extract_ubifights_dataset()
