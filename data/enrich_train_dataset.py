"""
Enrich Training Dataset with Complete UBI-Fights Video Coverage
==============================================================
Extracts representative frames from all remaining training videos in UBI-Fights,
strictly ensuring:
1. Zero leakage: test_videos.csv and validation videos are never touched.
2. Broad coverage: all normal and fight surveillance videos are represented.
3. Realistic class imbalance: natural fight vs. normal distribution preserved.
"""

import os
import cv2
import pandas as pd
import numpy as np

def enrich_training_set(
    ubi_root: str = "UBI_FIGHTS",
    dataset_dir: str = "data/dataset",
    frames_per_normal: int = 2,
    frames_per_fight: int = 6
):
    train_violent_dir = os.path.join(dataset_dir, "train", "violent")
    train_non_violent_dir = os.path.join(dataset_dir, "train", "non_violent")
    os.makedirs(train_violent_dir, exist_ok=True)
    os.makedirs(train_non_violent_dir, exist_ok=True)

    # 1. Identify excluded test and validation videos
    test_csv_path = os.path.join(ubi_root, "test_videos.csv")
    test_vids = set(line.strip() for line in open(test_csv_path) if line.strip())

    val_f = os.listdir(os.path.join(dataset_dir, "val", "non_violent")) + \
            os.listdir(os.path.join(dataset_dir, "val", "violent"))
    val_vids = set('_'.join(x.split('_')[:6]) for x in val_f)

    # Videos already in training set
    existing_train_vids = set('_'.join(x.split('_')[:6]) for x in os.listdir(train_non_violent_dir) + os.listdir(train_violent_dir))

    print(f"[INFO] Excluded: {len(test_vids)} test videos, {len(val_vids)} val videos.")
    print(f"[INFO] Already existing in train: {len(existing_train_vids)} videos.")

    # 2. Process unused Normal videos
    normal_dir = os.path.join(ubi_root, "videos", "normal")
    all_normal_files = [f for f in os.listdir(normal_dir) if f.endswith(".mp4")]
    unused_normals = [
        f for f in all_normal_files 
        if os.path.splitext(f)[0] not in test_vids 
        and os.path.splitext(f)[0] not in val_vids
        and os.path.splitext(f)[0] not in existing_train_vids
    ]
    print(f"[INFO] Found {len(unused_normals)} new normal videos to sample.")

    added_normals = 0
    for idx, vid_file in enumerate(unused_normals):
        vid_base = os.path.splitext(vid_file)[0]
        vid_path = os.path.join(normal_dir, vid_file)
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 30:
            sample_points = [0.33, 0.66] if frames_per_normal == 2 else [0.25, 0.5, 0.75]
            for sp in sample_points:
                fpos = int(total_frames * sp)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{vid_base}_n_f{fpos:06d}.jpg"
                    cv2.imwrite(os.path.join(train_non_violent_dir, out_name), resized)
                    added_normals += 1
        cap.release()

    print(f"[INFO] Added {added_normals} new diverse normal frames to train/non_violent.")

    # 3. Process unused Fight videos
    fight_dir = os.path.join(ubi_root, "videos", "fight")
    annotation_dir = os.path.join(ubi_root, "annotation")
    all_fight_files = [f for f in os.listdir(fight_dir) if f.endswith(".mp4")]
    unused_fights = [
        f for f in all_fight_files 
        if os.path.splitext(f)[0] not in test_vids 
        and os.path.splitext(f)[0] not in val_vids
        and os.path.splitext(f)[0] not in existing_train_vids
    ]
    print(f"[INFO] Found {len(unused_fights)} new fight videos to sample.")

    added_violent = 0
    added_hard_neg = 0
    for idx, vid_file in enumerate(unused_fights):
        vid_base = os.path.splitext(vid_file)[0]
        csv_path = os.path.join(annotation_dir, f"{vid_base}.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            labels_df = pd.read_csv(csv_path, header=None)
            labels = labels_df[0].values
        except Exception:
            continue

        violent_indices = np.where(labels == 1)[0]
        non_violent_indices = np.where(labels == 0)[0]

        vid_path = os.path.join(fight_dir, vid_file)
        cap = cv2.VideoCapture(vid_path)

        # Extract violent frames
        if len(violent_indices) > 0:
            step = max(1, len(violent_indices) // frames_per_fight)
            chosen_v = violent_indices[::step][:frames_per_fight]
            for fpos in chosen_v:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fpos))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{vid_base}_v_f{fpos:06d}.jpg"
                    cv2.imwrite(os.path.join(train_violent_dir, out_name), resized)
                    added_violent += 1

        # Extract 2 hard negative frames from calm intervals
        if len(non_violent_indices) > 20:
            step_hn = max(1, len(non_violent_indices) // 2)
            chosen_hn = non_violent_indices[::step_hn][:2]
            for fpos in chosen_hn:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fpos))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    out_name = f"{vid_base}_hn_f{fpos:06d}.jpg"
                    cv2.imwrite(os.path.join(train_non_violent_dir, out_name), resized)
                    added_hard_neg += 1

        cap.release()

    print(f"[INFO] Added {added_violent} violent frames and {added_hard_neg} hard-negative frames to train.")
    
    total_train_v = len(os.listdir(train_violent_dir))
    total_train_nv = len(os.listdir(train_non_violent_dir))
    print(f"[STATUS] Final Training Dataset: {total_train_v} violent, {total_train_nv} non-violent frames.")
    print(f"[STATUS] Total Training Samples: {total_train_v + total_train_nv}")


def mine_train_hard_negatives(
    ubi_root: str = "UBI_FIGHTS",
    dataset_dir: str = "data/dataset",
    target_hn_per_video: int = 12
):
    """
    Specifically samples hard negative frames (ground truth label 0) from training fight videos
    to balance fight backgrounds with non-violent ground truth, eliminating false alarms.
    Strictly excludes all test and validation videos.
    """
    train_nv_dir = os.path.join(dataset_dir, "train", "non_violent")
    os.makedirs(train_nv_dir, exist_ok=True)

    test_csv_path = os.path.join(ubi_root, "test_videos.csv")
    test_vids = set(line.strip() for line in open(test_csv_path) if line.strip())

    val_f = os.listdir(os.path.join(dataset_dir, "val", "non_violent")) + \
            os.listdir(os.path.join(dataset_dir, "val", "violent"))
    val_vids = set('_'.join(x.split('_')[:6]) for x in val_f)

    # Fight videos already present in train
    train_v_files = os.listdir(os.path.join(dataset_dir, "train", "violent"))
    train_fight_vids = sorted(list(set('_'.join(x.split('_')[:6]) for x in train_v_files if x.startswith("F_"))))

    # Existing hard negative files in train
    existing_hn_files = set(os.listdir(train_nv_dir))

    print(f"[INFO] Mining hard negatives across {len(train_fight_vids)} training fight videos...")
    print(f"[INFO] Excluded: {len(test_vids)} test videos, {len(val_vids)} val videos.")

    annotation_dir = os.path.join(ubi_root, "annotation")
    videos_fight_dir = os.path.join(ubi_root, "videos", "fight")

    new_hn_extracted = 0

    for vid_base in train_fight_vids:
        if vid_base in test_vids or vid_base in val_vids:
            continue

        csv_path = os.path.join(annotation_dir, f"{vid_base}.csv")
        vid_path = os.path.join(videos_fight_dir, f"{vid_base}.mp4")
        if not os.path.exists(csv_path) or not os.path.exists(vid_path):
            continue

        try:
            labels_df = pd.read_csv(csv_path, header=None)
            labels = labels_df[0].values
        except Exception:
            continue

        non_violent_indices = np.where(labels == 0)[0]
        if len(non_violent_indices) < 20:
            continue

        # Count how many HN already exist for this video
        current_hn = [f for f in existing_hn_files if f.startswith(vid_base) and "_hn_" in f]
        needed = max(0, target_hn_per_video - len(current_hn))
        if needed <= 0:
            continue

        step = max(1, len(non_violent_indices) // needed)
        selected_indices = set(non_violent_indices[::step][:needed])

        cap = cv2.VideoCapture(vid_path)
        fpos = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fpos in selected_indices:
                out_name = f"{vid_base}_hn_f{fpos:06d}.jpg"
                if out_name not in existing_hn_files:
                    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(train_nv_dir, out_name), resized)
                    existing_hn_files.add(out_name)
                    new_hn_extracted += 1
            fpos += 1
        cap.release()

    print(f"[INFO] Successfully mined {new_hn_extracted} additional hard-negative frames!")
    total_hn = sum(1 for f in os.listdir(train_nv_dir) if "_hn_" in f)
    total_n = sum(1 for f in os.listdir(train_nv_dir) if "_n_" in f)
    total_nv_count = len(os.listdir(train_nv_dir))
    total_v = len(os.listdir(os.path.join(dataset_dir, "train", "violent")))
    print(f"[STATUS] Updated Train Composition: {total_v} violent, {total_nv_count} non-violent ({total_hn} hard-negatives, {total_n} normal).")


if __name__ == "__main__":
    mine_train_hard_negatives(target_hn_per_video=14)


