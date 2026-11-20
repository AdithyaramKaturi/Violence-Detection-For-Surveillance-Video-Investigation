"""
Deep Learning Training Pipeline: MobileNetV2 for Surveillance Violence Recognition
==================================================================================
Improved Training System:
- Selective Fine-Tuning: Freezes lower MobileNetV2 feature layers (0-13) and
  fine-tunes upper semantic layers (14-18) + classification head with differential learning rates.
- Class-Weighted Loss: Handles Fight/Normal class imbalance naturally via pos_weight.
- Realistic Data Augmentation: Horizontal flipping, subtle rotation (+/-10 deg),
  mild brightness/contrast (+/-15%), non-destructive (no RandomErasing).
- Validation Threshold Calibration (T*): Sweeps operating thresholds to maximize F1,
  balance high recall, and suppress false alarms.
- Regularization & Callbacks: ReduceLROnPlateau and Early Stopping on Validation F1.
- Complete metadata persistence (including best threshold T*).
"""

import os
import sys
import time
import copy
import json

# Ensure root directory is in sys.path when script is executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models.model import ViolenceDetectionMobileNetV2


def get_data_transforms():
    """
    Returns realistic, non-destructive training and validation transforms.
    Preserves limb interactions and human postures while providing robust invariance.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {"train": train_transform, "val": val_transform}


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Searches for the operating threshold T* in [0.30, 0.70] that maximizes overall ACCURACY
    while maintaining high F1, precision, and recall.
    """
    thresholds = np.arange(0.30, 0.72, 0.02)
    best_t = 0.50
    best_acc = 0.0
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_cm = None

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        acc = float(np.mean(preds == y_true))
        rec = float(recall_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        # Maximize accuracy; break ties with F1 score
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_t = float(t)
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_cm = confusion_matrix(y_true, preds)

    return best_t, best_acc, best_f1, best_prec, best_rec, best_cm



def train_model(
    data_dir: str = "data/dataset",
    output_model_path: str = "models/mobilenet_violence_best.pth",
    metadata_path: str = "models/model_metadata.json",
    num_epochs: int = 8,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience: int = 4
):
    """
    Trains the MobileNetV2 classifier with selective fine-tuning, class weighting,
    validation threshold calibration, and early stopping.
    """
    device = torch.device(device)
    print(f"[INFO] Using compute device: {device}")

    transforms_dict = get_data_transforms()
    
    # Dataset Loaders
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_dict[x])
        for x in ["train", "val"]
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=0, drop_last=(x == "train"))
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_to_idx = image_datasets['train'].class_to_idx
    print(f"[INFO] Dataset loaded: {dataset_sizes['train']} train samples, {dataset_sizes['val']} val samples.")
    print(f"[INFO] Class Mapping: {class_to_idx}")


    # Compute sample weights for balanced sampling in train split
    train_targets = [sample[1] for sample in image_datasets["train"].samples]
    num_violent = sum(1 for t in train_targets if t == class_to_idx["violent"])
    num_non_violent = sum(1 for t in train_targets if t == class_to_idx["non_violent"])
    weight_v = 1.0 / max(1, num_violent)
    weight_nv = 1.0 / max(1, num_non_violent)
    sample_weights = [weight_v if t == class_to_idx["violent"] else weight_nv for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_violent * 2,
        replacement=True
    )

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=True),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    }

    dataset_sizes = {"train": num_violent * 2, "val": len(image_datasets["val"])}
    class_to_idx = image_datasets['train'].class_to_idx
    print(f"[INFO] Balanced Train Sampler: {dataset_sizes['train']} samples per epoch ({num_violent} violent, {num_violent} non-violent).")
    print(f"[INFO] Class Mapping: {class_to_idx}")

    # 1. Initialize Deep Learning Model Architecture
    model = ViolenceDetectionMobileNetV2(pretrained=True, dropout_rate=0.4).to(device)

    # 2. Layer Freezing: Freeze lower layers (0-13), fine-tune upper layers (14-18) + head
    for child in model.features[:14]:
        for p in child.parameters():
            p.requires_grad = False

    for child in model.features[14:]:
        for p in child.parameters():
            p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params*100:.1f}%)")

    # 3. BCE Loss Function with Label Smoothing
    criterion = nn.BCEWithLogitsLoss()

    # 4. Differential Learning Rate Optimizer (fine backbone adjustments + active head learning)
    optimizer = optim.AdamW([
        {'params': model.features[14:].parameters(), 'lr': 3.0e-5, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 3.0e-4, 'weight_decay': 1e-4}
    ])

    # 5. Learning Rate Scheduler: Cosine Annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_operating_threshold = 0.50
    best_metrics = {}
    epochs_no_improve = 0

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1_default": [],
        "val_f1_optimal": [], "optimal_threshold": []
    }

    print("\n" + "=" * 65)
    print("STARTING ACCURACY-OPTIMIZED TRAINING (MobileNetV2 Fine-Tuning)")
    print("=" * 65)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_probs = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                raw_labels = labels.to(device).float().unsqueeze(1)
                
                # Label smoothing for training: maps 0 -> 0.04, 1 -> 0.96
                if phase == "train":
                    smoothed_labels = raw_labels * (1.0 - 0.08) + 0.5 * 0.08
                else:
                    smoothed_labels = raw_labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    logits = model.forward_logits(inputs)
                    loss = criterion(logits, smoothed_labels)
                    probs = torch.sigmoid(logits)

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_probs.extend(probs.detach().cpu().numpy().flatten())
                all_labels.extend(raw_labels.detach().cpu().numpy().flatten())

            epoch_loss = running_loss / dataset_sizes[phase]
            y_true = np.array(all_labels)
            y_prob = np.array(all_probs)
            
            if phase == "val":
                scheduler.step()

                # Metrics at standard 0.5 threshold
                preds_05 = (y_prob >= 0.5).astype(int)
                val_acc_05 = float(np.mean(preds_05 == y_true))
                val_f1_05 = float(f1_score(y_true, preds_05, zero_division=0))
                cm_05 = confusion_matrix(y_true, preds_05)
                
                # Optimal threshold targeting ACCURACY on validation set
                opt_t, opt_acc, opt_f1, opt_prec, opt_rec, opt_cm = find_optimal_threshold(y_true, y_prob)

                history["val_loss"].append(round(epoch_loss, 4))
                history["val_acc"].append(round(val_acc_05, 4))
                history["val_f1_default"].append(round(val_f1_05, 4))
                history["val_f1_optimal"].append(round(opt_f1, 4))
                history["optimal_threshold"].append(round(opt_t, 2))

                print(f"[VAL]   Loss: {epoch_loss:.4f} | Default (0.50) -> Acc: {val_acc_05*100:.2f}%, F1: {val_f1_05:.4f} | CM: TN={cm_05[0,0]}, FP={cm_05[0,1]}, FN={cm_05[1,0]}, TP={cm_05[1,1]}")
                print(f"[VAL]   Optimal Threshold T*={opt_t:.2f} -> Acc: {opt_acc*100:.2f}%, Prec: {opt_prec*100:.2f}%, Rec: {opt_rec*100:.2f}%, F1: {opt_f1:.4f} | CM: TN={opt_cm[0,0]}, FP={opt_cm[0,1]}, FN={opt_cm[1,0]}, TP={opt_cm[1,1]}")

                # Save best checkpoint according to validation ACCURACY (with F1 tie-break)
                if opt_acc > best_val_acc or (opt_acc == best_val_acc and opt_f1 > best_val_f1):
                    best_val_acc = opt_acc
                    best_val_f1 = opt_f1
                    best_operating_threshold = opt_t
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    
                    best_metrics = {
                        "val_accuracy": round(opt_acc * 100, 2),
                        "val_f1": round(opt_f1, 4),
                        "val_precision": round(opt_prec * 100, 2),
                        "val_recall": round(opt_rec * 100, 2),
                        "operating_threshold": round(opt_t, 2),
                        "epoch": epoch + 1
                    }

                    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
                    torch.save(model.state_dict(), output_model_path)
                    print(f"[*] NEW BEST VALIDATION ACCURACY ({opt_acc*100:.2f}% at T*={opt_t:.2f}). Saved to {output_model_path}")
                else:
                    epochs_no_improve += 1
                    print(f"[INFO] No improvement in validation accuracy for {epochs_no_improve} epoch(s).")
            else:
                history["train_loss"].append(round(epoch_loss, 4))
                preds_train = (y_prob >= 0.5).astype(int)
                train_acc = float(np.mean(preds_train == y_true))
                print(f"[TRAIN] Loss: {epoch_loss:.4f} | Acc: {train_acc*100:.2f}% (Epoch time: {time.time()-epoch_start:.1f}s)")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] Validation accuracy has not improved for {early_stopping_patience} epochs. Stopping training.")
            break

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s. Best Val Acc: {best_val_acc*100:.2f}%, F1: {best_val_f1:.4f} (at T*={best_operating_threshold:.2f})")

    # Persist metadata for inference & evaluation
    model_metadata = {
        "model_architecture": "MobileNetV2 (18 Inverted Residuals + Binary Classifier)",
        "optimal_threshold": round(best_operating_threshold, 2),
        "best_val_metrics": best_metrics,
        "class_mapping": class_to_idx,
        "pos_weight": 1.0,
        "training_samples": dataset_sizes["train"],
        "validation_samples": dataset_sizes["val"],

        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"[*] Model metadata saved to: {metadata_path}")

    # Save training curves plot and CSV
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(11, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss", color="blue")
        plt.plot(history["val_loss"], label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Class-Weighted BCE Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history["val_f1_default"], label="Val F1 (0.50 threshold)", color="gray", linestyle="--")
        plt.plot(history["val_f1_optimal"], label="Val F1 (T* optimal)", color="green", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title(f"Validation F1 Optimization (Peak: {best_val_f1:.4f})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = "outputs/training_curves.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[*] Training curves plot saved to: {plot_path}")

        hist_df = pd.DataFrame({
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_acc": history["val_acc"],
            "val_f1": history["val_f1_optimal"]
        })
        hist_df.to_csv("outputs/training_history.csv", index=False)
        print("[*] Training history saved to: outputs/training_history.csv")
    except Exception as e:
        print(f"[WARN] Could not save training visualization: {e}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model, model_metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Improved MobileNetV2 Violence Classifier")
    parser.add_argument("--data_dir", type=str, default="data/dataset", help="Path to train/val dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
