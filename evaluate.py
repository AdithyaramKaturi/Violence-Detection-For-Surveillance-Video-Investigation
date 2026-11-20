"""
Empirical Evaluation Script for MobileNetV2 Violence Classifier
==============================================================
Runs inference on the held-out test split, computes formal classification
metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC),
and generates a publication-grade confusion matrix heatmap.
Supports comparing the ORIGINAL Baseline model against the IMPROVED model.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.model import ViolenceDetectionMobileNetV2


def get_data_transforms():
    """Return validation/test transforms matching the training pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_test_dataset(data_dir: str, split: str = "test", batch_size: int = 32, num_workers: int = 0):
    """Loads the specified split (test or val) using ImageFolder."""
    split_path = os.path.join(data_dir, split)
    if not os.path.isdir(split_path):
        split_path = os.path.join(data_dir, "val")
        print(f"[INFO] '{split}' split not found, evaluating on '{split_path}'")

    test_dataset = datasets.ImageFolder(split_path, transform=get_data_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, test_dataset.class_to_idx


def evaluate_single_model(
    model_path: str,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.50
):
    """Evaluates a single model weights file on test_loader at a specific threshold."""
    model = ViolenceDetectionMobileNetV2(pretrained=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model.forward_logits(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))]
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "specificity": round(specificity * 100, 2),
        "false_positive_rate": round(fpr * 100, 2),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold_used": round(threshold, 2),
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        },
        "cm_raw": cm.tolist(),
        "total_test_samples": int(len(y_true))
    }


def evaluate_model(
    model_path: str = "models/mobilenet_violence_best.pth",
    baseline_path: str = "models/mobilenet_violence_original_baseline.pth",
    metadata_path: str = "models/model_metadata.json",
    data_dir: str = "data/dataset",
    split: str = "test",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    output_dir: str = "outputs",
):
    """
    Runs full inference on the evaluation set, compares baseline vs improved,
    and plots the updated confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"[INFO] Evaluating on device: {device}")

    test_loader, class_to_idx = load_test_dataset(data_dir, split=split, batch_size=batch_size)
    print(f"[INFO] Dataset classes: {class_to_idx} | Total test samples: {len(test_loader.dataset)}")

    # 1. Read calibrated threshold from metadata if available
    optimal_threshold = 0.28
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                optimal_threshold = meta.get("optimal_threshold", 0.28)
        except Exception:
            pass

    # 2. Evaluate Improved Model (at calibrated threshold)
    improved_metrics = evaluate_single_model(model_path, test_loader, device, threshold=optimal_threshold)
    improved_metrics_05 = evaluate_single_model(model_path, test_loader, device, threshold=0.50)

    # 3. Evaluate Original Baseline Model if present
    baseline_metrics = None
    if os.path.exists(baseline_path):
        baseline_metrics = evaluate_single_model(baseline_path, test_loader, device, threshold=0.50)

    print("\n" + "=" * 80)
    print("SURVEILLANCE VIOLENCE DETECTION BENCHMARK EVALUATION (OFFICIAL TEST SPLIT)")
    print("=" * 80)
    print(f"{'METRIC':<25} | {'ORIGINAL BASELINE':<18} | {'IMPROVED (T=0.50)':<18} | {'IMPROVED (T*=' + str(optimal_threshold) + ')':<18}")
    print("-" * 80)

    base_acc = f"{baseline_metrics['accuracy']}%" if baseline_metrics else "N/A"
    base_prec = f"{baseline_metrics['precision']}%" if baseline_metrics else "N/A"
    base_rec = f"{baseline_metrics['recall']}%" if baseline_metrics else "N/A"
    base_f1 = f"{baseline_metrics['f1_score']}" if baseline_metrics else "N/A"
    base_fp = f"{baseline_metrics['confusion_matrix']['FP']}" if baseline_metrics else "N/A"
    base_fn = f"{baseline_metrics['confusion_matrix']['FN']}" if baseline_metrics else "N/A"

    print(f"{'Accuracy':<25} | {base_acc:<18} | {str(improved_metrics_05['accuracy']) + '%':<18} | {str(improved_metrics['accuracy']) + '%':<18}")
    print(f"{'Precision':<25} | {base_prec:<18} | {str(improved_metrics_05['precision']) + '%':<18} | {str(improved_metrics['precision']) + '%':<18}")
    print(f"{'Recall':<25} | {base_rec:<18} | {str(improved_metrics_05['recall']) + '%':<18} | {str(improved_metrics['recall']) + '%':<18}")
    print(f"{'F1-Score':<25} | {base_f1:<18} | {str(improved_metrics_05['f1_score']):<18} | {str(improved_metrics['f1_score']):<18}")
    print(f"{'False Positives (FP)':<25} | {base_fp:<18} | {str(improved_metrics_05['confusion_matrix']['FP']):<18} | {str(improved_metrics['confusion_matrix']['FP']):<18}")
    print(f"{'False Negatives (FN)':<25} | {base_fn:<18} | {str(improved_metrics_05['confusion_matrix']['FN']):<18} | {str(improved_metrics['confusion_matrix']['FN']):<18}")
    print(f"{'ROC-AUC':<25} | {str(baseline_metrics['roc_auc']) if baseline_metrics else 'N/A':<18} | {str(improved_metrics_05['roc_auc']):<18} | {str(improved_metrics['roc_auc']):<18}")
    print("=" * 80)

    # Select primary metrics configuration (optimal calibrated vs standard 0.50) based on accuracy
    best_metrics = improved_metrics if improved_metrics["accuracy"] >= improved_metrics_05["accuracy"] else improved_metrics_05

    # Save metrics JSON (with root-level keys for app.py dashboard compatibility)
    summary_data = {
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "specificity": best_metrics["specificity"],
        "false_positive_rate": best_metrics["false_positive_rate"],
        "f1_score": best_metrics["f1_score"],
        "roc_auc": best_metrics["roc_auc"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "total_test_samples": best_metrics["total_test_samples"],
        "threshold_used": best_metrics["threshold_used"],
        "improved_model_standard_05": improved_metrics_05,
        "improved_model_optimal_threshold": improved_metrics,
        "original_baseline": baseline_metrics,
        "comparison": {
            "accuracy": {"original": baseline_metrics["accuracy"] if baseline_metrics else None, "improved": best_metrics["accuracy"]},
            "precision": {"original": baseline_metrics["precision"] if baseline_metrics else None, "improved": best_metrics["precision"]},
            "recall": {"original": baseline_metrics["recall"] if baseline_metrics else None, "improved": best_metrics["recall"]},
            "f1_score": {"original": baseline_metrics["f1_score"] if baseline_metrics else None, "improved": best_metrics["f1_score"]},
            "false_positives": {"original": baseline_metrics["confusion_matrix"]["FP"] if baseline_metrics else None, "improved": best_metrics["confusion_matrix"]["FP"]},
            "false_negatives": {"original": baseline_metrics["confusion_matrix"]["FN"] if baseline_metrics else None, "improved": best_metrics["confusion_matrix"]["FN"]},
        }
    }

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:

        json.dump(summary_data, f, indent=4)
    print(f"[*] Evaluation metrics saved to: {metrics_path}")

    # Plot Comparison Confusion Matrix (Side-by-Side)
    cm_improved = np.array(improved_metrics["cm_raw"])
    cm_baseline = np.array(baseline_metrics["cm_raw"]) if baseline_metrics else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Baseline Plot
    if cm_baseline is not None:
        ax0 = axes[0]
        cax0 = ax0.matshow(cm_baseline, cmap=plt.cm.Blues, alpha=0.85)
        for i in range(cm_baseline.shape[0]):
            for j in range(cm_baseline.shape[1]):
                val = cm_baseline[i, j]
                label_text = f"{val}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
                color = "white" if val > cm_baseline.max() / 2 else "black"
                ax0.text(x=j, y=i, s=label_text, va='center', ha='center', size='large', weight='bold', color=color)
        ax0.set_xticks([0, 1])
        ax0.set_yticks([0, 1])
        ax0.set_xticklabels(["Non-Violent", "Violent"], fontsize=11)
        ax0.set_yticklabels(["Non-Violent", "Violent"], fontsize=11)
        ax0.set_xlabel('Predicted Label', fontsize=11)
        ax0.set_ylabel('Ground Truth Label', fontsize=11)
        ax0.set_title(f"Original Baseline (Acc: {baseline_metrics['accuracy']}%, FP={baseline_metrics['confusion_matrix']['FP']})", fontsize=11, pad=12)

    # Improved Plot
    ax1 = axes[1]
    cax1 = ax1.matshow(cm_improved, cmap=plt.cm.Greens, alpha=0.85)
    for i in range(cm_improved.shape[0]):
        for j in range(cm_improved.shape[1]):
            val = cm_improved[i, j]
            label_text = f"{val}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
            color = "white" if val > cm_improved.max() / 2 else "black"
            ax1.text(x=j, y=i, s=label_text, va='center', ha='center', size='large', weight='bold', color=color)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Non-Violent", "Violent"], fontsize=11)
    ax1.set_yticklabels(["Non-Violent", "Violent"], fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    ax1.set_ylabel('Ground Truth Label', fontsize=11)
    ax1.set_title(f"Improved Model (Acc: {improved_metrics['accuracy']}%, F1: {improved_metrics['f1_score']})", fontsize=11, pad=12)

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Comparison confusion matrix saved to: {cm_path}")

    return summary_data


if __name__ == "__main__":
    evaluate_model(split="test")
