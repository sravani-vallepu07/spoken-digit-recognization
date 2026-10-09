import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import SpokenDigitDataset, load_records
from model import (
    MODELS,
    SpokenDigitCNN,
    SpokenDigitGAP,
    SpokenDigitMobile,
    count_parameters,
    estimated_parameter_size_mb,
    get_model,
)

try:
    from torchinfo import summary
except ImportError:
    summary = None


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data/recordings"
CHECKPOINT_DIR = "checkpoints"
RUNS_BASE_DIR = "runs/spoken_digit_augmented"

BATCH_SIZE = 32
EPOCHS = 18
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 8e-4
PATIENCE = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speaker_split(records):
    speakers = sorted({r[2] for r in records})

    if len(speakers) < 3:
        raise ValueError("Need at least 3 speakers for train/validation/test split.")

    # 4 speakers for training, 1 for validation, 1 for testing
    train_speakers = speakers[:-2]
    val_speakers = [speakers[-2]]
    test_speakers = [speakers[-1]]

    train_records = [r for r in records if r[2] in train_speakers]
    val_records = [r for r in records if r[2] in val_speakers]
    test_records = [r for r in records if r[2] in test_speakers]

    return (
        train_records,
        val_records,
        test_records,
        train_speakers,
        val_speakers,
        test_speakers,
    )


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    return avg_loss, accuracy, precision, recall, f1, cm


def train_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    clean_train_loader: DataLoader,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
):
    """Train and evaluate an individual model architecture."""
    seed_everything(seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" TRAINING MODEL: {model_name} ")
    print("=" * 70)

    model = get_model(model_name).to(DEVICE)
    total_params = count_parameters(model)
    macs_str = "N/A"
    if summary is not None:
        try:
            stats = summary(model, input_size=(1, 1, 40, 63), verbose=0)
            macs_str = f"{stats.total_mult_adds:,}"
        except Exception:
            pass

    print(f"Device: {DEVICE}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Estimated FP32 Size: {estimated_parameter_size_mb(model):.4f} MB")
    print(f"MACs / Operations: {macs_str}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    run_dir = os.path.join(RUNS_BASE_DIR, model_name)
    writer = SummaryWriter(run_dir)

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"best_model_{model_name}.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            y_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)

        val_loss, val_acc, val_precision, val_recall, val_f1, _ = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.add_scalar("F1/validation", val_f1, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "sample_rate": 8000,
                    "duration": 1.0,
                    "n_mels": 40,
                    "architecture": model_name,
                    "best_val_accuracy": best_val_acc,
                    "best_epoch": epoch,
                },
                checkpoint_file,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    writer.close()

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Clean train evaluation
    train_loss, train_acc, train_precision, train_recall, train_f1, _ = evaluate(
        model, clean_train_loader, criterion
    )

    # Unseen test speaker evaluation
    test_loss, test_acc, test_precision, test_recall, test_f1, cm = evaluate(
        model, test_loader, criterion
    )

    actual_size_kb = os.path.getsize(checkpoint_file) / 1024
    actual_size_mb = actual_size_kb / 1024

    print(f"\n--- Results for {model_name} ---")
    print(f"Best Val Accuracy : {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Clean Train Acc   : {train_acc:.4f}")
    print(f"Test Accuracy     : {test_acc:.4f}")
    print(f"Test Macro F1     : {test_f1:.4f}")
    print(f"Checkpoint Size   : {actual_size_kb:.2f} KB ({actual_size_mb:.4f} MB)")

    result = {
        "model_name": model_name,
        "parameters": total_params,
        "macs": macs_str,
        "checkpoint_file": checkpoint_file,
        "checkpoint_size_kb": actual_size_kb,
        "checkpoint_size_mb": actual_size_mb,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "test_f1_macro": test_f1,
        "confusion_matrix": cm.tolist(),
    }

    return result, model, cm


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare spoken digit recognition models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["SpokenDigitCNN", "SpokenDigitGAP", "SpokenDigitMobile", "all"],
        help="Select model to train ('all' to compare all architectures).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})",
    )
    args = parser.parse_args()

    seed_everything(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    records = load_records(DATA_DIR)
    print(f"Total recordings: {len(records)}")

    (
        train_records,
        val_records,
        test_records,
        train_speakers,
        val_speakers,
        test_speakers,
    ) = speaker_split(records)

    print("Train speakers:", train_speakers)
    print("Validation speakers:", val_speakers)
    print("Test speakers:", test_speakers)
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(val_records)}")
    print(f"Test samples: {len(test_records)}")

    split_info = {
        "train_speakers": train_speakers,
        "validation_speakers": val_speakers,
        "test_speakers": test_speakers,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "seed": SEED,
    }

    with open("split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    train_ds = SpokenDigitDataset(train_records, augment=True)
    val_ds = SpokenDigitDataset(val_records, augment=False)
    test_ds = SpokenDigitDataset(test_records, augment=False)
    clean_train_ds = SpokenDigitDataset(train_records, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    clean_train_loader = DataLoader(clean_train_ds, batch_size=BATCH_SIZE, shuffle=False)

    models_to_train = (
        ["SpokenDigitCNN", "SpokenDigitGAP", "SpokenDigitMobile"]
        if args.model == "all"
        else [args.model]
    )

    results = []
    models_dict = {}

    for model_name in models_to_train:
        res, trained_model, cm = train_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            clean_train_loader=clean_train_loader,
            epochs=args.epochs,
        )
        results.append(res)
        models_dict[model_name] = trained_model

    # Save overall best model to best_model.pth and keep metrics.json updated
    # Default priority: if baseline is present, save SpokenDigitCNN to best_model.pth or best test accuracy
    best_result = max(results, key=lambda x: x["test_accuracy"])
    
    # Save standard best_model.pth (saving the best performing model or SpokenDigitCNN)
    # If SpokenDigitCNN was trained, also make sure checkpoints/best_model.pth has SpokenDigitCNN for inference.py
    cnn_res = next((r for r in results if r["model_name"] == "SpokenDigitCNN"), results[0])
    torch.save(
        torch.load(cnn_res["checkpoint_file"], map_location="cpu"),
        os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    )

    # Save metrics.json for the baseline/active model
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(cnn_res, f, indent=2)

    # Save comprehensive comparison metrics json
    comparison_data = {
        "timestamp": "speaker_independent_benchmark",
        "split_info": split_info,
        "best_overall_model": best_result["model_name"],
        "results": results,
    }
    with open("comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # Print Comparative Summary Table
    print("\n" + "=" * 88)
    print("                      MODEL ACCURACY & EFFICIENCY COMPARISON TABLE")
    print("=" * 88)
    header = (
        f"{'Model Architecture':<20} | "
        f"{'Params':<10} | "
        f"{'Size (KB)':<10} | "
        f"{'Train Acc':<10} | "
        f"{'Val Acc':<10} | "
        f"{'Test Acc':<10} | "
        f"{'Test F1':<10}"
    )
    print(header)
    print("-" * 88)
    for r in results:
        row = (
            f"{r['model_name']:<20} | "
            f"{r['parameters']:<10,d} | "
            f"{r['checkpoint_size_kb']:<10.1f} | "
            f"{r['train_accuracy']*100:<9.2f}% | "
            f"{r['best_val_accuracy']*100:<9.2f}% | "
            f"{r['test_accuracy']*100:<9.2f}% | "
            f"{r['test_f1_macro']*100:<9.2f}%"
        )
        print(row)
    print("=" * 88)
    print(f" WINNING MODEL (Highest Test Accuracy): {best_result['model_name']} ({best_result['test_accuracy']*100:.2f}%)")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
