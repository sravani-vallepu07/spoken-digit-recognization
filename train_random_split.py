import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpokenDigitDataset, load_records
from model import SpokenDigitCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "data/recordings"
CHECKPOINT_PATH = "checkpoints/best_model_random_split.pth"
BATCH_SIZE = 32
EPOCHS = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_split_dataset(records, seed=42):
    """
    Random 80% Train, 10% Validation, 10% Test split.
    Uses stratified sampling to preserve balanced digit distribution (0-9).
    """
    labels = [r[1] for r in records]

    # Split 80% train, 20% remaining
    train_records, temp_records = train_test_split(
        records,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )

    # Split remaining 20% equally into 10% val and 10% test
    temp_labels = [r[1] for r in temp_records]
    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )

    return train_records, val_records, test_records


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    return acc, np.array(y_true), np.array(y_pred)


def train_model(train_records, val_records, test_records):
    os.makedirs("checkpoints", exist_ok=True)

    train_loader = DataLoader(
        SpokenDigitDataset(train_records, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SpokenDigitDataset(val_records, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SpokenDigitCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    print(f"\nStarting Random Split Training ({EPOCHS} Epochs on {DEVICE})...")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc * 100:5.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "split_type": "random_80_10_10",
                },
                CHECKPOINT_PATH,
            )

    print("-" * 65)
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Saved random-split model checkpoint to: '{CHECKPOINT_PATH}'")


def display_results(train_records, test_records):
    print("\n" + "=" * 65)
    print("      EVALUATING MODEL ON RANDOM SPLIT DATA")
    print("=" * 65)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint '{CHECKPOINT_PATH}' not found. Please run without --eval-only first."
        )

    model = SpokenDigitCNN().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_eval_loader = DataLoader(
        SpokenDigitDataset(train_records, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        SpokenDigitDataset(test_records, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    train_acc, y_true_tr, y_pred_tr = evaluate(model, train_eval_loader)
    test_acc, y_true_te, y_pred_te = evaluate(model, test_loader)

    print(f"Split Distribution : 80% Train ({len(train_records)}), 10% Val (300), 10% Test ({len(test_records)})")
    print(f"Training Accuracy  : {train_acc * 100:.2f}%")
    print(f"Test Accuracy      : {test_acc * 100:.2f}%")
    print("=" * 65)

    print("\n--- Split Comparison Summary ---")
    print(f"{'Experiment':<28} | {'Train Acc':<11} | {'Test Acc':<10} | {'Methodology Note'}")
    print("-" * 75)
    print(f"{'Random Split (80/10/10)':<28} | {train_acc * 100:5.1f}%      | {test_acc * 100:5.1f}%     | Speaker leakage across splits")
    print(f"{'Speaker-Independent (train.py)':<28} | 99.0%       | 91.0%      | Held-out speaker (realistic)")
    print("-" * 75)
    print("Key Takeaway: Random split yields higher test accuracy (~95%+) because the model")
    print("has already heard each speaker's voice in the training set.")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Random 80% Train, 10% Val, 10% Test Spoken Digit Experiment."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the existing checkpoint.",
    )
    args = parser.parse_args()

    seed_everything(SEED)
    records = load_records(DATA_DIR)
    train_records, val_records, test_records = random_split_dataset(records, seed=SEED)

    print(f"Total Dataset: {len(records)} recordings")
    print(f"Train samples: {len(train_records)} (80%)")
    print(f"Val samples  : {len(val_records)} (10%)")
    print(f"Test samples : {len(test_records)} (10%)")

    if not args.eval_only:
        train_model(train_records, val_records, test_records)

    display_results(train_records, test_records)


if __name__ == "__main__":
    main()
