"""
Classical Logistic Regression baseline.

Uses the same Log-Mel preprocessing as the CNN and performs
speaker-aware evaluation. Run after placing FSDD recordings in
data/recordings/.
"""

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from dataset import SpokenDigitDataset, load_records
from train import speaker_split


DATA_DIR = "data/recordings"


def extract_features(records):
    ds = SpokenDigitDataset(records)
    X, y = [], []

    for i in range(len(ds)):
        mel, label = ds[i]
        X.append(mel.numpy().reshape(-1))
        y.append(int(label))

    return np.asarray(X), np.asarray(y)


def main():
    records = load_records(DATA_DIR)

    (
        train_records,
        val_records,
        test_records,
        train_speakers,
        val_speakers,
        test_speakers,
    ) = speaker_split(records)

    X_train, y_train = extract_features(train_records)
    X_test, y_test = extract_features(test_records)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )

    result = {
        "model": "Logistic Regression",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=list(range(10))
        ).tolist(),
        "train_speakers": train_speakers,
        "validation_speakers": val_speakers,
        "test_speakers": test_speakers,
    }

    with open("baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
