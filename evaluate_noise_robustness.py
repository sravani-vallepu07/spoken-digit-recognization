import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import audio_to_logmel, load_audio, load_records, pad_or_truncate
from model import SpokenDigitCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DATA_DIR = "data/recordings"
TEST_SPEAKER = "yweweler"


class NoisySpokenDigitDataset(Dataset):
    """
    Dataset that dynamically injects Gaussian noise at a specified SNR (in dB)
    into the peak-normalized waveform before feature extraction.
    """

    def __init__(self, records, snr_db=None, seed=42):
        self.records = records
        self.snr_db = snr_db
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filepath, label, speaker = self.records[idx]
        waveform = load_audio(filepath)

        # Peak normalization
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        # Add Gaussian noise at controlled SNR
        if self.snr_db is not None:
            # Signal power
            signal_power = torch.mean(waveform ** 2)
            if signal_power > 0:
                # Target noise power based on SNR_dB = 10 * log10(P_signal / P_noise)
                noise_power = signal_power / (10 ** (self.snr_db / 10.0))
                std = torch.sqrt(noise_power)
                noise = torch.randn_like(waveform) * std
                waveform = waveform + noise

                # Re-normalize after noise addition
                new_max = waveform.abs().max()
                if new_max > 0:
                    waveform = waveform / new_max

        waveform = pad_or_truncate(waveform)
        features = audio_to_logmel(waveform)
        return features, torch.tensor(label, dtype=torch.long)


def evaluate_snr(model, records, snr_db, batch_size=32):
    dataset = NoisySpokenDigitDataset(records, snr_db=snr_db)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            preds = model(feats).argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(
        description="Systematic noise robustness evaluation on held-out test speaker."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="noise_robustness_metrics.json",
        help="Path to save output metrics JSON.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      SYSTEMATIC NOISE ROBUSTNESS EVALUATION (SNR BENCHMARK)")
    print("=" * 70)

    # Load test records
    records = load_records(DATA_DIR)
    test_records = [r for r in records if r[2] == TEST_SPEAKER]
    print(f"Held-out test speaker : '{TEST_SPEAKER}' ({len(test_records)} clean samples)")
    print(f"Device                 : {DEVICE}")

    # Load model
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    model = SpokenDigitCNN().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # SNR levels to test (from clean to severe noise)
    snr_levels = [None, 30, 25, 20, 15, 10, 5, 0]
    results = []

    print("-" * 70)
    print(f"{'Condition':<15} | {'Accuracy':<10} | {'Macro F1':<10} | {'Degradation':<12} | Qualitative Impact")
    print("-" * 70)

    clean_acc = 0.0

    for snr in snr_levels:
        acc, prec, rec, f1 = evaluate_snr(model, test_records, snr_db=snr)
        condition_name = "Clean (No noise)" if snr is None else f"{snr} dB SNR"

        if snr is None:
            clean_acc = acc
            degradation_str = "0.00%"
            impact = "Clean baseline"
        else:
            deg = (clean_acc - acc) * 100
            degradation_str = f"-{deg:5.2f}%"
            if snr >= 25:
                impact = "Negligible effect"
            elif snr >= 20:
                impact = "Mild room noise"
            elif snr >= 15:
                impact = "Noticeable noise"
            elif snr >= 10:
                impact = "Moderate degradation"
            elif snr >= 5:
                impact = "Severe noise"
            else:
                impact = "Extreme noise (P_signal = P_noise)"

        print(
            f"{condition_name:<15} | {acc * 100:6.2f}%    | {f1 * 100:6.2f}%    | "
            f"{degradation_str:<12} | {impact}"
        )

        results.append(
            {
                "snr_db": snr if snr is not None else "clean",
                "condition": condition_name,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "accuracy_degradation": round(clean_acc - acc, 4) if snr is not None else 0.0,
                "qualitative_impact": impact,
            }
        )

    print("=" * 70)

    # Save metrics JSON
    with open(args.output, "w") as f:
        json.dump(
            {
                "test_speaker": TEST_SPEAKER,
                "total_test_samples": len(test_records),
                "checkpoint": CHECKPOINT_PATH,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results successfully saved to '{args.output}'.\n")


if __name__ == "__main__":
    main()
