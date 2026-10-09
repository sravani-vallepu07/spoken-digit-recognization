import argparse
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from model import SpokenDigitCNN


SAMPLE_RATE = 8000
NUM_SAMPLES = 8000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=256, win_length=256, hop_length=128, n_mels=40
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def preprocess_audio(filepath):
    """Loads, resamples to 8kHz, trims ambient silence via VAD, and extracts log-Mel."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio not found: {filepath}")

    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Energy-based Voice Activity Detection (VAD) to trim ambient silence
    sig = waveform.squeeze(0).numpy()
    win = int(SAMPLE_RATE * 0.025)
    env = np.convolve(np.abs(sig), np.ones(win) / win, mode="same")
    peak_idx = int(np.argmax(env))
    threshold = env[peak_idx] * 0.08

    start_idx, end_idx = peak_idx, peak_idx
    while start_idx > 0 and env[start_idx] > threshold:
        start_idx -= 1
    while end_idx < len(env) - 1 and env[end_idx] > threshold:
        end_idx += 1

    buf = int(SAMPLE_RATE * 0.025)
    start_samp = max(0, start_idx - buf)
    end_samp = min(len(sig), end_idx + buf)
    speech = waveform[:, start_samp:end_samp]

    peak = speech.abs().max()
    if peak > 1e-8:
        speech = speech / peak

    # Pad or truncate to 1.0s (8000 samples)
    if speech.shape[1] < NUM_SAMPLES:
        speech = F.pad(speech, (0, NUM_SAMPLES - speech.shape[1]))
    else:
        speech = speech[:, :NUM_SAMPLES]

    mel = to_db(mel_spec(speech))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0), (end_samp - start_samp) / SAMPLE_RATE, len(sig) / SAMPLE_RATE


def load_model(checkpoint_path="checkpoints/best_model.pth"):
    model = SpokenDigitCNN().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def predict(filepath, model):
    feats, speech_dur, total_dur = preprocess_audio(filepath)
    with torch.no_grad():
        logits = model(feats.to(DEVICE))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

    top_vals, top_idxs = torch.topk(probs, 3, dim=1)
    top3 = [(top_idxs[0][k].item(), top_vals[0][k].item()) for k in range(3)]
    return pred.item(), conf.item(), speech_dur, total_dur, top3


def run_batch(myvoice_dir="myvoice", model=None):
    if model is None:
        model = load_model()

    files = sorted(
        Path(myvoice_dir).glob("*.wav"),
        key=lambda p: int(re.search(r"(\d+)", p.name).group(1)) if re.search(r"(\d+)", p.name) else p.name,
    )
    if not files:
        print(f"No WAV files in '{myvoice_dir}'.")
        return

    print("\n" + "=" * 65)
    print(f"   RUNNING INFERENCE ON RECORDINGS IN '{myvoice_dir}'")
    print("=" * 65)
    print(f"{'Filename':<18} | {'Actual':<6} | {'Predicted':<9} | {'Confidence':<10} | {'Status'}")
    print("-" * 65)

    correct = 0
    for p in files:
        pred, conf, _, _, _ = predict(str(p), model)
        match = re.search(r"(\d+)", p.name)
        actual = int(match.group(1)) if match else "-"
        is_ok = actual == pred
        if is_ok:
            correct += 1
        status = "CORRECT [OK]" if is_ok else "MISMATCH [X]"
        print(f"{p.name:<18} | {str(actual):<6} | {pred:<9} | {conf * 100:6.2f}%    | {status}")

    acc = (correct / len(files)) * 100
    print("=" * 65)
    print(f"Batch Accuracy: {correct}/{len(files)} ({acc:.1f}%)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Spoken digit recognition for voice recordings.")
    parser.add_argument("audio", nargs="?", default=None, help="WAV file path. If omitted, runs on myvoice/.")
    args = parser.parse_args()

    model = load_model()
    if args.audio is None:
        run_batch("myvoice", model)
    else:
        pred, conf, sp_dur, tot_dur, top3 = predict(args.audio, model)
        print("\n" + "=" * 38)
        print(f"Audio file     : {args.audio}")
        print(f"Total duration : {tot_dur:.3f} sec")
        print(f"Speech duration: {sp_dur:.3f} sec")
        print(f"Predicted digit: {pred}")
        print(f"Confidence     : {conf * 100:.2f}%")
        print(f"Top 3 ranks    : {', '.join([f'Digit {d}: {c*100:.1f}%' for d, c in top3])}")
        print("=" * 38 + "\n")


if __name__ == "__main__":
    main()
