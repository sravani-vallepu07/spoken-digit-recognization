import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio

import soundfile as sf
import numpy as np

from model import SpokenDigitCNN


SAMPLE_RATE = 8000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 40

CONFIDENCE_THRESHOLD = 0.60

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Mel Spectrogram
# --------------------------------------------------

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=256,
    win_length=256,
    hop_length=128,
    n_mels=N_MELS,
)

db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power",
    top_db=80,
)


# --------------------------------------------------
# Audio Preprocessing
# --------------------------------------------------

def preprocess_audio(filepath):

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Audio file not found: {filepath}"
        )

    # Load audio using soundfile
    audio, sr = sf.read(
        filepath,
        dtype="float32"
    )

    if audio.size == 0:
        raise ValueError(
            "The audio file is empty."
        )

    # Convert stereo audio to mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    waveform = torch.from_numpy(
        audio
    ).float().unsqueeze(0)

    # Resample to 8000 Hz
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            SAMPLE_RATE
        )

    # Check silence
    peak = waveform.abs().max()

    if peak <= 1e-8:
        raise ValueError(
            "The audio appears to contain silence."
        )

    # Normalize amplitude
    waveform = waveform / peak

    # Original duration
    duration_seconds = (
        waveform.shape[1] / SAMPLE_RATE
    )

    # Pad or truncate to exactly 1 second
    if waveform.shape[1] < NUM_SAMPLES:

        waveform = F.pad(
            waveform,
            (
                0,
                NUM_SAMPLES - waveform.shape[1]
            )
        )

    else:

        waveform = waveform[:, :NUM_SAMPLES]

    # Convert audio -> Mel Spectrogram
    mel = mel_transform(waveform)

    # Convert to decibel scale
    mel = db_transform(mel)

    # Normalize features
    mel = (
        mel - mel.mean()
    ) / (
        mel.std() + 1e-6
    )

    # Add channel dimension
    # [1, 40, 63] -> [1, 1, 40, 63]
    return mel.unsqueeze(0), duration_seconds


# --------------------------------------------------
# Main Inference
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict a spoken digit "
            "from an unseen WAV file."
        )
    )

    parser.add_argument(
        "audio",
        help="Path to a WAV file."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=(
            "Confidence threshold for "
            "an uncertainty warning."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------

    checkpoint_path = (
        "checkpoints/best_model.pth"
    )

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "Checkpoint not found. "
            "Run train.py first."
        )

    # --------------------------------------------------
    # Preprocess input audio
    # --------------------------------------------------

    features, original_duration = (
        preprocess_audio(args.audio)
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = SpokenDigitCNN().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # Supports project checkpoint dictionary
    # and plain state_dict
    state_dict = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(
            features.to(DEVICE)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = (
            probabilities.max(dim=1)
        )

    confidence = confidence.item()

    prediction = prediction.item()

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    print(
        "\n========== INFERENCE =========="
    )

    print(
        f"Audio duration : "
        f"{original_duration:.3f} sec"
    )

    print(
        f"Predicted digit: "
        f"{prediction}"
    )

    print(
        f"Confidence     : "
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------
    # Confidence warning
    # --------------------------------------------------

    if confidence < args.threshold:

        print(
            f"Warning: confidence is below "
            f"the {args.threshold * 100:.0f}% threshold."
        )

        print(
            "Consider providing a clearer "
            "spoken-digit recording."
        )


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()