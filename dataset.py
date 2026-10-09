from pathlib import Path
import re

import numpy as np
import soundfile as sf

import torch
import torchaudio


TARGET_SR = 8000
MAX_SECONDS = 1.0
TARGET_LENGTH = int(TARGET_SR * MAX_SECONDS)

N_MELS = 40
N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 128


def load_records(data_dir):
    """
    Read FSDD WAV files.

    Filename format:
        digit_speaker_index.wav

    Example:
        0_george_0.wav
    """

    data_dir = Path(data_dir)
    records = []

    pattern = re.compile(
        r"^([0-9])_([^_]+)_(\d+)\.wav$",
        re.IGNORECASE
    )

    for filepath in sorted(data_dir.glob("*.wav")):

        match = pattern.match(filepath.name)

        if not match:
            continue

        digit = int(match.group(1))
        speaker = match.group(2)

        records.append(
            (str(filepath), digit, speaker)
        )

    if not records:
        raise RuntimeError(
            f"No valid FSDD WAV files found in: {data_dir}"
        )

    return records


def load_audio(filepath):
    """
    Load WAV using soundfile instead of torchaudio.load().

    This avoids the TorchCodec/FFmpeg dependency.
    """

    audio, sample_rate = sf.read(
        filepath,
        dtype="float32"
    )

    # Convert stereo -> mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    waveform = torch.from_numpy(audio).float()

    # Resample only if source sample rate differs
    if sample_rate != TARGET_SR:

        waveform = waveform.unsqueeze(0)

        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SR,
        ).squeeze(0)

    return waveform


def pad_or_truncate(waveform):
    """
    Make every audio sample exactly 1 second.
    """

    if waveform.numel() > TARGET_LENGTH:

        waveform = waveform[:TARGET_LENGTH]

    elif waveform.numel() < TARGET_LENGTH:

        padding = TARGET_LENGTH - waveform.numel()

        waveform = torch.nn.functional.pad(
            waveform,
            (0, padding)
        )

    return waveform


def audio_to_logmel(waveform):
    """
    Convert waveform to normalized Log-Mel spectrogram.

    Output shape:
        [1, n_mels, time]
    """

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel = mel_transform(waveform)

    # Convert power spectrogram to dB
    mel_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80,
    )(mel)

    # Normalize each sample
    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-6
    )

    return mel_db.unsqueeze(0)


class SpokenDigitDataset(torch.utils.data.Dataset):

    def __init__(self, records, augment=False):

        self.records = records
        self.augment = augment
        if self.augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=4)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=6)

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        filepath, label, speaker = self.records[index]

        waveform = load_audio(filepath)

        # -----------------------------------------
        # Normalize amplitude
        # -----------------------------------------

        max_value = waveform.abs().max()

        if max_value > 0:
            waveform = waveform / max_value

        # -----------------------------------------
        # Training-only audio augmentation
        # -----------------------------------------

        if self.augment:

            # Random gain
            if np.random.random() < 0.5:

                gain = np.random.uniform(
                    0.8,
                    1.2
                )

                waveform = waveform * gain

            # Add small amount of Gaussian noise
            if np.random.random() < 0.5:

                noise_level = np.random.uniform(
                    0.001,
                    0.005
                )

                noise = (
                    torch.randn_like(waveform)
                    * noise_level
                )

                waveform = waveform + noise

        # -----------------------------------------
        # Make audio exactly 1 second
        # -----------------------------------------

        waveform = pad_or_truncate(waveform)

        # -----------------------------------------
        # Extract Log-Mel spectrogram
        # -----------------------------------------

        features = audio_to_logmel(waveform)

        # Training-only SpecAugment (Frequency & Time Masking)
        if self.augment:
            features = self.freq_mask(features)
            features = self.time_mask(features)

        return (
            features,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )