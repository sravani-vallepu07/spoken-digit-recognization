# Spoken Digit Recognition

A robust, speaker‑independent spoken‑digit classification system built with **PyTorch**, **torchaudio**, and **SoundFile**. It classifies audio recordings of digits (0‑9) using log‑Mel spectrograms and a lightweight 2‑D CNN (`SpokenDigitCNN`).

---

## Table of Contents

1. [Project Overview & Pipeline](#project-overview--pipeline)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Training](#training)
   - [Inference (dataset)](#inference-dataset)
   - [Inference (your voice)](#inference-your-voice)
   - [Baseline & Random‑Split Experiments](#baseline--random-split)
   - [Noise Robustness Benchmark](#noise-robustness)
   - [Error Analysis](#error-analysis)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License & Citations](#license--citations)

---

## Project Overview & Pipeline

The system transforms raw WAV files into a 10‑class probability distribution:

```text
Raw Audio WAV → Pre‑processing → Augmentation (train only) → Fixed‑length framing → Log‑Mel Spectrogram → Spectrogram Augmentation → 2‑D CNN → Softmax → Predicted Digit + Confidence
```

Key components:
- **Pre‑processing**: mono conversion, resampling to 8 kHz, peak‑amplitude normalisation.
- **Training‑time Augmentation**: random gain (0.8‑1.2), Gaussian noise (σ = 0.001‑0.005).
- **SpecAugment**: time‑masking & frequency‑masking applied on‑the‑fly.
- **Model**: `SpokenDigitCNN` – 156 k parameters, ~0.60 MiB checkpoint, 89.8‑91.0 % accuracy on a held‑out speaker.

---

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

The repository requires Python 3.9+ and the packages listed in `requirements.txt`.

---

## Usage

### Training

```bash
python train.py        # Train all models and generate comparison metrics
python train_random_split.py  # Train with an 80/10/10 random split
```

### Inference (dataset)

```bash
python inference.py    # Predict on the official test split (held‑out speaker)
```

### Inference (your voice)

Place your recordings (`myvoice0.wav` … `myvoice9.wav`) in the `myvoice/` folder and run:

```bash
python inference_with_my_voice.py
```

### Baseline & Random‑Split Experiments

- **Logistic Regression baseline**: `python baseline.py`
- **Random split benchmark**: `python train_random_split.py`

### Noise Robustness

```bash
python evaluate_noise_robustness.py
```
Evaluates model performance over a range of SNRs (30 dB → 0 dB).

### Error Analysis

```bash
python error_analysis.py
```
Produces a confusion matrix and a list of mis‑classified samples.

---

## Project Structure

```
spoken_digit_recognition_main/
├─ data/                     # Free Spoken Digit Dataset (FSDD)
│   └─ recordings/           # 3 000 wav files
├─ myvoice/                  # Your own recordings for inference
├─ checkpoints/              # Model checkpoints (≈0.6 MiB each)
├─ runs/                     # TensorBoard logs
├─ dataset.py                # Data loading & augmentation
├─ model.py                  # Model definitions (CNN, GAP, MobileNet)
├─ train.py                  # Training & benchmark script
├─ train_random_split.py     # Random split training script
├─ baseline.py               # Logistic regression baseline
├─ inference.py              # Dataset‑wide inference script
├─ inference_with_my_voice.py# Inference on `myvoice/`
├─ evaluate_noise_robustness.py
├─ error_analysis.py         # Confusion matrix & error report
├─ speaker_digit_analysis.py# Per‑speaker / per‑digit stats
├─ comparison_metrics.json   # Benchmarks for all models
├─ metrics.json              # Primary evaluation metrics
├─ baseline_metrics.json
├─ split_info.json           # Split‑description metadata
├─ requirements.txt
└─ README.md                 # This file
```

---

## Contributing

Contributions are welcome! Feel free to:
- Add new data augmentations.
- Experiment with alternative lightweight architectures.
- Improve the documentation or add unit tests.

Please open an issue or a pull request on the repository.

---

## License & Citations

- **Dataset**: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) – CC BY‑SA 4.0.
- **Framework**: Built with PyTorch and torchaudio.
- **Paper references** are listed in the original documentation.
