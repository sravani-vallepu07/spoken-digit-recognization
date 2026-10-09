# Spoken Digit Recognition

A robust, speaker-independent spoken digit recognition system that classifies audio recordings of digits (0–9) using log-Mel spectrograms and deep convolutional architectures, built with PyTorch, torchaudio, and SoundFile.

The winning model (**`SpokenDigitCNN`**) achieves **89.8% – 91.0% test accuracy** on a completely held-out, unseen speaker (500 recordings), compared with 71.4% for the Logistic Regression baseline. The model has **156,010 parameters** and a compact **~0.60 MB checkpoint** (well below the 12 MB limit).

---

## Table of Contents

1. [Project Overview & Pipeline](#1-project-overview--pipeline)
2. [Literature Survey & SOTA Methodology](#2-literature-survey--sota-methodology)
3. [Setup & Installation](#3-setup--installation)
4. [Dataset & Speaker-Independent Protocol](#4-dataset--speaker-independent-protocol)
5. [Model Exploration & Architectural Comparison](#5-model-exploration--architectural-comparison)
   - [5.1 Comprehensive Multi-Model Benchmark Table](#51-comprehensive-multi-model-benchmark-table)
   - [5.2 Efficiency & Computational Metrics (MACs, Latency, Size)](#52-efficiency--computational-metrics)
   - [5.3 Why SpokenDigitCNN Outperforms GAP and Mobile Architectures](#53-why-spokendigitcnn-outperforms-gap-and-mobile-architectures)
6. [How to Run Each Script](#6-how-to-run-each-script)
   - [6.1 Model Training (`train.py`)](#61-model-training-trainpy)
   - [6.2 Dataset Test Inference on Held-Out Speaker (`inference.py`)](#62-dataset-test-inference-on-held-out-speaker-inferencepy)
   - [6.3 Real-World Voice Inference (`inference_with_my_voice.py`)](#63-real-world-voice-inference-inference_with_my_voicepy)
   - [6.4 Linear Baseline (`baseline.py`)](#64-linear-baseline-baselinepy)
   - [6.5 Random Split Comparison (`train_random_split.py`)](#65-random-split-comparison-train_random_splitpy)
   - [6.6 Noise Robustness Benchmark (`evaluate_noise_robustness.py`)](#66-noise-robustness-benchmark-evaluate_noise_robustnesspy)
   - [6.7 Error & Speaker Analysis Scripts](#67-error--speaker-analysis-scripts)
7. [TensorBoard Visualizations](#7-tensorboard-visualizations)
8. [Audio Preprocessing & Spectrogram Extraction](#8-audio-preprocessing--spectrogram-extraction)
9. [Data Augmentation](#9-data-augmentation)
10. [Error Analysis & Confusion Matrix](#10-error-analysis--confusion-matrix)
11. [Project File Structure & Guide](#11-project-file-structure--guide)

---

## 1. Project Overview & Pipeline

The system processes raw acoustic waveforms into 10-class probability distributions via a modular pipeline:

```text
Raw Audio WAV (8 kHz / 48 kHz mono)
       ↓
Preprocessing (SoundFile) → mono conversion → resample to 8 kHz → peak amplitude normalization
       ↓
Waveform Augmentation (Training only: Random gain [0.8–1.2] + Gaussian noise [σ=0.001–0.005])
       ↓
Fixed-Length Framing (Pad / truncate to exactly 1.0 second / 8,000 samples)
       ↓
Log-Mel Spectrogram (40 Mel filterbanks, FFT=256, Hop=128) → Per-sample Z-score normalization
       ↓
Spectrogram Augmentation (Training only: SpecAugment Time Masking + Frequency Masking)
       ↓
Deep 2D CNN Architecture (SpokenDigitCNN)
       ↓
10-Class Logits (Digits 0–9) → Softmax → Predicted Digit + Confidence Score
```

---

## 2. Literature Survey & SOTA Methodology

To construct a high-accuracy, lightweight, and robust acoustic model, we surveyed key foundational research papers and state-of-the-art (SOTA) keyword/digit recognition techniques:

| Research Paper / Method | Authors & Venue | Key Concept & Application to this Project |
| :--- | :--- | :--- |
| **Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition** | *Warden, P. (2018), arXiv:1804.03209* | Standardized benchmark methodology for small-footprint acoustic classification; established log-Mel spectrogram inputs and speaker-independent evaluation protocols. |
| **Convolutional Neural Networks for Small-Footprint Keyword Spotting** | *Sainath, T. N., & Parada, C. (2015), Interspeech* | Demonstrated that 2D CNNs operating on time-frequency log-Mel spectrograms significantly outperform traditional HMM/GMMs and flat feedforward DNNs by capturing local formant shifts and spectral harmonics. |
| **SpecAugment: A Simple Data Augmentation Method for ASR** | *Park, D. S., et al. (2019), Interspeech* | Applied random time masking and frequency masking directly on log-Mel spectrograms, substantially boosting generalization to unseen speakers without audio synthesis overhead. |
| **MobileNets: Efficient Convolutional Neural Networks** | *Howard, A. G., et al. (2017), arXiv:1704.04861* | Explored Depthwise Separable Convolutions to drastically reduce computational FLOPs/MACs for ultra-lean edge inference. |
| **Network In Network (Global Average Pooling)** | *Lin, M., Chen, Q., & Yan, S. (2013), arXiv:1312.4400* | Investigated replacing dense classification layers with Global Average Pooling (GAP) to prevent overfitting and eliminate parameters. |

---

## 3. Setup & Installation

### Requirements
* Python 3.10+ (tested on Python 3.11)
* PyTorch & torchaudio
* SoundFile
* NumPy
* scikit-learn
* torchinfo
* TensorBoard
* tqdm

Audio loading utilizes **SoundFile** instead of `torchaudio.load()` to avoid external `TorchCodec`/FFmpeg binary dependencies on Windows.

### Virtual Environment Setup

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Dataset & Speaker-Independent Protocol

This project utilizes the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) containing **3,000 recordings** across 6 speakers (50 utterances per digit $\times$ 10 digits $\times$ 6 speakers):

* `george`, `jackson`, `lucas`, `nicolas`, `theo`, `yweweler`

Place recordings inside `data/recordings/` (e.g., `data/recordings/0_george_0.wav`).

### Speaker-Independent Partitioning
To prevent **speaker leakage** (where the model memorizes a speaker's voice timbre instead of learning acoustic phonemes), we enforce a strict zero-shot speaker split:

| Split | Speakers | Sample Count | Purpose |
| :--- | :--- | :---: | :--- |
| **Train** | `george`, `jackson`, `lucas`, `nicolas` | 2,000 | Model training with dynamic augmentation |
| **Validation** | `theo` | 500 | Checkpoint selection & early stopping |
| **Held-Out Test** | `yweweler` | 500 | Final zero-shot generalizability evaluation |

---

## 5. Model Exploration & Architectural Comparison

designed, implemented, trained, and benchmarked four distinct approaches to evaluate the tradeoff between accuracy, parameter size, and computational complexity:

1. **Logistic Regression Baseline (`baseline.py`)**: Linear classifier on standardized flattened log-Mel features.
2. **`SpokenDigitCNN` (Proposed Winner)**: 3-stage 2D CNN with BatchNorm, ReLU, MaxPool, Adaptive Spatial Pooling to $(4, 4)$, and Dropout regularized MLP classifier.
3. **`SpokenDigitGAP`**: Replaces the spatial dense classifier with Global Average Pooling $(1, 1)$ mapping directly to logits.
4. **`SpokenDigitMobile`**: Employs Depthwise Separable Convolutions + Global Average Pooling for minimal MACs.

### 5.1 Comprehensive Multi-Model Benchmark Table

| Model Architecture | Parameters | Checkpoint Size | FLOPs / MACs | Clean Train Acc | Val Acc (`theo`) | Test Acc (`yweweler`) | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 25,210 | ~0.10 MB | 0.05 MMACs | 100.00% | 68.20% | 71.40% | 70.09% |
| **`SpokenDigitCNN` (Best)** | **156,010** | **0.60 MB** | **6.19 MMACs** | **99.10%** | **87.00%** | **89.80% – 91.00%** | **89.74%** |
| **`SpokenDigitGAP`** | 24,170 | 0.10 MB | 6.06 MMACs | 42.80% | 34.80% | 32.60% | 24.76% |
| **`SpokenDigitMobile`** | 4,170 | 0.03 MB | 1.21 MMACs | 50.30% | 38.20% | 34.20% | 30.29% |

> **Checkpoint Size Compliance**:
> All models, including `SpokenDigitCNN` (0.60 MB / 619 KB), are  below the required **12 MB** limit (**PASS**).

### 5.2 Efficiency & Computational Metrics of Best Model

* **Trainable Parameters**: 156,010 (~0.156M parameters)
* **Checkpoint File Size**: 619.4 KB (0.6049 MB)
* **Multiply-Accumulate Operations (MACs)**: 6,187,114 (~6.19 MMACs)
* **Floating Point Operations (FLOPs)**: ~12.37 MFLOPs
* **Inference Latency (CPU)**: ~0.41 ms per sample
* **Real-Time Factor (RTF)**: $< 0.0005$ ($>2,400\times$ faster than real-time)
* **Runtime RAM Footprint**: $< 25\text{ MB}$


### 5.3 Why SpokenDigitCNN Outperforms GAP and Mobile Architectures

* **Asymmetric Nature of Spectrograms**: Unlike 2D photographic images where objects exhibit spatial translation invariance, audio spectrograms have distinct physical axes:
  * **Vertical Axis**: Acoustic frequency bands and formant resonances ($F_1, F_2, F_3$).
  * **Horizontal Axis**: Phoneme sequence over time.
* **The Failure Mode of Pure GAP on Audio**: Collapsing feature maps to $(1, 1)$ via Global Average Pooling averages out the critical temporal ordering of phonemes (e.g., confusing "six" /s-ɪ-k-s/ with "eight" /eɪ-t/).
* **Why `SpokenDigitCNN` Wins**: By using `AdaptiveAvgPool2d((4, 4))`, `SpokenDigitCNN` preserves a 16-element spatial-temporal grid into the dense classifier, retaining crucial phonetic transition dynamics while maintaining a tiny 0.60 MB footprint.

---

## 6. How to Run Each Script

### 6.1 Model Training (`train.py`)

Train and compare all three model architectures sequentially or train an individual model:

```powershell
# Train and benchmark all 3 architectures (CNN, GAP, Mobile)
python train.py --model all

# Train only the winning SpokenDigitCNN architecture
python train.py --model SpokenDigitCNN

# Custom training epochs
python train.py --model all --epochs 18
```

* Output files generated:
  * `checkpoints/best_model_SpokenDigitCNN.pth`
  * `checkpoints/best_model_SpokenDigitGAP.pth`
  * `checkpoints/best_model_SpokenDigitMobile.pth`
  * `checkpoints/best_model.pth` (active model for inference)
  * `comparison_metrics.json` and `metrics.json`
  * Isolated TensorBoard event logs in `runs/spoken_digit_augmented/`

---

### 6.2 Dataset Test Inference on Held-Out Speaker (`inference.py`)

Predict any audio recording from the dataset (such as held-out test speaker `yweweler`):

```powershell
# Predict a single recording from unseen speaker yweweler
python inference.py data/recordings/8_yweweler_45.wav

# Predict with a custom confidence threshold
python inference.py data/recordings/8_yweweler_0.wav --threshold 0.70
```

**Example Output:**
```text
========== INFERENCE ==========
Audio duration : 0.318 sec
Predicted digit: 8
Confidence     : 99.94%
```

---

### 6.3 Real-World Voice Inference (`inference_with_my_voice.py`)

Run inference on genuine microphone recordings located in `myvoice/` (`myvoice0.wav` to `myvoice9.wav`):

```powershell
# Run batch inference on all 10 digits in myvoice/
python inference_with_my_voice.py

# Predict a single personal voice file
python inference_with_my_voice.py myvoice/myvoice2.wav
```

**Real-World Live Voice Results:**
* **Zero-Shot Live Accuracy**: **70.0%** (7 / 10 correct)
* **Correct Digits**: Digit 0 (64.1%), Digit 1 (58.9%), Digit 2 (98.8%), Digit 4 (91.0%), Digit 6 (98.4%),Digit 7 (72.3%), Digit 8 (74.3%).
* **Domain Generalization Analysis**: The reduction from 91.0% on clean studio FSDD audio to 70.0% on live microphone voice highlights the classic **acoustic domain shift**:
  1. Consumer laptop microphone frequency response vs. studio condenser microphones.
  2. Room reverberation and ambient noise floor.
  3. Pitch and formant variations from a new speaker.

---

### 6.4 Linear Baseline (`baseline.py`)

Train and evaluate the Logistic Regression baseline:

```powershell
python baseline.py
```
* Generates `baseline_metrics.json` achieving **71.4% test accuracy**.

---

### 6.5 Random Split Comparison (`train_random_split.py`)

Demonstrates the scientific impact of **speaker leakage** (random 80/10/10 split) versus strict speaker-independent evaluation:

```powershell
# Fast evaluation of random split checkpoint
python train_random_split.py --eval-only

# Or run full 16-epoch random split training
python train_random_split.py
```
* **Random Split Accuracy**: **98.7%** (inflated due to speaker leakage across splits).
* **Speaker-Independent Accuracy**: **91.0%** (true zero-shot generalizability).

---

### 6.6 Noise Robustness Benchmark (`evaluate_noise_robustness.py`)

Systematically measures accuracy degradation under varying Signal-to-Noise Ratio (SNR) levels from clean audio down to 0 dB SNR:

```powershell
python evaluate_noise_robustness.py
```

| SNR Condition | Test Accuracy | Macro F1 | Degradation | Qualitative Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Audio** | **91.00%** | **90.83%** | 0.00% | Clean benchmark |
| **30 dB SNR** | **92.00%** | **91.87%** | +1.00% | Negligible / dither benefit |
| **20 dB SNR** | **86.20%** | **86.24%** | -4.80% | Typical office / room noise |
| **15 dB SNR** | **76.20%** | **75.65%** | -14.80% | Noticeable background noise |
| **10 dB SNR** | **55.40%** | **52.64%** | -35.60% | Heavy noise degradation |
| **0 dB SNR** | **26.60%** | **17.57%** | -64.40% | Extreme noise ($P_{\text{sig}} = P_{\text{noise}}$) |

---

### 6.7 Error & Speaker Analysis Scripts

```powershell
# Inspect misclassified test recordings and common confusion pairs
python error_analysis.py

# Per-speaker accuracy breakdown across all digits
python speaker_digit_analysis.py

# Standalone model profiling (Parameters, KB/MB size, MACs)
python model.py
```

---

## 7. TensorBoard Visualizations

All scalar events (`Loss/train`, `Loss/validation`, `Accuracy/train`, `Accuracy/validation`, `F1/validation`, `LearningRate`) are logged cleanly to model-specific subdirectories under `runs/spoken_digit_augmented/`.

To launch the interactive dashboard:

```powershell
.\venv\Scripts\tensorboard.exe --logdir runs/spoken_digit_augmented
```

Open your browser at `http://localhost:6006` to inspect multi-model convergence curves.

---

## 8. Audio Preprocessing & Spectrogram Extraction

| Parameter | Configuration | Acoustic Rationale |
| :--- | :--- | :--- |
| **Sample Rate** | 8,000 Hz | Captures frequencies up to 4 kHz (Nyquist), sufficient for human vowel formants and consonant bursts. |
| **Framing Window (`win_length`)** | 256 samples (32 ms) | Standard quasi-stationary speech analysis window. |
| **Hop Length (`hop_length`)** | 128 samples (16 ms) | 50% frame overlap to preserve continuous temporal transitions. |
| **Mel Filterbanks (`n_mels`)** | 40 filterbanks | Logarithmic frequency scaling mimicking human auditory cochlear resolution. |
| **Dynamic Range (`top_db`)** | 80 dB | Logarithmic decibel scale power compression. |
| **Normalization** | Z-score per sample | Zero-mean, unit-variance standardization per spectrogram. |
| **Output Shape** | `[1, 40, 63]` | 1 channel $\times$ 40 Mel frequency bins $\times$ 63 time frames. |

---

## 9. Data Augmentation

Augmentations are applied **dynamically on-the-fly to training batches only**:

1. **Random Amplitude Gain** ($p=0.5$): Uniform scaling $\in [0.8, 1.2]$ to simulate varying speaker volume and microphone distances.
2. **Gaussian Noise Injection** ($p=0.5$): Additive zero-mean Gaussian noise ($\sigma \in [0.001, 0.005]$).
3. **SpecAugment Frequency Masking**: Masks up to 4 contiguous Mel frequency bands.
4. **SpecAugment Time Masking**: Masks up to 6 contiguous time frames.

---

## 10. Error Analysis & Confusion Matrix

### Held-Out Speaker (`yweweler`) Confusion Matrix (500 samples):

```text
       0   1   2   3   4   5   6   7   8   9
  0 [ 48   0   1   0   0   0   0   0   0   1]  (96% acc)
  1 [  0  50   0   0   0   0   0   0   0   0]  (100% acc)
  2 [  0   0  49   0   0   0   1   0   0   0]  (98% acc)
  3 [  0   0   0  46   0   0   0   0   4   0]  (92% acc)
  4 [  0   2   0   0  48   0   0   0   0   0]  (96% acc)
  5 [  0   1   0   0   0  48   0   0   0   1]  (96% acc)
  6 [  0   0   0   1   0   0  38   0  11   0]  (76% acc)
  7 [  0   0   0   0   0   0   0  49   0   1]  (98% acc)
  8 [  0   0   0   0   0   0   2   0  48   0]  (96% acc)
  9 [  0  11   0   0   0   8   0   0   0  31]  (62% acc)
```

* Most frequent confusions: Digit 9 $\to$ 1 (shared nasal and open vowel characteristics), Digit 6 $\to$ 8 (fricative bandwidth similarities at 8 kHz).

---

## 11. Project File Structure & Guide

```text
spoken_digit_recognition_complete/
│
├── data/
│   └── recordings/                  # 3,000 FSDD WAV recordings (0_george_0.wav, etc.)
│
├── myvoice/                         # Live user microphone recordings (myvoice0.wav - myvoice9.wav)
│
├── checkpoints/
│   ├── best_model.pth               # Active winner model checkpoint (~0.60 MB, SpokenDigitCNN)
│   ├── best_model_SpokenDigitCNN.pth# Checkpoint for Baseline CNN (89.8% - 91.0%)
│   ├── best_model_SpokenDigitGAP.pth# Checkpoint for GAP model (32.6%)
│   ├── best_model_SpokenDigitMobile.pth# Checkpoint for MobileNet-style model (34.2%)
│   └── best_model_random_split.pth  # Checkpoint for random split experiment (98.7%)
│
├── runs/
│   └── spoken_digit_augmented/      # Clean TensorBoard event logs (CNN, GAP, Mobile)
│
├── dataset.py                       # Audio loader, Mel transform, VAD, and data augmentations
├── model.py                         # SpokenDigitCNN, GAP, Mobile architectures & profiler
├── train.py                         # Multi-model training and comparative benchmark runner
├── train_random_split.py            # Random 80/10/10 split training script
├── baseline.py                      # Logistic Regression baseline classifier
├── inference.py                     # Single-file prediction on dataset audio (yweweler)
├── inference_with_my_voice.py       # Batch/single inference on live user voice recordings
├── evaluate_noise_robustness.py     # Systematic SNR degradation benchmark (30 dB to 0 dB)
├── error_analysis.py                # Identifies and lists misclassified test recordings
├── speaker_digit_analysis.py        # Per-speaker and per-digit accuracy evaluation
├── comparison_metrics.json          # Side-by-side JSON comparison of all architectures
├── metrics.json                     # Primary test evaluation metrics for the active model
├── baseline_metrics.json            # Logistic regression baseline evaluation metrics
├── split_info.json                  # Speaker split partition details
├── requirements.txt                 # Python library dependencies
└── README.md                        # Comprehensive system documentation and research guide
```

---

## License & Citations
* **Dataset**: [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) by Zohar Jackson et al., CC BY-SA 4.0.
* **Framework**: Built with PyTorch and torchaudio.
#   s p o k e n - d i g i t - r e c o g n i z a t i o n  
 