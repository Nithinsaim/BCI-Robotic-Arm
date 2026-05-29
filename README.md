# 🧠 BCI-Based Robotic Arm for Medical Rehabilitation

> **Published in IEEE** | Amrita Vishwa Vidyapeetham, Coimbatore

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white)](https://mathworks.com)
[![Arduino](https://img.shields.io/badge/Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://arduino.cc)
[![IEEE](https://img.shields.io/badge/Published-IEEE-00629B?style=for-the-badge&logo=ieee&logoColor=white)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.89%25-brightgreen?style=for-the-badge)]()

---

## 📄 Abstract

Stroke and other neurological disorders often result in significant motor impairments, limiting the effectiveness of traditional rehabilitation. This project proposes a non-invasive **Brain-Computer Interface (BCI)** system designed to assist motor rehabilitation by decoding motor imagery (MI) tasks from EEG signals.

Using the publicly available **PhysioNet EEG Motor Imagery dataset**, EEG signals from only **three strategically selected electrodes (C3, Cz, C4)** over the sensorimotor cortex are utilized — significantly reducing hardware complexity without compromising performance. Signals from **30 subjects** are preprocessed, then transformed into 2D spectrograms via **Short-Time Fourier Transform (STFT)**. A **hybrid CNN-LSTM** deep learning model classifies imagined movements (Left, Right, Rest), and the output is transmitted to an **Arduino Uno** to actuate a **4-DOF robotic arm** using SG90 servo motors.

---

## 🏗️ System Architecture

```
EEG Dataset (PhysioNet)
        │
        ▼
Preprocessing (Notch Filter 60Hz + Bandpass 8–30Hz + Z-score Normalization)
        │
        ▼
STFT → Spectrogram (224×224×3 RGB — C3, Cz, C4 stacked)
        │
        ▼
CNN Feature Extractor (16→32→64 filters, BatchNorm, ReLU)
        │
        ▼
LSTM Temporal Classifier (256 hidden units)
        │
        ▼
Classification: Left | Right | Rest
        │
        ▼
Serial Communication (MATLAB → Arduino Uno)
        │
        ▼
4-DOF Robotic Arm (SG90 Servo Motors, PWM Pins D3,D5,D6,D9)
```

---

## 📁 Repository Structure

```
BCI-Robotic-Arm/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing.py        # Notch + Bandpass filtering, normalization
│   ├── spectrogram.py          # STFT-based spectrogram generation
│   ├── cnn_model.py            # CNN spatial feature extractor
│   ├── lstm_model.py           # LSTM temporal classifier
│   ├── train.py                # Training pipeline (CNN 30 epochs, LSTM 70 epochs)
│   ├── predict.py              # Real-time inference
│   └── arduino_control.ino    # Arduino Uno servo control sketch
├── notebooks/
│   └── BCI_EEG_Pipeline.ipynb  # Full end-to-end notebook
├── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   └── ablation_study.csv
├── images/
│   └── system_overview.png
└── LICENSE
```

---

## ⚙️ Methods

### Dataset
- **Source:** PhysioNet EEG Motor Imagery Dataset
- **Subjects:** 30 (first valid subjects)
- **Electrodes:** C3, Cz, C4 (sensorimotor cortex — ERD/ERS)
- **Classes:** T0 (Rest), T1 (Left), T2 (Right)
- **Split:** 70% Train | 20% Test | 10% Validation

### Preprocessing
| Step | Parameters |
|------|-----------|
| Notch Filter | 60 Hz (powerline noise removal) |
| Bandpass Filter | 8–30 Hz (mu & beta rhythms), 4th order Butterworth |
| Normalization | Z-score (zero mean, unit variance) |
| STFT | Hamming window=64, overlap=32, FFT=128 |
| Spectrogram | 224×224, RGB-stacked (C3, Cz, C4) |

### CNN Architecture
| Layer | Details |
|-------|---------|
| Input | 224×224×3 RGB spectrogram |
| Conv Block 1 | 16 filters, BatchNorm, ReLU, MaxPool |
| Conv Block 2 | 32 filters, BatchNorm, ReLU, MaxPool |
| Conv Block 3 | 64 filters, BatchNorm, ReLU, AvgPool |
| FC | 128 units (spatial feature vector) |

### LSTM Architecture
| Layer | Details |
|-------|---------|
| Input | 128-dim CNN feature vector (as sequence) |
| LSTM | 256 hidden units |
| Output | Softmax → 3 classes |

### Training
| Stage | Epochs | Optimizer | LR |
|-------|--------|-----------|-----|
| CNN | 30 | Adam | 1e-4 |
| LSTM | 70 | Adam | 1e-4 |

---

## 📊 Results

### Classification Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| T0 (Rest) | 0.88 | 0.88 | 0.88 |
| T1 (Left) | 0.75 | 0.76 | 0.75 |
| T2 (Right) | 0.82 | 0.81 | 0.82 |

### Model Accuracy

| Model Variant | Accuracy (%) |
|---------------|-------------|
| **CNN-LSTM (Proposed)** | **85.89** |
| CNN Only | 82.20 |
| Shallow CNN | 70.55 |
| Shallow LSTM | 82.10 |
| No Batch Normalization | 77.54 |

### Comparison with Related Work

| Model | Accuracy | Subjects | Electrodes |
|-------|----------|----------|------------|
| **CNN-LSTM (This Work)** | **85.89%** | **30** | **3** |
| CNN-LSTM (Fadel et al.) | 70.64% | 109 | 64 |
| CNN+LSTM+DNN (Li et al.) | 75.52% | 12 | 64 |
| CNN1D MF (Alnaanah et al.) | 58.0% | 105 | 64 |

> Our model outperforms all baselines using only 3 electrodes — demonstrating hardware efficiency without sacrificing accuracy.

### Training Time
- CNN Feature Extractor: ~4 minutes 56 seconds
- LSTM Classifier: ~35 seconds

---

## 🔧 Hardware Setup

| Component | Specification |
|-----------|--------------|
| Microcontroller | Arduino Uno |
| Servo Motors | SG90 (4×) — Torque: 11 kg·cm at 6V |
| DOF | 4 (lift, rotate, grasp, release) |
| PWM Pins | D3, D5, D6, D9 |
| Communication | MATLAB → Arduino via Serial (UART) |
| Commands | LEFT, RIGHT, REST → servo angle PWM |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Preprocessing
```bash
python src/preprocessing.py --data_path ./data/physionet/
```

### Train the Model
```bash
python src/train.py --epochs_cnn 30 --epochs_lstm 70 --lr 1e-4
```

### Real-time Prediction
```bash
python src/predict.py --port COM3
```

---

## 📦 requirements.txt

See `requirements.txt` for full dependencies (numpy, scipy, torch, mne, pyserial, matplotlib).

---

## 👥 Authors

| Name | Roll No. | Institution |
|------|----------|-------------|
| Hari Sudharsan G | CB.AI.U4AIM24113 | Amrita Vishwa Vidyapeetham |
| **Nithin S** | **CB.AI.U4AIM24133** | **Amrita Vishwa Vidyapeetham** |
| Amritavarshini B | CB.AI.U4AIM24154 | Amrita Vishwa Vidyapeetham |
| Devadharshini M | CB.AI.U4AIM24126 | Amrita Vishwa Vidyapeetham |

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{harisudharsan2024bci,
  title     = {BCI-Based Robotic Arm for Medical Rehabilitation},
  author    = {Hari Sudharsan, G. and Nithin, S. and Amritavarshini, B. and Devadharshini, M. and Amrutha Veluppal},
  booktitle = {IEEE},
  year      = {2024},
  institution = {Amrita Vishwa Vidyapeetham, Coimbatore}
}
```

---

## 📜 References

1. Goldberger et al. PhysioBank, PhysioToolkit, PhysioNet. *Circulation*, 2000.
2. Fadel et al. Multi-class classification of motor imagery EEG. *IEEE BCI*, 2020.
3. Li et al. Improving EEG-based MI classification using hybrid neural network. *IEEE ICICN*, 2021.
4. Pfurtscheller & Da Silva. Event-related EEG/MEG synchronization. *Clinical Neurophysiology*, 1999.

---

<div align="center">
📍 Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu &nbsp;|&nbsp; Published in IEEE
</div>
