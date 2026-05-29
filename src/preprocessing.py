"""
preprocessing.py
EEG Signal Preprocessing for BCI Motor Imagery Classification
PhysioNet Dataset — Electrodes: C3, Cz, C4

Pipeline:
    1. Load EEG data (MNE)
    2. Select C3, Cz, C4 channels
    3. Notch filter @ 60 Hz
    4. Bandpass filter 8-30 Hz (mu/beta rhythms)
    5. Z-score normalization
    6. Epoch segmentation
    7. STFT spectrogram generation (224x224x3)
"""

import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch
from scipy import signal
import matplotlib.pyplot as plt
import os

ELECTRODES = ['C3', 'Cz', 'C4']
SFREQ      = 160        # PhysioNet sampling rate (Hz)
NOTCH_FREQ = 60.0       # Powerline noise
BP_LOW     = 8.0        # Mu rhythm lower bound
BP_HIGH    = 30.0       # Beta rhythm upper bound
EPOCH_SEC  = 4.0        # Trial window length
IMG_SIZE   = 224        # Spectrogram output size


def load_physionet_subject(subject_id: int, data_dir: str):
    """Load raw EEG for one subject from PhysioNet local files."""
    runs = [4, 8, 12]   # Motor imagery runs
    raws = []
    for run in runs:
        fname = os.path.join(data_dir, f'S{subject_id:03d}', f'S{subject_id:03d}R{run:02d}.edf')
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    raw.pick_channels(ELECTRODES)
    return raw


def notch_filter(data: np.ndarray, fs: float = SFREQ, freq: float = NOTCH_FREQ) -> np.ndarray:
    """Remove powerline interference with a notch filter."""
    b, a = iirnotch(freq, Q=35, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def bandpass_filter(data: np.ndarray, fs: float = SFREQ,
                    low: float = BP_LOW, high: float = BP_HIGH) -> np.ndarray:
    """4th-order Butterworth bandpass — isolates mu (8-13 Hz) and beta (13-30 Hz)."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization per channel."""
    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True) + 1e-8
    return (data - mean) / std


def eeg_to_spectrogram(epoch: np.ndarray, fs: float = SFREQ,
                        img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Convert 3-channel EEG epoch to 224x224x3 RGB-like spectrogram.

    Args:
        epoch: shape (3, n_samples) — C3, Cz, C4
        fs: sampling frequency
        img_size: output spatial resolution

    Returns:
        spectrogram: shape (img_size, img_size, 3)
    """
    specs = []
    for ch in range(epoch.shape[0]):
        f, t, Zxx = signal.stft(epoch[ch], fs=fs,
                                 window='hamming',
                                 nperseg=64,
                                 noverlap=32,
                                 nfft=128)
        mag = np.abs(Zxx)
        mag = np.log1p(mag)                        # log-scale compression
        # Resize to img_size x img_size
        from PIL import Image
        img = Image.fromarray(mag)
        img = img.resize((img_size, img_size), Image.BILINEAR)
        specs.append(np.array(img))

    stacked = np.stack(specs, axis=-1).astype(np.float32)
    stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min() + 1e-8)
    return stacked


def preprocess_subject(subject_id: int, data_dir: str):
    """Full preprocessing pipeline for one subject."""
    raw = load_physionet_subject(subject_id, data_dir)
    data = raw.get_data()                  # shape: (3, n_times)

    data = notch_filter(data)
    data = bandpass_filter(data)
    data = zscore_normalize(data)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epoch_len = int(EPOCH_SEC * SFREQ)

    spectrograms, labels = [], []
    label_map = {'T0': 0, 'T1': 1, 'T2': 2}

    for ev in events:
        onset  = ev[0]
        label  = ev[2]
        if onset + epoch_len > data.shape[1]:
            continue
        epoch = data[:, onset: onset + epoch_len]
        spec  = eeg_to_spectrogram(epoch)
        spectrograms.append(spec)
        labels.append(label - 1)           # 1-indexed → 0-indexed

    return np.array(spectrograms), np.array(labels)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/physionet/')
    parser.add_argument('--n_subjects', type=int, default=30)
    parser.add_argument('--out_dir', type=str, default='./data/processed/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_specs, all_labels = [], []

    for sid in range(1, args.n_subjects + 1):
        print(f'Processing subject {sid}/{args.n_subjects}...')
        specs, labels = preprocess_subject(sid, args.data_dir)
        all_specs.append(specs)
        all_labels.append(labels)

    X = np.concatenate(all_specs,  axis=0)
    y = np.concatenate(all_labels, axis=0)
    np.save(os.path.join(args.out_dir, 'X.npy'), X)
    np.save(os.path.join(args.out_dir, 'y.npy'), y)
    print(f'Saved: X={X.shape}, y={y.shape}')
