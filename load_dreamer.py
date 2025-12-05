from scipy.signal import butter, sosfiltfilt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    if data.ndim == 2:
        filtered = sosfiltfilt(sos, data, axis=-1)
    elif data.ndim == 3:
        filtered = sosfiltfilt(sos, data, axis=-1)
    else:
        raise ValueError("Unsupported data dimension.")
    return filtered


def zscore_normalize(data):
    if data.ndim == 2:
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std == 0, 1, std)
        normalized = (data - mean) / std
    elif data.ndim == 3:
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std == 0, 1, std)
        normalized = (data - mean) / std
    else:
        raise ValueError("Unsupported data dimension.")
    return normalized


def process_single_subject_eeg(eeg_data, fs=128.0, lowcut=4.0, highcut=45.0):
    filtered = butter_bandpass_filter(eeg_data, lowcut, highcut, fs, order=3)
    processed = zscore_normalize(filtered)
    return processed


def process_single_subject_ecg(ecg_data, fs=256.0, lowcut=0.5, highcut=40.0):
    filtered = butter_bandpass_filter(ecg_data, lowcut, highcut, fs, order=3)
    processed = zscore_normalize(filtered)
    return processed


def batch_process_dreamer(
        input_dir="Dreamer_data_all",
        output_dir="P_dreamer_data_all",
        subject_ids=range(0, 23),  # 0 to 22
        eeg_fs=128.0,
        eeg_lowcut=4.0,
        eeg_highcut=45.0,
        ecg_fs=256.0,
        ecg_lowcut=0.5,
        ecg_highcut=40.0
):

    os.makedirs(output_dir, exist_ok=True)

    for sid in subject_ids:
        input_path = os.path.join(input_dir, f"dreamer{sid}.npy")
        output_path = os.path.join(output_dir, f"dreamer{sid}.npy")

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found. Skipping.")
            continue

        data_dict = np.load(input_path, allow_pickle=True).item()

        if 'eeg_data' not in data_dict:
            print(f"Warning: 'eeg_data' not found in {input_path}. Skipping.")
            continue

        if 'ecg_data' not in data_dict:
            print(f"Warning: 'ecg_data' not found in {input_path}. Skipping.")
            continue

        eeg = data_dict['eeg_data']  # shape: [18, 14, 7680]
        ecg = data_dict['ecg_data']  # shape: [18, 2, 15360]

        print(f"Processing subject {sid} EEG...")
        processed_eeg = process_single_subject_eeg(eeg, fs=eeg_fs, lowcut=eeg_lowcut, highcut=eeg_highcut)

        print(f"Processing subject {sid} ECG...")
        processed_ecg = process_single_subject_ecg(ecg, fs=ecg_fs, lowcut=ecg_lowcut, highcut=ecg_highcut)

        data_dict['eeg_data'] = processed_eeg
        data_dict['ecg_data'] = processed_ecg

        np.save(output_path, data_dict)
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    batch_process_dreamer()