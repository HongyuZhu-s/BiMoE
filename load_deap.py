import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
import os

# Define the channel indices of each expert
EXPERT_CHANNELS = {
    'prefrontal': ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'central': ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'Cz', 'T7', 'T8'],
    'parietal': ['CP5', 'CP1', 'CP2', 'CP6', 'P3', 'P4', 'P7', 'P8', 'Pz'],
    'occipital': ['PO3', 'PO4', 'O1', 'O2', 'Oz'],
    'temporal': ['T7', 'T8'],
    'EEG': ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
            'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
            'CP2', 'P4', 'P8', 'PO4', 'O2'],
    'peripheral': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
}

ORIGINAL_CHANNELS = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                     'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                     'CP2', 'P4', 'P8', 'PO4', 'O2', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

BRAIN_REGIONS = {
    'prefrontal': ['Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'central': ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'Cz', 'T7', 'T8'],
    'parietal': ['CP5', 'CP1', 'CP2', 'CP6', 'P3', 'P4', 'P7', 'P8', 'Pz'],
    'occipital': ['PO3', 'PO4', 'O1', 'O2', 'Oz'],
    'temporal': ['T7', 'T8']
}


def get_channel_indices(channel_names):
    indices = []
    for channel in channel_names:
        indices.append(ORIGINAL_CHANNELS.index(channel))
    return indices


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


def normalize_per_subject(experts_data):
    """
    For each expert's data, perform Z-score normalization separately on the channel dimension of each subject.
    """
    normalized_experts_data = {expert: [] for expert in experts_data.keys()}

    for expert, subject_data_list in experts_data.items():
        print(f"The data of {expert} expert is being normalized...")

        for sub_id, subject_data in enumerate(subject_data_list):
            n_samples, n_channels, n_timesteps = subject_data.shape

            data_reshaped = subject_data.permute(1, 0, 2).reshape(n_channels, -1)
            channel_means = data_reshaped.mean(dim=1, keepdim=True)
            channel_stds = data_reshaped.std(dim=1, keepdim=True)

            channel_stds[channel_stds == 0] = 1.0

            normalized_data = (data_reshaped - channel_means) / channel_stds

            normalized_data = normalized_data.reshape(n_channels, n_samples, n_timesteps).permute(1, 0, 2)

            normalized_experts_data[expert].append(normalized_data)

    return normalized_experts_data


def load_all_subjects(data_folder, subject_count=32, normalize=True):
    """
    Initialize the expert data dictionary.
    """
    experts_data = {expert: [] for expert in EXPERT_CHANNELS.keys()}
    all_labels = []
    subject_info = []

    expert_indices = {}
    for expert, channels in EXPERT_CHANNELS.items():
        expert_indices[expert] = get_channel_indices(channels)

    for sub_id in range(subject_count):
        file_path = os.path.join(data_folder, f'sub{sub_id}.hdf')

        if not os.path.exists(file_path):
            print(f"Warning: The file {file_path} does not exist. Skip this subject.")
            continue

        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
            labels = f['label'][:]

        # reshape data: [40, 15, 40, 512] -> [600, 40, 512]
        data_reshaped = data.reshape(-1, data.shape[2], data.shape[3])

        # reshape label: [40, 15] -> [600, 1]
        labels_reshaped = labels.reshape(-1, 1)

        labels_tensor = torch.from_numpy(labels_reshaped).long()
        all_labels.append(labels_tensor)

        for expert, indices in expert_indices.items():
            expert_data = data_reshaped[:, indices, :]  # 提取该专家的通道
            expert_tensor = torch.from_numpy(expert_data).float()
            experts_data[expert].append(expert_tensor)

        # Record the information of subjects
        subject_info.append({
            'subject_id': sub_id,
            'original_data_shape': data.shape,
            'reshaped_data_shape': data_reshaped.shape,
            'reshaped_label_shape': labels_reshaped.shape,
            'expert_shapes': {expert: experts_data[expert][-1].shape
                              for expert in EXPERT_CHANNELS.keys()}
        })
        if sub_id == 0:
            for expert in EXPERT_CHANNELS.keys():
                print(f"         {expert}: {experts_data[expert][-1].shape}")

    if normalize:
        experts_data = normalize_per_subject(experts_data)

    return experts_data, all_labels, subject_info