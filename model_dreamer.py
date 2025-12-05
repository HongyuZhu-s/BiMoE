
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


EEG_SAMPLING_RATE = 128  # Hz
ECG_SAMPLING_RATE = 256  # Hz
SEGMENT_DURATION = 1     # s
EEG_SEGMENT_LENGTH = EEG_SAMPLING_RATE * SEGMENT_DURATION  # 128
ECG_SEGMENT_LENGTH = ECG_SAMPLING_RATE * SEGMENT_DURATION  # 256
TOTAL_DURATION = 60      # s
EEG_TOTAL_LENGTH = EEG_SAMPLING_RATE * TOTAL_DURATION  # 7680
ECG_TOTAL_LENGTH = ECG_SAMPLING_RATE * TOTAL_DURATION  # 15360

DREAMER_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(DREAMER_CHANNELS)}

DREAMER_REGIONS = {
    'prefrontal': ['AF3', 'F7', 'F3', 'F4', 'F8', 'AF4'],
    'central': ['FC5', 'FC6', 'T7', 'T8'],
    'parietal': ['P7', 'P8'],
    'occipital': ['O1', 'O2'],
    'temporal': ['T7', 'T8'],
    'EEG': DREAMER_CHANNELS
}


class WPLICalculator(nn.Module):
    def __init__(self):
        super(WPLICalculator, self).__init__()

    def hilbert_transform(self, signal):
        n = signal.size(-1)
        analytic_signal = torch.fft.fft(signal, dim=-1)
        if n % 2 == 0:
            analytic_signal[..., n // 2:] = 0
        else:
            analytic_signal[..., (n + 1) // 2:] = 0
        analytic_signal *= 2
        return torch.fft.ifft(analytic_signal, dim=-1)

    def forward(self, eeg_data):
        batch_size, num_channels, seq_len = eeg_data.shape

        analytic_signal = self.hilbert_transform(eeg_data)
        phase_data = torch.angle(analytic_signal)

        phase_i = phase_data.unsqueeze(2)
        phase_j = phase_data.unsqueeze(1)
        phase_diff = phase_i - phase_j

        sin_phase_diff = torch.sin(phase_diff)
        sign_sin_phase_diff = torch.sign(sin_phase_diff)

        pli_matrix = torch.abs(torch.mean(sign_sin_phase_diff, dim=-1))

        eye_mask = 1 - torch.eye(num_channels, device=eeg_data.device).unsqueeze(0)
        pli_matrix = pli_matrix * eye_mask

        return pli_matrix


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = adj.shape

        degree = torch.sum(adj, dim=2)
        degree_sqrt = torch.sqrt(degree + 1e-8)
        degree_norm = 1.0 / degree_sqrt

        D_norm = torch.diag_embed(degree_norm)
        adj_norm = torch.bmm(torch.bmm(D_norm, adj), D_norm)

        support = torch.bmm(adj_norm, x)
        output = self.linear(support)

        return output


class PLI_GCN_Extractor(nn.Module):
    def __init__(self, num_channels, hidden_dim=128):
        super().__init__()
        # 移除时间池化，直接使用1秒窗口数据
        self.pli_calculator = WPLICalculator()
        self.gcn1 = GCNLayer(EEG_SEGMENT_LENGTH, hidden_dim)  # 输入128个时间点
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, eeg_data):
        # eeg_data形状: [B, channels, 128]
        x = eeg_data  # 直接使用1秒窗口数据

        # 计算 wPLI
        wpli_matrix = self.pli_calculator(x)

        # GCN
        x = F.relu(self.bn1(self.gcn1(x, wpli_matrix)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.gcn2(x, wpli_matrix)))
        x = self.dropout(x)

        # 注意力
        x_attn, _ = self.attention(x, x, x)
        # x = x + self.dropout(x_attn)

        # 全局池化
        features = torch.mean(x_attn, dim=1)  # [B, 64]

        return features, wpli_matrix


class EnhancedPeripheralExpert(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super(EnhancedPeripheralExpert, self).__init__()

        # 针对ECG信号的特殊处理，使用1秒窗口数据
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x形状: [B, 2, 256] - 1秒ECG数据
        features = self.conv_layers(x).squeeze(-1)  # [B, 64]
        output = self.classifier(features)
        return output


class EnhancedBrainRegionExpert(nn.Module):
    def __init__(self, input_channels, num_classes=2, expert_type='eeg'):
        super(EnhancedBrainRegionExpert, self).__init__()
        self.expert_type = expert_type

        # CNN路径 - 适应1秒窗口
        self.cnn_path = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64
            nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 对于EEG专家，添加PLI-GCN路径
        if expert_type == 'eeg':
            self.gcn_extractor = PLI_GCN_Extractor(input_channels, hidden_dim=128)
            self.gcn_fusion = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.classifier = nn.Sequential(
                nn.Linear(192, 64),  # 64 (CNN) + 128 (GCN) = 192
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes)
            )

    def forward(self, x):
        # x形状: [B, channels, 128] - 1秒EEG数据
        cnn_features = self.cnn_path(x).squeeze(-1)  # [batch, 64]

        if self.expert_type == 'eeg':
            gcn_features, _ = self.gcn_extractor(x)  # [batch, 64]
            gcn_features = self.gcn_fusion(gcn_features)  # [batch, 128]
            combined_features = torch.cat([cnn_features, gcn_features], dim=1)  # [B, 192]
            output = self.classifier(combined_features)
        else:
            output = self.classifier(cnn_features)

        return output


class EnhancedGatingNetwork(nn.Module):
    def __init__(self, eeg_input_dim, ecg_input_dim, num_experts, hidden_dim=128):
        super().__init__()
        self.eeg_input_dim = eeg_input_dim
        self.ecg_input_dim = ecg_input_dim

        self.ecg_feature_extractor = nn.Sequential(
            nn.Linear(ecg_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

        self.multimodal_encoder = nn.Sequential(
            nn.Linear(eeg_input_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.gate_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, eeg_global_features, ecg_global_features):
        assert eeg_global_features.shape[-1] == self.eeg_input_dim
        assert ecg_global_features.shape[-1] == self.ecg_input_dim

        ecg_features = self.ecg_feature_extractor(ecg_global_features)
        fused_features = torch.cat([eeg_global_features, ecg_features], dim=1)
        gate_input = self.multimodal_encoder(fused_features)
        expert_weights = self.gate_output(gate_input)

        return expert_weights


class EnhancedMixtureOfExperts(nn.Module):
    def __init__(self, experts_config, num_classes=2):
        super(EnhancedMixtureOfExperts, self).__init__()
        self.num_classes = num_classes

        self.experts = nn.ModuleDict()
        for expert_name, config in experts_config.items():
            if expert_name == 'EEG':
                self.experts[expert_name] = EnhancedBrainRegionExpert(
                    input_channels=config['input_channels'],
                    num_classes=config['num_classes'],
                    expert_type='eeg'
                )
            elif expert_name == 'peripheral':
                self.experts[expert_name] = EnhancedPeripheralExpert(
                    input_channels=config['input_channels'],
                    num_classes=config['num_classes']
                )
            else:
                self.experts[expert_name] = EnhancedBrainRegionExpert(
                    input_channels=config['input_channels'],
                    num_classes=config['num_classes'],
                    expert_type='brain_region'
                )

        self.num_experts = len(experts_config)

        # ECG全局特征提取器 - 适应1秒窗口
        self.ecg_global_extractor = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 256 -> 128
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 128 -> 64
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.ecg_feature_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.gating_network = EnhancedGatingNetwork(
            eeg_input_dim=64,
            ecg_input_dim=32,
            num_experts=self.num_experts
        )

        self.global_gcn = PLI_GCN_Extractor(14, hidden_dim=128)

    def forward(self, expert_inputs):
        expert_outputs = []
        expert_logits = []
        expert_names = []

        eeg_global_features, _ = self.global_gcn(expert_inputs['EEG'])

        ecg_data = expert_inputs['peripheral']
        ecg_features = self.ecg_global_extractor(ecg_data).squeeze(-1)
        ecg_global_features = self.ecg_feature_fc(ecg_features)

        for expert_name, expert_input in expert_inputs.items():
            if expert_name in self.experts:
                output = self.experts[expert_name](expert_input)
                expert_outputs.append(output)
                expert_logits.append(output)
                expert_names.append(expert_name)

        gate_weights = self.gating_network(eeg_global_features, ecg_global_features)
        final_output = self.weighted_voting(expert_logits, gate_weights)

        return final_output, gate_weights, expert_logits

    def weighted_voting(self, expert_logits, gate_weights):
        expert_logits_stack = torch.stack(expert_logits, dim=1)
        gate_weights_expanded = gate_weights.unsqueeze(-1).expand(-1, -1, self.num_classes)
        weighted_logits = (expert_logits_stack * gate_weights_expanded).sum(dim=1)
        return weighted_logits

    def get_expert_decisions(self, expert_logits):
        expert_predictions = []
        expert_confidences = []

        for logits in expert_logits:
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            expert_predictions.append(prediction)
            expert_confidences.append(confidence)

        expert_predictions = torch.stack(expert_predictions, dim=1)
        expert_confidences = torch.stack(expert_confidences, dim=1)

        return expert_predictions, expert_confidences


def preprocess_data(eeg_data, ecg_data, labels):
    """
    预处理数据：保留最后60秒，并按1秒窗口划分
    """
    num_trials, eeg_channels, eeg_length = eeg_data.shape
    _, ecg_channels, ecg_length = ecg_data.shape

    # 确保数据长度足够
    assert eeg_length >= EEG_TOTAL_LENGTH, f"EEG数据长度不足: {eeg_length} < {EEG_TOTAL_LENGTH}"
    assert ecg_length >= ECG_TOTAL_LENGTH, f"ECG数据长度不足: {ecg_length} < {ECG_TOTAL_LENGTH}"

    # 取最后60秒数据
    eeg_60s = eeg_data[:, :, -EEG_TOTAL_LENGTH:]
    ecg_60s = ecg_data[:, :, -ECG_TOTAL_LENGTH:]

    # 按1秒窗口划分
    eeg_segments = []
    ecg_segments = []
    segment_labels = []

    for trial_idx in range(num_trials):
        for segment_idx in range(TOTAL_DURATION):
            # EEG片段
            start_idx = segment_idx * EEG_SEGMENT_LENGTH
            end_idx = start_idx + EEG_SEGMENT_LENGTH
            eeg_segment = eeg_60s[trial_idx, :, start_idx:end_idx]
            eeg_segments.append(eeg_segment)

            # ECG片段
            start_idx = segment_idx * ECG_SEGMENT_LENGTH
            end_idx = start_idx + ECG_SEGMENT_LENGTH
            ecg_segment = ecg_60s[trial_idx, :, start_idx:end_idx]
            ecg_segments.append(ecg_segment)

            # 标签（每个片段使用相同的trial标签）
            segment_labels.append(labels[trial_idx])

    # 转换为numpy数组
    eeg_segments = np.array(eeg_segments)  # (num_trials * 60, eeg_channels, 128)
    ecg_segments = np.array(ecg_segments)  # (num_trials * 60, ecg_channels, 256)
    segment_labels = np.array(segment_labels)  # (num_trials * 60,)

    return eeg_segments, ecg_segments, segment_labels

# ==================== 专家分歧损失 ====================
class ExpertDisagreementLoss(nn.Module):
    """专家分歧损失，鼓励专家学习不同的决策边界"""

    def __init__(self, disagreement_coef=0.05):
        super(ExpertDisagreementLoss, self).__init__()
        self.disagreement_coef = disagreement_coef

    def forward(self, expert_logits):
        if len(expert_logits) <= 1:
            return torch.tensor(0.0, device=expert_logits[0].device)

        batch_size = expert_logits[0].size(0)
        num_experts = len(expert_logits)

        # 计算每个专家的预测概率
        expert_probs = [F.softmax(logits, dim=1) for logits in expert_logits]

        # 计算专家之间的成对KL散度
        total_disagreement = 0.0
        pair_count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算KL散度: KL(P_i || P_j)
                kl_ij = F.kl_div(
                    torch.log(expert_probs[j] + 1e-8),
                    expert_probs[i] + 1e-8,
                    reduction='batchmean'
                )
                # 计算KL散度: KL(P_j || P_i)
                kl_ji = F.kl_div(
                    torch.log(expert_probs[i] + 1e-8),
                    expert_probs[j] + 1e-8,
                    reduction='batchmean'
                )
                # 使用对称KL散度
                symmetric_kl = (kl_ij + kl_ji) / 2.0
                total_disagreement += symmetric_kl
                pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=expert_logits[0].device)

        # 平均分歧损失
        avg_disagreement = total_disagreement / pair_count

        # 鼓励分歧，所以最大化KL散度（最小化负KL散度）
        disagreement_loss = -self.disagreement_coef * avg_disagreement

        return disagreement_loss

class ExpertLoadLoss(nn.Module):
    """专家负载损失，用于平衡专家利用率"""

    def __init__(self, num_experts, importance_coef=1.0, load_coef=0.1):
        super(ExpertLoadLoss, self).__init__()
        self.num_experts = num_experts
        self.importance_coef = importance_coef
        self.load_coef = load_coef

    def forward(self, gate_weights, batch_size):
        # 计算每个专家的利用率（批次平均）
        expert_utilization = torch.mean(gate_weights, dim=0)  # [num_experts]

        # 计算重要性损失 - 鼓励所有专家都有相似的利用率
        target_utilization = torch.ones_like(expert_utilization) / self.num_experts
        importance_loss = F.mse_loss(expert_utilization, target_utilization)

        # 计算负载损失 - 防止单个专家被过度使用
        load_loss = torch.var(expert_utilization)

        # 总负载损失
        total_load_loss = self.importance_coef * importance_loss + self.load_coef * load_loss

        return total_load_loss


class MultiExpertDreamerDataset(Dataset):
    def __init__(self, expert_data_dict, labels):
        self.expert_data = expert_data_dict  # dict: {expert_name: tensor}
        self.labels = labels  # shape [N,]
        self.length = labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        experts_input = {name: data[idx] for name, data in self.expert_data.items()}
        label = self.labels[idx]
        return experts_input, label