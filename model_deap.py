import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


class MultiExpertDEAPDataset(Dataset):
    def __init__(self, experts_data, labels):
        self.experts_data = experts_data
        self.labels = labels
        self.expert_names = list(experts_data.keys())

        data_lengths = [data.shape[0] for data in experts_data.values()]
        assert all(length == data_lengths[0] for length in data_lengths), "The length of all expert data must be the same."
        assert data_lengths[0] == len(labels), "The data length must be the same as the label length."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        expert_data = {expert: self.experts_data[expert][idx] for expert in self.expert_names}
        return expert_data, self.labels[idx]


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
        phase_data = torch.angle(analytic_signal)  # [batch, channels, seq_len]

        phase_i = phase_data.unsqueeze(2)  # [batch, channels, 1, seq_len]
        phase_j = phase_data.unsqueeze(1)  # [batch, 1, channels, seq_len]
        phase_diff = phase_i - phase_j  # [batch, channels, channels, seq_len]

        sin_phase_diff = torch.sin(phase_diff)  # [batch, channels, channels, seq_len]

        numerator = torch.abs(torch.mean(sin_phase_diff, dim=-1))  # [batch, channels, channels]

        denominator = torch.mean(torch.abs(sin_phase_diff), dim=-1)  # [batch, channels, channels]

        denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=denominator.device), denominator)

        wpli_matrix = numerator / denominator  # [batch, channels, channels]

        eye_mask = 1 - torch.eye(num_channels, device=eeg_data.device).unsqueeze(0)
        wpli_matrix = wpli_matrix * eye_mask

        return wpli_matrix


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = adj.shape

        degree = torch.sum(adj, dim=2)  # [batch, nodes]
        degree_sqrt = torch.sqrt(degree + 1e-8)
        degree_norm = 1.0 / degree_sqrt

        D_norm = torch.diag_embed(degree_norm)  # [batch, nodes, nodes]
        adj_norm = torch.bmm(torch.bmm(D_norm, adj), D_norm)

        support = torch.bmm(adj_norm, x)  # [batch, nodes, features]
        output = self.linear(support)  # [batch, nodes, out_features]

        return output


class PLI_GCN_Extractor(nn.Module):

    def __init__(self, num_channels, hidden_dim=128):
        super(PLI_GCN_Extractor, self).__init__()
        self.pli_calculator = WPLICalculator()

        self.gcn1 = GCNLayer(512, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim // 2)

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, batch_first=True)

        self.residual_fc = nn.Conv1d(512, hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, eeg_data):

        wpli_matrix = self.pli_calculator(eeg_data)  # [batch, 32, 32]

        x = eeg_data  # [batch, 32, 512]
        x_gcn = F.relu(self.bn1(self.gcn1(x, wpli_matrix)))  # [batch, 32, 128]
        x_gcn = self.dropout(x_gcn)
        x_gcn = F.relu(self.bn2(self.gcn2(x_gcn, wpli_matrix)))  # [batch, 32, 64]
        x_gcn = self.dropout(x_gcn)

        residual = self.residual_fc(x.transpose(1, 2))  # [batch, 512, 32] -> [batch, 64, 32]
        residual = residual.transpose(1, 2)  # [batch, 32, 64]

        x_gcn = x_gcn + residual

        x_attn, _ = self.attention(x_gcn, x_gcn, x_gcn)
        x_attn = self.dropout(x_attn)

        x_final = x_attn

        global_features = torch.mean(x_final, dim=1)  # [batch, 64]

        return global_features, wpli_matrix


class EnhancedBrainRegionExpert(nn.Module):

    def __init__(self, input_channels, time_points=512, num_classes=2, expert_type='eeg'):
        super(EnhancedBrainRegionExpert, self).__init__()
        self.expert_type = expert_type

        self.cnn_path = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        if expert_type == 'eeg':
            self.gcn_extractor = PLI_GCN_Extractor(input_channels, hidden_dim=128)
            self.gcn_fusion = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        self.classifier1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        cnn_features = self.cnn_path(x).squeeze(-1)  # [batch, 256]

        if self.expert_type == 'eeg':

            gcn_features, _ = self.gcn_extractor(x)  # [batch, 64]
            gcn_features = self.gcn_fusion(gcn_features)  # [batch, 128]

            combined_features = torch.cat([cnn_features, gcn_features], dim=1)
            output = self.classifier1(combined_features)
        else:
            combined_features = cnn_features
            output = self.classifier2(combined_features)

        return output


class EnhancedPeripheralExpert(nn.Module):

    def __init__(self, input_channels=8, time_points=512, num_classes=2):
        super(EnhancedPeripheralExpert, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        features = self.conv_layers(x).squeeze(-1)
        output = self.classifier(features)
        return output


class EnhancedGatingNetwork(nn.Module):

    def __init__(self, eeg_input_dim, peripheral_input_dim, num_experts, hidden_dim=256):
        super(EnhancedGatingNetwork, self).__init__()

        self.multimodal_encoder = nn.Sequential(
            nn.Linear(eeg_input_dim + peripheral_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.peripheral_processor = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.gate_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, eeg_global_features, peripheral_data):
        peripheral_features = self.peripheral_processor(peripheral_data).squeeze(-1)

        multimodal_input = torch.cat([eeg_global_features, peripheral_features], dim=1)
        gate_input = self.multimodal_encoder(multimodal_input)

        expert_weights = self.gate_output(gate_input)

        return expert_weights


class MixtureOfExperts(nn.Module):

    def __init__(self, experts_config, num_classes=2, gating_input_dim=512):
        super(MixtureOfExperts, self).__init__()
        self.num_classes = num_classes

        self.experts = nn.ModuleDict()
        for expert_name, config in experts_config.items():
            if expert_name == 'EEG':
                self.experts[expert_name] = EnhancedBrainRegionExpert(
                    input_channels=config['input_channels'],
                    time_points=config['time_points'],
                    num_classes=config['num_classes'],
                    expert_type='eeg'
                )
            elif expert_name == 'peripheral':
                self.experts[expert_name] = EnhancedPeripheralExpert(
                    input_channels=config['input_channels'],
                    time_points=config['time_points'],
                    num_classes=config['num_classes']
                )
            else:
                self.experts[expert_name] = EnhancedBrainRegionExpert(
                    input_channels=config['input_channels'],
                    time_points=config['time_points'],
                    num_classes=config['num_classes'],
                    expert_type='brain_region'
                )

        self.num_experts = len(experts_config)

        self.gating_network = EnhancedGatingNetwork(
            eeg_input_dim=64,
            peripheral_input_dim=32,
            num_experts=self.num_experts
        )

        self.global_gcn = PLI_GCN_Extractor(32, hidden_dim=128)  # 32个EEG通道

    def forward(self, expert_inputs):
        expert_outputs = []
        expert_logits = []
        expert_names = []

        eeg_global_features, _ = self.global_gcn(expert_inputs['EEG'])

        for expert_name, expert_input in expert_inputs.items():
            if expert_name in self.experts:
                output = self.experts[expert_name](expert_input)
                expert_outputs.append(output)
                expert_logits.append(output)
                expert_names.append(expert_name)

        # 计算门控权重
        gate_weights = self.gating_network(eeg_global_features, expert_inputs['peripheral'])

        final_output = self.weighted_voting(expert_logits, gate_weights)

        return final_output, gate_weights, expert_logits

    def weighted_voting(self, expert_logits, gate_weights):

        # [batch_size, num_experts, num_classes]
        expert_logits_stack = torch.stack(expert_logits, dim=1)

        # [batch_size, num_experts, 1] -> [batch_size, num_experts, num_classes]
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


class ExpertDisagreementLoss(nn.Module):

    def __init__(self, disagreement_coef = 0.05):
        super(ExpertDisagreementLoss, self).__init__()
        self.disagreement_coef = disagreement_coef

    def forward(self, expert_logits):

        if len(expert_logits) <= 1:
            return torch.tensor(0.0, device=expert_logits[0].device)

        batch_size = expert_logits[0].size(0)
        num_experts = len(expert_logits)

        expert_probs = [F.softmax(logits, dim=1) for logits in expert_logits]

        total_disagreement = 0.0
        pair_count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):

                kl_ij = F.kl_div(
                    torch.log(expert_probs[j] + 1e-8),
                    expert_probs[i] + 1e-8,
                    reduction='batchmean'
                )

                kl_ji = F.kl_div(
                    torch.log(expert_probs[i] + 1e-8),
                    expert_probs[j] + 1e-8,
                    reduction='batchmean'
                )
                symmetric_kl = (kl_ij + kl_ji) / 2.0
                total_disagreement += symmetric_kl
                pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=expert_logits[0].device)

        avg_disagreement = total_disagreement / pair_count

        disagreement_loss = -self.disagreement_coef * avg_disagreement

        return disagreement_loss


class ExpertLoadLoss(nn.Module):

    def __init__(self, num_experts, importance_coef=1.0, load_coef=0.1):
        super(ExpertLoadLoss, self).__init__()
        self.num_experts = num_experts
        self.importance_coef = importance_coef
        self.load_coef = load_coef

    def forward(self, gate_weights, batch_size):

        expert_utilization = torch.mean(gate_weights, dim=0)  # [num_experts]

        target_utilization = torch.ones_like(expert_utilization) / self.num_experts
        importance_loss = F.mse_loss(expert_utilization, target_utilization)

        load_loss = torch.var(expert_utilization)

        total_load_loss = self.importance_coef * importance_loss + self.load_coef * load_loss

        return total_load_loss