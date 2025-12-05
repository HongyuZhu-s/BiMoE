import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from load_dreamer import FocalLoss
from model_dreamer import *
import warnings

warnings.filterwarnings('ignore')

num_epochs = 100
lr = 0.002
patience = 25
batch_size = 128
label_type = "A"
data_folder = "../Work2/P_dreamer_data_all"
subject_count = 23

class BiMoETrainer:
    def __init__(self, model, device, num_classes=2, num_experts=8):  # 注意：专家数量改为8
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.num_experts = num_experts

        self.load_loss_fn = ExpertLoadLoss(num_experts=num_experts,
                                           importance_coef=1.0,
                                           load_coef=0.1)

        self.disagreement_loss_fn = ExpertDisagreementLoss()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion1 = FocalLoss(alpha=1, gamma=2)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                                    patience=patience, factor=0.5)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_load_loss = 0
        total_disagreement_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels_list = []

        expert_agreement_stats = []

        for batch_idx, (expert_data, labels) in enumerate(dataloader):
            expert_data_device = {}
            for expert_name, data in expert_data.items():
                expert_data_device[expert_name] = data.to(self.device)
            labels = labels.to(self.device).squeeze()

            self.optimizer.zero_grad()

            outputs, gate_weights, expert_logits = self.model(expert_data_device)

            classification_loss = self.criterion1(outputs, labels)

            load_loss = 100 * self.load_loss_fn(gate_weights, labels.size(0))

            disagreement_loss = self.disagreement_loss_fn(expert_logits)

            loss = classification_loss + load_loss + disagreement_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_load_loss += load_loss.item()
            total_disagreement_loss += disagreement_loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())
            expert_predictions, expert_confidences = self.model.get_expert_decisions(expert_logits)
            agreement_rate = self.calculate_expert_agreement(expert_predictions)
            expert_agreement_stats.append(agreement_rate)

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        avg_classification_loss = total_classification_loss / len(dataloader)
        avg_load_loss = total_load_loss / len(dataloader)
        avg_agreement = np.mean(expert_agreement_stats) if expert_agreement_stats else 0
        f1 = f1_score(all_labels_list, all_predictions, average='weighted') * 100

        return avg_loss, accuracy, f1, avg_classification_loss, avg_load_loss, total_disagreement_loss, avg_agreement

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels_list = []
        all_gate_weights = []
        expert_agreement_stats = []
        expert_individual_accuracies = {f'expert_{i}': [] for i in range(self.num_experts)}

        with torch.no_grad():
            for batch_idx, (expert_data, labels) in enumerate(dataloader):
                expert_data_device = {}
                for expert_name, data in expert_data.items():
                    expert_data_device[expert_name] = data.to(self.device)
                labels = labels.to(self.device).squeeze()

                outputs, gate_weights, expert_logits = self.model(expert_data_device)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu())

                expert_predictions, expert_confidences = self.model.get_expert_decisions(expert_logits)
                agreement_rate = self.calculate_expert_agreement(expert_predictions)
                expert_agreement_stats.append(agreement_rate)

                for i in range(self.num_experts):
                    expert_correct = (expert_predictions[:, i] == labels).sum().item()
                    expert_accuracy = expert_correct / labels.size(0)
                    expert_individual_accuracies[f'expert_{i}'].append(expert_accuracy)

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def calculate_expert_agreement(self, expert_predictions):
        batch_size, num_experts = expert_predictions.shape
        agreement_rates = []

        for i in range(batch_size):
            predictions = expert_predictions[i]
            majority_vote = torch.mode(predictions).values
            agreement = (predictions == majority_vote).float().mean()
            agreement_rates.append(agreement.item())

        return np.mean(agreement_rates) if agreement_rates else 0


def leave_one_subject_out_cv_dreamer():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_experts_data = {region: [] for region in DREAMER_REGIONS}
    all_experts_data['peripheral'] = []
    all_labels_raw = []

    for sub_id in range(subject_count):
        file_path = os.path.join(data_folder, f"dreamer_{sub_id}.npy")
        if not os.path.exists(file_path):
            file_path = os.path.join(data_folder, f"dreamer{sub_id}.npy")

        loaded = np.load(file_path, allow_pickle=True).item()
        eeg = loaded['eeg_data']  # (18, 14, 7680)
        ecg = loaded['ecg_data']  # (18, 2, 15360)
        labels = loaded['labels']  # (18, 3)

        if label_type == "V":
            target_labels = labels[:, 0]
        elif label_type == "A":
            target_labels = labels[:, 1]
        else:
            raise ValueError("label_type must be 'V', 'A'")

        eeg_segments, ecg_segments, segment_labels = preprocess_data(eeg, ecg, target_labels)

        all_labels_raw.append(segment_labels)

        for region, ch_list in DREAMER_REGIONS.items():
            indices = [CHANNEL_TO_IDX[ch] for ch in ch_list]
            region_eeg = eeg_segments[:, indices, :]  # (1080, num_channels, 128)
            all_experts_data[region].append(torch.tensor(region_eeg, dtype=torch.float32))

        all_experts_data['peripheral'].append(torch.tensor(ecg_segments, dtype=torch.float32))

    all_binary_labels = []
    for raw_labels in all_labels_raw:
        raw_labels = np.array(raw_labels)
        binary = np.where(raw_labels <= 2, 0, 1).astype(np.int64)
        all_binary_labels.append(torch.tensor(binary, dtype=torch.long))

    experts_config = {
        'EEG': {'input_channels': 14, 'num_classes': 2},
        'prefrontal': {'input_channels': 6, 'num_classes': 2},
        'central': {'input_channels': 4, 'num_classes': 2},
        'parietal': {'input_channels': 2, 'num_classes': 2},
        'occipital': {'input_channels': 2, 'num_classes': 2},
        'temporal': {'input_channels': 2, 'num_classes': 2},
        'peripheral': {'input_channels': 2, 'num_classes': 2},
    }

    expert_names = list(experts_config.keys())
    results = []

    for test_subject in range(subject_count):
        print(f"\n=== Leave-one-out cross-validation: Subject {test_subject} as the test set ===")

        train_data = {expert: [] for expert in expert_names}
        train_labels = []
        test_data = {expert: [] for expert in expert_names}
        test_labels = []

        for sub_id in range(subject_count):
            if sub_id == test_subject:
                for expert in expert_names:
                    test_data[expert].append(all_experts_data[expert][sub_id])
                test_labels.append(all_binary_labels[sub_id])
            else:
                for expert in expert_names:
                    train_data[expert].append(all_experts_data[expert][sub_id])
                train_labels.append(all_binary_labels[sub_id])

        for expert in expert_names:
            train_data[expert] = torch.cat(train_data[expert], dim=0)
            test_data[expert] = torch.cat(test_data[expert], dim=0)

        train_labels = torch.cat(train_labels, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        train_dataset = MultiExpertDreamerDataset(train_data, train_labels)
        test_dataset = MultiExpertDreamerDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = EnhancedMixtureOfExperts(experts_config, num_classes=2)
        trainer = BiMoETrainer(
            model=model,
            device=device,
            num_experts=len(expert_names)
        )

        best_acc = 0.0
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            trainer.train_epoch(train_loader)
            _, test_acc = trainer.evaluate(test_loader)

            if hasattr(trainer, 'scheduler'):
                trainer.scheduler.step(test_acc)

            if test_acc > best_acc:
                if test_acc > best_acc:
                    best_acc = test_acc

                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        print(f"Sub {test_subject}: Acc={best_acc:.4f}")

        results.append((best_acc, best_f1))

    avg_acc = np.mean([r[0] for r in results])
    std_acc = np.std([r[0] for r in results])

    summary = (
        f"\n=== Result ===\n"
        f"Mean ACC: {avg_acc:.4f} ± {std_acc:.4f}\n"
    )
    print(summary)

    return avg_acc


for target in ["V", "A", "D"]:
    print(f"\n>>> Begin training {target} <<<")
    label_type = target
    acc = leave_one_subject_out_cv_dreamer()