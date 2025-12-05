import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from load_deap import FocalLoss, load_all_subjects
from model_deap import *
import warnings

warnings.filterwarnings('ignore')

lr = 0.002
patience = 25
batch_size = 128

class BiMoETrainer:
    def __init__(self, model, device, num_classes=2, num_experts=7):
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

            disagreement_loss =  self.disagreement_loss_fn(expert_logits)

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

        return avg_loss, accuracy, avg_classification_loss, avg_load_loss, total_disagreement_loss, avg_agreement

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

                # 计算每个专家的准确率
                for i in range(self.num_experts):
                    expert_correct = (expert_predictions[:, i] == labels).sum().item()
                    expert_accuracy = expert_correct / labels.size(0)
                    expert_individual_accuracies[f'expert_{i}'].append(expert_accuracy)

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)

        expert_accuracies = {}
        for expert_key, acc_list in expert_individual_accuracies.items():
            expert_accuracies[expert_key] = np.mean(acc_list) * 100

        return avg_loss, accuracy

    def calculate_expert_agreement(self, expert_predictions):

        batch_size, num_experts = expert_predictions.shape
        agreement_rates = []

        for i in range(batch_size):
            predictions = expert_predictions[i]
            # 计算最多的预测类别
            majority_vote = torch.mode(predictions).values
            # 计算与多数投票一致的专家比例
            agreement = (predictions == majority_vote).float().mean()
            agreement_rates.append(agreement.item())

        return np.mean(agreement_rates) if agreement_rates else 0


def leave_one_subject_out_cv(experts_data, all_labels, subject_count=32, num_epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    experts_config = {
        'EEG': {'input_channels': 32, 'time_points': 512, 'num_classes': 2},
        'prefrontal': {'input_channels': 9, 'time_points': 512, 'num_classes': 2},
        'central': {'input_channels': 9, 'time_points': 512, 'num_classes': 2},
        'parietal': {'input_channels': 9, 'time_points': 512, 'num_classes': 2},
        'occipital': {'input_channels': 5, 'time_points': 512, 'num_classes': 2},
        'temporal': {'input_channels': 2, 'time_points': 512, 'num_classes': 2},
        'peripheral': {'input_channels': 8, 'time_points': 512, 'num_classes': 2}
    }

    results = []
    expert_names = list(experts_config.keys())

    for test_subject in range(subject_count):
        print(f"\n=== Leave-one-out cross-validation: Subject {test_subject} as the test set ===")

        train_data = {expert: [] for expert in expert_names}
        train_labels = []
        test_data = {expert: [] for expert in expert_names}
        test_labels = []

        for sub_id in range(subject_count):
            if sub_id == test_subject:
                for expert in expert_names:
                    test_data[expert].append(experts_data[expert][sub_id])
                test_labels.append(all_labels[sub_id])
            else:
                for expert in expert_names:
                    train_data[expert].append(experts_data[expert][sub_id])
                train_labels.append(all_labels[sub_id])

        for expert in expert_names:
            train_data[expert] = torch.cat(train_data[expert], dim=0)
            test_data[expert] = torch.cat(test_data[expert], dim=0)

        train_labels = torch.cat(train_labels, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        train_dataset = MultiExpertDEAPDataset(train_data, train_labels)
        test_dataset = MultiExpertDEAPDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MixtureOfExperts(experts_config, num_classes=2)
        trainer = BiMoETrainer(model, device, num_experts=len(expert_names))

        best_acc = 0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc, train_cls_loss, train_load_loss, train_disagreement_loss, train_agreement = trainer.train_epoch(
                train_loader)
            test_loss, test_acc = trainer.evaluate(test_loader)

            trainer.scheduler.step(test_acc)

            if test_acc > best_acc:
                if test_acc > best_acc:
                    best_acc = test_acc

                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stop on {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train_acc: {train_acc:.2f}%, Test_acc: {test_acc:.2f}%"
                      f"Loss: {train_cls_loss:.4f}, ")

        print(f"Sub {test_subject}: The best performance: {best_acc:.2f}% (Epoch: {best_epoch})")

        results.append({
            'test_subject': test_subject,
            'best_accuracy': best_acc,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'best_epoch': best_epoch
        })

    print("\n=== Experimental results of leave-one-out cv ===")
    accuracies = [result['best_accuracy'] for result in results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"Mean ACC: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Acc for each subject: {[f'{acc:.2f}%' for acc in accuracies]}")

    return results

if __name__ == "__main__":
    data_folder = "../Work2/data_raw_DEAP_A"

    experts_data, all_labels, subject_info = load_all_subjects(data_folder, normalize=True)

    results = leave_one_subject_out_cv(experts_data, all_labels)