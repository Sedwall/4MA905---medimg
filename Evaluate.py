import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

class Evaluate:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(X)
            y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred_labels)
            precision = precision_score(y_true, y_pred_labels, zero_division=0)
            recall = recall_score(y_true, y_pred_labels, zero_division=0)
            f1 = f1_score(y_true, y_pred_labels, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_pred[:, 1].cpu().numpy())

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
    
    def save_metrics(self, metrics, filepath):
        longest_name = max(len(m) for m in metrics.keys())
        with open(filepath, 'w') as f:
            f.write("=" * (longest_name + 14) + "\n")
            f.write("Model Evaluation Metrics\n")
            f.write("=" * (longest_name + 14) + "\n")
            for metric, value in metrics.items():
                if metric is 'training_time': f.write(f"{metric:<{longest_name}} : {value:>8}\n")
                else: f.write(f"{metric:<{longest_name}} : {value:>8.4f}\n")
            f.write("=" * (longest_name + 14) + "\n")
        print(f"Metrics saved to {filepath}")


    def print_metrics(self, metrics):
        longest_name = max(len(m) for m in metrics.keys())
        print("=" * (longest_name + 14))
        print("Model Evaluation Metrics")
        print("=" * (longest_name + 14))
        for metric, value in metrics.items():
            if metric is 'training_time': print(f"{metric:<{longest_name}} : {value:>8}")
            else: print(f"{metric:<{longest_name}} : {value:>8.4f}")
        print("=" * (longest_name + 14))
        print()