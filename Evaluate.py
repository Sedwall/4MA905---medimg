import torch
from PCAMdataset import PCAMdataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

class Evaluate:
    def __init__(self, model, val_data: DataLoader, device):
        self.model = model
        self.device = device
        self.strObj = ['training_time', 'num_params', 'model_size']
        self.dataset = val_data

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            # Get one big batch from the DataLoader
            data_iter = iter(self.dataset)
            X, F, y = next(data_iter)
            X = X.to(self.device)
            y = y.to(self.device)
            
            if F == None:
                y_pred = self.model(X)
            else:
                y_pred = self.model(X, F)
            y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred_labels)
            precision = precision_score(y_true, y_pred_labels, zero_division=0)
            recall = recall_score(y_true, y_pred_labels, zero_division=0)
            f1 = f1_score(y_true, y_pred_labels, zero_division=0)
            # For binary classification, use y_pred[:, 1] as probability for class 1
            roc_auc = roc_auc_score(y_true, y_pred[:, 1].cpu().numpy()) if y_pred.shape[1] > 1 else 0.0
        num_params = sum(p.numel() for p in self.model.parameters())
        param_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        param_size_mb = param_size_bytes / (1024 ** 2)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "num_params": num_params,
            "model_size": f"{param_size_mb:.2f} MB"
        }
    
    def save_metrics(self, metrics, filepath):
        longest_name = max(len(m) for m in metrics.keys())
        with open(filepath, 'w') as f:
            f.write("=" * (longest_name + 14) + "\n")
            f.write("Model Evaluation Metrics\n")
            f.write("=" * (longest_name + 14) + "\n")
            for metric, value in metrics.items():
                if metric in self.strObj: f.write(f"{metric:<{longest_name}} : {value:>8}\n")
                else: f.write(f"{metric:<{longest_name}} : {value:>8.4f}\n")
            f.write("=" * (longest_name + 14) + "\n")
        print(f"Metrics saved to {filepath}")


    def print_metrics(self, metrics):
        longest_name = max(len(m) for m in metrics.keys())
        print("=" * (longest_name + 14))
        print("Model Evaluation Metrics")
        print("=" * (longest_name + 14))
        for metric, value in metrics.items():
            if metric in self.strObj: print(f"{metric:<{longest_name}} : {value:>8}")
            else: print(f"{metric:<{longest_name}} : {value:>8.4f}")
        print("=" * (longest_name + 14))
        print()